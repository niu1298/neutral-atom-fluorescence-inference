"""Repeated-imaging and loss-budget models for the optimized measurements.

The inputs are held-out-emission scores in the standardized schema-4.0 table.
All occupancy and retention quantities remain apparent/model-based because no
external occupancy labels or exact loss timestamps are available.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from .cluster_bootstrap import (
    ClusterBootstrapResult,
    condition_stratified_cluster_bootstrap,
)


REPEATED_MODELS = (
    "continuous_only",
    "common_additional_readout",
    "exposure_specific_additional_readout",
    "interval_specific_diagnostic",
)
_EPS = 1e-9


def _require_columns(data: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise ValueError(f"repeated-imaging data are missing {missing}")


def _shot_columns(data: pd.DataFrame) -> tuple[str, ...]:
    if "_bootstrap_cluster_id" in data.columns:
        return ("run_id", "_bootstrap_cluster_id")
    return ("run_id", "shot_id")


def _exposure_label(exposure_s: float) -> str:
    return f"{int(round(float(exposure_s) * 1000.0))}ms"


def _binary_nll(response: np.ndarray, probability: np.ndarray) -> float:
    y = np.asarray(response, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), _EPS, 1.0 - _EPS)
    return float(-(y * np.log(p) + (1.0 - y) * np.log1p(-p)).sum())


def _aggregate_calls(data: pd.DataFrame, response_col: str) -> pd.DataFrame:
    shot_cols = _shot_columns(data)
    required = [
        *shot_cols,
        "run_id",
        "frame_index",
        "exposure_s",
        "cumulative_bright_s",
        "cumulative_dark_s",
        "pulse_count",
        response_col,
    ]
    _require_columns(data, required)
    response = data[response_col]
    if response.isna().any() or not response.astype(int).isin([0, 1]).all():
        raise ValueError(f"{response_col!r} must contain non-missing binary calls")
    model_cols = [
        "run_id",
        *[col for col in shot_cols if col != "run_id"],
        "frame_index",
        "exposure_s",
        "cumulative_bright_s",
        "cumulative_dark_s",
        "pulse_count",
    ]
    out = (
        data.groupby(model_cols, observed=True, sort=False)[response_col]
        .agg(["sum", "size"])
        .reset_index()
        .rename(columns={"sum": "successes", "size": "trials"})
    )
    return out


@dataclass(frozen=True)
class RepeatedImagingFit:
    """Bernoulli/binomial fit with fixed bright and dark hazards."""

    model: str
    lambda_bright: float
    lambda_dark: float
    initial_occupancy_by_run: Mapping[str, float]
    q_common: float
    q_by_exposure: Mapping[float, float]
    q_by_boundary: Mapping[int, float]
    nll_value: float
    n_site_observations: int
    n_independent_shots: int
    converged: bool
    parameter_on_boundary: bool
    optimizer_message: str

    def _log_additional_factor(
        self, exposure_s: np.ndarray, pulse_count: np.ndarray
    ) -> np.ndarray:
        pulse = np.asarray(pulse_count, dtype=int)
        if np.any(pulse < 1):
            raise ValueError("pulse_count must be one-based and positive")
        if self.model == "continuous_only":
            return np.zeros(len(pulse), dtype=float)
        if self.model == "common_additional_readout":
            return (pulse - 1) * np.log(np.clip(self.q_common, _EPS, 1.0))
        if self.model == "exposure_specific_additional_readout":
            q = np.asarray(
                [self.q_by_exposure[float(value)] for value in exposure_s],
                dtype=float,
            )
            return (pulse - 1) * np.log(np.clip(q, _EPS, 1.0))
        if self.model == "interval_specific_diagnostic":
            return np.asarray(
                [
                    sum(
                        np.log(np.clip(self.q_by_boundary[index], _EPS, 1.0))
                        for index in range(1, int(count))
                    )
                    for count in pulse
                ],
                dtype=float,
            )
        raise ValueError(f"unsupported repeated model {self.model!r}")

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        _require_columns(
            data,
            [
                "run_id",
                "exposure_s",
                "cumulative_bright_s",
                "cumulative_dark_s",
                "pulse_count",
            ],
        )
        run_probability = data["run_id"].astype(str).map(
            self.initial_occupancy_by_run
        )
        if run_probability.isna().any():
            missing = sorted(
                set(data.loc[run_probability.isna(), "run_id"].astype(str))
            )
            raise ValueError(f"unseen repeated-imaging runs: {missing}")
        exposure = data["exposure_s"].astype(float).to_numpy()
        bright = data["cumulative_bright_s"].astype(float).to_numpy()
        dark = data["cumulative_dark_s"].astype(float).to_numpy()
        pulse = data["pulse_count"].astype(int).to_numpy()
        log_probability = (
            np.log(run_probability.to_numpy(float))
            - self.lambda_bright * bright
            - self.lambda_dark * dark
            + self._log_additional_factor(exposure, pulse)
        )
        return np.clip(np.exp(log_probability), _EPS, 1.0 - _EPS)

    def nll(self, data: pd.DataFrame, response_col: str = "apparent_occupied") -> float:
        return _binary_nll(
            data[response_col].astype(int).to_numpy(), self.predict(data)
        )

    def parameter_dict(self) -> dict[str, float]:
        values = {
            f"initial_occupancy__{run}": float(probability)
            for run, probability in self.initial_occupancy_by_run.items()
        }
        if self.model == "common_additional_readout":
            values["additional_readout_retention__common"] = float(self.q_common)
        elif self.model == "exposure_specific_additional_readout":
            values.update(
                {
                    f"additional_readout_retention__{_exposure_label(exposure)}": (
                        float(q)
                    )
                    for exposure, q in self.q_by_exposure.items()
                }
            )
        elif self.model == "interval_specific_diagnostic":
            values.update(
                {
                    f"additional_readout_retention__boundary_{index}": float(q)
                    for index, q in self.q_by_boundary.items()
                }
            )
        return values

    def q_for_boundary(self, exposure_s: float, boundary: int) -> float:
        if boundary < 1:
            raise ValueError("boundary must be at least one")
        if self.model == "continuous_only":
            return 1.0
        if self.model == "common_additional_readout":
            return float(self.q_common)
        if self.model == "exposure_specific_additional_readout":
            return float(self.q_by_exposure[float(exposure_s)])
        return float(self.q_by_boundary[int(boundary)])


def fit_repeated_imaging(
    data: pd.DataFrame,
    *,
    model: str,
    lambda_bright: float,
    lambda_dark: float,
    response_col: str = "apparent_occupied",
    split_col: str | None = "split",
    training_label: str = "train",
) -> RepeatedImagingFit:
    """Fit run loading and optional pulse-boundary retention factors."""
    if model not in REPEATED_MODELS:
        raise ValueError(f"unsupported repeated model {model!r}")
    if min(lambda_bright, lambda_dark) < 0.0:
        raise ValueError("bright and dark rates must be non-negative")
    if split_col is not None and split_col in data.columns:
        labels = set(data[split_col].dropna().astype(str))
        if labels != {training_label}:
            raise ValueError(
                "repeated-imaging fitting accepts training rows only; "
                f"{split_col!r} contains {sorted(labels)}"
            )
    cells = _aggregate_calls(data, response_col)
    runs = tuple(sorted(cells["run_id"].astype(str).unique()))
    exposures = tuple(sorted(float(value) for value in cells["exposure_s"].unique()))
    boundaries = tuple(
        range(1, int(cells["pulse_count"].astype(int).max()))
    )
    run_index = {run: index for index, run in enumerate(runs)}
    exposure_index = {
        exposure: index for index, exposure in enumerate(exposures)
    }
    n_run = len(runs)
    n_q = {
        "continuous_only": 0,
        "common_additional_readout": 1,
        "exposure_specific_additional_readout": len(exposures),
        "interval_specific_diagnostic": len(boundaries),
    }[model]

    first = cells.loc[cells["pulse_count"].astype(int) == 1]
    initial = []
    for run in runs:
        block = first.loc[first["run_id"].astype(str) == run]
        observed = float(block["successes"].sum() / block["trials"].sum())
        exposure = float(block["exposure_s"].iloc[0])
        adjusted = np.clip(
            observed * np.exp(lambda_bright * exposure), 0.02, 0.98
        )
        initial.append(float(logit(adjusted)))

    successes = cells["successes"].to_numpy(float)
    trials = cells["trials"].to_numpy(float)
    run_codes = np.asarray(
        [run_index[str(value)] for value in cells["run_id"]], dtype=int
    )
    exposure_values = cells["exposure_s"].to_numpy(float)
    bright = cells["cumulative_bright_s"].to_numpy(float)
    dark = cells["cumulative_dark_s"].to_numpy(float)
    pulse = cells["pulse_count"].to_numpy(int)

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pi = expit(parameters[:n_run])
        q = expit(parameters[n_run:]) if n_q else np.empty(0, dtype=float)
        return pi, q

    def probabilities(parameters: np.ndarray) -> np.ndarray:
        pi, q = unpack(parameters)
        logp = (
            np.log(pi[run_codes])
            - lambda_bright * bright
            - lambda_dark * dark
        )
        if model == "common_additional_readout":
            logp += (pulse - 1) * np.log(q[0])
        elif model == "exposure_specific_additional_readout":
            q_for_row = np.asarray(
                [q[exposure_index[float(value)]] for value in exposure_values]
            )
            logp += (pulse - 1) * np.log(q_for_row)
        elif model == "interval_specific_diagnostic":
            logp += np.asarray(
                [sum(np.log(q[: max(count - 1, 0)])) for count in pulse]
            )
        return np.clip(np.exp(logp), _EPS, 1.0 - _EPS)

    def objective(parameters: np.ndarray) -> float:
        probability = probabilities(parameters)
        return float(
            -(
                successes * np.log(probability)
                + (trials - successes) * np.log1p(-probability)
            ).sum()
        )

    starts = (0.995, 0.98, 0.94) if n_q else (1.0,)
    best = None
    for q_start in starts:
        x0 = np.asarray(
            [*initial, *([float(logit(q_start))] * n_q)], dtype=float
        )
        candidate = minimize(objective, x0, method="L-BFGS-B")
        if best is None or float(candidate.fun) < float(best.fun):
            best = candidate
    assert best is not None
    if not np.isfinite(best.fun):
        raise RuntimeError("repeated-imaging fit did not produce a finite likelihood")
    pi, q = unpack(best.x)
    initial_by_run = {run: float(pi[index]) for run, index in run_index.items()}
    q_common = float(q[0]) if model == "common_additional_readout" else 1.0
    q_by_exposure = (
        {
            exposure: float(q[index])
            for exposure, index in exposure_index.items()
        }
        if model == "exposure_specific_additional_readout"
        else {}
    )
    q_by_boundary = (
        {boundary: float(q[index]) for index, boundary in enumerate(boundaries)}
        if model == "interval_specific_diagnostic"
        else {}
    )
    q_values = (
        [q_common]
        if model == "common_additional_readout"
        else list(q_by_exposure.values())
        if model == "exposure_specific_additional_readout"
        else list(q_by_boundary.values())
    )
    on_boundary = any(value > 1.0 - 1e-6 or value < 1e-6 for value in q_values)
    return RepeatedImagingFit(
        model=model,
        lambda_bright=float(lambda_bright),
        lambda_dark=float(lambda_dark),
        initial_occupancy_by_run=initial_by_run,
        q_common=q_common,
        q_by_exposure=q_by_exposure,
        q_by_boundary=q_by_boundary,
        nll_value=float(best.fun),
        n_site_observations=int(trials.sum()),
        n_independent_shots=int(len(cells[list(_shot_columns(data))].drop_duplicates())),
        converged=bool(best.success),
        parameter_on_boundary=bool(on_boundary),
        optimizer_message=str(best.message),
    )


def _paired_validation_gate(
    baseline: RepeatedImagingFit,
    candidate: RepeatedImagingFit,
    validation: pd.DataFrame,
    *,
    seed: int,
    n_resamples: int = 2000,
) -> dict[str, Any]:
    y = validation["apparent_occupied"].astype(int).to_numpy()
    p0 = baseline.predict(validation)
    p1 = candidate.predict(validation)
    row_improvement = (
        -(y * np.log(p0) + (1 - y) * np.log1p(-p0))
        + (y * np.log(p1) + (1 - y) * np.log1p(-p1))
    )
    work = validation[["run_id", "shot_id"]].copy()
    work["improvement"] = row_improvement
    shot = work.groupby(["run_id", "shot_id"], observed=True)[
        "improvement"
    ].mean()
    values = shot.to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [
            np.mean(rng.choice(values, size=len(values), replace=True))
            for _ in range(n_resamples)
        ]
    )
    estimate = float(values.mean())
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "mean_nll_improvement_per_row": estimate,
        "cluster_lower": float(lower),
        "cluster_upper": float(upper),
        "n_independent_validation_shots": int(len(values)),
        "passed": bool(estimate > 1e-4 and lower > 0.0),
    }


def compare_repeated_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    lambda_bright: float,
    lambda_dark: float,
    seed: int = 20260731,
) -> tuple[str, pd.DataFrame, dict[str, RepeatedImagingFit]]:
    """Select M0/M1/M2 on validation; retain M3 as diagnostic only."""
    fits = {
        model: fit_repeated_imaging(
            train,
            model=model,
            lambda_bright=lambda_bright,
            lambda_dark=lambda_dark,
        )
        for model in REPEATED_MODELS
    }
    validation_nll = {
        model: fit.nll(validation) for model, fit in fits.items()
    }
    m1_gate = _paired_validation_gate(
        fits["continuous_only"],
        fits["common_additional_readout"],
        validation,
        seed=seed,
    )
    m2_gate = _paired_validation_gate(
        fits["common_additional_readout"],
        fits["exposure_specific_additional_readout"],
        validation,
        seed=seed + 1,
    )
    selected = "continuous_only"
    if m1_gate["passed"]:
        selected = "common_additional_readout"
        if m2_gate["passed"]:
            selected = "exposure_specific_additional_readout"
    rows = []
    for model in REPEATED_MODELS:
        gate = (
            m1_gate
            if model == "common_additional_readout"
            else m2_gate
            if model == "exposure_specific_additional_readout"
            else None
        )
        rows.append(
            {
                "candidate": model,
                "validation_nll": float(validation_nll[model]),
                "validation_mean_nll": float(
                    validation_nll[model] / len(validation)
                ),
                "delta_nll_vs_continuous": float(
                    validation_nll[model]
                    - validation_nll["continuous_only"]
                ),
                "complexity_gate": gate,
                "eligible_for_primary": model != "interval_specific_diagnostic",
                "selected_on_validation": model == selected,
            }
        )
    return selected, pd.DataFrame(rows), fits


def bootstrap_repeated_imaging(
    development: pd.DataFrame,
    *,
    model: str,
    lambda_bright: float,
    lambda_dark: float,
    n_boot: int,
    seed: int,
) -> ClusterBootstrapResult:
    def statistic(sample: pd.DataFrame) -> Mapping[str, float]:
        fit = fit_repeated_imaging(
            sample,
            model=model,
            lambda_bright=lambda_bright,
            lambda_dark=lambda_dark,
            split_col=None,
        )
        return fit.parameter_dict()

    return condition_stratified_cluster_bootstrap(
        development,
        statistic,
        n_boot=n_boot,
        seed=seed,
        shot_cols=("run_id", "shot_id"),
        condition_cols="condition_id",
    )


def _resample_contiguous_blocks(
    data: pd.DataFrame,
    *,
    rng: np.random.Generator,
    block_size: int,
) -> pd.DataFrame:
    """Resample chronological blocks while retaining every row of each shot."""
    if block_size < 1:
        raise ValueError("block_size must be positive")
    _require_columns(data, ["run_id", "shot_id", "shot_order"])
    pieces = []
    cluster = 0
    for run, run_data in data.groupby("run_id", observed=True, sort=True):
        shots = (
            run_data[["shot_id", "shot_order"]]
            .drop_duplicates()
            .sort_values("shot_order", kind="stable")
            .reset_index(drop=True)
        )
        shots["_source_block"] = np.arange(len(shots)) // block_size
        blocks = sorted(int(value) for value in shots["_source_block"].unique())
        sampled = rng.choice(blocks, size=len(blocks), replace=True)
        for draw_block, source_block in enumerate(sampled):
            selected = shots.loc[shots["_source_block"] == int(source_block)]
            for shot_id in selected["shot_id"]:
                block = run_data.loc[run_data["shot_id"] == shot_id].copy()
                block["_bootstrap_cluster_id"] = cluster
                block["_bootstrap_block_id"] = f"{run}|{draw_block}"
                pieces.append(block)
                cluster += 1
    if not pieces:
        raise ValueError("cannot block-bootstrap an empty repeated table")
    return pd.concat(pieces, ignore_index=True)


def bootstrap_repeated_imaging_blocks(
    development: pd.DataFrame,
    *,
    model: str,
    lambda_bright: float,
    lambda_dark: float,
    n_boot: int,
    seed: int,
    block_size: int = 20,
) -> ClusterBootstrapResult:
    """Drift sensitivity using complete contiguous acquisition-order blocks."""
    if n_boot < 1:
        raise ValueError("n_boot must be positive")

    def fit_values(sample: pd.DataFrame) -> dict[str, float]:
        return fit_repeated_imaging(
            sample,
            model=model,
            lambda_bright=lambda_bright,
            lambda_dark=lambda_dark,
            split_col=None,
        ).parameter_dict()

    point = fit_values(development)
    names = tuple(point)
    rng = np.random.default_rng(seed)
    draws = []
    failures = []
    for replicate in range(n_boot):
        try:
            sample = _resample_contiguous_blocks(
                development, rng=rng, block_size=block_size
            )
            values = fit_values(sample)
            if tuple(values) != names:
                raise ValueError("block bootstrap changed parameter names")
            row = np.asarray([values[name] for name in names], dtype=float)
            if not np.isfinite(row).all():
                raise ValueError("block bootstrap returned non-finite parameters")
            draws.append(row)
        except (ArithmeticError, FloatingPointError, RuntimeError, ValueError) as exc:
            failures.append(
                {
                    "replicate": int(replicate),
                    "exception": type(exc).__name__,
                    "message": str(exc),
                }
            )
    if len(failures) / n_boot > 0.1 or not draws:
        raise RuntimeError(
            f"{len(failures)}/{n_boot} contiguous-block fits failed"
        )
    return ClusterBootstrapResult(
        estimate=pd.Series(point, dtype=float),
        draws=pd.DataFrame(np.vstack(draws), columns=names),
        confidence=0.95,
        n_requested=int(n_boot),
        n_successful=len(draws),
        n_failed=len(failures),
        seed=int(seed),
        failures=tuple(failures),
    )


def repeated_shot_order_sensitivity(
    development: pd.DataFrame,
    *,
    selected_model: str,
    lambda_bright: float,
    lambda_dark: float,
) -> dict[str, Any]:
    """Refit frozen structures in early and late complete-shot halves.

    The sensitivity does not reselect a model after looking at acquisition
    order. It keeps the validation-selected structure fixed and also reports
    the common-factor alternative in both halves so a time-localized residual
    cannot be mistaken for a stable pulse-count effect.
    """
    _require_columns(development, ["run_id", "shot_id", "shot_order"])
    assignments = []
    counts: dict[str, dict[str, int]] = {}
    for run, block in development.groupby("run_id", observed=True, sort=True):
        shots = (
            block[["run_id", "shot_id", "shot_order"]]
            .drop_duplicates()
            .sort_values("shot_order", kind="stable")
            .reset_index(drop=True)
        )
        midpoint = len(shots) // 2
        if midpoint < 2 or len(shots) - midpoint < 2:
            raise ValueError("shot-order sensitivity needs at least four shots per run")
        shots["acquisition_half"] = [
            "early" if index < midpoint else "late"
            for index in range(len(shots))
        ]
        assignments.append(shots[["run_id", "shot_id", "acquisition_half"]])
        counts[str(run)] = {
            "early": int(midpoint),
            "late": int(len(shots) - midpoint),
        }
    tagged = development.merge(
        pd.concat(assignments, ignore_index=True),
        on=["run_id", "shot_id"],
        how="left",
        validate="many_to_one",
    )
    halves: dict[str, Any] = {}
    for half in ("early", "late"):
        subset = tagged.loc[tagged["acquisition_half"] == half]
        selected_fit = fit_repeated_imaging(
            subset,
            model=selected_model,
            lambda_bright=lambda_bright,
            lambda_dark=lambda_dark,
            split_col=None,
        )
        common_fit = (
            selected_fit
            if selected_model == "common_additional_readout"
            else fit_repeated_imaging(
                subset,
                model="common_additional_readout",
                lambda_bright=lambda_bright,
                lambda_dark=lambda_dark,
                split_col=None,
            )
        )
        halves[half] = {
            "selected_structure_parameters": selected_fit.parameter_dict(),
            "common_factor_parameters": common_fit.parameter_dict(),
            "n_independent_shots": int(
                len(subset[["run_id", "shot_id"]].drop_duplicates())
            ),
        }
    return {
        "split_rule": (
            "within each exposure run, first versus second chronological half "
            "of training-plus-validation complete shots"
        ),
        "shots_by_run_and_half": counts,
        "selected_structure_frozen": selected_model,
        "halves": halves,
        "interpretation": (
            "Acquisition-order sensitivity only; no model was selected on these "
            "subsets and the common-factor fits do not override validation selection."
        ),
    }


def prefix_curve(
    scored: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Shot-clustered posterior apparent occupancy for every run prefix."""
    _require_columns(
        scored,
        [
            "run_id",
            "shot_id",
            "frame_index",
            "exposure_s",
            "cumulative_bright_s",
            "cumulative_dark_s",
            "pulse_count",
            "posterior_occupied",
        ],
    )
    shot = (
        scored.groupby(
            [
                "run_id",
                "shot_id",
                "frame_index",
                "exposure_s",
                "cumulative_bright_s",
                "cumulative_dark_s",
                "pulse_count",
            ],
            observed=True,
        )["posterior_occupied"]
        .mean()
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    rows = []
    keys = [
        "run_id",
        "frame_index",
        "exposure_s",
        "cumulative_bright_s",
        "cumulative_dark_s",
        "pulse_count",
    ]
    for key, block in shot.groupby(keys, observed=True, sort=True):
        values = block["posterior_occupied"].to_numpy(float)
        draws = np.asarray(
            [
                np.mean(rng.choice(values, size=len(values), replace=True))
                for _ in range(n_boot)
            ]
        )
        rows.append(
            {
                **dict(zip(keys, key)),
                "apparent_occupancy": float(values.mean()),
                "cluster_lower": float(np.quantile(draws, 0.025)),
                "cluster_upper": float(np.quantile(draws, 0.975)),
                "n_independent_shots": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def matched_total_time_contrasts(
    fit: RepeatedImagingFit,
    *,
    dark_gap_s: float = 0.01,
) -> list[dict[str, Any]]:
    """Model-normalized survival at the three predeclared matched totals."""
    comparisons = {
        0.1: ((0.05, 2), (0.10, 1)),
        0.2: ((0.05, 4), (0.10, 2), (0.20, 1)),
        0.4: ((0.10, 4), (0.20, 2)),
    }
    rows = []
    for total, designs in comparisons.items():
        estimates = []
        for exposure, pulses in designs:
            log_survival = (
                -fit.lambda_bright * total
                - fit.lambda_dark * (pulses - 1) * dark_gap_s
                + fit._log_additional_factor(
                    np.asarray([exposure]), np.asarray([pulses])
                )[0]
            )
            estimates.append(
                {
                    "exposure_s": exposure,
                    "pulse_count": pulses,
                    "cumulative_dark_s": (pulses - 1) * dark_gap_s,
                    "model_normalized_survival": float(np.exp(log_survival)),
                }
            )
        reference = estimates[-1]["model_normalized_survival"]
        for estimate in estimates:
            estimate["difference_vs_fewest_pulses"] = float(
                estimate["model_normalized_survival"] - reference
            )
            estimate["ratio_vs_fewest_pulses"] = float(
                estimate["model_normalized_survival"] / reference
            )
        rows.append(
            {
                "total_bright_s": total,
                "designs": estimates,
                "interpretation": (
                    "Normalized by the fitted run-specific initial occupancy; "
                    "not a raw cross-run occupancy contrast."
                ),
            }
        )
    return rows


def heldout_matched_prefix_contrasts(
    test_scored: pd.DataFrame,
    fit: RepeatedImagingFit,
    *,
    n_boot: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Test-shot clustered direct contrasts after run-intercept normalization."""
    _require_columns(
        test_scored,
        [
            "run_id",
            "shot_id",
            "exposure_s",
            "pulse_count",
            "posterior_occupied",
        ],
    )
    shot = (
        test_scored.groupby(
            ["run_id", "shot_id", "exposure_s", "pulse_count"],
            observed=True,
        )["posterior_occupied"]
        .mean()
        .reset_index()
    )
    run_by_exposure = {
        float(exposure): str(block["run_id"].iloc[0])
        for exposure, block in shot.groupby("exposure_s", observed=True)
    }
    comparisons = {
        0.1: ((0.05, 2), (0.10, 1)),
        0.2: ((0.05, 4), (0.10, 2), (0.20, 1)),
        0.4: ((0.10, 4), (0.20, 2)),
    }

    def estimates(sample: pd.DataFrame, designs: Sequence[tuple[float, int]]):
        values = []
        for exposure, pulses in designs:
            run = run_by_exposure[exposure]
            block = sample.loc[
                (sample["run_id"].astype(str) == run)
                & (sample["pulse_count"].astype(int) == pulses)
            ]
            values.append(
                float(
                    block["posterior_occupied"].mean()
                    / fit.initial_occupancy_by_run[run]
                )
            )
        return values

    point = {total: estimates(shot, designs) for total, designs in comparisons.items()}
    rng = np.random.default_rng(seed)
    draws = {total: [] for total in comparisons}
    run_groups = {
        str(run): block for run, block in shot.groupby("run_id", observed=True)
    }
    for _ in range(n_boot):
        pieces = []
        for run, block in run_groups.items():
            shot_ids = block["shot_id"].drop_duplicates().to_numpy()
            sampled = rng.choice(shot_ids, size=len(shot_ids), replace=True)
            for occurrence, shot_id in enumerate(sampled):
                selected = block.loc[block["shot_id"] == shot_id].copy()
                selected["_draw_shot"] = f"{run}|{occurrence}"
                pieces.append(selected)
        sample = pd.concat(pieces, ignore_index=True)
        for total, designs in comparisons.items():
            draws[total].append(estimates(sample, designs))

    rows = []
    for total, designs in comparisons.items():
        array = np.asarray(draws[total], dtype=float)
        reference = point[total][-1]
        reference_draw = array[:, -1]
        design_rows = []
        for index, (exposure, pulses) in enumerate(designs):
            difference_draw = array[:, index] - reference_draw
            lower, upper = np.quantile(difference_draw, [0.025, 0.975])
            run = run_by_exposure[exposure]
            design_rows.append(
                {
                    "exposure_s": exposure,
                    "pulse_count": pulses,
                    "normalized_apparent_occupancy": point[total][index],
                    "difference_vs_fewest_pulses": float(
                        point[total][index] - reference
                    ),
                    "difference_cluster_lower": float(lower),
                    "difference_cluster_upper": float(upper),
                    "n_independent_test_shots": int(
                        run_groups[run]["shot_id"].nunique()
                    ),
                }
            )
        rows.append(
            {
                "total_bright_s": total,
                "designs": design_rows,
                "bootstrap_unit": "complete held-out test shot within exposure run",
                "n_bootstrap": int(n_boot),
                "interpretation": (
                    "Posterior apparent occupancy divided by the development-fit "
                    "run intercept; cross-run contrast uncertainty resamples held-out "
                    "shots and does not identify a unique physical mechanism."
                ),
            }
        )
    return rows


def loss_budget(
    fit: RepeatedImagingFit,
    *,
    exposure_s: float,
    lambda_bright: float | None = None,
    lambda_dark: float | None = None,
    dark_gap_s: float = 0.01,
    n_frames: int = 5,
) -> dict[str, Any]:
    """Multiplicative prior attribution across bright, dark, and residual stages."""
    bright_rate = fit.lambda_bright if lambda_bright is None else float(lambda_bright)
    dark_rate = fit.lambda_dark if lambda_dark is None else float(lambda_dark)
    survival = 1.0
    segments = []
    cumulative = []

    def add(kind: str, index: int, conditional_loss: float) -> None:
        nonlocal survival
        before = survival
        loss = before * conditional_loss
        survival = before * (1.0 - conditional_loss)
        segments.append(
            {
                "segment": kind,
                "segment_index": index,
                "conditional_loss_probability": float(conditional_loss),
                "unconditional_loss_probability": float(loss),
                "survival_after": float(survival),
            }
        )

    for frame in range(1, n_frames + 1):
        add("bright_exposure", frame, 1.0 - np.exp(-bright_rate * exposure_s))
        cumulative.append(
            {"frame": frame, "survival_after_frame": float(survival)}
        )
        if frame == n_frames:
            continue
        add("dark_gap", frame, 1.0 - np.exp(-dark_rate * dark_gap_s))
        add(
            "residual_readout_associated",
            frame,
            1.0 - fit.q_for_boundary(exposure_s, frame),
        )
    component_loss = {
        kind: float(
            sum(
                row["unconditional_loss_probability"]
                for row in segments
                if row["segment"] == kind
            )
        )
        for kind in (
            "bright_exposure",
            "dark_gap",
            "residual_readout_associated",
        )
    }
    total_loss = float(1.0 - survival)
    if not np.isclose(sum(component_loss.values()), total_loss, atol=1e-12):
        raise RuntimeError("loss-segment probabilities do not sum to total loss")
    return {
        "exposure_s": float(exposure_s),
        "dark_gap_s": float(dark_gap_s),
        "n_frames": int(n_frames),
        "segments": segments,
        "component_loss": component_loss,
        "cumulative_survival": cumulative,
        "final_survival": float(survival),
        "total_loss": total_loss,
        "interpretation": (
            "Rate-based prior attribution over integrated stages; not an exact "
            "observed loss time or count-conditioned posterior location."
        ),
    }


def loss_budget_uncertainty(
    fit: RepeatedImagingFit,
    repeated_draws: pd.DataFrame,
    bright_rate_draws: Sequence[float],
    dark_rate_draws: Sequence[float],
    *,
    exposures_s: Sequence[float] = (0.05, 0.10, 0.20),
    seed: int = 20260731,
) -> dict[str, Any]:
    """Propagate independent clustered rate and readout-factor draws."""
    bright = np.asarray(bright_rate_draws, dtype=float)
    dark = np.asarray(dark_rate_draws, dtype=float)
    if min(len(repeated_draws), len(bright), len(dark)) < 2:
        raise ValueError("loss-budget uncertainty requires at least two draws")
    # Combining already-fitted bootstrap draws is cheap; use enough Monte
    # Carlo combinations for stable percentile interpolation even in focused
    # tests with only a few supplied draws.
    n_draw = max(1000, len(repeated_draws), len(bright), len(dark))
    rng = np.random.default_rng(seed)
    summaries: dict[str, Any] = {}
    for exposure in exposures_s:
        rows = []
        for _ in range(n_draw):
            rep = repeated_draws.iloc[int(rng.integers(0, len(repeated_draws)))]
            q_common = fit.q_common
            q_by_exposure = dict(fit.q_by_exposure)
            q_by_boundary = dict(fit.q_by_boundary)
            if fit.model == "common_additional_readout":
                q_common = float(rep["additional_readout_retention__common"])
            elif fit.model == "exposure_specific_additional_readout":
                for value in q_by_exposure:
                    q_by_exposure[value] = float(
                        rep[
                            "additional_readout_retention__"
                            + _exposure_label(value)
                        ]
                    )
            elif fit.model == "interval_specific_diagnostic":
                for boundary in q_by_boundary:
                    q_by_boundary[boundary] = float(
                        rep[
                            f"additional_readout_retention__boundary_{boundary}"
                        ]
                    )
            draw_fit = RepeatedImagingFit(
                **{
                    **fit.__dict__,
                    "q_common": q_common,
                    "q_by_exposure": q_by_exposure,
                    "q_by_boundary": q_by_boundary,
                }
            )
            budget = loss_budget(
                draw_fit,
                exposure_s=float(exposure),
                lambda_bright=float(bright[int(rng.integers(0, len(bright)))]),
                lambda_dark=float(dark[int(rng.integers(0, len(dark)))]),
            )
            rows.append(
                {
                    **budget["component_loss"],
                    "final_survival": budget["final_survival"],
                }
            )
        frame = pd.DataFrame(rows)
        summaries[_exposure_label(exposure)] = {
            column: {
                "lower": float(frame[column].quantile(0.025)),
                "median": float(frame[column].quantile(0.5)),
                "upper": float(frame[column].quantile(0.975)),
            }
            for column in frame.columns
        }
    return {
        "draw_combination": (
            "independent resampling of complete-shot bright, dark, and "
            "repeated-imaging bootstrap draws"
        ),
        "n_monte_carlo_draws": int(n_draw),
        "by_exposure": summaries,
    }
