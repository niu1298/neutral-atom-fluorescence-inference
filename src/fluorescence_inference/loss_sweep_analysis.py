"""Held-out operational analysis for the 2026-07-28 loss sweeps.

This module consumes standardized tables.  It never opens raw shots and never
fits an emission, threshold, background template or geometry on a test shot.
The public responses remain explicitly *apparent* because no empirical
occupancy labels exist.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import expit, logit

from .heldout_emissions import (
    EmissionModel,
    evaluate_heldout,
    fit_emission_baseline,
    fit_required_baselines,
)
from .latent_state import (
    LatentGateCriteria,
    LatentGateEvidence,
    LatentModelSpec,
    LatentStateParameters,
    evaluate_latent_state_gate,
    fit_latent_state,
    forward_log_likelihood,
    posterior_state_probabilities,
    profile_switch_off_rate,
    score_against_held_out_baseline,
    simulate_latent_state,
)
from .splits import SEEDED_STRATEGY, attach_split, build_cycle_split_manifest
from .sweep_models import (
    bootstrap_bright_decay,
    bootstrap_bright_decay_by_cycle,
    bootstrap_control_retention,
    bootstrap_dark_retention,
    bootstrap_dark_retention_by_cycle,
    compare_bright_models,
    compare_control_models,
    compare_dark_models,
    diagnose_bright_bootstrap_multistart,
    diagnose_dark_bootstrap_multistart,
    exact_multiplicative_loss,
    fit_bright_decay,
    fit_control_retention,
    fit_dark_retention,
    make_retention_events,
)


PRIMARY_COUNT_COLUMN = "count_corrected_template"
BACKGROUND_VALUE_COLUMNS: dict[str, str] = {
    "raw": "roi_sum_raw",
    "global": "count_corrected_global",
    "spatial": "count_corrected_spatial",
    "template": "count_corrected_template",
    "annulus_contaminated": "count_corrected_annulus_contaminated",
}
COUNT_TO_BACKGROUND_COLUMN: dict[str, str | None] = {
    "roi_sum_raw": None,
    "count_corrected_global": "background_global",
    "count_corrected_spatial": "background_spatial",
    "count_corrected_template": "background_template_offset",
    "count_corrected_fixed_offset": "background_fixed_offset",
    "count_corrected_annulus_contaminated": "background_annulus_contaminated",
}
DEFAULT_SHOT_COLUMNS = ("run_id", "shot_id")


def jsonable(value: Any) -> Any:
    """Convert numpy/pandas/dataclass values to deterministic JSON objects."""
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, pd.DataFrame):
        return [jsonable(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (str, bytes, bool, int)) else False:
        return None
    return value


def _interval_lookup(intervals: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        str(row["parameter"]): {
            "estimate": float(row["estimate"]),
            "lower": float(row["lower"]),
            "upper": float(row["upper"]),
            "confidence": float(row["confidence"]),
            "n_bootstrap": int(row["n_bootstrap"]),
        }
        for row in intervals.to_dict(orient="records")
    }


def _invert_positive_rate_interval(
    rate_interval: Mapping[str, float] | None,
) -> dict[str, float] | None:
    """Transform a positive rate interval into its reciprocal lifetime interval."""
    if rate_interval is None:
        return None
    estimate = float(rate_interval["estimate"])
    lower_rate = float(rate_interval["lower"])
    upper_rate = float(rate_interval["upper"])
    if estimate <= 0.0 or lower_rate <= 0.0 or upper_rate <= 0.0:
        return None
    return {
        "estimate": 1.0 / estimate,
        "lower": 1.0 / upper_rate,
        "upper": 1.0 / lower_rate,
    }


def _shot_count(df: pd.DataFrame) -> int:
    return int(df[list(DEFAULT_SHOT_COLUMNS)].drop_duplicates().shape[0])


def _find_count_column(df: pd.DataFrame, requested: str) -> str:
    if requested in df.columns:
        return requested
    compatibility = {
        "count_corrected_template": "count_corrected_fixed_offset",
    }
    fallback = compatibility.get(requested)
    if fallback and fallback in df.columns:
        return fallback
    raise ValueError(f"processed table does not contain {requested!r}")


def shot_cluster_curve(
    data: pd.DataFrame,
    *,
    value_col: str,
    group_cols: Sequence[str],
    n_boot: int,
    seed: int,
    shot_cols: Sequence[str] = DEFAULT_SHOT_COLUMNS,
) -> pd.DataFrame:
    """Equal-shot mean curve with clustered and naive site-binomial intervals."""
    required = {*group_cols, *shot_cols, value_col}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"curve data are missing {sorted(missing)}")
    shot = (
        data.groupby([*group_cols, *shot_cols], observed=True, dropna=False)[value_col]
        .agg(["mean", "size"])
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    rows = []
    z = float(norm.ppf(0.975))
    grouper: str | list[str] = (
        group_cols[0] if len(group_cols) == 1 else list(group_cols)
    )
    for key, block in shot.groupby(grouper, observed=True, dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        values = block["mean"].to_numpy(float)
        draws = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            draws[i] = float(np.mean(rng.choice(values, size=len(values), replace=True)))
        n_site_observations = int(block["size"].sum())
        p_naive = float(
            np.sum(block["mean"].to_numpy(float) * block["size"].to_numpy(float))
            / n_site_observations
        )
        naive_se = float(
            np.sqrt(
                max(p_naive * (1.0 - p_naive), 0.0)
                / n_site_observations
            )
        )
        row = dict(zip(group_cols, key_tuple))
        row.update(
            {
                "estimate": float(np.mean(values)),
                "cluster_lower": float(np.quantile(draws, 0.025)),
                "cluster_upper": float(np.quantile(draws, 0.975)),
                "naive_lower": float(max(0.0, p_naive - z * naive_se)),
                "naive_upper": float(min(1.0, p_naive + z * naive_se)),
                "n_independent_shots": int(len(values)),
                "n_site_observations": n_site_observations,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _emission_candidates(
    df: pd.DataFrame,
    *,
    primary_count_col: str,
) -> tuple[dict[str, EmissionModel], pd.DataFrame]:
    train = df.loc[df["split"].astype(str) == "train"].copy()
    validation = df.loc[df["split"].astype(str) == "validation"].copy()
    models = fit_required_baselines(
        train,
        corrected_value_col=primary_count_col,
        frame_col="frame_index",
        split_col="split",
    )
    corrected = models["corrected_global_threshold"]
    frame = models["frame_pooled_mixture"]
    shared = models["shared_frame_offsets"]
    # Validate shrinkage instead of treating a single arbitrary strength as fixed.
    for strength in (5.0, 10.0, 20.0, 50.0, 100.0):
        name = f"shrinkage_site_offsets_k{int(strength)}"
        if strength == 20.0:
            models[name] = models["shrinkage_site_offsets"]
            continue
        models[name] = fit_emission_baseline(
            train,
            value_col=primary_count_col,
            kind="site_shrinkage",
            name=name,
            frame_col="frame_index",
            split_col="split",
            shrinkage_strength=strength,
            pooled_components=corrected.components,
            cached_frame_components=frame.frame_components,
            cached_frame_offsets=shared.frame_offsets,
            cached_frame_adjusted_components=shared.components,
        )

    rows = []
    for name, model in models.items():
        result = evaluate_heldout(
            model,
            validation,
            condition_cols="condition_id",
            include_occupancy=False,
            include_transitions=False,
        )
        rows.append(
            {
                **result.scalar_metrics(),
                "candidate": name,
                "kind": model.kind,
                "eligible_for_primary": name not in {
                    "raw_global_threshold",
                    "prototype_per_site_diagnostic",
                },
                "shrinkage_strength": model.shrinkage_strength,
            }
        )
    ranking = pd.DataFrame(rows)
    eligible = ranking.loc[ranking["eligible_for_primary"]].sort_values(
        ["mean_count_nll", "candidate"], kind="stable"
    )
    selected = str(eligible.iloc[0]["candidate"])
    ranking["selected_on_validation"] = ranking["candidate"] == selected
    ranking["delta_validation_nll_per_row"] = (
        ranking["mean_count_nll"] - float(eligible.iloc[0]["mean_count_nll"])
    )
    return models, ranking.sort_values(
        ["mean_count_nll", "candidate"], kind="stable"
    ).reset_index(drop=True)


def fit_emission_analysis(
    df: pd.DataFrame,
    *,
    primary_count_col: str = PRIMARY_COUNT_COLUMN,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select emissions on validation and score every baseline on test once."""
    count_col = _find_count_column(df, primary_count_col)
    models, validation_ranking = _emission_candidates(df, primary_count_col=count_col)
    selected_name = str(
        validation_ranking.loc[
            validation_ranking["selected_on_validation"], "candidate"
        ].iloc[0]
    )
    selected = models[selected_name]
    test = df.loc[df["split"].astype(str) == "test"].copy()
    test_rows = []
    test_evaluations = {}
    for name, model in models.items():
        result = evaluate_heldout(
            model,
            test,
            condition_cols="condition_id",
            include_occupancy=False,
            include_transitions=False,
        )
        test_evaluations[name] = result
        test_rows.append({**result.scalar_metrics(), "candidate": name})

    scored = selected.score(df)
    prototype = models["prototype_per_site_diagnostic"].parameter_table()
    weak = prototype.loc[
        (prototype["separation_d_prime"] < 2.0)
        | (prototype["model_implied_overlap"] > 0.10),
        "group",
    ].tolist()
    weak_sites = sorted(int(v) for v in weak if str(v) != "all")
    selected_test = evaluate_heldout(
        selected,
        test,
        condition_cols="condition_id",
        include_occupancy=True,
        include_transitions=True,
    )
    test_evaluations[selected_name] = selected_test
    report = {
        "selected_model": selected_name,
        "selected_kind": selected.kind,
        "selected_count_column": count_col,
        "split_counts": {
            split: _shot_count(df.loc[df["split"].astype(str) == split])
            for split in ("train", "validation", "test")
        },
        "validation_ranking": validation_ranking,
        "test_metrics_once_per_candidate": pd.DataFrame(test_rows).sort_values(
            ["mean_count_nll", "candidate"], kind="stable"
        ),
        "selected_test_apparent_occupancy": selected_test.apparent_occupancy,
        "selected_test_apparent_transitions": selected_test.apparent_transitions,
        "weak_site_rule": (
            "training-only per-site diagnostic d-prime < 2 or "
            "model-implied overlap > 0.10"
        ),
        "weak_site_ids": weak_sites,
        "n_weak_sites": len(weak_sites),
        "interpretation": (
            "Count likelihood, overlap, separation, entropy and calls are "
            "model-implied held-out quantities, not empirical fidelity."
        ),
    }
    return jsonable(report), {
        "models": models,
        "selected_model": selected,
        "scored": scored,
        "weak_sites": weak_sites,
        "test_evaluations": test_evaluations,
    }


def _dark_rate_column(model: str, interval: str) -> str:
    if model == "shared":
        suffix = "shared"
    elif model == "first_separate":
        suffix = "first_interval" if interval == "0->1" else "later_intervals"
    else:
        suffix = interval
    return f"lambda_switch_off__{suffix}"


def _dark_model_band(fit, bootstrap, *, n_grid: int = 60) -> pd.DataFrame:
    t_values = np.linspace(0.0, 2.2, n_grid)
    rows = []
    draws = bootstrap.draws
    for interval in fit.interval_levels:
        interval = str(interval)
        q_col = f"fixed_interreadout_survival__{interval}"
        rate_col = _dark_rate_column(fit.model, interval)
        if q_col not in draws or rate_col not in draws:
            continue
        q_draw = draws[q_col].to_numpy(float)
        rate_draw = draws[rate_col].to_numpy(float)
        for t in t_values:
            sample = q_draw * np.exp(-rate_draw * t)
            rows.append(
                {
                    "interval": interval,
                    "hold_s": float(t),
                    "prediction": float(
                        fit.q_by_interval[interval]
                        * np.exp(-fit.lambda_by_interval[interval] * t)
                    ),
                    "lower": float(np.quantile(sample, 0.025)),
                    "upper": float(np.quantile(sample, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def _selected_dark_rate(model: str) -> str | None:
    if model == "shared":
        return "lambda_switch_off__shared"
    if model == "first_separate":
        return "lambda_switch_off__later_intervals"
    return None


def _training_site_coordinates(scored: pd.DataFrame) -> pd.DataFrame:
    """Return frozen training geometry after checking all split copies agree."""
    coordinate_columns = ["site_row", "site_col", "site_y", "site_x"]
    required = {"site_id", "split", *coordinate_columns}
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"site-coordinate rows are missing {sorted(missing)}")
    coordinate_variants = (
        scored[["site_id", "split", *coordinate_columns]]
        .drop_duplicates()
        .groupby("site_id", observed=True)[coordinate_columns]
        .nunique(dropna=False)
    )
    if (coordinate_variants > 1).any().any():
        raise ValueError("site coordinates differ across frozen shot splits")
    coordinates = (
        scored.loc[
            scored["split"].astype(str) == "train",
            ["site_id", *coordinate_columns],
        ]
        .drop_duplicates("site_id")
        .sort_values("site_id", kind="stable")
    )
    if set(coordinates["site_id"]) != set(scored["site_id"].unique()):
        raise ValueError("one or more held-out sites are absent from training geometry")
    return coordinates


def dark_endpoint_sensitivity(
    development: pd.DataFrame,
    *,
    selected_model: str,
    primary_fit: Any | None = None,
) -> list[dict[str, Any]]:
    """Fit the four predeclared switch-off endpoint variants."""
    if development.empty:
        raise ValueError("endpoint sensitivity requires development rows")
    minimum = development["sweep_value_s"].min()
    maximum = development["sweep_value_s"].max()
    variants = {
        "all_points": development,
        "exclude_shortest": development.loc[
            development["sweep_value_s"] > minimum
        ],
        "exclude_longest": development.loc[
            development["sweep_value_s"] < maximum
        ],
        "exclude_both": development.loc[
            (development["sweep_value_s"] > minimum)
            & (development["sweep_value_s"] < maximum)
        ],
    }

    def values(fit) -> dict[str, Any]:
        selected_rate_name = _selected_dark_rate(fit.model)
        rate_group = (
            None
            if selected_rate_name is None
            else selected_rate_name.removeprefix("lambda_switch_off__")
        )
        rate = (
            None if rate_group is None else float(fit.lambda_groups[rate_group])
        )
        later = [
            interval
            for interval in fit.interval_levels
            if interval != fit.first_interval
        ]
        later_loss = float(
            1.0 - np.mean([fit.q_by_interval[interval] for interval in later])
        )
        return {
            "lambda_switch_off": rate,
            "tau_switch_off": (
                None if rate is None or rate <= 0.0 else 1.0 / rate
            ),
            "fixed_interreadout_survival": {
                str(key): float(value) for key, value in fit.q_by_interval.items()
            },
            "later_interval_apparent_loss": later_loss,
        }

    primary_fit = primary_fit or fit_dark_retention(
        development, model=selected_model, split_col=None
    )
    primary = values(primary_fit)
    rows: list[dict[str, Any]] = []
    for name, table in variants.items():
        included = sorted(
            float(value) for value in table["sweep_value_s"].unique()
        )
        try:
            fit = fit_dark_retention(
                table, model=selected_model, split_col=None
            )
            fitted = values(fit)
            rows.append(
                {
                    "variant": name,
                    "included_conditions_s": included,
                    **fitted,
                    "difference_from_primary": {
                        key: (
                            None
                            if fitted[key] is None or primary[key] is None
                            else float(fitted[key] - primary[key])
                        )
                        for key in (
                            "lambda_switch_off",
                            "tau_switch_off",
                            "later_interval_apparent_loss",
                        )
                    },
                    "fit_success": bool(fit.converged),
                    "parameter_on_boundary": bool(fit.parameter_on_boundary),
                    "optimizer_message": fit.optimizer_message,
                    "interpretation": (
                        "Endpoint sensitivity under the validation-frozen "
                        f"{selected_model} structure."
                    ),
                }
            )
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            rows.append(
                {
                    "variant": name,
                    "included_conditions_s": included,
                    "fit_success": False,
                    "fit_error": f"{type(exc).__name__}: {exc}",
                    "interpretation": (
                        "The predeclared endpoint sensitivity fit failed and "
                        "was retained explicitly."
                    ),
                }
            )
    return rows


def fit_dark_analysis(
    scored: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
    weak_sites: Sequence[int] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select and estimate the operational five-frame retention model."""
    events = make_retention_events(
        scored,
        frame_col="frame_index",
        condition_cols=("condition_id", "split"),
        passthrough_cols=("cycle_index",),
    )
    train = events.loc[events["split"].astype(str) == "train"].copy()
    validation = events.loc[events["split"].astype(str) == "validation"].copy()
    test = events.loc[events["split"].astype(str) == "test"].copy()
    selection = compare_dark_models(
        train,
        validation,
        split_col="split",
        tolerance_per_observation=1e-4,
    )
    selected_name = selection.selected_name
    development = events.loc[
        events["split"].astype(str).isin(["train", "validation"])
    ].copy()
    final_fit = fit_dark_retention(
        development,
        model=selected_name,
        split_col=None,
    )
    test_nll = float(final_fit.nll(test))
    bootstrap = bootstrap_dark_retention(
        development,
        model=selected_name,
        split_col=None,
        n_boot=n_boot,
        seed=seed,
    )
    intervals = bootstrap.intervals()
    interval_lookup = _interval_lookup(intervals)
    rate_parameter = _selected_dark_rate(selected_name)
    rate_ci = interval_lookup.get(rate_parameter) if rate_parameter else None
    rate_resolved = bool(
        rate_ci is not None
        and rate_ci["lower"] > 0.0
        and not final_fit.parameter_on_boundary
    )
    tau_ci = _invert_positive_rate_interval(rate_ci)

    multistart = diagnose_dark_bootstrap_multistart(
        development,
        model=selected_name,
        split_col=None,
        n_replicates=50,
        n_starts=5,
        seed=seed,
    )
    if not multistart.passed:
        raise RuntimeError(
            "dark bootstrap multi-start diagnostic found a materially "
            "distinct optimization mode"
        )
    cycle_seed = seed + 1000
    cycle_bootstrap = bootstrap_dark_retention_by_cycle(
        development,
        model=selected_name,
        split_col=None,
        n_boot=n_boot,
        seed=cycle_seed,
    )

    curve = shot_cluster_curve(
        events,
        value_col="retained_next",
        group_cols=("condition_id", "sweep_value_s", "interval"),
        n_boot=n_boot,
        seed=seed + 11,
    )
    site_curve = shot_cluster_curve(
        test,
        value_col="retained_next",
        group_cols=("site_id",),
        n_boot=n_boot,
        seed=seed + 14,
    )
    site_coordinates = _training_site_coordinates(scored)
    site_curve = site_curve.merge(
        site_coordinates,
        on="site_id",
        how="left",
        validate="one_to_one",
    )
    q_rows = []
    for interval, value in final_fit.q_by_interval.items():
        name = f"fixed_interreadout_survival__{interval}"
        q_rows.append(
            {
                "interval": str(interval),
                "estimate": float(value),
                "lower": interval_lookup[name]["lower"],
                "upper": interval_lookup[name]["upper"],
            }
        )

    sensitivity_rows = []
    no_first = development.loc[development["interval"].astype(str) != "0->1"]
    if no_first["interval"].nunique() >= 1:
        try:
            fit = fit_dark_retention(
                no_first, model="shared", split_col=None
            )
            sensitivity_rows.append(
                {
                    "variant": "exclude_0_to_1",
                    **fit.parameter_dict(),
                    "n_independent_shots": fit.n_independent_shots,
                }
            )
        except ValueError as exc:
            sensitivity_rows.append(
                {"variant": "exclude_0_to_1", "fit_error": str(exc)}
            )
    if weak_sites:
        strong = development.loc[~development["site_id"].isin(weak_sites)]
        try:
            fit = fit_dark_retention(
                strong, model=selected_name, split_col=None
            )
            sensitivity_rows.append(
                {
                    "variant": "exclude_predeclared_weak_sites",
                    **fit.parameter_dict(),
                    "n_independent_shots": fit.n_independent_shots,
                }
            )
        except ValueError as exc:
            sensitivity_rows.append(
                {"variant": "exclude_predeclared_weak_sites", "fit_error": str(exc)}
            )

    endpoint_rows = dark_endpoint_sensitivity(
        development,
        selected_model=selected_name,
        primary_fit=final_fit,
    )

    report = {
        "selected_model": selected_name,
        "model_selection": selection.table,
        "final_fit_scope": "training_plus_validation_after_structure_freeze",
        "test_scored_once": {
            "nll": test_nll,
            "mean_nll": test_nll / len(test),
            "n_events": len(test),
            "n_independent_shots": _shot_count(test),
        },
        "lambda_switch_off_parameter": rate_parameter,
        "lambda_switch_off": rate_ci,
        "tau_switch_off": tau_ci,
        "rate_resolved": rate_resolved,
        "fixed_interreadout_survival": q_rows,
        "clustered_apparent_retention_curve": curve,
        "heldout_per_site_apparent_retention": site_curve,
        "model_band": _dark_model_band(final_fit, bootstrap),
        "bootstrap": {
            "unit": "complete shot within condition",
            "n_requested": bootstrap.n_requested,
            "n_successful": bootstrap.n_successful,
            "n_failed": bootstrap.n_failed,
            "seed": bootstrap.seed,
            "intervals": intervals,
            "failures": bootstrap.failures,
        },
        "whole_cycle_bootstrap_sensitivity": {
            "unit": "complete acquisition cycle",
            "primary_condition_stratified_seed": bootstrap.seed,
            "seed": cycle_bootstrap.seed,
            "seeds_are_distinct": cycle_bootstrap.seed != bootstrap.seed,
            "n_requested": cycle_bootstrap.n_requested,
            "n_successful": cycle_bootstrap.n_successful,
            "n_failed": cycle_bootstrap.n_failed,
            "intervals": cycle_bootstrap.intervals(),
            "failures": cycle_bootstrap.failures,
            "interpretation": (
                "Sensitivity preserving repeated-cycle dependence across "
                "conditions; the condition-stratified complete-shot bootstrap "
                "remains primary."
            ),
        },
        "bootstrap_multistart_diagnostic": {
            **multistart.summary(),
            "replicates": multistart.rows,
            "subset_rule": (
                "first 50 condition-stratified bootstrap replicates from the "
                "primary bootstrap seed"
            ),
            "primary_one_start_bootstrap_retained": multistart.passed,
        },
        "endpoint_sensitivity": endpoint_rows,
        "sensitivity": pd.DataFrame(sensitivity_rows),
        "exact_loss_definition": "1 - q_j * exp(-lambda_switch_off * hold_s)",
        "interpretation": (
            "Operational retention under the switch-off command with DDS "
            "settings retained; not an intrinsic dark lifetime."
        ),
    }
    return jsonable(report), {
        "events": events,
        "development_events": development,
        "test_events": test,
        "fit": final_fit,
        "bootstrap": bootstrap,
        "cycle_bootstrap": cycle_bootstrap,
        "multistart_diagnostic": multistart,
        "selection": selection,
    }


def _bright_model_band(
    fit, bootstrap, *, max_time_s: float = 2.0, n_grid: int = 80
) -> pd.DataFrame:
    t = np.linspace(0.0, float(max_time_s), n_grid)
    draws = bootstrap.draws
    pi0 = draws["pi_0"].to_numpy(float)
    floor = draws["pi_floor"].to_numpy(float)
    rate = draws["lambda_bright_effective"].to_numpy(float)
    rows = []
    for value in t:
        sample = floor + (pi0 - floor) * np.exp(-rate * value)
        rows.append(
            {
                "wait_s": float(value),
                "prediction": float(fit.predict_time([value])[0]),
                "lower": float(np.quantile(sample, 0.025)),
                "upper": float(np.quantile(sample, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def _control_model_band(
    fit, bootstrap, *, max_time_s: float = 2.0, n_grid: int = 80
) -> pd.DataFrame:
    t = np.linspace(0.0, float(max_time_s), n_grid)
    draws = bootstrap.draws
    q0 = draws["post_wait_retention_q_0"].to_numpy(float)
    kappa = draws["post_wait_retention_kappa"].to_numpy(float)
    rows = []
    for value in t:
        sample = q0 * np.exp(-kappa * value)
        rows.append(
            {
                "wait_s": float(value),
                "prediction": float(fit.q_0 * np.exp(-fit.kappa * value)),
                "lower": float(np.quantile(sample, 0.025)),
                "upper": float(np.quantile(sample, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def _selected_kappa_semantics(
    fit,
    interval: Mapping[str, float],
    *,
    trend_resolved: bool,
) -> dict[str, Any]:
    """Serialize a flat structural zero separately from an estimated trend."""
    if fit.model == "flat":
        return {
            "selected_structure": "flat",
            "kappa_status": "structurally_fixed",
            "kappa_fixed_value": 0.0,
            "trend_resolved": False,
        }
    return {
        "selected_structure": fit.model,
        "kappa_status": "estimated_under_selected_monotone_structure",
        "kappa_fixed_value": None,
        "kappa_estimate": float(fit.kappa),
        "kappa_sampling_interval": dict(interval),
        "trend_resolved": bool(trend_resolved),
    }


def _shot_cycle_residual_slope(
    first_frame: pd.DataFrame, fit
) -> dict[str, Any]:
    if "cycle_index" not in first_frame.columns:
        return {"available": False}
    shot = (
        first_frame.groupby(
            ["run_id", "shot_id", "cycle_index", "sweep_value_s"],
            observed=True,
        )["apparent_occupied"]
        .mean()
        .reset_index()
    )
    shot["model_prediction"] = fit.predict_time(shot["sweep_value_s"].to_numpy(float))
    shot["residual"] = shot["apparent_occupied"] - shot["model_prediction"]
    x = shot["cycle_index"].to_numpy(float)
    y = shot["residual"].to_numpy(float)
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return {
        "available": True,
        "residual_slope_per_cycle": float(coef[1]),
        "residual_range_over_run": float(coef[1] * np.ptp(x)),
        "n_independent_shots": int(len(shot)),
        "interpretation": (
            "Sensitivity only: a cycle trend can indicate loading or run drift "
            "left after the bright-wait model."
        ),
    }


def fit_bright_analysis(
    scored: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
    weak_sites: Sequence[int] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select effective bright decay and the post-wait retention control."""
    first = scored.loc[scored["frame_index"] == 0].copy()
    train = first.loc[first["split"].astype(str) == "train"].copy()
    validation = first.loc[first["split"].astype(str) == "validation"].copy()
    test = first.loc[first["split"].astype(str) == "test"].copy()
    selection = compare_bright_models(
        train,
        validation,
        split_col="split",
        tolerance_per_observation=1e-4,
    )
    development = first.loc[
        first["split"].astype(str).isin(["train", "validation"])
    ].copy()
    final_fit = fit_bright_decay(
        development, model=selection.selected_name, split_col=None
    )
    test_nll = float(final_fit.nll(test))
    bootstrap = bootstrap_bright_decay(
        development,
        model=selection.selected_name,
        split_col=None,
        n_boot=n_boot,
        seed=seed,
    )
    rate_ci = _interval_lookup(bootstrap.intervals())[
        "lambda_bright_effective"
    ]
    tau_ci = _invert_positive_rate_interval(rate_ci)
    floor_sensitivity_fit = fit_bright_decay(
        development, model="floor", split_col=None
    )
    multistart = diagnose_bright_bootstrap_multistart(
        development,
        model=selection.selected_name,
        split_col=None,
        n_replicates=50,
        n_starts=5,
        seed=seed,
    )
    if not multistart.passed:
        raise RuntimeError(
            "bright bootstrap multi-start diagnostic found a materially "
            "distinct optimization mode"
        )
    cycle_seed = seed + 1000
    cycle_bootstrap = bootstrap_bright_decay_by_cycle(
        development,
        model=selection.selected_name,
        split_col=None,
        n_boot=n_boot,
        seed=cycle_seed,
    )

    control_events = make_retention_events(
        scored,
        frame_col="frame_index",
        condition_cols=("condition_id", "split"),
        passthrough_cols=("cycle_index",),
    )
    c_train = control_events.loc[
        control_events["split"].astype(str) == "train"
    ].copy()
    c_validation = control_events.loc[
        control_events["split"].astype(str) == "validation"
    ].copy()
    c_test = control_events.loc[
        control_events["split"].astype(str) == "test"
    ].copy()
    control_selection = compare_control_models(
        c_train,
        c_validation,
        split_col="split",
        tolerance_per_observation=1e-4,
    )
    c_development = control_events.loc[
        control_events["split"].astype(str).isin(["train", "validation"])
    ].copy()
    control_fit = fit_control_retention(
        c_development,
        model=control_selection.selected_name,
        split_col=None,
    )
    control_test_nll = float(control_fit.nll(c_test))
    control_bootstrap = bootstrap_control_retention(
        c_development,
        model=control_selection.selected_name,
        split_col=None,
        n_boot=n_boot,
        seed=seed + 1,
    )
    control_intervals = _interval_lookup(control_bootstrap.intervals())
    kappa_ci = control_intervals["post_wait_retention_kappa"]
    control_trend_resolved = bool(
        control_fit.model == "monotone"
        and kappa_ci["lower"] > 0.0
        and not control_fit.parameter_on_boundary
    )
    monotone_sensitivity_fit = fit_control_retention(
        c_development,
        model="monotone",
        split_col=None,
    )
    monotone_sensitivity_bootstrap = bootstrap_control_retention(
        c_development,
        model="monotone",
        split_col=None,
        n_boot=n_boot,
        seed=seed + 2,
    )
    monotone_intervals = _interval_lookup(
        monotone_sensitivity_bootstrap.intervals()
    )
    monotone_kappa = monotone_intervals["post_wait_retention_kappa"]
    monotone_trend_resolved = bool(
        monotone_kappa["lower"] > 0.0
        and not monotone_sensitivity_fit.parameter_on_boundary
    )

    occupancy_curve = shot_cluster_curve(
        first,
        value_col="apparent_occupied",
        group_cols=("condition_id", "sweep_value_s"),
        n_boot=n_boot,
        seed=seed + 12,
    )
    control_curve = shot_cluster_curve(
        control_events,
        value_col="retained_next",
        group_cols=("condition_id", "sweep_value_s"),
        n_boot=n_boot,
        seed=seed + 13,
    )

    sensitivity_rows = []
    if weak_sites:
        strong = development.loc[~development["site_id"].isin(weak_sites)]
        fit = fit_bright_decay(
            strong, model=selection.selected_name, split_col=None
        )
        sensitivity_rows.append(
            {
                "variant": "exclude_predeclared_weak_sites",
                **fit.parameter_dict(),
                "n_independent_shots": fit.n_independent_shots,
            }
        )

    report = {
        "selected_model": selection.selected_name,
        "model_selection": selection.table,
        "final_fit_scope": "training_plus_validation_after_structure_freeze",
        "test_scored_once": {
            "nll": test_nll,
            "mean_nll": test_nll / len(test),
            "n_site_observations": len(test),
            "n_independent_shots": _shot_count(test),
        },
        "lambda_bright_effective": rate_ci,
        "tau_bright_effective": tau_ci,
        "rate_resolved": bool(
            rate_ci["lower"] > 0.0 and not final_fit.parameter_on_boundary
        ),
        "pi_0": float(final_fit.pi_0),
        "pi_floor": float(final_fit.pi_floor),
        "clustered_apparent_occupancy_curve": occupancy_curve,
        "model_band": _bright_model_band(
            final_fit,
            bootstrap,
            max_time_s=float(first["sweep_value_s"].max()),
        ),
        "cycle_drift_sensitivity": _shot_cycle_residual_slope(first, final_fit),
        "post_wait_control": {
            "selected_model": control_selection.selected_name,
            **_selected_kappa_semantics(
                control_fit,
                kappa_ci,
                trend_resolved=control_trend_resolved,
            ),
            "model_selection": control_selection.table,
            "test_scored_once": {
                "nll": control_test_nll,
                "mean_nll": control_test_nll / len(c_test),
                "n_events": len(c_test),
                "n_independent_shots": _shot_count(c_test),
            },
            "q_0": control_intervals["post_wait_retention_q_0"],
            "alternative_model_sensitivity": {
                "structure": "monotone",
                "selected_on_validation": False,
                "q_0": monotone_intervals["post_wait_retention_q_0"],
                "kappa": monotone_kappa,
                "trend_resolved": monotone_trend_resolved,
                "parameter_on_boundary": (
                    monotone_sensitivity_fit.parameter_on_boundary
                ),
                "bootstrap": {
                    "unit": "complete shot within condition",
                    "n_requested": monotone_sensitivity_bootstrap.n_requested,
                    "n_successful": (
                        monotone_sensitivity_bootstrap.n_successful
                    ),
                    "n_failed": monotone_sensitivity_bootstrap.n_failed,
                    "seed": monotone_sensitivity_bootstrap.seed,
                    "failures": monotone_sensitivity_bootstrap.failures,
                },
                "interpretation": (
                    "Alternative-model sensitivity only; it did not determine "
                    "the selected structure and does not identify heating."
                ),
            },
            "clustered_curve": control_curve,
            "model_band": _control_model_band(
                control_fit,
                control_bootstrap,
                max_time_s=float(first["sweep_value_s"].max()),
            ),
            "interpretation": (
                "Validation selected a flat post-wait retention model. Kappa "
                "is fixed to zero by that selected structure; the data do not "
                "resolve a monotone post-wait trend."
            ),
        },
        "bootstrap": {
            "unit": "complete shot within condition",
            "n_requested": bootstrap.n_requested,
            "n_successful": bootstrap.n_successful,
            "n_failed": bootstrap.n_failed,
            "seed": bootstrap.seed,
            "intervals": bootstrap.intervals(),
            "failures": bootstrap.failures,
        },
        "whole_cycle_bootstrap_sensitivity": {
            "unit": "complete acquisition cycle",
            "primary_condition_stratified_seed": bootstrap.seed,
            "seed": cycle_bootstrap.seed,
            "seeds_are_distinct": cycle_bootstrap.seed != bootstrap.seed,
            "n_requested": cycle_bootstrap.n_requested,
            "n_successful": cycle_bootstrap.n_successful,
            "n_failed": cycle_bootstrap.n_failed,
            "intervals": cycle_bootstrap.intervals(),
            "failures": cycle_bootstrap.failures,
            "interpretation": (
                "Sensitivity preserving repeated-cycle dependence across "
                "conditions; the condition-stratified complete-shot bootstrap "
                "remains primary."
            ),
        },
        "bootstrap_multistart_diagnostic": {
            **multistart.summary(),
            "replicates": multistart.rows,
            "subset_rule": (
                "first 50 condition-stratified bootstrap replicates from the "
                "primary bootstrap seed"
            ),
            "primary_one_start_bootstrap_retained": multistart.passed,
        },
        "unresolved_floor_structure": {
            "fit_scope": (
                "training_plus_validation_after candidate structures were "
                "declared; test data were not used"
            ),
            "pi_0": floor_sensitivity_fit.pi_0,
            "pi_floor": floor_sensitivity_fit.pi_floor,
            "lambda_bright_effective": (
                floor_sensitivity_fit.lambda_bright_effective
            ),
            "tau_bright_effective": floor_sensitivity_fit.tau_bright_effective,
            "predicted_50ms_loss": float(
                1.0
                - np.exp(
                    -floor_sensitivity_fit.lambda_bright_effective * 0.05
                )
            ),
            "selected_on_validation": False,
            "interpretation": (
                "Unresolved model-structure sensitivity, not a confidence "
                "interval and not a replacement selected using test data."
            ),
        },
        "sensitivity": pd.DataFrame(sensitivity_rows),
        "interpretation": (
            "Effective apparent decay during the commanded illuminated wait; "
            "not an intrinsic bright-state lifetime."
        ),
    }
    return jsonable(report), {
        "first_frame": first,
        "development_first": development,
        "test_first": test,
        "fit": final_fit,
        "floor_sensitivity_fit": floor_sensitivity_fit,
        "bootstrap": bootstrap,
        "cycle_bootstrap": cycle_bootstrap,
        "multistart_diagnostic": multistart,
        "selection": selection,
        "control_events": control_events,
        "control_fit": control_fit,
        "control_bootstrap": control_bootstrap,
        "control_selection": control_selection,
    }


def _refit_emission_kind(
    df: pd.DataFrame,
    *,
    selected_model: EmissionModel,
    value_col: str,
) -> tuple[EmissionModel, pd.DataFrame]:
    train = df.loc[df["split"].astype(str) == "train"].copy()
    model = fit_emission_baseline(
        train,
        value_col=value_col,
        kind=selected_model.kind,
        name=f"{value_col}:{selected_model.kind}",
        frame_col="frame_index",
        split_col="split",
        shrinkage_strength=selected_model.shrinkage_strength or 20.0,
    )
    return model, model.score(df)


def background_model_sensitivity(
    df: pd.DataFrame,
    *,
    selected_emission: EmissionModel,
    response: str,
    selected_physical_model: str,
) -> pd.DataFrame:
    """Refit the frozen emission/physics structure across count corrections."""
    rows: list[dict[str, Any]] = []
    for method, requested in BACKGROUND_VALUE_COLUMNS.items():
        try:
            value_col = _find_count_column(df, requested)
            _model, scored = _refit_emission_kind(
                df, selected_model=selected_emission, value_col=value_col
            )
            if response == "dark":
                events = make_retention_events(
                    scored,
                    frame_col="frame_index",
                    condition_cols=("condition_id", "split"),
                )
                dev = events.loc[
                    events["split"].astype(str).isin(["train", "validation"])
                ]
                fit = fit_dark_retention(
                    dev, model=selected_physical_model, split_col=None
                )
                params = fit.parameter_dict()
            elif response == "bright":
                first = scored.loc[
                    (scored["frame_index"] == 0)
                    & scored["split"].astype(str).isin(["train", "validation"])
                ]
                fit = fit_bright_decay(
                    first, model=selected_physical_model, split_col=None
                )
                params = fit.parameter_dict()
                control_events = make_retention_events(
                    scored,
                    frame_col="frame_index",
                    condition_cols=("condition_id", "split"),
                )
                c_train = control_events.loc[
                    control_events["split"].astype(str) == "train"
                ]
                c_validation = control_events.loc[
                    control_events["split"].astype(str) == "validation"
                ]
                control_selection = compare_control_models(
                    c_train,
                    c_validation,
                    split_col="split",
                    tolerance_per_observation=1e-4,
                )
                c_development = control_events.loc[
                    control_events["split"]
                    .astype(str)
                    .isin(["train", "validation"])
                ]
                control_fit = fit_control_retention(
                    c_development,
                    model=control_selection.selected_name,
                    split_col=None,
                )
                params.update(
                    {
                        "post_wait_control_model": control_selection.selected_name,
                        **control_fit.parameter_dict(),
                    }
                )
            else:
                raise ValueError("response must be 'dark' or 'bright'")
            rows.append(
                {
                    "background_method": method,
                    "count_column": value_col,
                    "eligible_as_primary": method != "annulus_contaminated",
                    "n_independent_shots": fit.n_independent_shots,
                    "converged": fit.converged,
                    "parameter_on_boundary": fit.parameter_on_boundary,
                    **params,
                }
            )
        except (RuntimeError, ValueError) as exc:
            rows.append(
                {
                    "background_method": method,
                    "count_column": requested,
                    "eligible_as_primary": method != "annulus_contaminated",
                    "fit_error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def emission_model_sensitivity(
    df: pd.DataFrame,
    *,
    models: Mapping[str, EmissionModel],
    response: str,
    selected_physical_model: str,
) -> pd.DataFrame:
    """Refit the frozen physical structure across held-out emission baselines."""
    rows: list[dict[str, Any]] = []
    for name, model in models.items():
        try:
            scored = model.score(df)
            if response == "dark":
                events = make_retention_events(
                    scored,
                    frame_col="frame_index",
                    condition_cols=("condition_id", "split"),
                )
                development = events.loc[
                    events["split"]
                    .astype(str)
                    .isin(["train", "validation"])
                ]
                fit = fit_dark_retention(
                    development,
                    model=selected_physical_model,
                    split_col=None,
                )
                params = fit.parameter_dict()
            elif response == "bright":
                development = scored.loc[
                    (scored["frame_index"] == 0)
                    & scored["split"]
                    .astype(str)
                    .isin(["train", "validation"])
                ]
                fit = fit_bright_decay(
                    development,
                    model=selected_physical_model,
                    split_col=None,
                )
                params = fit.parameter_dict()
                events = make_retention_events(
                    scored,
                    frame_col="frame_index",
                    condition_cols=("condition_id", "split"),
                )
                c_train = events.loc[events["split"].astype(str) == "train"]
                c_validation = events.loc[
                    events["split"].astype(str) == "validation"
                ]
                control_selection = compare_control_models(
                    c_train,
                    c_validation,
                    split_col="split",
                    tolerance_per_observation=1e-4,
                )
                c_development = events.loc[
                    events["split"]
                    .astype(str)
                    .isin(["train", "validation"])
                ]
                control_fit = fit_control_retention(
                    c_development,
                    model=control_selection.selected_name,
                    split_col=None,
                )
                params.update(
                    {
                        "post_wait_control_model": control_selection.selected_name,
                        **control_fit.parameter_dict(),
                    }
                )
            else:
                raise ValueError("response must be 'dark' or 'bright'")
            rows.append(
                {
                    "emission_model": name,
                    "emission_kind": model.kind,
                    "primary_emission_eligible": name
                    not in {
                        "raw_global_threshold",
                        "prototype_per_site_diagnostic",
                    },
                    "n_fitted_emission_parameters_approximate": int(
                        5 * len(model.site_components)
                        if model.kind == "prototype_per_site"
                        else (
                            5 * len(model.frame_components)
                            if model.kind == "frame_pooled"
                            else len(model.site_offsets)
                            + len(model.frame_offsets)
                            + 5
                        )
                    ),
                    "n_independent_shots": fit.n_independent_shots,
                    "parameter_on_boundary": fit.parameter_on_boundary,
                    **params,
                }
            )
        except (RuntimeError, ValueError) as exc:
            rows.append(
                {
                    "emission_model": name,
                    "emission_kind": model.kind,
                    "fit_error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def seeded_split_sensitivity(
    df: pd.DataFrame,
    *,
    selected_emission: EmissionModel,
    response: str,
    selected_physical_model: str,
    seed: int,
) -> dict[str, Any]:
    """Repeat the frozen structures under an alternate whole-cycle split."""
    shot_cols = [
        c
        for c in (
            "shot_id",
            "shot_order",
            "condition_id",
            "sweep_value_s",
            "repetition_index",
            "cycle_index",
        )
        if c in df.columns
    ]
    shots = df[shot_cols].drop_duplicates("shot_id")
    manifest, metadata = build_cycle_split_manifest(
        shots, strategy=SEEDED_STRATEGY, seed=seed
    )
    alternate = attach_split(df, manifest)
    count_col = selected_emission.value_col
    model, scored = _refit_emission_kind(
        alternate,
        selected_model=selected_emission,
        value_col=count_col,
    )
    if response == "dark":
        events = make_retention_events(
            scored,
            frame_col="frame_index",
            condition_cols=("condition_id", "split"),
        )
        dev = events.loc[
            events["split"].astype(str).isin(["train", "validation"])
        ]
        fit = fit_dark_retention(
            dev, model=selected_physical_model, split_col=None
        )
    else:
        dev = scored.loc[
            (scored["frame_index"] == 0)
            & scored["split"].astype(str).isin(["train", "validation"])
        ]
        fit = fit_bright_decay(
            dev, model=selected_physical_model, split_col=None
        )
    return jsonable(
        {
            "split": metadata,
            "emission_kind": model.kind,
            "physical_model": selected_physical_model,
            "parameters": fit.parameter_dict(),
            "n_independent_shots": fit.n_independent_shots,
            "primary_background_template_note": (
                "The count table retains the primary chronological-training "
                "geometry/template; this sensitivity changes statistical split "
                "assignment, while all-data geometry is assessed separately."
            ),
        }
    )


def background_occupancy_coupling(
    scored: pd.DataFrame,
) -> dict[str, Any]:
    """Correlate site-free background with frozen apparent occupancy by shot."""
    from .sweep_validation import safe_correlation

    background_cols = [
        c
        for c in (
            "background_global",
            "background_spatial",
            "background_template_offset",
            "background_fixed_offset",
            "background_annulus_contaminated",
        )
        if c in scored.columns
    ]
    group = ["run_id", "shot_id", "shot_order", "frame_index", "sweep_value_s"]
    agg: dict[str, str] = {"apparent_occupied": "mean"}
    agg.update({c: "mean" for c in background_cols})
    shot = scored.groupby(group, observed=True).agg(agg).reset_index()
    out: dict[str, Any] = {}
    for frame, part in shot.groupby("frame_index", observed=True):
        out[str(int(frame))] = {
            col: {
                "background_vs_apparent_occupancy_r": safe_correlation(
                    part[col], part["apparent_occupied"]
                ),
                "background_change_shortest_to_longest": float(
                    part.loc[
                        np.isclose(
                            part["sweep_value_s"],
                            float(part["sweep_value_s"].max()),
                        ),
                        col,
                    ].mean()
                    - part.loc[
                        np.isclose(
                            part["sweep_value_s"],
                            float(part["sweep_value_s"].min()),
                        ),
                        col,
                    ].mean()
                ),
            }
            for col in background_cols
        }
    return jsonable(out)


def site_free_background_curve(
    scored: pd.DataFrame,
    *,
    selected_count_column: str,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Condition-level ROI-equivalent background with complete-shot intervals."""
    background_col = COUNT_TO_BACKGROUND_COLUMN.get(selected_count_column)
    if background_col is None or background_col not in scored.columns:
        return {
            "available": False,
            "reason": "selected count has no eligible site-free background column",
        }
    curve = shot_cluster_curve(
        scored,
        value_col=background_col,
        group_cols=("condition_id", "sweep_value_s", "frame_index"),
        n_boot=n_boot,
        seed=seed,
    )
    return jsonable(
        {
            "available": True,
            "background_column": background_col,
            "units": "ROI-equivalent camera counts",
            "scope": (
                "Selected training-shot template plus per-frame site-free "
                "offset, averaged across sites within each complete shot."
            ),
            "curve": curve,
        }
    )


def compare_datasets(
    dark_context: Mapping[str, Any],
    bright_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Joint-bootstrap rate ratio and 50 ms predicted-versus-observed loss."""
    dark_fit = dark_context["fit"]
    dark_boot = dark_context["bootstrap"]
    bright_fit = bright_context["fit"]
    bright_boot = bright_context["bootstrap"]
    dark_parameter = _selected_dark_rate(dark_fit.model)
    if dark_parameter is None or dark_parameter not in dark_boot.draws:
        return {
            "rate_ratio_available": False,
            "reason": (
                "The selected switch-off model has interval-specific slopes, "
                "so no single rate ratio is identified."
            ),
            "sequences_pooled": False,
        }
    n = min(len(dark_boot.draws), len(bright_boot.draws))
    dark_rate = dark_boot.draws[dark_parameter].to_numpy(float)[:n]
    bright_rate = bright_boot.draws["lambda_bright_effective"].to_numpy(float)[:n]
    valid = dark_rate > 0.0
    ratio = bright_rate[valid] / dark_rate[valid]
    predicted = 1.0 - np.exp(-bright_rate * 0.05)

    q_columns = [
        f"fixed_interreadout_survival__{interval}"
        for interval in dark_fit.interval_levels
        if str(interval) != "0->1"
    ]
    q_columns = [c for c in q_columns if c in dark_boot.draws]
    fixed_survival = dark_boot.draws[q_columns].mean(axis=1).to_numpy(float)[:n]
    observed_fixed_loss = 1.0 - fixed_survival
    gap = observed_fixed_loss - predicted

    later_intervals = [
        interval for interval in dark_fit.interval_levels
        if str(interval) != "0->1"
    ]
    point_ratio = (
        float(bright_fit.lambda_bright_effective)
        / float(dark_fit.lambda_groups[dark_parameter.removeprefix(
            "lambda_switch_off__"
        )])
    )
    point_predicted = float(
        1.0 - np.exp(-float(bright_fit.lambda_bright_effective) * 0.05)
    )
    point_fixed_loss = float(
        1.0
        - np.mean([dark_fit.q_by_interval[interval] for interval in later_intervals])
    )
    point_gap = point_fixed_loss - point_predicted

    def summary(
        values: np.ndarray, point: float
    ) -> dict[str, float | int | None]:
        values = values[np.isfinite(values)]
        if not len(values):
            return {"estimate": None, "lower": None, "upper": None, "n": 0}
        return {
            "estimate": float(point),
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
            "n": int(len(values)),
        }

    discrepancy_resolved = bool(
        len(gap) and np.quantile(gap, 0.025) > 0.0
    )
    report = {
        "rate_ratio_available": bool(len(ratio)),
        "lambda_bright_over_lambda_switch_off": summary(ratio, point_ratio),
        "bright_model_predicted_loss_over_50ms": summary(
            predicted, point_predicted
        ),
        "apparent_later_interval_fixed_loss": summary(
            observed_fixed_loss, point_fixed_loss
        ),
        "observed_minus_predicted_loss": summary(gap, point_gap),
        "predicted_vs_observed_discrepancy_resolved": discrepancy_resolved,
        # Failure to resolve a discrepancy would not establish that this
        # simple model explains the physical loss, so this is never promoted
        # to an affirmative conclusion from these unmatched sequences.
        "simple_bright_model_explains_full_interreadout_loss": False,
        "allowed_conclusion": (
            "The simple constant-rate bright-wait model does not explain the "
            "full apparent inter-readout loss."
            if discrepancy_resolved
            else "The current clustered interval does not resolve a discrepancy."
        ),
        "fixed_per_pulse_cost_proven": False,
        "sequences_pooled": False,
        "pooling_reason": (
            "Separate acquisitions have a one-pitch coordinate shift, distinct "
            "background trajectories, and no randomized cross-sequence control."
        ),
    }
    floor_fit = bright_context.get("floor_sensitivity_fit")
    if floor_fit is not None:
        alternative_predicted = float(
            1.0 - np.exp(-floor_fit.lambda_bright_effective * 0.05)
        )
        alternative_gap = point_fixed_loss - alternative_predicted
        selected_gap_pp = 100.0 * point_gap
        alternative_gap_pp = 100.0 * alternative_gap
        report["structural_sensitivity"] = {
            "selected_structure": bright_fit.model,
            "selected_predicted_loss_percentage": 100.0 * point_predicted,
            "selected_gap_percentage_points": selected_gap_pp,
            "selected_sampling_ci_percentage_points": [
                100.0 * float(np.quantile(gap, 0.025)),
                100.0 * float(np.quantile(gap, 0.975)),
            ],
            "alternative_structure": "floor",
            "alternative_fit_scope": (
                "training_plus_validation after validation-frozen candidate "
                "definition; test data were not used"
            ),
            "alternative_floor": float(floor_fit.pi_floor),
            "alternative_effective_rate_per_s": float(
                floor_fit.lambda_bright_effective
            ),
            "alternative_predicted_loss_percentage": (
                100.0 * alternative_predicted
            ),
            "alternative_gap_percentage_points": alternative_gap_pp,
            "structural_range_percentage_points": sorted(
                [selected_gap_pp, alternative_gap_pp]
            ),
            "gap_sign_positive_under_both_structures": bool(
                point_gap > 0.0 and alternative_gap > 0.0
            ),
            "interpretation": (
                "The selected interval is complete-shot sampling uncertainty "
                "conditional on the no-floor structure. The range across the "
                "selected and unresolved floor structures is model-structure "
                "sensitivity, not a confidence interval or total uncertainty."
            ),
        }
    return report


def compare_cycle_bootstraps(
    dark_context: Mapping[str, Any],
    bright_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-dataset sensitivity under complete acquisition-cycle resampling."""
    dark_fit = dark_context["fit"]
    bright_fit = bright_context["fit"]
    dark_boot = dark_context["cycle_bootstrap"]
    bright_boot = bright_context["cycle_bootstrap"]
    dark_parameter = _selected_dark_rate(dark_fit.model)
    if dark_parameter is None:
        return {
            "available": False,
            "reason": "selected dark structure has no single comparable rate",
            "qualitative_conclusion_consistent": False,
        }
    n = min(len(dark_boot.draws), len(bright_boot.draws))
    dark_rate = dark_boot.draws[dark_parameter].to_numpy(float)[:n]
    bright_rate = bright_boot.draws[
        "lambda_bright_effective"
    ].to_numpy(float)[:n]
    predicted = bright_boot.draws["predicted_50ms_loss"].to_numpy(float)[:n]
    later_loss = dark_boot.draws[
        "apparent_later_interval_fixed_loss"
    ].to_numpy(float)[:n]
    gap = later_loss - predicted

    def interval(values: np.ndarray, estimate: float) -> dict[str, Any]:
        finite = values[np.isfinite(values)]
        return {
            "estimate": float(estimate),
            "lower": (
                None if not len(finite) else float(np.quantile(finite, 0.025))
            ),
            "upper": (
                None if not len(finite) else float(np.quantile(finite, 0.975))
            ),
            "n_successful_joint_draws": int(len(finite)),
        }

    rate_group = dark_parameter.removeprefix("lambda_switch_off__")
    point_dark_rate = float(dark_fit.lambda_groups[rate_group])
    point_bright_rate = float(bright_fit.lambda_bright_effective)
    point_predicted = float(1.0 - np.exp(-point_bright_rate * 0.05))
    later = [
        value
        for value in dark_fit.interval_levels
        if value != dark_fit.first_interval
    ]
    point_later_loss = float(
        1.0 - np.mean([dark_fit.q_by_interval[value] for value in later])
    )
    point_gap = point_later_loss - point_predicted
    dark_interval = interval(dark_rate, point_dark_rate)
    tau_values = np.divide(
        1.0,
        dark_rate,
        out=np.full_like(dark_rate, np.nan),
        where=dark_rate > 0.0,
    )
    tau_interval = interval(tau_values, 1.0 / point_dark_rate)
    bright_interval = interval(bright_rate, point_bright_rate)
    predicted_interval = interval(predicted, point_predicted)
    later_interval = interval(later_loss, point_later_loss)
    gap_interval = interval(gap, point_gap)
    resolved_rates = bool(
        dark_interval["lower"] is not None
        and dark_interval["lower"] > 0.0
        and bright_interval["lower"] is not None
        and bright_interval["lower"] > 0.0
    )
    positive_gap = bool(
        gap_interval["lower"] is not None and gap_interval["lower"] > 0.0
    )
    return {
        "available": True,
        "unit": "complete acquisition cycle within each run",
        "dark_seed": dark_boot.seed,
        "bright_seed": bright_boot.seed,
        "seeds_are_distinct": dark_boot.seed != bright_boot.seed,
        "dark_n_requested": dark_boot.n_requested,
        "dark_n_successful": dark_boot.n_successful,
        "dark_n_failed": dark_boot.n_failed,
        "dark_failures": dark_boot.failures,
        "bright_n_requested": bright_boot.n_requested,
        "bright_n_successful": bright_boot.n_successful,
        "bright_n_failed": bright_boot.n_failed,
        "bright_failures": bright_boot.failures,
        "lambda_switch_off": dark_interval,
        "tau_switch_off": tau_interval,
        "lambda_bright_effective": bright_interval,
        "predicted_50ms_loss": predicted_interval,
        "later_interval_apparent_loss": later_interval,
        "observed_minus_predicted_apparent_loss": gap_interval,
        "rates_resolved": resolved_rates,
        "apparent_gap_positive": positive_gap,
        "qualitative_conclusion_consistent": bool(
            resolved_rates and positive_gap
        ),
        "interpretation": (
            "Sensitivity to dependence shared by repeated acquisition cycles; "
            "it does not replace the primary condition-stratified complete-shot "
            "sampling interval."
        ),
    }


def _trajectory_arrays(
    scored: pd.DataFrame,
    *,
    value_col: str,
) -> dict[str, Any]:
    identity = [
        "run_id",
        "shot_id",
        "shot_order",
        "site_id",
        "condition_id",
        "sweep_value_s",
        "split",
    ]
    index = scored[identity].drop_duplicates(
        ["run_id", "shot_id", "site_id"]
    )
    values = scored.pivot_table(
        index=["run_id", "shot_id", "site_id"],
        columns="frame_index",
        values=value_col,
        aggfunc="first",
        observed=True,
    ).sort_index(axis=1)
    index = index.set_index(["run_id", "shot_id", "site_id"]).loc[values.index]
    observations = values.to_numpy(float)
    n_intervals = observations.shape[1] - 1
    hold = np.repeat(
        index["sweep_value_s"].to_numpy(float)[:, None],
        n_intervals,
        axis=1,
    )
    return {
        "index": index.reset_index(),
        "observations": observations,
        "transition_times": hold,
    }


def _subset_trajectories(trajectories: Mapping[str, Any], split: str) -> dict[str, Any]:
    mask = trajectories["index"]["split"].astype(str).to_numpy() == split
    return {
        "index": trajectories["index"].loc[mask].reset_index(drop=True),
        "observations": trajectories["observations"][mask],
        "transition_times": trajectories["transition_times"][mask],
    }


def _latent_ppc(fit, observations: np.ndarray, transition_times: np.ndarray, seed: int):
    simulated = simulate_latent_state(
        fit.parameters,
        fit.spec,
        transition_times,
        random_seed=seed,
    )
    observed_mean = np.mean(observations, axis=0)
    simulated_mean = np.mean(simulated.observations, axis=0)
    observed_sd = np.std(observations, axis=0, ddof=1)
    simulated_sd = np.std(simulated.observations, axis=0, ddof=1)
    mean_shift_sd = np.abs(simulated_mean - observed_mean) / np.maximum(
        observed_sd, 1e-9
    )
    sd_ratio = simulated_sd / np.maximum(observed_sd, 1e-9)
    passed = bool(
        np.all(mean_shift_sd < 0.25)
        and np.all((sd_ratio > 0.70) & (sd_ratio < 1.30))
    )
    return {
        "passed": passed,
        "observed_mean": observed_mean,
        "simulated_mean": simulated_mean,
        "mean_shift_in_observed_sd": mean_shift_sd,
        "observed_sd": observed_sd,
        "simulated_sd": simulated_sd,
        "simulated_to_observed_sd_ratio": sd_ratio,
        "thresholds": {
            "max_abs_mean_shift_sd": 0.25,
            "sd_ratio_range": [0.70, 1.30],
        },
    }


def _latent_synthetic_recovery(
    fit,
    transition_times: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    n = min(len(transition_times), 2500)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(transition_times), size=n, replace=False)
    known = fit.parameters
    synthetic = simulate_latent_state(
        known,
        fit.spec,
        transition_times[chosen],
        random_seed=seed + 1,
    )
    recovered = fit_latent_state(
        synthetic.observations,
        synthetic.transition_times_s,
        fit.spec,
        n_starts=5,
        random_seed=seed + 2,
        max_iterations=250,
    )
    known_rate = np.asarray(known.switch_off_rate_per_s, float)
    found_rate = np.asarray(recovered.parameters.switch_off_rate_per_s, float)
    rate_relative_error = np.abs(found_rate - known_rate) / np.maximum(
        known_rate, 1e-6
    )
    q_error = np.abs(
        recovered.parameters.fixed_interreadout_survival
        - known.fixed_interreadout_survival
    )
    passed = bool(
        recovered.converged
        and np.max(rate_relative_error) < 0.35
        and np.max(q_error) < 0.10
    )
    return {
        "passed": passed,
        "n_trajectories": n,
        "known_rate": known_rate,
        "recovered_rate": found_rate,
        "rate_relative_error": rate_relative_error,
        "known_q": known.fixed_interreadout_survival,
        "recovered_q": recovered.parameters.fixed_interreadout_survival,
        "q_absolute_error": q_error,
        "thresholds": {
            "max_rate_relative_error": 0.35,
            "max_q_absolute_error": 0.10,
        },
    }


def _profile_for_fit(fit, train: Mapping[str, Any]):
    mle = float(fit.parameters.switch_off_rate_per_s[0])
    lo = max(fit.spec.rate_bounds_per_s[0] * 1.01, mle / 20.0)
    hi = min(fit.spec.rate_bounds_per_s[1] / 1.01, max(mle * 20.0, lo * 2.0))
    grid = np.geomspace(lo, hi, 31)
    return profile_switch_off_rate(
        fit,
        train["observations"],
        train["transition_times"],
        grid,
        interval_index=0,
        max_iterations=120,
    )


def _latent_transition_curvature(
    fit,
    train: Mapping[str, Any],
    *,
    step: float = 1e-3,
) -> dict[str, Any]:
    """Local observed curvature for transition parameters with emissions fixed."""
    parameters = fit.parameters
    q = np.asarray(parameters.fixed_interreadout_survival, dtype=float)
    rates = np.asarray(parameters.switch_off_rate_per_s, dtype=float)
    theta = np.r_[
        logit(parameters.initial_occupancy),
        logit(q),
        np.log(rates),
    ]
    names = [
        "logit_initial_occupancy",
        *[f"logit_fixed_survival_{i}_to_{i + 1}" for i in range(len(q))],
        *[
            (
                "log_lambda_switch_off_shared"
                if len(rates) == 1
                else f"log_lambda_switch_off_interval_{i}"
            )
            for i in range(len(rates))
        ],
    ]

    def decode(value: np.ndarray) -> LatentStateParameters:
        offset = 1 + len(q)
        return replace(
            parameters,
            initial_occupancy=float(expit(value[0])),
            fixed_interreadout_survival=np.asarray(
                expit(value[1:offset]), dtype=float
            ),
            switch_off_rate_per_s=np.asarray(
                np.exp(value[offset:]), dtype=float
            ),
        )

    def objective(value: np.ndarray) -> float:
        return -float(
            forward_log_likelihood(
                train["observations"],
                train["transition_times"],
                decode(value),
                fit.spec,
            ).total_log_likelihood
        )

    n = len(theta)
    hessian = np.empty((n, n), dtype=float)
    f0 = objective(theta)
    basis = np.eye(n) * step
    for i in range(n):
        hessian[i, i] = (
            objective(theta + basis[i])
            - 2.0 * f0
            + objective(theta - basis[i])
        ) / step**2
        for j in range(i):
            value = (
                objective(theta + basis[i] + basis[j])
                - objective(theta + basis[i] - basis[j])
                - objective(theta - basis[i] + basis[j])
                + objective(theta - basis[i] - basis[j])
            ) / (4.0 * step**2)
            hessian[i, j] = value
            hessian[j, i] = value
    eigenvalues = np.linalg.eigvalsh(hessian)
    positive_definite = bool(np.all(eigenvalues > 0.0))
    covariance = np.linalg.pinv(hessian, hermitian=True)
    scale = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = np.outer(scale, scale)
    correlation = np.divide(
        covariance,
        denominator,
        out=np.full_like(covariance, np.nan),
        where=denominator > 0.0,
    )
    off_diagonal = correlation[~np.eye(n, dtype=bool)]
    max_abs_correlation = float(np.nanmax(np.abs(off_diagonal)))
    condition_number = float(np.linalg.cond(hessian))
    identifiable = bool(
        positive_definite
        and np.isfinite(condition_number)
        and condition_number < 1e10
        and max_abs_correlation < 0.995
    )
    return {
        "parameterization": names,
        "hessian_eigenvalues": eigenvalues,
        "hessian_condition_number": condition_number,
        "positive_definite": positive_definite,
        "correlation_matrix": correlation,
        "max_absolute_off_diagonal_correlation": max_abs_correlation,
        "identifiability_diagnostic_passed": identifiable,
        "scope": (
            "Local observed-likelihood curvature for initial occupancy, q_j, "
            "and rate with fitted emissions held fixed. The nuisance-refitted "
            "profile likelihood is the primary rate uncertainty."
        ),
    }


def _latent_parameter_dict(parameters: LatentStateParameters) -> dict[str, Any]:
    return {
        "initial_occupancy": parameters.initial_occupancy,
        "fixed_interreadout_survival": parameters.fixed_interreadout_survival,
        "switch_off_rate_per_s": parameters.switch_off_rate_per_s,
        "dark_mean": parameters.dark_mean,
        "bright_mean": parameters.bright_mean,
        "dark_sigma": parameters.dark_sigma,
        "bright_sigma": parameters.bright_sigma,
        "reload_probability": parameters.reload_probability,
    }


def fit_latent_analysis(
    scored: pd.DataFrame,
    *,
    selected_emission: EmissionModel,
    timing_verified: bool,
    geometry_validated: bool,
    background_frozen: bool,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit candidate HMM structures and apply the real-data acceptance gate."""
    trajectories = _trajectory_arrays(
        scored, value_col=selected_emission.value_col
    )
    train = _subset_trajectories(trajectories, "train")
    validation = _subset_trajectories(trajectories, "validation")
    test = _subset_trajectories(trajectories, "test")

    candidate_fits = {}
    validation_rows = []
    val_rows = scored.loc[scored["split"].astype(str) == "validation"]
    baseline_val_ll = float(
        selected_emission.score(val_rows)["log_predictive_density"].sum()
    )
    for name, shared in (("shared_rate", True), ("interval_specific_rate", False)):
        spec = LatentModelSpec(n_frames=5, shared_switch_off_rate=shared)
        fit = fit_latent_state(
            train["observations"],
            train["transition_times"],
            spec,
            n_starts=8,
            random_seed=seed + int(not shared),
            max_iterations=300,
        )
        likelihood = forward_log_likelihood(
            validation["observations"],
            validation["transition_times"],
            fit.parameters,
            spec,
        )
        delta = likelihood.total_log_likelihood - baseline_val_ll
        candidate_fits[name] = fit
        validation_rows.append(
            {
                "candidate": name,
                "validation_log_likelihood": likelihood.total_log_likelihood,
                "independent_frame_baseline_log_likelihood": baseline_val_ll,
                "delta_log_likelihood": delta,
                "delta_per_trajectory": delta / len(validation["observations"]),
                "n_transition_rates": spec.n_switch_off_rates,
                "converged": fit.converged,
                "boundary_parameters": list(fit.diagnostics.boundary_parameters),
            }
        )
    ranking = pd.DataFrame(validation_rows).sort_values(
        ["validation_log_likelihood", "n_transition_rates"],
        ascending=[False, True],
        kind="stable",
    )
    best = ranking.iloc[0]
    close = ranking.loc[
        ranking["validation_log_likelihood"]
        >= float(best["validation_log_likelihood"])
        - 0.001 * len(validation["observations"])
    ].sort_values("n_transition_rates")
    selected_name = str(close.iloc[0]["candidate"])
    fit = candidate_fits[selected_name]

    test_rows = scored.loc[scored["split"].astype(str) == "test"]
    baseline_test_ll = float(
        selected_emission.score(test_rows)["log_predictive_density"].sum()
    )
    comparison = score_against_held_out_baseline(
        fit.parameters,
        fit.spec,
        test["observations"],
        test["transition_times"],
        baseline_test_ll,
    )
    ppc = _latent_ppc(
        fit,
        train["observations"],
        train["transition_times"],
        seed + 20,
    )
    synthetic = _latent_synthetic_recovery(
        fit, train["transition_times"], seed=seed + 30
    )
    profile = _profile_for_fit(fit, train)
    curvature = _latent_transition_curvature(fit, train)
    evidence = LatentGateEvidence(
        timing_semantics_verified=timing_verified,
        geometry_validated=geometry_validated,
        background_method_frozen=background_frozen,
        shot_split_frozen=True,
        held_out_baselines_complete=True,
        synthetic_recovery_passed=bool(synthetic["passed"]),
        posterior_predictive_checks_passed=bool(ppc["passed"]),
        # The current HMM engine profiles the ordinary trajectory likelihood.
        # It does not yet refit emissions/transitions under complete-shot
        # resampling, so the exploratory fit must not pass the public gate.
        shot_cluster_uncertainty_passed=False,
        held_out_comparison=comparison,
        rate_profile=profile,
    )
    decision = evaluate_latent_state_gate(
        fit,
        evidence,
        LatentGateCriteria(
            min_delta_loglik_per_trajectory=0.0,
            min_frame_separation_d_prime=1.0,
            min_near_best_starts=2,
            max_start_agreement_distance=0.25,
            max_profile_confidence_ratio=20.0,
            reject_boundary_parameters=True,
        ),
    )
    gate_failures = list(decision.failures)
    gate_cautions = list(decision.cautions)
    if not curvature["identifiability_diagnostic_passed"]:
        gate_failures.append(
            "Local transition-parameter curvature is singular, ill-conditioned, "
            "or has an absolute correlation at or above 0.995."
        )
    gate_passed = bool(decision.accepted and not gate_failures)

    representative = None
    if gate_passed:
        posterior = posterior_state_probabilities(
            test["observations"],
            test["transition_times"],
            fit.parameters,
            fit.spec,
        )
        p = np.clip(posterior.state_probability, 1e-12, 1.0 - 1e-12)
        entropy = np.mean(-(p * np.log(p) + (1 - p) * np.log1p(-p)), axis=1)
        target = float(np.median(entropy))
        index = int(np.argmin(np.abs(entropy - target)))
        meta = test["index"].iloc[index]
        baseline_calls = (
            selected_emission.score(
                test_rows.loc[
                    (test_rows["shot_id"] == meta["shot_id"])
                    & (test_rows["site_id"] == meta["site_id"])
                ].sort_values("frame_index")
            )["apparent_occupied"]
            .astype(int)
            .tolist()
        )
        representative = {
            "selection_rule": (
                "test trajectory whose mean posterior entropy is closest to "
                "the test-set median"
            ),
            "shot_order": int(meta["shot_order"]),
            "site_id": int(meta["site_id"]),
            "hold_s": float(meta["sweep_value_s"]),
            "observed_corrected_counts": test["observations"][index],
            "threshold_baseline_calls": baseline_calls,
            "posterior_state_probability": posterior.state_probability[index],
            "posterior_occupied_probability": posterior.state_probability[
                index, :, 1
            ],
        }

    report = {
        "gate_passed": gate_passed,
        "gate_failures": gate_failures,
        "gate_cautions": gate_cautions,
        "selected_structure": selected_name,
        "validation_structure_selection": ranking,
        "parameters": _latent_parameter_dict(fit.parameters),
        "training_fit": {
            "converged": fit.converged,
            "log_likelihood": fit.log_likelihood,
            "n_trajectories": fit.n_trajectories,
            "selected_start": fit.selected_start,
            "starts": fit.starts,
            "diagnostics": fit.diagnostics,
        },
        "held_out_comparison": comparison,
        "profile_likelihood": profile,
        "transition_parameter_correlations": curvature,
        "posterior_predictive_checks": ppc,
        "synthetic_recovery": synthetic,
        "uncertainty_scope": {
            "profile_likelihood": (
                "working-independence exploratory profile over site "
                "trajectories; not a complete-shot clustered interval"
            ),
            "complete_shot_cluster_bootstrap_available": False,
            "public_gate_requires_complete_shot_clustering": True,
        },
        "physical_rate_gate_rationale": {
            "held_out_likelihood_improvement_supports_predictive_structure": (
                comparison.latent_predicts_better
            ),
            "complete_shot_clustered_transition_uncertainty_available": False,
            "synthetic_rate_recovery_max_relative_error": float(
                np.max(synthetic["rate_relative_error"])
            ),
            "transition_rate_weakly_identified_for_physical_claim": True,
            "interpretation": (
                "Held-out likelihood improvement supports predictive sequence "
                "structure, not a precise physical transition rate. Complete-"
                "shot clustered transition uncertainty is absent and the "
                "declared synthetic recovery experiment has approximately "
                f"{100.0 * float(np.max(synthetic['rate_relative_error'])):.1f}% "
                "relative rate error."
            ),
        },
        "representative_example": representative,
        "interpretation": (
            "Predictive latent-state structure improves held-out likelihood, "
            "but the transition rate remains weakly identified for the intended "
            "physical claim. Parameters remain conditional on the stated model "
            "and are not empirical fidelity."
        ),
    }
    return jsonable(report), {
        "fit": fit,
        "decision": decision,
        "trajectories": trajectories,
        "test": test,
        "profile": profile,
        "comparison": comparison,
        "representative": representative,
    }


def _validate_analysis_table(
    data: pd.DataFrame,
    *,
    dataset_label: str,
    expected_frames: int,
) -> None:
    """Reject a sweep table whose frozen shot contract is not intact."""
    required = {
        "run_id",
        "shot_id",
        "shot_order",
        "condition_id",
        "sweep_value_s",
        "frame_index",
        "site_id",
        "split",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"{dataset_label} analysis table is missing {sorted(missing)}"
        )
    if data.empty:
        raise ValueError(f"{dataset_label} analysis table is empty")
    observed_frames = sorted(data["frame_index"].dropna().astype(int).unique())
    if observed_frames != list(range(expected_frames)):
        raise ValueError(
            f"{dataset_label} has frames {observed_frames}, expected "
            f"{list(range(expected_frames))}"
        )

    shot_split = (
        data[["run_id", "shot_id", "split"]]
        .drop_duplicates()
        .groupby(["run_id", "shot_id"], observed=True)["split"]
        .nunique()
    )
    if int(shot_split.max()) != 1:
        raise ValueError(f"{dataset_label} splits one or more shots")
    allowed = {"train", "validation", "test"}
    found = set(data["split"].dropna().astype(str).unique())
    if found != allowed:
        raise ValueError(
            f"{dataset_label} split labels are {sorted(found)}, expected "
            f"{sorted(allowed)}"
        )
    coverage = (
        data[["condition_id", "split", "shot_id"]]
        .drop_duplicates()
        .groupby(["condition_id", "split"], observed=True)["shot_id"]
        .nunique()
        .unstack(fill_value=0)
    )
    if set(coverage.columns.astype(str)) != allowed or (coverage <= 0).any().any():
        raise ValueError(
            f"{dataset_label} does not represent every condition in every split"
        )


def run_loss_sweep_analysis(
    dark_data: pd.DataFrame,
    bright_data: pd.DataFrame,
    *,
    dark_count_column: str = PRIMARY_COUNT_COLUMN,
    bright_count_column: str = PRIMARY_COUNT_COLUMN,
    n_boot: int = 1000,
    random_seed: int = 20260728,
    timing_verified: bool,
    dark_geometry_validated: bool,
    bright_geometry_validated: bool,
    dark_background_frozen: bool,
    bright_background_frozen: bool,
    run_latent_gate: bool = True,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the frozen held-out analyses and, conditionally, the latent gate.

    Geometry and background evidence are supplied explicitly by the preceding
    validation scripts.  This function therefore cannot silently turn a
    missing prerequisite into a successful scientific gate.
    """
    if n_boot < 100:
        raise ValueError("n_boot must be at least 100 for public intervals")
    _validate_analysis_table(
        dark_data, dataset_label="switch-off hold", expected_frames=5
    )
    _validate_analysis_table(
        bright_data, dataset_label="bright wait", expected_frames=2
    )
    announce = progress or (lambda _message: None)

    announce("fit dark held-out emission baselines")
    dark_emission, dark_emission_context = fit_emission_analysis(
        dark_data, primary_count_col=dark_count_column
    )
    announce("fit bright held-out emission baselines")
    bright_emission, bright_emission_context = fit_emission_analysis(
        bright_data, primary_count_col=bright_count_column
    )

    announce("fit and bootstrap switch-off retention")
    dark, dark_context = fit_dark_analysis(
        dark_emission_context["scored"],
        n_boot=n_boot,
        seed=random_seed + 100,
        weak_sites=dark_emission_context["weak_sites"],
    )
    announce("fit and bootstrap bright decay and post-wait control")
    bright, bright_context = fit_bright_analysis(
        bright_emission_context["scored"],
        n_boot=n_boot,
        seed=random_seed + 200,
        weak_sites=bright_emission_context["weak_sites"],
    )

    announce("evaluate background-method sensitivities")
    dark["background_method_sensitivity"] = jsonable(
        background_model_sensitivity(
            dark_data,
            selected_emission=dark_emission_context["selected_model"],
            response="dark",
            selected_physical_model=str(dark["selected_model"]),
        )
    )
    bright["background_method_sensitivity"] = jsonable(
        background_model_sensitivity(
            bright_data,
            selected_emission=bright_emission_context["selected_model"],
            response="bright",
            selected_physical_model=str(bright["selected_model"]),
        )
    )
    dark["emission_model_sensitivity"] = jsonable(
        emission_model_sensitivity(
            dark_data,
            models=dark_emission_context["models"],
            response="dark",
            selected_physical_model=str(dark["selected_model"]),
        )
    )
    bright["emission_model_sensitivity"] = jsonable(
        emission_model_sensitivity(
            bright_data,
            models=bright_emission_context["models"],
            response="bright",
            selected_physical_model=str(bright["selected_model"]),
        )
    )
    announce("evaluate alternate whole-cycle splits")
    dark["alternate_whole_cycle_split"] = seeded_split_sensitivity(
        dark_data,
        selected_emission=dark_emission_context["selected_model"],
        response="dark",
        selected_physical_model=str(dark["selected_model"]),
        seed=random_seed + 300,
    )
    bright["alternate_whole_cycle_split"] = seeded_split_sensitivity(
        bright_data,
        selected_emission=bright_emission_context["selected_model"],
        response="bright",
        selected_physical_model=str(bright["selected_model"]),
        seed=random_seed + 400,
    )
    announce("evaluate site-free background coupling")
    dark["site_free_background_occupancy_coupling"] = (
        background_occupancy_coupling(dark_emission_context["scored"])
    )
    bright["site_free_background_occupancy_coupling"] = (
        background_occupancy_coupling(bright_emission_context["scored"])
    )
    dark["site_free_background_curve"] = site_free_background_curve(
        dark_emission_context["scored"],
        selected_count_column=str(dark_emission["selected_count_column"]),
        n_boot=n_boot,
        seed=random_seed + 610,
    )
    bright["site_free_background_curve"] = site_free_background_curve(
        bright_emission_context["scored"],
        selected_count_column=str(bright_emission["selected_count_column"]),
        n_boot=n_boot,
        seed=random_seed + 620,
    )

    prerequisites = {
        "timing_semantics_verified": bool(timing_verified),
        "dark_geometry_validated": bool(dark_geometry_validated),
        "bright_geometry_validated": bool(bright_geometry_validated),
        "dark_background_method_frozen": bool(dark_background_frozen),
        "bright_background_method_frozen": bool(bright_background_frozen),
        "shot_split_frozen": True,
        "held_out_baselines_complete": True,
    }
    dark_claim_gate = bool(
        timing_verified and dark_geometry_validated and dark_background_frozen
    )
    bright_claim_gate = bool(
        timing_verified and bright_geometry_validated and bright_background_frozen
    )
    dark["public_rate_claim_gate_passed"] = bool(
        dark_claim_gate and dark["rate_resolved"]
    )
    bright["public_rate_claim_gate_passed"] = bool(
        bright_claim_gate and bright["rate_resolved"]
    )
    if not dark_claim_gate:
        dark["claim_gate_failures"] = [
            name
            for name, passed in {
                "timing semantics": timing_verified,
                "dark-run geometry": dark_geometry_validated,
                "dark-run background method": dark_background_frozen,
            }.items()
            if not passed
        ]
    if not bright_claim_gate:
        bright["claim_gate_failures"] = [
            name
            for name, passed in {
                "timing semantics": timing_verified,
                "bright-run geometry": bright_geometry_validated,
                "bright-run background method": bright_background_frozen,
            }.items()
            if not passed
        ]

    comparison = compare_datasets(dark_context, bright_context)
    structural = comparison.get("structural_sensitivity")
    if structural and not structural["gap_sign_positive_under_both_structures"]:
        raise RuntimeError(
            "the unresolved floor structure makes the apparent "
            "observed-minus-predicted gap non-positive"
        )
    cycle_comparison = compare_cycle_bootstraps(dark_context, bright_context)
    if not cycle_comparison.get("qualitative_conclusion_consistent"):
        raise RuntimeError(
            "whole-cycle bootstrap sensitivity does not support the "
            "validation-frozen qualitative comparison"
        )
    comparison["whole_cycle_bootstrap_sensitivity"] = cycle_comparison
    comparison["public_comparison_gate_passed"] = bool(
        dark["public_rate_claim_gate_passed"]
        and bright["public_rate_claim_gate_passed"]
    )

    latent_context: dict[str, Any] | None = None
    latent_prerequisites = bool(
        run_latent_gate
        and timing_verified
        and dark_geometry_validated
        and dark_background_frozen
    )
    if not latent_prerequisites:
        latent = {
            "attempted": False,
            "gate_passed": False,
            "gate_failures": [
                name
                for name, passed in {
                    "latent analysis requested": run_latent_gate,
                    "timing semantics verified": timing_verified,
                    "dark-run geometry validated": dark_geometry_validated,
                    "dark-run background method frozen": dark_background_frozen,
                }.items()
                if not passed
            ],
            "interpretation": (
                "The latent model was not fitted because a predeclared "
                "prerequisite failed."
            ),
        }
    else:
        announce("fit and evaluate gated latent-state candidates")
        try:
            latent, latent_context = fit_latent_analysis(
                dark_emission_context["scored"],
                selected_emission=dark_emission_context["selected_model"],
                timing_verified=timing_verified,
                geometry_validated=dark_geometry_validated,
                background_frozen=dark_background_frozen,
                seed=random_seed + 500,
            )
            latent["attempted"] = True
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            latent = {
                "attempted": True,
                "gate_passed": False,
                "gate_failures": [
                    f"latent fitting or identifiability evaluation failed: "
                    f"{type(exc).__name__}: {exc}"
                ],
                "interpretation": (
                    "Baseline results remain available; no latent-state result "
                    "is accepted."
                ),
            }
    announce("assemble public result")

    report = {
        "analysis_version": "loss-sweeps-v1.0",
        "random_seed": int(random_seed),
        "bootstrap_replicates": int(n_boot),
        "independent_experimental_unit": "shot",
        "prerequisite_evidence": prerequisites,
        "acquisition_order_caveat": (
            "Conditions repeat in fixed ascending order within every cycle. "
            "Sweep value is perfectly confounded with within-cycle position, "
            "although each condition recurs across ten cycles."
        ),
        "dark_hold": {
            "emission_baselines": dark_emission,
            "operational_model": dark,
        },
        "bright_wait": {
            "emission_baselines": bright_emission,
            "effective_model": bright,
        },
        "cross_dataset_comparison": comparison,
        "latent_state": latent,
    }
    contexts = {
        "dark_emission": dark_emission_context,
        "bright_emission": bright_emission_context,
        "dark": dark_context,
        "bright": bright_context,
        "latent": latent_context,
    }
    return jsonable(report), contexts


__all__ = [
    "BACKGROUND_VALUE_COLUMNS",
    "PRIMARY_COUNT_COLUMN",
    "background_model_sensitivity",
    "background_occupancy_coupling",
    "compare_cycle_bootstraps",
    "compare_datasets",
    "dark_endpoint_sensitivity",
    "fit_bright_analysis",
    "fit_dark_analysis",
    "fit_emission_analysis",
    "fit_latent_analysis",
    "jsonable",
    "run_loss_sweep_analysis",
    "seeded_split_sensitivity",
    "shot_cluster_curve",
]
