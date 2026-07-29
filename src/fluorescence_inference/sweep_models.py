"""Likelihood models for operational switch-off and bright-wait sweeps.

The response variables are apparent occupancy calls under a stated emission
model.  Consequently, parameters here are operational/model-based:
``lambda_switch_off``, ``lambda_bright_effective`` and fixed inter-readout
retention factors.  They are not intrinsic lifetimes or labelled physical
imaging-loss probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .cluster_bootstrap import (
    ClusterBootstrapResult,
    condition_stratified_cluster_bootstrap,
    whole_cycle_cluster_bootstrap,
    resample_shot_clusters,
)


DARK_MODELS = ("shared", "first_separate", "interval_specific")
BRIGHT_MODELS = ("no_floor", "floor")
CONTROL_MODELS = ("flat", "monotone")
_EPS = 1e-8
SUCCESS_COUNT_COL = "_successes"
TRIAL_COUNT_COL = "_trials"


def _as_tuple(columns: str | Sequence[str]) -> tuple[str, ...]:
    return (columns,) if isinstance(columns, str) else tuple(columns)


def _require_columns(data: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _check_training_only(
    data: pd.DataFrame, *, split_col: str | None, training_label: str
) -> None:
    if split_col is None or split_col not in data.columns:
        return
    labels = set(data[split_col].dropna().astype(str).unique())
    if labels != {training_label}:
        raise ValueError(
            "sweep-model fitting accepts training rows only; "
            f"{split_col!r} contains {sorted(labels)}"
        )


def _binary_response(data: pd.DataFrame, response_col: str) -> np.ndarray:
    y = data[response_col].to_numpy()
    if pd.isna(y).any():
        raise ValueError(f"{response_col!r} contains missing values")
    y_float = y.astype(float)
    if not np.isin(y_float, [0.0, 1.0]).all():
        raise ValueError(f"{response_col!r} must contain only binary values")
    return y_float


def _time_values(data: pd.DataFrame, time_col: str) -> np.ndarray:
    time = data[time_col].to_numpy(float)
    if not np.isfinite(time).all() or np.any(time < 0.0):
        raise ValueError(f"{time_col!r} must contain finite non-negative seconds")
    return time


def _bernoulli_nll(response: np.ndarray, probability: np.ndarray) -> float:
    p = np.clip(np.asarray(probability, dtype=float), _EPS, 1.0 - _EPS)
    y = np.asarray(response, dtype=float)
    return float(-(y * np.log(p) + (1.0 - y) * np.log1p(-p)).sum())


def _response_counts(
    data: pd.DataFrame, response_col: str
) -> tuple[np.ndarray, np.ndarray]:
    """Binary rows or their exact binomial sufficient statistics."""
    if SUCCESS_COUNT_COL not in data.columns and TRIAL_COUNT_COL not in data.columns:
        response = _binary_response(data, response_col)
        return response, np.ones_like(response)
    _require_columns(data, [SUCCESS_COUNT_COL, TRIAL_COUNT_COL])
    successes = data[SUCCESS_COUNT_COL].to_numpy(float)
    trials = data[TRIAL_COUNT_COL].to_numpy(float)
    if (
        not np.isfinite(successes).all()
        or not np.isfinite(trials).all()
        or np.any(trials <= 0.0)
        or np.any(successes < 0.0)
        or np.any(successes > trials)
    ):
        raise ValueError("aggregated response counts are invalid")
    return successes, trials


def _binomial_nll(
    successes: np.ndarray, trials: np.ndarray, probability: np.ndarray
) -> float:
    p = np.clip(np.asarray(probability, dtype=float), _EPS, 1.0 - _EPS)
    return float(
        -(
            successes * np.log(p)
            + (trials - successes) * np.log1p(-p)
        ).sum()
    )


def _aggregate_shot_binomial(
    data: pd.DataFrame,
    *,
    response_col: str,
    shot_cols: Sequence[str],
    condition_cols: Sequence[str],
    model_cols: Sequence[str],
) -> pd.DataFrame:
    """One exact likelihood row per shot and model-design cell."""
    _require_columns(
        data, [response_col, *shot_cols, *condition_cols, *model_cols]
    )
    group_cols = list(dict.fromkeys([*condition_cols, *shot_cols, *model_cols]))
    grouped = data.groupby(
        group_cols, observed=True, dropna=False, sort=False
    )[response_col]
    out = grouped.agg(["sum", "size"]).reset_index().rename(
        columns={"sum": SUCCESS_COUNT_COL, "size": TRIAL_COUNT_COL}
    )
    out[response_col] = (out[SUCCESS_COUNT_COL] > 0).astype(int)
    return out


def _ordered_levels(values: pd.Series) -> tuple[Any, ...]:
    return tuple(sorted(values.drop_duplicates().tolist(), key=lambda x: str(x)))


def _shot_count(data: pd.DataFrame, shot_cols: tuple[str, ...]) -> int:
    return (
        len(data[list(shot_cols)].drop_duplicates())
        if set(shot_cols).issubset(data.columns)
        else 0
    )


def make_retention_events(
    scored: pd.DataFrame,
    *,
    call_col: str = "apparent_occupied",
    time_col: str = "sweep_value_s",
    frame_col: str = "frame_index",
    frame_order_col: str | None = None,
    site_col: str = "site_id",
    shot_cols: str | Sequence[str] = ("run_id", "shot_id"),
    condition_cols: str | Sequence[str] = "condition_id",
    response_col: str = "retained_next",
    interval_col: str = "interval",
    passthrough_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Create consecutive-frame retention events conditional on a prior 1 call."""
    shot_cols = _as_tuple(shot_cols)
    condition_cols = _as_tuple(condition_cols)
    order_col = frame_order_col or frame_col
    required = [
        *shot_cols,
        *condition_cols,
        site_col,
        frame_col,
        order_col,
        time_col,
        call_col,
        *passthrough_cols,
    ]
    _require_columns(scored, required)
    identity_cols = [*shot_cols, site_col]
    ordered = scored.sort_values(
        [*identity_cols, order_col],
        kind="stable",
    )
    grouped = ordered.groupby(
        identity_cols,
        observed=True,
        dropna=False,
        sort=False,
    )
    events = ordered[
        [
            *shot_cols,
            *condition_cols,
            site_col,
            time_col,
            frame_col,
            call_col,
            *passthrough_cols,
        ]
    ].copy()
    events["_next_frame"] = grouped[frame_col].shift(-1)
    events["_next_call"] = grouped[call_col].shift(-1)
    events = events.loc[
        events["_next_frame"].notna() & events[call_col].astype(bool)
    ].copy()
    if events.empty:
        return pd.DataFrame(
            columns=[
                *shot_cols,
                *condition_cols,
                site_col,
                time_col,
                interval_col,
                response_col,
                *passthrough_cols,
            ]
        )
    if pd.api.types.is_integer_dtype(ordered[frame_col].dtype):
        events["_next_frame"] = events["_next_frame"].astype(
            ordered[frame_col].dtype
        )
    events[interval_col] = (
        events[frame_col].astype(str) + "->" + events["_next_frame"].astype(str)
    )
    events[response_col] = events["_next_call"].astype(bool).astype(int)
    return events[
        [
            *shot_cols,
            *condition_cols,
            site_col,
            time_col,
            interval_col,
            response_col,
            *passthrough_cols,
        ]
    ].reset_index(drop=True)


def operational_survival(
    fixed_interreadout_survival: float | np.ndarray,
    rate: float | np.ndarray,
    hold_s: float | np.ndarray,
) -> np.ndarray:
    """Exact ``q * exp(-lambda * t)`` operational survival."""
    q = np.asarray(fixed_interreadout_survival, dtype=float)
    lam = np.asarray(rate, dtype=float)
    time = np.asarray(hold_s, dtype=float)
    if np.any(~np.isfinite(q)) or np.any((q < 0.0) | (q > 1.0)):
        raise ValueError("fixed survival q must be finite and in [0, 1]")
    if np.any(~np.isfinite(lam)) or np.any(lam < 0.0):
        raise ValueError("rate must be finite and non-negative")
    if np.any(~np.isfinite(time)) or np.any(time < 0.0):
        raise ValueError("hold time must be finite and non-negative")
    return q * np.exp(-lam * time)


def exact_multiplicative_loss(
    fixed_interreadout_survival: float | np.ndarray,
    rate: float | np.ndarray,
    hold_s: float | np.ndarray,
) -> np.ndarray:
    """Exact total loss, ``1 - q * exp(-lambda*t)`` (never an additive sum)."""
    return 1.0 - operational_survival(
        fixed_interreadout_survival, rate, hold_s
    )


def constant_rate_loss(
    rate: float | np.ndarray, duration_s: float | np.ndarray
) -> np.ndarray:
    """Loss predicted by a simple constant-rate model over a duration."""
    return exact_multiplicative_loss(1.0, rate, duration_s)


@dataclass(frozen=True)
class DarkRetentionFit:
    """Operational consecutive-readout retention fit."""

    model: str
    interval_levels: tuple[Any, ...]
    first_interval: Any
    q_by_interval: dict[Any, float]
    lambda_by_interval: dict[Any, float]
    lambda_groups: dict[str, float]
    n_observations: int
    n_independent_shots: int
    n_parameters: int
    train_nll: float
    converged: bool
    optimizer_message: str
    parameter_on_boundary: bool
    response_col: str
    time_col: str
    interval_col: str

    @property
    def aic(self) -> float:
        return 2.0 * self.n_parameters + 2.0 * self.train_nll

    @property
    def bic(self) -> float:
        return (
            self.n_parameters * np.log(max(self.n_independent_shots, 1))
            + 2.0 * self.train_nll
        )

    @property
    def lambda_switch_off(self) -> float | None:
        return self.lambda_groups.get("shared")

    @property
    def tau_switch_off(self) -> float | None:
        rate = self.lambda_switch_off
        return None if rate is None or rate <= 0.0 else 1.0 / rate

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        _require_columns(data, [self.time_col, self.interval_col])
        intervals = data[self.interval_col]
        unseen = set(intervals.dropna().unique()) - set(self.interval_levels)
        if unseen:
            raise ValueError(f"unseen retention intervals: {sorted(unseen, key=str)}")
        q = intervals.map(self.q_by_interval).to_numpy(float)
        rate = intervals.map(self.lambda_by_interval).to_numpy(float)
        return operational_survival(q, rate, _time_values(data, self.time_col))

    def nll(self, data: pd.DataFrame) -> float:
        _require_columns(data, [self.response_col])
        successes, trials = _response_counts(data, self.response_col)
        return _binomial_nll(successes, trials, self.predict(data))

    def parameter_dict(self) -> dict[str, float]:
        out = {
            f"fixed_interreadout_survival__{key}": value
            for key, value in self.q_by_interval.items()
        }
        out.update(
            {f"lambda_switch_off__{key}": value for key, value in self.lambda_groups.items()}
        )
        for key, value in self.lambda_groups.items():
            out[f"tau_switch_off__{key}"] = (
                float("inf") if value <= 0.0 else 1.0 / value
            )
        return out


def fit_dark_retention(
    train: pd.DataFrame,
    *,
    model: str = "shared",
    response_col: str = "retained_next",
    time_col: str = "sweep_value_s",
    interval_col: str = "interval",
    shot_cols: str | Sequence[str] = ("run_id", "shot_id"),
    first_interval: Any | None = None,
    split_col: str | None = "split",
    training_label: str = "train",
    max_rate: float = 20.0,
    initial_parameters: Mapping[str, float] | None = None,
    n_starts: int = 5,
) -> DarkRetentionFit:
    """Fit ``q_j exp(-lambda*t)`` by constrained Bernoulli likelihood."""
    if model not in DARK_MODELS:
        raise ValueError(f"unknown dark model {model!r}; choose from {DARK_MODELS}")
    if max_rate <= 0.0:
        raise ValueError("max_rate must be positive")
    if n_starts <= 0:
        raise ValueError("n_starts must be positive")
    shot_cols = _as_tuple(shot_cols)
    _require_columns(train, [response_col, time_col, interval_col])
    _check_training_only(train, split_col=split_col, training_label=training_label)
    successes, trials = _response_counts(train, response_col)
    time = _time_values(train, time_col)
    intervals = _ordered_levels(train[interval_col])
    if not intervals:
        raise ValueError("no retention intervals")
    if train[interval_col].isna().any():
        raise ValueError("retention interval contains missing values")
    first_interval = intervals[0] if first_interval is None else first_interval
    if first_interval not in intervals:
        raise ValueError("first_interval is absent from the training data")

    interval_index = {value: i for i, value in enumerate(intervals)}
    row_interval = train[interval_col].map(interval_index).to_numpy(int)
    if model == "shared":
        rate_group_names = ("shared",)
        interval_rate_group = np.zeros(len(intervals), dtype=int)
    elif model == "first_separate":
        rate_group_names = ("first_interval", "later_intervals")
        interval_rate_group = np.asarray(
            [0 if value == first_interval else 1 for value in intervals], dtype=int
        )
    else:
        rate_group_names = tuple(str(value) for value in intervals)
        interval_rate_group = np.arange(len(intervals), dtype=int)

    # A rate and an intercept need at least two distinct hold times per rate group.
    for group_index, group_name in enumerate(rate_group_names):
        selected_intervals = np.where(interval_rate_group == group_index)[0]
        selected_rows = np.isin(row_interval, selected_intervals)
        if np.unique(time[selected_rows]).size < 2:
            raise ValueError(
                f"rate group {group_name!r} has fewer than two distinct hold times"
            )

    n_q = len(intervals)
    n_rate = len(rate_group_names)

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return theta[:n_q], theta[n_q:]

    def objective(theta: np.ndarray) -> float:
        q, rates = unpack(theta)
        row_rate = rates[interval_rate_group[row_interval]]
        p = q[row_interval] * np.exp(-row_rate * time)
        return _binomial_nll(successes, trials, p)

    bounds = [(_EPS, 1.0 - _EPS)] * n_q + [(0.0, max_rate)] * n_rate
    initial_thetas: list[np.ndarray] = []
    if initial_parameters is not None:
        q0 = np.asarray(
            [
                initial_parameters[f"fixed_interreadout_survival__{value}"]
                for value in intervals
            ],
            dtype=float,
        )
        rate0 = np.asarray(
            [
                initial_parameters[f"lambda_switch_off__{name}"]
                for name in rate_group_names
            ],
            dtype=float,
        )
        initial_thetas.append(
            np.r_[
                np.clip(q0, _EPS, 1.0 - _EPS),
                np.clip(rate0, 0.0, max_rate),
            ]
        )
    for initial_rate in (0.01, 0.05, 0.2, 0.8, 2.0):
        rate0 = min(initial_rate, 0.8 * max_rate)
        q0 = np.empty(n_q, dtype=float)
        for i in range(n_q):
            rows = row_interval == i
            group_rate = rate0
            denom = float(np.mean(np.exp(-group_rate * time[rows])))
            observed = float(successes[rows].sum() / trials[rows].sum())
            q0[i] = np.clip(observed / max(denom, _EPS), 0.05, 0.98)
        initial_thetas.append(np.r_[q0, np.full(n_rate, rate0)])
    best = None
    for theta0 in initial_thetas[:n_starts]:
        result = minimize(
            objective,
            theta0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 3000},
        )
        if best is None or result.fun < best.fun:
            best = result
    if best is None or not np.isfinite(best.fun):
        raise RuntimeError("dark-retention optimization failed")
    q_fit, rates_fit = unpack(np.asarray(best.x, dtype=float))
    q_by = {value: float(q_fit[i]) for value, i in interval_index.items()}
    group_rates = {
        name: float(rates_fit[i]) for i, name in enumerate(rate_group_names)
    }
    lambda_by = {
        value: float(rates_fit[interval_rate_group[i]])
        for i, value in enumerate(intervals)
    }
    boundary = bool(
        np.any(q_fit < 1e-6)
        or np.any(q_fit > 1.0 - 1e-6)
        or np.any(rates_fit < 1e-7)
        or np.any(rates_fit > max_rate - 1e-6)
    )
    return DarkRetentionFit(
        model=model,
        interval_levels=intervals,
        first_interval=first_interval,
        q_by_interval=q_by,
        lambda_by_interval=lambda_by,
        lambda_groups=group_rates,
        n_observations=int(trials.sum()),
        n_independent_shots=_shot_count(train, shot_cols),
        n_parameters=n_q + n_rate,
        train_nll=float(best.fun),
        converged=bool(best.success),
        optimizer_message=str(best.message),
        parameter_on_boundary=boundary,
        response_col=response_col,
        time_col=time_col,
        interval_col=interval_col,
    )


@dataclass(frozen=True)
class BrightDecayFit:
    """Apparent first-image occupancy versus prior bright wait."""

    model: str
    pi_0: float
    lambda_bright_effective: float
    pi_floor: float
    n_observations: int
    n_independent_shots: int
    n_parameters: int
    train_nll: float
    converged: bool
    optimizer_message: str
    parameter_on_boundary: bool
    response_col: str
    time_col: str

    @property
    def tau_bright_effective(self) -> float:
        return (
            float("inf")
            if self.lambda_bright_effective <= 0.0
            else 1.0 / self.lambda_bright_effective
        )

    @property
    def aic(self) -> float:
        return 2.0 * self.n_parameters + 2.0 * self.train_nll

    @property
    def bic(self) -> float:
        return (
            self.n_parameters * np.log(max(self.n_independent_shots, 1))
            + 2.0 * self.train_nll
        )

    def predict_time(self, time_s: np.ndarray | Sequence[float]) -> np.ndarray:
        time = np.asarray(time_s, dtype=float)
        if np.any(~np.isfinite(time)) or np.any(time < 0.0):
            raise ValueError("bright-wait time must be finite and non-negative")
        return self.pi_floor + (self.pi_0 - self.pi_floor) * np.exp(
            -self.lambda_bright_effective * time
        )

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        _require_columns(data, [self.time_col])
        return self.predict_time(_time_values(data, self.time_col))

    def nll(self, data: pd.DataFrame) -> float:
        _require_columns(data, [self.response_col])
        successes, trials = _response_counts(data, self.response_col)
        return _binomial_nll(successes, trials, self.predict(data))

    def parameter_dict(self) -> dict[str, float]:
        return {
            "pi_0": self.pi_0,
            "pi_floor": self.pi_floor,
            "lambda_bright_effective": self.lambda_bright_effective,
            "tau_bright_effective": self.tau_bright_effective,
        }


def fit_bright_decay(
    train: pd.DataFrame,
    *,
    model: str = "no_floor",
    response_col: str = "apparent_occupied",
    time_col: str = "sweep_value_s",
    shot_cols: str | Sequence[str] = ("run_id", "shot_id"),
    split_col: str | None = "split",
    training_label: str = "train",
    max_rate: float = 20.0,
    initial_parameters: Mapping[str, float] | None = None,
    n_starts: int = 5,
) -> BrightDecayFit:
    """Fit no-floor or floor apparent occupancy decay by Bernoulli likelihood."""
    if model not in BRIGHT_MODELS:
        raise ValueError(f"unknown bright model {model!r}; choose from {BRIGHT_MODELS}")
    if max_rate <= 0.0:
        raise ValueError("max_rate must be positive")
    if n_starts <= 0:
        raise ValueError("n_starts must be positive")
    shot_cols = _as_tuple(shot_cols)
    _require_columns(train, [response_col, time_col])
    _check_training_only(train, split_col=split_col, training_label=training_label)
    successes, trials = _response_counts(train, response_col)
    time = _time_values(train, time_col)
    if np.unique(time).size < 2:
        raise ValueError("bright decay needs at least two distinct wait times")

    def unpack(theta: np.ndarray) -> tuple[float, float, float]:
        pi0 = float(theta[0])
        if model == "floor":
            floor = pi0 * float(theta[1])
            rate = float(theta[2])
        else:
            floor = 0.0
            rate = float(theta[1])
        return pi0, floor, rate

    def objective(theta: np.ndarray) -> float:
        pi0, floor, rate = unpack(theta)
        p = floor + (pi0 - floor) * np.exp(-rate * time)
        return _binomial_nll(successes, trials, p)

    bounds = (
        [(_EPS, 1.0 - _EPS), (0.0, 1.0), (0.0, max_rate)]
        if model == "floor"
        else [(_EPS, 1.0 - _EPS), (0.0, max_rate)]
    )
    initial_thetas: list[np.ndarray] = []
    if initial_parameters is not None:
        initial_pi0 = float(initial_parameters["pi_0"])
        initial_rate = float(initial_parameters["lambda_bright_effective"])
        if model == "floor":
            initial_floor = float(initial_parameters["pi_floor"])
            fraction = initial_floor / max(initial_pi0, _EPS)
            initial_thetas.append(
                np.asarray(
                    [
                        np.clip(initial_pi0, _EPS, 1.0 - _EPS),
                        np.clip(fraction, 0.0, 1.0),
                        np.clip(initial_rate, 0.0, max_rate),
                    ]
                )
            )
        else:
            initial_thetas.append(
                np.asarray(
                    [
                        np.clip(initial_pi0, _EPS, 1.0 - _EPS),
                        np.clip(initial_rate, 0.0, max_rate),
                    ]
                )
            )
    for initial_rate in (0.01, 0.1, 0.4, 1.0, 2.0):
        pi0 = np.clip(
            float(
                successes[time == np.min(time)].sum()
                / trials[time == np.min(time)].sum()
            )
            * np.exp(initial_rate * float(np.min(time))),
            0.1,
            0.98,
        )
        theta0 = (
            np.array([pi0, 0.1, min(initial_rate, 0.8 * max_rate)])
            if model == "floor"
            else np.array([pi0, min(initial_rate, 0.8 * max_rate)])
        )
        initial_thetas.append(theta0)
    best = None
    for theta0 in initial_thetas[:n_starts]:
        result = minimize(
            objective,
            theta0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 3000},
        )
        if best is None or result.fun < best.fun:
            best = result
    if best is None or not np.isfinite(best.fun):
        raise RuntimeError("bright-decay optimization failed")
    pi0, floor, rate = unpack(np.asarray(best.x, dtype=float))
    theta = np.asarray(best.x, dtype=float)
    boundary = bool(
        pi0 < 1e-6
        or pi0 > 1.0 - 1e-6
        or rate < 1e-7
        or rate > max_rate - 1e-6
        or (model == "floor" and (theta[1] < 1e-7 or theta[1] > 1.0 - 1e-7))
    )
    return BrightDecayFit(
        model=model,
        pi_0=pi0,
        lambda_bright_effective=rate,
        pi_floor=floor,
        n_observations=int(trials.sum()),
        n_independent_shots=_shot_count(train, shot_cols),
        n_parameters=3 if model == "floor" else 2,
        train_nll=float(best.fun),
        converged=bool(best.success),
        optimizer_message=str(best.message),
        parameter_on_boundary=boundary,
        response_col=response_col,
        time_col=time_col,
    )


@dataclass(frozen=True)
class ControlRetentionFit:
    """Image-1 to image-2 apparent retention versus prior bright wait."""

    model: str
    q_0: float
    kappa: float
    n_observations: int
    n_independent_shots: int
    n_parameters: int
    train_nll: float
    converged: bool
    optimizer_message: str
    parameter_on_boundary: bool
    response_col: str
    time_col: str

    @property
    def aic(self) -> float:
        return 2.0 * self.n_parameters + 2.0 * self.train_nll

    @property
    def bic(self) -> float:
        return (
            self.n_parameters * np.log(max(self.n_independent_shots, 1))
            + 2.0 * self.train_nll
        )

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        time = _time_values(data, self.time_col)
        return self.q_0 * np.exp(-self.kappa * time)

    def nll(self, data: pd.DataFrame) -> float:
        successes, trials = _response_counts(data, self.response_col)
        return _binomial_nll(successes, trials, self.predict(data))

    def parameter_dict(self) -> dict[str, float]:
        return {"post_wait_retention_q_0": self.q_0, "post_wait_retention_kappa": self.kappa}


def fit_control_retention(
    train: pd.DataFrame,
    *,
    model: str = "flat",
    response_col: str = "retained_next",
    time_col: str = "sweep_value_s",
    shot_cols: str | Sequence[str] = ("run_id", "shot_id"),
    split_col: str | None = "split",
    training_label: str = "train",
    max_rate: float = 20.0,
) -> ControlRetentionFit:
    """Fit flat or constrained monotone post-wait retention."""
    if model not in CONTROL_MODELS:
        raise ValueError(f"unknown control model {model!r}; choose from {CONTROL_MODELS}")
    shot_cols = _as_tuple(shot_cols)
    _require_columns(train, [response_col, time_col])
    _check_training_only(train, split_col=split_col, training_label=training_label)
    successes, trials = _response_counts(train, response_col)
    time = _time_values(train, time_col)
    if model == "monotone" and np.unique(time).size < 2:
        raise ValueError("monotone control needs at least two wait times")

    def objective(theta: np.ndarray) -> float:
        q0 = theta[0]
        kappa = 0.0 if model == "flat" else theta[1]
        return _binomial_nll(
            successes, trials, q0 * np.exp(-kappa * time)
        )

    bounds = [(_EPS, 1.0 - _EPS)] + (
        [] if model == "flat" else [(0.0, max_rate)]
    )
    theta0 = np.array([
        np.clip(float(successes.sum() / trials.sum()), 0.05, 0.98)
    ])
    if model == "monotone":
        theta0 = np.r_[theta0, 0.05]
    result = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 3000},
    )
    if not np.isfinite(result.fun):
        raise RuntimeError("control-retention optimization failed")
    q0 = float(result.x[0])
    kappa = 0.0 if model == "flat" else float(result.x[1])
    boundary = bool(
        q0 < 1e-6
        or q0 > 1.0 - 1e-6
        or (model == "monotone" and (kappa < 1e-7 or kappa > max_rate - 1e-6))
    )
    return ControlRetentionFit(
        model=model,
        q_0=q0,
        kappa=kappa,
        n_observations=int(trials.sum()),
        n_independent_shots=_shot_count(train, shot_cols),
        n_parameters=1 if model == "flat" else 2,
        train_nll=float(result.fun),
        converged=bool(result.success),
        optimizer_message=str(result.message),
        parameter_on_boundary=boundary,
        response_col=response_col,
        time_col=time_col,
    )


@dataclass
class ModelSelection:
    """Validation ranking and the corresponding train-only fitted objects."""

    selected_name: str
    fits: dict[str, Any]
    table: pd.DataFrame

    @property
    def selected_fit(self) -> Any:
        return self.fits[self.selected_name]


def _comparison_table(
    fits: Mapping[str, Any],
    validation: pd.DataFrame,
    *,
    tolerance_per_observation: float = 0.0,
) -> ModelSelection:
    rows = []
    for order, (name, fit) in enumerate(fits.items()):
        validation_nll = float(fit.nll(validation))
        rows.append(
            {
                "candidate": name,
                "complexity_order": order,
                "n_parameters": fit.n_parameters,
                "train_nll": fit.train_nll,
                "validation_nll": validation_nll,
                "validation_mean_nll": validation_nll / len(validation),
                "aic": fit.aic,
                "bic": fit.bic,
                "converged": fit.converged,
                "parameter_on_boundary": fit.parameter_on_boundary,
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["validation_mean_nll", "n_parameters", "complexity_order"], kind="stable"
    ).reset_index(drop=True)
    best_value = float(table.loc[0, "validation_mean_nll"])
    eligible = table[
        table["validation_mean_nll"] <= best_value + tolerance_per_observation
    ].sort_values(["n_parameters", "complexity_order"], kind="stable")
    selected = str(eligible.iloc[0]["candidate"])
    table["selected_on_validation"] = table["candidate"] == selected
    table["delta_validation_nll_per_observation"] = (
        table["validation_mean_nll"] - best_value
    )
    return ModelSelection(selected_name=selected, fits=dict(fits), table=table)


def _paired_shot_validation_improvement(
    simple_fit: Any,
    complex_fit: Any,
    validation: pd.DataFrame,
    *,
    shot_cols: Sequence[str],
) -> dict[str, float | int | bool]:
    """Paired validation-NLL improvement with shots as independent clusters."""
    shot_cols = tuple(shot_cols)
    _require_columns(validation, [*shot_cols, simple_fit.response_col])
    successes, trials = _response_counts(validation, simple_fit.response_col)
    simple_p = np.clip(simple_fit.predict(validation), _EPS, 1.0 - _EPS)
    complex_p = np.clip(complex_fit.predict(validation), _EPS, 1.0 - _EPS)
    simple_row = -(
        successes * np.log(simple_p)
        + (trials - successes) * np.log1p(-simple_p)
    )
    complex_row = -(
        successes * np.log(complex_p)
        + (trials - successes) * np.log1p(-complex_p)
    )
    paired = validation[list(shot_cols)].copy()
    paired["_complex_model_nll_improvement"] = simple_row - complex_row
    per_shot = (
        paired.groupby(
            list(shot_cols),
            observed=True,
            dropna=False,
            sort=False,
        )["_complex_model_nll_improvement"]
        .sum()
        .to_numpy(float)
    )
    n_shots = len(per_shot)
    mean = float(np.mean(per_shot))
    se = (
        float(np.std(per_shot, ddof=1) / np.sqrt(n_shots))
        if n_shots > 1
        else float("inf")
    )
    z = 1.959963984540054
    lower = mean - z * se
    upper = mean + z * se
    identifiable = bool(not complex_fit.parameter_on_boundary)
    return {
        "mean_nll_improvement_per_shot": mean,
        "lower_95": float(lower),
        "upper_95": float(upper),
        "n_independent_validation_shots": int(n_shots),
        "clustered_improvement_resolved": bool(lower > 0.0),
        "complex_parameter_interior": identifiable,
        "complexity_gate_passed": bool(lower > 0.0 and identifiable),
    }


def compare_dark_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    candidates: Sequence[str] = DARK_MODELS,
    tolerance_per_observation: float = 0.0,
    **fit_kwargs: Any,
) -> ModelSelection:
    """Fit dark alternatives on train and select by validation likelihood."""
    fits = {
        name: fit_dark_retention(train, model=name, **fit_kwargs)
        for name in candidates
    }
    selection = _comparison_table(
        fits,
        validation,
        tolerance_per_observation=tolerance_per_observation,
    )
    if "shared" not in fits:
        return selection
    shot_cols = _as_tuple(fit_kwargs.get("shot_cols", ("run_id", "shot_id")))
    raw_choice = selection.selected_name
    gate_by_candidate: dict[str, dict[str, float | int | bool]] = {}
    for name, fit in fits.items():
        if name == "shared":
            continue
        gate_by_candidate[name] = _paired_shot_validation_improvement(
            fits["shared"], fit, validation, shot_cols=shot_cols
        )
    for name, gate in gate_by_candidate.items():
        mask = selection.table["candidate"] == name
        for key, value in gate.items():
            selection.table.loc[mask, f"cluster_gate__{key}"] = value
    selected = (
        raw_choice
        if raw_choice != "shared"
        and bool(gate_by_candidate[raw_choice]["complexity_gate_passed"])
        else "shared"
    )
    selection.table["selected_on_validation"] = (
        selection.table["candidate"] == selected
    )
    return ModelSelection(
        selected_name=selected,
        fits=selection.fits,
        table=selection.table,
    )


def compare_bright_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    candidates: Sequence[str] = BRIGHT_MODELS,
    tolerance_per_observation: float = 0.0,
    **fit_kwargs: Any,
) -> ModelSelection:
    """Select a floor only when held-out gain resolves across shot clusters."""
    fits = {
        name: fit_bright_decay(train, model=name, **fit_kwargs)
        for name in candidates
    }
    selection = _comparison_table(
        fits,
        validation,
        tolerance_per_observation=tolerance_per_observation,
    )
    if {"no_floor", "floor"}.issubset(fits):
        raw_choice = selection.selected_name
        shot_cols = _as_tuple(
            fit_kwargs.get("shot_cols", ("run_id", "shot_id"))
        )
        gate = _paired_shot_validation_improvement(
            fits["no_floor"],
            fits["floor"],
            validation,
            shot_cols=shot_cols,
        )
        selection.table[
            "floor_mean_validation_nll_improvement_per_shot"
        ] = gate["mean_nll_improvement_per_shot"]
        selection.table["floor_validation_improvement_lower_95"] = gate[
            "lower_95"
        ]
        selection.table["floor_validation_improvement_upper_95"] = gate[
            "upper_95"
        ]
        selection.table["n_independent_validation_shots"] = gate[
            "n_independent_validation_shots"
        ]
        selection.table["floor_complexity_gate_passed"] = gate[
            "complexity_gate_passed"
        ]
        selected = (
            "floor"
            if raw_choice == "floor" and gate["complexity_gate_passed"]
            else "no_floor"
        )
        selection.table["selected_on_validation"] = (
            selection.table["candidate"] == selected
        )
        selection = ModelSelection(
            selected_name=selected,
            fits=selection.fits,
            table=selection.table,
        )
    return selection


def compare_control_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    candidates: Sequence[str] = CONTROL_MODELS,
    tolerance_per_observation: float = 0.0,
    **fit_kwargs: Any,
) -> ModelSelection:
    """Select flat versus monotone post-wait retention on validation data."""
    fits = {
        name: fit_control_retention(train, model=name, **fit_kwargs)
        for name in candidates
    }
    selection = _comparison_table(
        fits,
        validation,
        tolerance_per_observation=tolerance_per_observation,
    )
    if not {"flat", "monotone"}.issubset(fits):
        return selection
    shot_cols = _as_tuple(fit_kwargs.get("shot_cols", ("run_id", "shot_id")))
    gate = _paired_shot_validation_improvement(
        fits["flat"], fits["monotone"], validation, shot_cols=shot_cols
    )
    for key, value in gate.items():
        selection.table[f"monotone_cluster_gate__{key}"] = value
    selected = (
        "monotone"
        if selection.selected_name == "monotone"
        and bool(gate["complexity_gate_passed"])
        else "flat"
    )
    selection.table["selected_on_validation"] = (
        selection.table["candidate"] == selected
    )
    return ModelSelection(
        selected_name=selected,
        fits=selection.fits,
        table=selection.table,
    )


def bootstrap_dark_retention(
    data: pd.DataFrame,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
    shot_cols: str | Sequence[str] = ("run_id", "shot_id"),
    condition_cols: str | Sequence[str] = "condition_id",
    **fit_kwargs: Any,
) -> ClusterBootstrapResult:
    """Shot-cluster uncertainty for all dark-rate and fixed-retention parameters."""
    original_shot_cols = _as_tuple(shot_cols)
    original_condition_cols = _as_tuple(condition_cols)
    response_col = str(fit_kwargs.get("response_col", "retained_next"))
    time_col = str(fit_kwargs.get("time_col", "sweep_value_s"))
    interval_col = str(fit_kwargs.get("interval_col", "interval"))
    sufficient = _aggregate_shot_binomial(
        data,
        response_col=response_col,
        shot_cols=original_shot_cols,
        condition_cols=original_condition_cols,
        model_cols=(time_col, interval_col),
    )
    point_fit = fit_dark_retention(
        sufficient,
        shot_cols=original_shot_cols,
        **fit_kwargs,
    )
    bootstrap_fit_kwargs = {
        **fit_kwargs,
        "initial_parameters": point_fit.parameter_dict(),
        "n_starts": 1,
    }

    def statistic(sample: pd.DataFrame) -> dict[str, float]:
        active_shot_cols: str | Sequence[str] = (
            "_bootstrap_cluster_id"
            if "_bootstrap_cluster_id" in sample.columns
            else original_shot_cols
        )
        fit = fit_dark_retention(
            sample, shot_cols=active_shot_cols, **bootstrap_fit_kwargs
        )
        return fit.parameter_dict()

    return condition_stratified_cluster_bootstrap(
        sufficient,
        statistic,
        n_boot=n_boot,
        seed=seed,
        confidence=confidence,
        shot_cols=original_shot_cols,
        condition_cols=original_condition_cols,
    )


def bootstrap_bright_decay(
    data: pd.DataFrame,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
    shot_cols: str | Sequence[str] = ("run_id", "shot_id"),
    condition_cols: str | Sequence[str] = "condition_id",
    **fit_kwargs: Any,
) -> ClusterBootstrapResult:
    """Shot-cluster uncertainty for effective bright-wait decay."""
    original_shot_cols = _as_tuple(shot_cols)
    original_condition_cols = _as_tuple(condition_cols)
    response_col = str(fit_kwargs.get("response_col", "apparent_occupied"))
    time_col = str(fit_kwargs.get("time_col", "sweep_value_s"))
    sufficient = _aggregate_shot_binomial(
        data,
        response_col=response_col,
        shot_cols=original_shot_cols,
        condition_cols=original_condition_cols,
        model_cols=(time_col,),
    )
    point_fit = fit_bright_decay(
        sufficient,
        shot_cols=original_shot_cols,
        **fit_kwargs,
    )
    bootstrap_fit_kwargs = {
        **fit_kwargs,
        "initial_parameters": point_fit.parameter_dict(),
        "n_starts": 1,
    }

    def statistic(sample: pd.DataFrame) -> dict[str, float]:
        active_shot_cols: str | Sequence[str] = (
            "_bootstrap_cluster_id"
            if "_bootstrap_cluster_id" in sample.columns
            else original_shot_cols
        )
        return fit_bright_decay(
            sample, shot_cols=active_shot_cols, **bootstrap_fit_kwargs
        ).parameter_dict()

    return condition_stratified_cluster_bootstrap(
        sufficient,
        statistic,
        n_boot=n_boot,
        seed=seed,
        confidence=confidence,
        shot_cols=original_shot_cols,
        condition_cols=original_condition_cols,
    )


def bootstrap_control_retention(
    data: pd.DataFrame,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
    shot_cols: str | Sequence[str] = ("run_id", "shot_id"),
    condition_cols: str | Sequence[str] = "condition_id",
    **fit_kwargs: Any,
) -> ClusterBootstrapResult:
    """Shot-cluster uncertainty for the post-wait retention trend."""
    original_shot_cols = _as_tuple(shot_cols)
    original_condition_cols = _as_tuple(condition_cols)
    response_col = str(fit_kwargs.get("response_col", "retained_next"))
    time_col = str(fit_kwargs.get("time_col", "sweep_value_s"))
    sufficient = _aggregate_shot_binomial(
        data,
        response_col=response_col,
        shot_cols=original_shot_cols,
        condition_cols=original_condition_cols,
        model_cols=(time_col,),
    )

    def statistic(sample: pd.DataFrame) -> dict[str, float]:
        active_shot_cols: str | Sequence[str] = (
            "_bootstrap_cluster_id"
            if "_bootstrap_cluster_id" in sample.columns
            else original_shot_cols
        )
        return fit_control_retention(
            sample, shot_cols=active_shot_cols, **fit_kwargs
        ).parameter_dict()

    return condition_stratified_cluster_bootstrap(
        sufficient,
        statistic,
        n_boot=n_boot,
        seed=seed,
        confidence=confidence,
        shot_cols=original_shot_cols,
        condition_cols=original_condition_cols,
    )


def bootstrap_dark_retention_by_cycle(
    data: pd.DataFrame,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
    cycle_cols: str | Sequence[str] = ("run_id", "cycle_index"),
    shot_cols: str | Sequence[str] = ("run_id", "shot_id"),
    condition_cols: str | Sequence[str] = "condition_id",
    **fit_kwargs: Any,
) -> ClusterBootstrapResult:
    """Whole-cycle sensitivity for switch-off retention parameters."""
    cycle_cols = _as_tuple(cycle_cols)
    shot_cols = _as_tuple(shot_cols)
    condition_cols = _as_tuple(condition_cols)
    response_col = str(fit_kwargs.get("response_col", "retained_next"))
    time_col = str(fit_kwargs.get("time_col", "sweep_value_s"))
    interval_col = str(fit_kwargs.get("interval_col", "interval"))
    sufficient = _aggregate_shot_binomial(
        data,
        response_col=response_col,
        shot_cols=shot_cols,
        condition_cols=condition_cols,
        model_cols=(time_col, interval_col, *cycle_cols),
    )
    point_fit = fit_dark_retention(
        sufficient,
        shot_cols=shot_cols,
        **fit_kwargs,
    )
    bootstrap_fit_kwargs = {
        **fit_kwargs,
        "initial_parameters": point_fit.parameter_dict(),
        "n_starts": 1,
    }

    def parameters(fit: DarkRetentionFit) -> dict[str, float]:
        values = fit.parameter_dict()
        later = [
            interval
            for interval in fit.interval_levels
            if interval != fit.first_interval
        ]
        values["apparent_later_interval_fixed_loss"] = float(
            1.0 - np.mean([fit.q_by_interval[interval] for interval in later])
        )
        return values

    def statistic(sample: pd.DataFrame) -> dict[str, float]:
        active_shot_cols: str | Sequence[str] = (
            "_bootstrap_cluster_id"
            if "_bootstrap_cluster_id" in sample.columns
            else shot_cols
        )
        fit = fit_dark_retention(
            sample, shot_cols=active_shot_cols, **bootstrap_fit_kwargs
        )
        return parameters(fit)

    return whole_cycle_cluster_bootstrap(
        sufficient,
        statistic,
        n_boot=n_boot,
        seed=seed,
        confidence=confidence,
        cycle_cols=cycle_cols,
        shot_cols=shot_cols,
        condition_cols=condition_cols,
    )


def bootstrap_bright_decay_by_cycle(
    data: pd.DataFrame,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
    cycle_cols: str | Sequence[str] = ("run_id", "cycle_index"),
    shot_cols: str | Sequence[str] = ("run_id", "shot_id"),
    condition_cols: str | Sequence[str] = "condition_id",
    **fit_kwargs: Any,
) -> ClusterBootstrapResult:
    """Whole-cycle sensitivity for effective bright-wait decay."""
    cycle_cols = _as_tuple(cycle_cols)
    shot_cols = _as_tuple(shot_cols)
    condition_cols = _as_tuple(condition_cols)
    response_col = str(fit_kwargs.get("response_col", "apparent_occupied"))
    time_col = str(fit_kwargs.get("time_col", "sweep_value_s"))
    sufficient = _aggregate_shot_binomial(
        data,
        response_col=response_col,
        shot_cols=shot_cols,
        condition_cols=condition_cols,
        model_cols=(time_col, *cycle_cols),
    )
    point_fit = fit_bright_decay(
        sufficient,
        shot_cols=shot_cols,
        **fit_kwargs,
    )
    bootstrap_fit_kwargs = {
        **fit_kwargs,
        "initial_parameters": point_fit.parameter_dict(),
        "n_starts": 1,
    }

    def parameters(fit: BrightDecayFit) -> dict[str, float]:
        values = fit.parameter_dict()
        values["predicted_50ms_loss"] = float(
            constant_rate_loss(fit.lambda_bright_effective, 0.05)
        )
        return values

    def statistic(sample: pd.DataFrame) -> dict[str, float]:
        active_shot_cols: str | Sequence[str] = (
            "_bootstrap_cluster_id"
            if "_bootstrap_cluster_id" in sample.columns
            else shot_cols
        )
        fit = fit_bright_decay(
            sample, shot_cols=active_shot_cols, **bootstrap_fit_kwargs
        )
        return parameters(fit)

    return whole_cycle_cluster_bootstrap(
        sufficient,
        statistic,
        n_boot=n_boot,
        seed=seed,
        confidence=confidence,
        cycle_cols=cycle_cols,
        shot_cols=shot_cols,
        condition_cols=condition_cols,
    )


@dataclass(frozen=True)
class MultiStartBootstrapDiagnostic:
    """Comparison of anchored one-start and multi-start bootstrap refits."""

    rows: pd.DataFrame
    failures: tuple[dict[str, Any], ...]
    n_requested: int
    n_successful: int
    n_failed: int
    seed: int
    n_starts: int
    objective_tolerance: float
    relative_parameter_tolerance: float

    @property
    def materially_distinct_modes(self) -> int:
        if self.rows.empty:
            return 0
        return int(self.rows["materially_distinct_mode"].astype(bool).sum())

    @property
    def passed(self) -> bool:
        return self.n_successful > 0 and self.materially_distinct_modes == 0

    def summary(self) -> dict[str, Any]:
        return {
            "n_requested": self.n_requested,
            "n_successful": self.n_successful,
            "n_failed": self.n_failed,
            "seed": self.seed,
            "n_starts": self.n_starts,
            "objective_tolerance": self.objective_tolerance,
            "relative_parameter_tolerance": self.relative_parameter_tolerance,
            "max_objective_improvement": (
                None
                if self.rows.empty
                else float(self.rows["objective_improvement"].max())
            ),
            "max_absolute_parameter_difference": (
                None
                if self.rows.empty
                else float(self.rows["max_absolute_parameter_difference"].max())
            ),
            "max_relative_parameter_difference": (
                None
                if self.rows.empty
                else float(self.rows["max_relative_parameter_difference"].max())
            ),
            "one_start_boundary_hits": (
                0
                if self.rows.empty
                else int(self.rows["one_start_boundary"].astype(bool).sum())
            ),
            "multi_start_boundary_hits": (
                0
                if self.rows.empty
                else int(self.rows["multi_start_boundary"].astype(bool).sum())
            ),
            "materially_distinct_modes": self.materially_distinct_modes,
            "passed": self.passed,
            "failures": list(self.failures),
        }


def _multistart_bootstrap_diagnostic(
    data: pd.DataFrame,
    *,
    point_fit: Any,
    fitter: Callable[..., Any],
    n_replicates: int,
    seed: int,
    n_starts: int,
    shot_cols: tuple[str, ...],
    condition_cols: tuple[str, ...],
    objective_tolerance: float,
    relative_parameter_tolerance: float,
    max_fail_fraction: float,
) -> MultiStartBootstrapDiagnostic:
    if n_replicates < 50:
        raise ValueError("multi-start diagnostic requires at least 50 replicates")
    if n_starts < 3:
        raise ValueError("multi-start diagnostic requires at least 3 starts")
    if objective_tolerance <= 0.0 or relative_parameter_tolerance <= 0.0:
        raise ValueError("multi-start diagnostic tolerances must be positive")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    initial = point_fit.parameter_dict()
    for replicate in range(n_replicates):
        try:
            sample = resample_shot_clusters(
                data,
                rng=rng,
                shot_cols=shot_cols,
                condition_cols=condition_cols,
            )
            one = fitter(
                sample,
                shot_cols="_bootstrap_cluster_id",
                initial_parameters=initial,
                n_starts=1,
            )
            multi = fitter(
                sample,
                shot_cols="_bootstrap_cluster_id",
                initial_parameters=initial,
                n_starts=n_starts,
            )
            one_parameters = one.parameter_dict()
            multi_parameters = multi.parameter_dict()
            if one_parameters.keys() != multi_parameters.keys():
                raise ValueError("one- and multi-start parameter names differ")
            differences = {
                name: float(one_parameters[name] - multi_parameters[name])
                for name in one_parameters
            }
            absolute = {
                name: abs(value) for name, value in differences.items()
            }
            relative = {
                name: absolute[name]
                / max(abs(float(multi_parameters[name])), 1e-8)
                for name in differences
            }
            objective_improvement = float(one.train_nll - multi.train_nll)
            max_absolute = max(absolute.values(), default=0.0)
            max_relative = max(relative.values(), default=0.0)
            boundary_changed = bool(
                one.parameter_on_boundary != multi.parameter_on_boundary
            )
            distinct = bool(
                objective_improvement > objective_tolerance
                or max_relative > relative_parameter_tolerance
                or boundary_changed
            )
            rows.append(
                {
                    "replicate": int(replicate),
                    "one_start_nll": float(one.train_nll),
                    "multi_start_nll": float(multi.train_nll),
                    "objective_improvement": objective_improvement,
                    "max_absolute_parameter_difference": max_absolute,
                    "max_relative_parameter_difference": max_relative,
                    "one_start_boundary": bool(one.parameter_on_boundary),
                    "multi_start_boundary": bool(multi.parameter_on_boundary),
                    "boundary_status_changed": boundary_changed,
                    "materially_distinct_mode": distinct,
                    "parameter_differences": differences,
                }
            )
        except (ArithmeticError, FloatingPointError, RuntimeError, ValueError) as exc:
            failures.append(
                {
                    "replicate": int(replicate),
                    "exception": type(exc).__name__,
                    "message": str(exc),
                }
            )
    if len(failures) / n_replicates > max_fail_fraction:
        raise RuntimeError(
            f"{len(failures)}/{n_replicates} multi-start diagnostics failed"
        )
    return MultiStartBootstrapDiagnostic(
        rows=pd.DataFrame(rows),
        failures=tuple(failures),
        n_requested=int(n_replicates),
        n_successful=len(rows),
        n_failed=len(failures),
        seed=int(seed),
        n_starts=int(n_starts),
        objective_tolerance=float(objective_tolerance),
        relative_parameter_tolerance=float(relative_parameter_tolerance),
    )


def diagnose_dark_bootstrap_multistart(
    data: pd.DataFrame,
    *,
    n_replicates: int = 100,
    seed: int = 0,
    n_starts: int = 5,
    shot_cols: str | Sequence[str] = ("run_id", "shot_id"),
    condition_cols: str | Sequence[str] = "condition_id",
    objective_tolerance: float = 1e-6,
    relative_parameter_tolerance: float = 1e-2,
    max_fail_fraction: float = 0.1,
    **fit_kwargs: Any,
) -> MultiStartBootstrapDiagnostic:
    """Recheck a deterministic subset of dark bootstrap refits."""
    shot_cols = _as_tuple(shot_cols)
    condition_cols = _as_tuple(condition_cols)
    response_col = str(fit_kwargs.get("response_col", "retained_next"))
    time_col = str(fit_kwargs.get("time_col", "sweep_value_s"))
    interval_col = str(fit_kwargs.get("interval_col", "interval"))
    sufficient = _aggregate_shot_binomial(
        data,
        response_col=response_col,
        shot_cols=shot_cols,
        condition_cols=condition_cols,
        model_cols=(time_col, interval_col),
    )
    point_fit = fit_dark_retention(
        sufficient, shot_cols=shot_cols, **fit_kwargs
    )

    def fitter(sample: pd.DataFrame, **kwargs: Any) -> DarkRetentionFit:
        return fit_dark_retention(sample, **fit_kwargs, **kwargs)

    return _multistart_bootstrap_diagnostic(
        sufficient,
        point_fit=point_fit,
        fitter=fitter,
        n_replicates=n_replicates,
        seed=seed,
        n_starts=n_starts,
        shot_cols=shot_cols,
        condition_cols=condition_cols,
        objective_tolerance=objective_tolerance,
        relative_parameter_tolerance=relative_parameter_tolerance,
        max_fail_fraction=max_fail_fraction,
    )


def diagnose_bright_bootstrap_multistart(
    data: pd.DataFrame,
    *,
    n_replicates: int = 100,
    seed: int = 0,
    n_starts: int = 5,
    shot_cols: str | Sequence[str] = ("run_id", "shot_id"),
    condition_cols: str | Sequence[str] = "condition_id",
    objective_tolerance: float = 1e-6,
    relative_parameter_tolerance: float = 1e-2,
    max_fail_fraction: float = 0.1,
    **fit_kwargs: Any,
) -> MultiStartBootstrapDiagnostic:
    """Recheck a deterministic subset of bright bootstrap refits."""
    shot_cols = _as_tuple(shot_cols)
    condition_cols = _as_tuple(condition_cols)
    response_col = str(fit_kwargs.get("response_col", "apparent_occupied"))
    time_col = str(fit_kwargs.get("time_col", "sweep_value_s"))
    sufficient = _aggregate_shot_binomial(
        data,
        response_col=response_col,
        shot_cols=shot_cols,
        condition_cols=condition_cols,
        model_cols=(time_col,),
    )
    point_fit = fit_bright_decay(
        sufficient, shot_cols=shot_cols, **fit_kwargs
    )

    def fitter(sample: pd.DataFrame, **kwargs: Any) -> BrightDecayFit:
        return fit_bright_decay(sample, **fit_kwargs, **kwargs)

    return _multistart_bootstrap_diagnostic(
        sufficient,
        point_fit=point_fit,
        fitter=fitter,
        n_replicates=n_replicates,
        seed=seed,
        n_starts=n_starts,
        shot_cols=shot_cols,
        condition_cols=condition_cols,
        objective_tolerance=objective_tolerance,
        relative_parameter_tolerance=relative_parameter_tolerance,
        max_fail_fraction=max_fail_fraction,
    )


@dataclass(frozen=True)
class RateRatioBootstrapResult:
    """Independent-shot bootstrap for bright/switch-off effective-rate ratio."""

    point_ratio: float
    draws: pd.Series
    confidence: float
    n_requested: int
    n_failed: int
    seed: int

    def interval(self) -> dict[str, float | int]:
        alpha = (1.0 - self.confidence) / 2.0
        return {
            "lambda_bright_over_lambda_switch_off": self.point_ratio,
            "lower": float(self.draws.quantile(alpha)),
            "upper": float(self.draws.quantile(1.0 - alpha)),
            "confidence": self.confidence,
            "n_bootstrap": len(self.draws),
            "n_failed": self.n_failed,
        }


def bootstrap_rate_ratio(
    dark_data: pd.DataFrame,
    bright_data: pd.DataFrame,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
    dark_shot_cols: str | Sequence[str] = ("run_id", "shot_id"),
    dark_condition_cols: str | Sequence[str] = "condition_id",
    bright_shot_cols: str | Sequence[str] = ("run_id", "shot_id"),
    bright_condition_cols: str | Sequence[str] = "condition_id",
    dark_fit_kwargs: Mapping[str, Any] | None = None,
    bright_fit_kwargs: Mapping[str, Any] | None = None,
    max_fail_fraction: float = 0.1,
) -> RateRatioBootstrapResult:
    """Resample the two runs independently and form the effective-rate ratio.

    A ratio is returned only for a shared dark slope.  This comparison does not
    imply the sequences share all optical conditions or are poolable.
    """
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie between zero and one")
    dark_kwargs = dict(dark_fit_kwargs or {})
    bright_kwargs = dict(bright_fit_kwargs or {})
    dark_shot_cols = _as_tuple(dark_shot_cols)
    bright_shot_cols = _as_tuple(bright_shot_cols)

    dark_point = fit_dark_retention(
        dark_data, shot_cols=dark_shot_cols, **dark_kwargs
    )
    bright_point = fit_bright_decay(
        bright_data, shot_cols=bright_shot_cols, **bright_kwargs
    )
    if dark_point.lambda_switch_off is None:
        raise ValueError("rate ratio requires dark model='shared'")
    if dark_point.lambda_switch_off <= 0.0:
        raise ValueError("shared switch-off rate is on the zero boundary")
    point_ratio = (
        bright_point.lambda_bright_effective / dark_point.lambda_switch_off
    )

    rng = np.random.default_rng(seed)
    ratios: list[float] = []
    failed = 0
    for _ in range(n_boot):
        dark_sample = resample_shot_clusters(
            dark_data,
            rng=rng,
            shot_cols=dark_shot_cols,
            condition_cols=dark_condition_cols,
        )
        bright_sample = resample_shot_clusters(
            bright_data,
            rng=rng,
            shot_cols=bright_shot_cols,
            condition_cols=bright_condition_cols,
        )
        try:
            dark_fit = fit_dark_retention(
                dark_sample,
                shot_cols="_bootstrap_cluster_id",
                **dark_kwargs,
            )
            bright_fit = fit_bright_decay(
                bright_sample,
                shot_cols="_bootstrap_cluster_id",
                **bright_kwargs,
            )
            dark_rate = dark_fit.lambda_switch_off
            if dark_rate is None or dark_rate <= 0.0:
                raise ValueError("bootstrap dark rate is on the zero boundary")
            ratio = bright_fit.lambda_bright_effective / dark_rate
            if not np.isfinite(ratio):
                raise ValueError("non-finite bootstrap rate ratio")
            ratios.append(float(ratio))
        except (ArithmeticError, FloatingPointError, RuntimeError, ValueError):
            failed += 1
    if failed / n_boot > max_fail_fraction:
        raise RuntimeError(
            f"{failed}/{n_boot} rate-ratio bootstrap fits failed; ratio is unreliable"
        )
    if not ratios:
        raise RuntimeError("all rate-ratio bootstrap fits failed")
    return RateRatioBootstrapResult(
        point_ratio=float(point_ratio),
        draws=pd.Series(ratios, name="lambda_bright_over_lambda_switch_off"),
        confidence=float(confidence),
        n_requested=int(n_boot),
        n_failed=failed,
        seed=int(seed),
    )


def _sensitivity_table(
    variants: Mapping[str, pd.DataFrame],
    fitter: Callable[[pd.DataFrame], Any],
) -> pd.DataFrame:
    rows = []
    for name, table in variants.items():
        fit = fitter(table)
        rows.append(
            {
                "sensitivity_variant": name,
                "n_observations": fit.n_observations,
                "n_independent_shots": fit.n_independent_shots,
                "train_nll": fit.train_nll,
                "converged": fit.converged,
                "parameter_on_boundary": fit.parameter_on_boundary,
                **fit.parameter_dict(),
            }
        )
    return pd.DataFrame(rows)


def dark_sensitivity_table(
    variants: Mapping[str, pd.DataFrame], **fit_kwargs: Any
) -> pd.DataFrame:
    """Fit one declared dark model across predeclared sensitivity variants."""
    return _sensitivity_table(
        variants, lambda table: fit_dark_retention(table, **fit_kwargs)
    )


def bright_sensitivity_table(
    variants: Mapping[str, pd.DataFrame], **fit_kwargs: Any
) -> pd.DataFrame:
    """Fit one declared bright model across predeclared sensitivity variants."""
    return _sensitivity_table(
        variants, lambda table: fit_bright_decay(table, **fit_kwargs)
    )
