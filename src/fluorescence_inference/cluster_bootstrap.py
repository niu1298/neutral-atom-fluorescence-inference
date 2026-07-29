"""Condition-stratified bootstrap with complete shots as clusters.

Sites and frames are repeated observations inside one experimental shot.  The
functions here resample the shot key, then copy every row belonging to the
selected shot.  They never resample site-frame rows independently.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd


def _as_tuple(columns: str | Sequence[str]) -> tuple[str, ...]:
    return (columns,) if isinstance(columns, str) else tuple(columns)


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def resample_shot_clusters(
    data: pd.DataFrame,
    *,
    rng: np.random.Generator,
    shot_cols: str | Sequence[str] = ("run_id", "shot_id"),
    condition_cols: str | Sequence[str] = "condition_id",
    cluster_col: str = "_bootstrap_cluster_id",
    source_col: str = "_bootstrap_source_shot",
) -> pd.DataFrame:
    """Draw one condition-stratified bootstrap sample of complete shots.

    Original shot columns remain unchanged for traceability.  Because a source
    shot can be selected multiple times, downstream shot-level aggregation
    should use ``cluster_col`` inside bootstrap statistics.
    """
    shot_cols = _as_tuple(shot_cols)
    condition_cols = _as_tuple(condition_cols)
    keys = [*condition_cols, *shot_cols]
    _require_columns(data, keys)
    if cluster_col in data.columns or source_col in data.columns:
        raise ValueError("bootstrap bookkeeping columns already exist")

    shot_table = data[keys].drop_duplicates().reset_index(drop=True)
    if shot_table[keys].isna().any().any():
        raise ValueError("shot and condition keys must not be missing")
    if shot_table.duplicated(list(shot_cols)).any():
        raise ValueError("each shot must belong to exactly one condition")
    # Build the row lookup once.  A 1,000-draw analysis with ~100 source shots
    # must not scan all site-frame rows separately for every selected shot.
    grouped_indices = data.groupby(
        keys, observed=True, dropna=False, sort=False
    ).indices

    pieces: list[pd.DataFrame] = []
    occurrence = 0
    grouper: str | list[str]
    grouper = condition_cols[0] if len(condition_cols) == 1 else list(condition_cols)
    for _, condition_shots in shot_table.groupby(
        grouper, observed=True, dropna=False, sort=True
    ):
        condition_shots = condition_shots.reset_index(drop=True)
        sampled = rng.integers(0, len(condition_shots), size=len(condition_shots))
        for idx in sampled:
            key_row = condition_shots.iloc[int(idx)]
            lookup = tuple(key_row[col] for col in keys)
            block = data.iloc[grouped_indices[lookup]].copy()
            if block.empty:
                raise RuntimeError("internal bootstrap key did not select any rows")
            block[cluster_col] = occurrence
            block[source_col] = "|".join(str(key_row[c]) for c in shot_cols)
            pieces.append(block)
            occurrence += 1

    if not pieces:
        raise ValueError("cannot bootstrap an empty table")
    return pd.concat(pieces, ignore_index=True)


def _coerce_statistic(
    value: float | Sequence[float] | np.ndarray | Mapping[str, float],
    parameter_names: Sequence[str] | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if isinstance(value, Mapping):
        names = tuple(map(str, value.keys()))
        vector = np.asarray(list(value.values()), dtype=float)
        if parameter_names is not None and tuple(parameter_names) != names:
            raise ValueError("mapping keys changed between bootstrap evaluations")
        return vector, names
    vector = np.atleast_1d(np.asarray(value, dtype=float))
    names = (
        tuple(parameter_names)
        if parameter_names is not None
        else tuple(f"value_{i}" for i in range(vector.size))
    )
    if len(names) != vector.size:
        raise ValueError("parameter_names length does not match statistic")
    return vector, names


@dataclass(frozen=True)
class ClusterBootstrapResult:
    """Point estimate, bootstrap draws, and percentile intervals."""

    estimate: pd.Series
    draws: pd.DataFrame
    confidence: float
    n_requested: int
    n_successful: int
    n_failed: int
    seed: int

    def intervals(self) -> pd.DataFrame:
        alpha = (1.0 - self.confidence) / 2.0
        rows = []
        for name, point in self.estimate.items():
            values = self.draws[name].to_numpy(float)
            rows.append(
                {
                    "parameter": name,
                    "estimate": float(point),
                    "lower": float(np.quantile(values, alpha)),
                    "upper": float(np.quantile(values, 1.0 - alpha)),
                    "confidence": self.confidence,
                    "n_bootstrap": self.n_successful,
                }
            )
        return pd.DataFrame(rows)

    def correlations(self) -> pd.DataFrame:
        """Bootstrap parameter correlations for identifiability diagnostics."""
        return self.draws.corr()


def condition_stratified_cluster_bootstrap(
    data: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], object],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
    shot_cols: str | Sequence[str] = ("run_id", "shot_id"),
    condition_cols: str | Sequence[str] = "condition_id",
    parameter_names: Sequence[str] | None = None,
    max_fail_fraction: float = 0.1,
) -> ClusterBootstrapResult:
    """Evaluate ``statistic`` under a complete-shot clustered bootstrap.

    Fit failures are recorded rather than converted into finite values.  More
    than ``max_fail_fraction`` failures aborts the interval because silently
    retaining only easy bootstrap samples would bias it.
    """
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie between zero and one")
    if not 0.0 <= max_fail_fraction < 1.0:
        raise ValueError("max_fail_fraction must be in [0, 1)")

    point, names = _coerce_statistic(statistic(data.copy()), parameter_names)
    if not np.isfinite(point).all():
        raise ValueError("point statistic is non-finite")

    rng = np.random.default_rng(seed)
    draws: list[np.ndarray] = []
    failed = 0
    for _ in range(int(n_boot)):
        sample = resample_shot_clusters(
            data,
            rng=rng,
            shot_cols=shot_cols,
            condition_cols=condition_cols,
        )
        try:
            value, draw_names = _coerce_statistic(statistic(sample), names)
            if draw_names != names or value.shape != point.shape:
                raise ValueError("bootstrap statistic changed shape")
            if not np.isfinite(value).all():
                raise ValueError("non-finite bootstrap statistic")
            draws.append(value)
        except (ArithmeticError, FloatingPointError, RuntimeError, ValueError):
            failed += 1

    if failed / n_boot > max_fail_fraction:
        raise RuntimeError(
            f"{failed}/{n_boot} clustered bootstrap fits failed; interval is unreliable"
        )
    if not draws:
        raise RuntimeError("all clustered bootstrap fits failed")
    draw_frame = pd.DataFrame(np.vstack(draws), columns=names)
    return ClusterBootstrapResult(
        estimate=pd.Series(point, index=names, dtype=float),
        draws=draw_frame,
        confidence=float(confidence),
        n_requested=int(n_boot),
        n_successful=len(draws),
        n_failed=failed,
        seed=int(seed),
    )
