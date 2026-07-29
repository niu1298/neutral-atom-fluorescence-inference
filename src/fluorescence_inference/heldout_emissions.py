"""Train-only Gaussian emission baselines and held-out diagnostics.

These models describe fluorescence count distributions.  Their overlap and
posterior calls are model-implied quantities, not empirical readout fidelity,
false-positive rates, false-negative rates, or physical loss labels.

The API deliberately accepts a training table, rather than a full table plus a
row mask.  If a split column is present, fitting rejects any non-training row.
This makes train/validation/test leakage a testable error.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, log, sqrt
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import logsumexp


MODEL_KINDS = (
    "global",
    "frame_pooled",
    "shared_frame_offsets",
    "site_shrinkage",
    "prototype_per_site",
)


def _as_tuple(columns: str | Sequence[str]) -> tuple[str, ...]:
    return (columns,) if isinstance(columns, str) else tuple(columns)


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _normal_logpdf(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    z = (np.asarray(x, dtype=float) - mean) / sigma
    return -0.5 * z * z - log(sigma) - 0.5 * log(2.0 * np.pi)


def _normal_cdf(x: float, mean: float, sigma: float) -> float:
    return 0.5 * (1.0 + erf((x - mean) / (sigma * sqrt(2.0))))


def _normal_intersections(
    mean0: float, sigma0: float, mean1: float, sigma1: float, *, log_ratio: float = 0.0
) -> list[float]:
    """Roots of ``log f0 - log f1 + log_ratio == 0``."""
    a = 0.5 / (sigma1 * sigma1) - 0.5 / (sigma0 * sigma0)
    b = mean0 / (sigma0 * sigma0) - mean1 / (sigma1 * sigma1)
    c = (
        mean1 * mean1 / (2.0 * sigma1 * sigma1)
        - mean0 * mean0 / (2.0 * sigma0 * sigma0)
        + log(sigma1 / sigma0)
        + log_ratio
    )
    scale = max(abs(a), abs(b), abs(c), 1.0)
    if abs(a) < 1e-12 * scale:
        return [] if abs(b) < 1e-12 * scale else [float(-c / b)]
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return []
    root = sqrt(max(disc, 0.0))
    return sorted([float((-b - root) / (2.0 * a)), float((-b + root) / (2.0 * a))])


@dataclass(frozen=True)
class GaussianComponents:
    """Ordered empty/occupied Gaussian components."""

    weight_empty: float
    mean_empty: float
    sigma_empty: float
    weight_occupied: float
    mean_occupied: float
    sigma_occupied: float
    log_likelihood: float
    n_fit: int
    converged: bool

    def log_density(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=float)
        terms = np.column_stack(
            [
                log(self.weight_empty)
                + _normal_logpdf(x, self.mean_empty, self.sigma_empty),
                log(self.weight_occupied)
                + _normal_logpdf(x, self.mean_occupied, self.sigma_occupied),
            ]
        )
        return logsumexp(terms, axis=1)

    def posterior_occupied(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=float)
        low = log(self.weight_empty) + _normal_logpdf(
            x, self.mean_empty, self.sigma_empty
        )
        high = log(self.weight_occupied) + _normal_logpdf(
            x, self.mean_occupied, self.sigma_occupied
        )
        return np.exp(high - logsumexp(np.column_stack([low, high]), axis=1))

    @property
    def separation_d_prime(self) -> float:
        denom = sqrt(
            0.5 * (self.sigma_empty**2 + self.sigma_occupied**2)
        )
        return float((self.mean_occupied - self.mean_empty) / denom)

    @property
    def model_implied_overlap(self) -> float:
        """Equal-prior Bayes error implied by the two Gaussian shapes."""
        roots = _normal_intersections(
            self.mean_empty,
            self.sigma_empty,
            self.mean_occupied,
            self.sigma_occupied,
        )
        boundaries = [-np.inf, *roots, np.inf]
        overlap_coefficient = 0.0
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            if np.isneginf(lo):
                probe = hi - 10.0 * max(self.sigma_empty, self.sigma_occupied)
            elif np.isposinf(hi):
                probe = lo + 10.0 * max(self.sigma_empty, self.sigma_occupied)
            else:
                probe = 0.5 * (lo + hi)
            f0 = _normal_logpdf(np.array([probe]), self.mean_empty, self.sigma_empty)[0]
            f1 = _normal_logpdf(
                np.array([probe]), self.mean_occupied, self.sigma_occupied
            )[0]
            if f0 <= f1:
                cdf = lambda x: (  # noqa: E731
                    0.0
                    if np.isneginf(x)
                    else 1.0
                    if np.isposinf(x)
                    else _normal_cdf(x, self.mean_empty, self.sigma_empty)
                )
            else:
                cdf = lambda x: (  # noqa: E731
                    0.0
                    if np.isneginf(x)
                    else 1.0
                    if np.isposinf(x)
                    else _normal_cdf(x, self.mean_occupied, self.sigma_occupied)
                )
            overlap_coefficient += cdf(hi) - cdf(lo)
        return float(np.clip(0.5 * overlap_coefficient, 0.0, 0.5))

    @property
    def posterior_half_threshold(self) -> float:
        roots = _normal_intersections(
            self.mean_empty,
            self.sigma_empty,
            self.mean_occupied,
            self.sigma_occupied,
            log_ratio=log(self.weight_empty / self.weight_occupied),
        )
        between = [
            x for x in roots if self.mean_empty <= x <= self.mean_occupied
        ]
        return float(
            between[0] if between else 0.5 * (self.mean_empty + self.mean_occupied)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "weight_empty": self.weight_empty,
            "mean_empty": self.mean_empty,
            "sigma_empty": self.sigma_empty,
            "weight_occupied": self.weight_occupied,
            "mean_occupied": self.mean_occupied,
            "sigma_occupied": self.sigma_occupied,
            "log_likelihood": self.log_likelihood,
            "n_fit": self.n_fit,
            "converged": self.converged,
            "separation_d_prime": self.separation_d_prime,
            "model_implied_overlap": self.model_implied_overlap,
            "posterior_half_threshold": self.posterior_half_threshold,
            "_interpretation": (
                "Gaussian component overlap and posterior calls are model-implied; "
                "they are not empirical fidelity or labelled physical loss."
            ),
        }


def _fit_two_gaussian(
    values: np.ndarray,
    *,
    min_component_weight: float = 0.02,
    max_iter: int = 500,
    tolerance: float = 1e-8,
    starts: Sequence[tuple[int, int]] | None = None,
) -> GaussianComponents:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 12:
        raise ValueError(f"need at least 12 finite counts, got {x.size}")
    spread = float(np.std(x, ddof=1))
    if not np.isfinite(spread) or spread <= 0.0:
        raise ValueError("count distribution has no finite spread")
    sigma_floor = max(np.finfo(float).eps * max(abs(float(np.mean(x))), 1.0), 0.02 * spread)
    starts = (
        ((10, 90), (20, 80), (30, 70), (5, 75), (25, 95))
        if starts is None
        else tuple(starts)
    )
    if not starts:
        raise ValueError("at least one Gaussian-mixture start is required")
    best: GaussianComponents | None = None

    for q0, q1 in starts:
        means = np.asarray(np.percentile(x, [q0, q1]), dtype=float)
        sigmas = np.full(2, max(0.5 * spread, sigma_floor), dtype=float)
        weights = np.array([0.5, 0.5], dtype=float)
        previous = -np.inf
        converged = False
        for _ in range(max_iter):
            log_terms = np.column_stack(
                [
                    log(weights[k]) + _normal_logpdf(x, means[k], sigmas[k])
                    for k in range(2)
                ]
            )
            norm = logsumexp(log_terms, axis=1)
            responsibilities = np.exp(log_terms - norm[:, None])
            mass = responsibilities.sum(axis=0)
            weights = np.clip(mass / x.size, min_component_weight, 1.0)
            weights /= weights.sum()
            means = (responsibilities * x[:, None]).sum(axis=0) / mass
            variance = (
                responsibilities * (x[:, None] - means[None, :]) ** 2
            ).sum(axis=0) / mass
            sigmas = np.sqrt(np.maximum(variance, sigma_floor**2))
            ll = float(norm.sum())
            if np.isfinite(previous) and abs(ll - previous) <= tolerance * (
                1.0 + abs(previous)
            ):
                converged = True
                break
            previous = ll
        order = np.argsort(means)
        candidate = GaussianComponents(
            weight_empty=float(weights[order[0]]),
            mean_empty=float(means[order[0]]),
            sigma_empty=float(sigmas[order[0]]),
            weight_occupied=float(weights[order[1]]),
            mean_occupied=float(means[order[1]]),
            sigma_occupied=float(sigmas[order[1]]),
            log_likelihood=float(ll),
            n_fit=int(x.size),
            converged=converged,
        )
        if best is None or candidate.log_likelihood > best.log_likelihood:
            best = candidate
    if best is None or not np.isfinite(best.log_likelihood):
        raise RuntimeError("two-Gaussian fit did not produce a finite solution")
    return best


def _check_training_only(
    data: pd.DataFrame, *, split_col: str | None, training_label: str
) -> None:
    if split_col is None or split_col not in data.columns:
        return
    labels = set(data[split_col].dropna().astype(str).unique())
    if labels != {training_label}:
        raise ValueError(
            "emission fitting accepts training rows only; "
            f"{split_col!r} contains {sorted(labels)}"
        )


def _shot_keys(data: pd.DataFrame, shot_cols: tuple[str, ...]) -> tuple[tuple[Any, ...], ...]:
    if not set(shot_cols).issubset(data.columns):
        return ()
    unique = data[list(shot_cols)].drop_duplicates()
    return tuple(tuple(row) for row in unique.itertuples(index=False, name=None))


@dataclass
class EmissionModel:
    """A fitted count-emission baseline."""

    name: str
    kind: str
    value_col: str
    frame_col: str
    site_col: str
    shot_cols: tuple[str, ...]
    components: GaussianComponents
    n_train_rows: int
    training_shot_keys: tuple[tuple[Any, ...], ...]
    frame_components: dict[Any, GaussianComponents] = field(default_factory=dict)
    site_components: dict[Any, GaussianComponents] = field(default_factory=dict)
    frame_offsets: dict[Any, float] = field(default_factory=dict)
    site_offsets: dict[Any, float] = field(default_factory=dict)
    fallback_sites: tuple[Any, ...] = ()
    shrinkage_strength: float | None = None

    def _adjust(self, data: pd.DataFrame) -> np.ndarray:
        values = data[self.value_col].to_numpy(float).copy()
        if self.frame_offsets:
            missing = set(data[self.frame_col].dropna().unique()) - set(self.frame_offsets)
            if missing:
                raise ValueError(f"unseen frame groups at scoring time: {sorted(missing)}")
            values -= data[self.frame_col].map(self.frame_offsets).to_numpy(float)
        if self.site_offsets:
            values -= (
                data[self.site_col].map(self.site_offsets).fillna(0.0).to_numpy(float)
            )
        return values

    def score(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return row-level predictive density and posterior apparent occupancy."""
        _require_columns(data, [self.value_col, self.frame_col, self.site_col])
        adjusted = self._adjust(data)
        log_density = np.empty(len(data), dtype=float)
        posterior = np.empty(len(data), dtype=float)
        threshold = np.empty(len(data), dtype=float)
        d_prime = np.empty(len(data), dtype=float)
        overlap = np.empty(len(data), dtype=float)

        if self.kind == "frame_pooled":
            groups = data[self.frame_col].to_numpy()
            mapping = self.frame_components
        elif self.kind == "prototype_per_site":
            groups = data[self.site_col].to_numpy()
            mapping = self.site_components
        else:
            groups = np.zeros(len(data), dtype=int)
            mapping = {0: self.components}

        for group in pd.unique(groups):
            mask = groups == group
            comp = mapping.get(group, self.components)
            log_density[mask] = comp.log_density(adjusted[mask])
            posterior[mask] = comp.posterior_occupied(adjusted[mask])
            # thresholds are expressed on the original count scale.
            base_threshold = comp.posterior_half_threshold
            if self.frame_offsets:
                frame_adjustment = (
                    data.loc[mask, self.frame_col]
                    .map(self.frame_offsets)
                    .to_numpy(float)
                )
            else:
                frame_adjustment = 0.0
            if self.site_offsets:
                site_adjustment = (
                    data.loc[mask, self.site_col]
                    .map(self.site_offsets)
                    .fillna(0.0)
                    .to_numpy(float)
                )
            else:
                site_adjustment = 0.0
            threshold[mask] = base_threshold + frame_adjustment + site_adjustment
            d_prime[mask] = comp.separation_d_prime
            overlap[mask] = comp.model_implied_overlap

        if not np.isfinite(log_density).all():
            raise RuntimeError("non-finite held-out predictive density")
        out = data.copy()
        out["count_adjusted_for_emission"] = adjusted
        out["log_predictive_density"] = log_density
        out["posterior_occupied"] = posterior
        out["apparent_occupied"] = posterior >= 0.5
        out["posterior_half_threshold"] = threshold
        out["separation_d_prime"] = d_prime
        out["model_implied_overlap"] = overlap
        return out

    def parameter_table(self) -> pd.DataFrame:
        """Machine-readable parameters, including all fitted group offsets."""
        rows: list[dict[str, Any]] = []
        if self.kind == "frame_pooled":
            iterator = (("frame", key, comp) for key, comp in self.frame_components.items())
        elif self.kind == "prototype_per_site":
            iterator = (("site", key, comp) for key, comp in self.site_components.items())
        else:
            iterator = iter([("global", "all", self.components)])
        for group_type, key, comp in iterator:
            rows.append({"group_type": group_type, "group": key, **comp.to_dict()})
        return pd.DataFrame(rows)


def fit_emission_baseline(
    train: pd.DataFrame,
    *,
    value_col: str,
    kind: str = "global",
    name: str | None = None,
    frame_col: str = "frame_index",
    site_col: str = "site_id",
    shot_cols: str | Sequence[str] = ("run_id", "shot_id"),
    split_col: str | None = "split",
    training_label: str = "train",
    shrinkage_strength: float = 20.0,
    min_site_observations: int = 20,
    pooled_components: GaussianComponents | None = None,
    cached_frame_components: Mapping[Any, GaussianComponents] | None = None,
    cached_frame_offsets: Mapping[Any, float] | None = None,
    cached_frame_adjusted_components: GaussianComponents | None = None,
) -> EmissionModel:
    """Fit one Gaussian baseline using training rows and training rows only."""
    if kind not in MODEL_KINDS:
        raise ValueError(f"unknown emission kind {kind!r}; choose from {MODEL_KINDS}")
    shot_cols = _as_tuple(shot_cols)
    _require_columns(train, [value_col, frame_col, site_col])
    _check_training_only(train, split_col=split_col, training_label=training_label)
    finite = np.isfinite(train[value_col].to_numpy(float))
    fit_data = train.loc[finite].copy()
    if len(fit_data) < 12:
        raise ValueError("too few finite training counts")

    frame_components: dict[Any, GaussianComponents] = {}
    site_components: dict[Any, GaussianComponents] = {}
    frame_offsets: dict[Any, float] = {}
    site_offsets: dict[Any, float] = {}
    fallback_sites: list[Any] = []
    pooled = pooled_components or _fit_two_gaussian(
        fit_data[value_col].to_numpy(float)
    )

    if kind == "frame_pooled":
        for key, group in fit_data.groupby(frame_col, observed=True, sort=True):
            frame_components[key] = (
                cached_frame_components[key]
                if cached_frame_components is not None
                and key in cached_frame_components
                else _fit_two_gaussian(group[value_col].to_numpy(float))
            )
        components = pooled
    elif kind in ("shared_frame_offsets", "site_shrinkage"):
        locations: dict[Any, float] = {}
        weights: dict[Any, int] = {}
        for key, group in fit_data.groupby(frame_col, observed=True, sort=True):
            comp = (
                cached_frame_components[key]
                if cached_frame_components is not None
                and key in cached_frame_components
                else _fit_two_gaussian(group[value_col].to_numpy(float))
            )
            frame_components[key] = comp
            locations[key] = 0.5 * (comp.mean_empty + comp.mean_occupied)
            weights[key] = len(group)
        if cached_frame_offsets is None:
            reference = float(
                np.average(
                    np.fromiter(locations.values(), dtype=float),
                    weights=np.fromiter(weights.values(), dtype=float),
                )
            )
            frame_offsets = {
                key: value - reference for key, value in locations.items()
            }
        else:
            missing_frames = set(locations) - set(cached_frame_offsets)
            if missing_frames:
                raise ValueError(
                    f"cached frame offsets are missing {sorted(missing_frames)}"
                )
            frame_offsets = {
                key: float(cached_frame_offsets[key]) for key in locations
            }
        corrected = (
            fit_data[value_col].to_numpy(float)
            - fit_data[frame_col].map(frame_offsets).to_numpy(float)
        )
        components = (
            cached_frame_adjusted_components
            or _fit_two_gaussian(corrected)
        )

        if kind == "site_shrinkage":
            if shrinkage_strength < 0:
                raise ValueError("shrinkage_strength must be non-negative")
            post = components.posterior_occupied(corrected)
            expected_mean = (
                (1.0 - post) * components.mean_empty
                + post * components.mean_occupied
            )
            work = pd.DataFrame(
                {
                    "site": fit_data[site_col].to_numpy(),
                    "residual": corrected - expected_mean,
                }
            )
            for key, group in work.groupby("site", observed=True, sort=True):
                n = len(group)
                factor = n / (n + shrinkage_strength) if n else 0.0
                site_offsets[key] = float(factor * group["residual"].mean())
            corrected -= fit_data[site_col].map(site_offsets).fillna(0.0).to_numpy(float)
            components = _fit_two_gaussian(corrected)
    elif kind == "prototype_per_site":
        components = pooled
        for key, group in fit_data.groupby(site_col, observed=True, sort=True):
            if len(group) < min_site_observations:
                site_components[key] = pooled
                fallback_sites.append(key)
                continue
            try:
                site_components[key] = _fit_two_gaussian(
                    group[value_col].to_numpy(float),
                    # This all-data-per-site prototype is deliberately
                    # diagnostic and cannot be selected as primary. Three
                    # ordered starts retain local-mode protection while a
                    # looser convergence tolerance keeps 100-site sweeps
                    # reproducible in practical time.
                    max_iter=200,
                    tolerance=1e-6,
                    starts=((10, 90), (20, 80), (30, 70)),
                )
            except (RuntimeError, ValueError):
                site_components[key] = pooled
                fallback_sites.append(key)
    else:
        components = pooled

    return EmissionModel(
        name=name or f"{value_col}:{kind}",
        kind=kind,
        value_col=value_col,
        frame_col=frame_col,
        site_col=site_col,
        shot_cols=shot_cols,
        components=components,
        n_train_rows=len(fit_data),
        training_shot_keys=_shot_keys(fit_data, shot_cols),
        frame_components=frame_components,
        site_components=site_components,
        frame_offsets=frame_offsets,
        site_offsets=site_offsets,
        fallback_sites=tuple(fallback_sites),
        shrinkage_strength=shrinkage_strength if kind == "site_shrinkage" else None,
    )


def fit_required_baselines(
    train: pd.DataFrame,
    *,
    raw_value_col: str = "roi_sum_raw",
    corrected_value_col: str = "count_corrected_template",
    frame_col: str = "frame_index",
    site_col: str = "site_id",
    shot_cols: str | Sequence[str] = ("run_id", "shot_id"),
    split_col: str | None = "split",
    shrinkage_strength: float = 20.0,
) -> dict[str, EmissionModel]:
    """Fit the six required baselines on the supplied training shots."""
    common = {
        "frame_col": frame_col,
        "site_col": site_col,
        "shot_cols": shot_cols,
        "split_col": split_col,
    }
    raw = fit_emission_baseline(
        train,
        value_col=raw_value_col,
        kind="global",
        name="raw_global_threshold",
        **common,
    )
    corrected = fit_emission_baseline(
        train,
        value_col=corrected_value_col,
        kind="global",
        name="corrected_global_threshold",
        **common,
    )
    frame = fit_emission_baseline(
        train,
        value_col=corrected_value_col,
        kind="frame_pooled",
        name="frame_pooled_mixture",
        pooled_components=corrected.components,
        **common,
    )
    shared = fit_emission_baseline(
        train,
        value_col=corrected_value_col,
        kind="shared_frame_offsets",
        name="shared_frame_offsets",
        pooled_components=corrected.components,
        cached_frame_components=frame.frame_components,
        **common,
    )
    shrinkage = fit_emission_baseline(
        train,
        value_col=corrected_value_col,
        kind="site_shrinkage",
        name="shrinkage_site_offsets",
        shrinkage_strength=shrinkage_strength,
        pooled_components=corrected.components,
        cached_frame_components=frame.frame_components,
        cached_frame_offsets=shared.frame_offsets,
        cached_frame_adjusted_components=shared.components,
        **common,
    )
    prototype = fit_emission_baseline(
        train,
        value_col=corrected_value_col,
        kind="prototype_per_site",
        name="prototype_per_site_diagnostic",
        pooled_components=corrected.components,
        **common,
    )
    return {
        "raw_global_threshold": raw,
        "corrected_global_threshold": corrected,
        "frame_pooled_mixture": frame,
        "shared_frame_offsets": shared,
        "shrinkage_site_offsets": shrinkage,
        "prototype_per_site_diagnostic": prototype,
    }


def _posterior_entropy(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1e-15, 1.0 - 1e-15)
    return -(p * np.log(p) + (1.0 - p) * np.log1p(-p))


def apparent_occupancy_summary(
    scored: pd.DataFrame,
    *,
    frame_col: str = "frame_index",
    condition_cols: str | Sequence[str] | None = "condition_id",
    shot_cols: str | Sequence[str] = ("run_id", "shot_id"),
) -> pd.DataFrame:
    """Apparent occupancy and posterior summaries by condition and frame."""
    shot_cols = _as_tuple(shot_cols)
    conditions = () if condition_cols is None else _as_tuple(condition_cols)
    groups = [*conditions, frame_col]
    _require_columns(
        scored,
        [*groups, *shot_cols, "apparent_occupied", "posterior_occupied"],
    )
    rows = []
    grouper: str | list[str] = groups[0] if len(groups) == 1 else groups
    for key, block in scored.groupby(grouper, observed=True, dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, key_tuple))
        row.update(
            {
                "n_site_frames": len(block),
                "n_independent_shots": len(block[list(shot_cols)].drop_duplicates()),
                "apparent_occupancy": float(block["apparent_occupied"].mean()),
                "mean_posterior_occupancy": float(block["posterior_occupied"].mean()),
                "mean_posterior_entropy": float(
                    _posterior_entropy(block["posterior_occupied"].to_numpy(float)).mean()
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def apparent_transition_summary(
    scored: pd.DataFrame,
    *,
    frame_col: str = "frame_index",
    frame_order_col: str | None = None,
    site_col: str = "site_id",
    shot_cols: str | Sequence[str] = ("run_id", "shot_id"),
    condition_cols: str | Sequence[str] | None = "condition_id",
) -> pd.DataFrame:
    """Summarize consecutive-frame apparent calls without physical labels."""
    shot_cols = _as_tuple(shot_cols)
    conditions = () if condition_cols is None else _as_tuple(condition_cols)
    order_col = frame_order_col or frame_col
    required = [*shot_cols, *conditions, site_col, frame_col, order_col, "apparent_occupied"]
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
        [*shot_cols, *conditions, frame_col, "apparent_occupied"]
    ].copy()
    events["next_frame"] = grouped[frame_col].shift(-1)
    events["next_call"] = grouped["apparent_occupied"].shift(-1)
    events = events.loc[events["next_frame"].notna()].copy()
    if events.empty:
        return pd.DataFrame(
            columns=[
                *conditions,
                "interval",
                "n_pairs",
                "n_independent_shots",
                "apparent_agreement",
                "apparent_1_to_0",
                "apparent_0_to_1",
                "apparent_retention_given_1",
            ]
        )
    if pd.api.types.is_integer_dtype(ordered[frame_col].dtype):
        events["next_frame"] = events["next_frame"].astype(ordered[frame_col].dtype)
    events["interval"] = (
        events[frame_col].astype(str) + "->" + events["next_frame"].astype(str)
    )
    events["previous_call"] = events["apparent_occupied"].astype(bool)
    events["next_call"] = events["next_call"].astype(bool)
    rows = []
    group_cols = [*conditions, "interval"]
    grouper: str | list[str] = group_cols[0] if len(group_cols) == 1 else group_cols
    for key, block in events.groupby(grouper, observed=True, dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        previous = block["previous_call"].to_numpy(bool)
        current = block["next_call"].to_numpy(bool)
        at_risk = previous.sum()
        row = dict(zip(group_cols, key_tuple))
        row.update(
            {
                "n_pairs": len(block),
                "n_independent_shots": len(block[list(shot_cols)].drop_duplicates()),
                "apparent_agreement": float(np.mean(previous == current)),
                "apparent_1_to_0": float(np.mean(previous & ~current)),
                "apparent_0_to_1": float(np.mean(~previous & current)),
                "apparent_retention_given_1": (
                    float(current[previous].mean()) if at_risk else float("nan")
                ),
                "n_apparently_occupied_previous": int(at_risk),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class HeldoutEmissionEvaluation:
    """Held-out count score plus explicitly apparent, model-based summaries."""

    model_name: str
    n_rows: int
    n_independent_shots: int
    total_count_nll: float
    mean_count_nll: float
    mean_model_implied_overlap: float
    mean_separation_d_prime: float
    mean_posterior_entropy: float
    apparent_occupancy: pd.DataFrame
    apparent_transitions: pd.DataFrame
    scored_rows: pd.DataFrame

    def scalar_metrics(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "n_rows": self.n_rows,
            "n_independent_shots": self.n_independent_shots,
            "total_count_nll": self.total_count_nll,
            "mean_count_nll": self.mean_count_nll,
            "mean_model_implied_overlap": self.mean_model_implied_overlap,
            "mean_separation_d_prime": self.mean_separation_d_prime,
            "mean_posterior_entropy": self.mean_posterior_entropy,
            "_interpretation": (
                "Held-out predictive count metrics and apparent posterior calls; "
                "not empirical fidelity, FPR, FNR, or physical loss."
            ),
        }


def evaluate_heldout(
    model: EmissionModel,
    heldout: pd.DataFrame,
    *,
    condition_cols: str | Sequence[str] | None = "condition_id",
    frame_order_col: str | None = None,
    require_disjoint_shots: bool = True,
    include_occupancy: bool = True,
    include_transitions: bool = True,
) -> HeldoutEmissionEvaluation:
    """Score a held-out table once and reject train/test shot overlap."""
    if require_disjoint_shots and model.training_shot_keys and set(model.shot_cols).issubset(
        heldout.columns
    ):
        test_keys = set(_shot_keys(heldout, model.shot_cols))
        overlap = test_keys.intersection(model.training_shot_keys)
        if overlap:
            raise ValueError(
                f"held-out data overlap {len(overlap)} fitted training shots"
            )
    scored = model.score(heldout)
    log_density = scored["log_predictive_density"].to_numpy(float)
    entropy = _posterior_entropy(scored["posterior_occupied"].to_numpy(float))
    n_shots = (
        len(scored[list(model.shot_cols)].drop_duplicates())
        if set(model.shot_cols).issubset(scored.columns)
        else 0
    )
    occupancy = (
        apparent_occupancy_summary(
            scored,
            frame_col=model.frame_col,
            condition_cols=condition_cols,
            shot_cols=model.shot_cols,
        )
        if include_occupancy
        else pd.DataFrame()
    )
    transitions = (
        apparent_transition_summary(
            scored,
            frame_col=model.frame_col,
            frame_order_col=frame_order_col,
            site_col=model.site_col,
            shot_cols=model.shot_cols,
            condition_cols=condition_cols,
        )
        if include_transitions
        else pd.DataFrame()
    )
    return HeldoutEmissionEvaluation(
        model_name=model.name,
        n_rows=len(scored),
        n_independent_shots=n_shots,
        total_count_nll=float(-log_density.sum()),
        mean_count_nll=float(-log_density.mean()),
        mean_model_implied_overlap=float(scored["model_implied_overlap"].mean()),
        mean_separation_d_prime=float(scored["separation_d_prime"].mean()),
        mean_posterior_entropy=float(entropy.mean()),
        apparent_occupancy=occupancy,
        apparent_transitions=transitions,
        scored_rows=scored,
    )


def compare_emission_models(
    models: Mapping[str, EmissionModel],
    validation: pd.DataFrame,
    *,
    condition_cols: str | Sequence[str] | None = "condition_id",
    require_disjoint_shots: bool = True,
) -> pd.DataFrame:
    """Rank pre-fitted candidates by validation predictive count likelihood."""
    rows = []
    for name, model in models.items():
        result = evaluate_heldout(
            model,
            validation,
            condition_cols=condition_cols,
            require_disjoint_shots=require_disjoint_shots,
        )
        row = result.scalar_metrics()
        row["candidate"] = name
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(
        ["mean_count_nll", "candidate"], kind="stable"
    ).reset_index(drop=True)
    out["selected_on_validation"] = False
    if not out.empty:
        out.loc[0, "selected_on_validation"] = True
        out["delta_validation_nll_per_row"] = (
            out["mean_count_nll"] - out.loc[0, "mean_count_nll"]
        )
    return out
