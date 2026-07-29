"""Gated binary latent-state models for fluorescence trajectories.

This module deliberately separates three questions:

* whether a binary hidden-state model predicts held-out counts;
* whether its transition parameters are statistically identifiable; and
* whether the experimental controls permit a physical interpretation.

The first does not imply the other two.  In particular, the model has no
empirical occupancy labels and therefore cannot establish readout fidelity.
The :func:`evaluate_latent_state_gate` function rejects a real-data result
unless all prerequisite and validation evidence is supplied explicitly.

The default transition convention is ``0 -> 0`` with probability one and

``P(z[j + 1] = 1 | z[j] = 1) = q[j] * exp(-lambda * hold_time)``.

Reloading can be enabled for a sequence that physically permits it, but it is
off by default.  Emissions are frame-specific Gaussians with an enforced
dark-mean < bright-mean ordering.  Fits use deterministic multi-start
Baum-Welch iterations; all data-dependent fitting must still be performed by
the caller on training shots only.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp
from scipy.stats import chi2

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_LOG_2PI = float(np.log(2.0 * np.pi))
_TINY = np.finfo(float).tiny


@dataclass(frozen=True)
class LatentModelSpec:
    """Structure and numerical bounds for a binary latent-state model."""

    n_frames: int
    shared_switch_off_rate: bool = True
    bright_initial: bool = False
    allow_reloading: bool = False
    probability_bounds: tuple[float, float] = (1e-4, 1.0 - 1e-4)
    rate_bounds_per_s: tuple[float, float] = (1e-6, 100.0)
    sigma_floor_fraction: float = 1e-3

    def __post_init__(self) -> None:
        if self.n_frames < 2:
            raise ValueError("a transition model needs at least two frames")
        plo, phi = self.probability_bounds
        if not 0.0 < plo < phi < 1.0:
            raise ValueError("probability_bounds must lie strictly inside (0, 1)")
        rlo, rhi = self.rate_bounds_per_s
        if not 0.0 < rlo < rhi:
            raise ValueError("rate_bounds_per_s must be positive and ordered")
        if self.sigma_floor_fraction <= 0.0:
            raise ValueError("sigma_floor_fraction must be positive")

    @property
    def n_intervals(self) -> int:
        return self.n_frames - 1

    @property
    def n_switch_off_rates(self) -> int:
        return 1 if self.shared_switch_off_rate else self.n_intervals


@dataclass(frozen=True)
class LatentStateParameters:
    """Parameters in physical count and time units."""

    initial_occupancy: float
    fixed_interreadout_survival: FloatArray
    switch_off_rate_per_s: FloatArray
    dark_mean: FloatArray
    bright_mean: FloatArray
    dark_sigma: FloatArray
    bright_sigma: FloatArray
    bright_initial_rate_per_s: float | None = None
    reload_probability: FloatArray | None = None

    def survival_probability(self, transition_times_s: ArrayLike) -> FloatArray:
        """Return occupied-to-occupied transition probabilities."""
        t = np.asarray(transition_times_s, dtype=float)
        q = np.asarray(self.fixed_interreadout_survival, dtype=float)
        rates = np.asarray(self.switch_off_rate_per_s, dtype=float)
        if rates.size == 1:
            rate = rates[0]
        else:
            if t.shape[-1] != rates.size:
                raise ValueError("last time dimension must match interval rates")
            rate = rates
        return q * np.exp(-rate * t)

    def initial_probability(self, bright_wait_times_s: ArrayLike | None) -> FloatArray:
        """Return the occupied probability before the first emission."""
        if self.bright_initial_rate_per_s is None:
            if bright_wait_times_s is None:
                return np.asarray(self.initial_occupancy, dtype=float)
            return np.full(
                np.asarray(bright_wait_times_s, dtype=float).shape,
                self.initial_occupancy,
                dtype=float,
            )
        if bright_wait_times_s is None:
            raise ValueError("bright_wait_times_s is required by this parameter set")
        t = np.asarray(bright_wait_times_s, dtype=float)
        return self.initial_occupancy * np.exp(
            -self.bright_initial_rate_per_s * t
        )


@dataclass(frozen=True)
class LikelihoodResult:
    """Forward-algorithm likelihood, including shot/trajectory contributions."""

    total_log_likelihood: float
    trajectory_log_likelihood: FloatArray
    n_finite_observations: int


@dataclass(frozen=True)
class PosteriorResult:
    """Smoothed state and transition probabilities from forward-backward."""

    total_log_likelihood: float
    trajectory_log_likelihood: FloatArray
    state_probability: FloatArray
    transition_probability: FloatArray


@dataclass(frozen=True)
class FitStartSummary:
    """Compact audit record for one deterministic initialization."""

    start_index: int
    converged: bool
    iterations: int
    log_likelihood: float
    switch_off_rate_per_s: tuple[float, ...]
    fixed_interreadout_survival: tuple[float, ...]
    initial_occupancy: float
    message: str


@dataclass(frozen=True)
class LatentFitDiagnostics:
    """Diagnostics that are necessary but not sufficient for interpretation."""

    boundary_parameters: tuple[str, ...]
    frame_separation_d_prime: FloatArray
    design_has_hold_time_variation: bool
    design_has_bright_wait_variation: bool | None
    near_best_start_count: int
    near_best_log_likelihood_tolerance: float
    start_agreement_max_distance: float | None
    fixed_nuisance_rate_loglik_drop_half: float | None
    fixed_nuisance_rate_loglik_drop_double: float | None
    locally_rate_informative: bool


@dataclass(frozen=True)
class LatentStateFit:
    """Result of a training-data fit.

    ``log_likelihood`` is an in-sample likelihood.  Use
    :func:`score_against_held_out_baseline` on untouched test trajectories for
    model comparison.
    """

    parameters: LatentStateParameters
    spec: LatentModelSpec
    log_likelihood: float
    n_trajectories: int
    n_finite_observations: int
    converged: bool
    selected_start: int
    starts: tuple[FitStartSummary, ...]
    diagnostics: LatentFitDiagnostics
    sigma_floor: float


@dataclass(frozen=True)
class SyntheticLatentData:
    """A reproducible simulated data set with its hidden states retained."""

    observations: FloatArray
    states: NDArray[np.int8]
    transition_times_s: FloatArray
    bright_wait_times_s: FloatArray | None


@dataclass(frozen=True)
class HeldOutLikelihoodComparison:
    """Predictive comparison on one untouched set of complete trajectories."""

    latent_total_log_likelihood: float
    baseline_total_log_likelihood: float
    delta_total_log_likelihood: float
    delta_per_trajectory: float
    delta_per_finite_observation: float
    n_trajectories: int
    n_finite_observations: int

    @property
    def latent_predicts_better(self) -> bool:
        return self.delta_total_log_likelihood > 0.0


@dataclass(frozen=True)
class ProfileLikelihood:
    """One-dimensional profile likelihood for a switch-off rate."""

    rate_grid_per_s: FloatArray
    log_likelihood: FloatArray
    converged: BoolArray
    confidence_level: float
    lower_confidence_limit_per_s: float | None
    upper_confidence_limit_per_s: float | None
    maximum_rate_per_s: float
    maximum_log_likelihood: float
    interval_index: int

    @property
    def has_bounded_interval(self) -> bool:
        return (
            self.lower_confidence_limit_per_s is not None
            and self.upper_confidence_limit_per_s is not None
        )

    @property
    def confidence_ratio(self) -> float | None:
        if not self.has_bounded_interval:
            return None
        assert self.lower_confidence_limit_per_s is not None
        assert self.upper_confidence_limit_per_s is not None
        return (
            self.upper_confidence_limit_per_s
            / self.lower_confidence_limit_per_s
        )


@dataclass(frozen=True)
class LatentGateEvidence:
    """Evidence required before accepting a latent model on real data.

    Defaults are deliberately failing.  Missing evidence is not interpreted as
    a pass.
    """

    timing_semantics_verified: bool = False
    geometry_validated: bool = False
    background_method_frozen: bool = False
    shot_split_frozen: bool = False
    held_out_baselines_complete: bool = False
    synthetic_recovery_passed: bool | None = None
    posterior_predictive_checks_passed: bool | None = None
    shot_cluster_uncertainty_passed: bool | None = None
    held_out_comparison: HeldOutLikelihoodComparison | None = None
    rate_profile: ProfileLikelihood | None = None


@dataclass(frozen=True)
class LatentGateCriteria:
    """Predeclared numerical thresholds for the real-data gate."""

    min_delta_loglik_per_trajectory: float = 0.0
    min_frame_separation_d_prime: float = 1.0
    min_near_best_starts: int = 2
    max_start_agreement_distance: float = 0.25
    max_profile_confidence_ratio: float = 20.0
    reject_boundary_parameters: bool = True


@dataclass(frozen=True)
class LatentGateDecision:
    """Machine-readable gate outcome with every failed condition retained."""

    accepted: bool
    failures: tuple[str, ...]
    cautions: tuple[str, ...]


@dataclass(frozen=True)
class _EMResult:
    parameters: LatentStateParameters
    log_likelihood: float
    converged: bool
    iterations: int
    message: str


def _as_observations(
    observations: ArrayLike, spec: LatentModelSpec
) -> FloatArray:
    x = np.asarray(observations, dtype=float)
    if x.ndim != 2 or x.shape[1] != spec.n_frames:
        raise ValueError(
            f"observations must have shape (n, {spec.n_frames}), got {x.shape}"
        )
    if x.shape[0] < 2:
        raise ValueError("at least two complete trajectories are required")
    if np.any(np.isinf(x)):
        raise ValueError("observations may contain NaN but not infinity")
    if np.any(np.sum(np.isfinite(x), axis=1) == 0):
        raise ValueError("every trajectory must contain at least one observation")
    for frame in range(spec.n_frames):
        if np.count_nonzero(np.isfinite(x[:, frame])) < 2:
            raise ValueError(f"frame {frame} has fewer than two finite counts")
    return x


def _as_transition_times(
    transition_times_s: ArrayLike, n: int, spec: LatentModelSpec
) -> FloatArray:
    t = np.asarray(transition_times_s, dtype=float)
    if t.ndim == 0:
        out = np.full((n, spec.n_intervals), float(t), dtype=float)
    elif t.ndim == 1:
        if t.size != n:
            raise ValueError(
                "one-dimensional transition_times_s must give one common "
                "hold time per trajectory"
            )
        out = np.repeat(t[:, None], spec.n_intervals, axis=1)
    elif t.shape == (n, spec.n_intervals):
        out = t.copy()
    else:
        raise ValueError(
            "transition_times_s must be scalar, shape (n,), or shape "
            f"(n, {spec.n_intervals}); got {t.shape}"
        )
    if not np.all(np.isfinite(out)) or np.any(out < 0.0):
        raise ValueError("transition times must be finite and nonnegative")
    return out


def _as_bright_wait_times(
    bright_wait_times_s: ArrayLike | None, n: int, spec: LatentModelSpec
) -> FloatArray | None:
    if bright_wait_times_s is None:
        if spec.bright_initial:
            raise ValueError("bright_wait_times_s is required for bright_initial")
        return None
    t = np.asarray(bright_wait_times_s, dtype=float)
    if t.shape != (n,):
        raise ValueError(f"bright_wait_times_s must have shape ({n},)")
    if not np.all(np.isfinite(t)) or np.any(t < 0.0):
        raise ValueError("bright wait times must be finite and nonnegative")
    return t


def _validate_parameters(
    parameters: LatentStateParameters, spec: LatentModelSpec
) -> None:
    p = parameters
    arrays_and_lengths = (
        ("fixed_interreadout_survival", p.fixed_interreadout_survival,
         spec.n_intervals),
        ("switch_off_rate_per_s", p.switch_off_rate_per_s,
         spec.n_switch_off_rates),
        ("dark_mean", p.dark_mean, spec.n_frames),
        ("bright_mean", p.bright_mean, spec.n_frames),
        ("dark_sigma", p.dark_sigma, spec.n_frames),
        ("bright_sigma", p.bright_sigma, spec.n_frames),
    )
    for name, values, length in arrays_and_lengths:
        a = np.asarray(values, dtype=float)
        if a.shape != (length,) or not np.all(np.isfinite(a)):
            raise ValueError(f"{name} must be a finite vector of length {length}")
    if not 0.0 < p.initial_occupancy < 1.0:
        raise ValueError("initial_occupancy must lie strictly inside (0, 1)")
    q = np.asarray(p.fixed_interreadout_survival)
    if np.any((q <= 0.0) | (q >= 1.0)):
        raise ValueError("fixed survival factors must lie strictly inside (0, 1)")
    if np.any(np.asarray(p.switch_off_rate_per_s) <= 0.0):
        raise ValueError("switch-off rates must be positive")
    if np.any(np.asarray(p.bright_mean) <= np.asarray(p.dark_mean)):
        raise ValueError("bright means must exceed dark means in every frame")
    if np.any(np.asarray(p.dark_sigma) <= 0.0) or np.any(
        np.asarray(p.bright_sigma) <= 0.0
    ):
        raise ValueError("emission standard deviations must be positive")
    if spec.bright_initial != (p.bright_initial_rate_per_s is not None):
        raise ValueError("bright_initial spec and parameterization disagree")
    if p.bright_initial_rate_per_s is not None:
        if not np.isfinite(p.bright_initial_rate_per_s):
            raise ValueError("bright initial rate must be finite")
        if p.bright_initial_rate_per_s <= 0.0:
            raise ValueError("bright initial rate must be positive")
    if spec.allow_reloading:
        if p.reload_probability is None:
            raise ValueError("reload probabilities are required when enabled")
        r = np.asarray(p.reload_probability, dtype=float)
        if r.shape != (spec.n_intervals,) or np.any((r <= 0.0) | (r >= 1.0)):
            raise ValueError("reload probabilities must be inside (0, 1)")
    elif p.reload_probability is not None:
        if np.any(np.asarray(p.reload_probability, dtype=float) != 0.0):
            raise ValueError("reload probabilities were supplied but disabled")


def _emission_log_probability(
    observations: FloatArray, parameters: LatentStateParameters
) -> FloatArray:
    n, n_frames = observations.shape
    out = np.empty((n, n_frames, 2), dtype=float)
    means = np.stack((parameters.dark_mean, parameters.bright_mean), axis=1)
    sigmas = np.stack((parameters.dark_sigma, parameters.bright_sigma), axis=1)
    for state in (0, 1):
        z = (observations - means[None, :, state]) / sigmas[None, :, state]
        out[:, :, state] = (
            -0.5 * z * z
            - np.log(sigmas[None, :, state])
            - 0.5 * _LOG_2PI
        )
    out[~np.isfinite(observations), :] = 0.0
    return out


def _transition_log_probability(
    parameters: LatentStateParameters,
    transition_times_s: FloatArray,
    spec: LatentModelSpec,
) -> FloatArray:
    n = transition_times_s.shape[0]
    rates = np.asarray(parameters.switch_off_rate_per_s, dtype=float)
    if rates.size == 1:
        rate_by_interval = np.repeat(rates, spec.n_intervals)
    else:
        rate_by_interval = rates
    survival = (
        parameters.fixed_interreadout_survival[None, :]
        * np.exp(-transition_times_s * rate_by_interval[None, :])
    )
    survival = np.clip(survival, _TINY, 1.0 - np.finfo(float).eps)

    log_a = np.full((n, spec.n_intervals, 2, 2), -np.inf, dtype=float)
    log_a[:, :, 1, 1] = np.log(survival)
    log_a[:, :, 1, 0] = np.log1p(-survival)
    if spec.allow_reloading:
        assert parameters.reload_probability is not None
        reload = np.broadcast_to(
            parameters.reload_probability[None, :],
            (n, spec.n_intervals),
        )
        log_a[:, :, 0, 1] = np.log(reload)
        log_a[:, :, 0, 0] = np.log1p(-reload)
    else:
        log_a[:, :, 0, 0] = 0.0
    return log_a


def _prior_log_probability(
    parameters: LatentStateParameters,
    bright_wait_times_s: FloatArray | None,
    n: int,
) -> FloatArray:
    p1 = parameters.initial_probability(bright_wait_times_s)
    if np.ndim(p1) == 0:
        p1 = np.full(n, float(p1), dtype=float)
    p1 = np.clip(np.asarray(p1, dtype=float), _TINY, 1.0 - np.finfo(float).eps)
    return np.stack((np.log1p(-p1), np.log(p1)), axis=1)


def _forward_components(
    observations: FloatArray,
    transition_times_s: FloatArray,
    parameters: LatentStateParameters,
    spec: LatentModelSpec,
    bright_wait_times_s: FloatArray | None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    log_b = _emission_log_probability(observations, parameters)
    log_a = _transition_log_probability(parameters, transition_times_s, spec)
    log_prior = _prior_log_probability(
        parameters, bright_wait_times_s, observations.shape[0]
    )
    alpha = np.empty_like(log_b)
    alpha[:, 0, :] = log_prior + log_b[:, 0, :]
    for frame in range(1, spec.n_frames):
        propagated = (
            alpha[:, frame - 1, :, None]
            + log_a[:, frame - 1, :, :]
        )
        alpha[:, frame, :] = (
            log_b[:, frame, :] + logsumexp(propagated, axis=1)
        )
    trajectory_ll = logsumexp(alpha[:, -1, :], axis=1)
    return alpha, trajectory_ll, log_a, log_b


def forward_log_likelihood(
    observations: ArrayLike,
    transition_times_s: ArrayLike,
    parameters: LatentStateParameters,
    spec: LatentModelSpec,
    *,
    bright_wait_times_s: ArrayLike | None = None,
) -> LikelihoodResult:
    """Evaluate the binary HMM using the forward algorithm."""
    x = _as_observations(observations, spec)
    t = _as_transition_times(transition_times_s, x.shape[0], spec)
    bright_t = _as_bright_wait_times(bright_wait_times_s, x.shape[0], spec)
    _validate_parameters(parameters, spec)
    _, trajectory_ll, _, _ = _forward_components(
        x, t, parameters, spec, bright_t
    )
    return LikelihoodResult(
        total_log_likelihood=float(np.sum(trajectory_ll)),
        trajectory_log_likelihood=trajectory_ll,
        n_finite_observations=int(np.count_nonzero(np.isfinite(x))),
    )


def posterior_state_probabilities(
    observations: ArrayLike,
    transition_times_s: ArrayLike,
    parameters: LatentStateParameters,
    spec: LatentModelSpec,
    *,
    bright_wait_times_s: ArrayLike | None = None,
) -> PosteriorResult:
    """Run forward-backward and return smoothed states and transitions."""
    x = _as_observations(observations, spec)
    t = _as_transition_times(transition_times_s, x.shape[0], spec)
    bright_t = _as_bright_wait_times(bright_wait_times_s, x.shape[0], spec)
    _validate_parameters(parameters, spec)
    alpha, trajectory_ll, log_a, log_b = _forward_components(
        x, t, parameters, spec, bright_t
    )

    beta = np.zeros_like(alpha)
    for frame in range(spec.n_frames - 2, -1, -1):
        terms = (
            log_a[:, frame, :, :]
            + log_b[:, frame + 1, None, :]
            + beta[:, frame + 1, None, :]
        )
        beta[:, frame, :] = logsumexp(terms, axis=2)

    log_gamma = alpha + beta - trajectory_ll[:, None, None]
    gamma = np.exp(log_gamma)
    gamma /= np.sum(gamma, axis=2, keepdims=True)

    xi = np.empty(
        (x.shape[0], spec.n_intervals, 2, 2), dtype=float
    )
    for interval in range(spec.n_intervals):
        log_xi = (
            alpha[:, interval, :, None]
            + log_a[:, interval, :, :]
            + log_b[:, interval + 1, None, :]
            + beta[:, interval + 1, None, :]
            - trajectory_ll[:, None, None]
        )
        xi[:, interval, :, :] = np.exp(log_xi)
        xi_sum = np.sum(xi[:, interval, :, :], axis=(1, 2), keepdims=True)
        xi[:, interval, :, :] /= xi_sum

    return PosteriorResult(
        total_log_likelihood=float(np.sum(trajectory_ll)),
        trajectory_log_likelihood=trajectory_ll,
        state_probability=gamma,
        transition_probability=xi,
    )


def simulate_latent_state(
    parameters: LatentStateParameters,
    spec: LatentModelSpec,
    transition_times_s: ArrayLike,
    *,
    n_trajectories: int | None = None,
    bright_wait_times_s: ArrayLike | None = None,
    random_seed: int = 0,
) -> SyntheticLatentData:
    """Simulate counts and hidden states exactly under the stated model."""
    raw_t = np.asarray(transition_times_s, dtype=float)
    if n_trajectories is None:
        if raw_t.ndim == 0:
            raise ValueError("n_trajectories is required for scalar times")
        n_trajectories = int(raw_t.shape[0])
    if n_trajectories < 2:
        raise ValueError("n_trajectories must be at least two")
    t = _as_transition_times(raw_t, n_trajectories, spec)
    bright_t = _as_bright_wait_times(
        bright_wait_times_s, n_trajectories, spec
    )
    _validate_parameters(parameters, spec)

    rng = np.random.default_rng(random_seed)
    states = np.zeros((n_trajectories, spec.n_frames), dtype=np.int8)
    p_initial = parameters.initial_probability(bright_t)
    if np.ndim(p_initial) == 0:
        p_initial = np.full(n_trajectories, float(p_initial))
    states[:, 0] = rng.binomial(1, p_initial).astype(np.int8)

    survival = parameters.survival_probability(t)
    if spec.allow_reloading:
        assert parameters.reload_probability is not None
        reload = np.broadcast_to(
            parameters.reload_probability[None, :], t.shape
        )
    else:
        reload = np.zeros_like(t)
    for interval in range(spec.n_intervals):
        occupied = states[:, interval] == 1
        p_next = np.where(occupied, survival[:, interval], reload[:, interval])
        states[:, interval + 1] = rng.binomial(1, p_next).astype(np.int8)

    observations = np.empty((n_trajectories, spec.n_frames), dtype=float)
    for frame in range(spec.n_frames):
        occupied = states[:, frame] == 1
        means = np.where(
            occupied, parameters.bright_mean[frame], parameters.dark_mean[frame]
        )
        sigmas = np.where(
            occupied, parameters.bright_sigma[frame], parameters.dark_sigma[frame]
        )
        observations[:, frame] = rng.normal(means, sigmas)
    return SyntheticLatentData(observations, states, t, bright_t)


def _initial_parameters(
    observations: FloatArray,
    transition_times_s: FloatArray,
    spec: LatentModelSpec,
) -> tuple[LatentStateParameters, float]:
    dark_mean = np.empty(spec.n_frames)
    bright_mean = np.empty(spec.n_frames)
    dark_sigma = np.empty(spec.n_frames)
    bright_sigma = np.empty(spec.n_frames)
    finite_values = observations[np.isfinite(observations)]
    scale = max(float(np.std(finite_values)), 1.0)
    sigma_floor = max(spec.sigma_floor_fraction * scale, np.finfo(float).eps)
    for frame in range(spec.n_frames):
        values = observations[np.isfinite(observations[:, frame]), frame]
        q25, q50, q75 = np.percentile(values, [25.0, 50.0, 75.0])
        spread = max(float(q75 - q25), 0.2 * scale, sigma_floor)
        dark = values[values <= q50]
        bright = values[values > q50]
        dark_mean[frame] = q25
        bright_mean[frame] = max(q75, q25 + 0.1 * spread)
        dark_sigma[frame] = max(
            float(np.std(dark)) if dark.size > 1 else 0.5 * spread,
            sigma_floor,
        )
        bright_sigma[frame] = max(
            float(np.std(bright)) if bright.size > 1 else 0.5 * spread,
            sigma_floor,
        )
    midpoint = 0.5 * (dark_mean[0] + bright_mean[0])
    initial = float(np.mean(observations[:, 0] > midpoint))
    plo, phi = spec.probability_bounds
    initial = float(np.clip(initial, plo * 2.0, phi - plo))

    positive_times = transition_times_s[transition_times_s > 0.0]
    time_scale = float(np.median(positive_times)) if positive_times.size else 1.0
    base_rate = float(np.clip(
        -np.log(0.85) / time_scale,
        spec.rate_bounds_per_s[0] * 10.0,
        spec.rate_bounds_per_s[1] / 10.0,
    ))
    rates = np.full(spec.n_switch_off_rates, base_rate)
    fixed_survival = np.full(spec.n_intervals, 0.9)
    reload = (
        np.full(spec.n_intervals, 0.01)
        if spec.allow_reloading
        else None
    )
    return LatentStateParameters(
        initial_occupancy=initial,
        fixed_interreadout_survival=fixed_survival,
        switch_off_rate_per_s=rates,
        dark_mean=dark_mean,
        bright_mean=bright_mean,
        dark_sigma=dark_sigma,
        bright_sigma=bright_sigma,
        bright_initial_rate_per_s=base_rate if spec.bright_initial else None,
        reload_probability=reload,
    ), sigma_floor


def _jitter_parameters(
    base: LatentStateParameters,
    spec: LatentModelSpec,
    rng: np.random.Generator,
    data_scale: float,
) -> LatentStateParameters:
    plo, phi = spec.probability_bounds
    dark = base.dark_mean + rng.normal(
        0.0, 0.08 * data_scale, spec.n_frames
    )
    separation = (base.bright_mean - base.dark_mean) * np.exp(
        rng.normal(0.0, 0.2, spec.n_frames)
    )
    bright = dark + np.maximum(separation, 0.02 * data_scale)
    q = expit(
        logit(np.clip(base.fixed_interreadout_survival, plo, phi))
        + rng.normal(0.0, 0.6, spec.n_intervals)
    )
    q = np.clip(q, plo, phi)
    rates = np.clip(
        base.switch_off_rate_per_s
        * np.exp(rng.normal(0.0, 1.0, spec.n_switch_off_rates)),
        spec.rate_bounds_per_s[0],
        spec.rate_bounds_per_s[1],
    )
    pi = float(expit(
        logit(np.clip(base.initial_occupancy, plo, phi))
        + rng.normal(0.0, 0.5)
    ))
    bright_rate = base.bright_initial_rate_per_s
    if bright_rate is not None:
        bright_rate = float(np.clip(
            bright_rate * np.exp(rng.normal(0.0, 1.0)),
            spec.rate_bounds_per_s[0],
            spec.rate_bounds_per_s[1],
        ))
    reload = base.reload_probability
    if reload is not None:
        reload = expit(
            logit(np.clip(reload, plo, phi))
            + rng.normal(0.0, 0.5, spec.n_intervals)
        )
        reload = np.clip(reload, plo, phi)
    return LatentStateParameters(
        initial_occupancy=pi,
        fixed_interreadout_survival=np.asarray(q, dtype=float),
        switch_off_rate_per_s=np.asarray(rates, dtype=float),
        dark_mean=dark,
        bright_mean=bright,
        dark_sigma=np.maximum(
            base.dark_sigma * np.exp(rng.normal(0.0, 0.2, spec.n_frames)),
            np.finfo(float).eps,
        ),
        bright_sigma=np.maximum(
            base.bright_sigma * np.exp(rng.normal(0.0, 0.2, spec.n_frames)),
            np.finfo(float).eps,
        ),
        bright_initial_rate_per_s=bright_rate,
        reload_probability=None if reload is None else np.asarray(reload),
    )


def _m_step_initial(
    gamma_initial_occupied: FloatArray,
    current: LatentStateParameters,
    bright_wait_times_s: FloatArray | None,
    spec: LatentModelSpec,
) -> tuple[float, float | None]:
    plo, phi = spec.probability_bounds
    if not spec.bright_initial:
        return float(np.clip(np.mean(gamma_initial_occupied), plo, phi)), None
    assert bright_wait_times_s is not None
    assert current.bright_initial_rate_per_s is not None
    bounds = (
        (float(logit(plo)), float(logit(phi))),
        (
            float(np.log(spec.rate_bounds_per_s[0])),
            float(np.log(spec.rate_bounds_per_s[1])),
        ),
    )

    def objective(theta: FloatArray) -> float:
        pi0 = float(expit(theta[0]))
        rate = float(np.exp(theta[1]))
        probability = np.clip(
            pi0 * np.exp(-rate * bright_wait_times_s),
            _TINY,
            1.0 - np.finfo(float).eps,
        )
        value = np.sum(
            gamma_initial_occupied * np.log(probability)
            + (1.0 - gamma_initial_occupied) * np.log1p(-probability)
        )
        return -float(value)

    start = np.array([
        logit(np.clip(current.initial_occupancy, plo, phi)),
        np.log(current.bright_initial_rate_per_s),
    ])
    result = minimize(objective, start, method="L-BFGS-B", bounds=bounds)
    theta = result.x if np.all(np.isfinite(result.x)) else start
    return float(expit(theta[0])), float(np.exp(theta[1]))


def _m_step_transitions(
    expected_transitions: FloatArray,
    transition_times_s: FloatArray,
    current: LatentStateParameters,
    spec: LatentModelSpec,
    fixed_switch_rates: FloatArray | None,
) -> tuple[FloatArray, FloatArray, FloatArray | None]:
    plo, phi = spec.probability_bounds
    rate_lo, rate_hi = spec.rate_bounds_per_s
    if fixed_switch_rates is None:
        fixed = np.full(spec.n_switch_off_rates, np.nan)
    else:
        fixed = np.asarray(fixed_switch_rates, dtype=float)
        if fixed.shape != (spec.n_switch_off_rates,):
            raise ValueError("fixed_switch_rates has the wrong length")
        finite_fixed = np.isfinite(fixed)
        if np.any((fixed[finite_fixed] < rate_lo) | (fixed[finite_fixed] > rate_hi)):
            raise ValueError("fixed switch-off rate lies outside model bounds")
    free_rate_indices = np.flatnonzero(~np.isfinite(fixed))
    q_start = np.clip(current.fixed_interreadout_survival, plo, phi)
    theta_start = [*logit(q_start)]
    theta_start.extend(
        np.log(current.switch_off_rate_per_s[free_rate_indices])
    )
    bounds = [(float(logit(plo)), float(logit(phi)))] * spec.n_intervals
    bounds.extend([(float(np.log(rate_lo)), float(np.log(rate_hi)))] * len(
        free_rate_indices
    ))

    w11 = expected_transitions[:, :, 1, 1]
    w10 = expected_transitions[:, :, 1, 0]

    def decode(theta: FloatArray) -> tuple[FloatArray, FloatArray]:
        q = expit(theta[:spec.n_intervals])
        rates = fixed.copy()
        rates[free_rate_indices] = np.exp(theta[spec.n_intervals:])
        return np.asarray(q), rates

    def objective(theta: FloatArray) -> float:
        q, rates = decode(theta)
        rate_by_interval = (
            np.repeat(rates, spec.n_intervals)
            if rates.size == 1
            else rates
        )
        survival = np.clip(
            q[None, :]
            * np.exp(-transition_times_s * rate_by_interval[None, :]),
            _TINY,
            1.0 - np.finfo(float).eps,
        )
        value = np.sum(w11 * np.log(survival) + w10 * np.log1p(-survival))
        return -float(value)

    result = minimize(
        objective,
        np.asarray(theta_start),
        method="L-BFGS-B",
        bounds=bounds,
    )
    theta = result.x if np.all(np.isfinite(result.x)) else np.asarray(theta_start)
    q, rates = decode(theta)

    reload: FloatArray | None = None
    if spec.allow_reloading:
        w01 = np.sum(expected_transitions[:, :, 0, 1], axis=0)
        w00 = np.sum(expected_transitions[:, :, 0, 0], axis=0)
        reload = np.clip(w01 / np.maximum(w00 + w01, _TINY), plo, phi)
    return q, rates, reload


def _m_step(
    observations: FloatArray,
    transition_times_s: FloatArray,
    bright_wait_times_s: FloatArray | None,
    posterior: PosteriorResult,
    current: LatentStateParameters,
    spec: LatentModelSpec,
    sigma_floor: float,
    fixed_switch_rates: FloatArray | None,
) -> LatentStateParameters:
    gamma = posterior.state_probability
    dark_mean = np.empty(spec.n_frames)
    bright_mean = np.empty(spec.n_frames)
    dark_sigma = np.empty(spec.n_frames)
    bright_sigma = np.empty(spec.n_frames)
    for frame in range(spec.n_frames):
        finite = np.isfinite(observations[:, frame])
        values = observations[finite, frame]
        for state in (0, 1):
            weights = gamma[finite, frame, state]
            total_weight = float(np.sum(weights))
            if total_weight <= 1e-8:
                raise FloatingPointError(
                    f"state {state} has no emission weight in frame {frame}"
                )
            mean = float(np.sum(weights * values) / total_weight)
            variance = float(
                np.sum(weights * (values - mean) ** 2) / total_weight
            )
            sigma = max(float(np.sqrt(max(variance, 0.0))), sigma_floor)
            if state == 0:
                dark_mean[frame], dark_sigma[frame] = mean, sigma
            else:
                bright_mean[frame], bright_sigma[frame] = mean, sigma
        if bright_mean[frame] <= dark_mean[frame]:
            raise FloatingPointError(
                f"emission ordering collapsed in frame {frame}"
            )

    initial, bright_rate = _m_step_initial(
        gamma[:, 0, 1], current, bright_wait_times_s, spec
    )
    q, rates, reload = _m_step_transitions(
        posterior.transition_probability,
        transition_times_s,
        current,
        spec,
        fixed_switch_rates,
    )
    return LatentStateParameters(
        initial_occupancy=initial,
        fixed_interreadout_survival=q,
        switch_off_rate_per_s=rates,
        dark_mean=dark_mean,
        bright_mean=bright_mean,
        dark_sigma=dark_sigma,
        bright_sigma=bright_sigma,
        bright_initial_rate_per_s=bright_rate,
        reload_probability=reload,
    )


def _run_em(
    observations: FloatArray,
    transition_times_s: FloatArray,
    bright_wait_times_s: FloatArray | None,
    initial: LatentStateParameters,
    spec: LatentModelSpec,
    sigma_floor: float,
    *,
    max_iterations: int,
    tolerance_per_trajectory: float,
    fixed_switch_rates: FloatArray | None = None,
) -> _EMResult:
    current = initial
    current_ll = forward_log_likelihood(
        observations,
        transition_times_s,
        current,
        spec,
        bright_wait_times_s=bright_wait_times_s,
    ).total_log_likelihood
    tolerance = tolerance_per_trajectory * observations.shape[0]
    for iteration in range(1, max_iterations + 1):
        try:
            posterior = posterior_state_probabilities(
                observations,
                transition_times_s,
                current,
                spec,
                bright_wait_times_s=bright_wait_times_s,
            )
            updated = _m_step(
                observations,
                transition_times_s,
                bright_wait_times_s,
                posterior,
                current,
                spec,
                sigma_floor,
                fixed_switch_rates,
            )
            updated_ll = forward_log_likelihood(
                observations,
                transition_times_s,
                updated,
                spec,
                bright_wait_times_s=bright_wait_times_s,
            ).total_log_likelihood
        except (FloatingPointError, ValueError) as exc:
            return _EMResult(
                current, current_ll, False, iteration - 1, str(exc)
            )
        if not np.isfinite(updated_ll):
            return _EMResult(
                current, current_ll, False, iteration, "non-finite likelihood"
            )
        improvement = updated_ll - current_ll
        if improvement < -max(1e-7, 1e-9 * abs(current_ll)):
            return _EMResult(
                current,
                current_ll,
                False,
                iteration,
                f"EM likelihood decreased by {improvement:.6g}",
            )
        current, current_ll = updated, updated_ll
        if improvement <= tolerance:
            return _EMResult(
                current, current_ll, True, iteration, "converged"
            )
    return _EMResult(
        current,
        current_ll,
        False,
        max_iterations,
        "maximum iterations reached",
    )


def _parameter_distance(
    a: LatentStateParameters,
    b: LatentStateParameters,
) -> float:
    """Dimensionless maximum discrepancy for multi-start agreement."""
    distances: list[float] = [
        abs(a.initial_occupancy - b.initial_occupancy),
        float(np.max(np.abs(
            a.fixed_interreadout_survival - b.fixed_interreadout_survival
        ))),
        float(np.max(np.abs(np.log(
            a.switch_off_rate_per_s / b.switch_off_rate_per_s
        )))),
        float(np.max(np.abs(np.log(a.dark_sigma / b.dark_sigma)))),
        float(np.max(np.abs(np.log(a.bright_sigma / b.bright_sigma)))),
    ]
    pooled_sigma = np.maximum(
        0.5 * (a.dark_sigma + a.bright_sigma),
        np.finfo(float).eps,
    )
    distances.append(float(np.max(np.abs(a.dark_mean - b.dark_mean) / pooled_sigma)))
    distances.append(float(np.max(
        np.abs(a.bright_mean - b.bright_mean) / pooled_sigma
    )))
    if a.bright_initial_rate_per_s is not None:
        assert b.bright_initial_rate_per_s is not None
        distances.append(abs(np.log(
            a.bright_initial_rate_per_s / b.bright_initial_rate_per_s
        )))
    if a.reload_probability is not None:
        assert b.reload_probability is not None
        distances.append(float(np.max(np.abs(
            a.reload_probability - b.reload_probability
        ))))
    return max(distances)


def _boundary_parameters(
    parameters: LatentStateParameters,
    spec: LatentModelSpec,
    sigma_floor: float,
) -> tuple[str, ...]:
    names: list[str] = []
    probability_tolerance = 0.005
    if (
        parameters.initial_occupancy <= probability_tolerance
        or parameters.initial_occupancy >= 1.0 - probability_tolerance
    ):
        names.append("initial_occupancy")
    for interval, value in enumerate(parameters.fixed_interreadout_survival):
        if value <= probability_tolerance or value >= 1.0 - probability_tolerance:
            names.append(f"fixed_interreadout_survival[{interval}]")
    if parameters.reload_probability is not None:
        for interval, value in enumerate(parameters.reload_probability):
            if value <= probability_tolerance or value >= 1.0 - probability_tolerance:
                names.append(f"reload_probability[{interval}]")
    rate_lo, rate_hi = spec.rate_bounds_per_s
    for index, value in enumerate(parameters.switch_off_rate_per_s):
        if value <= rate_lo * 1.05 or value >= rate_hi / 1.05:
            names.append(f"switch_off_rate_per_s[{index}]")
    if parameters.bright_initial_rate_per_s is not None:
        value = parameters.bright_initial_rate_per_s
        if value <= rate_lo * 1.05 or value >= rate_hi / 1.05:
            names.append("bright_initial_rate_per_s")
    for frame, value in enumerate(parameters.dark_sigma):
        if value <= sigma_floor * 1.05:
            names.append(f"dark_sigma[{frame}]")
    for frame, value in enumerate(parameters.bright_sigma):
        if value <= sigma_floor * 1.05:
            names.append(f"bright_sigma[{frame}]")
    return tuple(names)


def _replace_rate(
    parameters: LatentStateParameters, rates: FloatArray
) -> LatentStateParameters:
    return replace(
        parameters,
        switch_off_rate_per_s=np.asarray(rates, dtype=float),
    )


def fit_latent_state(
    observations: ArrayLike,
    transition_times_s: ArrayLike,
    spec: LatentModelSpec,
    *,
    bright_wait_times_s: ArrayLike | None = None,
    n_starts: int = 8,
    random_seed: int = 0,
    max_iterations: int = 300,
    tolerance_per_trajectory: float = 1e-7,
    near_best_log_likelihood_tolerance: float = 0.1,
) -> LatentStateFit:
    """Fit frame-specific Gaussian emissions and operational transitions.

    Every row is one indivisible trajectory.  This function does not create a
    split and must never receive validation or test rows during training.
    Random initializations are reproducible for a fixed ``random_seed``.
    """
    if n_starts < 2:
        raise ValueError("n_starts must be at least two for agreement diagnostics")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if tolerance_per_trajectory <= 0.0:
        raise ValueError("tolerance_per_trajectory must be positive")
    x = _as_observations(observations, spec)
    t = _as_transition_times(transition_times_s, x.shape[0], spec)
    bright_t = _as_bright_wait_times(bright_wait_times_s, x.shape[0], spec)
    base, sigma_floor = _initial_parameters(x, t, spec)
    data_scale = max(float(np.nanstd(x)), 1.0)
    rng = np.random.default_rng(random_seed)

    starts = [base]
    starts.extend(
        _jitter_parameters(base, spec, rng, data_scale)
        for _ in range(n_starts - 1)
    )
    results: list[_EMResult] = []
    summaries: list[FitStartSummary] = []
    for index, initial in enumerate(starts):
        result = _run_em(
            x,
            t,
            bright_t,
            initial,
            spec,
            sigma_floor,
            max_iterations=max_iterations,
            tolerance_per_trajectory=tolerance_per_trajectory,
        )
        results.append(result)
        summaries.append(FitStartSummary(
            start_index=index,
            converged=result.converged,
            iterations=result.iterations,
            log_likelihood=result.log_likelihood,
            switch_off_rate_per_s=tuple(
                float(v) for v in result.parameters.switch_off_rate_per_s
            ),
            fixed_interreadout_survival=tuple(
                float(v)
                for v in result.parameters.fixed_interreadout_survival
            ),
            initial_occupancy=float(result.parameters.initial_occupancy),
            message=result.message,
        ))
    converged_indices = [i for i, result in enumerate(results) if result.converged]
    candidate_indices = converged_indices or list(range(len(results)))
    selected = max(candidate_indices, key=lambda i: results[i].log_likelihood)
    best = results[selected]

    near_best_indices = [
        i for i in converged_indices
        if best.log_likelihood - results[i].log_likelihood
        <= near_best_log_likelihood_tolerance
    ]
    if len(near_best_indices) >= 2:
        agreement = max(
            _parameter_distance(best.parameters, results[i].parameters)
            for i in near_best_indices
        )
    else:
        agreement = None

    dprime = (
        (best.parameters.bright_mean - best.parameters.dark_mean)
        / np.sqrt(
            0.5 * (
                best.parameters.dark_sigma ** 2
                + best.parameters.bright_sigma ** 2
            )
        )
    )
    unique_time_count = np.array([
        np.unique(t[:, interval]).size for interval in range(spec.n_intervals)
    ])
    time_range = np.ptp(t, axis=0)
    design_varies = bool(np.any(
        (unique_time_count >= 3)
        & (time_range > np.finfo(float).eps)
    ))
    if bright_t is None:
        bright_design_varies: bool | None = None
    else:
        bright_design_varies = bool(
            np.unique(bright_t).size >= 3
            and np.ptp(bright_t) > np.finfo(float).eps
        )

    drop_half: float | None = None
    drop_double: float | None = None
    if spec.shared_switch_off_rate:
        mle_rate = best.parameters.switch_off_rate_per_s[0]
        rate_lo, rate_hi = spec.rate_bounds_per_s
        half = max(0.5 * mle_rate, rate_lo)
        double = min(2.0 * mle_rate, rate_hi)
        half_ll = forward_log_likelihood(
            x,
            t,
            _replace_rate(best.parameters, np.array([half])),
            spec,
            bright_wait_times_s=bright_t,
        ).total_log_likelihood
        double_ll = forward_log_likelihood(
            x,
            t,
            _replace_rate(best.parameters, np.array([double])),
            spec,
            bright_wait_times_s=bright_t,
        ).total_log_likelihood
        drop_half = float(best.log_likelihood - half_ll)
        drop_double = float(best.log_likelihood - double_ll)
    local_informative = bool(
        design_varies
        and drop_half is not None
        and drop_double is not None
        and min(drop_half, drop_double) > 0.05
    )
    diagnostics = LatentFitDiagnostics(
        boundary_parameters=_boundary_parameters(
            best.parameters, spec, sigma_floor
        ),
        frame_separation_d_prime=dprime,
        design_has_hold_time_variation=design_varies,
        design_has_bright_wait_variation=bright_design_varies,
        near_best_start_count=len(near_best_indices),
        near_best_log_likelihood_tolerance=near_best_log_likelihood_tolerance,
        start_agreement_max_distance=agreement,
        fixed_nuisance_rate_loglik_drop_half=drop_half,
        fixed_nuisance_rate_loglik_drop_double=drop_double,
        locally_rate_informative=local_informative,
    )
    return LatentStateFit(
        parameters=best.parameters,
        spec=spec,
        log_likelihood=best.log_likelihood,
        n_trajectories=x.shape[0],
        n_finite_observations=int(np.count_nonzero(np.isfinite(x))),
        converged=best.converged,
        selected_start=selected,
        starts=tuple(summaries),
        diagnostics=diagnostics,
        sigma_floor=sigma_floor,
    )


def _profile_confidence_limits(
    rates: FloatArray,
    log_likelihood: FloatArray,
    confidence_level: float,
) -> tuple[float | None, float | None, int]:
    finite = np.isfinite(log_likelihood)
    if not np.any(finite):
        return None, None, 0
    best_index = int(np.nanargmax(log_likelihood))
    cutoff = float(np.nanmax(log_likelihood)) - 0.5 * float(
        chi2.ppf(confidence_level, df=1)
    )

    def interpolate(i_out: int, i_in: int) -> float:
        x0, x1 = np.log(rates[[i_out, i_in]])
        y0, y1 = log_likelihood[[i_out, i_in]]
        if not np.isfinite(y0) or y1 == y0:
            return float(np.exp(x1))
        fraction = np.clip((cutoff - y0) / (y1 - y0), 0.0, 1.0)
        return float(np.exp(x0 + fraction * (x1 - x0)))

    lower_index = best_index
    while (
        lower_index > 0
        and np.isfinite(log_likelihood[lower_index - 1])
        and log_likelihood[lower_index - 1] >= cutoff
    ):
        lower_index -= 1
    lower = (
        None
        if lower_index == 0 and log_likelihood[0] >= cutoff
        else interpolate(lower_index - 1, lower_index)
    )

    upper_index = best_index
    while (
        upper_index < rates.size - 1
        and np.isfinite(log_likelihood[upper_index + 1])
        and log_likelihood[upper_index + 1] >= cutoff
    ):
        upper_index += 1
    upper = (
        None
        if upper_index == rates.size - 1
        and log_likelihood[-1] >= cutoff
        else interpolate(upper_index + 1, upper_index)
    )
    return lower, upper, best_index


def profile_switch_off_rate(
    fit: LatentStateFit,
    observations: ArrayLike,
    transition_times_s: ArrayLike,
    rate_grid_per_s: Sequence[float],
    *,
    bright_wait_times_s: ArrayLike | None = None,
    interval_index: int = 0,
    confidence_level: float = 0.95,
    max_iterations: int = 200,
    tolerance_per_trajectory: float = 1e-7,
) -> ProfileLikelihood:
    """Profile one switch-off rate while refitting all nuisance parameters."""
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie inside (0, 1)")
    spec = fit.spec
    if spec.shared_switch_off_rate:
        if interval_index != 0:
            raise ValueError("shared-rate models have only interval_index=0")
        rate_parameter_index = 0
    else:
        if not 0 <= interval_index < spec.n_intervals:
            raise ValueError("interval_index is out of range")
        rate_parameter_index = interval_index
    rates = np.asarray(rate_grid_per_s, dtype=float)
    if rates.ndim != 1 or rates.size < 5:
        raise ValueError("rate_grid_per_s must contain at least five values")
    if not np.all(np.isfinite(rates)) or np.any(rates <= 0.0):
        raise ValueError("profile rates must be finite and positive")
    if np.any(np.diff(rates) <= 0.0):
        raise ValueError("profile rates must be strictly increasing")
    rate_lo, rate_hi = spec.rate_bounds_per_s
    if np.any((rates < rate_lo) | (rates > rate_hi)):
        raise ValueError("profile rates must lie inside the model rate bounds")

    x = _as_observations(observations, spec)
    t = _as_transition_times(transition_times_s, x.shape[0], spec)
    bright_t = _as_bright_wait_times(bright_wait_times_s, x.shape[0], spec)
    profile_ll = np.empty(rates.size)
    profile_converged = np.zeros(rates.size, dtype=bool)
    for grid_index, rate in enumerate(rates):
        fixed = np.full(spec.n_switch_off_rates, np.nan)
        fixed[rate_parameter_index] = rate
        initial_rates = fit.parameters.switch_off_rate_per_s.copy()
        initial_rates[rate_parameter_index] = rate
        initial = _replace_rate(fit.parameters, initial_rates)
        result = _run_em(
            x,
            t,
            bright_t,
            initial,
            spec,
            fit.sigma_floor,
            max_iterations=max_iterations,
            tolerance_per_trajectory=tolerance_per_trajectory,
            fixed_switch_rates=fixed,
        )
        profile_ll[grid_index] = result.log_likelihood
        profile_converged[grid_index] = result.converged
    lower, upper, best_index = _profile_confidence_limits(
        rates, profile_ll, confidence_level
    )
    return ProfileLikelihood(
        rate_grid_per_s=rates,
        log_likelihood=profile_ll,
        converged=profile_converged,
        confidence_level=confidence_level,
        lower_confidence_limit_per_s=lower,
        upper_confidence_limit_per_s=upper,
        maximum_rate_per_s=float(rates[best_index]),
        maximum_log_likelihood=float(profile_ll[best_index]),
        interval_index=interval_index,
    )


def compare_held_out_log_likelihood(
    latent_total_log_likelihood: float,
    baseline_total_log_likelihood: float,
    *,
    n_trajectories: int,
    n_finite_observations: int,
) -> HeldOutLikelihoodComparison:
    """Normalize a held-out likelihood difference without refitting either model."""
    if n_trajectories <= 0 or n_finite_observations <= 0:
        raise ValueError("held-out sample sizes must be positive")
    values = (latent_total_log_likelihood, baseline_total_log_likelihood)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("held-out log likelihoods must be finite")
    delta = float(latent_total_log_likelihood - baseline_total_log_likelihood)
    return HeldOutLikelihoodComparison(
        latent_total_log_likelihood=float(latent_total_log_likelihood),
        baseline_total_log_likelihood=float(baseline_total_log_likelihood),
        delta_total_log_likelihood=delta,
        delta_per_trajectory=delta / n_trajectories,
        delta_per_finite_observation=delta / n_finite_observations,
        n_trajectories=n_trajectories,
        n_finite_observations=n_finite_observations,
    )


def score_against_held_out_baseline(
    parameters: LatentStateParameters,
    spec: LatentModelSpec,
    observations: ArrayLike,
    transition_times_s: ArrayLike,
    baseline_total_log_likelihood: float,
    *,
    bright_wait_times_s: ArrayLike | None = None,
) -> HeldOutLikelihoodComparison:
    """Score frozen parameters once and compare with a frozen baseline."""
    likelihood = forward_log_likelihood(
        observations,
        transition_times_s,
        parameters,
        spec,
        bright_wait_times_s=bright_wait_times_s,
    )
    return compare_held_out_log_likelihood(
        likelihood.total_log_likelihood,
        baseline_total_log_likelihood,
        n_trajectories=likelihood.trajectory_log_likelihood.size,
        n_finite_observations=likelihood.n_finite_observations,
    )


def evaluate_latent_state_gate(
    fit: LatentStateFit,
    evidence: LatentGateEvidence,
    criteria: LatentGateCriteria = LatentGateCriteria(),
) -> LatentGateDecision:
    """Accept only a converged, predictive, identifiable, permitted model.

    The decision concerns whether the model is suitable for a public
    model-based result.  It never upgrades posterior state probabilities into
    empirical labels or a measured readout fidelity.
    """
    failures: list[str] = []
    cautions: list[str] = [
        "Posterior occupancy is conditional on the stated emission and "
        "transition model; it is not empirical occupancy ground truth."
    ]
    prerequisites = (
        ("timing semantics were not verified", evidence.timing_semantics_verified),
        ("geometry was not validated", evidence.geometry_validated),
        ("background method was not frozen", evidence.background_method_frozen),
        ("shot-level split was not frozen", evidence.shot_split_frozen),
        ("held-out baselines are incomplete", evidence.held_out_baselines_complete),
    )
    failures.extend(message for message, passed in prerequisites if not passed)
    if not fit.converged:
        failures.append("selected multi-start fit did not converge")
    if (
        criteria.reject_boundary_parameters
        and fit.diagnostics.boundary_parameters
    ):
        failures.append(
            "parameters reached or approached boundaries: "
            + ", ".join(fit.diagnostics.boundary_parameters)
        )
    if not fit.diagnostics.design_has_hold_time_variation:
        failures.append("hold-time design cannot separate a rate from fixed survival")
    if fit.diagnostics.near_best_start_count < criteria.min_near_best_starts:
        failures.append("too few independent starts reached the best likelihood")
    agreement = fit.diagnostics.start_agreement_max_distance
    if agreement is None:
        failures.append("multi-start parameter agreement is unavailable")
    elif agreement > criteria.max_start_agreement_distance:
        failures.append(
            f"multi-start parameter distance {agreement:.3g} exceeds "
            f"{criteria.max_start_agreement_distance:.3g}"
        )
    if np.min(fit.diagnostics.frame_separation_d_prime) < (
        criteria.min_frame_separation_d_prime
    ):
        failures.append("one or more frames have insufficient emission separation")
    if evidence.synthetic_recovery_passed is not True:
        failures.append("synthetic parameter recovery did not pass")
    if evidence.posterior_predictive_checks_passed is not True:
        failures.append("posterior predictive checks did not pass")
    if evidence.shot_cluster_uncertainty_passed is not True:
        failures.append(
            "complete-shot clustered uncertainty for latent transition "
            "parameters was not supplied"
        )

    comparison = evidence.held_out_comparison
    if comparison is None:
        failures.append("held-out predictive likelihood was not supplied")
    elif comparison.delta_per_trajectory <= (
        criteria.min_delta_loglik_per_trajectory
    ):
        failures.append(
            "latent model did not improve held-out likelihood by the "
            "predeclared amount"
        )

    profile = evidence.rate_profile
    if profile is None:
        failures.append("profile-likelihood identifiability evidence was not supplied")
    elif not profile.has_bounded_interval:
        failures.append("switch-off rate profile interval is not bounded")
    else:
        ratio = profile.confidence_ratio
        assert ratio is not None
        if ratio > criteria.max_profile_confidence_ratio:
            failures.append(
                f"switch-off rate profile ratio {ratio:.3g} exceeds "
                f"{criteria.max_profile_confidence_ratio:.3g}"
            )
        if not np.all(profile.converged):
            failures.append("one or more profile-grid nuisance fits did not converge")
    return LatentGateDecision(
        accepted=not failures,
        failures=tuple(failures),
        cautions=tuple(cautions),
    )


__all__ = [
    "FitStartSummary",
    "HeldOutLikelihoodComparison",
    "LatentFitDiagnostics",
    "LatentGateCriteria",
    "LatentGateDecision",
    "LatentGateEvidence",
    "LatentModelSpec",
    "LatentStateFit",
    "LatentStateParameters",
    "LikelihoodResult",
    "PosteriorResult",
    "ProfileLikelihood",
    "SyntheticLatentData",
    "compare_held_out_log_likelihood",
    "evaluate_latent_state_gate",
    "fit_latent_state",
    "forward_log_likelihood",
    "posterior_state_probabilities",
    "profile_switch_off_rate",
    "score_against_held_out_baseline",
    "simulate_latent_state",
]
