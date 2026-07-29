"""Synthetic tests for the gated latent-state engine.

The synthetic states are available only to verify implementation recovery.
Real experimental analyses do not have these labels.
"""
from __future__ import annotations

import numpy as np

from fluorescence_inference.latent_state import (
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


def _parameters(
    n_frames: int,
    *,
    switch_rate: float = 0.28,
    bright_initial_rate: float | None = None,
) -> LatentStateParameters:
    frame = np.arange(n_frames, dtype=float)
    return LatentStateParameters(
        initial_occupancy=0.78,
        fixed_interreadout_survival=np.linspace(
            0.91, 0.95, n_frames - 1
        ),
        switch_off_rate_per_s=np.array([switch_rate]),
        dark_mean=0.10 * np.sin(frame),
        bright_mean=4.5 + 0.12 * np.cos(frame),
        dark_sigma=np.full(n_frames, 0.72),
        bright_sigma=np.full(n_frames, 0.82),
        bright_initial_rate_per_s=bright_initial_rate,
    )


def _condition_times(n: int) -> np.ndarray:
    points = np.linspace(0.1, 2.1, 11)
    return np.resize(points, n)


def test_forward_backward_normalizes_and_forbids_reloading():
    spec = LatentModelSpec(n_frames=4)
    parameters = _parameters(4)
    times = _condition_times(120)
    data = simulate_latent_state(
        parameters, spec, times, random_seed=141
    )

    likelihood = forward_log_likelihood(
        data.observations, times, parameters, spec
    )
    posterior = posterior_state_probabilities(
        data.observations, times, parameters, spec
    )

    assert np.isfinite(likelihood.total_log_likelihood)
    np.testing.assert_allclose(
        likelihood.trajectory_log_likelihood,
        posterior.trajectory_log_likelihood,
    )
    np.testing.assert_allclose(
        posterior.state_probability.sum(axis=2), 1.0, atol=1e-12
    )
    np.testing.assert_allclose(
        posterior.transition_probability.sum(axis=(2, 3)), 1.0, atol=1e-12
    )
    assert np.all(posterior.transition_probability[:, :, 0, 1] == 0.0)
    expected = (
        parameters.fixed_interreadout_survival[None, :]
        * np.exp(-parameters.switch_off_rate_per_s[0] * data.transition_times_s)
    )
    np.testing.assert_allclose(
        parameters.survival_probability(data.transition_times_s), expected
    )


def test_deterministic_multiple_initializations():
    spec = LatentModelSpec(n_frames=3)
    parameters = _parameters(3, switch_rate=0.34)
    times = _condition_times(420)
    data = simulate_latent_state(
        parameters, spec, times, random_seed=287
    )

    kwargs = dict(
        n_starts=3,
        random_seed=20260728,
        max_iterations=220,
        tolerance_per_trajectory=2e-7,
    )
    first = fit_latent_state(data.observations, times, spec, **kwargs)
    second = fit_latent_state(data.observations, times, spec, **kwargs)

    assert first.selected_start == second.selected_start
    assert first.converged == second.converged
    assert first.log_likelihood == second.log_likelihood
    np.testing.assert_array_equal(
        first.parameters.switch_off_rate_per_s,
        second.parameters.switch_off_rate_per_s,
    )
    np.testing.assert_array_equal(
        first.parameters.fixed_interreadout_survival,
        second.parameters.fixed_interreadout_survival,
    )
    assert first.starts == second.starts


def test_synthetic_switch_off_parameter_recovery():
    spec = LatentModelSpec(n_frames=4)
    truth = _parameters(4, switch_rate=0.31)
    times = _condition_times(1_100)
    data = simulate_latent_state(truth, spec, times, random_seed=9182)
    fit = fit_latent_state(
        data.observations,
        times,
        spec,
        n_starts=4,
        random_seed=46,
        max_iterations=300,
        tolerance_per_trajectory=1e-7,
    )

    assert fit.converged
    estimate = fit.parameters.switch_off_rate_per_s[0]
    assert abs(estimate - truth.switch_off_rate_per_s[0]) < 0.08
    np.testing.assert_allclose(
        fit.parameters.fixed_interreadout_survival,
        truth.fixed_interreadout_survival,
        atol=0.045,
    )
    assert np.all(fit.parameters.bright_mean > fit.parameters.dark_mean)
    assert np.min(fit.diagnostics.frame_separation_d_prime) > 4.0
    assert fit.diagnostics.design_has_hold_time_variation
    assert fit.diagnostics.locally_rate_informative
    assert not fit.diagnostics.boundary_parameters


def test_synthetic_bright_initial_decay_recovery():
    spec = LatentModelSpec(n_frames=2, bright_initial=True)
    truth = _parameters(
        2, switch_rate=0.10, bright_initial_rate=0.48
    )
    bright_wait = np.resize(np.linspace(0.1, 1.9, 10), 1_000)
    transition_times = np.resize(np.linspace(0.02, 0.12, 6), 1_000)
    data = simulate_latent_state(
        truth,
        spec,
        transition_times,
        bright_wait_times_s=bright_wait,
        random_seed=776,
    )
    fit = fit_latent_state(
        data.observations,
        transition_times,
        spec,
        bright_wait_times_s=bright_wait,
        n_starts=4,
        random_seed=11,
        max_iterations=300,
        tolerance_per_trajectory=1e-7,
    )

    assert fit.converged
    assert fit.parameters.bright_initial_rate_per_s is not None
    assert abs(fit.parameters.bright_initial_rate_per_s - 0.48) < 0.09
    assert abs(fit.parameters.initial_occupancy - 0.78) < 0.06
    assert fit.diagnostics.design_has_bright_wait_variation


def test_held_out_helper_scores_frozen_parameters_and_gate_defaults_reject():
    spec = LatentModelSpec(n_frames=3)
    parameters = _parameters(3)
    times = _condition_times(300)
    data = simulate_latent_state(
        parameters, spec, times, random_seed=39
    )
    train = np.arange(200)
    test = np.arange(200, 300)
    fit = fit_latent_state(
        data.observations[train],
        times[train],
        spec,
        n_starts=3,
        random_seed=8,
        max_iterations=220,
    )
    frozen_score = forward_log_likelihood(
        data.observations[test], times[test], fit.parameters, spec
    )
    comparison = score_against_held_out_baseline(
        fit.parameters,
        spec,
        data.observations[test],
        times[test],
        baseline_total_log_likelihood=frozen_score.total_log_likelihood - 5.0,
    )

    assert comparison.delta_total_log_likelihood == 5.0
    assert comparison.n_trajectories == len(test)
    assert comparison.latent_predicts_better

    decision = evaluate_latent_state_gate(
        fit,
        LatentGateEvidence(held_out_comparison=comparison),
    )
    assert not decision.accepted
    assert "timing semantics were not verified" in decision.failures
    assert any("profile-likelihood" in reason for reason in decision.failures)
    assert any("synthetic parameter recovery" in reason for reason in decision.failures)
    assert any("complete-shot clustered" in reason for reason in decision.failures)


def test_profile_likelihood_refits_nuisance_parameters():
    spec = LatentModelSpec(n_frames=3)
    truth = _parameters(3, switch_rate=0.30)
    times = _condition_times(500)
    data = simulate_latent_state(truth, spec, times, random_seed=991)
    fit = fit_latent_state(
        data.observations,
        times,
        spec,
        n_starts=3,
        random_seed=17,
        max_iterations=220,
    )
    grid = np.geomspace(0.08, 0.9, 7)
    profile = profile_switch_off_rate(
        fit,
        data.observations,
        times,
        grid,
        max_iterations=160,
    )

    assert profile.rate_grid_per_s.shape == (7,)
    assert profile.log_likelihood.shape == (7,)
    assert np.all(np.isfinite(profile.log_likelihood))
    assert np.count_nonzero(profile.converged) >= 5
    assert 0.12 < profile.maximum_rate_per_s < 0.7
