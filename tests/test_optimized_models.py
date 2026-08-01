"""Synthetic identification tests for optimized repeated-imaging models."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fluorescence_inference.heldout_emissions import (
    EmissionModel,
    GaussianComponents,
)
from fluorescence_inference.optimized_analysis import (
    _public_emission_distribution,
    emission_residual_diagnostic,
)
from fluorescence_inference.optimized_models import (
    bootstrap_repeated_imaging,
    bootstrap_repeated_imaging_blocks,
    compare_repeated_models,
    fit_repeated_imaging,
    heldout_matched_prefix_contrasts,
    loss_budget,
    loss_budget_uncertainty,
    matched_total_time_contrasts,
    prefix_curve,
    repeated_shot_order_sensitivity,
)


def _repeated_calls(
    *,
    seed: int = 301,
    q_common: float = 0.95,
    q_by_exposure: dict[float, float] | None = None,
    n_shots: int = 120,
    n_sites: int = 24,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lambda_bright = 0.08
    lambda_dark = 0.03
    rows = []
    run_specs = (
        ("run_50", 0.05, 0.43),
        ("run_100", 0.10, 0.55),
        ("run_200", 0.20, 0.67),
    )
    for run, exposure, initial in run_specs:
        q = q_common if q_by_exposure is None else q_by_exposure[exposure]
        condition = f"exposure_{int(exposure * 1000)}ms"
        for shot in range(n_shots):
            split = (
                "train"
                if shot < int(0.6 * n_shots)
                else "validation"
                if shot < int(0.8 * n_shots)
                else "test"
            )
            for frame in range(5):
                pulses = frame + 1
                bright = pulses * exposure
                dark = frame * 0.01
                probability = (
                    initial
                    * np.exp(-lambda_bright * bright - lambda_dark * dark)
                    * q**frame
                )
                for site in range(n_sites):
                    call = int(rng.random() < probability)
                    rows.append(
                        {
                            "run_id": run,
                            "shot_id": shot,
                            "shot_order": shot,
                            "condition_id": condition,
                            "split": split,
                            "frame_index": frame,
                            "site_id": site,
                            "exposure_s": exposure,
                            "cumulative_bright_s": bright,
                            "cumulative_dark_s": dark,
                            "pulse_count": pulses,
                            "apparent_occupied": call,
                            "posterior_occupied": float(call),
                        }
                    )
    return pd.DataFrame(rows)


def test_common_additional_readout_and_run_loading_recovery():
    data = _repeated_calls(q_common=0.94)
    train = data.loc[data["split"] == "train"]
    fit = fit_repeated_imaging(
        train,
        model="common_additional_readout",
        lambda_bright=0.08,
        lambda_dark=0.03,
    )

    assert fit.q_common == pytest.approx(0.94, abs=0.012)
    assert fit.initial_occupancy_by_run == pytest.approx(
        {"run_50": 0.43, "run_100": 0.55, "run_200": 0.67}, abs=0.025
    )
    assert fit.n_independent_shots == 3 * 72
    assert fit.converged


def test_validation_selects_pulse_model_and_recovers_continuous_null():
    pulse = _repeated_calls(seed=302, q_common=0.93)
    selected, table, _ = compare_repeated_models(
        pulse.loc[pulse["split"] == "train"],
        pulse.loc[pulse["split"] == "validation"],
        lambda_bright=0.08,
        lambda_dark=0.03,
    )
    assert selected == "common_additional_readout"
    assert table.loc[
        table["candidate"] == selected, "selected_on_validation"
    ].iloc[0]

    null = _repeated_calls(seed=303, q_common=1.0)
    null_selected, null_table, _ = compare_repeated_models(
        null.loc[null["split"] == "train"],
        null.loc[null["split"] == "validation"],
        lambda_bright=0.08,
        lambda_dark=0.03,
    )
    assert null_selected == "continuous_only"
    assert not null_table.loc[
        null_table["candidate"] == "common_additional_readout",
        "complexity_gate",
    ].iloc[0]["passed"]


def test_exposure_specific_additional_readout_recovery():
    truth = {0.05: 0.91, 0.10: 0.95, 0.20: 0.985}
    data = _repeated_calls(seed=304, q_by_exposure=truth)
    fit = fit_repeated_imaging(
        data.loc[data["split"] == "train"],
        model="exposure_specific_additional_readout",
        lambda_bright=0.08,
        lambda_dark=0.03,
    )
    assert fit.q_by_exposure == pytest.approx(truth, abs=0.015)


def test_prefix_curve_keeps_complete_shots_and_exact_cumulative_timing():
    data = _repeated_calls(n_shots=10, n_sites=4)
    curve = prefix_curve(data, n_boot=20, seed=8)

    assert len(curve) == 15
    assert set(curve["n_independent_shots"]) == {10}
    last_50 = curve.loc[
        (curve["exposure_s"] == 0.05) & (curve["pulse_count"] == 5)
    ].iloc[0]
    assert last_50["cumulative_bright_s"] == pytest.approx(0.25)
    assert last_50["cumulative_dark_s"] == pytest.approx(0.04)


def test_matched_total_time_comparisons_are_the_predeclared_prefixes():
    data = _repeated_calls(n_shots=30, n_sites=8)
    fit = fit_repeated_imaging(
        data.loc[data["split"] == "train"],
        model="common_additional_readout",
        lambda_bright=0.08,
        lambda_dark=0.03,
    )
    rows = matched_total_time_contrasts(fit)

    observed = {
        row["total_bright_s"]: [
            (item["exposure_s"], item["pulse_count"])
            for item in row["designs"]
        ]
        for row in rows
    }
    assert observed == {
        0.1: [(0.05, 2), (0.10, 1)],
        0.2: [(0.05, 4), (0.10, 2), (0.20, 1)],
        0.4: [(0.10, 4), (0.20, 2)],
    }
    assert all(
        item["cumulative_dark_s"] == pytest.approx(
            (item["pulse_count"] - 1) * 0.01
        )
        for row in rows
        for item in row["designs"]
    )

    heldout = heldout_matched_prefix_contrasts(
        data.loc[data["split"] == "test"], fit, n_boot=20, seed=13
    )
    assert [row["total_bright_s"] for row in heldout] == [0.1, 0.2, 0.4]
    assert all(
        item["difference_cluster_lower"]
        <= item["difference_vs_fewest_pulses"]
        <= item["difference_cluster_upper"]
        for row in heldout
        for item in row["designs"]
    )


def test_loss_segments_sum_to_total_and_uncertainty_propagates_rates():
    data = _repeated_calls(n_shots=30, n_sites=8)
    development = data.loc[data["split"].isin(["train", "validation"])]
    fit = fit_repeated_imaging(
        development,
        model="common_additional_readout",
        lambda_bright=0.08,
        lambda_dark=0.03,
        split_col=None,
    )
    budget = loss_budget(fit, exposure_s=0.1)

    assert len(budget["segments"]) == 5 + 4 + 4
    assert sum(budget["component_loss"].values()) == pytest.approx(
        budget["total_loss"]
    )
    assert budget["final_survival"] == pytest.approx(
        np.prod(
            [
                1.0 - row["conditional_loss_probability"]
                for row in budget["segments"]
            ]
        )
    )

    repeated = pd.DataFrame(
        {
            "additional_readout_retention__common": [
                fit.q_common - 0.01,
                fit.q_common,
                fit.q_common + 0.005,
            ]
        }
    )
    propagated = loss_budget_uncertainty(
        fit,
        repeated,
        bright_rate_draws=[0.06, 0.08, 0.11],
        dark_rate_draws=[0.02, 0.03, 0.05],
        seed=9,
    )
    interval = propagated["by_exposure"]["100ms"]["final_survival"]
    assert interval["lower"] < interval["median"] < interval["upper"]


def test_repeated_bootstrap_resamples_complete_shots():
    data = _repeated_calls(n_shots=20, n_sites=6)
    development = data.loc[data["split"].isin(["train", "validation"])]
    result = bootstrap_repeated_imaging(
        development,
        model="common_additional_readout",
        lambda_bright=0.08,
        lambda_dark=0.03,
        n_boot=10,
        seed=11,
    )
    assert result.n_successful == 10
    assert "additional_readout_retention__common" in result.draws

    blocks = bootstrap_repeated_imaging_blocks(
        development,
        model="common_additional_readout",
        lambda_bright=0.08,
        lambda_dark=0.03,
        n_boot=5,
        seed=12,
        block_size=4,
    )
    assert blocks.n_successful == 5
    assert "additional_readout_retention__common" in blocks.draws


def test_repeated_fit_rejects_mixed_splits():
    data = _repeated_calls(n_shots=10, n_sites=3)
    with pytest.raises(ValueError, match="training rows only"):
        fit_repeated_imaging(
            data,
            model="continuous_only",
            lambda_bright=0.08,
            lambda_dark=0.03,
        )


def test_shot_order_sensitivity_keeps_complete_run_specific_halves():
    data = _repeated_calls(n_shots=20, n_sites=8)
    development = data.loc[data["split"].isin(["train", "validation"])]
    result = repeated_shot_order_sensitivity(
        development,
        selected_model="continuous_only",
        lambda_bright=0.08,
        lambda_dark=0.03,
    )

    assert result["selected_structure_frozen"] == "continuous_only"
    assert result["shots_by_run_and_half"] == {
        run: {"early": 8, "late": 8}
        for run in ("run_50", "run_100", "run_200")
    }
    assert result["halves"]["early"]["n_independent_shots"] == 24
    assert result["halves"]["late"]["n_independent_shots"] == 24
    assert "additional_readout_retention__common" in result["halves"][
        "early"
    ]["common_factor_parameters"]


def test_validation_residual_gate_controls_extra_emission_structures():
    components = GaussianComponents(
        weight_empty=0.5,
        mean_empty=-5.0,
        sigma_empty=1.0,
        weight_occupied=0.5,
        mean_occupied=5.0,
        sigma_occupied=1.0,
        log_likelihood=-1.0,
        n_fit=100,
        converged=True,
    )
    model = EmissionModel(
        name="synthetic_two_component",
        kind="global",
        value_col="count",
        frame_col="frame_index",
        site_col="site_id",
        shot_cols=("run_id", "shot_id"),
        components=components,
        n_train_rows=100,
        training_shot_keys=(),
    )
    rng = np.random.default_rng(41)
    values = np.concatenate(
        [rng.normal(-5.0, 0.8, 1000), rng.normal(5.0, 0.8, 1000)]
    )
    table = pd.DataFrame(
        {
            "run_id": "run",
            "shot_id": np.arange(len(values)) // 20,
            "split": ["validation" if index % 2 == 0 else "test" for index in range(len(values))],
            "frame_index": np.arange(len(values)) % 5,
            "site_id": np.arange(len(values)) % 100,
            "count": values,
        }
    )
    quiet = emission_residual_diagnostic(
        {"selected_model": model, "scored": model.score(table)}
    )
    assert quiet["gate_passed_without_additional_state_or_tail_model"]
    assert not quiet["partial_exposure_state_required"]
    assert not quiet["heavy_tail_alternative_required"]
    public = _public_emission_distribution(
        {"selected_model": model, "scored": model.score(table)}
    )
    assert len(public["bin_centers"]) == 60
    assert len(public["heldout_density"]) == 60
    assert public["n_independent_test_shots"] == 100
    assert "labelled truth" in public["interpretation"]

    ambiguous_table = table.copy()
    ambiguous_table.loc[:149, "count"] = 0.0
    triggered = emission_residual_diagnostic(
        {"selected_model": model, "scored": model.score(ambiguous_table)}
    )
    assert triggered["partial_exposure_state_required"]
    assert triggered["heavy_tail_alternative_required"]
    assert not triggered["third_state_attempted"]
