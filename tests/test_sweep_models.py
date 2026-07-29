"""Synthetic recovery and selection tests for operational sweep models."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fluorescence_inference.sweep_models import (
    bootstrap_dark_retention,
    bootstrap_rate_ratio,
    bright_sensitivity_table,
    compare_bright_models,
    compare_control_models,
    compare_dark_models,
    constant_rate_loss,
    dark_sensitivity_table,
    exact_multiplicative_loss,
    fit_bright_decay,
    fit_control_retention,
    fit_dark_retention,
    make_retention_events,
)


def _split_label(repetition: int, n_repetitions: int) -> str:
    if repetition < round(0.6 * n_repetitions):
        return "train"
    if repetition < round(0.8 * n_repetitions):
        return "validation"
    return "test"


def _dark_events(
    *,
    seed: int = 8,
    n_repetitions: int = 60,
    n_sites: int = 15,
    rates=(0.18, 0.18, 0.18, 0.18),
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = np.linspace(0.1, 2.1, 11)
    intervals = ("0->1", "1->2", "2->3", "3->4")
    q = (0.80, 0.87, 0.90, 0.88)
    rows = []
    shot = 0
    for condition, time in enumerate(times):
        for repetition in range(n_repetitions):
            split = _split_label(repetition, n_repetitions)
            for site in range(n_sites):
                for interval, q_j, rate in zip(intervals, q, rates):
                    p = q_j * np.exp(-rate * time)
                    rows.append(
                        {
                            "run_id": "dark",
                            "shot_id": shot,
                            "condition_id": condition,
                            "site_id": site,
                            "sweep_value_s": time,
                            "interval": interval,
                            "retained_next": int(rng.random() < p),
                            "split": split,
                        }
                    )
            shot += 1
    return pd.DataFrame(rows)


def _bright_calls(
    *,
    seed: int = 18,
    n_repetitions: int = 80,
    n_sites: int = 20,
    pi0: float = 0.84,
    rate: float = 0.42,
    floor: float = 0.0,
    response_col: str = "apparent_occupied",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    shot = 0
    for condition, time in enumerate(np.linspace(0.1, 1.9, 10)):
        probability = floor + (pi0 - floor) * np.exp(-rate * time)
        for repetition in range(n_repetitions):
            split = _split_label(repetition, n_repetitions)
            for site in range(n_sites):
                rows.append(
                    {
                        "run_id": "bright",
                        "shot_id": shot,
                        "condition_id": condition,
                        "site_id": site,
                        "sweep_value_s": time,
                        response_col: int(rng.random() < probability),
                        "split": split,
                    }
                )
            shot += 1
    return pd.DataFrame(rows)


def test_dark_shared_rate_and_interval_retention_recovery():
    data = _dark_events()
    fit = fit_dark_retention(data[data["split"] == "train"], model="shared")
    assert fit.lambda_switch_off == pytest.approx(0.18, abs=0.025)
    assert fit.tau_switch_off == pytest.approx(1.0 / 0.18, rel=0.16)
    truth = {"0->1": 0.80, "1->2": 0.87, "2->3": 0.90, "3->4": 0.88}
    for interval, q in truth.items():
        assert fit.q_by_interval[interval] == pytest.approx(q, abs=0.035)
    assert fit.n_independent_shots == 11 * 36
    assert np.isfinite(fit.aic)
    assert np.isfinite(fit.bic)


def test_validation_selects_interval_specific_rates_when_slopes_really_differ():
    data = _dark_events(
        seed=21,
        n_repetitions=80,
        n_sites=20,
        rates=(0.03, 0.18, 0.48, 0.82),
    )
    selection = compare_dark_models(
        data[data["split"] == "train"],
        data[data["split"] == "validation"],
    )
    assert selection.selected_name == "interval_specific"
    assert selection.table["selected_on_validation"].sum() == 1
    selected_row = selection.table.loc[
        selection.table["candidate"] == "interval_specific"
    ].iloc[0]
    assert bool(selected_row["cluster_gate__complexity_gate_passed"])
    assert selected_row[
        "cluster_gate__n_independent_validation_shots"
    ] == 11 * 16
    assert (
        selection.table.loc[
            selection.table["candidate"] == "interval_specific",
            "validation_nll",
        ].iloc[0]
        < selection.table.loc[
            selection.table["candidate"] == "shared", "validation_nll"
        ].iloc[0]
    )


def test_exact_operational_loss_is_multiplicative_not_additive():
    q, rate, hold = 0.8, 0.5, 0.4
    exact = float(exact_multiplicative_loss(q, rate, hold))
    assert exact == pytest.approx(1.0 - q * np.exp(-rate * hold))
    additive_bug = (1.0 - q) + rate * hold
    assert exact != pytest.approx(additive_bug)
    assert exact < additive_bug
    assert float(constant_rate_loss(rate, hold)) == pytest.approx(
        1.0 - np.exp(-rate * hold)
    )


def test_bright_no_floor_rate_recovery():
    data = _bright_calls()
    fit = fit_bright_decay(data[data["split"] == "train"], model="no_floor")
    assert fit.pi_0 == pytest.approx(0.84, abs=0.025)
    assert fit.lambda_bright_effective == pytest.approx(0.42, abs=0.025)
    assert fit.tau_bright_effective == pytest.approx(1.0 / 0.42, rel=0.07)
    assert fit.pi_floor == 0.0


def test_floor_model_is_selected_only_when_validation_supports_it():
    data = _bright_calls(
        seed=31,
        n_repetitions=100,
        n_sites=25,
        pi0=0.90,
        rate=2.0,
        floor=0.35,
    )
    selection = compare_bright_models(
        data[data["split"] == "train"],
        data[data["split"] == "validation"],
    )
    assert selection.selected_name == "floor"
    fit = selection.selected_fit
    assert fit.pi_floor == pytest.approx(0.35, abs=0.04)
    assert fit.lambda_bright_effective == pytest.approx(2.0, abs=0.20)
    assert bool(selection.table["floor_complexity_gate_passed"].iloc[0])


def test_nominal_floor_gain_that_is_unresolved_across_shots_selects_no_floor():
    data = _bright_calls(
        seed=31,
        n_repetitions=100,
        n_sites=25,
        pi0=0.86,
        rate=1.1,
        floor=0.24,
    )
    selection = compare_bright_models(
        data[data["split"] == "train"],
        data[data["split"] == "validation"],
    )
    assert selection.table.iloc[0]["candidate"] == "floor"
    assert selection.selected_name == "no_floor"
    assert not bool(selection.table["floor_complexity_gate_passed"].iloc[0])


def test_flat_and_monotone_control_are_compared_on_validation():
    data = _bright_calls(
        seed=41,
        n_repetitions=80,
        n_sites=20,
        pi0=0.91,
        rate=0.18,
        response_col="retained_next",
    )
    train = data[data["split"] == "train"]
    validation = data[data["split"] == "validation"]
    selection = compare_control_models(train, validation)
    assert selection.selected_name == "monotone"
    assert bool(
        selection.table[
            "monotone_cluster_gate__complexity_gate_passed"
        ].iloc[0]
    )
    fit = fit_control_retention(train, model="monotone")
    assert fit.q_0 == pytest.approx(0.91, abs=0.03)
    assert fit.kappa == pytest.approx(0.18, abs=0.03)


def test_retention_event_builder_conditions_on_previous_apparent_occupancy():
    scored = pd.DataFrame(
        {
            "run_id": ["R"] * 6,
            "shot_id": [0] * 6,
            "condition_id": [0] * 6,
            "site_id": [0, 0, 0, 1, 1, 1],
            "frame_index": [0, 1, 2, 0, 1, 2],
            "sweep_value_s": [0.3] * 6,
            "apparent_occupied": [True, True, False, False, True, True],
        }
    )
    events = make_retention_events(scored)
    # site 0 contributes 0->1 and 1->2; site 1 contributes only 1->2
    assert list(events["interval"]) == ["0->1", "1->2", "1->2"]
    assert list(events["retained_next"]) == [1, 0, 1]


def test_sweep_fits_reject_nontraining_rows():
    dark = _dark_events(n_repetitions=10, n_sites=3)
    bright = _bright_calls(n_repetitions=10, n_sites=3)
    with pytest.raises(ValueError, match="training rows only"):
        fit_dark_retention(dark)
    with pytest.raises(ValueError, match="training rows only"):
        fit_bright_decay(bright)
    control = bright.rename(columns={"apparent_occupied": "retained_next"})
    with pytest.raises(ValueError, match="training rows only"):
        fit_control_retention(control)


def test_dark_cluster_bootstrap_returns_rate_and_interval_parameters():
    data = _dark_events(n_repetitions=12, n_sites=5)
    train = data[data["split"] == "train"].copy()
    result = bootstrap_dark_retention(
        train, model="shared", n_boot=12, seed=73
    )
    assert result.n_successful == 12
    assert "lambda_switch_off__shared" in result.draws
    assert {
        "fixed_interreadout_survival__0->1",
        "fixed_interreadout_survival__1->2",
        "fixed_interreadout_survival__2->3",
        "fixed_interreadout_survival__3->4",
    }.issubset(result.draws.columns)


def test_rate_ratio_bootstrap_resamples_the_runs_independently():
    dark = _dark_events(n_repetitions=12, n_sites=5)
    dark = dark[dark["split"] == "train"].copy()
    bright = _bright_calls(n_repetitions=12, n_sites=5)
    bright = bright[bright["split"] == "train"].copy()
    result = bootstrap_rate_ratio(
        dark,
        bright,
        n_boot=8,
        seed=29,
        dark_fit_kwargs={"model": "shared"},
        bright_fit_kwargs={"model": "no_floor"},
    )
    assert len(result.draws) == 8
    assert np.isfinite(result.point_ratio)
    interval = result.interval()
    assert interval["lower"] <= interval["upper"]


def test_sensitivity_tables_record_declared_variants():
    dark = _dark_events(n_repetitions=20, n_sites=5)
    dark = dark[dark["split"] == "train"]
    dark_variants = {
        "all": dark,
        "exclude_shortest": dark[
            dark["sweep_value_s"] > dark["sweep_value_s"].min()
        ],
    }
    dark_table = dark_sensitivity_table(dark_variants, model="shared")
    assert set(dark_table["sensitivity_variant"]) == set(dark_variants)
    assert "lambda_switch_off__shared" in dark_table

    bright = _bright_calls(n_repetitions=20, n_sites=5)
    bright = bright[bright["split"] == "train"]
    bright_variants = {
        "all": bright,
        "exclude_longest": bright[
            bright["sweep_value_s"] < bright["sweep_value_s"].max()
        ],
    }
    bright_table = bright_sensitivity_table(bright_variants, model="no_floor")
    assert set(bright_table["sensitivity_variant"]) == set(bright_variants)
    assert "lambda_bright_effective" in bright_table
