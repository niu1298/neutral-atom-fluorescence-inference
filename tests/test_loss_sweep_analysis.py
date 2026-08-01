"""Focused integration invariants for assembled loss-sweep claims."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fluorescence_inference.loss_sweep_analysis import (
    _bright_model_band,
    _invert_positive_rate_interval,
    _selected_kappa_semantics,
    _training_site_coordinates,
    compare_datasets,
    dark_endpoint_sensitivity,
)


def test_bright_model_band_covers_the_configured_wait_range():
    fit = SimpleNamespace(
        pi_0=0.5,
        pi_floor=0.0,
        predict_time=lambda time: 0.5 * np.exp(-0.05 * np.asarray(time)),
    )
    bootstrap = SimpleNamespace(
        draws=pd.DataFrame(
            {
                "pi_0": [0.48, 0.50, 0.52],
                "pi_floor": [0.0, 0.0, 0.0],
                "lambda_bright_effective": [0.04, 0.05, 0.06],
            }
        )
    )
    band = _bright_model_band(fit, bootstrap, max_time_s=8.1)
    assert band["wait_s"].iloc[-1] == pytest.approx(8.1)


def test_cross_dataset_estimates_use_point_fits_and_never_assert_explanation():
    dark_fit = SimpleNamespace(
        model="shared",
        interval_levels=("0->1", "1->2", "2->3", "3->4"),
        lambda_groups={"shared": 0.05},
        q_by_interval={
            "0->1": 0.80,
            "1->2": 0.86,
            "2->3": 0.88,
            "3->4": 0.90,
        },
    )
    dark_draws = pd.DataFrame(
        {
            "lambda_switch_off__shared": [0.04, 0.05, 0.08],
            "fixed_interreadout_survival__0->1": [0.79, 0.80, 0.81],
            "fixed_interreadout_survival__1->2": [0.84, 0.86, 0.88],
            "fixed_interreadout_survival__2->3": [0.86, 0.88, 0.90],
            "fixed_interreadout_survival__3->4": [0.88, 0.90, 0.92],
        }
    )
    bright_fit = SimpleNamespace(
        model="no_floor", lambda_bright_effective=0.60
    )
    bright_draws = pd.DataFrame(
        {"lambda_bright_effective": [0.50, 0.60, 0.72]}
    )
    report = compare_datasets(
        {
            "fit": dark_fit,
            "bootstrap": SimpleNamespace(draws=dark_draws),
        },
        {
            "fit": bright_fit,
            "bootstrap": SimpleNamespace(draws=bright_draws),
        },
    )

    assert report["lambda_bright_over_lambda_switch_off"][
        "estimate"
    ] == pytest.approx(12.0)
    predicted = 1.0 - np.exp(-0.60 * 0.05)
    assert report["bright_model_predicted_loss_over_50ms"][
        "estimate"
    ] == pytest.approx(predicted)
    assert report["apparent_later_interval_fixed_loss"][
        "estimate"
    ] == pytest.approx(0.12)
    assert report["observed_minus_predicted_loss"][
        "estimate"
    ] == pytest.approx(0.12 - predicted)
    assert report["simple_bright_model_explains_full_interreadout_loss"] is False
    assert "apparent inter-readout loss" in report["allowed_conclusion"]


def test_structural_floor_sensitivity_is_not_a_sampling_interval():
    dark_fit = SimpleNamespace(
        model="shared",
        first_interval="0->1",
        interval_levels=("0->1", "1->2", "2->3", "3->4"),
        lambda_groups={"shared": 0.05},
        q_by_interval={
            "0->1": 0.80,
            "1->2": 0.86,
            "2->3": 0.88,
            "3->4": 0.90,
        },
    )
    dark_draws = pd.DataFrame(
        {
            "lambda_switch_off__shared": [0.04, 0.05, 0.08],
            "fixed_interreadout_survival__0->1": [0.79, 0.80, 0.81],
            "fixed_interreadout_survival__1->2": [0.84, 0.86, 0.88],
            "fixed_interreadout_survival__2->3": [0.86, 0.88, 0.90],
            "fixed_interreadout_survival__3->4": [0.88, 0.90, 0.92],
        }
    )
    bright_fit = SimpleNamespace(
        model="no_floor", lambda_bright_effective=0.60
    )
    floor_fit = SimpleNamespace(
        pi_floor=0.12,
        lambda_bright_effective=1.20,
    )
    report = compare_datasets(
        {
            "fit": dark_fit,
            "bootstrap": SimpleNamespace(draws=dark_draws),
        },
        {
            "fit": bright_fit,
            "floor_sensitivity_fit": floor_fit,
            "bootstrap": SimpleNamespace(
                draws=pd.DataFrame(
                    {"lambda_bright_effective": [0.50, 0.60, 0.72]}
                )
            ),
        },
    )
    sensitivity = report["structural_sensitivity"]
    assert sensitivity["selected_structure"] == "no_floor"
    assert sensitivity["alternative_structure"] == "floor"
    assert sensitivity["alternative_gap_percentage_points"] > 0.0
    assert sensitivity["structural_range_percentage_points"] == sorted(
        [
            sensitivity["selected_gap_percentage_points"],
            sensitivity["alternative_gap_percentage_points"],
        ]
    )
    assert "not a confidence interval" in sensitivity["interpretation"]
    assert "test data were not used" in sensitivity["alternative_fit_scope"]


def test_rate_interval_to_lifetime_reverses_bounds():
    lifetime = _invert_positive_rate_interval(
        {"estimate": 0.5, "lower": 0.25, "upper": 1.0}
    )
    assert lifetime == {
        "estimate": 2.0,
        "lower": 1.0,
        "upper": 4.0,
    }
    assert _invert_positive_rate_interval(
        {"estimate": 0.5, "lower": 0.0, "upper": 1.0}
    ) is None


def test_flat_kappa_is_structural_not_a_measured_zero_interval():
    semantics = _selected_kappa_semantics(
        SimpleNamespace(model="flat", kappa=0.0),
        {
            "estimate": 0.0,
            "lower": 0.0,
            "upper": 0.0,
            "confidence": 0.95,
        },
        trend_resolved=False,
    )
    assert semantics == {
        "selected_structure": "flat",
        "kappa_status": "structurally_fixed",
        "kappa_fixed_value": 0.0,
        "trend_resolved": False,
    }
    assert "kappa_sampling_interval" not in semantics


def test_endpoint_sensitivity_has_all_predeclared_deterministic_variants():
    rng = np.random.default_rng(37)
    rows = []
    times = (0.1, 0.3, 0.5, 0.7, 0.9)
    for condition, time in enumerate(times):
        for shot in range(12):
            for interval, q in zip(
                ("0->1", "1->2", "2->3", "3->4"),
                (0.80, 0.87, 0.90, 0.88),
            ):
                for site in range(4):
                    rows.append(
                        {
                            "run_id": "dark",
                            "shot_id": condition * 12 + shot,
                            "condition_id": condition,
                            "site_id": site,
                            "sweep_value_s": time,
                            "interval": interval,
                            "retained_next": int(
                                rng.random() < q * np.exp(-0.12 * time)
                            ),
                        }
                    )
    events = pd.DataFrame(rows)
    first = dark_endpoint_sensitivity(events, selected_model="shared")
    second = dark_endpoint_sensitivity(events, selected_model="shared")
    assert [row["variant"] for row in first] == [
        "all_points",
        "exclude_shortest",
        "exclude_longest",
        "exclude_both",
    ]
    assert first == second
    assert all(row["fit_success"] for row in first)
    assert first[1]["included_conditions_s"] == list(times[1:])
    assert first[2]["included_conditions_s"] == list(times[:-1])
    assert first[3]["included_conditions_s"] == list(times[1:-1])


def test_public_site_coordinates_are_sourced_from_train_and_checked_across_splits():
    rows = []
    for split in ("train", "validation", "test"):
        for site_id in (0, 1):
            rows.append(
                {
                    "site_id": site_id,
                    "split": split,
                    "site_row": 0,
                    "site_col": site_id,
                    "site_y": 10.0,
                    "site_x": 20.0 + site_id,
                }
            )
    scored = pd.DataFrame(rows)
    coordinates = _training_site_coordinates(scored)
    assert len(coordinates) == 2
    assert coordinates["site_id"].tolist() == [0, 1]

    inconsistent = scored.copy()
    inconsistent.loc[
        (inconsistent["split"] == "test")
        & (inconsistent["site_id"] == 1),
        "site_x",
    ] += 0.5
    with pytest.raises(ValueError, match="differ across"):
        _training_site_coordinates(inconsistent)
