"""Focused integration invariants for assembled loss-sweep claims."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fluorescence_inference.loss_sweep_analysis import compare_datasets


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
    bright_fit = SimpleNamespace(lambda_bright_effective=0.60)
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
