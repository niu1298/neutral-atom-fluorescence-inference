"""Held-out Gaussian count baselines and non-leakage checks."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fluorescence_inference.heldout_emissions import (
    GaussianComponents,
    compare_emission_models,
    evaluate_heldout,
    fit_emission_baseline,
    fit_required_baselines,
)


def _synthetic_counts(seed: int = 902, n_shots: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    frame_offsets = {0: 0.0, 1: 2.1, 2: -1.4}
    site_offsets = rng.normal(0.0, 0.35, size=12)
    for shot in range(n_shots):
        condition = shot % 5
        split = "train" if shot < 60 else "validation" if shot < 75 else "test"
        common = rng.normal(0.0, 0.18)
        previous = rng.random(12) < 0.58
        for frame in range(3):
            if frame:
                previous = previous & (rng.random(12) < 0.91)
            for site in range(12):
                occupied = bool(previous[site])
                corrected = (
                    6.0 * occupied
                    + frame_offsets[frame]
                    + site_offsets[site]
                    + common
                    + rng.normal(0.0, 0.75 if occupied else 0.65)
                )
                rows.append(
                    {
                        "run_id": "synthetic",
                        "shot_id": shot,
                        "condition_id": condition,
                        "frame_index": frame,
                        "site_id": site,
                        "split": split,
                        "roi_sum_raw": corrected + 100.0 + 4.0 * frame,
                        "count_corrected_template": corrected,
                        "synthetic_state": occupied,
                    }
                )
    return pd.DataFrame(rows)


def test_gaussian_component_overlap_has_known_equal_variance_value():
    component = GaussianComponents(
        weight_empty=0.5,
        mean_empty=0.0,
        sigma_empty=1.0,
        weight_occupied=0.5,
        mean_occupied=4.0,
        sigma_occupied=1.0,
        log_likelihood=0.0,
        n_fit=100,
        converged=True,
    )
    assert component.separation_d_prime == pytest.approx(4.0)
    assert component.model_implied_overlap == pytest.approx(0.0227501, rel=2e-5)
    assert component.posterior_half_threshold == pytest.approx(2.0)


def test_required_baselines_fit_training_rows_only_and_score_test_counts():
    data = _synthetic_counts()
    train = data[data["split"] == "train"].copy()
    test = data[data["split"] == "test"].copy()
    models = fit_required_baselines(train)
    assert set(models) == {
        "raw_global_threshold",
        "corrected_global_threshold",
        "frame_pooled_mixture",
        "shared_frame_offsets",
        "shrinkage_site_offsets",
        "prototype_per_site_diagnostic",
    }
    assert all(model.n_train_rows == len(train) for model in models.values())
    for model in models.values():
        result = evaluate_heldout(model, test)
        assert result.n_rows == len(test)
        assert result.n_independent_shots == 15
        assert np.isfinite(result.mean_count_nll)
        assert 0.0 <= result.mean_model_implied_overlap <= 0.5
        assert result.mean_separation_d_prime > 2.0
        assert set(result.apparent_transitions["interval"]) == {"0->1", "1->2"}


def test_shared_frame_offsets_recover_injected_pedestal_without_test_data():
    data = _synthetic_counts()
    train = data[data["split"] == "train"].copy()
    model = fit_emission_baseline(
        train,
        value_col="count_corrected_template",
        kind="shared_frame_offsets",
    )
    recovered = model.frame_offsets
    assert recovered[1] - recovered[0] == pytest.approx(2.1, abs=0.25)
    assert recovered[2] - recovered[0] == pytest.approx(-1.4, abs=0.25)
    assert model.components.mean_occupied - model.components.mean_empty == pytest.approx(
        6.0, abs=0.25
    )

    changed_test = data.copy()
    changed_test.loc[changed_test["split"] == "test", "count_corrected_template"] += 1e6
    again = fit_emission_baseline(
        changed_test[changed_test["split"] == "train"],
        value_col="count_corrected_template",
        kind="shared_frame_offsets",
    )
    assert again.components.to_dict() == model.components.to_dict()
    assert again.frame_offsets == model.frame_offsets


def test_fit_rejects_validation_or_test_rows_in_a_split_aware_table():
    data = _synthetic_counts()
    with pytest.raises(ValueError, match="training rows only"):
        fit_emission_baseline(
            data, value_col="count_corrected_template", kind="global"
        )


def test_heldout_evaluation_rejects_shot_overlap():
    data = _synthetic_counts()
    train = data[data["split"] == "train"]
    model = fit_emission_baseline(
        train, value_col="count_corrected_template", kind="global"
    )
    with pytest.raises(ValueError, match="overlap"):
        evaluate_heldout(model, train.iloc[:100])


def test_validation_model_comparison_uses_predictive_nll():
    data = _synthetic_counts()
    train = data[data["split"] == "train"]
    validation = data[data["split"] == "validation"]
    models = {
        kind: fit_emission_baseline(
            train,
            value_col="count_corrected_template",
            kind=kind,
            name=kind,
        )
        for kind in ("global", "frame_pooled", "shared_frame_offsets")
    }
    comparison = compare_emission_models(models, validation)
    assert comparison["selected_on_validation"].sum() == 1
    assert comparison.loc[0, "candidate"] in {
        "frame_pooled",
        "shared_frame_offsets",
    }
    assert comparison.loc[0, "delta_validation_nll_per_row"] == 0.0


def test_site_shrinkage_is_finite_and_unknown_site_uses_zero_offset():
    data = _synthetic_counts()
    train = data[data["split"] == "train"]
    model = fit_emission_baseline(
        train,
        value_col="count_corrected_template",
        kind="site_shrinkage",
        shrinkage_strength=15.0,
    )
    assert set(model.site_offsets) == set(range(12))
    novel = data[data["split"] == "test"].iloc[[0]].copy()
    novel["site_id"] = 999
    scored = model.score(novel)
    assert np.isfinite(scored["log_predictive_density"]).all()
