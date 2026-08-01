"""Determinism and public gates for optimized README assets."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_optimized_assets as assets  # noqa: E402


def _curve(time_key: str):
    return [
        {time_key: float(x), "prediction": 0.52 * np.exp(-0.04 * x), "lower": 0.49 * np.exp(-0.04 * x), "upper": 0.55 * np.exp(-0.04 * x)}
        for x in np.linspace(0, 2, 8)
    ]


def _result() -> dict:
    emission = {}
    for label, dprime in (("50ms", 4.8), ("100ms", 5.1), ("200ms", 5.4)):
        x = np.linspace(-5, 8, 60)
        emission[label] = {
            "selected_model": "shrinkage_site_offsets_k5",
            "test_metrics_once_per_candidate": [
                {
                    "candidate": "shrinkage_site_offsets_k5",
                    "mean_count_nll": 8.0,
                    "mean_separation_d_prime": dprime,
                    "mean_model_implied_overlap": 0.005,
                    "mean_posterior_entropy": 0.014,
                }
            ],
            "public_heldout_distribution": {
                "bin_centers": x.tolist(),
                "heldout_density": (0.5 * np.exp(-0.5 * (x + 2) ** 2) + 0.5 * np.exp(-0.5 * (x - 5) ** 2)).tolist(),
                "model_empty_density": (0.5 * np.exp(-0.5 * (x + 2) ** 2)).tolist(),
                "model_occupied_density": (0.5 * np.exp(-0.5 * (x - 5) ** 2)).tolist(),
            },
        }
    dark_curve = [
        {"interval": "1->2", "sweep_value_s": float(x), "estimate": 0.99 * np.exp(-x / 36), "cluster_lower": 0.98 * np.exp(-x / 36), "cluster_upper": 1.0 * np.exp(-x / 36)}
        for x in np.linspace(0.1, 2.1, 6)
    ]
    dark_band = [{**row, "interval": "1->2"} for row in _curve("hold_s")]
    bright_curve = [
        {"sweep_value_s": float(x), "estimate": 0.52 * np.exp(-x / 20), "cluster_lower": 0.49 * np.exp(-x / 20), "cluster_upper": 0.55 * np.exp(-x / 20)}
        for x in np.linspace(0.1, 8.1, 8)
    ]
    matched = []
    for total, designs in ((0.1, ((0.05, 2), (0.1, 1))), (0.2, ((0.05, 4), (0.1, 2), (0.2, 1))), (0.4, ((0.1, 4), (0.2, 2)) )):
        matched.append({
            "total_bright_s": total,
            "designs": [
                {
                    "exposure_s": exposure,
                    "pulse_count": pulses,
                    "difference_vs_fewest_pulses": -0.004 * (len(designs) - index - 1),
                    "difference_cluster_lower": -0.012 * (len(designs) - index - 1),
                    "difference_cluster_upper": 0.004 * (len(designs) - index - 1),
                }
                for index, (exposure, pulses) in enumerate(designs)
            ],
        })
    budgets = {}
    for label, bright in (("50ms", 0.012), ("100ms", 0.024), ("200ms", 0.048)):
        budgets[label] = {
            "component_loss": {"bright_exposure": bright, "dark_gap": 0.0011, "residual_readout_associated": 0.0},
            "final_survival": 1.0 - bright - 0.0011,
        }
    return {
        "bootstrap_replicates": 1000,
        "provenance": {"publishable_clean_provenance": True, "analysis_code_commit": "a" * 40},
        "prerequisite_evidence": {"timing_and_compiled_commands_verified": True, "geometry_validated": {"all": True}, "background_frozen": {"all": True}},
        "optimized_dark_bright": {
            "dark_hold": {"operational_model": {"tau_switch_off": {"estimate": 35.7, "lower": 32.3, "upper": 40.1}, "model_band": dark_band, "clustered_apparent_retention_curve": dark_curve, "bootstrap": {"intervals": [{"parameter": "tau_switch_off__shared", "estimate": 35.7, "lower": 32.3, "upper": 40.1}]}}},
            "bright_wait": {"effective_model": {"tau_bright_effective": {"estimate": 20.3, "lower": 18.0, "upper": 23.1}, "model_band": _curve("wait_s"), "clustered_apparent_occupancy_curve": bright_curve, "bootstrap": {"intervals": [{"parameter": "tau_bright_effective", "estimate": 20.3, "lower": 18.0, "upper": 23.1}]}}},
        },
        "repeated_imaging": {"emission_models_by_exposure": emission, "heldout_matched_prefix_contrasts": matched, "selected_model": "continuous_only", "allowed_pulse_claim": False},
        "loss_budget": {"point_estimates": budgets},
    }


def _hashes(directory: Path):
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in assets.ASSET_NAMES}


def _relative_luminance(rgb: tuple[float, float, float]) -> float:
    channels = [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in rgb
    ]
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _hex_rgb(value: str) -> tuple[float, float, float]:
    return tuple(int(value[index:index + 2], 16) / 255 for index in (1, 3, 5))


def _contrast_ratio(foreground: str, background: str, alpha: float) -> float:
    fg = _hex_rgb(foreground)
    bg = _hex_rgb(background)
    composite = tuple(alpha * front + (1 - alpha) * back for front, back in zip(fg, bg))
    light = _relative_luminance(bg)
    dark = _relative_luminance(composite)
    return (light + 0.05) / (dark + 0.05)


def test_histogram_fill_has_visible_panel_contrast():
    assert _contrast_ratio(assets.HIST_FILL, "#ffffff", assets.HIST_ALPHA) >= 3.0
    assert assets.HIST_EDGE.lower() != "#ffffff"


def _story() -> assets.OccupancyStoryData:
    y, x = np.mgrid[:120, :140]
    raw = 100 + 20 * np.sin(x / 13) + 15 * np.cos(y / 11)
    sites = pd.DataFrame({
        "site_id": [0, 1, 2, 3],
        "site_x": [50.0, 70.0, 50.0, 70.0],
        "site_y": [45.0, 45.0, 65.0, 65.0],
        "site_row": [0, 0, 1, 1],
        "site_col": [0, 1, 0, 1],
    })
    rows = pd.DataFrame({
        "frame_id": range(5),
        "site_x": [50.0] * 5,
        "site_y": [45.0] * 5,
        "roi_sum_raw": [3200, 3300, 3250, 3350, 3280],
        "background_fixed_offset": [1800, 1810, 1790, 1805, 1795],
        "background_corrected_count": [1400, 1490, 1460, 1545, 1485],
    })
    return assets.OccupancyStoryData(
        raw_image=raw,
        sites=sites,
        representative_rows=rows,
        heldout_raw_counts=np.linspace(1800, 6500, 500),
        heldout_corrected_counts=np.r_[np.linspace(-100, 900, 250), np.linspace(2200, 5200, 250)],
        posterior_by_frame=np.asarray([0.53, 0.525, 0.52, 0.515, 0.51]),
        selected_shot_id=85,
        selected_site_id=0,
        shot_rule="held-out median shot rule",
        site_rule="within-shot median site rule",
    )


def test_assets_are_deterministic_and_gated(tmp_path):
    result = tmp_path / "result.json"
    result.write_text(json.dumps(_result()), encoding="utf-8")
    first = tmp_path / "first"
    second = tmp_path / "second"
    assets.generate_assets(
        result, first, metrics_path=tmp_path / "metrics.md", story_data=_story()
    )
    assets.generate_assets(result, second, story_data=_story())
    assert _hashes(first) == _hashes(second)
    with Image.open(first / "optimized_occupancy_inference.gif") as animation:
        assert animation.size == (1000, 600)
        assert animation.n_frames >= 40
        assert sum(
            animation.seek(index) or animation.info["duration"]
            for index in range(animation.n_frames)
        ) / 1000 == pytest.approx(11.5)
        target = np.asarray([int(assets.HIST_FILL[index:index + 2], 16) for index in (1, 3, 5)])
        for scene_index in (3, 5):
            animation.seek(scene_index * (assets.STORY_TRANSITION_FRAMES + 1))
            pixels = np.asarray(animation.convert("RGB"), dtype=int)
            distance = np.max(np.abs(pixels - target), axis=2)
            assert np.count_nonzero(distance < 48) > 2_000
    metadata = json.loads((first / "optimized_asset_metadata.json").read_text())
    assert metadata["representative_selection"]["uses_frozen_geometry_background_and_emissions"] is True
    assert metadata["animation"]["bytes"] < 8_000_000
    metrics = (tmp_path / "metrics.md").read_text(encoding="utf-8")
    assert "model-implied" in metrics
    assert "no externally labelled occupancy truth" in metrics

    payload = _result()
    payload["bootstrap_replicates"] = 100
    result.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(assets.AssetInputError, match="1000-bootstrap"):
        assets.generate_assets(result, tmp_path / "rejected", story_data=_story())
