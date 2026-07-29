"""Determinism, content, and gating of public loss-sweep figures."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_loss_sweep_assets as assets  # noqa: E402

from fluorescence_inference.reporting import assert_public_safe  # noqa: E402


def _curve(times, baseline, slope, n_shots):
    rows = []
    for time in times:
        estimate = baseline * np.exp(-slope * time)
        rows.append(
            {
                "condition_id": f"c{time}",
                "sweep_value_s": time,
                "estimate": estimate,
                "cluster_lower": max(0.0, estimate - 0.035),
                "cluster_upper": min(1.0, estimate + 0.035),
                "n_independent_shots": n_shots,
            }
        )
    return rows


def _band(times, baseline, slope, time_key):
    return [
        {
            time_key: time,
            "prediction": baseline * np.exp(-slope * time),
            "lower": max(0.0, baseline * np.exp(-slope * time) - 0.025),
            "upper": min(1.0, baseline * np.exp(-slope * time) + 0.025),
        }
        for time in times
    ]


def _result_fixture() -> dict:
    dark_times = list(np.round(np.arange(0.1, 2.11, 0.2), 10))
    bright_times = list(np.round(np.arange(0.1, 1.91, 0.2), 10))
    band_times = list(np.linspace(0.0, 2.2, 24))
    intervals = ("0->1", "1->2", "2->3", "3->4")
    q_values = (0.80, 0.87, 0.90, 0.88)
    dark_curve = []
    dark_band = []
    fixed = []
    for interval, q in zip(intervals, q_values):
        for row in _curve(dark_times, q, 0.055, 10):
            dark_curve.append({**row, "interval": interval})
        for row in _band(band_times, q, 0.055, "hold_s"):
            dark_band.append({**row, "interval": interval})
        fixed.append(
            {
                "interval": interval,
                "estimate": q,
                "lower": q - 0.025,
                "upper": q + 0.025,
            }
        )

    bright_curve = _curve(bright_times, 0.82, 0.39, 10)
    bright_band = _band(list(np.linspace(0.0, 2.0, 30)), 0.82, 0.39, "wait_s")
    control_curve = _curve(bright_times, 0.90, 0.07, 10)
    control_band = _band(
        list(np.linspace(0.0, 2.0, 30)), 0.90, 0.07, "wait_s"
    )
    per_site = []
    for row in range(4):
        for col in range(4):
            value = 0.76 + 0.015 * row + 0.01 * col
            per_site.append(
                {
                    "site_id": row * 4 + col,
                    "site_row": row,
                    "site_col": col,
                    "estimate": value,
                    "cluster_lower": value - 0.06,
                    "cluster_upper": value + 0.06,
                    "n_independent_shots": 22,
                }
            )

    def coupling(n_frames, step):
        return {
            str(frame): {
                "background_global": {
                    "background_vs_apparent_occupancy_r": 0.03 * frame,
                    "background_change_shortest_to_longest": step * (frame + 1),
                },
                "background_template_offset": {
                    "background_vs_apparent_occupancy_r": 0.02 * frame,
                    "background_change_shortest_to_longest": 0.5
                    * step
                    * (frame + 1),
                },
            }
            for frame in range(n_frames)
        }

    def pedestal(n_frames, n_shots, base):
        frames = {
            str(frame): {
                "median_roi_counts": base - 30 * frame,
                "mean_roi_counts": base - 30 * frame,
                "sd_across_shots": 12.0,
                "n_shots": n_shots,
            }
            for frame in range(n_frames)
        }
        return {
            "available": True,
            "frames": frames,
            "median_spread_max_minus_min_roi_counts": 30 * (n_frames - 1),
            "first_minus_later_median_roi_counts": 45.0,
        }

    def background_curve(times, n_frames, base):
        rows = []
        for frame in range(n_frames):
            for condition, time in enumerate(times):
                estimate = base - 18.0 * frame - 7.0 * time
                rows.append(
                    {
                        "condition_id": condition,
                        "sweep_value_s": time,
                        "frame_index": frame,
                        "estimate": estimate,
                        "cluster_lower": estimate - 4.0,
                        "cluster_upper": estimate + 4.0,
                        "n_independent_shots": 10,
                    }
                )
        return {
            "available": True,
            "background_column": "background_template_offset",
            "units": "ROI counts",
            "scope": "selected site-free background by condition and frame",
            "curve": rows,
        }

    def condition_split_counts(times):
        return [
            {
                "condition_id": f"condition:{time:g}",
                "sweep_value_s": time,
                "split": split,
                "n_independent_shots": count,
            }
            for time in times
            for split, count in (
                ("train", 6),
                ("validation", 2),
                ("test", 2),
            )
        ]

    def geometry_evidence(pitch, early_median, early_max, endpoint):
        return {
            "subset_fits": {
                "train": {
                    "grids": [
                        {
                            "row_pitch_px": pitch,
                            "col_pitch_px": pitch,
                        }
                    ]
                }
            },
            "comparisons": {
                "early_vs_late": {
                    "gate_source": "registered",
                    "registered": {
                        "matching": {
                            "median_px": early_median,
                            "max_px": early_max,
                        }
                    },
                },
                "shortest_vs_longest": {
                    "gate_source": "rigid_train_shape_registration",
                    "rigid_train_shape_registration": {
                        "matching": {
                            "median_px": endpoint,
                            "max_px": endpoint,
                        }
                    },
                },
            },
            "gate": {"passed": True},
        }

    def selected_metrics(n_shots, n_rows, nll, overlap, d_prime, entropy):
        return {
            "candidate": "shrinkage_site_offsets_k5",
            "model": "shrinkage_site_offsets_k5",
            "n_independent_shots": n_shots,
            "n_rows": n_rows,
            "mean_count_nll": nll,
            "mean_model_implied_overlap": overlap,
            "mean_separation_d_prime": d_prime,
            "mean_posterior_entropy": entropy,
        }

    return {
        "analysis_version": "loss-sweeps-test",
        "independent_experimental_unit": "shot",
        "acquisition_order_caveat": (
            "Conditions repeat in fixed ascending order within every cycle. "
            "Sweep value is confounded with within-cycle position."
        ),
        "dark_hold": {
            "emission_baselines": {
                "selected_model": "shrinkage_site_offsets_k5",
                "split_counts": {"train": 66, "validation": 22, "test": 22},
                "test_metrics_once_per_candidate": [
                    selected_metrics(22, 11000, 7.65, 0.045, 3.25, 0.088)
                ],
            },
            "operational_model": {
                "public_rate_claim_gate_passed": True,
                "rate_resolved": True,
                "selected_model": "shared",
                "lambda_switch_off": {
                    "estimate": 0.055,
                    "lower": 0.040,
                    "upper": 0.072,
                },
                "tau_switch_off": {
                    "estimate": 18.18,
                    "lower": 13.89,
                    "upper": 25.00,
                },
                "clustered_apparent_retention_curve": dark_curve,
                "fixed_interreadout_survival": fixed,
                "model_band": dark_band,
                "heldout_per_site_apparent_retention": per_site,
                "site_free_background_curve": background_curve(
                    dark_times, 5, 9400.0
                ),
                "site_free_background_occupancy_coupling": coupling(5, -8.0),
            }
        },
        "bright_wait": {
            "emission_baselines": {
                "selected_model": "shrinkage_site_offsets_k5",
                "split_counts": {"train": 60, "validation": 20, "test": 20},
                "test_metrics_once_per_candidate": [
                    selected_metrics(20, 4000, 7.91, 0.041, 3.30, 0.080)
                ],
            },
            "effective_model": {
                "public_rate_claim_gate_passed": True,
                "rate_resolved": True,
                "selected_model": "no_floor",
                "lambda_bright_effective": {
                    "estimate": 0.39,
                    "lower": 0.34,
                    "upper": 0.45,
                },
                "tau_bright_effective": {
                    "estimate": 2.56,
                    "lower": 2.22,
                    "upper": 2.94,
                },
                "clustered_apparent_occupancy_curve": bright_curve,
                "model_band": bright_band,
                "post_wait_control": {
                    "selected_model": "flat",
                    "trend_resolved": False,
                    "clustered_curve": control_curve,
                    "model_band": control_band,
                },
                "site_free_background_curve": background_curve(
                    bright_times, 2, 9300.0
                ),
                "site_free_background_occupancy_coupling": coupling(2, -12.0),
            }
        },
        "background_evidence": {
            "dark_hold": {
                "selection": {
                    "selected_method": "template",
                    "selected_residual_structure": 5.17,
                },
                "annulus_contamination": {
                    "n_sites": 100,
                    "sites_with_any_overlap": 100,
                },
                "site_free_frame_pedestal": pedestal(5, 110, 10450.0)
            },
            "bright_wait": {
                "selection": {
                    "selected_method": "template",
                    "selected_residual_structure": 3.96,
                },
                "annulus_contamination": {
                    "n_sites": 100,
                    "sites_with_any_overlap": 100,
                },
                "site_free_frame_pedestal": pedestal(2, 100, 9700.0)
            },
        },
        "dataset_audit": {
            "dark_hold": {
                "n_complete_shots": 110,
                "n_incomplete_shots": 0,
                "n_frames": 5,
                "commanded_exposure_s": [0.05],
                "sweep_values_s": dark_times,
                "condition_split_counts": condition_split_counts(dark_times),
            },
            "bright_wait": {
                "n_complete_shots": 100,
                "n_incomplete_shots": 0,
                "n_frames": 2,
                "commanded_exposure_s": [0.05],
                "sweep_values_s": bright_times,
                "condition_split_counts": condition_split_counts(bright_times),
            },
        },
        "geometry_evidence": {
            "dark_hold": geometry_evidence(11.205, 0.001, 0.001, 0.007),
            "bright_wait": geometry_evidence(11.233, 0.145, 0.258, 0.088),
            "cross_run": {
                "wide_radius_matching": {
                    "bulk_shift": {"magnitude_px": 11.09}
                },
                "interpretation": (
                    "Recorded crop and camera settings are identical. Image "
                    "data alone cannot distinguish a one-pitch translation "
                    "from a lattice-index alias."
                ),
            },
        },
        "cross_dataset_comparison": {
            "public_comparison_gate_passed": True,
            "lambda_bright_over_lambda_switch_off": {
                "estimate": 7.09,
                "lower": 4.90,
                "upper": 10.20,
            },
            "bright_model_predicted_loss_over_50ms": {
                "estimate": 0.0193,
                "lower": 0.0169,
                "upper": 0.0222,
            },
            "apparent_later_interval_fixed_loss": {
                "estimate": 0.125,
                "lower": 0.110,
                "upper": 0.142,
            },
            "observed_minus_predicted_loss": {
                "estimate": 0.106,
                "lower": 0.088,
                "upper": 0.124,
            },
            "allowed_conclusion": (
                "The simple constant-rate bright-wait model does not explain "
                "the full inter-readout loss."
            ),
            "pooling_reason": (
                "the acquisitions have a one-pitch coordinate shift and "
                "distinct background trajectories."
            ),
        },
        "latent_state": {
            "gate_passed": True,
            "uncertainty_scope": {
                "complete_shot_cluster_bootstrap_available": True,
                "public_gate_requires_complete_shot_clustering": True,
            },
            "selected_structure": "shared_rate",
            "held_out_comparison": {
                "delta_per_trajectory": 0.79,
                "n_trajectories": 2200,
            },
            "interpretation": (
                "Predictive latent-state parameters remain conditional on the "
                "stated model and are not empirical fidelity."
            ),
            "representative_example": {
                "selection_rule": (
                    "test trajectory whose mean posterior entropy is closest "
                    "to the test-set median"
                ),
                "hold_s": 0.9,
                "observed_corrected_counts": [120.0, 105.0, 92.0, 20.0, 12.0],
                "threshold_baseline_calls": [1, 1, 1, 0, 0],
                "posterior_occupied_probability": [0.99, 0.97, 0.90, 0.08, 0.02],
                "posterior_state_probability": [
                    [0.01, 0.99],
                    [0.03, 0.97],
                    [0.10, 0.90],
                    [0.92, 0.08],
                    [0.98, 0.02],
                ],
            },
        },
    }


def _v0_qc_fixture() -> dict:
    return {
        "dataset": {
            "n_shots": 101,
            "n_frames_per_shot": 3,
            "exposure_ms": 87.5,
        }
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_loss_sweep_assets_are_deterministic_and_have_real_content(scratch):
    result_path = scratch / "loss_sweep_results.json"
    result_path.write_text(
        json.dumps(_result_fixture(), sort_keys=True), encoding="utf-8"
    )
    out_a, out_b = scratch / "a", scratch / "b"
    first = assets.generate_assets(result_path, out_a)
    second = assets.generate_assets(result_path, out_b)

    expected = {
        *assets.CORE_ASSETS,
        "per_site_retention_map.png",
        "latent_state_example.png",
    }
    assert set(first["assets"]) == expected
    assert set(second["assets"]) == expected
    for filename in sorted(expected):
        assert _sha(out_a / filename) == _sha(out_b / filename)
        with Image.open(out_a / filename) as image:
            array = np.asarray(image.convert("RGB"))
            assert image.width >= 700
            assert image.height >= 450
            assert float(array.std()) > 20.0
            assert len(np.unique(array.reshape(-1, 3), axis=0)) > 100
    assert _sha(out_a / assets.METADATA_NAME) == _sha(
        out_b / assets.METADATA_NAME
    )
    assert first["source_file"] == "loss_sweep_results.json"
    assert first["independent_experimental_unit"] == "shot"
    assert not first["optional_assets_skipped"]
    overview_text = first["visible_text"]["loss_sweep_overview.png"]
    assert "88 independent development shots" in overview_text
    assert (
        "Points show 95% complete-shot cluster intervals."
        in first["visible_text"]["background_drift_sweeps.png"]
    )
    visible_blob = json.dumps(first["visible_text"]).lower()
    assert "fidelity" not in visible_blob
    assert "fixed per-exposure" not in visible_blob
    assert "pair loss" not in visible_blob
    assert_public_safe(
        (out_a / assets.METADATA_NAME).read_text(encoding="utf-8"),
        "test metadata",
    )


def test_optional_assets_obey_data_and_latent_gate(scratch):
    result = _result_fixture()
    del result["dark_hold"]["operational_model"][
        "heldout_per_site_apparent_retention"
    ]
    result["latent_state"]["uncertainty_scope"][
        "complete_shot_cluster_bootstrap_available"
    ] = False
    result["latent_state"]["gate_failures"] = [
        "complete-shot clustered uncertainty was not supplied"
    ]
    result_path = scratch / "loss_sweep_results.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    output = scratch / "output"
    output.mkdir(parents=True)
    (output / "per_site_retention_map.png").write_bytes(b"stale")
    (output / "latent_state_example.png").write_bytes(b"stale")
    metadata = assets.generate_assets(result_path, output)

    assert set(metadata["assets"]) == set(assets.CORE_ASSETS)
    assert set(metadata["optional_assets_skipped"]) == set(assets.OPTIONAL_ASSETS)
    assert not (output / "per_site_retention_map.png").exists()
    assert not (output / "latent_state_example.png").exists()


def test_background_endpoint_uses_the_selected_site_free_method():
    result = _result_fixture()
    rows = assets._background_endpoint_rows(result, "dark_hold")
    assert set(rows["method"]) == {"background_template_offset"}
    assert rows.sort_values("frame_index")["estimate"].tolist() == [
        -4.0,
        -8.0,
        -12.0,
        -16.0,
        -20.0,
    ]


def test_selected_background_curve_uses_cluster_bounds_and_template_column():
    result = _result_fixture()
    rows = assets._optional_background_curve(result)
    assert rows is not None
    assert set(rows["background_method"]) == {"template"}
    assert set(rows["background_column"]) == {"background_template_offset"}
    np.testing.assert_allclose(rows["upper"] - rows["lower"], 8.0)
    assert set(rows["n_independent_shots"]) == {10}


def test_generator_requires_reviewed_curve_records_and_reads_no_dataset(scratch):
    result_path = scratch / "incomplete.json"
    result_path.write_text(json.dumps({"analysis_version": "incomplete"}), encoding="utf-8")
    with pytest.raises(assets.AssetInputError, match="operational_model"):
        assets.generate_assets(result_path, scratch / "output")

    source = (ROOT / "scripts" / "generate_loss_sweep_assets.py").read_text(
        encoding="utf-8"
    )
    assert "load_dataset" not in source
    assert "discover_shots" not in source
    assert "raw_image_path" not in source
    assert "--results" in source
    assert "--output-dir" in source


def test_probability_bounds_and_duplicate_points_are_rejected(scratch):
    result = _result_fixture()
    curve = result["bright_wait"]["effective_model"][
        "clustered_apparent_occupancy_curve"
    ]
    curve[0]["cluster_lower"] = curve[0]["estimate"] + 0.1
    path = scratch / "bad_bounds.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(assets.AssetInputError, match="lower <= estimate"):
        assets.generate_assets(path, scratch / "bad_bounds")

    result = _result_fixture()
    curve = result["bright_wait"]["effective_model"][
        "clustered_apparent_occupancy_curve"
    ]
    curve.append(copy.deepcopy(curve[0]))
    path = scratch / "duplicate.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(assets.AssetInputError, match="duplicate points"):
        assets.generate_assets(path, scratch / "duplicate")


def _readme_stub(newline="\n"):
    parts = ["V0 prefix", ""]
    for marker in assets.README_MARKERS:
        parts.extend(
            [
                f"<!-- BEGIN:{marker} -->",
                f"old {marker}",
                f"<!-- END:{marker} -->",
                "",
            ]
        )
    parts.append("V0 suffix")
    return newline.join(parts) + newline


def test_publication_fragments_are_deterministic_source_backed_and_private():
    result = _result_fixture()
    v0_qc = _v0_qc_fixture()
    first = assets.build_publication_fragments(result, v0_qc)
    second = assets.build_publication_fragments(
        copy.deepcopy(result), copy.deepcopy(v0_qc)
    )

    assert first == second
    assert set(first) == {*assets.README_MARKERS, "readme-metrics-v1"}
    matrix = first["experiment-matrix"]
    assert "| paired readout | 101 | 3 | 87.5 ms | none |" in matrix
    assert "| switch-off hold | 110 | 5 | 50 ms | 0.1–2.1 s |" in matrix
    assert "| bright wait | 100 | 2 | 50 ms | 0.1–1.9 s |" in matrix

    changed = copy.deepcopy(v0_qc)
    changed["dataset"]["n_shots"] = 321
    changed_matrix = assets.render_experiment_matrix(result, changed)
    assert "| paired readout | 321 |" in changed_matrix
    assert "| paired readout | 101 |" not in changed_matrix

    validation = first["sweep-validation"]
    assert "11.205 px" in validation
    assert "11.09 px" in validation
    assert "fixed ascending order" in validation
    results = first["loss-sweep-results"]
    assert "6/2/2 per condition; 66/22/22 total" in results
    assert "no post-wait retention trend was resolved" in results
    assert "does not prove a fixed per-pulse cost" in results
    assert "not empirical fidelity" in results
    assert first["readme-metrics-v1"].startswith(
        "### V1 held-out loss-sweep inference"
    )
    for name, fragment in first.items():
        assert_public_safe(fragment, f"test {name} fragment")
        assert "C:\\" not in fragment
        assert "20260728_0044" not in fragment
        assert "20260728_0050" not in fragment


def test_readme_and_metrics_injection_is_exact_idempotent_and_v0_preserving():
    fragments = assets.build_publication_fragments(
        _result_fixture(), _v0_qc_fixture()
    )
    readme = _readme_stub("\r\n")
    readme_replacements = {
        marker: fragments[marker] for marker in assets.README_MARKERS
    }
    once = assets.inject_readme_fragments(readme, readme_replacements)
    twice = assets.inject_readme_fragments(once, readme_replacements)

    assert once == twice
    assert once.startswith("V0 prefix\r\n")
    assert once.endswith("V0 suffix\r\n")
    for marker in assets.README_MARKERS:
        assert once.count(f"<!-- BEGIN:{marker} -->") == 1
        assert once.count(f"<!-- END:{marker} -->") == 1
        assert f"old {marker}" not in once

    v0_metrics = "V0 metrics line 1\r\nV0 metrics line 2\r\n"
    metrics_once = assets.update_v1_metrics_text(
        v0_metrics, fragments["readme-metrics-v1"]
    )
    metrics_twice = assets.update_v1_metrics_text(
        metrics_once, fragments["readme-metrics-v1"]
    )
    assert metrics_once == metrics_twice
    assert metrics_once.startswith(v0_metrics)
    assert metrics_once.count(
        f"<!-- BEGIN:{assets.METRICS_V1_MARKER} -->"
    ) == 1
    assert_public_safe(metrics_once, "injected metrics")


def test_marker_injection_rejects_missing_or_duplicate_regions():
    fragments = assets.build_publication_fragments(
        _result_fixture(), _v0_qc_fixture()
    )
    replacements = {
        marker: fragments[marker] for marker in assets.README_MARKERS
    }
    missing = _readme_stub().replace(
        "<!-- END:sweep-validation -->", ""
    )
    with pytest.raises(assets.AssetInputError, match="exactly one"):
        assets.inject_readme_fragments(missing, replacements)

    duplicate = _readme_stub() + (
        "<!-- BEGIN:experiment-matrix -->\nextra\n"
        "<!-- END:experiment-matrix -->\n"
    )
    with pytest.raises(assets.AssetInputError, match="exactly one"):
        assets.inject_readme_fragments(duplicate, replacements)

    with pytest.raises(assets.AssetInputError, match="incomplete or duplicated"):
        assets.update_v1_metrics_text(
            "V0\n<!-- BEGIN:loss-sweeps-v1 -->\n",
            fragments["readme-metrics-v1"],
        )


def test_isolated_publication_emits_fragments_without_repo_mutation(scratch):
    result_path = scratch / "loss_sweep_results.json"
    v0_path = scratch / "qc_summary.json"
    result_path.write_text(
        json.dumps(_result_fixture(), sort_keys=True), encoding="utf-8"
    )
    v0_path.write_text(
        json.dumps(_v0_qc_fixture(), sort_keys=True), encoding="utf-8"
    )
    readme_hash = _sha(ROOT / "README.md")
    metrics_hash = _sha(ROOT / "reports" / "readme_metrics.md")
    output = scratch / "isolated"

    metadata = assets.generate_publication(
        result_path,
        v0_path,
        output,
        publish=False,
    )

    assert metadata["injection"] == {
        "mode": "isolated_preview",
        "readme_updated": False,
        "readme_metrics_updated": False,
    }
    for filename in assets.FRAGMENT_FILES.values():
        path = output / filename
        assert path.exists()
        assert path.stat().st_size > 50
        assert_public_safe(path.read_text(encoding="utf-8"), filename)
    assert (output / assets.PUBLICATION_METADATA_NAME).exists()
    assert _sha(ROOT / "README.md") == readme_hash
    assert _sha(ROOT / "reports" / "readme_metrics.md") == metrics_hash


def test_publish_mode_updates_only_temp_markdown_surfaces(scratch, monkeypatch):
    result_path = scratch / "loss_sweep_results.json"
    v0_path = scratch / "qc_summary.json"
    result_path.write_text(json.dumps(_result_fixture()), encoding="utf-8")
    v0_path.write_text(json.dumps(_v0_qc_fixture()), encoding="utf-8")
    readme_path = scratch / "README.md"
    metrics_path = scratch / "readme_metrics.md"
    readme_path.write_text(_readme_stub(), encoding="utf-8", newline="")
    v0_metrics = "V0 content stays byte-identical\n"
    metrics_path.write_text(v0_metrics, encoding="utf-8", newline="")

    monkeypatch.setattr(
        assets,
        "generate_assets",
        lambda *_: {
            "assets": {},
            "optional_assets_skipped": {},
            "visible_text": {},
        },
    )
    metadata = assets.generate_publication(
        result_path,
        v0_path,
        scratch / "published",
        publish=True,
        readme_path=readme_path,
        readme_metrics_path=metrics_path,
    )
    first_readme = readme_path.read_bytes()
    first_metrics = metrics_path.read_bytes()
    assert metadata["injection"]["mode"] == "published"
    assert metadata["injection"]["readme_updated"] is True
    assert metadata["injection"]["readme_metrics_updated"] is True
    assert first_metrics.startswith(v0_metrics.encode())

    second = assets.generate_publication(
        result_path,
        v0_path,
        scratch / "published",
        publish=True,
        readme_path=readme_path,
        readme_metrics_path=metrics_path,
    )
    assert readme_path.read_bytes() == first_readme
    assert metrics_path.read_bytes() == first_metrics
    assert second["injection"]["readme_updated"] is False
    assert second["injection"]["readme_metrics_updated"] is False
