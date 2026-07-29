from __future__ import annotations

import numpy as np
import pandas as pd

from fluorescence_inference.sweep_validation import (
    GeometryGate,
    annulus_neighbour_overlap,
    geometry_match_with_lattice_registration,
    geometry_match_report,
    lattice_step_explanation,
    refine_rigid_lattice_center,
    select_background_method,
    summarize_background_methods,
)


def test_annulus_contamination_detects_neighbour_masks():
    sites = pd.DataFrame(
        {
            "site_id": [0, 1, 2],
            "site_y": [25.0, 25.0, 25.0],
            "site_x": [20.0, 31.0, 42.0],
        }
    )
    report = annulus_neighbour_overlap(
        sites,
        (60, 70),
        inner_half_width=6,
        outer_half_width=16,
        neighbour_mask_radius_px=5.0,
    )
    assert report["sites_with_any_overlap"] == 3
    assert report["overlap_fraction_median"] > 0
    assert report["neighbour_masks_intersected_median"] >= 1


def test_annulus_far_from_neighbours_has_no_overlap():
    sites = pd.DataFrame(
        {"site_id": [0, 1], "site_y": [25.0, 25.0], "site_x": [20.0, 80.0]}
    )
    report = annulus_neighbour_overlap(
        sites,
        (60, 110),
        inner_half_width=6,
        outer_half_width=16,
        neighbour_mask_radius_px=5.0,
    )
    assert report["sites_with_any_overlap"] == 0
    assert report["overlap_fraction_max"] == 0


def _background_frame() -> pd.DataFrame:
    rows = []
    for shot in range(8):
        for frame in (0, 1):
            bg = 100.0 + shot + 3 * frame
            for site in range(4):
                signal = 30.0 + 5 * site
                rows.append(
                    {
                        "run_id": "r",
                        "shot_id": shot,
                        "shot_order": shot,
                        "frame_id": frame,
                        "site_id": site,
                        "condition_id": f"c{shot % 2}",
                        "sweep_value_s": float(shot % 2),
                        "roi_sum_raw": signal + bg,
                        "background_global": bg,
                        "count_corrected_global": signal,
                        "background_spatial": bg + 0.2 * site,
                        "count_corrected_spatial": signal - 0.2 * site,
                        "background_template_offset": bg + 0.1 * site,
                        "count_corrected_template": signal - 0.1 * site,
                        "background_annulus_contaminated": bg + signal / 3,
                        "count_corrected_annulus_contaminated": 2 * signal / 3,
                    }
                )
    return pd.DataFrame(rows)


def test_background_summary_uses_unique_shot_frames():
    df = _background_frame()
    diag = []
    for shot in range(8):
        for frame in (0, 1):
            diag.append(
                {
                    "shot_order": shot,
                    "frame_id": frame,
                    "structure_global": {"block_median_std": 5.0},
                    "structure_spatial": {"block_median_std": 2.0},
                    "structure_fixed": {"block_median_std": 1.0},
                }
            )
    summary = summarize_background_methods(df, frame_diagnostics=diag)
    assert summary["global"]["per_frame"]["0"]["n_independent_shots"] == 8
    assert summary["template"]["site_free_residual_structure"][
        "median_block_median_std"
    ] == 1.0
    assert summary["annulus_contaminated"]["eligible_as_primary"] is False


def test_background_selection_ignores_separation_and_rejects_annulus():
    summary = {
        "global": {
            "eligible_as_primary": True,
            "site_free_residual_structure": {"median_block_median_std": 4.0},
            "per_frame": {"0": {"corrected_vs_background_r": 0.1}},
        },
        "spatial": {
            "eligible_as_primary": True,
            "site_free_residual_structure": {"median_block_median_std": 2.0},
            "per_frame": {"0": {"corrected_vs_background_r": 0.2}},
        },
        "template": {
            "eligible_as_primary": True,
            "site_free_residual_structure": {"median_block_median_std": 1.0},
            "per_frame": {"0": {"corrected_vs_background_r": 0.05}},
        },
        "annulus_contaminated": {
            "eligible_as_primary": False,
            "site_free_residual_structure": {"median_block_median_std": 0.0},
            "per_frame": {"0": {"corrected_vs_background_r": 0.0}},
        },
    }
    selected = select_background_method(summary)
    assert selected["selected_method"] == "template"
    assert selected["background_coupling_gate_passed"]
    assert selected["separation_used_for_selection"] is False
    assert selected["annulus_eligible"] is False


def test_background_selection_stops_on_strong_residual_coupling():
    summary = {
        "template": {
            "eligible_as_primary": True,
            "site_free_residual_structure": {"median_block_median_std": 1.0},
            "per_frame": {
                "0": {"corrected_vs_background_r": 0.72},
                "1": {"corrected_vs_background_r": 0.64},
            },
        }
    }
    selected = select_background_method(summary)
    assert selected["selected_method"] == "template"
    assert not selected["passed"]
    assert not selected["background_coupling_gate_passed"]
    assert selected["reason"]


def test_geometry_report_separates_bulk_shift_from_distortion():
    a = np.array([(y, x) for y in range(4) for x in range(4)], dtype=float)
    b = a + np.array([0.2, -0.1])
    report = geometry_match_report(a, b, match_radius_px=1.0)
    assert report["matching"]["n_matched"] == len(a)
    assert report["bulk_shift"]["magnitude_px"] > 0
    assert report["bulk_shift"]["residual_max_px"] < 1e-12
    assert GeometryGate().evaluate(report)["passed"]


def test_lattice_step_explanation_recognizes_one_index_alias():
    basis = np.array([[11.0, 1.5], [-1.5, 11.0]])
    report = lattice_step_explanation([11.1, 1.4], basis)
    assert report["nearest_index_shift"] == [1, 0]
    assert report["residual_px"] < 0.2
    assert report["consistent_with_one_basis_step"]


def test_geometry_registration_resolves_integer_alias_not_distortion():
    yy, xx = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
    reference = np.column_stack([10.0 * yy.ravel(), 10.0 * xx.ravel()])
    candidate = reference + np.array([20.0, -10.0]) + np.array([0.12, -0.08])
    result = geometry_match_with_lattice_registration(
        reference,
        candidate,
        [[10.0, 0.0], [0.0, 10.0]],
        max_index_shift=3,
    )
    assert result["registration"]["candidate_index_shift_applied"] == [-2, 1]
    assert result["registration"]["integer_lattice_alias_required"]
    matching = result["registered"]["matching"]
    assert matching["n_matched"] == 16
    assert matching["max_px"] < 0.15


def test_rigid_lattice_refinement_recovers_bulk_shift():
    yy, xx = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
    centers = np.column_stack(
        [25.0 + 12.0 * yy.ravel(), 30.0 + 12.0 * xx.ravel()]
    )
    shift = np.array([0.45, -0.35])
    image_y, image_x = np.mgrid[:80, :85]
    image = np.zeros((80, 85), dtype=float)
    for y, x in centers + shift:
        image += np.exp(-((image_y - y) ** 2 + (image_x - x) ** 2) / 2.0)
    moved, diagnostics = refine_rigid_lattice_center(
        image,
        centers,
        bandpass_fn=lambda value: value,
        max_shift_px=2.0,
    )
    np.testing.assert_allclose(
        moved - centers, np.broadcast_to(shift, centers.shape), atol=0.12
    )
    assert diagnostics["converged"]
    assert not diagnostics["parameter_on_boundary"]
