"""Geometry validation: matching determinism, stability, and the gate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fluorescence_inference import geometry_validation as gv
from fluorescence_inference import schema

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------- matching
def test_matching_is_optimal_and_one_to_one():
    a = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    b = a + np.array([0.3, -0.2])
    m = gv.match_site_sets(a, b, max_distance_px=4.0)
    assert m.distances.size == 3
    assert len(set(m.pairs[:, 1].tolist())) == 3, "a target was matched twice"
    assert m.distances.max() < 0.4


def test_matching_is_independent_of_input_order():
    """Greedy nearest-neighbour would fail this; assignment must not."""
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 100, (40, 2))
    b = a + rng.normal(0, 0.2, (40, 2))
    m1 = gv.match_site_sets(a, b, max_distance_px=4.0)

    perm = rng.permutation(40)
    m2 = gv.match_site_sets(a, b[perm], max_distance_px=4.0)
    # the same pairs, expressed through the permutation
    back = {int(i): int(perm[j]) for i, j in m2.pairs}
    assert {int(i): int(j) for i, j in m1.pairs} == back
    np.testing.assert_allclose(sorted(m1.distances), sorted(m2.distances), atol=1e-9)


def test_matching_is_deterministic_across_repeats():
    rng = np.random.default_rng(5)
    a = rng.uniform(0, 100, (30, 2))
    b = a + rng.normal(0, 0.3, (30, 2))
    first = gv.match_site_sets(a, b, max_distance_px=3.0)
    second = gv.match_site_sets(a, b, max_distance_px=3.0)
    np.testing.assert_array_equal(first.pairs, second.pairs)
    np.testing.assert_array_equal(first.distances, second.distances)


def test_matching_reports_sites_beyond_the_radius_as_unmatched():
    a = np.array([[0.0, 0.0], [10.0, 0.0]])
    b = np.array([[0.1, 0.0], [90.0, 90.0]])
    m = gv.match_site_sets(a, b, max_distance_px=4.0)
    assert m.distances.size == 1
    assert m.unmatched_a.tolist() == [1]
    assert m.unmatched_b.tolist() == [1]


def test_match_summary_reports_the_requested_percentiles():
    a = np.zeros((100, 2))
    b = np.column_stack([np.linspace(0, 1, 100), np.zeros(100)])
    s = gv.match_site_sets(a, b, max_distance_px=4.0).summary()
    for key in ("median_px", "p90_px", "p99_px", "max_px", "n_unmatched_a"):
        assert key in s


# --------------------------------------------- stability on synthetic halves
def _synthetic_halves(shift_px: float = 0.0):
    """Two variance maps of the same lattice, optionally displaced."""
    from fluorescence_inference.sites import build_site_map
    from tests.test_sites import _cfg, _kmeans_2d

    h, w = 160, 180
    ny = nx = 4
    pitch = 9.0
    yy, xx = np.mgrid[0:h, 0:w]

    def make(dy: float):
        v = np.full((h, w), 40.0)
        for oy, ox in ((40.0, 35.0), (95.0, 105.0)):
            for i in range(ny):
                for j in range(nx):
                    cy = oy + i * pitch + 0.15 * j + dy
                    cx = ox + j * pitch - 0.15 * i
                    v += 9000.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.2 ** 2))
        return v

    cfg = _cfg({"n_grids": 2, "ny": ny, "nx": nx})
    a = build_site_map(make(0.0), cfg, trap_half_width=2, kmeans_2d=_kmeans_2d)
    b = build_site_map(make(shift_px), cfg, trap_half_width=2, kmeans_2d=_kmeans_2d)
    return a, b


def test_identical_halves_match_with_zero_drift():
    a, b = _synthetic_halves(shift_px=0.0)
    m = gv.match_site_sets(a.centers_yx, b.centers_yx, max_distance_px=4.0)
    assert m.distances.size == a.n_sites
    assert m.unmatched_a.size == 0 and m.unmatched_b.size == 0
    assert m.distances.max() < 1e-6


def test_a_displaced_half_is_detected_as_a_bulk_shift():
    a, b = _synthetic_halves(shift_px=2.0)
    m = gv.match_site_sets(a.centers_yx, b.centers_yx, max_distance_px=4.0)
    assert m.distances.size == a.n_sites
    delta = b.centers_yx[m.pairs[:, 1]] - a.centers_yx[m.pairs[:, 0]]
    assert float(np.median(delta[:, 0])) == pytest.approx(2.0, abs=0.3)
    residual = np.linalg.norm(delta - np.median(delta, axis=0), axis=1)
    assert residual.max() < 0.5, "a pure translation must leave no residual"


# ------------------------------------------------------------ duplicate test
def _paired_frame(correlated: bool, seed: int = 0):
    import pandas as pd

    from fluorescence_inference.sites import build_site_map
    from tests.test_sites import _cfg, _kmeans_2d

    a, _ = _synthetic_halves()
    rng = np.random.default_rng(seed)
    n_shots = 60
    rows = []
    n_per = a.n_sites // 2
    occ_a = rng.random((n_shots, n_per)) < 0.5
    occ_b = occ_a if correlated else (rng.random((n_shots, n_per)) < 0.5)
    for s in range(n_shots):
        for k in range(a.n_sites):
            grid = a.grid_name[k]
            idx = (int(a.row_index[k]), int(a.col_index[k]))
            flat = idx[0] * 4 + idx[1]
            occ = occ_a[s, flat] if grid == a.grids[0].name else occ_b[s, flat]
            rows.append({"shot_order": s, "site_id": k, "frame_id": 0,
                         "v": 3000.0 * occ + rng.normal(0, 100.0)})
    return a, pd.DataFrame(rows)


def test_duplicate_image_test_rejects_independent_blocks():
    smap, df = _paired_frame(correlated=False, seed=1)
    r = gv.duplicate_image_test(df, smap, "v", seed=3)
    assert r["applicable"]
    assert not r["duplicate_image_hypothesis_supported"]
    assert abs(r["matched_correlation_median"]) < 0.3


def test_duplicate_image_test_detects_a_duplicated_array():
    """The test has to be able to fire, or passing it means nothing."""
    smap, df = _paired_frame(correlated=True, seed=2)
    r = gv.duplicate_image_test(df, smap, "v", seed=3)
    assert r["duplicate_image_hypothesis_supported"]
    assert r["matched_correlation_median"] > 0.8


# ------------------------------------------------------------- real artefact
@pytest.mark.real_data
def test_geometry_validation_report_exists_and_passes():
    path = ROOT / "reports" / "validation" / "site_geometry_validation.json"
    if not path.exists():
        pytest.skip("run scripts/validate_site_geometry.py first")
    r = json.loads(path.read_text(encoding="utf-8"))
    assert r["gate"]["passed"], r["gate"]["checks"]
    assert r["half_run_stability"]["matching"]["max_px"] < 4.0
    assert r["n_sites_modelled"] == 200
    assert not r["two_block_independence"]["duplicate_image_test"][
        "duplicate_image_hypothesis_supported"]


@pytest.mark.real_data
def test_background_comparison_report_backs_the_configured_primary():
    path = ROOT / "reports" / "validation" / "background_method_comparison.json"
    if not path.exists():
        pytest.skip("run scripts/compare_background_methods.py first")
    r = json.loads(path.read_text(encoding="utf-8"))
    assert r["configured_matches_recommendation"], (
        f"config says {r['configured_primary_method']}, evidence says "
        f"{r['recommendation']['selected']}")
    assert r["configured_primary_method"] not in schema.CONTAMINATED_METHODS
