"""This repository must not fork the lab numerics.

Where a helper here mirrors one in the general analysis package, the two are
required to agree. Skips cleanly where that checkout is not available.
"""
from __future__ import annotations

import numpy as np
import pytest

from fluorescence_inference.background import annulus_background

pytestmark = pytest.mark.real_data


@pytest.fixture(autouse=True)
def _require_rydlab(rydlab_available):
    if not rydlab_available:
        pytest.skip("the general lab analysis package is not configured")


def test_annulus_background_matches_the_lab_implementation():
    from rydlab.atoms.grids import local_background_for_box

    rng = np.random.default_rng(11)
    img = rng.normal(520.0, 30.0, (120, 130))
    img[40:45, 40:45] += 4000.0
    for box in [(40, 45, 40, 45), (10, 15, 100, 105), (2, 7, 3, 8),
                (115, 120, 125, 130)]:
        mine, mine_px, _ = annulus_background(
            img, box, inner_half_width=6, outer_half_width=16, stat="median")
        theirs, theirs_px = local_background_for_box(
            img, box, inner_half_width=6, outer_half_width=16, stat="median")
        assert mine == pytest.approx(theirs, rel=0, abs=1e-9), box
        assert mine_px == pytest.approx(theirs_px, rel=0, abs=1e-9), box


def test_roi_box_construction_matches_the_lab_implementation():
    from rydlab.atoms.grids import make_box_from_center

    from fluorescence_inference.sites import _box

    shape = (600, 700)
    for y, x in [(300.4, 400.6), (0.2, 0.2), (599.8, 699.9), (12.5, 33.5)]:
        assert _box(y, x, 2, shape) == make_box_from_center(y, x, 2, shape)


def test_kmeans_helper_matches_the_lab_implementation():
    from rydlab.atoms.grids import kmeans_2d as lab_kmeans

    from tests.test_sites import _kmeans_2d as local_kmeans

    rng = np.random.default_rng(3)
    pts = np.vstack([rng.normal((20, 20), 3, (60, 2)),
                     rng.normal((60, 70), 3, (60, 2))])
    a_lab, c_lab = lab_kmeans(pts, 2)
    a_loc, c_loc = local_kmeans(pts, 2)
    np.testing.assert_array_equal(a_lab, a_loc)
    np.testing.assert_allclose(c_lab, c_loc)


def test_roi_sums_match_the_lab_counting_routine(cfg, real_dataset):
    """One shot, all 200 ROIs, straight through the lab's own counter."""
    from rydlab.atoms.grids import count_dataset_all_frames_all_traps

    from fluorescence_inference.dataset import SourceDataError, _frame_loader

    df, sites_df, _ = real_dataset
    try:
        load, _ = _frame_loader(cfg)
        from fluorescence_inference.dataset import discover_shots

        shots = discover_shots(cfg)
    except SourceDataError as exc:
        pytest.skip(f"raw shots unavailable: {exc}")

    shot_order = 3
    img = load(shots[shot_order], cfg.frame_specs[0]["h5_path"])
    boxes = [(int(r.roi_y0), int(r.roi_y1), int(r.roi_x0), int(r.roi_x1))
             for r in sites_df.sort_values("site_id").itertuples()]
    lab = count_dataset_all_frames_all_traps(img[None], boxes)

    mine = (df[(df["shot_order"] == shot_order) & (df["frame_id"] == 0)]
            .sort_values("site_id"))
    np.testing.assert_allclose(mine["roi_sum"].to_numpy(), lab["raw"][0], atol=1e-9)
    np.testing.assert_allclose(mine["local_background"].to_numpy(),
                               lab["local_bg"][0], atol=1e-9)
    np.testing.assert_allclose(mine["background_corrected_count"].to_numpy(),
                               lab["bgsub"][0], atol=1e-9)
