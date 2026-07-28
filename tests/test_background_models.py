"""Background models: masking, robustness, and column separation."""
from __future__ import annotations

import numpy as np
import pytest

from fluorescence_inference import background_models as bm
from fluorescence_inference import schema


# ------------------------------------------------------------------- masking
def _centers(pitch: float = 10.5, n: int = 6, origin=(40.0, 40.0)) -> np.ndarray:
    return np.array([[origin[0] + i * pitch, origin[1] + j * pitch]
                     for i in range(n) for j in range(n)], dtype=float)


def test_mask_excludes_every_site_to_the_stated_radius():
    c = _centers()
    mask = bm.build_site_mask((140, 140), c, radius_px=5.0)
    # offsets strictly inside the radius: the mask is defined on continuous
    # distance, so a pixel rounded from exactly r away can land just outside
    for cy, cx in c:
        for dy, dx in ((0, 0), (0, 4), (4, 0), (3, 3), (0, -4), (-4, 0), (-3, -3)):
            y, x = int(round(cy + dy)), int(round(cx + dx))
            assert not mask.usable[y, x], f"({dy},{dx}) from a site is not masked"


def test_mask_leaves_pixels_well_away_from_sites_usable():
    mask = bm.build_site_mask((140, 140), _centers(), radius_px=5.0)
    assert mask.usable[5, 5]
    assert mask.usable[135, 135]
    assert mask.n_usable == int(mask.usable.sum())


def test_no_roi_pixel_can_enter_a_background_fit():
    """The ROI is 5x5; the mask radius must strictly contain it."""
    c = _centers()
    mask = bm.build_site_mask((140, 140), c, radius_px=5.0)
    half = 2
    for cy, cx in c:
        y0, x0 = int(round(cy)) - half, int(round(cx)) - half
        roi = mask.usable[y0:y0 + 2 * half + 1, x0:x0 + 2 * half + 1]
        assert not roi.any(), "an ROI pixel is usable for background estimation"


def test_mask_excludes_hot_pixels():
    rng = np.random.default_rng(0)
    img = rng.normal(500.0, 4.0, (140, 140))
    img[3, 3] = 50000.0
    mask = bm.build_site_mask((140, 140), _centers(), radius_px=5.0,
                              reference_image=img, hot_pixel_sigma=8.0)
    assert not mask.usable[3, 3]
    assert mask.hot_pixel_count >= 1


def test_mask_reports_the_masked_fraction_inside_the_array():
    mask = bm.build_site_mask((140, 140), _centers(), radius_px=5.0)
    assert 0.3 < mask.masked_fraction_in_array_bbox < 1.0
    s = mask.summary()
    assert s["mask_radius_px"] == 5.0
    assert s["n_usable_px"] == mask.n_usable


# ------------------------------------------------------------- surface fits
def test_surface_recovers_a_known_polynomial():
    h, w = 200, 220
    yy, xx = np.mgrid[0:h, 0:w]
    truth = 500.0 + 0.02 * yy + 0.03 * xx + 1e-4 * (yy - 100) ** 2
    usable = np.ones((h, w), dtype=bool)
    surface, info = bm.fit_robust_surface(truth, usable, degree=4,
                                          max_fit_pixels=20000)
    assert np.abs(surface - truth).max() < 1.0
    assert info["residual_rms"] < 0.5


def test_surface_is_robust_to_injected_bright_sites():
    """Unmasked fluorescence must not drag the background surface upward.

    This is the failure the annulus suffers from, so the replacement has to be
    demonstrably immune to it.
    """
    rng = np.random.default_rng(3)
    h, w = 200, 220
    yy, xx = np.mgrid[0:h, 0:w]
    truth = 500.0 + 0.02 * yy + 0.03 * xx
    img = truth + rng.normal(0, 3.0, (h, w))
    contaminated = img.copy()
    for cy, cx in _centers(pitch=18.0, n=6, origin=(60.0, 60.0)):
        y, x = int(cy), int(cx)
        contaminated[y - 1:y + 2, x - 1:x + 2] += 4000.0

    usable = np.ones((h, w), dtype=bool)
    clean, _ = bm.fit_robust_surface(img, usable, degree=4, max_fit_pixels=40000)
    dirty, _ = bm.fit_robust_surface(contaminated, usable, degree=4,
                                     max_fit_pixels=40000)
    assert np.abs(dirty - clean).max() < 30.0, "bright sites moved the surface"

    # and a least-squares fit would have been dragged much further
    ys, xs = np.where(usable)
    A = bm._poly_design(ys.astype(float)[::7], xs.astype(float)[::7],
                        img.shape, 4)
    coef, *_ = np.linalg.lstsq(A, contaminated[ys[::7], xs[::7]], rcond=None)
    ls = (bm._poly_design(np.mgrid[0:h, 0:w][0].ravel().astype(float),
                          np.mgrid[0:h, 0:w][1].ravel().astype(float),
                          img.shape, 4) @ coef).reshape(h, w)
    assert np.abs(ls - clean).max() > np.abs(dirty - clean).max()


def test_surface_fit_is_deterministic():
    rng = np.random.default_rng(1)
    img = 500.0 + rng.normal(0, 5.0, (150, 160))
    usable = np.ones((150, 160), dtype=bool)
    a, ia = bm.fit_robust_surface(img, usable, degree=3, max_fit_pixels=8000)
    b, ib = bm.fit_robust_surface(img, usable, degree=3, max_fit_pixels=8000)
    np.testing.assert_array_equal(a, b)
    assert ia == ib


# ---------------------------------------------------------------- template
def test_template_fills_site_pixels_from_the_surface():
    h, w = 160, 170
    c = _centers(pitch=12.0, n=5, origin=(50.0, 50.0))
    yy, xx = np.mgrid[0:h, 0:w]
    base = 500.0 + 0.05 * yy
    mean_img = base.copy()
    for cy, cx in c:                       # a bright average atom at each site
        y, x = int(cy), int(cx)
        mean_img[y - 1:y + 2, x - 1:x + 2] += 3000.0

    mask = bm.build_site_mask((h, w), c, radius_px=5.0)
    tpl = bm.build_fixed_template(mean_img, mask, degree=3, max_fit_pixels=20000)

    for cy, cx in c:                       # site pixels must not carry the atom
        y, x = int(cy), int(cx)
        assert tpl.template[y, x] < 200.0
    assert abs(float(np.median(tpl.template[mask.usable]))) < 1e-6


def test_template_offset_tracks_a_pure_level_shift():
    h, w = 140, 150
    c = _centers(pitch=12.0, n=4, origin=(45.0, 45.0))
    yy, _ = np.mgrid[0:h, 0:w]
    mean_img = 500.0 + 0.04 * yy
    mask = bm.build_site_mask((h, w), c, radius_px=5.0)
    tpl = bm.build_fixed_template(mean_img, mask, degree=3, max_fit_pixels=20000)
    # the offset is an absolute level, so a pure shift must move it by exactly
    # the shift and by nothing else
    base = tpl.level(mean_img, mask.usable)
    assert tpl.level(mean_img + 17.0, mask.usable) - base == pytest.approx(17.0, abs=1e-6)
    assert tpl.level(mean_img - 4.5, mask.usable) - base == pytest.approx(-4.5, abs=1e-6)


# ---------------------------------------------------------------- integration
def test_evaluate_frame_returns_all_methods_and_residuals():
    rng = np.random.default_rng(7)
    h, w = 160, 170
    c = _centers(pitch=12.0, n=5, origin=(50.0, 50.0))
    yy, xx = np.mgrid[0:h, 0:w]
    base = 500.0 + 0.04 * yy + 8.0 * np.sin((yy + xx) / 9.0)
    mask = bm.build_site_mask((h, w), c, radius_px=5.0)
    tpl = bm.build_fixed_template(base, mask, degree=3, max_fit_pixels=20000)

    img = base + rng.normal(0, 2.0, (h, w))
    fb = bm.evaluate_frame(img, mask, tpl, degree=3, max_fit_pixels=20000)

    assert fb.spatial.shape == img.shape
    assert fb.fixed_offset.shape == img.shape
    # the template knows about the fringes; a flat level and a smooth surface
    # do not, so it must leave the smallest residual
    assert fb.residual_rms_fixed < fb.residual_rms_spatial < fb.residual_rms_global


def test_residual_structure_detects_a_left_behind_pattern():
    h, w = 160, 170
    yy, xx = np.mgrid[0:h, 0:w]
    img = 500.0 + 30.0 * np.sin((yy + xx) / 25.0)
    mask = bm.build_site_mask((h, w), _centers(pitch=12.0, n=4, origin=(50., 50.)),
                              radius_px=5.0)
    left = bm.residual_spatial_structure(img, 500.0, mask, block=25)
    removed = bm.residual_spatial_structure(img, img, mask, block=25)
    assert left["block_median_std"] > 5.0
    assert removed["block_median_std"] < 1e-6


def test_roi_sums_integrate_the_model():
    model = np.full((60, 60), 4.0)
    boxes = [(10, 15, 10, 15), (20, 25, 30, 35)]
    np.testing.assert_allclose(bm.roi_sums(model, boxes), [100.0, 100.0])


# ---------------------------------------------------------- schema plumbing
def test_every_method_has_its_own_pair_of_columns():
    seen: set[str] = set()
    for method, (bg, ct) in schema.METHOD_COLUMNS.items():
        assert ct in schema.ALL_COLUMNS, f"{method}: {ct} not declared"
        assert ct not in seen, f"{method} shares its corrected column"
        seen.add(ct)
        if bg is not None:
            assert bg in schema.ALL_COLUMNS, f"{method}: {bg} not declared"
            assert bg not in seen, f"{method} shares its background column"
            seen.add(bg)


def test_the_contaminated_method_is_marked_and_not_a_candidate():
    assert "annulus_contaminated" in schema.CONTAMINATED_METHODS
    for col in ("background_annulus_contaminated",
                "count_corrected_annulus_contaminated"):
        assert "DIAGNOSTIC ONLY" in schema.ALL_COLUMNS[col][2]
