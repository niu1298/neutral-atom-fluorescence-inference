"""Site finding, exercised against a synthetic array with known truth."""
from __future__ import annotations

import numpy as np
import pytest

from fluorescence_inference import sites


def _kmeans_2d(coords, k, iters=100):
    """Local copy of the lab's deterministic k-means, so this test file runs
    without the private analysis checkout. `test_reuse_matches_lab.py` asserts
    the two agree."""
    coords = np.asarray(coords, dtype=float)
    axis = int(np.argmax(coords.var(axis=0)))
    order = np.argsort(coords[:, axis])
    seed_idx = np.linspace(0, len(coords) - 1, k + 2)[1:-1].astype(int)
    centroids = coords[order[seed_idx]].copy()
    labels = np.full(len(coords), -1, dtype=int)
    for _ in range(iters):
        d = np.linalg.norm(coords[:, None, :] - centroids[None, :, :], axis=2)
        new = np.argmin(d, axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            if np.any(labels == j):
                centroids[j] = coords[labels == j].mean(axis=0)
    return labels, centroids


def _cfg(fx) -> dict:
    return {
        "highpass_sigma": 20.0, "score_smooth_sigma": 0.8, "clip_negative": True,
        "density_sigma": 8.0, "density_percentile": 95.0,
        "min_component_px": 100, "region_pad_px": 12,
        "peak_neighborhood": 5, "peak_percentile": 96.0,
        "n_grids": fx["n_grids"], "grid_names": ["grid_A", "grid_B"],
        "ny": fx["ny"], "nx": fx["nx"],
        "nn_vector_tol_lo": 0.6, "nn_vector_tol_hi": 1.45,
        "angle_bin_deg": 5.0, "angle_tol_deg": 15.0,
        "min_angle_between_basis_deg": 50.0,
        "refine_iterations": 6, "detection_radius_px": 3.0,
    }


@pytest.fixture
def var_img(synthetic_array):
    f = synthetic_array["frames"].reshape(-1, *synthetic_array["shape"])
    return sites.variance_map(f.sum(axis=0), (f * f).sum(axis=0), f.shape[0])


def test_variance_map_matches_numpy(synthetic_array):
    f = synthetic_array["frames"].reshape(-1, *synthetic_array["shape"])
    v = sites.variance_map(f.sum(axis=0), (f * f).sum(axis=0), f.shape[0])
    np.testing.assert_allclose(v, f.var(axis=0), rtol=1e-8, atol=1e-6)


def test_variance_needs_two_frames():
    z = np.zeros((4, 4))
    with pytest.raises(ValueError, match="at least two"):
        sites.variance_map(z, z, 1)


def test_sites_are_recovered_to_subpixel(synthetic_array, var_img):
    fx = synthetic_array
    sm = sites.build_site_map(var_img, _cfg(fx), trap_half_width=2,
                              kmeans_2d=_kmeans_2d)

    assert sm.n_sites == fx["n_grids"] * fx["ny"] * fx["nx"]
    d = np.linalg.norm(sm.centers_yx[:, None, :] - fx["truth_yx"][None, :, :],
                       axis=2)
    assert d.min(axis=1).max() < 1.0, "every modelled site within 1 px of truth"
    # and the mapping is one-to-one
    assert len(set(np.argmin(d, axis=1))) == sm.n_sites


def test_mean_image_would_have_failed(synthetic_array):
    """The fixture reproduces the real problem: the mean is fringe dominated.

    If this ever starts passing on the mean image, the fixture has stopped
    representing the dataset and the variance argument needs re-examining.
    """
    fx = synthetic_array
    mean_img = fx["frames"].reshape(-1, *fx["shape"]).mean(axis=0)
    score = sites.finding_score(mean_img, highpass_sigma=20.0, smooth_sigma=0.8)
    roi = (0, fx["shape"][1], 0, fx["shape"][0])
    peaks = sites.detect_peaks(score, roi, neighborhood=5, percentile=96.0)
    d = np.linalg.norm(peaks[:, None, :] - fx["truth_yx"][None, :, :], axis=2)
    on_site = int((d.min(axis=1) < 1.5).sum())
    assert on_site < 0.6 * len(peaks), (
        "the mean image is unexpectedly clean; re-check the fringe amplitude")


def test_lattice_basis_is_orthogonal_and_correct_pitch(synthetic_array, var_img):
    fx = synthetic_array
    sm = sites.build_site_map(var_img, _cfg(fx), trap_half_width=2,
                              kmeans_2d=_kmeans_2d)
    for g in sm.grids:
        s = g.summary()
        assert abs(s["row_pitch_px"] - fx["pitch"]) < 0.5
        assert abs(s["col_pitch_px"] - fx["pitch"]) < 0.5
        assert abs(s["basis_angle_deg"] - 90.0) < 5.0
        assert s["fit_residual_px_median"] < 0.7


def test_boxes_have_the_requested_size(synthetic_array, var_img):
    sm = sites.build_site_map(var_img, _cfg(synthetic_array), trap_half_width=2,
                              kmeans_2d=_kmeans_2d)
    for y0, y1, x0, x1 in sm.boxes:
        assert (y1 - y0, x1 - x0) == (5, 5)


def test_site_finding_is_deterministic(synthetic_array, var_img):
    cfg = _cfg(synthetic_array)
    a = sites.build_site_map(var_img, cfg, trap_half_width=2, kmeans_2d=_kmeans_2d)
    b = sites.build_site_map(var_img, cfg, trap_half_width=2, kmeans_2d=_kmeans_2d)
    np.testing.assert_array_equal(a.centers_yx, b.centers_yx)
    assert a.boxes == b.boxes


def test_array_region_rejects_an_empty_image():
    with pytest.raises(RuntimeError):
        sites.array_region(np.zeros((80, 80)), density_sigma=6.0,
                           density_percentile=99.0, min_component_px=500,
                           pad_px=5)


def test_undetected_site_still_gets_a_modelled_position(synthetic_array):
    """A site that is never occupied must still be measured, not dropped."""
    fx = synthetic_array
    frames = fx["frames"].copy()
    cy, cx = fx["truth_yx"][7]
    yy, xx = np.mgrid[0:fx["shape"][0], 0:fx["shape"][1]]
    hole = ((yy - cy) ** 2 + (xx - cx) ** 2) < 9.0
    base = frames[..., :, :][0, 0][hole].mean()
    frames[:, :, hole] = base

    flat = frames.reshape(-1, *fx["shape"])
    v = sites.variance_map(flat.sum(axis=0), (flat * flat).sum(axis=0), len(flat))
    sm = sites.build_site_map(v, _cfg(fx), trap_half_width=2, kmeans_2d=_kmeans_2d)

    assert sm.n_sites == fx["n_grids"] * fx["ny"] * fx["nx"]
    d = np.linalg.norm(sm.centers_yx - np.array([cy, cx]), axis=1)
    k = int(np.argmin(d))
    assert d[k] < 1.5, "the blanked site still has a modelled position"
    assert not sm.detected[k], "and it is flagged as undetected"
