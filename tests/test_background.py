"""Background references."""
from __future__ import annotations

import numpy as np
import pytest

from fluorescence_inference.background import annulus_background, site_free_mask


def test_annulus_recovers_a_flat_level():
    img = np.full((60, 60), 7.0)
    equiv, per_px, n = annulus_background(img, (28, 33, 28, 33),
                                          inner_half_width=6, outer_half_width=16)
    assert per_px == pytest.approx(7.0)
    assert equiv == pytest.approx(7.0 * 25)
    assert n == 33 * 33 - 13 * 13


def test_annulus_excludes_the_inner_box():
    img = np.full((60, 60), 7.0)
    img[24:37, 24:37] = 1000.0          # the whole inner exclusion zone
    _, per_px, _ = annulus_background(img, (28, 33, 28, 33),
                                      inner_half_width=6, outer_half_width=16)
    assert per_px == pytest.approx(7.0)


def test_annulus_median_survives_neighbour_contamination():
    """The real array has ~10 neighbours inside the annulus; the median holds."""
    rng = np.random.default_rng(0)
    img = rng.normal(500.0, 5.0, (80, 80))
    for dy in (-11, 0, 11):
        for dx in (-11, 0, 11):
            if dy == dx == 0:
                continue
            img[40 + dy - 1:40 + dy + 2, 40 + dx - 1:40 + dx + 2] += 3000.0
    _, per_px, _ = annulus_background(img, (38, 43, 38, 43),
                                      inner_half_width=6, outer_half_width=16)
    assert abs(per_px - 500.0) < 15.0


def test_annulus_mean_does_not_survive_it():
    """Which is why the config uses the median."""
    rng = np.random.default_rng(0)
    img = rng.normal(500.0, 5.0, (80, 80))
    for dy in (-11, 0, 11):
        for dx in (-11, 0, 11):
            if dy == dx == 0:
                continue
            img[40 + dy - 1:40 + dy + 2, 40 + dx - 1:40 + dx + 2] += 3000.0
    _, per_px, _ = annulus_background(img, (38, 43, 38, 43), inner_half_width=6,
                                      outer_half_width=16, stat="mean")
    assert per_px > 550.0


def test_annulus_rejects_an_unknown_statistic():
    with pytest.raises(ValueError, match="median.*mean"):
        annulus_background(np.zeros((40, 40)), (18, 23, 18, 23),
                           inner_half_width=6, outer_half_width=16, stat="mode")


def test_site_free_mask_excludes_the_array_with_a_margin():
    ref = site_free_mask((600, 700), (210, 460, 150, 400),
                         exclusion_pad_px=40, min_reference_px=1000)
    assert not ref.mask[150:400, 210:460].any()
    assert not ref.mask[120, 400], "the margin is excluded too"
    assert ref.mask[10, 10]
    assert ref.n_pixels == int(ref.mask.sum())


def test_site_free_mask_refuses_a_too_small_reference():
    with pytest.raises(RuntimeError, match="too small"):
        site_free_mask((100, 100), (0, 100, 0, 100), exclusion_pad_px=10,
                       min_reference_px=1000)


def test_site_free_level_is_a_median():
    ref = site_free_mask((200, 200), (80, 120, 80, 120), exclusion_pad_px=5,
                         min_reference_px=100)
    img = np.full((200, 200), 3.0)
    img[0, 0] = 1e6                       # one hot pixel must not matter
    assert ref.level(img) == pytest.approx(3.0)
