"""Background references for the ROI counts.

Three measurement variants are carried through the standardized table so that
downstream work can compare them instead of inheriting one choice:

===========  ==============================================================
``A``        ``roi_sum`` - raw sum over the ROI box.
``B``        ``roi_sum - local_background``, the annulus median scaled to the
             ROI pixel count. This is the primary measurement.
``C``        ``roi_sum - global_background``, using a common-mode level
             measured on genuinely site-free pixels of the same frame.
===========  ==============================================================

Variant C is only defensible because this frame is much larger than the site
array, so a large region containing no site exists. Taking a median over the
*sites* instead would fold occupancy changes into "background" and is not done
anywhere in this package.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SiteFreeReference:
    """Boolean mask of pixels used for the common-mode level."""

    mask: np.ndarray
    n_pixels: int
    exclusion_pad_px: int

    def level(self, img: np.ndarray) -> float:
        """Median count per pixel over the site-free region of one frame."""
        return float(np.median(img[self.mask]))


def site_free_mask(shape: tuple[int, int], array_roi: tuple[int, int, int, int],
                   *, exclusion_pad_px: int, min_reference_px: int) -> SiteFreeReference:
    """Everything outside the array region, dilated by a margin.

    The margin keeps the tails of the outermost sites' point-spread functions
    out of the reference.
    """
    h, w = shape
    x0, x1, y0, y1 = array_roi
    mask = np.ones((h, w), dtype=bool)
    ey0 = max(0, y0 - exclusion_pad_px)
    ey1 = min(h, y1 + exclusion_pad_px)
    ex0 = max(0, x0 - exclusion_pad_px)
    ex1 = min(w, x1 + exclusion_pad_px)
    mask[ey0:ey1, ex0:ex1] = False
    n = int(mask.sum())
    if n < min_reference_px:
        raise RuntimeError(
            f"site-free reference has only {n} px (< {min_reference_px}); "
            f"the frame is too small for a common-mode reference at this padding"
        )
    return SiteFreeReference(mask=mask, n_pixels=n, exclusion_pad_px=exclusion_pad_px)


def annulus_background(img: np.ndarray, box: tuple[int, int, int, int],
                       *, inner_half_width: int, outer_half_width: int,
                       stat: str = "median") -> tuple[float, float, int]:
    """Local background for one ROI box: ``(roi_equivalent, per_pixel, n_px)``.

    Numerically identical to the lab helper ``local_background_for_box``; it
    additionally returns the number of annulus pixels actually used, which the
    quality flags need.
    """
    y0, y1, x0, x1 = box
    cy, cx = 0.5 * (y0 + y1 - 1), 0.5 * (x0 + x1 - 1)
    h, w = img.shape

    def _box(hw: int) -> tuple[int, int, int, int]:
        yi, xi = int(round(cy)), int(round(cx))
        return (max(0, yi - hw), min(h, yi + hw + 1),
                max(0, xi - hw), min(w, xi + hw + 1))

    oy0, oy1, ox0, ox1 = _box(outer_half_width)
    iy0, iy1, ix0, ix1 = _box(inner_half_width)

    patch = img[oy0:oy1, ox0:ox1]
    mask = np.ones(patch.shape, dtype=bool)
    mask[max(0, iy0 - oy0):min(patch.shape[0], iy1 - oy0),
         max(0, ix0 - ox0):min(patch.shape[1], ix1 - ox0)] = False

    vals = patch[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        per_pixel = 0.0
    elif stat == "median":
        per_pixel = float(np.median(vals))
    elif stat == "mean":
        per_pixel = float(np.mean(vals))
    else:
        raise ValueError("stat must be 'median' or 'mean'")

    n_trap_px = (y1 - y0) * (x1 - x0)
    return per_pixel * n_trap_px, per_pixel, int(vals.size)
