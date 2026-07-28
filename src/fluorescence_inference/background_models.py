"""Background models for the ROI counts.

Four methods are computed for every ROI and stored in separate columns. None
overwrites another, so the choice of correction stays visible and revisable
downstream instead of being frozen into a single number here.

======  ============================================================
``A``   no correction — the raw ROI sum
``B``   global site-free median of the same frame, per pixel
``C``   robust smooth spatial surface fitted to site-free pixels of the
        same frame
``D``   a fixed spatial template plus a per-(shot, frame) common-mode
        offset
======  ============================================================

The local annulus used in the first version of this pipeline is kept only as a
diagnostic. At a 10–11 px site pitch the 13–33 px annulus contains roughly ten
neighbouring sites; it is contaminated by array light and is unsuitable as a
primary background. See ``docs/DATA_AUDIT.md``.

The masking rule shared by B, C and D
-------------------------------------
Every detected site is masked out to ``mask_radius_px``, which must comfortably
exceed the ROI half-width plus the point-spread tail. At this site pitch that
masks essentially the whole array footprint, so C and D necessarily *interpolate*
the background underneath the array from surrounding site-free pixels. That is
the honest consequence of a dense array and is reported, not hidden: the
validation report records the masked fraction inside the array bounding box.

No estimator here uses a statistic computed over the sites themselves. A median
over sites would absorb changes in occupied fraction into "background" and make
occupancy partly unobservable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


# --------------------------------------------------------------------- masks
@dataclass(frozen=True)
class SiteMask:
    """Which pixels may be used to estimate background."""

    usable: np.ndarray            # True where a pixel may enter a background fit
    site_mask: np.ndarray         # True where a pixel is inside a site exclusion disk
    radius_px: float
    n_usable: int
    n_site_masked: int
    array_bbox: tuple[int, int, int, int]        # y0, y1, x0, x1
    masked_fraction_in_array_bbox: float
    hot_pixel_count: int

    def summary(self) -> dict[str, Any]:
        return {
            "mask_radius_px": self.radius_px,
            "n_usable_px": self.n_usable,
            "n_site_masked_px": self.n_site_masked,
            "array_bbox_y0y1x0x1": [int(v) for v in self.array_bbox],
            "masked_fraction_in_array_bbox": self.masked_fraction_in_array_bbox,
            "hot_pixels_excluded": self.hot_pixel_count,
        }


def build_site_mask(shape: tuple[int, int], centers_yx: np.ndarray, *,
                    radius_px: float, reference_image: np.ndarray | None = None,
                    hot_pixel_sigma: float = 8.0) -> SiteMask:
    """Exclude a disk of ``radius_px`` around every site, plus hot pixels.

    ``reference_image`` is used only to find hot pixels; pass the run-mean so
    that a single bright shot cannot decide the mask, which would make the mask
    depend on the data it is later used to correct.
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    site = np.zeros((h, w), dtype=bool)
    r2 = float(radius_px) ** 2
    pad = int(np.ceil(radius_px)) + 1
    for cy, cx in np.asarray(centers_yx, dtype=float):
        y0, y1 = max(0, int(cy) - pad), min(h, int(cy) + pad + 1)
        x0, x1 = max(0, int(cx) - pad), min(w, int(cx) + pad + 1)
        d2 = (yy[y0:y1, x0:x1] - cy) ** 2 + (xx[y0:y1, x0:x1] - cx) ** 2
        site[y0:y1, x0:x1] |= d2 <= r2

    usable = ~site
    hot = 0
    if reference_image is not None:
        vals = np.asarray(reference_image, dtype=float)[usable]
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) * 1.4826
        if mad <= 0:          # a perfectly flat reference; fall back to the std
            mad = float(np.std(vals))
        if mad > 0:
            hot_mask = (np.asarray(reference_image, dtype=float) - med) > hot_pixel_sigma * mad
            hot = int(np.count_nonzero(hot_mask & usable))
            usable = usable & ~hot_mask

    c = np.asarray(centers_yx, dtype=float)
    by0, by1 = int(np.floor(c[:, 0].min())), int(np.ceil(c[:, 0].max())) + 1
    bx0, bx1 = int(np.floor(c[:, 1].min())), int(np.ceil(c[:, 1].max())) + 1
    in_box = site[by0:by1, bx0:bx1]

    return SiteMask(
        usable=usable, site_mask=site, radius_px=float(radius_px),
        n_usable=int(usable.sum()), n_site_masked=int(site.sum()),
        array_bbox=(by0, by1, bx0, bx1),
        masked_fraction_in_array_bbox=float(in_box.mean()),
        hot_pixel_count=hot,
    )


# ------------------------------------------------------------- poly surface
def _poly_design(y: np.ndarray, x: np.ndarray, shape: tuple[int, int],
                 degree: int) -> np.ndarray:
    """Total-degree 2D polynomial basis on coordinates normalised to [-1, 1]."""
    h, w = shape
    yn = (2.0 * y / max(h - 1, 1)) - 1.0
    xn = (2.0 * x / max(w - 1, 1)) - 1.0
    cols = [yn ** i * xn ** j
            for i in range(degree + 1) for j in range(degree + 1 - i)]
    return np.column_stack(cols)


def fit_robust_surface(img: np.ndarray, usable: np.ndarray, *, degree: int,
                       max_fit_pixels: int, huber_k: float = 1.345,
                       iterations: int = 4) -> tuple[np.ndarray, dict[str, Any]]:
    """Iteratively reweighted least-squares polynomial surface.

    Huber weights rather than plain least squares: a handful of unmasked
    fluorescence tails or cosmic rays would otherwise pull the surface up
    everywhere, and the surface is what gets subtracted from the signal.

    The fit pixels are a fixed stride through the usable set, never a random
    sample, so the result is reproducible bit for bit.
    """
    img = np.asarray(img, dtype=float)
    ys, xs = np.where(usable)
    stride = max(1, int(np.ceil(ys.size / max_fit_pixels)))
    ys, xs = ys[::stride], xs[::stride]
    z = img[ys, xs]

    A = _poly_design(ys.astype(float), xs.astype(float), img.shape, degree)
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    scale = float("nan")
    for _ in range(iterations):
        resid = z - A @ coef
        mad = float(np.median(np.abs(resid - np.median(resid))))
        scale = max(mad * 1.4826, 1e-9)
        u = np.abs(resid) / scale
        wt = np.where(u <= huber_k, 1.0, huber_k / np.maximum(u, 1e-12))
        sw = np.sqrt(wt)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], z * sw, rcond=None)

    resid = z - A @ coef
    info = {
        "degree": degree,
        "n_fit_pixels": int(ys.size),
        "stride": stride,
        "robust_scale": scale,
        "residual_rms": float(np.sqrt(np.mean(resid ** 2))),
        "residual_median_abs": float(np.median(np.abs(resid))),
    }

    h, w = img.shape
    gy, gx = np.mgrid[0:h, 0:w]
    surface = (_poly_design(gy.ravel().astype(float), gx.ravel().astype(float),
                            img.shape, degree) @ coef).reshape(h, w)
    return surface, info


# ----------------------------------------------------------------- template
@dataclass
class FixedTemplate:
    """Static spatial structure, with its own level removed.

    Built from the run-mean of every frame so that the fringe pattern is
    measured at 200-frame signal-to-noise rather than re-estimated in each
    shot. Site pixels in the run-mean carry the mean atom signal, so they are
    replaced by the robust surface: the template must describe background, not
    an average atom.
    """

    template: np.ndarray
    fit_info: dict[str, Any] = field(default_factory=dict)

    def level(self, img: np.ndarray, usable: np.ndarray) -> float:
        """Common-mode offset of one frame against the template."""
        return float(np.median(img[usable] - self.template[usable]))


def build_fixed_template(mean_img: np.ndarray, mask: SiteMask, *, degree: int,
                         max_fit_pixels: int) -> FixedTemplate:
    surface, info = fit_robust_surface(mean_img, mask.usable, degree=degree,
                                       max_fit_pixels=max_fit_pixels)
    template = np.where(mask.usable, mean_img, surface)
    template = template - float(np.median(template[mask.usable]))
    info = dict(info)
    info["site_pixels_filled_from_surface"] = int(mask.n_site_masked)
    return FixedTemplate(template=template, fit_info=info)


# --------------------------------------------------------------- evaluation
@dataclass
class FrameBackgrounds:
    """The four background estimates for one frame, as full-frame maps."""

    global_level: float                       # B, counts per pixel
    spatial: np.ndarray                       # C, counts per pixel
    fixed_offset: np.ndarray                  # D, counts per pixel
    offset: float                             # D, the fitted common-mode scalar
    spatial_fit: dict[str, Any]
    residual_rms_global: float
    residual_rms_spatial: float
    residual_rms_fixed: float


def evaluate_frame(img: np.ndarray, mask: SiteMask, template: FixedTemplate, *,
                   degree: int, max_fit_pixels: int) -> FrameBackgrounds:
    """All background models for one frame, plus their site-free residuals."""
    img = np.asarray(img, dtype=float)
    usable = mask.usable

    g = float(np.median(img[usable]))
    surface, fit_info = fit_robust_surface(img, usable, degree=degree,
                                           max_fit_pixels=max_fit_pixels)
    offset = template.level(img, usable)
    fixed = template.template + offset

    def rms(model: np.ndarray | float) -> float:
        r = img[usable] - (model[usable] if isinstance(model, np.ndarray) else model)
        return float(np.sqrt(np.mean(r ** 2)))

    return FrameBackgrounds(
        global_level=g, spatial=surface, fixed_offset=fixed, offset=offset,
        spatial_fit=fit_info,
        residual_rms_global=rms(g), residual_rms_spatial=rms(surface),
        residual_rms_fixed=rms(fixed),
    )


def roi_sums(model: np.ndarray, boxes: Iterable[tuple[int, int, int, int]]
             ) -> np.ndarray:
    """Integrate a background map over each ROI box."""
    return np.array([float(model[y0:y1, x0:x1].sum()) for y0, y1, x0, x1 in boxes])


def residual_spatial_structure(img: np.ndarray, model: np.ndarray | float,
                               mask: SiteMask, *, block: int = 25) -> dict[str, Any]:
    """How much structure a background model leaves behind.

    A model that only removes the level leaves the fringes; a model that
    reproduces them does not. Block medians of the site-free residual make that
    difference measurable rather than a matter of opinion.
    """
    img = np.asarray(img, dtype=float)
    resid = img - (model if isinstance(model, np.ndarray) else float(model))
    usable = mask.usable
    h, w = img.shape
    meds: list[float] = []
    for y in range(0, h - block + 1, block):
        for x in range(0, w - block + 1, block):
            m = usable[y:y + block, x:x + block]
            if m.sum() >= 0.25 * block * block:
                meds.append(float(np.median(resid[y:y + block, x:x + block][m])))
    meds_arr = np.asarray(meds, dtype=float)
    return {
        "block_px": block,
        "n_blocks": int(meds_arr.size),
        "block_median_std": float(np.std(meds_arr, ddof=1)) if meds_arr.size > 1 else float("nan"),
        "block_median_p5_p95": [float(np.percentile(meds_arr, 5)),
                                float(np.percentile(meds_arr, 95))] if meds_arr.size else [float("nan")] * 2,
        "pixel_residual_rms": float(np.sqrt(np.mean(resid[usable] ** 2))),
    }
