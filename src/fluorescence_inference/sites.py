"""Site localisation from the shot-to-shot pixel variance map.

Why variance and not the mean image
-----------------------------------
On this run the *mean* fluorescence image is dominated by a static, high
contrast fringe pattern from the imaging light; the site array is barely
visible in it, and the frozen grid corners measured on an earlier day do not
apply. The *variance* across shots is the opposite: a static fringe contributes
almost nothing, while a site whose occupancy fluctuates from shot to shot
contributes a large, spatially compact variance excess. Both sub-arrays come
out cleanly, and the fitted lattice residual is well under one pixel.

Consequence worth stating plainly: site positions are inferred from atom
fluorescence, so a trap that was never loaded during these 100 shots cannot be
localised. The lattice model mitigates this by predicting a position for every
site of the fitted ``ny x nx`` block, including ones with no detected peak;
those rows carry the ``site_not_detected`` flag. A trap outside the fitted
block, or an entire never-loaded sub-array, would still be invisible.

Numerics that already exist in the general lab package (k-means seeding, corner
extraction, bilinear grids, ROI boxes) are imported from it rather than
reimplemented.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import find_objects, gaussian_filter, label, maximum_filter


@dataclass
class GridFit:
    """One fitted sub-array."""

    name: str
    ny: int
    nx: int
    origin_yx: np.ndarray          # (2,) modelled position of index (0, 0)
    row_vector_yx: np.ndarray      # (2,) displacement per row index step
    col_vector_yx: np.ndarray      # (2,) displacement per column index step
    centers_yx: np.ndarray         # (ny*nx, 2) modelled site centres, full frame
    row_index: np.ndarray          # (ny*nx,)
    col_index: np.ndarray          # (ny*nx,)
    detected: np.ndarray           # (ny*nx,) bool, a peak was found here
    residual_px: np.ndarray        # (ny*nx,) distance to nearest detected peak
    n_peaks_used: int

    def summary(self) -> dict[str, Any]:
        det = self.detected
        res = self.residual_px[det]
        return {
            "name": self.name,
            "ny": self.ny,
            "nx": self.nx,
            "n_sites": int(self.centers_yx.shape[0]),
            "n_detected": int(det.sum()),
            "n_not_detected": int((~det).sum()),
            "n_peaks_used": int(self.n_peaks_used),
            "row_pitch_px": float(np.linalg.norm(self.row_vector_yx)),
            "col_pitch_px": float(np.linalg.norm(self.col_vector_yx)),
            "basis_angle_deg": float(_angle_between(self.row_vector_yx, self.col_vector_yx)),
            "fit_residual_px_median": float(np.median(res)) if res.size else float("nan"),
            "fit_residual_px_max": float(res.max()) if res.size else float("nan"),
            "bbox_y": [float(self.centers_yx[:, 0].min()), float(self.centers_yx[:, 0].max())],
            "bbox_x": [float(self.centers_yx[:, 1].min()), float(self.centers_yx[:, 1].max())],
        }


@dataclass
class SiteMap:
    """All sub-arrays plus the derived ROI boxes, in site_id order."""

    grids: list[GridFit]
    site_id: np.ndarray
    grid_name: list[str]
    centers_yx: np.ndarray
    row_index: np.ndarray
    col_index: np.ndarray
    detected: np.ndarray
    boxes: list[tuple[int, int, int, int]]   # (y0, y1, x0, x1), half-open
    array_roi: tuple[int, int, int, int]     # (x0, x1, y0, y1) lab convention
    score_image: np.ndarray = field(repr=False, default=None)  # type: ignore[assignment]

    @property
    def n_sites(self) -> int:
        return int(self.site_id.size)

    def summary(self) -> dict[str, Any]:
        return {
            "n_sites": self.n_sites,
            "n_detected": int(self.detected.sum()),
            "array_roi_x0x1y0y1": [int(v) for v in self.array_roi],
            "grids": [g.summary() for g in self.grids],
        }


def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    c = abs(float(u @ v)) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def variance_map(sum_img: np.ndarray, sumsq_img: np.ndarray, n: int) -> np.ndarray:
    """Per-pixel variance from streaming sums (never holds the whole stack)."""
    if n < 2:
        raise ValueError("variance needs at least two frames")
    mean = sum_img / n
    var = sumsq_img / n - mean * mean
    return np.maximum(var, 0.0)


def finding_score(img: np.ndarray, *, highpass_sigma: float, smooth_sigma: float,
                  clip_negative: bool = True) -> np.ndarray:
    """High-pass then smooth - same recipe the lab uses for trap finding."""
    img = np.asarray(img, dtype=float)
    score = img - gaussian_filter(img, sigma=highpass_sigma)
    if clip_negative:
        score = np.maximum(score, 0.0)
    if smooth_sigma and smooth_sigma > 0:
        score = gaussian_filter(score, sigma=smooth_sigma)
    return score


def array_region(score: np.ndarray, *, density_sigma: float, density_percentile: float,
                 min_component_px: int, pad_px: int) -> tuple[int, int, int, int]:
    """Bounding box of the site array, as ``(x0, x1, y0, y1)``.

    Heavily smoothing the score turns the array into a compact blob while
    isolated hot pixels wash out, so thresholding the smoothed map isolates the
    array without a hand-drawn region.
    """
    dens = gaussian_filter(score, sigma=density_sigma)
    lab, n = label(dens > np.percentile(dens, density_percentile))
    if n == 0:
        raise RuntimeError("no array region found in the score image")
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = [k for k in range(1, n + 1) if sizes[k] >= min_component_px]
    if not keep:
        raise RuntimeError(
            f"no connected region reached min_component_px={min_component_px}; "
            f"largest was {int(sizes.max())} px"
        )
    objs = find_objects(lab)
    y0 = min(objs[k - 1][0].start for k in keep)
    y1 = max(objs[k - 1][0].stop for k in keep)
    x0 = min(objs[k - 1][1].start for k in keep)
    x1 = max(objs[k - 1][1].stop for k in keep)
    h, w = score.shape
    return (max(0, x0 - pad_px), min(w, x1 + pad_px),
            max(0, y0 - pad_px), min(h, y1 + pad_px))


def detect_peaks(score: np.ndarray, roi: tuple[int, int, int, int], *,
                 neighborhood: int, percentile: float) -> np.ndarray:
    """Local maxima inside the array region, as an ``(n, 2)`` array of (y, x)."""
    x0, x1, y0, y1 = roi
    mask = np.zeros(score.shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    thr = float(np.percentile(score[mask], percentile))
    peaks = (score == maximum_filter(score, size=neighborhood)) & (score > thr) & mask
    ys, xs = np.where(peaks)
    return np.column_stack([ys, xs]).astype(float)


def lattice_basis(points: np.ndarray, *, tol_lo: float, tol_hi: float,
                  angle_bin_deg: float, angle_tol_deg: float,
                  min_angle_deg: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate the two lattice vectors from nearest-neighbour displacements.

    Robust to a wrongly guessed orientation and to a handful of spurious peaks,
    which a four-corner seed is not: the corner of a noisy point cloud is often
    an outlier, and one bad corner tilts the whole grid.
    """
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = float(np.median(d.min(axis=1)))
    ii, jj = np.where((d > tol_lo * nn) & (d < tol_hi * nn))
    if ii.size < 8:
        raise RuntimeError("too few nearest-neighbour pairs to estimate a lattice basis")
    vecs = points[jj] - points[ii]

    ang = np.mod(np.arctan2(vecs[:, 0], vecs[:, 1]), np.pi)   # direction, sign folded
    nbins = max(4, int(round(180.0 / angle_bin_deg)))
    hist, edges = np.histogram(ang, bins=nbins, range=(0.0, np.pi))
    centres = 0.5 * (edges[:-1] + edges[1:])

    a1 = centres[int(np.argmax(hist))]
    sep = np.abs(np.mod(centres - a1 + np.pi / 2, np.pi) - np.pi / 2)
    cand = np.where(sep > np.deg2rad(min_angle_deg))[0]
    if cand.size == 0:
        raise RuntimeError("could not find a second lattice direction")
    a2 = centres[cand[int(np.argmax(hist[cand]))]]

    def mode_vector(a0: float) -> np.ndarray:
        dd = np.abs(np.mod(ang - a0 + np.pi / 2, np.pi) - np.pi / 2)
        sel = dd < np.deg2rad(angle_tol_deg)
        w = vecs[sel].copy()
        ref = np.array([np.sin(a0), np.cos(a0)])
        w[(w @ ref) < 0] *= -1.0
        return np.median(w, axis=0)

    return mode_vector(a1), mode_vector(a2), nn


def fit_grid(points: np.ndarray, *, ny: int, nx: int, name: str,
             tol_lo: float, tol_hi: float, angle_bin_deg: float,
             angle_tol_deg: float, min_angle_deg: float,
             refine_iterations: int, detection_radius_px: float) -> GridFit:
    """Fit an ``ny x nx`` lattice to a peak cloud and model every site position.

    Peaks are integer-indexed against the estimated basis, the densest
    ``ny x nx`` index window is selected (this drops spurious peaks outside the
    array without dropping genuine dark sites inside it), and an affine map
    ``(row, col) -> (y, x)`` is least-squares fitted to the surviving peaks.
    The fitted map then supplies a position for *all* ``ny*nx`` sites.
    """
    v1, v2, _nn = lattice_basis(points, tol_lo=tol_lo, tol_hi=tol_hi,
                                angle_bin_deg=angle_bin_deg,
                                angle_tol_deg=angle_tol_deg,
                                min_angle_deg=min_angle_deg)
    basis = np.column_stack([v1, v2])
    ref = points[int(np.argmin(points[:, 0] + points[:, 1]))]
    idx = np.round(np.linalg.solve(basis, (points - ref).T).T).astype(int)

    coef = None
    for _ in range(max(1, refine_iterations)):
        design = np.column_stack([idx, np.ones(len(idx))])
        coef, *_ = np.linalg.lstsq(design, points, rcond=None)
        basis = coef[:2].T
        idx = np.round(
            np.linalg.lstsq(basis, (points - coef[2]).T, rcond=None)[0]
        ).T.astype(int)

    # densest ny x nx index window
    best, best_i0, best_j0 = -1, idx[:, 0].min(), idx[:, 1].min()
    for i0 in range(idx[:, 0].min(), idx[:, 0].max() - ny + 2):
        for j0 in range(idx[:, 1].min(), idx[:, 1].max() - nx + 2):
            inside = ((idx[:, 0] >= i0) & (idx[:, 0] < i0 + ny)
                      & (idx[:, 1] >= j0) & (idx[:, 1] < j0 + nx))
            if inside.sum() > best:
                best, best_i0, best_j0 = int(inside.sum()), i0, j0
    keep = ((idx[:, 0] >= best_i0) & (idx[:, 0] < best_i0 + ny)
            & (idx[:, 1] >= best_j0) & (idx[:, 1] < best_j0 + nx))
    kept_pts, kept_idx = points[keep], idx[keep] - np.array([best_i0, best_j0])

    design = np.column_stack([kept_idx, np.ones(len(kept_idx))])
    coef, *_ = np.linalg.lstsq(design, kept_pts, rcond=None)

    full_idx = np.array([(i, j) for i in range(ny) for j in range(nx)], dtype=int)
    centers = np.column_stack([full_idx, np.ones(len(full_idx))]) @ coef

    dist = np.linalg.norm(centers[:, None, :] - kept_pts[None, :, :], axis=2)
    residual = dist.min(axis=1) if kept_pts.size else np.full(len(centers), np.inf)

    return GridFit(
        name=name, ny=ny, nx=nx,
        origin_yx=coef[2], row_vector_yx=coef[0], col_vector_yx=coef[1],
        centers_yx=centers, row_index=full_idx[:, 0], col_index=full_idx[:, 1],
        detected=residual <= detection_radius_px,
        residual_px=residual, n_peaks_used=int(keep.sum()),
    )


def build_site_map(var_img: np.ndarray, cfg_sites: dict[str, Any],
                   *, trap_half_width: int, kmeans_2d) -> SiteMap:
    """End-to-end site localisation for one variance map.

    ``kmeans_2d`` is injected (the general lab package supplies it) so this
    module stays free of a hard dependency on that checkout during tests.
    """
    score = finding_score(
        var_img,
        highpass_sigma=float(cfg_sites["highpass_sigma"]),
        smooth_sigma=float(cfg_sites["score_smooth_sigma"]),
        clip_negative=bool(cfg_sites.get("clip_negative", True)),
    )
    roi = array_region(
        score,
        density_sigma=float(cfg_sites["density_sigma"]),
        density_percentile=float(cfg_sites["density_percentile"]),
        min_component_px=int(cfg_sites["min_component_px"]),
        pad_px=int(cfg_sites["region_pad_px"]),
    )
    points = detect_peaks(
        score, roi,
        neighborhood=int(cfg_sites["peak_neighborhood"]),
        percentile=float(cfg_sites["peak_percentile"]),
    )
    n_grids = int(cfg_sites["n_grids"])
    names = list(cfg_sites["grid_names"])
    if len(names) != n_grids:
        raise ValueError("grid_names length must equal n_grids")

    if n_grids == 1:
        clusters = [points]
    else:
        labels, centroids = kmeans_2d(points, n_grids)
        # order sub-arrays left to right so site_id is reproducible
        order = np.argsort(centroids[:, 1])
        clusters = [points[labels == j] for j in order]

    grids = [
        fit_grid(
            cluster, ny=int(cfg_sites["ny"]), nx=int(cfg_sites["nx"]), name=name,
            tol_lo=float(cfg_sites["nn_vector_tol_lo"]),
            tol_hi=float(cfg_sites["nn_vector_tol_hi"]),
            angle_bin_deg=float(cfg_sites["angle_bin_deg"]),
            angle_tol_deg=float(cfg_sites["angle_tol_deg"]),
            min_angle_deg=float(cfg_sites["min_angle_between_basis_deg"]),
            refine_iterations=int(cfg_sites["refine_iterations"]),
            detection_radius_px=float(cfg_sites["detection_radius_px"]),
        )
        for cluster, name in zip(clusters, names)
    ]

    centers = np.vstack([g.centers_yx for g in grids])
    grid_name = [g.name for g in grids for _ in range(g.centers_yx.shape[0])]
    row_index = np.concatenate([g.row_index for g in grids])
    col_index = np.concatenate([g.col_index for g in grids])
    detected = np.concatenate([g.detected for g in grids])
    h, w = var_img.shape
    boxes = [_box(y, x, trap_half_width, (h, w)) for y, x in centers]

    return SiteMap(
        grids=grids,
        site_id=np.arange(centers.shape[0], dtype=int),
        grid_name=grid_name,
        centers_yx=centers,
        row_index=row_index,
        col_index=col_index,
        detected=detected,
        boxes=boxes,
        array_roi=roi,
        score_image=score,
    )


def _box(y: float, x: float, half_width: int, shape: tuple[int, int]
         ) -> tuple[int, int, int, int]:
    """ROI box, identical convention to the lab helper (half-open, clipped)."""
    yi, xi, hw = int(round(y)), int(round(x)), int(half_width)
    return (max(0, yi - hw), min(shape[0], yi + hw + 1),
            max(0, xi - hw), min(shape[1], xi + hw + 1))
