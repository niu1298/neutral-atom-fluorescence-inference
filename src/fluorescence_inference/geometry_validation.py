"""Independent validation of the site geometry.

The Round-1 pipeline detected two 10x10 blocks and called them 200 sites. That
is a *fit result*, not a verified fact about the apparatus, and this module
exists to test it rather than trust it:

* refit the geometry on the first and second half of the run independently and
  measure how far the two answers drift apart;
* check whether the imposed ``ny x nx`` window is data-supported, by refitting
  with a larger window and asking whether the extra ring contains anything;
* diagnose every site that carries the geometry flag, individually;
* test the competing explanations for two blocks appearing at once, including
  the one that would invalidate treating them as independent sites — that they
  are two images of the same atoms.

Nothing here writes to the processed dataset. It produces evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .sites import SiteMap, build_site_map, variance_map


# ----------------------------------------------------------------- matching
@dataclass
class MatchResult:
    """Nearest-pair assignment between two independently fitted site sets."""

    pairs: np.ndarray            # (n, 2) index into (a, b)
    distances: np.ndarray        # (n,) px
    unmatched_a: np.ndarray
    unmatched_b: np.ndarray
    max_distance_px: float

    def summary(self) -> dict[str, Any]:
        d = self.distances
        return {
            "n_matched": int(d.size),
            "n_unmatched_a": int(self.unmatched_a.size),
            "n_unmatched_b": int(self.unmatched_b.size),
            "match_radius_px": self.max_distance_px,
            "median_px": float(np.median(d)) if d.size else float("nan"),
            "p90_px": float(np.percentile(d, 90)) if d.size else float("nan"),
            "p99_px": float(np.percentile(d, 99)) if d.size else float("nan"),
            "max_px": float(d.max()) if d.size else float("nan"),
            "mean_px": float(d.mean()) if d.size else float("nan"),
        }


def match_site_sets(a_yx: np.ndarray, b_yx: np.ndarray, *,
                    max_distance_px: float = 4.0) -> MatchResult:
    """Globally optimal one-to-one matching, then drop pairs beyond the radius.

    Optimal assignment rather than greedy nearest-neighbour: greedy matching is
    order dependent, so two runs of the same code could disagree, which is
    exactly what a stability test must not do.
    """
    a_yx = np.asarray(a_yx, dtype=float)
    b_yx = np.asarray(b_yx, dtype=float)
    cost = np.linalg.norm(a_yx[:, None, :] - b_yx[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(cost)
    d = cost[ri, ci]
    keep = d <= max_distance_px
    pairs = np.column_stack([ri[keep], ci[keep]])
    unmatched_a = np.setdiff1d(np.arange(len(a_yx)), pairs[:, 0])
    unmatched_b = np.setdiff1d(np.arange(len(b_yx)), pairs[:, 1])
    return MatchResult(pairs=pairs, distances=d[keep], unmatched_a=unmatched_a,
                       unmatched_b=unmatched_b, max_distance_px=max_distance_px)


# ------------------------------------------------------------- subset fits
def variance_over_shots(shots: Sequence, frame_paths: Sequence[str], load
                        ) -> tuple[np.ndarray, int]:
    """Streaming variance map over an arbitrary subset of shots."""
    total = sq = None
    n = 0
    for p in shots:
        for h5_path in frame_paths:
            img = load(p, h5_path)
            if total is None:
                total = np.zeros(img.shape, dtype=np.float64)
                sq = np.zeros(img.shape, dtype=np.float64)
            total += img
            sq += img * img
            n += 1
    if total is None or n < 2:
        raise RuntimeError("need at least two frames for a variance map")
    return variance_map(total, sq, n), n


def fit_on_subset(shots: Sequence, cfg_sites: dict[str, Any], frame_paths,
                  load, *, trap_half_width: int, kmeans_2d) -> SiteMap:
    var_img, _ = variance_over_shots(shots, frame_paths, load)
    return build_site_map(var_img, cfg_sites, trap_half_width=trap_half_width,
                          kmeans_2d=kmeans_2d)


def stability_across_halves(shots: Sequence, cfg_sites: dict[str, Any],
                            frame_paths, load, *, trap_half_width: int,
                            kmeans_2d, match_radius_px: float = 4.0
                            ) -> dict[str, Any]:
    """Refit on shots 0..n/2-1 and n/2..n-1, then compare."""
    half = len(shots) // 2
    first, second = list(shots[:half]), list(shots[half:])
    map_a = fit_on_subset(first, cfg_sites, frame_paths, load,
                          trap_half_width=trap_half_width, kmeans_2d=kmeans_2d)
    map_b = fit_on_subset(second, cfg_sites, frame_paths, load,
                          trap_half_width=trap_half_width, kmeans_2d=kmeans_2d)
    match = match_site_sets(map_a.centers_yx, map_b.centers_yx,
                            max_distance_px=match_radius_px)

    per_grid: dict[str, Any] = {}
    for name in sorted({g.name for g in map_a.grids}):
        idx_a = np.array([i for i, g in enumerate(map_a.grid_name) if g == name])
        sel = np.isin(match.pairs[:, 0], idx_a)
        d = match.distances[sel]
        per_grid[name] = {
            "n_matched": int(d.size),
            "median_px": float(np.median(d)) if d.size else float("nan"),
            "max_px": float(d.max()) if d.size else float("nan"),
        }

    # a bulk translation is a different failure from a geometry change
    if match.pairs.size:
        delta = (map_b.centers_yx[match.pairs[:, 1]]
                 - map_a.centers_yx[match.pairs[:, 0]])
        shift = {"median_dy_px": float(np.median(delta[:, 0])),
                 "median_dx_px": float(np.median(delta[:, 1]))}
        residual = delta - np.median(delta, axis=0)
        shift["residual_after_bulk_shift_median_px"] = float(
            np.median(np.linalg.norm(residual, axis=1)))
        shift["residual_after_bulk_shift_max_px"] = float(
            np.linalg.norm(residual, axis=1).max())
    else:
        shift = {}

    return {
        "split": {"first_half_shots": len(first), "second_half_shots": len(second)},
        "n_sites_first_half": map_a.n_sites,
        "n_sites_second_half": map_b.n_sites,
        "matching": match.summary(),
        "per_grid": per_grid,
        "bulk_shift": shift,
        "grids_first_half": [g.summary() for g in map_a.grids],
        "grids_second_half": [g.summary() for g in map_b.grids],
        "_maps": (map_a, map_b),
    }


def array_boundary_profile(var_img: np.ndarray, site_map: SiteMap, *,
                           rings_outside: int = 3, probe_half_width: int = 2,
                           highpass_sigma: float = 30.0,
                           smooth_sigma: float = 0.8) -> dict[str, Any]:
    """Does the occupied region stop at the configured window, or fade out?

    The lattice model is extrapolated outward and the variance score is read at
    each modelled position. A structured array of exactly ``ny x nx`` traps
    gives a step: edge positions as bright as interior ones, the first ring
    outside at the noise floor. A larger lattice that merely happens to be
    loaded in the middle gives a taper instead, and the window would then be
    clipping real sites.

    This replaces an earlier check that grew the fitting window and counted
    detected peaks in the added ring. That test was wrong: growing the window
    does not find new peaks, it only relabels the same peaks as "edge", so it
    reported a failure whatever the data looked like.
    """
    from .sites import finding_score

    score = finding_score(var_img, highpass_sigma=highpass_sigma,
                          smooth_sigma=smooth_sigma)
    h, w = score.shape
    hw = int(probe_half_width)

    def probe(p: np.ndarray) -> float:
        y, x = int(round(p[0])), int(round(p[1]))
        if hw <= y < h - hw and hw <= x < w - hw:
            return float(score[y - hw:y + hw + 1, x - hw:x + hw + 1].max())
        return float("nan")

    out: dict[str, Any] = {"probe_half_width": hw, "rings_outside": rings_outside,
                           "per_grid": {}}
    ratios: list[float] = []
    for g in site_map.grids:
        o, a, b = g.origin_yx, g.row_vector_yx, g.col_vector_yx
        ny, nx = g.ny, g.nx

        def band(sel) -> list[float]:
            vals = [probe(o + i * a + j * b)
                    for i in range(-rings_outside, ny + rings_outside)
                    for j in range(-rings_outside, nx + rings_outside)
                    if sel(i, j)]
            return [v for v in vals if np.isfinite(v)]

        interior = band(lambda i, j: 1 <= i < ny - 1 and 1 <= j < nx - 1)
        edge = band(lambda i, j: (0 <= i < ny and 0 <= j < nx)
                    and not (1 <= i < ny - 1 and 1 <= j < nx - 1))
        outside = {}
        for r in range(1, rings_outside + 1):
            outside[r] = band(
                lambda i, j, r=r: (-r <= i < ny + r and -r <= j < nx + r)
                and not (-(r - 1) <= i < ny + r - 1 and -(r - 1) <= j < nx + r - 1))

        med_edge = float(np.median(edge)) if edge else float("nan")
        med_out1 = float(np.median(outside[1])) if outside[1] else float("nan")
        ratio = med_edge / med_out1 if med_out1 > 0 else float("inf")
        ratios.append(ratio)
        out["per_grid"][g.name] = {
            "window": [ny, nx],
            "median_score_interior": float(np.median(interior)) if interior else float("nan"),
            "median_score_edge_ring": med_edge,
            "median_score_outside_rings": {str(r): float(np.median(v)) if v else float("nan")
                                           for r, v in outside.items()},
            "edge_to_first_outside_ratio": ratio,
            "edge_detection_fraction": float(
                g.detected.reshape(ny, nx)[[0, -1], :].mean() / 2
                + g.detected.reshape(ny, nx)[:, [0, -1]].mean() / 2),
        }
    out["min_edge_to_outside_ratio"] = float(np.nanmin(ratios))
    out["boundary_is_sharp"] = bool(np.nanmin(ratios) >= 5.0)
    out["criterion"] = ("the edge ring of the configured window must be at "
                        "least 5x brighter in the variance score than the first "
                        "ring of modelled positions outside it")
    return out


def peak_count_saturates(var_img: np.ndarray, cfg_sites: dict[str, Any], *,
                         trap_half_width: int, kmeans_2d, grow: int = 4,
                         tolerance: float = 0.05) -> dict[str, Any]:
    """Refit with a larger window and check the peak count does not grow.

    If the occupied region really extended past the configured window, a larger
    window would capture more detected peaks. If the same peaks simply get
    redistributed over more index positions, the window is not clipping
    anything.
    """
    base = build_site_map(var_img, cfg_sites, trap_half_width=trap_half_width,
                          kmeans_2d=kmeans_2d)
    bigger = dict(cfg_sites)
    bigger["ny"] = int(cfg_sites["ny"]) + grow
    bigger["nx"] = int(cfg_sites["nx"]) + grow
    grown = build_site_map(var_img, bigger, trap_half_width=trap_half_width,
                           kmeans_2d=kmeans_2d)

    per_grid: dict[str, Any] = {}
    growth: list[float] = []
    for gb, gg in zip(base.grids, grown.grids):
        n0, n1 = gb.n_peaks_used, gg.n_peaks_used
        rel = (n1 - n0) / max(n0, 1)
        growth.append(rel)
        per_grid[gb.name] = {
            "peaks_in_configured_window": int(n0),
            "peaks_in_grown_window": int(n1),
            "relative_increase": float(rel),
        }
    return {
        "grow": grow, "tolerance": tolerance, "per_grid": per_grid,
        "max_relative_increase": float(max(growth)),
        "peak_count_saturated": bool(max(growth) <= tolerance),
        "criterion": (f"growing the window by {grow} in each direction adds no "
                      f"more than {tolerance:.0%} additional detected peaks"),
    }


# ------------------------------------------------------- flagged-site triage
def diagnose_flagged_sites(site_map: SiteMap, var_img: np.ndarray,
                           df, *, frame_shape: tuple[int, int],
                           roi_half_width: int) -> list[dict[str, Any]]:
    """Say *why* each flagged site is flagged, one site at a time."""
    peak_var = np.array([var_img[y0:y1, x0:x1].max()
                         for y0, y1, x0, x1 in site_map.boxes])
    order = np.argsort(np.argsort(peak_var))          # rank, 0 = lowest
    centers = site_map.centers_yx
    nn = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(nn, np.inf)
    residual = np.concatenate([g.residual_px for g in site_map.grids])
    h, w = frame_shape
    box = 2 * roi_half_width + 1

    out: list[dict[str, Any]] = []
    for k in np.where(~site_map.detected)[0]:
        y0, y1, x0, x1 = site_map.boxes[k]
        cy, cx = centers[k]
        sub = df[(df["site_id"] == k) & (df["frame_id"] == 0)]
        counts = sub["roi_sum_raw"].to_numpy(float) if "roi_sum_raw" in sub else \
            sub["roi_sum"].to_numpy(float)
        checks = {
            "outside_valid_image_region": bool(
                cy < roi_half_width or cx < roi_half_width
                or cy > h - 1 - roi_half_width or cx > w - 1 - roi_half_width),
            "roi_clipped_by_frame": bool((y1 - y0) != box or (x1 - x0) != box),
            "overlaps_a_neighbour": bool(nn[k].min() < box),
            "low_variance_site": bool(order[k] < 0.05 * len(order)),
            "lattice_fit_outlier": bool(residual[k] > 5.0),
        }
        if checks["outside_valid_image_region"] or checks["roi_clipped_by_frame"]:
            verdict = "outside the valid image region"
        elif checks["overlaps_a_neighbour"]:
            verdict = "ROI overlaps a neighbouring site"
        elif checks["low_variance_site"]:
            verdict = ("low-signal site: real, but its variance peak is too weak "
                       "to be detected as a local maximum")
        elif checks["lattice_fit_outlier"]:
            verdict = ("peak-detection ambiguity: the nearest detected peak is "
                       "about one lattice step away, so no peak sits at the "
                       "modelled position")
        else:
            verdict = ("nearest detected peak lies just outside the detection "
                       "radius; the modelled position itself looks sound")
        out.append({
            "site_id": int(k),
            "grid": site_map.grid_name[k],
            "row": int(site_map.row_index[k]),
            "col": int(site_map.col_index[k]),
            "center_yx": [float(cy), float(cx)],
            "fit_residual_px": float(residual[k]),
            "peak_variance": float(peak_var[k]),
            "peak_variance_rank_of_n": [int(order[k]), int(len(order))],
            "nearest_neighbour_px": float(nn[k].min()),
            "mean_roi_sum_frame0": float(counts.mean()) if counts.size else float("nan"),
            "checks": checks,
            "verdict": verdict,
            "intentionally_absent_trap": "unknown: no trap-light reference image exists",
        })
    return out


# ------------------------------------------- are the two blocks independent?
def duplicate_image_test(df, site_map: SiteMap, value_col: str, *,
                         seed: int) -> dict[str, Any]:
    """Test whether the two blocks are two images of the same atoms.

    If one physical array were being imaged twice, the two blocks would have to
    share their shot-to-shot occupancy: site (row, col) of one block would
    track its counterpart in the other almost perfectly. Comparing the
    index-matched correlation against a permuted null makes that testable, and
    a null result is what licenses treating the blocks as independent sites.
    """
    grids = sorted({g.name for g in site_map.grids})
    if len(grids) != 2:
        return {"applicable": False, "reason": f"{len(grids)} sub-arrays, test needs 2"}

    wide = df.pivot_table(index="shot_order", columns="site_id", values=value_col,
                          observed=True).sort_index()
    key = {int(s): (site_map.grid_name[s], int(site_map.row_index[s]),
                    int(site_map.col_index[s])) for s in site_map.site_id}
    a = {(r, c): s for s, (g, r, c) in key.items() if g == grids[0]}
    b = {(r, c): s for s, (g, r, c) in key.items() if g == grids[1]}
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"applicable": False, "reason": "no shared lattice indices"}

    def corr(u: np.ndarray, v: np.ndarray) -> float:
        u = u - u.mean()
        v = v - v.mean()
        d = np.linalg.norm(u) * np.linalg.norm(v)
        return float(u @ v / d) if d > 0 else float("nan")

    matched = np.array([corr(wide[a[k]].to_numpy(float), wide[b[k]].to_numpy(float))
                        for k in shared])
    rng = np.random.default_rng(seed)
    null = []
    b_ids = [b[k] for k in shared]
    for _ in range(200):
        perm = rng.permutation(len(shared))
        null.extend(corr(wide[a[shared[i]]].to_numpy(float),
                         wide[b_ids[perm[i]]].to_numpy(float))
                    for i in range(len(shared)))
    null_arr = np.asarray(null, dtype=float)

    med, null_med = float(np.median(matched)), float(np.median(null_arr))
    duplicate = bool(med > 0.8 and med - null_med > 0.5)
    return {
        "applicable": True,
        "n_index_matched_pairs": len(shared),
        "matched_correlation_median": med,
        "matched_correlation_p95": float(np.percentile(matched, 95)),
        "matched_correlation_max": float(np.max(matched)),
        "permuted_null_median": null_med,
        "permuted_null_p95": float(np.percentile(null_arr, 95)),
        "duplicate_image_hypothesis_supported": duplicate,
        "interpretation": (
            "Two images of one physical array would give index-matched "
            "correlations near 1 and far above the permuted null. "
            + ("They do, so the two blocks are NOT independent."
               if duplicate else
               "They do not, so the blocks behave as independent sites.")),
    }


def occupancy_correlation_within_grid(df, site_map: SiteMap, value_col: str
                                      ) -> dict[str, Any]:
    """Neighbour correlation inside each block, as a sanity reference."""
    wide = df.pivot_table(index="shot_order", columns="site_id", values=value_col,
                          observed=True).sort_index()
    out: dict[str, Any] = {}
    for name in sorted({g.name for g in site_map.grids}):
        ids = [int(s) for s in site_map.site_id if site_map.grid_name[s] == name]
        m = wide[ids].to_numpy(float)
        c = np.corrcoef(m, rowvar=False)
        iu = np.triu_indices_from(c, k=1)
        out[name] = {
            "median_pairwise_correlation": float(np.median(c[iu])),
            "p95_pairwise_correlation": float(np.percentile(c[iu], 95)),
        }
    return out
