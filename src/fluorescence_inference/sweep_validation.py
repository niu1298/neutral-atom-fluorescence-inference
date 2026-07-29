"""Geometry and background diagnostics for multi-frame loss sweeps.

The functions in this module produce evidence, not occupancy labels.  They are
kept separate from the statistical models so background selection cannot be
made by choosing the correction that produces the cleanest lifetime.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize

from .geometry_validation import match_site_sets
from .sites import variance_map


BACKGROUND_COLUMNS: dict[str, tuple[str | None, tuple[str, ...]]] = {
    "raw": (None, ("roi_sum_raw",)),
    "global": ("background_global", ("count_corrected_global",)),
    "spatial": ("background_spatial", ("count_corrected_spatial",)),
    "template": (
        "background_template_offset",
        ("count_corrected_template", "count_corrected_fixed_offset"),
    ),
    "annulus_contaminated": (
        "background_annulus_contaminated",
        ("count_corrected_annulus_contaminated",),
    ),
}


def _existing(df: pd.DataFrame, names: Sequence[str]) -> str | None:
    return next((name for name in names if name in df.columns), None)


def _finite_pair(x: Iterable[float], y: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    xa = np.asarray(list(x), dtype=float)
    ya = np.asarray(list(y), dtype=float)
    keep = np.isfinite(xa) & np.isfinite(ya)
    return xa[keep], ya[keep]


def safe_correlation(x: Iterable[float], y: Iterable[float]) -> float:
    """Pearson correlation, returning NaN for an uninformative constant pair."""
    xa, ya = _finite_pair(x, y)
    if xa.size < 3 or np.std(xa) == 0 or np.std(ya) == 0:
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def linear_slope(x: Iterable[float], y: Iterable[float]) -> dict[str, float | int]:
    """OLS slope with an intercept, used only as a drift diagnostic."""
    xa, ya = _finite_pair(x, y)
    if xa.size < 3 or np.ptp(xa) == 0:
        return {"slope": float("nan"), "intercept": float("nan"), "n": int(xa.size)}
    design = np.column_stack([np.ones(xa.size), xa])
    coef, *_ = np.linalg.lstsq(design, ya, rcond=None)
    residual = ya - design @ coef
    dof = max(xa.size - 2, 1)
    scale2 = float(residual @ residual / dof)
    cov = scale2 * np.linalg.pinv(design.T @ design)
    return {
        "slope": float(coef[1]),
        "intercept": float(coef[0]),
        "slope_se_naive": float(np.sqrt(max(cov[1, 1], 0.0))),
        "n": int(xa.size),
    }


def annulus_neighbour_overlap(
    sites: pd.DataFrame,
    image_shape: tuple[int, int],
    *,
    inner_half_width: int,
    outer_half_width: int,
    neighbour_mask_radius_px: float,
) -> dict[str, Any]:
    """Measure overlap of each legacy annulus with every other site's mask.

    The legacy estimator uses a square outer box with a square inner box
    removed.  The neighbouring-site mask is circular, matching the site-free
    estimators.  A pixel can count only once even when two neighbour masks
    overlap.
    """
    required = {"site_id", "site_y", "site_x"}
    missing = required - set(sites.columns)
    if missing:
        raise ValueError(f"site table is missing {sorted(missing)}")
    h, w = (int(image_shape[0]), int(image_shape[1]))
    centers = sites.sort_values("site_id")[["site_y", "site_x"]].to_numpy(float)
    per_site: list[dict[str, Any]] = []
    r2 = float(neighbour_mask_radius_px) ** 2

    for i, (cy, cx) in enumerate(centers):
        yi, xi = int(round(cy)), int(round(cx))
        oy0, oy1 = max(0, yi - outer_half_width), min(h, yi + outer_half_width + 1)
        ox0, ox1 = max(0, xi - outer_half_width), min(w, xi + outer_half_width + 1)
        yy, xx = np.mgrid[oy0:oy1, ox0:ox1]
        annulus = np.ones(yy.shape, dtype=bool)
        iy0, iy1 = max(oy0, yi - inner_half_width), min(oy1, yi + inner_half_width + 1)
        ix0, ix1 = max(ox0, xi - inner_half_width), min(ox1, xi + inner_half_width + 1)
        annulus[iy0 - oy0:iy1 - oy0, ix0 - ox0:ix1 - ox0] = False

        neighbour_union = np.zeros(yy.shape, dtype=bool)
        neighbours = 0
        for j, (ny, nx) in enumerate(centers):
            if i == j:
                continue
            disk = (yy - ny) ** 2 + (xx - nx) ** 2 <= r2
            hit = bool(np.any(disk & annulus))
            neighbours += int(hit)
            neighbour_union |= disk
        n_annulus = int(annulus.sum())
        n_overlap = int(np.count_nonzero(annulus & neighbour_union))
        per_site.append(
            {
                "site_id": int(sites.sort_values("site_id").iloc[i]["site_id"]),
                "annulus_pixels": n_annulus,
                "overlap_pixels": n_overlap,
                "overlap_fraction": n_overlap / n_annulus if n_annulus else float("nan"),
                "neighbour_masks_intersected": neighbours,
            }
        )

    frac = np.asarray([r["overlap_fraction"] for r in per_site], dtype=float)
    neighbours = np.asarray([r["neighbour_masks_intersected"] for r in per_site])
    return {
        "inner_half_width_px": int(inner_half_width),
        "outer_half_width_px": int(outer_half_width),
        "neighbour_mask_radius_px": float(neighbour_mask_radius_px),
        "n_sites": int(len(per_site)),
        "sites_with_any_overlap": int(np.count_nonzero(frac > 0)),
        "overlap_fraction_median": float(np.nanmedian(frac)),
        "overlap_fraction_p90": float(np.nanpercentile(frac, 90)),
        "overlap_fraction_max": float(np.nanmax(frac)),
        "neighbour_masks_intersected_median": float(np.median(neighbours)),
        "neighbour_masks_intersected_max": int(np.max(neighbours)),
        "per_site": per_site,
    }


def _shot_frame_table(
    df: pd.DataFrame, background_col: str | None, corrected_col: str
) -> pd.DataFrame:
    identity = [
        c
        for c in (
            "dataset_id",
            "run_id",
            "shot_id",
            "shot_order",
            "frame_index",
            "frame_id",
            "frame_name",
            "condition_id",
            "sweep_value_s",
            "split",
        )
        if c in df.columns
    ]
    agg: dict[str, str] = {corrected_col: "mean"}
    if background_col:
        agg[background_col] = "mean"
    return df.groupby(identity, observed=True, dropna=False).agg(agg).reset_index()


def _frame_key(df: pd.DataFrame) -> str:
    return "frame_index" if "frame_index" in df.columns else "frame_id"


def summarize_background_methods(
    df: pd.DataFrame,
    *,
    frame_diagnostics: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Summarize physical background diagnostics without using separation.

    Residual structure is read from the site-free pixel diagnostics produced
    during extraction.  Drift and coupling use one shot-frame aggregate so 100
    duplicated site rows cannot masquerade as 100 background measurements.
    """
    frame_col = _frame_key(df)
    out: dict[str, Any] = {}
    for method, (background_name, corrected_names) in BACKGROUND_COLUMNS.items():
        corrected = _existing(df, corrected_names)
        background = background_name if background_name in df.columns else None
        if method == "template" and background is None:
            background = _existing(df, ("background_fixed_offset",))
        if corrected is None:
            continue

        method_block: dict[str, Any] = {
            "background_column": background,
            "corrected_column": corrected,
            "eligible_as_primary": method not in {"raw", "annulus_contaminated"},
            "disqualification": (
                "no background correction"
                if method == "raw"
                else "annulus intersects neighbouring-site masks"
                if method == "annulus_contaminated"
                else None
            ),
        }
        if background:
            method_block["row_level_corrected_vs_background_r"] = safe_correlation(
                df[corrected], df[background]
            )

        sf = _shot_frame_table(df, background, corrected)
        per_frame: dict[str, Any] = {}
        for fid, part in sf.groupby(frame_col, observed=True):
            record: dict[str, Any] = {
                "n_independent_shots": int(part["shot_id"].nunique()),
                "corrected_mean": float(part[corrected].mean()),
                "corrected_sd_across_shots": float(part[corrected].std(ddof=1)),
                "corrected_vs_shot_order": linear_slope(part["shot_order"], part[corrected]),
            }
            if background:
                record.update(
                    background_mean=float(part[background].mean()),
                    background_sd_across_shots=float(part[background].std(ddof=1)),
                    corrected_vs_background_r=safe_correlation(
                        part[corrected], part[background]
                    ),
                    background_vs_shot_order=linear_slope(
                        part["shot_order"], part[background]
                    ),
                )
                if "sweep_value_s" in part.columns:
                    record["background_vs_sweep"] = linear_slope(
                        part["sweep_value_s"], part[background]
                    )
            per_frame[str(int(fid))] = record
        method_block["per_frame"] = per_frame
        out[method] = method_block

    if frame_diagnostics:
        structure_map = {
            "global": "structure_global",
            "spatial": "structure_spatial",
            "template": "structure_fixed",
        }
        for method, key in structure_map.items():
            if method not in out:
                continue
            vals = [
                float(r[key]["block_median_std"])
                for r in frame_diagnostics
                if key in r and np.isfinite(float(r[key]["block_median_std"]))
            ]
            if vals:
                out[method]["site_free_residual_structure"] = {
                    "median_block_median_std": float(np.median(vals)),
                    "p90_block_median_std": float(np.percentile(vals, 90)),
                    "n_shot_frames": int(len(vals)),
                }
    return out


def select_background_method(summary: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Select among site-free methods using residual structure and coupling.

    Separation or test-set performance is intentionally absent.  Template and
    spatial methods are preferred only when their site-free residual structure
    is measured; otherwise selection stops rather than guessing.
    """
    candidates: list[tuple[float, float, str]] = []
    for method in ("global", "spatial", "template"):
        block = summary.get(method)
        if not block or not block.get("eligible_as_primary"):
            continue
        structure = block.get("site_free_residual_structure", {})
        residual = float(structure.get("median_block_median_std", np.nan))
        frame_corr = [
            abs(float(v.get("corrected_vs_background_r", np.nan)))
            for v in block.get("per_frame", {}).values()
        ]
        frame_corr = [v for v in frame_corr if np.isfinite(v)]
        coupling = float(np.median(frame_corr)) if frame_corr else float("inf")
        if np.isfinite(residual):
            candidates.append((residual, coupling, method))
    if not candidates:
        return {
            "passed": False,
            "selected_method": None,
            "reason": "no site-free candidate has residual-structure diagnostics",
        }
    candidates.sort()
    residual, coupling, selected = candidates[0]
    return {
        "passed": True,
        "selected_method": selected,
        "criterion": "minimum site-free residual block-median structure; coupling as tie-break",
        "selected_residual_structure": residual,
        "selected_abs_background_coupling": coupling,
        "ranking": [
            {"method": method, "residual_structure": r, "abs_background_coupling": c}
            for r, c, method in candidates
        ],
        "separation_used_for_selection": False,
        "annulus_eligible": False,
    }


def frame_pedestal_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Site-free ROI-equivalent background levels by frame."""
    if "background_global" not in df.columns:
        return {"available": False}
    frame_col = _frame_key(df)
    identity = [c for c in ("shot_id", frame_col, "background_global") if c in df.columns]
    one = df[identity].drop_duplicates([c for c in ("shot_id", frame_col) if c in identity])
    frames: dict[str, Any] = {}
    for fid, part in one.groupby(frame_col, observed=True):
        v = part["background_global"].to_numpy(float)
        frames[str(int(fid))] = {
            "median_roi_counts": float(np.median(v)),
            "mean_roi_counts": float(np.mean(v)),
            "sd_across_shots": float(np.std(v, ddof=1)),
            "n_shots": int(part["shot_id"].nunique()),
        }
    medians = np.asarray([v["median_roi_counts"] for v in frames.values()], dtype=float)
    return {
        "available": True,
        "frames": frames,
        "median_spread_max_minus_min_roi_counts": float(np.ptp(medians)),
        "first_minus_later_median_roi_counts": (
            float(medians[0] - np.median(medians[1:])) if medians.size > 1 else float("nan")
        ),
    }


def geometry_match_report(
    centers_a: np.ndarray,
    centers_b: np.ndarray,
    *,
    match_radius_px: float = 4.0,
) -> dict[str, Any]:
    """Match two independent geometry fits and separate bulk from distortion."""
    match = match_site_sets(
        np.asarray(centers_a, dtype=float),
        np.asarray(centers_b, dtype=float),
        max_distance_px=match_radius_px,
    )
    out = {"matching": match.summary()}
    if not match.pairs.size:
        out["bulk_shift"] = {}
        return out
    delta = centers_b[match.pairs[:, 1]] - centers_a[match.pairs[:, 0]]
    bulk = np.median(delta, axis=0)
    residual = np.linalg.norm(delta - bulk, axis=1)
    out["bulk_shift"] = {
        "median_dy_px": float(bulk[0]),
        "median_dx_px": float(bulk[1]),
        "magnitude_px": float(np.linalg.norm(bulk)),
        "residual_median_px": float(np.median(residual)),
        "residual_p90_px": float(np.percentile(residual, 90)),
        "residual_p99_px": float(np.percentile(residual, 99)),
        "residual_max_px": float(np.max(residual)),
    }
    return out


def lattice_step_explanation(
    center_shift_yx: Sequence[float],
    basis_vectors_yx: Sequence[Sequence[float]],
    *,
    max_index_shift: int = 2,
) -> dict[str, Any]:
    """Find the small integer lattice step closest to a cross-run shift."""
    shift = np.asarray(center_shift_yx, dtype=float)
    basis = np.asarray(basis_vectors_yx, dtype=float)
    if basis.shape != (2, 2):
        raise ValueError("basis_vectors_yx must contain two 2D vectors")
    rows = []
    for i in range(-max_index_shift, max_index_shift + 1):
        for j in range(-max_index_shift, max_index_shift + 1):
            candidate = i * basis[0] + j * basis[1]
            rows.append((float(np.linalg.norm(shift - candidate)), i, j, candidate))
    residual, i, j, candidate = min(rows, key=lambda r: r[0])
    return {
        "observed_shift_yx_px": [float(v) for v in shift],
        "nearest_index_shift": [int(i), int(j)],
        "nearest_lattice_vector_yx_px": [float(v) for v in candidate],
        "residual_px": residual,
        "consistent_with_one_basis_step": bool(abs(i) + abs(j) == 1 and residual < 1.0),
    }


def geometry_match_with_lattice_registration(
    centers_reference: np.ndarray,
    centers_candidate: np.ndarray,
    basis_vectors_yx: Sequence[Sequence[float]],
    *,
    match_radius_px: float = 4.0,
    max_index_shift: int = 3,
) -> dict[str, Any]:
    """Compare periodic lattices after resolving a small index-origin alias.

    Sparse fluorescence can identify pitch and tilt while leaving the
    row/column origin ambiguous by an integer lattice step.  The raw result is
    retained, then the candidate is translated over a bounded integer grid.
    Registration maximizes matched sites and only then minimizes residual
    distance; it cannot remove sub-pixel drift or non-lattice distortion.
    """
    reference = np.asarray(centers_reference, dtype=float)
    candidate = np.asarray(centers_candidate, dtype=float)
    basis = np.asarray(basis_vectors_yx, dtype=float)
    if basis.shape != (2, 2):
        raise ValueError("basis_vectors_yx must contain two 2D vectors")
    raw = geometry_match_report(
        reference, candidate, match_radius_px=match_radius_px
    )
    possibilities = []
    for row_shift in range(-max_index_shift, max_index_shift + 1):
        for col_shift in range(-max_index_shift, max_index_shift + 1):
            vector = row_shift * basis[0] + col_shift * basis[1]
            report = geometry_match_report(
                reference,
                candidate + vector,
                match_radius_px=match_radius_px,
            )
            matching = report["matching"]
            median = float(matching["median_px"])
            maximum = float(matching["max_px"])
            score = (
                -float(matching["n_matched"]),
                median if np.isfinite(median) else float("inf"),
                maximum if np.isfinite(maximum) else float("inf"),
                float(abs(row_shift) + abs(col_shift)),
            )
            possibilities.append(
                (score, row_shift, col_shift, vector, report)
            )
    _score, row_shift, col_shift, vector, registered = min(
        possibilities, key=lambda value: value[0]
    )
    return {
        "unregistered": raw,
        "registration": {
            "candidate_index_shift_applied": [int(row_shift), int(col_shift)],
            "candidate_translation_yx_px": [float(v) for v in vector],
            "integer_lattice_alias_required": bool(row_shift or col_shift),
            "max_index_shift_searched": int(max_index_shift),
        },
        "registered": registered,
    }


def refine_rigid_lattice_center(
    image: np.ndarray,
    reference_centers_yx: np.ndarray,
    *,
    bandpass_fn,
    max_shift_px: float = 2.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit only a bulk translation of a training-frozen lattice shape.

    Ten-shot endpoint subsets do not identify pitch, tilt and absolute index
    origin reliably.  This constrained diagnostic asks the narrower physical
    question needed for the gate: did the array translate relative to the
    training-frozen coordinates?  A deterministic coarse scan seeds a bounded
    continuous refinement.
    """
    centers = np.asarray(reference_centers_yx, dtype=float)
    band = np.asarray(bandpass_fn(np.asarray(image, dtype=float)), dtype=float)

    def objective(delta: Sequence[float]) -> float:
        moved = centers + np.asarray(delta, dtype=float)
        values = map_coordinates(
            band,
            [moved[:, 0], moved[:, 1]],
            order=3,
            mode="constant",
            cval=0.0,
        )
        return -float(np.sum(values))

    grid = np.linspace(-max_shift_px, max_shift_px, 17)
    starts = [(dy, dx) for dy in grid for dx in grid]
    initial = np.asarray(min(starts, key=objective), dtype=float)
    result = minimize(
        objective,
        initial,
        method="Powell",
        bounds=[(-max_shift_px, max_shift_px)] * 2,
        options={"xtol": 1e-4, "ftol": 1e-9, "maxiter": 500},
    )
    shift = np.asarray(result.x, dtype=float)
    boundary = bool(np.any(np.abs(shift) >= 0.98 * max_shift_px))
    return centers + shift, {
        "bulk_shift_yx_px": [float(v) for v in shift],
        "bulk_shift_magnitude_px": float(np.linalg.norm(shift)),
        "max_shift_bound_px": float(max_shift_px),
        "parameter_on_boundary": boundary,
        "converged": bool(result.success),
        "objective": float(result.fun),
    }


@dataclass(frozen=True)
class GeometryGate:
    """Predeclared coordinate-stability thresholds."""

    median_px: float = 0.5
    p90_px: float = 0.8
    p99_px: float = 1.2
    max_px: float = 1.5
    unmatched: int = 0

    def evaluate(self, report: Mapping[str, Any]) -> dict[str, Any]:
        m = report["matching"]
        failures = []
        limits = {
            "median_px": self.median_px,
            "p90_px": self.p90_px,
            "p99_px": self.p99_px,
            "max_px": self.max_px,
        }
        for key, limit in limits.items():
            if not np.isfinite(float(m[key])) or float(m[key]) > limit:
                failures.append(f"{key}={m[key]} exceeds {limit}")
        unmatched = int(m["n_unmatched_a"]) + int(m["n_unmatched_b"])
        if unmatched > self.unmatched:
            failures.append(f"{unmatched} unmatched sites exceeds {self.unmatched}")
        return {
            "passed": not failures,
            "thresholds": {**limits, "unmatched": self.unmatched},
            "failures": failures,
        }


def fit_geometry_subsets(
    cfg,
    shot_manifest: pd.DataFrame,
    *,
    gate: GeometryGate | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Independently refit geometry on train/time/condition subsets.

    Parameters
    ----------
    cfg:
        Repository :class:`~fluorescence_inference.config.Config`.
    shot_manifest:
        Exactly one row per shot, including ``shot_id``, ``shot_order``,
        ``split``, ``condition_id`` and ``sweep_value_s``.

    Returns
    -------
    report, context
        ``report`` is JSON-safe. ``context`` retains the fitted SiteMap objects
        for overlay figures and cross-run registration.
    """
    required = {
        "shot_id",
        "shot_order",
        "split",
        "condition_id",
        "sweep_value_s",
    }
    missing = required - set(shot_manifest.columns)
    if missing:
        raise ValueError(f"shot manifest is missing {sorted(missing)}")
    if shot_manifest["shot_id"].duplicated().any():
        raise ValueError("shot manifest must have one row per shot")

    from .dataset import (
        _frame_loader,
        accumulate_variance,
        build_configured_site_map,
        discover_shots,
        read_shot_meta,
    )

    load, loader_name = _frame_loader(cfg)
    from rydlab.atoms.grids import kmeans_2d

    paths = discover_shots(cfg)
    metas = [read_shot_meta(p, cfg, i) for i, p in enumerate(paths)]
    path_by_id = {int(m.shot_id): m.path for m in metas}
    unknown = sorted(set(shot_manifest["shot_id"].astype(int)) - set(path_by_id))
    if unknown:
        raise ValueError(f"shot manifest contains unknown shot ids: {unknown[:5]}")

    ordered = shot_manifest.sort_values("shot_order").reset_index(drop=True)
    train = ordered.loc[ordered["split"] == "train"]
    half = len(ordered) // 2
    train_half = max(1, len(train) // 2)
    sweep = ordered["sweep_value_s"].astype(float)
    lo, hi = float(sweep.min()), float(sweep.max())
    subset_rows = {
        "train": train,
        "all_unsupervised": ordered,
        "early": ordered.iloc[:half],
        "late": ordered.iloc[half:],
        "train_early": train.iloc[:train_half],
        "train_late": train.iloc[train_half:],
        "shortest_condition": ordered.loc[np.isclose(sweep, lo)],
        "longest_condition": ordered.loc[np.isclose(sweep, hi)],
    }
    hw = int(cfg["roi"]["trap_half_width"])
    fits: dict[str, Any] = {}
    mean_images: dict[str, np.ndarray] = {}
    failures: dict[str, str] = {}
    subset_constraints: dict[str, Any] | None = None
    for name, rows in subset_rows.items():
        subset_paths = [path_by_id[int(s)] for s in rows["shot_id"]]
        try:
            total, square, n_frame, _diag = accumulate_variance(
                subset_paths, cfg, load
            )
            mean_image = total / n_frame
            variance_image = variance_map(total, square, n_frame)
            mean_images[name] = mean_image
            settings = None
            if name != "train" and subset_constraints is not None:
                settings = deepcopy(cfg["sites"])
                settings.update(subset_constraints)
            fits[name] = build_configured_site_map(
                cfg,
                mean_image,
                variance_image,
                trap_half_width=hw,
                kmeans_2d=kmeans_2d,
                sites_cfg=settings,
            )
            if (
                name == "train"
                and str(cfg["sites"].get("method")) == "matched_template_average"
            ):
                grid = fits[name].grids[0]
                pitch = 0.5 * (
                    np.linalg.norm(grid.row_vector_yx)
                    + np.linalg.norm(grid.col_vector_yx)
                )
                tilt = float(
                    np.degrees(
                        np.arctan2(
                            grid.col_vector_yx[0],
                            grid.col_vector_yx[1],
                        )
                    )
                )
                subset_constraints = {
                    "matched_spacings_px": np.round(
                        np.arange(pitch - 0.45, pitch + 0.451, 0.05), 6
                    ).tolist(),
                    "matched_tilts_deg": np.round(
                        np.arange(tilt - 3.0, tilt + 3.001, 0.5), 6
                    ).tolist(),
                }
        except Exception as exc:  # an explicit failed gate, not a silent fallback
            failures[name] = f"{type(exc).__name__}: {exc}"

    comparisons = {
        "early_vs_late": ("early", "late"),
        "train_early_vs_train_late": ("train_early", "train_late"),
        "shortest_vs_longest": ("shortest_condition", "longest_condition"),
        "train_vs_all_unsupervised": ("train", "all_unsupervised"),
    }
    rigid_fits: dict[str, Any] = {}
    if "train" in fits:
        from rydlab.atoms.lattice_fit import bandpass

        for name, image in mean_images.items():
            centers, diagnostics = refine_rigid_lattice_center(
                image,
                fits["train"].centers_yx,
                bandpass_fn=bandpass,
                max_shift_px=2.0,
            )
            rigid_fits[name] = {
                "centers_yx": centers,
                "diagnostics": diagnostics,
            }
    gate = gate or GeometryGate()
    comparison_reports: dict[str, Any] = {}
    required_gate_names = {
        "early_vs_late",
        "train_early_vs_train_late",
        "shortest_vs_longest",
    }
    for label, (a, b) in comparisons.items():
        if a not in fits or b not in fits:
            comparison_reports[label] = {
                "available": False,
                "reason": failures.get(a) or failures.get(b) or "fit unavailable",
            }
            continue
        basis = [
            fits[a].grids[0].row_vector_yx,
            fits[a].grids[0].col_vector_yx,
        ]
        result = geometry_match_with_lattice_registration(
            fits[a].centers_yx,
            fits[b].centers_yx,
            basis,
        )
        result["available"] = True
        if a in rigid_fits and b in rigid_fits:
            rigid = geometry_match_report(
                rigid_fits[a]["centers_yx"],
                rigid_fits[b]["centers_yx"],
            )
            rigid["fit_a"] = rigid_fits[a]["diagnostics"]
            rigid["fit_b"] = rigid_fits[b]["diagnostics"]
            result["rigid_train_shape_registration"] = rigid
        gate_source = (
            "rigid_train_shape_registration"
            if label == "shortest_vs_longest"
            else "registered"
        )
        gate_report = result[gate_source]
        result["gate"] = gate.evaluate(gate_report)
        result["gate_source"] = gate_source
        if gate_source == "rigid_train_shape_registration":
            boundary = bool(
                gate_report["fit_a"]["parameter_on_boundary"]
                or gate_report["fit_b"]["parameter_on_boundary"]
            )
            converged = bool(
                gate_report["fit_a"]["converged"]
                and gate_report["fit_b"]["converged"]
            )
            if boundary:
                result["gate"]["passed"] = False
                result["gate"]["failures"].append(
                    "a rigid-translation parameter reached its bound"
                )
            if not converged:
                result["gate"]["passed"] = False
                result["gate"]["failures"].append(
                    "a rigid-translation fit did not converge"
                )
        result["interpretation"] = (
            "Free matched-template coordinates and integer index aliases are "
            "reported. Ten-shot endpoint gates use independently fitted bulk "
            "translations of the training-frozen pitch and tilt because the "
            "endpoint subsets do not identify the periodic index origin."
        )
        comparison_reports[label] = result

    fit_summaries = {
        name: {
            "n_shots": int(len(subset_rows[name])),
            "n_sites": int(site_map.n_sites),
            "n_detected": int(site_map.detected.sum()),
            "array_roi_x0x1y0y1": [int(v) for v in site_map.array_roi],
            "grids": [grid.summary() for grid in site_map.grids],
        }
        for name, site_map in fits.items()
    }
    required_failures = [
        name
        for name in required_gate_names
        if not comparison_reports.get(name, {}).get("available")
        or not comparison_reports[name]["gate"]["passed"]
    ]
    report = {
        "dataset_id": cfg.dataset_id,
        "loader": loader_name,
        "primary_geometry_subset": "train",
        "sensitivity_geometry_subset": "all_unsupervised",
        "subset_fits": fit_summaries,
        "fit_failures": failures,
        "rigid_train_shape_fits": {
            name: value["diagnostics"] for name, value in rigid_fits.items()
        },
        "subset_fit_constraints": (
            {
                "source": "derived from the complete training subset only",
                "purpose": (
                    "resolve sparse-subset lattice-index aliases without "
                    "tuning on validation or test performance"
                ),
                **(subset_constraints or {}),
            }
            if str(cfg["sites"].get("method")) == "matched_template_average"
            else None
        ),
        "comparisons": comparison_reports,
        "gate": {
            "passed": not required_failures,
            "required_comparisons": sorted(required_gate_names),
            "failed_comparisons": required_failures,
        },
    }
    return report, {
        "fits": fits,
        "subset_rows": subset_rows,
        "paths_by_shot_id": path_by_id,
        "load": load,
    }
