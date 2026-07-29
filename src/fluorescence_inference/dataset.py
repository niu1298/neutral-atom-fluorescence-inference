"""Build the standardized frame-site table from raw shot files.

Pipeline, deliberately two passes over the shots so that no more than one frame
is ever held in memory:

1. **Pass 1** streams every frame to accumulate per-pixel sum and sum of
   squares, and to collect per-shot metadata.
2. Site localisation runs on the resulting variance map (see :mod:`.sites`).
3. **Pass 2** streams the frames again and evaluates the ROI sums, background
   references and quality flags at the fitted site positions.

Image loading goes through the general lab package so that this repository does
not carry a second, silently diverging HDF5 reader.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import background_models as bm
from . import schema
from .background import annulus_background
from .config import Config, ensure_rydlab_importable
from .sites import (
    SiteMap,
    build_matched_template_site_map,
    build_site_map,
    variance_map,
)
from .splits import (
    CHRONOLOGICAL_STRATEGY,
    SEEDED_STRATEGY,
    build_cycle_split_manifest,
)


class SourceDataError(RuntimeError):
    """Raised when the raw shots are missing or contradict the config."""


@dataclass
class ShotMeta:
    path: Path
    shot_id: int
    shot_order: int
    timestamp: str | None
    sequence_index: int | None
    sequence_date: str | None
    script_basename: str | None
    frame_elapsed_s: dict[int, float]
    exposure_ms: dict[int, float]
    exposure_names: dict[int, str]
    n_runs: int | None
    globals_hash_input: dict[str, Any]


def discover_shots(cfg: Config) -> list[Path]:
    """Sorted shot files, or a clear failure."""
    d = cfg.shot_dir
    if not d.is_dir():
        raise SourceDataError(
            f"shot directory '{cfg['source']['shot_subdir']}' not found under the "
            f"configured experiment data root. Check configs/local.toml."
        )
    files = sorted(d.glob(cfg["source"]["file_glob"]))
    if not files:
        raise SourceDataError(
            f"no files matching '{cfg['source']['file_glob']}' in "
            f"'{cfg['source']['shot_subdir']}'"
        )
    return files


def read_shot_meta(path: Path, cfg: Config, order: int) -> ShotMeta:
    """Per-shot metadata. Missing values stay ``None``; nothing is invented."""
    import h5py

    h5cfg = cfg["source"]["h5"]
    frames = cfg.frame_specs
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        shot_id = attrs.get(h5cfg["run_number_attr"])
        if shot_id is None:
            raise SourceDataError(f"{path.name}: missing '{h5cfg['run_number_attr']}' attribute")

        exposures_path = h5cfg["exposures_dataset"]
        elapsed: dict[int, float] = {}
        exposure_ms: dict[int, float] = {}
        names: dict[int, str] = {}
        if exposures_path in f:
            rows = f[exposures_path][:]
            by_name = {}
            for r in rows:
                nm = r["name"]
                by_name[nm.decode() if isinstance(nm, bytes) else str(nm)] = r
            for spec in frames:
                r = by_name.get(spec["exposure_name"])
                if r is not None:
                    fid = int(spec["frame_id"])
                    elapsed[fid] = float(r["t"])
                    exposure_ms[fid] = float(r["trigger_duration"]) * 1e3
                    names[fid] = str(spec["exposure_name"])

        gl = {}
        gg = h5cfg.get("globals_group")
        if gg and gg in f:
            gl = {k: _py(v) for k, v in f[gg].attrs.items()}

        return ShotMeta(
            path=path,
            shot_id=int(shot_id),
            shot_order=order,
            timestamp=_str_or_none(attrs.get(h5cfg["run_time_attr"])),
            sequence_index=_int_or_none(attrs.get(h5cfg["sequence_index_attr"])),
            sequence_date=_str_or_none(attrs.get(h5cfg.get("sequence_date_attr", ""))),
            script_basename=_str_or_none(attrs.get(h5cfg["script_basename_attr"])),
            frame_elapsed_s=elapsed,
            exposure_ms=exposure_ms,
            exposure_names=names,
            n_runs=_int_or_none(attrs.get(h5cfg["n_runs_attr"])),
            globals_hash_input=gl,
        )


def _py(v: Any) -> Any:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return [_py(x) for x in v.tolist()]
    return v


def _str_or_none(v: Any) -> str | None:
    if v is None:
        return None
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)


def _int_or_none(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _float_or_none(v: Any) -> float | None:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _shot_design(
    cfg: Config, metas: list[ShotMeta]
) -> tuple[pd.DataFrame | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Build the exact condition/cycle/split map for a schema-V3 sweep."""
    if cfg.schema_version == schema.SCHEMA_VERSION:
        return None, None, None
    if cfg.schema_version != schema.V3_SCHEMA_VERSION:
        raise SourceDataError(f"unsupported configured schema version {cfg.schema_version!r}")

    sweep = cfg.get("sweep")
    split_cfg = cfg.get("split")
    if not isinstance(sweep, dict) or not isinstance(split_cfg, dict):
        raise SourceDataError("schema 3.0 configs require `sweep` and `split` mappings")

    global_name = str(sweep["global"])
    axis = str(sweep["axis"])
    expected_values = np.asarray(sweep["values_s"], dtype=float)
    if expected_values.ndim != 1 or len(expected_values) < 2:
        raise SourceDataError("sweep.values_s must contain at least two conditions")
    if len(np.unique(expected_values)) != len(expected_values):
        raise SourceDataError("sweep.values_s contains duplicate conditions")
    tolerance = float(sweep.get("value_tolerance_s", 1e-9))
    n_condition = len(expected_values)

    rows: list[dict[str, Any]] = []
    for meta in metas:
        value = _float_or_none(meta.globals_hash_input.get(global_name))
        if value is None:
            raise SourceDataError(
                f"{meta.path.name}: swept global {global_name!r} is absent or non-numeric")
        distances = np.abs(expected_values - value)
        condition_position = int(np.argmin(distances))
        if float(distances[condition_position]) > tolerance:
            raise SourceDataError(
                f"{meta.path.name}: {global_name}={value!r} does not match a configured "
                f"sweep value within {tolerance:g} s")
        cycle = int(meta.shot_order // n_condition)
        expected_position = int(meta.shot_order % n_condition)
        if bool(sweep.get("verify_cycle_order", True)) and (
                condition_position != expected_position):
            raise SourceDataError(
                f"{meta.path.name}: condition position {condition_position} contradicts "
                f"cyclic acquisition position {expected_position}")
        canonical = float(expected_values[condition_position])
        rows.append({
            "shot_id": meta.shot_id,
            "shot_order": meta.shot_order,
            "condition_id": f"{axis}:{canonical:.9g}",
            "sweep_value_s": canonical,
            # REPEAT is fixed at zero in these files.  This index is explicitly
            # inferred from cyclic acquisition order, not copied from REPEAT.
            "repetition_index": cycle,
            "cycle_index": cycle,
        })
    design = pd.DataFrame(rows)

    strategy = str(split_cfg.get("strategy", CHRONOLOGICAL_STRATEGY))
    seed = int(split_cfg.get("seed", 0))
    primary, primary_meta = build_cycle_split_manifest(
        design, strategy=strategy, seed=seed)

    sensitivity_strategy = str(
        split_cfg.get("sensitivity_strategy", SEEDED_STRATEGY))
    sensitivity_seed = int(split_cfg.get("sensitivity_seed", 20260728))
    sensitivity, sensitivity_meta = build_cycle_split_manifest(
        design, strategy=sensitivity_strategy, seed=sensitivity_seed)
    sensitivity = sensitivity.rename(columns={"split": "sensitivity_split"})
    combined = primary.merge(
        sensitivity[["shot_id", "sensitivity_split"]],
        on="shot_id", how="left", validate="one_to_one")
    return combined, primary_meta, sensitivity_meta


def select_fit_metas(
    metas: list[ShotMeta],
    split_manifest: pd.DataFrame | None,
    *,
    fit_scope: str,
) -> list[ShotMeta]:
    """Select only the shots authorized to fit geometry/background parameters."""
    if fit_scope == "all":
        return list(metas)
    if fit_scope != "train":
        raise SourceDataError("extraction.fit_scope must be `all` or `train`")
    if split_manifest is None:
        raise SourceDataError("train-only extraction requires a shot split")
    train_ids = set(split_manifest.loc[
        split_manifest["split"] == "train", "shot_id"].astype(int))
    selected = [meta for meta in metas if meta.shot_id in train_ids]
    if not selected:
        raise SourceDataError("train-only extraction selected no shots")
    return selected


def _validate_shot_metadata(cfg: Config, metas: list[ShotMeta]) -> None:
    """Fail before fitting if names, timing, or sequence identity disagree."""
    expected_ids = {int(s["frame_id"]) for s in cfg.frame_specs}
    expected_names = {
        int(s["frame_id"]): str(s["exposure_name"]) for s in cfg.frame_specs}
    exposure_expected = float(cfg["source"]["expected_exposure_s"]) * 1e3
    sequence_expected = cfg["source"].get("expected_sequence_index")
    date_expected = cfg.get("date")
    for meta in metas:
        if set(meta.exposure_names) != expected_ids:
            raise SourceDataError(
                f"{meta.path.name}: expected frame ids {sorted(expected_ids)}, "
                f"found {sorted(meta.exposure_names)}")
        if meta.exposure_names != expected_names:
            raise SourceDataError(
                f"{meta.path.name}: exposure-name mapping {meta.exposure_names} "
                f"does not match {expected_names}")
        bad_exposure = {
            fid: value for fid, value in meta.exposure_ms.items()
            if not np.isclose(value, exposure_expected, rtol=0, atol=1e-9)
        }
        if bad_exposure:
            raise SourceDataError(
                f"{meta.path.name}: commanded exposure mismatch {bad_exposure}")
        starts = [meta.frame_elapsed_s[fid] for fid in sorted(expected_ids)]
        if any(b <= a for a, b in zip(starts, starts[1:])):
            raise SourceDataError(f"{meta.path.name}: frame starts are not strictly ordered")
        if sequence_expected is not None and meta.sequence_index != int(sequence_expected):
            raise SourceDataError(
                f"{meta.path.name}: sequence_index={meta.sequence_index}, "
                f"expected {sequence_expected}")
        if date_expected is not None and meta.sequence_date != str(date_expected):
            raise SourceDataError(
                f"{meta.path.name}: sequence_date={meta.sequence_date!r}, "
                f"expected {date_expected!r}")


def _frame_loader(cfg: Config):
    """The lab package's HDF5 frame reader, or a minimal equivalent."""
    try:
        ensure_rydlab_importable(cfg)
        from rydlab.io.legacy_loaders import load_h5_frames

        def load(path: Path, h5_path: str) -> np.ndarray:
            frames = load_h5_frames(str(path), h5_path)
            if frames.shape[0] != 1:
                raise SourceDataError(
                    f"{path.name}:{h5_path} holds {frames.shape[0]} frames; "
                    f"this dataset assumes one image per exposure name"
                )
            return frames[0]

        return load, "rydlab.io.legacy_loaders.load_h5_frames"
    except Exception as exc:  # noqa: BLE001 - fall back but say so loudly
        raise SourceDataError(
            "the general lab analysis package could not be imported, so raw "
            "images cannot be read with the same loader the lab uses. Set "
            "paths.tweezer_analysis_src in configs/local.toml. "
            f"Underlying error: {exc}"
        ) from exc


def accumulate_variance(shots: Iterable[Path], cfg: Config, load
                        ) -> tuple[np.ndarray, np.ndarray, int, dict[str, Any]]:
    """Pass 1: streaming sum / sum-of-squares plus whole-frame diagnostics."""
    frames_cfg = {int(s["frame_id"]): s["h5_path"] for s in cfg.frame_specs}
    use = [int(f) for f in cfg["sites"]["variance_frames"]]
    total = sq = None
    n = 0
    shape: tuple[int, int] | None = None
    diag: dict[str, Any] = {"frame_median": {}, "frame_max": {}, "frame_shape": {}}
    for p in shots:
        for fid, h5_path in frames_cfg.items():
            img = load(p, h5_path)
            if shape is None:
                shape = img.shape
                total = np.zeros(shape, dtype=np.float64)
                sq = np.zeros(shape, dtype=np.float64)
            elif img.shape != shape:
                raise SourceDataError(
                    f"{p.name}: frame {fid} has shape {img.shape}, expected {shape}"
                )
            diag["frame_median"].setdefault(fid, []).append(float(np.median(img)))
            diag["frame_max"].setdefault(fid, []).append(float(img.max()))
            diag["frame_shape"][fid] = list(img.shape)
            if fid in use:
                total += img
                sq += img * img
                n += 1
    if total is None or n < 2:
        raise SourceDataError("not enough frames to build a variance map")
    return total, sq, n, diag


def build_configured_site_map(
    cfg: Config,
    mean_img: np.ndarray,
    var_img: np.ndarray,
    *,
    trap_half_width: int,
    kmeans_2d,
    sites_cfg: dict[str, Any] | None = None,
) -> SiteMap:
    """Dispatch geometry without changing the validated V0 default."""
    settings = cfg["sites"] if sites_cfg is None else sites_cfg
    method = str(settings.get("method", "variance_lattice"))
    if method == "variance_lattice":
        return build_site_map(
            var_img,
            settings,
            trap_half_width=trap_half_width,
            kmeans_2d=kmeans_2d,
        )
    if method == "matched_template_average":
        from rydlab.atoms.lattice_fit import fit_lattices

        return build_matched_template_site_map(
            mean_img,
            settings,
            trap_half_width=trap_half_width,
            fit_lattices=fit_lattices,
        )
    raise SourceDataError(f"unsupported sites.method {method!r}")


def build_frame_site_table(cfg: Config, *, progress: bool = True
                           ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build the standardized table, the site table and a metadata block."""
    shots = discover_shots(cfg)
    expected = cfg["source"].get("expected_n_shots")
    if expected is not None and len(shots) != int(expected):
        raise SourceDataError(
            f"expected {expected} shots, found {len(shots)}. Refusing to build a "
            f"dataset that does not match the configured acquisition."
        )

    load, loader_name = _frame_loader(cfg)
    ensure_rydlab_importable(cfg)
    from rydlab.atoms.grids import kmeans_2d

    metas = [read_shot_meta(p, cfg, order=i) for i, p in enumerate(shots)]
    _validate_shot_metadata(cfg, metas)
    split_manifest, split_meta, split_sensitivity_meta = _shot_design(cfg, metas)

    fit_scope = str(cfg.get("extraction", {}).get("fit_scope", "all"))
    fit_metas = select_fit_metas(
        metas, split_manifest, fit_scope=fit_scope)
    fit_shots = [m.path for m in fit_metas]

    total, sq, n_var, diag = accumulate_variance(fit_shots, cfg, load)
    var_img = variance_map(total, sq, n_var)
    mean_img = total / n_var

    roi_cfg = cfg["roi"]
    site_map = build_configured_site_map(
        cfg,
        mean_img,
        var_img,
        trap_half_width=int(roi_cfg["trap_half_width"]),
        kmeans_2d=kmeans_2d,
    )

    bg_cfg = cfg["background"]
    mask = bm.build_site_mask(
        var_img.shape, site_map.centers_yx,
        radius_px=float(bg_cfg["mask_radius_px"]),
        reference_image=mean_img,
        hot_pixel_sigma=float(bg_cfg["hot_pixel_sigma"]),
    )
    template = bm.build_fixed_template(
        mean_img, mask, degree=int(bg_cfg["surface_degree"]),
        max_fit_pixels=int(bg_cfg["max_fit_pixels"]),
    )

    rows, frame_diag = _extract_rows(
        cfg, metas, site_map, mask, template, load,
        split_manifest=split_manifest, progress=progress)
    df = schema.coerce(pd.DataFrame(rows), version=cfg.schema_version)

    sites_df = pd.DataFrame({
        "site_id": site_map.site_id,
        "grid": site_map.grid_name,
        "site_row": site_map.row_index,
        "site_col": site_map.col_index,
        "site_y": site_map.centers_yx[:, 0],
        "site_x": site_map.centers_yx[:, 1],
        "roi_y0": [b[0] for b in site_map.boxes],
        "roi_y1": [b[1] for b in site_map.boxes],
        "roi_x0": [b[2] for b in site_map.boxes],
        "roi_x1": [b[3] for b in site_map.boxes],
        "site_detected": site_map.detected,
        "fit_residual_px": np.concatenate([g.residual_px for g in site_map.grids]),
    })

    meta: dict[str, Any] = {
        "schema_version": cfg.schema_version,
        "geometry_version": cfg["sites"].get("geometry_version", "unversioned"),
        "loader": loader_name,
        "n_shots": len(shots),
        "n_frames_per_shot": cfg.n_frames,
        "frame_shape": list(var_img.shape),
        "variance_frames_used": int(n_var),
        "fit_scope": fit_scope,
        "fit_shot_ids": [m.shot_id for m in fit_metas],
        "sites": site_map.summary(),
        "background": {
            "primary_method": str(bg_cfg["primary_method"]),
            "methods": {k: {"background_column": b, "corrected_column": c}
                        for k, (b, c) in schema.METHOD_COLUMNS.items()},
            "mask": mask.summary(),
            "surface_degree": int(bg_cfg["surface_degree"]),
            "max_fit_pixels": int(bg_cfg["max_fit_pixels"]),
            "huber_k": float(bg_cfg["huber_k"]),
            "irls_iterations": int(bg_cfg["irls_iterations"]),
            "template_fit": template.fit_info,
            "annulus_is_diagnostic_only": True,
        },
        "shot_metadata": _metadata_consistency(cfg, metas),
        "whole_frame_diagnostics": _frame_diagnostics(diag),
        "frame_background_diagnostics": frame_diag,
    }
    run_meta = _run_level_metadata(
        cfg, metas, split_meta=split_meta,
        sensitivity_split_meta=split_sensitivity_meta)
    if split_meta is not None:
        meta["split"] = split_meta
        meta["split_sensitivity"] = split_sensitivity_meta
    return df, sites_df, {"meta": meta, "mean_image": mean_img,
                          "variance_image": var_img, "site_map": site_map,
                          "mask": mask, "template": template,
                          "shots": shots, "metas": metas,
                          "fit_shots": fit_shots,
                          "split_manifest": split_manifest,
                          "run_metadata": run_meta}


def _extract_rows(cfg: Config, metas: list[ShotMeta], site_map: SiteMap,
                  mask: bm.SiteMask, template: bm.FixedTemplate, load, *,
                  split_manifest: pd.DataFrame | None,
                  progress: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rows plus one per-frame background diagnostic record.

    All four background methods are evaluated for every ROI. None overwrites
    another: the method actually used downstream is a configuration choice made
    on the evidence in ``reports/validation/background_method_comparison.json``,
    and can be revisited without re-reading a single image.
    """
    roi_cfg = cfg["roi"]
    q_cfg = cfg["quality"]
    src = cfg["source"]
    bg_cfg = cfg["background"]
    ann_cfg = cfg.get("legacy_annulus", {}) or {}

    offset = float(roi_cfg["camera_offset_counts_per_pixel"])
    saturation = float(src["saturation_adu"])
    edge_margin = int(q_cfg["roi_edge_margin_px"])
    min_bg_px = int(q_cfg["local_bg_min_pixels"])
    hw = int(roi_cfg["trap_half_width"])
    full_box_px = (2 * hw + 1) ** 2
    degree = int(bg_cfg["surface_degree"])
    max_fit = int(bg_cfg["max_fit_pixels"])
    block = int(bg_cfg["residual_block_px"])
    primary = str(bg_cfg["primary_method"])
    _, primary_col = schema.METHOD_COLUMNS[primary]
    frame_specs = {int(s["frame_id"]): s for s in cfg.frame_specs}
    frames_cfg = {fid: spec["h5_path"] for fid, spec in frame_specs.items()}
    split_lookup: dict[int, dict[str, Any]] = {}
    if split_manifest is not None:
        split_lookup = {
            int(row["shot_id"]): row
            for row in split_manifest.to_dict(orient="records")
        }

    iterator: Iterable[ShotMeta] = metas
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(metas, desc="extracting frame-site rows")
        except ImportError:
            pass

    rows: list[dict[str, Any]] = []
    frame_diag: list[dict[str, Any]] = []
    for m in iterator:
        for fid, h5_path in frames_cfg.items():
            raw = load(m.path, h5_path)
            img = raw.astype(float) - offset

            bg = bm.evaluate_frame(img, mask, template, degree=degree,
                                   max_fit_pixels=max_fit)
            b_global = bm.roi_sums(np.full(img.shape, bg.global_level), site_map.boxes)
            b_spatial = bm.roi_sums(bg.spatial, site_map.boxes)
            b_fixed = bm.roi_sums(bg.fixed_offset, site_map.boxes)

            frame_diag.append({
                "shot_order": m.shot_order, "frame_id": fid,
                "global_level_per_px": bg.global_level,
                "fixed_offset_per_px": bg.offset,
                "residual_rms_global": bg.residual_rms_global,
                "residual_rms_spatial": bg.residual_rms_spatial,
                "residual_rms_fixed": bg.residual_rms_fixed,
                "spatial_fit_residual_rms": bg.spatial_fit["residual_rms"],
                "structure_global": bm.residual_spatial_structure(
                    img, bg.global_level, mask, block=block),
                "structure_spatial": bm.residual_spatial_structure(
                    img, bg.spatial, mask, block=block),
                "structure_fixed": bm.residual_spatial_structure(
                    img, bg.fixed_offset, mask, block=block),
            })

            for k, box in enumerate(site_map.boxes):
                y0, y1, x0, x1 = box
                patch = raw[y0:y1, x0:x1]
                n_px = int((y1 - y0) * (x1 - x0))
                roi_sum = float(np.sum(img[y0:y1, x0:x1]))
                roi_max = float(patch.max()) if patch.size else float("nan")

                ann = ann_d = float("nan")
                n_bg = min_bg_px
                if ann_cfg.get("enabled", False):
                    ann, ann_d, n_bg = annulus_background(
                        img, box,
                        inner_half_width=int(ann_cfg["inner_half_width"]),
                        outer_half_width=int(ann_cfg["outer_half_width"]),
                        stat=str(ann_cfg["stat"]))

                values = {
                    "roi_sum_raw": roi_sum,
                    "background_global": float(b_global[k]),
                    "count_corrected_global": roi_sum - float(b_global[k]),
                    "background_spatial": float(b_spatial[k]),
                    "count_corrected_spatial": roi_sum - float(b_spatial[k]),
                    "background_fixed_offset": float(b_fixed[k]),
                    "count_corrected_fixed_offset": roi_sum - float(b_fixed[k]),
                    "background_annulus_contaminated": ann,
                    "count_corrected_annulus_contaminated": roi_sum - ann,
                    "background_annulus_density_contaminated": ann_d,
                }

                flags: list[str] = []
                if roi_max >= saturation:
                    flags.append("roi_saturated")
                if n_px != full_box_px or _touches_edge(box, img.shape, edge_margin):
                    flags.append("roi_touches_edge")
                if ann_cfg.get("enabled", False) and n_bg < min_bg_px:
                    flags.append("local_bg_underdetermined")
                if not all(np.isfinite(values[c]) for c in
                           ("roi_sum_raw", "background_global", "background_spatial",
                            "background_fixed_offset")):
                    flags.append("nonfinite_count")
                if not site_map.detected[k]:
                    flags.append("site_not_detected")

                rows.append({
                    "run_id": cfg.run_id,
                    "shot_id": m.shot_id,
                    "shot_order": m.shot_order,
                    "frame_id": fid,
                    "site_id": int(site_map.site_id[k]),
                    "timestamp": m.timestamp,
                    "exposure_ms": m.exposure_ms.get(fid),
                    "frame_elapsed_s": m.frame_elapsed_s.get(fid),
                    "site_x": float(site_map.centers_yx[k, 1]),
                    "site_y": float(site_map.centers_yx[k, 0]),
                    **values,
                    "background_corrected_count": values[primary_col],
                    "raw_image_path": m.path.name,
                    "quality_flag": "|".join(flags) if flags else "ok",
                    "grid": site_map.grid_name[k],
                    "site_row": int(site_map.row_index[k]),
                    "site_col": int(site_map.col_index[k]),
                    "roi_n_pixels": n_px,
                    "roi_max_pixel": roi_max,
                    "site_detected": bool(site_map.detected[k]),
                    **_v3_row_fields(
                        cfg, m, fid, frame_specs[fid], split_lookup.get(m.shot_id),
                        grid_id=str(site_map.grid_name[k]), values=values),
                })
    return rows, frame_diag


def _v3_row_fields(
    cfg: Config,
    meta: ShotMeta,
    frame_id: int,
    frame_spec: dict[str, Any],
    design: dict[str, Any] | None,
    *,
    grid_id: str,
    values: dict[str, float],
) -> dict[str, Any]:
    """Add schema-V3 identifiers and commanded timing without inventing readback."""
    if cfg.schema_version == schema.SCHEMA_VERSION:
        return {}
    if design is None:
        raise SourceDataError(f"shot {meta.shot_id} is absent from the split manifest")

    exposure_s = (
        meta.exposure_ms.get(frame_id) / 1e3
        if meta.exposure_ms.get(frame_id) is not None else None)
    frame_ids = sorted(meta.frame_elapsed_s)
    position = frame_ids.index(frame_id)
    gap: float | None = None
    if position > 0:
        previous = frame_ids[position - 1]
        previous_exposure = meta.exposure_ms.get(previous)
        if previous_exposure is not None:
            gap = (
                meta.frame_elapsed_s[frame_id]
                - meta.frame_elapsed_s[previous]
                - previous_exposure / 1e3
            )
            if abs(gap) < 1e-12:
                gap = 0.0

    wait_global = str(cfg.get("timing", {}).get(
        "bright_wait_global", "WAIT_BEFORE_FIRST_FLUOR"))
    bright_wait = _float_or_none(meta.globals_hash_input.get(wait_global))
    hardware = cfg.get("hardware", {}) or {}
    return {
        "dataset_id": cfg.dataset_id,
        "date": str(cfg["date"]),
        "sequence_id": str(cfg["sequence_id"]),
        "sequence_type": cfg.sequence_type,
        "repetition_index": int(design["repetition_index"]),
        "cycle_index": int(design["cycle_index"]),
        "condition_id": str(design["condition_id"]),
        "split": str(design["split"]),
        "sweep_axis": str(cfg["sweep"]["axis"]),
        "sweep_value_s": float(design["sweep_value_s"]),
        "frame_index": int(frame_id),
        "frame_name": str(frame_spec["exposure_name"]),
        "frame_start_s": meta.frame_elapsed_s.get(frame_id),
        "exposure_s": exposure_s,
        "interframe_gap_s": gap,
        # The switch-off gap preceding a later fluorescence frame.  For the
        # bright-wait run this is the fixed 10 ms control gap.
        "dark_hold_s": gap,
        "bright_wait_before_first_s": bright_wait,
        # The files store commands and triggers but no optical-power readback.
        "actual_light_on_s": None,
        "switch_state": hardware.get("commanded_switch_state_during_exposure"),
        # Multiple DDS channels are active.  A scalar would be misleading;
        # channel-level commands live in the run sidecar.
        "dds_frequency": None,
        "dds_amplitude": None,
        "grid_id": grid_id,
        "background_template_offset": values["background_fixed_offset"],
        "count_corrected_template": values["count_corrected_fixed_offset"],
    }


def _touches_edge(box: tuple[int, int, int, int], shape: tuple[int, int],
                  margin: int) -> bool:
    y0, y1, x0, x1 = box
    h, w = shape
    return y0 < margin or x0 < margin or y1 > h - margin or x1 > w - margin


def _metadata_consistency(cfg: Config, metas: list[ShotMeta]) -> dict[str, Any]:
    """Cross-check the per-shot metadata against the configured expectations."""
    src = cfg["source"]
    exp_exposure_ms = float(src["expected_exposure_s"]) * 1e3
    exp_delta_raw = src.get("expected_interframe_start_delta_s")
    exp_delta = float(exp_delta_raw) if exp_delta_raw is not None else None

    exposures = sorted({round(v, 9) for m in metas for v in m.exposure_ms.values()})
    starts = {fid: sorted({round(m.frame_elapsed_s[fid], 9)
                           for m in metas if fid in m.frame_elapsed_s})
              for fid in sorted({fid for m in metas for fid in m.frame_elapsed_s})}
    deltas = sorted({
        round(m.frame_elapsed_s[f1] - m.frame_elapsed_s[f0], 9)
        for m in metas
        for f0, f1 in zip(sorted(m.frame_elapsed_s), sorted(m.frame_elapsed_s)[1:])
    })
    gaps = sorted({
        round(
            m.frame_elapsed_s[f1] - m.frame_elapsed_s[f0]
            - m.exposure_ms[f0] / 1e3,
            9,
        )
        for m in metas
        for f0, f1 in zip(sorted(m.frame_elapsed_s), sorted(m.frame_elapsed_s)[1:])
        if f0 in m.exposure_ms
    })

    return {
        "exposure_ms_values": exposures,
        "exposure_matches_config": exposures == [exp_exposure_ms],
        "frame_start_s_values": {str(k): v for k, v in starts.items()},
        "interframe_start_delta_s_values": deltas,
        "interframe_gap_s_values": gaps,
        "interframe_delta_matches_config": (
            None if exp_delta is None else deltas == [round(exp_delta, 9)]),
        "shot_ids": [m.shot_id for m in metas],
        "shot_ids_unique": len({m.shot_id for m in metas}) == len(metas),
        "timestamps_present": sum(m.timestamp is not None for m in metas),
        "timestamp_first": metas[0].timestamp if metas else None,
        "timestamp_last": metas[-1].timestamp if metas else None,
        "n_runs_attr_values": sorted({m.n_runs for m in metas if m.n_runs is not None}),
        "script_basenames": sorted({m.script_basename for m in metas
                                    if m.script_basename}),
        "sequence_indices": sorted({m.sequence_index for m in metas
                                    if m.sequence_index is not None}),
        "sequence_dates": sorted({m.sequence_date for m in metas
                                  if m.sequence_date is not None}),
        "varying_globals": _varying_globals(metas),
    }


def _varying_globals(metas: list[ShotMeta]) -> dict[str, int]:
    """Names of globals that change across shots, with their cardinality.

    Names only, never values: a global name is a physics parameter label, a
    global value can encode unpublished apparatus settings.
    """
    keys = sorted({k for m in metas for k in m.globals_hash_input})
    out: dict[str, int] = {}
    for k in keys:
        vals = {repr(m.globals_hash_input.get(k)) for m in metas}
        if len(vals) > 1:
            out[k] = len(vals)
    return out


def _run_level_metadata(
    cfg: Config,
    metas: list[ShotMeta],
    *,
    split_meta: dict[str, Any] | None,
    sensitivity_split_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    """Non-repeated run facts for the ignored machine-readable sidecar."""
    public_cfg = cfg.get("run_metadata", {}) or {}
    public_names = [str(v) for v in public_cfg.get("public_globals", [])]
    public_globals: dict[str, list[Any]] = {}
    for name in public_names:
        values = {_stable_scalar(m.globals_hash_input.get(name)) for m in metas}
        public_globals[name] = sorted(values, key=lambda value: repr(value))

    frame_names = {
        str(int(spec["frame_id"])): str(spec["exposure_name"])
        for spec in cfg.frame_specs
    }
    return {
        "schema_version": cfg.schema_version,
        "dataset_id": cfg.dataset_id,
        "run_id": cfg.run_id,
        "date": cfg.get("date"),
        "sequence_id": cfg.get("sequence_id"),
        "sequence_type": cfg.sequence_type,
        "n_shots": len(metas),
        "frame_names": frame_names,
        "public_globals": public_globals,
        "varying_globals": _varying_globals(metas),
        "timing": {
            "source": "commanded hamamatsu EXPOSURES table in each shot",
            "frame_starts_are_hardware_timestamps": False,
            "measured_exposure_s": None,
            "per_frame_hardware_timestamp": None,
            "leaf_metadata_conflict": public_cfg.get("leaf_metadata_conflict"),
        },
        "hardware": cfg.get("hardware", {}),
        "field_provenance": {
            "shot_id": "direct root HDF5 attribute",
            "timestamp": "direct root HDF5 attribute; timezone absent",
            "frame_name": "direct devices/hamamatsu/EXPOSURES name",
            "frame_start_s": "direct commanded EXPOSURES time",
            "exposure_s": "direct commanded EXPOSURES trigger_duration",
            "condition": "direct evaluated swept global, matched to tracked config",
            "cycle_index": "inferred from verified repeated cyclic acquisition order",
            "repetition_index": "inferred cycle_index; stored REPEAT is fixed at zero",
            "interframe_gap_s": "derived start-to-start minus preceding exposure",
            "switch_state": "compiled-command audit summarized in tracked config",
            "actual_light_on_s": "unrecoverable; no optical readback",
            "dds_frequency": "row scalar omitted; multiple channel commands",
            "dds_amplitude": "row scalar omitted; multiple channel commands",
            "labscript_commit": "unrecoverable; no source-repository revision in HDF5",
        },
        "split": split_meta,
        "split_sensitivity": sensitivity_split_meta,
    }


def _stable_scalar(value: Any) -> Any:
    """Hashable JSON-safe scalar for run-level public-global summaries."""
    value = _py(value)
    if isinstance(value, (list, dict)):
        import json

        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return value


def _frame_diagnostics(diag: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"frame_shape": diag["frame_shape"]}
    for key in ("frame_median", "frame_max"):
        out[key] = {
            str(fid): {
                "mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)),
                "min": float(np.min(v)), "max": float(np.max(v)),
            }
            for fid, v in diag[key].items()
        }
    med = diag["frame_median"]
    if {0, 1} <= set(med):
        d = np.asarray(med[1], float) - np.asarray(med[0], float)
        out["frame1_minus_frame0_median_shift"] = {
            "mean": float(d.mean()), "std": float(d.std(ddof=1)),
            "min": float(d.min()), "max": float(d.max()),
        }
    return out


def write_dataset(cfg: Config, df: pd.DataFrame, sites_df: pd.DataFrame,
                  meta: dict[str, Any], *,
                  run_metadata: dict[str, Any] | None = None,
                  split_manifest: pd.DataFrame | None = None) -> dict[str, Path]:
    """Write tables plus dataset, run, and split metadata sidecars."""
    import json

    out_dir = cfg.paths.processed_root
    out_dir.mkdir(parents=True, exist_ok=True)
    table = out_dir / f"{cfg.dataset_id}.parquet"
    sites = out_dir / f"{cfg.dataset_id}.sites.parquet"
    meta_path = out_dir / f"{cfg.dataset_id}.meta.json"

    df.to_parquet(table, index=False)
    sites_df.to_parquet(sites, index=False)
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    paths = {"table": table, "sites": sites, "meta": meta_path}
    if run_metadata is not None:
        run_path = out_dir / f"{cfg.dataset_id}.run.json"
        run_path.write_text(
            json.dumps(run_metadata, indent=2, default=str), encoding="utf-8")
        paths["run_meta"] = run_path
    if split_manifest is not None:
        split_path = out_dir / f"{cfg.dataset_id}.splits.parquet"
        split_manifest.to_parquet(split_path, index=False)
        paths["splits"] = split_path
    return paths


def load_dataset(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Read back the processed dataset, or fail with an actionable message."""
    import json

    table = cfg.paths.processed_root / f"{cfg.dataset_id}.parquet"
    sites = cfg.paths.processed_root / f"{cfg.dataset_id}.sites.parquet"
    meta_path = cfg.paths.processed_root / f"{cfg.dataset_id}.meta.json"
    missing = [p.name for p in (table, sites, meta_path) if not p.exists()]
    if missing:
        raise SourceDataError(
            "processed dataset is not available (missing: "
            + ", ".join(missing)
            + "). Run: python scripts/export_processed_dataset.py "
              "--config configs/paired_100ms.yaml"
        )
    return (pd.read_parquet(table), pd.read_parquet(sites),
            json.loads(meta_path.read_text(encoding="utf-8")))


def load_run_metadata(cfg: Config) -> dict[str, Any]:
    """Load the schema-V3 run sidecar without changing the V0 load API."""
    import json

    path = cfg.paths.processed_root / f"{cfg.dataset_id}.run.json"
    if not path.exists():
        raise SourceDataError(
            f"run metadata is not available (missing: {path.name}). "
            f"Run export_processed_dataset.py with {cfg.config_path.name}."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_split_manifest(cfg: Config) -> pd.DataFrame:
    """Load the one-row-per-shot split manifest."""
    path = cfg.paths.processed_root / f"{cfg.dataset_id}.splits.parquet"
    if not path.exists():
        raise SourceDataError(
            f"split manifest is not available (missing: {path.name}). "
            f"Run export_processed_dataset.py with {cfg.config_path.name}."
        )
    return pd.read_parquet(path)
