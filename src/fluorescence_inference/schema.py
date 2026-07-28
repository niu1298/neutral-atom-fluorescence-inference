"""The standardized frame-site long table: columns, dtypes, validation.

This schema is the contract between the general lab analysis package and this
one. Everything downstream reads the table, never the raw HDF5 files.

One row = one (shot, frame, site) observation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

PRIMARY_KEY = ("run_id", "shot_id", "frame_id", "site_id")

#: Bumped whenever the column set or the meaning of a column changes.
#: 1.0 — annulus background as the primary correction.
#: 2.0 — annulus demoted to a diagnostic; four background methods carried in
#:       separate columns; `roi_sum` renamed `roi_sum_raw`.
SCHEMA_VERSION = "2.0"

#: v1 column -> v2 column, for reading an older table.
V1_TO_V2 = {
    "roi_sum": "roi_sum_raw",
    "local_background": "background_annulus_contaminated",
    "global_background": "background_global",
    "background_corrected_count": "count_corrected_annulus_contaminated",
    "common_mode_corrected_count": "count_corrected_global",
    "local_background_density": "background_annulus_density_contaminated",
}

#: column -> (pandas dtype, nullable, description)
FRAME_SITE_COLUMNS: dict[str, tuple[str, bool, str]] = {
    "run_id":       ("string",  False, "Sequence identifier, one acquisition run."),
    "shot_id":      ("Int32",   False, "Shot index within the run (camera run number)."),
    "shot_order":   ("Int32",   False, "Rank of the shot in acquisition time order, 0-based."),
    "frame_id":     ("Int8",    False, "0 = first exposure, 1 = second exposure."),
    "site_id":      ("Int32",   False, "Site index, stable across shots and frames."),
    "timestamp":    ("string",  True,  "Shot acquisition timestamp, ISO-like, from the shot file."),
    "exposure_ms":  ("Float64", True,  "Exposure of THIS frame, milliseconds."),
    "frame_elapsed_s": ("Float64", True, "Exposure start of this frame, seconds from sequence t=0."),
    "site_x":       ("Float64", False, "Site centre column in full-frame pixel coordinates."),
    "site_y":       ("Float64", False, "Site centre row in full-frame pixel coordinates."),

    # ---- method A: no background correction
    "roi_sum_raw":  ("Float64", True,  "Method A: raw sum over the ROI box, camera counts."),

    # ---- method B: global site-free median of the same frame
    "background_global": ("Float64", True,
                          "Method B reference: site-free median of this frame x ROI pixels."),
    "count_corrected_global": ("Float64", True, "Method B: roi_sum_raw - background_global."),

    # ---- method C: robust smooth spatial surface, refitted per frame
    "background_spatial": ("Float64", True,
                           "Method C reference: robust polynomial surface integrated over the ROI."),
    "count_corrected_spatial": ("Float64", True, "Method C: roi_sum_raw - background_spatial."),

    "raw_image_path": ("string", True,  "Shot file NAME only; never an absolute path."),
    "quality_flag": ("string",  False, "'ok' or a '|'-joined list of flag names."),
}

#: extra columns this project carries beyond the required contract
OPTIONAL_COLUMNS: dict[str, tuple[str, bool, str]] = {
    # ---- method D: fixed spatial template plus a per-frame common-mode offset
    "background_fixed_offset": ("Float64", True,
                                "Method D reference: fixed template + frame offset, over the ROI."),
    "count_corrected_fixed_offset": ("Float64", True,
                                     "Method D: roi_sum_raw - background_fixed_offset."),

    # ---- the primary alias, so downstream code has one canonical column.
    # Which method it copies is recorded in the metadata sidecar under
    # `background.primary_method`; it is never a fifth, different estimate.
    "background_corrected_count": ("Float64", True,
                                   "Alias of the corrected count from the configured primary method."),

    # ---- retained diagnostics. NOT valid as a primary background: at a
    # 10-11 px site pitch the 13-33 px annulus contains ~10 neighbouring sites.
    "background_annulus_contaminated": ("Float64", True,
                                        "DIAGNOSTIC ONLY, contaminated by neighbouring sites."),
    "count_corrected_annulus_contaminated": ("Float64", True,
                                             "DIAGNOSTIC ONLY, do not use for inference."),
    "background_annulus_density_contaminated": ("Float64", True,
                                                "DIAGNOSTIC ONLY, annulus statistic per pixel."),

    "grid":          ("string",  False, "Sub-array the site belongs to."),
    "site_row":      ("Int16",   False, "Lattice row index within its sub-array."),
    "site_col":      ("Int16",   False, "Lattice column index within its sub-array."),
    "roi_n_pixels":  ("Int32",   False, "Pixels summed for roi_sum_raw."),
    "roi_max_pixel": ("Float64", True,  "Maximum single-pixel value inside the ROI, saturation check."),
    "site_detected":  ("boolean", False,
                       "A variance peak lies within the detection radius of the "
                       "modelled site position. Geometry confidence, not brightness."),
}

#: every (background, corrected) pair, keyed by method label
METHOD_COLUMNS: dict[str, tuple[str | None, str]] = {
    "raw": (None, "roi_sum_raw"),
    "global": ("background_global", "count_corrected_global"),
    "spatial": ("background_spatial", "count_corrected_spatial"),
    "fixed_offset": ("background_fixed_offset", "count_corrected_fixed_offset"),
    "annulus_contaminated": ("background_annulus_contaminated",
                             "count_corrected_annulus_contaminated"),
}


def migrate_v1_to_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Read a v1 table under v2 names.

    Renaming only. The v1 table has no spatial or fixed-template background, so
    those columns stay absent rather than being filled with a stand-in — a
    migrated table is explicitly missing methods C and D.
    """
    out = df.rename(columns={k: v for k, v in V1_TO_V2.items() if k in df.columns})
    out.attrs["schema_version"] = "2.0-migrated-from-1.0"
    out.attrs["missing_methods"] = ["spatial", "fixed_offset"]
    return out

ALL_COLUMNS = {**FRAME_SITE_COLUMNS, **OPTIONAL_COLUMNS}

#: quality flags this pipeline can raise
QUALITY_FLAGS = (
    "ok",
    "roi_saturated",          # a pixel in the ROI reached the ADC ceiling
    "roi_touches_edge",       # ROI box clipped by the frame boundary
    "local_bg_underdetermined",  # annulus had too few usable pixels
    "nonfinite_count",        # ROI sum or background not finite
    # No variance peak within the detection radius of the modelled position.
    # This is a statement about how confidently the site was localised, NOT a
    # statement that the site is dark: it fires both for a site whose variance
    # peak is unusually weak and for one where a neighbouring peak merged.
    # Flagged rows keep their modelled ROI and are not dropped.
    "site_not_detected",
)

#: methods whose corrected count must never be used for inference
CONTAMINATED_METHODS = ("annulus_contaminated",)


@dataclass
class ValidationReport:
    ok: bool
    n_rows: int
    errors: list[str]
    warnings: list[str]
    stats: dict[str, Any]

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise ValueError(
                "frame-site table failed validation:\n  - "
                + "\n  - ".join(self.errors)
            )


def empty_frame() -> pd.DataFrame:
    """Empty table with the full declared dtype set."""
    return pd.DataFrame({c: pd.Series(dtype=t) for c, (t, _, _) in ALL_COLUMNS.items()})


def coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Apply declared dtypes and column order; unknown columns are kept last."""
    out = df.copy()
    for col, (dtype, _, _) in ALL_COLUMNS.items():
        if col in out.columns:
            out[col] = out[col].astype(dtype)
    ordered = [c for c in ALL_COLUMNS if c in out.columns]
    rest = [c for c in out.columns if c not in ALL_COLUMNS]
    return out[ordered + rest]


def validate(df: pd.DataFrame, *, expected_frames: int | None = None,
             expected_shots: int | None = None,
             expected_sites: int | None = None) -> ValidationReport:
    """Structural validation of the standardized table.

    Checks structure and internal consistency only. It deliberately makes no
    judgement about whether the *physics* is right.
    """
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, Any] = {}

    missing = [c for c in FRAME_SITE_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"missing required columns: {missing}")
        return ValidationReport(False, len(df), errors, warnings, stats)

    # ---------------------------------------------------------- primary key
    dup = df.duplicated(subset=list(PRIMARY_KEY)).sum()
    stats["duplicate_primary_keys"] = int(dup)
    if dup:
        errors.append(f"{dup} duplicate rows for primary key {PRIMARY_KEY}")

    # ------------------------------------------------------------ null-ness
    for col, (_, nullable, _) in ALL_COLUMNS.items():
        if col in df.columns and not nullable:
            n_null = int(df[col].isna().sum())
            if n_null:
                errors.append(f"column '{col}' is non-nullable but has {n_null} nulls")

    # ------------------------------------------------------------ structure
    shots = df["shot_id"].dropna().unique()
    frames = sorted(df["frame_id"].dropna().unique().tolist())
    sites = df["site_id"].dropna().unique()
    stats.update(n_shots=int(len(shots)), n_sites=int(len(sites)),
                 frame_ids=[int(f) for f in frames], n_rows=int(len(df)))

    if expected_shots is not None and len(shots) != expected_shots:
        errors.append(f"expected {expected_shots} shots, found {len(shots)}")
    if expected_sites is not None and len(sites) != expected_sites:
        errors.append(f"expected {expected_sites} sites, found {len(sites)}")
    if expected_frames is not None and len(frames) != expected_frames:
        errors.append(f"expected {expected_frames} frame ids, found {frames}")

    # every shot must carry the same complete frame x site block
    per_shot = df.groupby("shot_id", observed=True).size()
    expected_block = len(frames) * len(sites)
    bad = per_shot[per_shot != expected_block]
    stats["shots_with_incomplete_block"] = int(len(bad))
    if len(bad):
        errors.append(
            f"{len(bad)} shots do not have exactly {expected_block} rows "
            f"({len(frames)} frames x {len(sites)} sites)"
        )

    counts_per_frame = df.groupby("frame_id", observed=True)["site_id"].nunique()
    if counts_per_frame.nunique() > 1:
        errors.append(f"site count differs between frames: {counts_per_frame.to_dict()}")

    # ----------------------------------------------------------- numerics
    for col in ("roi_sum_raw", "background_global", "count_corrected_global",
                "background_spatial", "count_corrected_spatial"):
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        n_bad = int(np.count_nonzero(~np.isfinite(vals)))
        stats[f"nonfinite_{col}"] = n_bad
        if n_bad:
            flagged = df.loc[~np.isfinite(vals), "quality_flag"].astype(str)
            if not flagged.str.contains("nonfinite_count").all():
                errors.append(f"{n_bad} non-finite values in '{col}' are not flagged")

    # -------------------------------------------------------- site geometry
    geo = df.groupby("site_id", observed=True)[["site_x", "site_y"]].nunique()
    inconsistent = int(((geo["site_x"] > 1) | (geo["site_y"] > 1)).sum())
    stats["sites_with_moving_coordinates"] = inconsistent
    if inconsistent:
        errors.append(f"{inconsistent} sites have non-constant coordinates across rows")

    # -------------------------------------------------------- quality flags
    flags = (df["quality_flag"].astype(str).str.split("|").explode().unique().tolist())
    unknown = sorted(set(flags) - set(QUALITY_FLAGS))
    stats["quality_flag_values"] = sorted(f for f in flags if f)
    if unknown:
        errors.append(f"unknown quality flags: {unknown}")
    stats["n_rows_flagged"] = int((df["quality_flag"].astype(str) != "ok").sum())

    # ------------------------------------------------------------- privacy
    if "raw_image_path" in df.columns:
        paths = df["raw_image_path"].dropna().astype(str)
        if paths.str.contains(r"[/\\]", regex=True).any():
            errors.append("raw_image_path must contain a bare file name, not a path")

    # ---------------------------------------------------------- exposures
    if "exposure_ms" in df.columns:
        ex = pd.to_numeric(df["exposure_ms"], errors="coerce").dropna().unique()
        stats["exposure_ms_values"] = sorted(float(v) for v in ex)
        if len(ex) > 1:
            warnings.append(f"more than one exposure value present: {sorted(ex)}")

    return ValidationReport(not errors, len(df), errors, warnings, stats)
