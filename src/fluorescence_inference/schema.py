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

# V0 remains a schema-2.0 product.  The loss-sweep work opts in to the
# additive schema below through ``schema_version: "3.0"`` in its tracked
# config.  Keeping the V0 constant here is intentional: old commands and
# sidecars must not silently change meaning.
V3_SCHEMA_VERSION = "3.0"
SUPPORTED_SCHEMA_VERSIONS = (SCHEMA_VERSION, V3_SCHEMA_VERSION)

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

V2_ALL_COLUMNS = {**FRAME_SITE_COLUMNS, **OPTIONAL_COLUMNS}

# Additive columns used by multi-run and swept acquisitions.  Existing V2
# names are retained in V3; aliases make the public terminology explicit
# without invalidating old tables.
V3_ADDITIONAL_COLUMNS: dict[str, tuple[str, bool, str]] = {
    "dataset_id": ("string", False, "Machine-independent dataset identifier."),
    "date": ("string", False, "Acquisition date, YYYY-MM-DD."),
    "sequence_id": ("string", False, "Machine-independent sequence identifier."),
    "sequence_type": ("string", False, "Acquisition family, e.g. switch_off_hold."),
    "repetition_index": ("Int16", False,
                         "Within-condition repetition index; inferred when not stored."),
    "cycle_index": ("Int16", False,
                    "Acquisition-cycle index used as the indivisible split block."),
    "condition_id": ("string", False, "Stable identifier for the sweep condition."),
    "split": ("string", False, "Shot-level train, validation, or test assignment."),
    "sweep_axis": ("string", True, "Swept physical axis in public terminology."),
    "sweep_value_s": ("Float64", True, "Sweep value in seconds."),
    "frame_index": ("Int8", False, "Zero-based temporal frame index."),
    "frame_name": ("string", False, "Saved exposure name, e.g. fluor2."),
    "frame_start_s": ("Float64", True,
                      "Commanded frame start relative to sequence t=0, seconds."),
    "exposure_s": ("Float64", True,
                   "Commanded trigger duration; not a measured exposure readback."),
    "interframe_gap_s": ("Float64", True,
                         "Commanded gap after the preceding frame, seconds."),
    "dark_hold_s": ("Float64", True,
                    "Commanded switch-off gap preceding this frame, seconds."),
    "bright_wait_before_first_s": ("Float64", True,
                                   "Commanded illuminated wait before frame 0, seconds."),
    "actual_light_on_s": ("Float64", True,
                          "Measured light-on duration, null when no readback exists."),
    "switch_state": ("string", True,
                     "Verified commanded switch state; not an optical-power readback."),
    "dds_frequency": ("Float64", True,
                      "Calibrated scalar DDS frequency, null for multi-channel commands."),
    "dds_amplitude": ("Float64", True,
                      "Calibrated scalar DDS amplitude, null for multi-channel commands."),
    "grid_id": ("string", False, "Grid identifier; alias of V2 `grid`."),
    "background_template_offset": (
        "Float64", True,
        "Method D reference: training-shot fixed template plus frame offset, over ROI."),
    "count_corrected_template": (
        "Float64", True,
        "Method D count; alias of V2 `count_corrected_fixed_offset`."),
}

V3_ALL_COLUMNS = {**V2_ALL_COLUMNS, **V3_ADDITIONAL_COLUMNS}

# Backwards-compatible public name used by all V0 tests and callers.
ALL_COLUMNS = V2_ALL_COLUMNS

V3_REQUIRED_COLUMNS = {
    **FRAME_SITE_COLUMNS,
    **{
        name: OPTIONAL_COLUMNS[name]
        for name in (
            "background_fixed_offset",
            "count_corrected_fixed_offset",
            "background_corrected_count",
            "background_annulus_contaminated",
            "count_corrected_annulus_contaminated",
            "grid",
            "site_row",
            "site_col",
        )
    },
    **V3_ADDITIONAL_COLUMNS,
}

V3_PRIMARY_KEY = ("dataset_id", "run_id", "shot_id", "frame_id", "site_id")

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


def columns_for_version(version: str = SCHEMA_VERSION
                        ) -> dict[str, tuple[str, bool, str]]:
    """Return the declared contract for one explicit schema version."""
    if version == SCHEMA_VERSION:
        return V2_ALL_COLUMNS
    if version == V3_SCHEMA_VERSION:
        return V3_ALL_COLUMNS
    raise ValueError(
        f"unsupported schema version {version!r}; expected one of "
        f"{SUPPORTED_SCHEMA_VERSIONS}"
    )


def primary_key_for_version(version: str = SCHEMA_VERSION) -> tuple[str, ...]:
    """Return the primary key without changing the legacy public constant."""
    columns_for_version(version)
    return PRIMARY_KEY if version == SCHEMA_VERSION else V3_PRIMARY_KEY


def empty_frame(version: str = SCHEMA_VERSION) -> pd.DataFrame:
    """Empty table with the selected version's full declared dtype set."""
    columns = columns_for_version(version)
    out = pd.DataFrame({c: pd.Series(dtype=t) for c, (t, _, _) in columns.items()})
    out.attrs["schema_version"] = version
    return out


def coerce(df: pd.DataFrame, *, version: str = SCHEMA_VERSION) -> pd.DataFrame:
    """Apply declared dtypes and column order; unknown columns are kept last."""
    out = df.copy()
    columns = columns_for_version(version)
    for col, (dtype, _, _) in columns.items():
        if col in out.columns:
            out[col] = out[col].astype(dtype)
    ordered = [c for c in columns if c in out.columns]
    rest = [c for c in out.columns if c not in columns]
    out = out[ordered + rest]
    out.attrs["schema_version"] = version
    return out


def migrate_v2_to_v3(
    df: pd.DataFrame,
    *,
    dataset_id: str,
    date: str,
    sequence_id: str,
    sequence_type: str,
    frame_names: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Add V3 fields to a V2 table without changing any V2 values.

    Identifiers are mandatory because deriving them from a local path would be
    ambiguous and could leak machine state.  Hardware readback fields stay
    null, with the reason recorded in ``DataFrame.attrs``.
    """
    missing = [c for c in FRAME_SITE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"cannot migrate V2 table; missing columns: {missing}")
    if not all(str(v).strip() for v in
               (dataset_id, date, sequence_id, sequence_type)):
        raise ValueError("dataset_id, date, sequence_id, and sequence_type are required")

    out = df.copy()
    out["dataset_id"] = str(dataset_id)
    out["date"] = str(date)
    out["sequence_id"] = str(sequence_id)
    out["sequence_type"] = str(sequence_type)
    out["repetition_index"] = out["shot_order"]
    out["cycle_index"] = out["shot_order"]
    out["condition_id"] = "single_condition"
    out["split"] = "unassigned"
    out["sweep_axis"] = pd.NA
    out["sweep_value_s"] = pd.NA
    out["frame_index"] = out["frame_id"]
    names = frame_names or {}
    out["frame_name"] = out["frame_id"].map(
        lambda x: names.get(int(x), f"frame_{int(x)}"))
    out["frame_start_s"] = out["frame_elapsed_s"]
    out["exposure_s"] = pd.to_numeric(out["exposure_ms"], errors="coerce") / 1e3

    timing = (out[["run_id", "shot_id", "frame_id", "frame_start_s", "exposure_s"]]
              .drop_duplicates()
              .sort_values(["run_id", "shot_id", "frame_id"]))
    grouped = timing.groupby(["run_id", "shot_id"], observed=True)
    timing["previous_end_s"] = (
        grouped["frame_start_s"].shift() + grouped["exposure_s"].shift())
    timing["interframe_gap_s"] = timing["frame_start_s"] - timing["previous_end_s"]
    gaps = timing.set_index(["run_id", "shot_id", "frame_id"])["interframe_gap_s"]
    out["interframe_gap_s"] = [
        gaps.get((r, s, f), np.nan)
        for r, s, f in out[["run_id", "shot_id", "frame_id"]].itertuples(
            index=False, name=None)
    ]

    for col in ("dark_hold_s", "bright_wait_before_first_s", "actual_light_on_s",
                "switch_state", "dds_frequency", "dds_amplitude"):
        out[col] = pd.NA
    out["grid_id"] = out["grid"]
    out["background_template_offset"] = out.get("background_fixed_offset", pd.NA)
    out["count_corrected_template"] = out.get("count_corrected_fixed_offset", pd.NA)

    out = coerce(out, version=V3_SCHEMA_VERSION)
    out.attrs.update({
        "schema_version": "3.0-migrated-from-2.0",
        "inferred_fields": {
            "repetition_index": "V2 shot_order; V2 is a single-condition run.",
            "cycle_index": "V2 shot_order; no acquisition-cycle metadata existed.",
            "condition_id": "single_condition",
            "frame_name": "caller mapping or deterministic frame_<id> fallback",
            "interframe_gap_s": "derived from commanded frame starts and exposures",
        },
        "unrecoverable_fields": {
            "actual_light_on_s": "no optical-power or light-on readback",
            "switch_state": "not represented in the V2 frame-site table",
            "dds_frequency": "no single calibrated scalar value",
            "dds_amplitude": "no single calibrated scalar value",
        },
    })
    return out


def validate(df: pd.DataFrame, *, expected_frames: int | None = None,
             expected_shots: int | None = None,
             expected_sites: int | None = None,
             version: str = SCHEMA_VERSION) -> ValidationReport:
    """Structural validation of the standardized table.

    Checks structure and internal consistency only. It deliberately makes no
    judgement about whether the *physics* is right.
    """
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, Any] = {}

    columns = columns_for_version(version)
    required = FRAME_SITE_COLUMNS if version == SCHEMA_VERSION else V3_REQUIRED_COLUMNS
    key = primary_key_for_version(version)
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"missing required columns: {missing}")
        return ValidationReport(False, len(df), errors, warnings, stats)

    # ---------------------------------------------------------- primary key
    dup = df.duplicated(subset=list(key)).sum()
    stats["duplicate_primary_keys"] = int(dup)
    if dup:
        errors.append(f"{dup} duplicate rows for primary key {key}")

    # ------------------------------------------------------------ null-ness
    for col, (_, nullable, _) in columns.items():
        if col in df.columns and not nullable:
            n_null = int(df[col].isna().sum())
            if n_null:
                errors.append(f"column '{col}' is non-nullable but has {n_null} nulls")

    # ------------------------------------------------------------ structure
    shot_key = ["shot_id"] if version == SCHEMA_VERSION else [
        "dataset_id", "run_id", "shot_id"]
    site_key = ["site_id"] if version == SCHEMA_VERSION else [
        "dataset_id", "run_id", "grid_id", "site_id"]
    shots = df[shot_key].drop_duplicates()
    frames = sorted(df["frame_id"].dropna().unique().tolist())
    sites = df[site_key].drop_duplicates()
    stats.update(n_shots=int(len(shots)), n_sites=int(len(sites)),
                 frame_ids=[int(f) for f in frames], n_rows=int(len(df)))
    run_frame_counts: pd.Series | None = None
    if version == V3_SCHEMA_VERSION:
        run_frame_ids = (
            df[["dataset_id", "run_id", "frame_id"]]
            .drop_duplicates()
            .groupby(["dataset_id", "run_id"], observed=True)["frame_id"]
            .agg(lambda values: sorted(int(v) for v in values))
        )
        run_frame_counts = run_frame_ids.map(len)
        stats["frame_ids_by_run"] = {
            f"{dataset_id}/{run_id}": frame_ids
            for (dataset_id, run_id), frame_ids in run_frame_ids.items()
        }

    if expected_shots is not None and len(shots) != expected_shots:
        errors.append(f"expected {expected_shots} shots, found {len(shots)}")
    if expected_sites is not None:
        if version == SCHEMA_VERSION:
            site_counts = [len(sites)]
        else:
            site_counts = (
                sites.groupby(["dataset_id", "run_id"], observed=True).size().tolist())
        if any(n != expected_sites for n in site_counts):
            errors.append(f"expected {expected_sites} sites, found {site_counts}")
    if expected_frames is not None:
        if version == SCHEMA_VERSION:
            if len(frames) != expected_frames:
                errors.append(
                    f"expected {expected_frames} frame ids, found {frames}")
        else:
            assert run_frame_counts is not None
            bad_frame_counts = {
                f"{dataset_id}/{run_id}": int(count)
                for (dataset_id, run_id), count in run_frame_counts.items()
                if int(count) != expected_frames
            }
            if bad_frame_counts:
                errors.append(
                    f"expected {expected_frames} frames per run, found "
                    f"{bad_frame_counts}")

    # every shot must carry the same complete frame x site block
    per_shot = df.groupby(shot_key, observed=True).size()
    if version == SCHEMA_VERSION:
        expected_by_shot = pd.Series(
            len(frames) * len(sites), index=per_shot.index)
        expected_block_text = str(len(frames) * len(sites))
    else:
        assert run_frame_counts is not None
        run_sites = (
            sites.groupby(["dataset_id", "run_id"], observed=True).size())
        expected_by_shot = pd.Series(
            [
                int(run_frame_counts.loc[(dataset_id, run_id)])
                * int(run_sites.loc[(dataset_id, run_id)])
                for dataset_id, run_id, _ in per_shot.index
            ],
            index=per_shot.index,
        )
        expected_block_text = "run-specific frame x run-specific site"
    bad = per_shot[per_shot.to_numpy() != expected_by_shot.to_numpy()]
    stats["shots_with_incomplete_block"] = int(len(bad))
    if len(bad):
        errors.append(
            f"{len(bad)} shots do not have exactly {expected_block_text} rows "
            f"(global union: {len(frames)} frames x {len(sites)} sites)"
        )

    if version == SCHEMA_VERSION:
        counts_per_frame = df.groupby(
            "frame_id", observed=True)["site_id"].nunique()
        if counts_per_frame.nunique() > 1:
            errors.append(
                f"site count differs between frames: {counts_per_frame.to_dict()}")
    else:
        counts_per_frame = (
            df[["dataset_id", "run_id", "frame_id", "grid_id", "site_id"]]
            .drop_duplicates()
            .groupby(["dataset_id", "run_id", "frame_id"], observed=True)
            .size()
        )
        if (counts_per_frame.groupby(
                level=["dataset_id", "run_id"]).nunique() > 1).any():
            errors.append(
                f"site count differs between frames: {counts_per_frame.to_dict()}")

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
    geo = df.groupby(site_key, observed=True)[["site_x", "site_y"]].nunique()
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

    if version == V3_SCHEMA_VERSION:
        allowed_splits = {"train", "validation", "test", "unassigned"}
        seen_splits = set(df["split"].dropna().astype(str))
        unknown_splits = sorted(seen_splits - allowed_splits)
        if unknown_splits:
            errors.append(f"unknown split values: {unknown_splits}")

        if not (df["frame_index"].astype("Int64") ==
                df["frame_id"].astype("Int64")).all():
            errors.append("frame_index must equal the legacy frame_id alias")
        if not (df["grid_id"].astype(str) == df["grid"].astype(str)).all():
            errors.append("grid_id must equal the legacy grid alias")
        for old, new in (
            ("background_fixed_offset", "background_template_offset"),
            ("count_corrected_fixed_offset", "count_corrected_template"),
        ):
            both = df[[old, new]].apply(pd.to_numeric, errors="coerce")
            mismatch = ~np.isclose(both[old], both[new], equal_nan=True)
            if mismatch.any():
                errors.append(f"{new} must equal its backwards-compatible alias {old}")

        split_per_shot = df.groupby(shot_key, observed=True)["split"].nunique()
        if (split_per_shot != 1).any():
            errors.append("a shot is split across train/validation/test")
        condition_per_shot = df.groupby(
            shot_key, observed=True)["condition_id"].nunique()
        if (condition_per_shot != 1).any():
            errors.append("a shot maps to more than one condition")

    return ValidationReport(not errors, len(df), errors, warnings, stats)
