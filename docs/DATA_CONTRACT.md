# Versioned frame-site data contract

| Version | Change |
|---|---|
| 1.0 | first standardized table; the local annulus was the primary background |
| **2.0** | annulus demoted to a diagnostic; four background methods in separate columns; `roi_sum` renamed `roi_sum_raw`; `schema_version` and `geometry_version` recorded in the metadata sidecar |
| **3.0** | additive multi-dataset, sweep, timing, split and hardware-command fields; V0 remains a schema-2.0 product |

`schema.migrate_v1_to_v2` renames a v1 table into v2 names. It fills nothing in: a migrated table is explicitly missing the spatial and template methods, and the old primary maps onto `count_corrected_annulus_contaminated` rather than being silently promoted.

Schema 3.0 extends, rather than replaces, the schema-2.0 frame-site table.
`schema.migrate_v2_to_v3` requires explicit run metadata for fields that did
not exist in V2 and leaves unrecoverable hardware measurements null. The
paired-100-ms config still requests schema 2.0, so its columns, commands and
sidecar meaning are unchanged.

The boundary between the two repositories, and the exact table that crosses it.

```
raw fluorescence shots  (private, read-only, never committed here)
          |
          v
mit-tweezer-array-analysis          image loading, ROI sums, local background,
  (general lab infrastructure)      grid geometry, lab numerics
          |
          v
standardized frame-site table       <-- THE CONTRACT
  data/processed/*.parquet
          |
          v
neutral-atom-fluorescence-inference  QC, background variants, statistics
```

This repository contains no laboratory control code, no camera or hardware
driver, no absolute data path, and no raw image. It reads the table.

The dependency runs one way. The general analysis checkout is imported through
`sys.path` (see `paths.tweezer_analysis_src` in `configs/local.toml`) and is
never written to, never installed into, and never modified.

---

## Files

| File | Contents |
|---|---|
| `<dataset_id>.parquet` | the frame-site long table, one row per (shot, frame, site) |
| `<dataset_id>.sites.parquet` | one row per site: geometry, ROI box, fit diagnostics |
| `<dataset_id>.meta.json` | provenance, site-fit summary, validation report, schema |
| `<dataset_id>.run.json` | run-level timing, command-state evidence, globals and missing-metadata explanations (V3) |
| `<dataset_id>.splits.parquet` | one row per complete shot with its frozen train/validation/test assignment (V3) |

None of them are committed. Regenerate with:

```bash
python scripts/export_processed_dataset.py --config configs/paired_100ms.yaml
python scripts/export_processed_dataset.py --config configs/dark_hold_50ms_20260728_0044.yaml
python scripts/export_processed_dataset.py --config configs/bright_wait_50ms_20260728_0050.yaml
```

---

## Primary key

```
(run_id, shot_id, frame_id, site_id)
```

Enforced by `schema.validate`, which also requires that every shot carry a
complete `frames × sites` block. A partial shot is an error, not a warning.
Schema V3 uses `(dataset_id, run_id, shot_id, frame_id, site_id)` so multiple
datasets cannot collide when concatenated.

---

## Required columns

| Column | Type | Null? | Meaning |
|---|---|---|---|
| `run_id` | string | no | one acquisition run |
| `shot_id` | Int32 | no | camera run number within the run |
| `shot_order` | Int32 | no | acquisition-time rank, 0-based |
| `frame_id` | Int8 | no | 0 = first exposure, 1 = second |
| `site_id` | Int32 | no | stable across shots and frames |
| `timestamp` | string | yes | per-shot acquisition time |
| `exposure_ms` | Float64 | yes | exposure of *this* frame |
| `frame_elapsed_s` | Float64 | yes | exposure start, seconds from sequence t=0 |
| `site_x`, `site_y` | Float64 | no | site centre in full-frame pixels |
| `roi_sum_raw` | Float64 | yes | **method A** — no correction |
| `background_global` | Float64 | yes | **B** reference: site-free median × ROI pixels |
| `count_corrected_global` | Float64 | yes | **B** corrected count |
| `background_spatial` | Float64 | yes | **C** reference: robust surface over the ROI |
| `count_corrected_spatial` | Float64 | yes | **C** corrected count |
| `raw_image_path` | string | yes | **file name only**, never a path |
| `quality_flag` | string | no | `ok`, or `|`-joined flags |

## Additional columns carried by this project

| Column | Meaning |
|---|---|
| `background_fixed_offset` | **D** reference: fixed template + per-frame offset |
| `count_corrected_fixed_offset` | **D** corrected count — the current primary |
| `background_corrected_count` | alias of the primary method's corrected count; which one is recorded in the metadata under `background.primary_method`, never a fifth estimate |
| `background_annulus_contaminated` | **diagnostic only** |
| `count_corrected_annulus_contaminated` | **diagnostic only, not valid for inference** |
| `background_annulus_density_contaminated` | **diagnostic only**, per pixel |
| `grid`, `site_row`, `site_col` | sub-array identity and lattice index |
| `roi_n_pixels` | pixels summed for `roi_sum_raw` |
| `roi_max_pixel` | brightest ROI pixel, for the saturation check |
| `site_detected` | a variance peak lies within the detection radius |

Every method keeps its own pair of columns. Nothing overwrites anything, so
changing the primary method is a configuration edit plus a rebuild, not a
re-derivation, and an analysis can always see what the alternatives would have
given. `tests/test_background_models.py` asserts the columns stay distinct and
`tests/test_dataset.py` asserts each corrected column equals raw minus its own
background.

## Additive schema-3.0 fields

| Group | Columns | Meaning |
|---|---|---|
| dataset identity | `dataset_id`, `date`, `sequence_id`, `sequence_type` | machine-independent identifiers; no local path |
| acquisition | `repetition_index`, `cycle_index`, `condition_id`, `shot_order`, `split` | complete-shot grouping and frozen 60/20/20 cycle split |
| sweep | `sweep_axis`, `sweep_value_s` | public sweep name and exact value read from shot globals |
| frame | `frame_index`, `frame_name`, `frame_start_s`, `frame_elapsed_s`, `exposure_s`, `interframe_gap_s` | commanded frame order and timing |
| optical interval | `dark_hold_s`, `bright_wait_before_first_s`, `actual_light_on_s` | commanded holds/waits; realized light-on time is null without readback |
| command state | `switch_state`, `dds_frequency`, `dds_amplitude` | scalar values only when directly recoverable; multi-channel state lives at run level |
| geometry | `grid_id`, `site_row`, `site_col`, `site_y`, `site_x` | per-run lattice identity and coordinates |
| template background | `background_template_offset`, `count_corrected_template` | public aliases for training-shot fixed template plus per-frame offset |

The retained V2 names (`grid`, `background_fixed_offset`,
`count_corrected_fixed_offset`) remain present in V3. The aliases have exactly
the same numeric values; they expose the public V1 terminology without
silently changing V0.

Run-level JSON stores globals and multi-channel DDS commands once instead of
duplicating them across every site-frame row. Each hardware field records its
evidence scope. In particular, compiled switch commands are not optical-power
readback, and commanded trigger duration is not a measured camera exposure.

---

## Quality flags

| Flag | Raised when |
|---|---|
| `ok` | nothing to report |
| `roi_saturated` | an ROI pixel reached the ADC ceiling |
| `roi_touches_edge` | the ROI box was clipped by the frame boundary |
| `local_bg_underdetermined` | too few usable annulus pixels |
| `nonfinite_count` | an ROI sum or background is not finite |
| `site_not_detected` | no variance peak within the detection radius of the modelled position |

`site_not_detected` is a **geometry-confidence** flag. It does not mean the site
is dark, and flagged rows keep their modelled ROI. See `DATA_AUDIT.md` §4.

Rows are never deleted. Everything excluded from a downstream analysis is
excluded by an explicit flag filter, visible in that analysis.

---

## Missing data

Absent metadata is written as null. It is never back-filled from a nominal
value, a neighbouring shot, or a configuration default. The run sidecar
records why values such as realized light-on time or scalar DDS settings are
unavailable. A null therefore means “not recoverable under the stated evidence
scope,” not zero.

---

## Guarantees the exporter enforces

1. The shot count matches `source.expected_n_shots`, or the export aborts.
2. Every shot has every configured frame, or the export aborts.
3. Per-shot exposure, frame ordering, sweep grouping and inter-frame timing
   are cross-checked against `source.expected_*`, the HDF5 `EXPOSURES` table
   and compiled command edges, then recorded with their evidence scope.
4. Schema validation runs on every export; failure is a non-zero exit code.
5. Provenance — inference commit, lab-analysis commit, dirty flag, config
   SHA-256, input-manifest hash, Python version, schema and geometry versions,
   background method, split definition, random seed and model version — is
   written into the metadata/result sidecars.
6. Nothing machine-identifying is written into any artefact: input paths appear
   only as `<configured:present>` / `<configured:missing>`, while output paths
   appear as `<configured>` so a first build and a repeat build hash equally.

---

## Reproducibility

Given the same raw shots and the same config, the exporter is deterministic.
For V3, geometry and the fixed background template are fitted on training
cycles only; validation and test shots never alter them. Split generation is
deterministic and every condition contributes 6/2/2 shots for
train/validation/test. Tests rebuild in isolated output roots and compare
tables and metadata without relying on operating-system-specific PNG bytes.

Any stochastic step elsewhere (the QC bootstrap) takes its seed from
`qc.random_seed` in the config and records it in the output.
