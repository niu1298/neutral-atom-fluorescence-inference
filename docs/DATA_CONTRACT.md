# Data contract — schema **2.0**

| Version | Change |
|---|---|
| 1.0 | first standardized table; the local annulus was the primary background |
| **2.0** | annulus demoted to a diagnostic; four background methods in separate columns; `roi_sum` renamed `roi_sum_raw`; `schema_version` and `geometry_version` recorded in the metadata sidecar |

`schema.migrate_v1_to_v2` renames a v1 table into v2 names. It fills nothing in: a migrated table is explicitly missing the spatial and template methods, and the old primary maps onto `count_corrected_annulus_contaminated` rather than being silently promoted.

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

None of them are committed. Regenerate with:

```bash
python scripts/export_processed_dataset.py --config configs/paired_100ms.yaml
```

---

## Primary key

```
(run_id, shot_id, frame_id, site_id)
```

Enforced by `schema.validate`, which also requires that every shot carry a
complete `frames × sites` block. A partial shot is an error, not a warning.

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
value, a neighbouring shot, or a configuration default. If a column is null,
the shot file did not contain it.

---

## Guarantees the exporter enforces

1. The shot count matches `source.expected_n_shots`, or the export aborts.
2. Every shot has every configured frame, or the export aborts.
3. Per-shot exposure and inter-frame timing are cross-checked against
   `source.expected_*` and recorded in the metadata sidecar.
4. Schema validation runs on every export; failure is a non-zero exit code.
5. Provenance — git commit, dirty flag, config SHA-256, input manifest hash,
   Python version, **schema version, geometry version, background method and
   mask parameters** — is written into the metadata sidecar.
6. Nothing machine-identifying is written into any artefact: paths appear only
   as `<configured:present>` / `<configured:missing>`.

---

## Reproducibility

Given the same raw shots and the same config, the exporter is deterministic:
site finding uses no random initialisation, and the k-means helper seeds
deterministically from coordinate quantiles. `tests/test_dataset.py` asserts
byte-level equality of the table across two independent builds.

Any stochastic step elsewhere (the QC bootstrap) takes its seed from
`qc.random_seed` in the config and records it in the output.
