# Data contract

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
| `roi_sum` | Float64 | yes | **variant A** |
| `local_background` | Float64 | yes | annulus median × ROI pixel count |
| `global_background` | Float64 | yes | site-free median × ROI pixel count |
| `background_corrected_count` | Float64 | yes | **variant B**, the primary measurement |
| `raw_image_path` | string | yes | **file name only**, never a path |
| `quality_flag` | string | no | `ok`, or `|`-joined flags |

## Additional columns carried by this project

| Column | Meaning |
|---|---|
| `grid`, `site_row`, `site_col` | sub-array identity and lattice index |
| `roi_n_pixels` | pixels summed for `roi_sum` |
| `local_background_density` | annulus statistic per pixel |
| `roi_max_pixel` | brightest ROI pixel, for the saturation check |
| `common_mode_corrected_count` | **variant C** |
| `site_detected` | a variance peak lies within the detection radius |

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
   Python version — is written into the metadata sidecar.
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
