# Neutral-Atom Fluorescence Inference

Statistical inference of latent atom occupancy from paired fluorescence
readout, on a standardized single-condition dataset with frame-dependent
background.

<p align="center">
  <img src="assets/readme/fluorescence_inference_overview.gif"
       alt="Animated walkthrough: two consecutive fluorescence frames, site ROIs, per-site extraction, raw and background-corrected count distributions, a descriptive two-component fit, and the frame-0 against frame-1 paired readout."
       width="1100">
</p>

<!-- BEGIN:dataset-line -->
**Current dataset** — 100 shots · 2 consecutive frames per shot · 100 ms exposure · 200 sites (40,000 site-frame observations, 20,000 site-shot pairs, 2 sites carrying a geometry flag). The independent experimental units are the 100 shots.
<!-- END:dataset-line -->

**Scientific question** — how can latent atom occupancy, and the disagreement
between two consecutive readouts of it, be inferred from noisy fluorescence
measurements whose background depends on which frame you are looking at?

**Current scope** — one imaging condition. Audit, standardized data layer,
paired-readout quality control, and descriptive baselines. No readout fidelity,
false-positive rate, false-negative rate or imaging-loss rate is estimated
here, and [Limitations](#limitations) says exactly why this dataset cannot
support those claims yet.

[**Method**](#method) · [**Validation**](#validation) · [**Results**](#results) ·
[**Reproduce**](#reproduce) · [**Limitations**](#limitations) ·
[**Data audit**](docs/DATA_AUDIT.md) · [**Validation report**](docs/VALIDATION.md) ·
[**Data contract**](docs/DATA_CONTRACT.md)

---

## Method

```mermaid
flowchart TD
    A["Raw fluorescence images<br/><i>private, read-only</i>"] --> B
    B["ROI and background extraction<br/><code>mit-tweezer-array-analysis</code>"] --> C
    C["Standardized frame-site table<br/><i>the repository boundary</i>"] --> D
    D["QC and background correction"] --> E
    E["Threshold / mixture baseline<br/><i>next milestone</i>"] --> F
    F["Paired latent-state inference<br/><i>next milestone</i>"] --> G
    G["Held-out evaluation and uncertainty<br/><i>next milestone</i>"]

    subgraph lab ["mit-tweezer-array-analysis — general lab infrastructure"]
        B
    end
    subgraph here ["neutral-atom-fluorescence-inference — this repository"]
        D
        E
        F
        G
    end
```

Image loading, ROI sums, local-background estimation and grid geometry come
from the general lab analysis package and are reused rather than reimplemented;
`tests/test_reuse_matches_lab.py` pins this repository's helpers to that
implementation numerically. This repository contains no laboratory control
code, no hardware driver, no absolute data path and no raw image. It starts at
the standardized table.

### Locating the sites

The frozen grid corners from an earlier day do not apply to this run, and the
100-shot **mean** image is dominated by a static fringe pattern that a
high-pass trap search cannot see through. The shot-to-shot **variance** map is
the opposite: static structure contributes almost nothing, and every loaded
site becomes a compact variance excess. Both 10 × 10 sub-arrays come out
cleanly, with a sub-pixel lattice residual.

| ![Consecutive frames with the site ROIs](assets/readme/paired_images_roi_overlay.png) |
|:--|
| Two consecutive frames of one shot, identical crop and identical intensity scale, with the fitted ROIs. |

### Extracting counts

Four background methods are carried through the table in **separate columns**,
so the choice stays visible and revisable instead of being frozen into one
number: no correction, a site-free median, a robust spatial surface refitted
per frame, and a fixed spatial template plus a per-frame common-mode offset.
All of them estimate the background from pixels outside a 5 px exclusion disk
around every site. A median over the sites is never used — a change in occupied
fraction would then be absorbed into "background".

The first version of this pipeline used a 13–33 px local annulus. At a 10–11 px
site pitch that ring contains roughly ten neighbouring sites, and it showed:
its "corrected" count still tracked its own background at |r| = 0.571, it
reported a frame-to-frame shift three times too large, and it gave the *worst*
separation of any method tested. It is retained only as a diagnostic column,
named `*_annulus_contaminated`, and is not used for inference.

| ![Four background methods compared on residual structure, drift and background coupling](assets/readme/background_method_comparison.png) |
|:--|
| Selection is on residual spatial structure, drift with acquisition order, and whether the corrected count still tracks its own background. `d'` is shown for completeness and deliberately **not** used to rank — it rewards subtracting less. |

| ![Background-corrected count distribution with a descriptive two-component fit](assets/readme/count_distribution_fit.png) |
|:--|
| Pooled background-corrected counts per frame. **Descriptive fit on the full dataset — not a held-out fidelity estimate.** No train/validation/test split, no per-site structure; the crossing is a display reference, not a validated classifier. |

### Pairing the two readouts

| ![Frame 0 against frame 1 paired counts](assets/readme/paired_frame_scatter.png) | ![Site-level descriptive maps](assets/readme/site_summary_map.png) |
|:--|:--|
| Every site-shot pair, with the four apparent regions. The reference lines come from a fit on the full dataset: a display reference, **not** a validated classifier. | Signal, background and spread across the array, with the geometry-flagged sites outlined. |

---

## Validation

The site geometry and the background correction were both fit results
presented as facts in the first pass. Round 1.5 tested them. Method and full
numbers: [`docs/VALIDATION.md`](docs/VALIDATION.md).

| Test | Result |
|---|---|
| Geometry refitted on shots 0–49 vs 50–99, optimally matched | median drift **0.080 px**, max **0.149 px**, 0 unmatched of 200 |
| Is the 10 × 10 window imposed or real? | **real** — the edge ring is 38–64× brighter than the first ring outside, and a larger fitting window finds +2.0% peaks at most |
| Are the two blocks two images of one array? | **no** — index-matched occupancy correlation −0.003 against a permuted null of −0.001 |
| The two geometry-flagged sites | one genuinely faint site, one peak-detection ambiguity; neither clipped, overlapping, or outside the valid region. Neither dropped |
| Background method | annulus **rejected**; template + per-frame offset selected on residual structure (8.74 vs 40.97 counts/px for a flat level) |
| Frame 1 − frame 0 shift, before → after | **−36.56 ± 9.94 → −12.53 ± 1.48** counts/px, now agreeing with the model-free whole-frame median (−13.77) |

What this certifies: the site set is stable, unclipped, non-overlapping and not
a duplicated image. What it does **not** certify: that each site is a
physically verified trap. Sites are localised from atom fluorescence, so a trap
never loaded during these 100 shots is invisible to the procedure, and no
trap-light reference image exists for this run. The phrase "200 valid traps"
appears nowhere in this repository.

One question stays open. Two sharply bounded 10 × 10 arrays with different
pitch and rotation appear in the same exposure, yet the sequence ramps the
second lattice to zero amplitude before imaging. Which potential holds each
block is unresolved; one shot with either lattice disabled from the start would
settle it. No result here depends on the answer.

---

## Results

Everything below is generated by the pipeline into
[`reports/readme_metrics.md`](reports/readme_metrics.md) and injected here. No
metric in this README is typed by hand.

> **These are descriptive quantities, not benchmarks.** The two-component fit
> behind `d'`, the overlap and the reference levels is fitted on the full
> dataset with no train/validation/test split. Nothing here is a held-out
> fidelity estimate or a measured error rate. **The next milestone is exactly
> that**: a shot-ordered 60/20/20 split, threshold and mixture baselines scored
> on held-out shots, and shot-cluster bootstrap intervals.

<!-- BEGIN:results -->
### Dataset, as measured

| quantity | value |
|---|---|
| Shots | 100 |
| Frames per shot | 2 |
| Exposure per frame | 100 ms |
| Inter-frame start separation | 110 ms (100 ms exposure + 10 ms gap) |
| Sites | 200 (two 10x10 sub-arrays) |
| ROI | 25 px per site |
| Site-frame observations | 40,000 |
| Site-shot pairs | 20,000 |
| Rows carrying a quality flag | 400 |
| Sites with a geometry flag | 2 of 200 |
| Lattice fit residual (median / P95) | 0.47 / 0.82 px |
| Saturated ROI pixels | 0 |
| Brightest pixel seen | 2,037 of 65,535 ADU |

### Frame-dependent background

| quantity | value |
|---|---|
| Frame 1 minus frame 0, template + offset (primary) | -12.53 ± 1.48 counts/px |
| Frame 1 minus frame 0, global site-free median | -12.92 ± 1.69 counts/px |
| Frame 1 minus frame 0, legacy annulus (contaminated) | -36.56 ± 9.94 counts/px |
| Background level under a ROI, frame 0 (primary) | 550.9 counts/px |
| Legacy annulus level, frame 0 | 591.6 counts/px |

The two site-masked estimators agree with each other to 0.4 counts/px and with
the whole-frame median shift. The legacy annulus reports a shift about three
times larger, with seven times the spread: at a 10–11 px site pitch its 13–33 px
ring contains roughly ten neighbouring sites, so part of what it calls
"background" is array light that itself changes between the frames. It is kept
as a diagnostic column and is not used for inference.

### Paired-readout agreement

| quantity | value |
|---|---|
| Paired-readout agreement | 91.98% (95% shot-cluster bootstrap 91.59–92.38%) |
| Apparent bright-to-dark | 11.98% |
| Apparent dark-to-bright | 3.49% |
| Above reference, frame 0 | 53.39% |
| Above reference, frame 1 | 48.63% |

Agreement is computed against per-frame descriptive reference levels.
"Apparent" is meant literally: this run has no matched-empty, dark-frame or
natural-loss control, so a disagreement cannot be assigned to atom loss rather
than to misclassification.

### Measurement variants

| variant | definition | frame | d' | model-implied overlap | drift / 100 shots |
|---|---|---|---|---|---|
| A | no correction | 0 | 2.80 | 8.3% | +190 |
| A | no correction | 1 | 3.29 | 5.1% | +64 |
| B | site-free median | 0 | 2.81 | 8.2% | +115 |
| B | site-free median | 1 | 3.29 | 5.0% | +21 |
| C | spatial surface | 0 | 2.98 | 7.0% | +117 |
| C | spatial surface | 1 | 3.42 | 4.4% | +47 |
| D | template + offset | 0 | 2.97 | 7.1% | +127 |
| D | template + offset | 1 | 3.41 | 4.4% | +19 |

`d'` and the overlap describe the descriptive two-component fit. They are
**not** a readout fidelity, a false-positive rate or a false-negative rate.

### Representative example used in the README assets

- **Shot:** shot whose mean frame-0 raw ROI count is closest to the run median, excluding shot 0 (documented background outlier) → shot order 50.
- **Site:** unflagged site whose mean frame-0 background-corrected count is closest to the median across sites → site 160
  (grid_B, row 6, column 0).
<!-- END:results -->

---

## Reproduce

```bash
git clone https://github.com/niu1298/neutral-atom-fluorescence-inference.git
cd neutral-atom-fluorescence-inference
python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev]"
cp configs/local.example.toml configs/local.toml   # then edit for your machine
```

Then, in order:

```bash
python scripts/audit_source_data.py         --config configs/paired_100ms.yaml
python scripts/export_processed_dataset.py   --config configs/paired_100ms.yaml
python scripts/validate_site_geometry.py     --config configs/paired_100ms.yaml
python scripts/compare_background_methods.py --config configs/paired_100ms.yaml
python scripts/generate_readme_assets.py     --config configs/paired_100ms.yaml
python -m pytest
```

In order: the data audit; the standardized table (schema 2.0) plus the QC
report; the geometry gate, which exits non-zero if it fails; the background
comparison, which reports whether the configured primary method matches what
the evidence recommends; and every asset on this page plus the metrics above.
Each script fails with an actionable message when the raw shots are
unavailable; none of them substitutes synthetic data for a missing
measurement.

`notebooks/01_data_qc.ipynb` is a presentation layer over the same package. No
asset and no number in this repository depends on running it.

Raw shots live under `Experiment-Data/<date>/<sequence>/` and are gitignored,
as are the processed table and the QC report. Provenance — git commit, config
hash, input-manifest hash — is recorded in every generated artefact; machine
paths never are.

---

## Limitations

**What this dataset can answer.** Whether the bright and dark populations
separate at 100 ms; whether there is a systematic background or signal shift
between the two frames; how consistent the two frames' apparent calls are; how
signal, background and spread vary across sites; the apparent bright-to-dark
and dark-to-bright rates; and which background treatment is more stable against
acquisition order.

**What it cannot.** With one imaging condition, two frames, and no control
data, this run has no identifying information for:

- a true false-positive or false-negative rate, or a readout fidelity —
  mixture overlap is a property of the fitted description, not a measured error
  rate;
- an imaging-induced loss rate — frame-to-frame disagreement mixes real loss
  with misclassification in both frames, and the apparent dark-to-bright rate
  is large enough to prove the misclassification term is not negligible;
- **when** an atom was lost, if it was: during frame 0, during the 10 ms gap,
  or during frame 1;
- an optimal exposure time — the array is nowhere near detector saturation, but
  that is not an exposure-time optimisation;
- generalisation to another day;
- a multi-frame transition hazard.

**Known data issues**, all recorded in [`docs/DATA_AUDIT.md`](docs/DATA_AUDIT.md)
and none of them silently repaired: frame 1 sits systematically below frame 0
by 12.5 counts/px;
the common-mode level is not stationary within the 231 s run; shot 0 is a
first-shot background outlier and is kept, not dropped; and two of 200 sites
carry a geometry-confidence flag. Sites are localised from atom fluorescence,
so a trap never loaded during these 100 shots could not be localised at all.

---

## Status

| Milestone | State |
|---|---|
| Data audit | done — [`docs/DATA_AUDIT.md`](docs/DATA_AUDIT.md) |
| Standardized data layer, schema 2.0 | done — [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md) |
| Image-level QC and background variants | done |
| Geometry validation and background remediation | done — [`docs/VALIDATION.md`](docs/VALIDATION.md) |
| Threshold and mixture baselines, 60/20/20 held-out split | **next** |
| Two-frame latent-state model | not started |
| Identification controls: matched-empty, dark frames, natural-loss | data not yet acquired |
| Multi-frame sequences, exposure scan, cross-day | out of scope for V0 |

The next milestone needs a shot-ordered 60/20/20 split, a threshold and mixture
baseline evaluated on held-out shots, and shot-cluster bootstrap intervals.
Until the identification controls exist, no result from this repository will be
labelled a fidelity or a loss rate.

## Layout

```
configs/     tracked machine-independent config; local paths stay untracked
docs/        data audit, validation report, repository-boundary data contract
src/         the fluorescence_inference package: schema, sites, background, QC
scripts/     audit, export, geometry gate, background comparison, assets
notebooks/   presentation only
tests/       schema, site finding, background models, geometry validation,
             determinism, migration, privacy, lab-parity
reports/     generated QC report and the README metric fragment
assets/      generated README figures and the hero animation
```

## License

MIT — see [LICENSE](LICENSE).
