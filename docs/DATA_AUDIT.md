# Data audit — V0 paired-readout dataset

Everything below was read from the shot files themselves. Where a fact could
not be established from the data it is listed as unresolved rather than
assumed.

**Regenerate:**

```bash
python scripts/audit_source_data.py --config configs/paired_100ms.yaml
```

Machine-readable output: `reports/audit/source_audit.json` (gitignored).

---

## 1. Sources inspected

| Source | Role | Access |
|---|---|---|
| `Experiment-Data/2026-07-27/0211/*.h5` | the 100 raw shots | read-only |
| `mit-tweezer-array-analysis` (`rydlab`) | image loading, ROI sums, local background, grid helpers | read-only, imported via `sys.path`, never installed into |
| this repository | standardization, QC, statistics | the only thing written to |

The general analysis checkout was not modified. It is put on `sys.path` rather
than pip-installed on purpose: an editable install would write build metadata
into a directory that is meant to stay untouched.

---

## 2. Dataset dimensions (verified)

| Fact | Value |
|---|---|
| Shot files | **100** |
| `run number` attribute | 0 … 99, unique, no gaps |
| `n_runs` attribute | 100 on every shot |
| Shots with both frames present | **100 / 100** |
| Frames per shot | **2** |
| Image datasets per frame | 1 (`ndim == 2`, not a frame bundle) |
| Image shape / dtype | **600 × 700**, `uint16` |
| Sequence index | 211, identical on all shots |
| Script name | one value, identical on all shots |
| Acquisition span | 19:54:43.78 → 19:58:34.95 on 2026-07-27, **231.2 s** |
| Mean shot period | **2.33 s** |
| Total raw bytes | 312,536,048 (~298 MiB) |

Derived table: 100 shots × 2 frames × 200 sites = **40,000 frame-site rows**,
**20,000 site-shot pairs**. The independent experimental units are still the
**100 shots**.

---

## 3. What the two frames actually are

Read from `devices/hamamatsu/EXPOSURES` and from the sequence script stored
inside each shot file.

| | frame 0 | frame 1 |
|---|---|---|
| Exposure name | `fluor` | `fluor2` |
| Frame type | `atoms` | `atoms` |
| Exposure start (s from sequence t=0) | 0.714111677 | 0.824111677 |
| Trigger duration | **0.1 s** | **0.1 s** |
| Camera `exposure` attribute | 0.1 | 0.1 |

Identical on all 100 shots — zero variation in any of these numbers.

**Temporal order is verified, not assumed.** Frame 0 precedes frame 1 both in
exposure start time and in the order the two `expose` calls appear in the
sequence script.

**Inter-frame timing.** Start-to-start separation is exactly **0.110 s** on
every shot, i.e. a 100 ms exposure followed by a **10 ms gap** before the
second 100 ms exposure. The gap equals the `SECOND_HAMAMATSU_FLUOR_DELAY`
global (0.01 s), and no other timed operation sits between the two exposures.

**Pulse history between the frames — verified from the script.** After frame 0
the science imaging light is switched off, the 10 ms delay elapses with the
light off, the light is switched back on, and frame 1 is exposed. The repump is
on continuously across both frames and is only turned off afterwards. The
absorption-imaging branch, which would have inserted extra operations between
the frames, is disabled in this run. So the two frames share the same nominal
imaging configuration, and the only difference between them is that frame 1
happens 110 ms later and follows one prior 100 ms illumination.

**Interpretation limit.** Both frames are labelled `atoms`. Neither is a
reference, a background, or a pre/post pair in the sense of a separate
reference exposure. There is no `frametype` distinction to exploit.

---

## 4. Site coordinates

### The frozen grid corners from the earlier day do not apply

The general analysis package carries grid corners measured on 2026-07-20. On
this run the array sits roughly 80 px away in the row direction from those
corners, so reusing them would have placed every ROI on empty sensor. Site
positions had to be re-derived from this run's own data.

### The mean image is the wrong image to find sites in

The 100-shot mean fluorescence image is dominated by a static, high-contrast
diagonal fringe pattern from the imaging light. The array is not cleanly
separable in it, and a high-pass peak search on the mean returns hundreds of
fringe maxima spread over the whole sensor.

### The variance map is the right one

The per-pixel variance across shots suppresses anything static and keeps
anything that fluctuates. Occupancy varies from shot to shot, so every loaded
site becomes a compact variance excess, and both sub-arrays appear cleanly. The
procedure implemented in `src/fluorescence_inference/sites.py`:

1. streaming per-pixel variance over all 200 frames;
2. high-pass (σ = 30 px) and smooth (σ = 0.8 px) — the lab's own trap-finding
   recipe, applied to the variance instead of the mean;
3. locate the array by heavy smoothing (σ = 12 px) and thresholding, so hot
   pixels elsewhere on the sensor cannot seed the search;
4. detect local maxima inside that region;
5. split into sub-arrays by k-means on coordinates (the lab helper);
6. per sub-array, estimate the two lattice vectors from the nearest-neighbour
   displacement distribution, index every peak against them, keep the densest
   10 × 10 index window, and least-squares fit an affine map
   `(row, col) → (y, x)`;
7. evaluate that map at all 100 index pairs, so a site with no detected peak
   still gets a modelled position and is measured.

### Result

| | grid A | grid B |
|---|---|---|
| Shape | 10 × 10 | 10 × 10 |
| Row pitch | 10.28 px | 11.21 px |
| Column pitch | 10.36 px | 11.15 px |
| Basis angle | 89.73° | 89.50° |
| Peaks used in the fit | 99 | 100 |
| Fit residual, median / max | 0.44 / 0.97 px | 0.49 / 0.83 px |

**Verified site count: 200** (not the ~50 assumed at project start). Sub-pixel
residuals across both sub-arrays are the evidence that the lattice model is
correct rather than merely fitted.

### The two flagged sites

`site_not_detected` marks a modelled position with no variance peak within
3 px. Two of 200 sites carry it, for two different reasons, and inspection
shows neither should be dropped:

* **site 29** (grid A, row 2, col 9): genuinely weak in the variance map — its
  peak pixel variance ranks 3rd lowest of 200. It is still a real site
  (mean background-corrected count 1450, P90 4170), just dimmer and more
  diffuse than its neighbours.
* **site 127** (grid B, row 2, col 7): not weak at all — it has the *highest*
  mean count of any site. The nearest detected peak sits 3.07 px away because
  a neighbouring diffuse feature merged with it during peak detection.

Both keep their modelled ROI and stay in the table with the flag set, so
downstream work can exclude them deliberately rather than silently.

### Independently validated in Round 1.5

The 200-site geometry is a fit result, so it was tested rather than trusted.
Full report: `reports/validation/site_geometry_validation.json`; method:
[`VALIDATION.md`](VALIDATION.md).

| Test | Result |
|---|---|
| Refit on shots 0–49 vs 50–99, optimally matched | median drift **0.080 px**, p90 0.117, p99 0.136, **max 0.149 px**, 0 unmatched of 200 |
| Bulk translation between halves | 0.069 px in row, 0.021 px in column |
| Array boundary sharpness | edge ring **38–64×** brighter than the first ring outside |
| Larger fitting window | **+2.0%** peaks at most — the window is not clipping |
| Two blocks = two images of one array? | **rejected**: index-matched correlation −0.003 against a permuted null of −0.001 |
| ROI overlays across the run | 10 shots, evenly spaced |

The boundary result is the informative one. Each block is exactly 10 × 10 with
edge sites as bright as interior sites and the next lattice position at the
noise floor. A larger lattice merely loaded in its middle would taper; this
does not. So the 10 × 10 window is a property of the data, not of the
configuration.

### Standing caveat

Sites are localised from atom fluorescence. **A trap never loaded during these
100 shots cannot be localised**, and an entire never-loaded block would be
invisible. The validation above establishes that the site set is stable,
unclipped, non-overlapping and non-duplicated. It does **not** establish that
each site is a physically verified trap — that needs a trap-light reference
image, which this run does not have. The phrase "200 valid traps" is therefore
not used anywhere in this repository.

---

## 5. ROI and background extraction

| Setting | Value |
|---|---|
| ROI | 5 × 5 px box (`trap_half_width = 2`), 25 px per site |
| Site exclusion mask | 5 px radius disk around every site, plus hot pixels |
| Spatial surface | total-degree-4 polynomial, Huber IRLS, 80,000 fit pixels |
| Camera offset | 0.0 counts/px (as frozen in the lab config) |

**The local annulus was demoted in Round 1.5.** It is retained only as a
diagnostic column, explicitly named `*_annulus_contaminated`.

With a 10–11 px site pitch the 13–33 px annulus contains roughly ten
neighbouring sites. The median keeps it from being dragged by them, but the
region is not atom-free, and the measured consequences are large:

* the annulus reports a frame 1 − frame 0 shift of **−36.56 ± 9.94 counts/px**
  against **−12.53 ± 1.48** from the site-masked estimator and **−13.77 ± 1.74**
  from the whole-frame median — roughly three times too large, with seven times
  the spread;
* its "corrected" count still correlates with its own background across shots
  at **|r| = 0.571**, against 0.15–0.22 for the site-masked estimators. A valid
  correction decorrelates the two; this one does not, because it is subtracting
  something that carries signal;
* it gives the *worst* two-component separation of any method tested
  (d′ = 2.76 / 3.17 against 2.97 / 3.41), so it is not even trading validity
  for contrast.

`background.annulus_background` still reproduces the lab's
`local_background_for_box` numerically (asserted by
`tests/test_reuse_matches_lab.py`); the problem is the estimator's geometry at
this site density, not the implementation.

---

## 6. Measurement variants carried in the table

Four background methods, in separate columns, none overwriting another.

| Variant | Corrected column | Definition |
|---|---|---|
| A | `roi_sum_raw` | no correction |
| B | `count_corrected_global` | site-free median of the same frame |
| C | `count_corrected_spatial` | robust degree-4 surface refitted per frame |
| **D** | `count_corrected_fixed_offset` | **primary** — fixed spatial template + per-frame common-mode offset |
| — | `count_corrected_annulus_contaminated` | diagnostic only, do not use |

All of B, C and D estimate the background from pixels outside a 5 px exclusion
disk around every site. A median over the sites is never used anywhere: a
change in the occupied fraction would then be absorbed into "background" and
make occupancy partly unobservable.

**D was selected on residual structure, not on separation.** It leaves
8.74 counts/px of block-median structure on site-free pixels, against
15.03 for C and 40.97 for B — a background model that leaves the fringe pattern
behind is wrong however flattering its histogram. Ranking on d′ was explicitly
rejected as a criterion because it rewards subtracting less, and would select
"no correction". Full comparison:
`reports/validation/background_method_comparison.json`.

**Honest limitation of C and D.** A 5 px exclusion radius at a 10–11 px pitch
masks about 74% of each array block, so both methods partly *interpolate* the
background underneath the array from surrounding site-free pixels. The
interpolation is smooth by construction and cannot represent structure finer
than the mask spacing. Enough unmasked pixels survive between sites
(**404,259 of 420,000 across the frame**) that the fit is well constrained
outside the blocks.

---

## 7. Metadata inventory

**Present and verified**

* root attributes: run number, run time, n_runs, sequence index, sequence date,
  script name;
* the full evaluated global-parameter set (139 names) on every shot;
* the camera exposure table with per-frame name, type, start time and trigger
  duration;
* camera configuration including the sensor sub-array window;
* the complete sequence script text, which is what made the pulse history
  between the frames verifiable.

**Absent**

* no per-frame acquisition timestamp (only a per-shot `run time`);
* no camera temperature, gain or read-noise record in the shot file;
* no trap-light or trap-position reference image;
* no site-labelling metadata — site identity is derived here, not supplied.

**Only one global varies across the run:** a repetition counter, taking 100
distinct values. It is a bookkeeping index, not a physics axis. Every physics
parameter is constant, which is what makes this a single-condition dataset.

---

## 8. Control data

| Control the analysis would need | Present? |
|---|---|
| Matched-empty two-frame sequence (same timing, same pulse history, no atoms) | **no** |
| Dark frames (imaging light off, same camera configuration) | **no** |
| Natural-loss control (atoms, imaging light off, same total wall-clock) | **no** |
| Pre / post reference exposure | **no** |
| Independent occupancy ground truth | **no** |

The audit does surface two `cam_mot` datasets named `exposure` and
`background`. They are **not** usable as controls for this measurement: they
come from a different camera (1080 × 1440 MOT camera, 0.2 ms exposure), not the
science camera, and the `background` frame is taken 100 ms after the repump is
switched off rather than under matched conditions.

**Consequence, stated plainly.** Without these controls this dataset cannot
identify a false-positive rate, a false-negative rate, a readout fidelity, or
an imaging-induced loss rate. Frame-to-frame disagreement is reported as
disagreement throughout.

---

## 9. Data problems found

1. **Frame-dependent background.** Frame 1 sits systematically below frame 0.
   Whole-frame median: **−13.77 ± 1.74 counts/px**, negative on **100 / 100**
   shots. On site-free pixels: **−9.34 ± 1.40 counts/px**. On the local
   annulus: **−36.56 ± 9.94 counts/px**. The annulus shift is about four times
   the site-free shift, consistent with the annulus carrying array-dependent
   light that itself decreases in frame 1. Any analysis that pools the frames
   without a per-frame background term is measuring this offset.

2. **Common mode is not stationary within the run.** The site-free level rises
   by roughly 5 counts/px around shots 45–60 and falls again after shot 80,
   with both frames tracking together. Over a 231 s run this is a real
   environmental drift, and it argues against a single background constant for
   the whole sequence.

3. **Shot 0 is an outlier in background.** Its mean annulus level is
   623 counts/px against a 591 counts/px median over the remaining shots. It is
   *not* excluded — it is a first-shot warm-up effect worth knowing about, and
   silently dropping it would be a worse decision than reporting it.

4. **Two sites carry the geometry flag** (section 4). No row is dropped.

**Not found:** no missing files, no missing frames, no truncated or corrupt
shot, no non-finite pixel, no duplicate primary key, no coordinate drift
requiring registration (the fitted lattice residual stays sub-pixel over the
whole run), and no inconsistent exposure metadata.

**No saturation, with room to spare.** The brightest single pixel anywhere in
the run is **2,038 counts** against a 16-bit ceiling of 65,535 — about 3% of
full scale. Whatever else limits this measurement, it is not detector
saturation. This does *not* license any claim about the optimal exposure time,
which this single-condition dataset cannot address.

---

## 10. Unresolved factual ambiguities

1. **Where atoms are lost, if they are lost.** The 10 ms gap, the frame-0
   exposure and the frame-1 exposure cannot be separated by two frames alone.
2. **How much of the frame 0 → frame 1 disagreement is misclassification.**
   The apparent dark→bright rate is not negligible (section: QC summary), which
   sets a floor on misclassification, but no control pins it down.
3. **Whether all traps are occupied-capable.** No trap-light reference exists,
   so a permanently empty trap is indistinguishable from no trap.
4. **Absolute count calibration.** No photons-per-count or atoms-per-count
   calibration is recorded in the shot files; all counts here are camera ADU.
5. **Whether the array geometry is exactly two 10 × 10 blocks.** Strongly
   supported by sub-pixel lattice residuals, not independently confirmed.
6. **Cause of the mid-run common-mode excursion.** Observed, not explained.

---

## 11. What this dataset can and cannot answer

**Can:**

* whether the bright and dark populations separate at 100 ms;
* whether there is a systematic background or signal shift from frame 0 to
  frame 1;
* how consistent the two frames' apparent occupancy calls are;
* how signal, background and spread vary across sites;
* the apparent bright→dark and dark→bright rates;
* which background treatment is more stable against acquisition order.

**Cannot:**

* true false-positive / false-negative rates or readout fidelity;
* imaging-induced loss rate;
* whether an atom was lost during frame 0, in the gap, or during frame 1;
* an optimal exposure time;
* generalisation across days;
* a multi-frame transition hazard.
