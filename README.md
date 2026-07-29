# Neutral-Atom Fluorescence Inference

From raw neutral-atom fluorescence images to held-out occupancy inference and operational imaging-loss diagnostics.

**Scientific question** — which fluorescence-count features generalize to
unseen shots, and what timing dependence remains after per-run geometry,
site-free background correction, frozen model selection, and complete-shot
uncertainty?

[**Measurements**](#from-fluorescence-images-to-measurements) ·
[**Experiments**](#data-and-experiments) ·
[**Statistical design**](#statistical-design) ·
[**Results**](#operational-loss-sweep-results) ·
[**Robustness**](#background-and-robustness) ·
[**Reproduce**](#reproduce) · [**Limitations**](#limitations)

## From fluorescence images to measurements

Per-run geometry maps two-dimensional fluorescence images to ROI counts
corrected with site-free pixels. The frame-site counts support descriptive or
held-out occupancy probabilities; **apparent occupancy** is used because no
empirical labels are available. The animation is a paired 100 ms extraction
example, not the repository-wide result.

<p align="center">
  <a href="assets/readme/fluorescence_inference_overview.gif">
    <img src="assets/readme/fluorescence_inference_overview.gif"
         alt="Paired-readout data-pipeline walkthrough: consecutive fluorescence frames, site ROIs, count extraction, background correction, a descriptive two-component fit, and paired counts."
         width="540">
  </a>
</p>

**Pipeline:** Raw frames → site geometry → ROI and site-free background → corrected frame-site counts → held-out emission model → apparent occupancy / retention

The independent experimental unit is a **shot**, not a site. Model-implied
component overlap is not empirical fidelity, and paired disagreement is not
automatically physical loss.

## Data and experiments

<!-- BEGIN:experiment-matrix -->
| dataset | shots | frames | exposure | swept variable |
|---|---:|---:|---:|---|
| paired readout | 100 | 2 | 100 ms | none |
| switch-off hold | 110 | 5 | 50 ms | 0.1–2.1 s |
| bright wait | 100 | 2 | 50 ms | 0.1–1.9 s |
<!-- END:experiment-matrix -->

<p align="center">
  <a href="assets/readme/sequence_design.png">
    <img src="assets/readme/sequence_design.png"
         alt="Commanded timing for the five-frame switch-off-hold sweep and the two-frame bright-wait sweep"
         width="620">
  </a>
</p>

*“Switch-off” describes a command state, not demonstrated optical darkness;
DDS frequency and amplitude commands remain configured during the hold.*

**Progression:** (1) paired 100 ms readout establishes extraction/background;
(2) five-frame 50 ms readout varies hold; (3) two-frame 50 ms readout varies
bright wait; (4) their comparison tests whether continuous bright-time decay
explains the apparent inter-readout loss.

The sweeps have independent geometry/background and are not pooled; see the
[command and shot-order audit](docs/DATA_AUDIT_2026-07-28_LOSS_SWEEPS.md).

## Statistical design

Each condition occurs once in each of ten complete acquisition cycles:

- cycles 0–5: training, six shots per condition;
- cycles 6–7: validation, two shots per condition;
- cycles 8–9: final test, two shots per condition.

Geometry, the fixed background template, and emissions are train-only;
validation selects structure, and test shots are scored once. Primary
intervals resample complete shots within condition. A separate whole-cycle
bootstrap preserves cross-condition cycle dependence as a sensitivity, not a
replacement confidence interval. Exact multiplicative loss accounting and
model details are in the [interpretation document](docs/LOSS_SWEEP_INTERPRETATION.md).

## Operational loss-sweep results

Parentheses for both reported time constants are 95% complete-shot cluster
confidence intervals. The apparent-gap row separately labels selected-model
sampling uncertainty and cross-structure sensitivity.

<!-- BEGIN:loss-sweep-results -->
| operational quantity | selected result and qualification |
|---|---:|
| `tau_switch_off` | 22.29 s (15.57–33.15 s) |
| `tau_bright_effective` | 1.64 s (1.50–1.83 s) |
| observed − predicted 50 ms apparent-loss gap | 10.26 percentage points (8.62–11.79 percentage points); 6.57–10.26 percentage points across the selected and unresolved floor structures |

Under the selected no-floor model, the apparent gap is 10.26 percentage points with a 95% complete-shot sampling interval of 8.62–11.79 percentage points. The unresolved floor alternative gives 6.57 percentage points, so the model-structure sensitivity is 6.57–10.26 percentage points. The sign remains positive, but the selected-model sampling interval is not total model uncertainty.

The simple constant-rate bright-wait model does not explain the full apparent inter-readout loss. This does not establish a fixed per-pulse mechanism.
<!-- END:loss-sweep-results -->

| ![Four-panel overview of switch-off retention, interval factors, bright-wait occupancy, and the post-wait control](assets/readme/loss_sweep_overview.png) |
|:--|
| Points summarize all 10 shots per condition; selected structures and fitted bands follow the frozen train/validation workflow. Intervals use complete shots as clusters. |

`tau_switch_off` is an operational time constant under the commanded
switch-off configuration: realized optical extinction was not measured and
cooling was not optimized. `tau_bright_effective` is specific to the recorded
illuminated-wait configuration. Neither is an intrinsic or best-achievable
apparatus lifetime.

The 8.62–11.79 percentage-point interval is sampling uncertainty conditional
on the validation-selected no-floor structure. The 6.57–10.26
percentage-point range compares that selected structure with the unresolved
floor alternative; it is **not** a confidence interval or total uncertainty.
Both structures leave a positive apparent gap.

The central conclusion is deliberately narrow: the simple constant-rate
bright-wait model does not explain the full apparent inter-readout loss. The
gap does not identify a fixed per-pulse cost, switching transient,
classification error, nonstationary heating, state selection, or any other
single mechanism.

## Background and robustness

| ![Per-frame site-free background change relative to the shortest sweep point](assets/readme/background_drift_sweeps.png) |
|:--|
| Selected site-free background by frame, expressed relative to that frame’s shortest sweep point. Error bars retain the pointwise complete-shot cluster interval widths; they are not paired-difference intervals. |

No correction, global site-free median, robust spatial surface, and a fixed
training template plus per-frame site-free offset were compared. Validation
selected the template-plus-offset method for both sweeps using physical
masking, residual structure, drift, and background coupling—not by maximizing
count separation.

The legacy local annulus is diagnostic only. At the 10–11 px pitch it
intersects neighbouring-site masks for every sweep site, so it cannot serve as
the primary background estimator. The first switch-off frame also has a large
site-free pedestal, and the bright-wait background changes with sweep
position; neither effect is estimated from the occupancy-dependent low tail of
trap counts.

The switch-off endpoint sensitivity keeps the validation-frozen shared-slope
structure. Removing the shortest point, the longest point, or both moves
`tau_switch_off` from 22.29 s to 21.54 s, 20.57 s, or 19.42 s,
respectively; all four fits remain positive and non-boundary. The whole-cycle
bootstrap gives an apparent gap of 10.26 percentage points
(8.81–11.61 percentage points), consistent with the primary conclusion.

Sweep values ascend in fixed order within every cycle. Repeated cycles and
complete-shot clustering capture repeat variability, but no analysis can
separate a true timing effect from an unexplained drift locked perfectly to
within-cycle position. This remains a design limitation, not a fitted-away
nuisance.

<!-- BEGIN:sweep-validation -->
Both sweep timing audits, independent per-run geometry gates, the selected site-free background (template + frame offset for both sweeps), and frozen complete-shot splits pass. The two runs remain separate because they are distinct acquisitions with different geometry and background trajectories. The exploratory latent-state model is not accepted as a public result because complete-shot clustered uncertainty for its transition parameters is unavailable.
<!-- END:sweep-validation -->

Detailed geometry shifts, A/B/C/D background metrics, representative-shot
rules, endpoint checks, and cycle sensitivities are in
[V0 validation](docs/VALIDATION.md),
[the V0 audit](docs/DATA_AUDIT.md), and
[the sweep audit](docs/DATA_AUDIT_2026-07-28_LOSS_SWEEPS.md).

## Scientific interpretation

The three datasets support a narrow chain of inference:

- the paired readout validates extraction, background handling, and
  descriptive readout consistency;
- held-out sweep counts support apparent occupancy under the selected emission
  model;
- the switch-off sweep supports an operational decay rate and apparent fixed
  interval factors `q_j`;
- the bright-wait sweep supports an effective apparent-occupancy decay and a
  separate post-wait apparent-retention control;
- their comparison rejects the simple constant-rate bright-wait explanation
  as sufficient for the full apparent inter-readout loss.

Validation selected a flat post-wait retention control. Its κ = 0 is fixed by
that selected structure, not estimated as a precise zero trend. The unselected
monotone sensitivity places κ on its boundary and does not
resolve a trend; neither result identifies heating.

The exploratory latent-state fit improves ordinary held-out likelihood, but
its public gate remains closed because complete-shot clustered transition
uncertainty is unavailable. Synthetic recovery passed its declared threshold
but retained 28.1% relative rate error, supporting predictive sequence
structure rather than a precise physical rate. No latent trajectory or
transition rate is promoted as a physical result.

The [claim table](docs/LOSS_SWEEP_INTERPRETATION.md) gives each quantity’s
operational definition, assumptions, uncertainty method, permitted claim, and
prohibited claim.

## Reproduce

Create the untracked local path configuration, install the package in a
repository-local environment, and run:

```text
.\.venv\Scripts\python.exe scripts\reproduce_all.py
```

The command requires a clean worktree, audits all three datasets, regenerates
the standardized local exports, validation reports, reviewed result, public
assets, generated metrics, full tests, and publication manifest. It never
substitutes synthetic measurements when raw data are unavailable.

The exact Windows commands, standalone stages, and non-overwriting reviewed
verification workflow are in [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Documentation

- [V0 data audit](docs/DATA_AUDIT.md)
- [2026-07-28 sweep audit](docs/DATA_AUDIT_2026-07-28_LOSS_SWEEPS.md)
- [Detailed validation](docs/VALIDATION.md)
- [Data contract and provenance](docs/DATA_CONTRACT.md)
- [Scientific claim table and robustness values](docs/LOSS_SWEEP_INTERPRETATION.md)
- [Prototype discrepancy ledger](docs/PROTOTYPE_DISCREPANCY_LEDGER.md)
- [Generated public metrics](reports/readme_metrics.md)
- [Prioritized next experiments](docs/NEXT_EXPERIMENTS.md)

## Limitations

- The nominal dark hold commands the imaging switches off while DDS frequency
  and amplitude commands remain configured. Without an extinction
  measurement, `tau_switch_off` is not an intrinsic dark lifetime.
- Cooling was not optimized, so neither reported rate is a best-achievable
  apparatus benchmark.
- Both sweeps are from one date with only 10 independent shots per condition;
  sites and frames within a shot are not additional experimental repeats.
- Cycles 0–5/6–7/8–9 define train/validation/test. The final test contains only
  two shots per condition.
- Sweep values ascend in fixed order within every cycle, perfectly confounding
  condition with within-cycle position.
- There is no empirical occupancy ground truth, matched-empty fluorescence
  sequence, complete dark-frame control, or pre/post nondestructive reference.
- The first switch-off frame has a large site-free pedestal. The sweep runs
  also differ by about one lattice pitch and have different background
  trajectories, so their rates are compared but their data are not pooled.
- Interval factors `q_j` mix apparent fixed loss, classification error,
  sequence transients, and state selection.
- The paired 100 ms mixture is a full-data descriptive fit, not held out, and
  paired disagreement is not identified as physical loss.
- The selected no-floor sampling interval is conditional on one model
  structure. The 6.57–10.26 percentage-point structural range is not a
  confidence interval or total model uncertainty.
- Endpoint deletion and whole-cycle bootstrap checks support the qualitative
  result, but they do not remove the fixed-order confounding or establish a
  causal mechanism.
- The selected flat post-wait model fixes κ = 0 structurally; it does
  not measure a precisely zero physical heating trend.
- The latent model lacks complete-shot clustered transition uncertainty and
  has about 28.1% synthetic rate-recovery error, so it remains exploratory.
- The observed-minus-predicted apparent gap leaves the mechanism unidentified.
  A randomized exposure-duration and pulse-count experiment is required to
  test a fixed-per-pulse hypothesis.

## License

MIT — see [LICENSE](LICENSE).
