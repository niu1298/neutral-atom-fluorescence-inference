# Data audit — 2026-07-31 optimized lifetime measurements

This audit separates facts stored in the shot files from interpretations of
compiled sequence commands and from physical facts supplied by the
experimenter. No result below is an occupancy label or an optical readback.

Machine-readable source audits are written under `reports/audit/`; geometry
and background reports are under `reports/validation/`. Those products are
ignored because they contain detailed experimental diagnostics. The tracked
configs and this reduced audit contain the public contract.

## Scope and completeness

| Measurement | Shots | Frames | Exposure | Start-to-start interval | Complete | Science image |
|---|---:|---:|---:|---:|---:|---|
| Bright lifetime | 200 | 2 | 50 ms | 60.002 ms between readouts | 200 / 200 | 600 × 700 `uint16` |
| Dark lifetime | 220 | 5 | 50 ms | hold + 50.002 ms | 220 / 220 | 600 × 700 `uint16` |
| Five-frame imaging | 100 | 5 | 50 ms | 60.002 ms | 100 / 100 | 600 × 700 `uint16` |
| Five-frame imaging | 100 | 5 | 100 ms | 110.002 ms | 100 / 100 | 600 × 700 `uint16` |
| Five-frame imaging | 100 | 5 | 200 ms | 210.002 ms | 100 / 100 | 600 × 700 `uint16` |

Across all 720 shots, shot identifiers are unique within their run, every
configured science frame is present exactly once, and the exposure-table frame
names are `fluor`, `fluor2`, and, where applicable, `fluor3` through `fluor5`.
No science image contains a saturated or non-finite pixel. The sequence index
is respectively 90, 94, 113, 114, and 115; every shot embeds the same sequence
script basename.

The shot files also contain a 1080 × 1440 `uint16` MOT-camera exposure and
background frame. Those are from a different camera and are not a
timing-matched empty-site or science-camera background control.

## Acquisition design and exact commanded timing

### Bright lifetime

`WAIT_BEFORE_FIRST_FLUOR` takes ten values from 0.1 s to 8.1 s, in fixed
ascending order, repeated for 20 complete cycles. The first 12 cycles are
training, the next four validation, and the last four test, giving 120 / 40 /
40 independent shots and 12 / 4 / 4 shots per wait value.

The varied wait is continuous commanded illumination before frame 0. Both
frames are 50 ms. Their compiled start separation is 60.002 ms: 50 ms of
exposure, the declared 10 ms dark gap, and 2 µs of command-update overhead.
The switch-low lead before frame 0 is the requested bright wait plus 1 µs of
command-update overhead.

### Dark lifetime

`SECOND_HAMAMATSU_FLUOR_DELAY` takes 11 values from 0.1 s to 2.1 s in 0.2 s
steps. Conditions occur in fixed ascending order within each of 20 complete
cycles. The split is 12 / 4 / 4 cycles, or 132 / 44 / 44 independent shots and
12 / 4 / 4 shots per hold value.

Every shot contains five 50 ms frames. Each later frame starts after the
declared dark hold, the preceding 50 ms exposure, and 2 µs of command-update
overhead. Thus the observed start-to-start values are 0.150002, 0.350002, …,
2.150002 s. The physical survival model uses the declared 0.1, 0.3, …, 2.1 s
dark holds, not the exposure duration or command overhead.

### Repeated imaging

Runs at 50, 100, and 200 ms contain 100 chronological fixed-condition shots
each. `REPEAT` runs from 0 through 99 and matches acquisition order. Frozen
splits are shots 0–59 training, 60–79 validation, and 80–99 test. A seeded
permutation of five contiguous 20-shot blocks is retained as a drift
sensitivity.

The declared dark gap is exactly 10 ms. The compiled start-to-start values are
exposure + 10.002 ms, so the 2 µs overhead is recorded separately from dark
time. For prefix `N`, schema 4.0 therefore records
`cumulative_bright_s = N × exposure_s`, `cumulative_dark_s = (N - 1) × 0.01`,
and `pulse_count = N`.

## Compiled command evidence

The audit independently compares the camera exposure table, compiled digital
pulse program, compiled DDS programs, and embedded sequence source for every
shot. All 720 command audits pass:

- camera-trigger edges agree with the exposure table within the compiled
  digital-clock tolerance;
- both non-inverted science switches are in the commanded-on state throughout
  bright waits and exposures;
- both switches are in the commanded-off state throughout every declared dark
  interval;
- configured imaging DDS channels are on during bright stages and have zero
  commanded amplitude during dark gaps;
- equivalent DDS science-stage states are constant across every shot of a run.

This proves consistency among compiled commands. It does not prove that a
camera, switch, DDS, or optical field physically followed those commands:
there is no independent optical-power, extinction, or per-frame hardware
timestamp readback. The experimenter-supplied physical fact is that the
relevant imaging light is fully off during the dark intervals.

## Apparatus variables and comparability

Only the target sweep changes in the lifetime runs. Only `REPEAT` changes in
the three fixed-condition runs. The imaging-stage power, detuning, repump,
trap-power, and magnetic-bias commands recorded for all five runs are constant
at the public values in the run sidecars. The three repeated-imaging runs match
one another except for camera exposure.

Bright-lifetime run 0090 uses different pre-heat power commands from 0094 and
0113–0115, although its main imaging power, detuning, repump, trap, and bias
commands match. Consequently, the bright rate from 0090 can constrain a
repeated-imaging model only with an explicit condition-comparability
sensitivity; it must not be treated as a universal fixed hazard without that
check.

## Geometry

Geometry is fitted independently per run from training shots only. Every gate
passes the required early/late or run-start/run-end refits, integer-lattice
registration, residual-distortion checks, and manual ROI overlay inspection.

| Measurement | Expected sites | Variance peaks within 3 px | Split used to fit |
|---|---:|---:|---|
| Bright lifetime | 100 | 89 | training cycles |
| Dark lifetime | 100 | 98 | training cycles |
| Five-frame 50 ms | 100 | 99 | training shots |
| Five-frame 100 ms | 100 | 97 | training shots |
| Five-frame 200 ms | 100 | 99 | training shots |

A missing nearby variance peak is a localization-confidence flag, not evidence
that a site is empty. All expected lattice positions retain their modelled ROI;
weak-site exclusion is predeclared from training-only emission diagnostics and
is used only as a sensitivity.

## Background diagnostics and frozen selection

Raw, global site-free, robust spatial, and fixed-training-template corrections
are all carried separately. The legacy annulus intersects neighbouring-site
masks at all 100 sites in every run and remains diagnostic only.

Background selection uses validation-shot site-free residual structure, with
corrected-count/background coupling as a physical-validity gate. It does not
use mixture separation or test shots.

| Measurement | Selected method | Residual block-median SD | Median absolute coupling | Validation shots |
|---|---|---:|---:|---:|
| Bright lifetime | fixed template + frame offset | 4.63 | 0.227 | 40 |
| Dark lifetime | fixed template + frame offset | 4.90 | 0.167 | 44 |
| Five-frame 50 ms | per-frame robust spatial surface | 6.55 | 0.063 | 20 |
| Five-frame 100 ms | fixed template + frame offset | 6.42 | 0.294 | 20 |
| Five-frame 200 ms | fixed template + frame offset | 11.82 | 0.112 | 20 |

All selected methods pass the predeclared median absolute coupling gate of
0.50. The 50 ms repeated run selects the spatial model over the template
(residual structure 6.55 versus 7.86); the tracked config records that result.
Its already-recovered development export contains both columns and remains
usable for code development, but its sidecar predates this one-field selection
update. The final clean reproduction must refresh that export and its hashes
before any reviewed result is published.

Site-free pedestal is exposure- and frame-dependent. Relative to later frames,
the median frame-0 pedestal is higher by 75 counts per ROI at 50 ms, 100 counts
at 100 ms, and 125 counts at 200 ms. The bright-lifetime and dark-lifetime
runs show corresponding first-versus-later differences of 87.5 and 50 counts
per ROI. These shifts are why emissions may vary by frame and exposure and why
background cannot be estimated from the low tail of trap counts.

## Standardized products

The recovered schema-4.0 exports validate structurally and preserve complete
shots across splits:

| Dataset | Rows | Sites | Split shots |
|---|---:|---:|---|
| Bright lifetime | 40,000 | 100 | 120 / 40 / 40 |
| Dark lifetime | 110,000 | 100 | 132 / 44 / 44 |
| Five-frame 50 ms | 50,000 | 100 | 60 / 20 / 20 |
| Five-frame 100 ms | 50,000 | 100 | 60 / 20 / 20 |
| Five-frame 200 ms | 50,000 | 100 | 60 / 20 / 20 |

Each product has a table, site table, metadata sidecar, run sidecar, and split
manifest bound to a tracked config hash and raw input-manifest hash. Geometry
and the fixed background template use training shots only. Full processed
tables and raw shots remain ignored.

## Fact classes and unresolved limits

Raw metadata facts are shot counts, frame names, shapes, dtypes, exposure-table
timing, evaluated globals, image statistics, and sequence identifiers.
Sequence-code interpretations are the named bright, exposure, and dark stages
deduced by matching compiled commands. The fully-dark optical interpretation
is an experimenter-supplied physical fact consistent with, but not measured by,
the compiled command audit.

Unresolved limitations are:

- occupancy is latent; naturally empty sites do not provide external labels;
- there is no empirical readout-fidelity measurement;
- there is one run per repeated-imaging exposure, so exposure and run are
  partly confounded;
- optical extinction and exact loss times are not observed;
- the 0090 pre-heat difference limits a universal bright-rate claim;
- acquisition order is fixed rather than randomized, so cycle/block and
  shot-order sensitivities are required.

These limits do not prevent apparent-occupancy, operational lifetime, or
matched-prefix analyses, but they constrain the wording of physical claims.
