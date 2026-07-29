# Data audit: 2026-07-28 loss sweeps

This audit covers the two fluorescence loss sweeps acquired on 2026-07-28:

- sequence 0044: five-frame, 50 ms switch-off-hold sweep;
- sequence 0050: two-frame, 50 ms bright-wait sweep.

The paired 100 ms sequence from 2026-07-27 (0211) was rechecked as the
backwards-compatibility reference. Raw HDF5 shots are read-only and remain
untracked. The machine-readable audit manifest is generated under
`reports/audit/`, which is also untracked.

## Audit status

| gate | status | evidence |
|---|---|---|
| files open and expected shots exist | pass | 110/110 in 0044; 100/100 in 0050 |
| expected fluorescence frames exist | pass | five per 0044 shot; two per 0050 shot |
| commanded exposure and frame order recoverable | pass, with metadata conflict | compiled trigger and exposure schedule agree on 50 ms |
| switch command state recoverable | pass | compiled digital outputs agree in every shot |
| DDS command state recoverable | pass | commands remain unchanged through waits and frames |
| independent optical-dark verification | not available | no extinction or leakage measurement |
| per-frame hardware timestamp/readback | not available | only commanded pseudoclock timing is stored |
| sweep randomization | fail | fixed ascending order inside every repeated cycle |
| run-specific geometry | pass | training-only grids; required subset comparisons pass |
| site-free background selection | pass, with sensitivity flag | fixed template plus per-frame offset minimizes residual spatial structure |
| physical-rate models | evaluated downstream | no physical-rate claim follows from this audit alone |

The timing audit therefore supports inference against the **commanded**
switch-off and bright-wait durations. It does not establish a fully dark
optical state or measure the realized exposure with an independent clock.

## Source hierarchy

The following labels are used throughout this document.

- **Direct metadata fact**: read from the HDF5 attributes, datasets, compiled
  device program or saved image arrays.
- **Code-derived fact**: reconstructed from the embedded sequence source or
  deterministic acquisition expressions.
- **Experimenter context**: supplied with the dataset but not independently
  measured by these files.
- **Prototype assumption**: found in the reference notebooks and retained only
  as an audit comparison.
- **Unresolved**: not identified by the available measurement.

Saved notebook plots and outputs are not sources of truth for public numbers.

## Dataset inventory

| dataset | complete shots | incomplete | fluorescence frames | auxiliary frames | shape / dtype | dtype-limit saturation |
|---|---:|---:|---:|---:|---|---:|
| paired 100 ms, 0211 | 100 | 0 | 200 | not used here | 600 x 700 / uint16 | 0 pixels |
| switch-off hold, 0044 | 110 | 0 | 550 | 220 | 600 x 700 / uint16 | 0 pixels |
| bright wait, 0050 | 100 | 0 | 200 | 200 | 600 x 700 / uint16 | 0 pixels |

Direct metadata and image-integrity checks found:

- consecutive, unique camera run numbers (`0..109` and `0..99`);
- `failed_shot=False` for both recorded cameras in every new shot;
- no missing expected frame;
- no exact duplicate image SHA-256 within either sequence;
- no pixel equal to the uint16 maximum;
- one identical HDF5 group/dataset schema within each sequence.

Absence of a uint16-maximum pixel is not a calibrated camera full-well test.

## HDF5 structure

Every new shot contains the same top-level entries:

```text
calibrations
connection table
data
devices
front_panel
globals
images
labscriptlib
script
shot_properties
time_markers
waits
```

The fluorescence arrays are stored at
`images/hamamatsu/<frame-name>/atoms`. Sequence 0044 uses `fluor`, `fluor2`,
`fluor3`, `fluor4`, and `fluor5`; sequence 0050 uses `fluor` and `fluor2`.
The auxiliary camera's background image is from a different camera and is not
a matched empty fluorescence reference.

## Exact commanded timing

### Sequence 0044: switch-off hold

For saved hold value \(d\), in seconds, the commanded start of zero-based frame
\(j\) is

\[
t_j = 0.6842116770321879 + j(0.05+d), \qquad j=0,\ldots,4.
\]

Each trigger is 0.05 s. Consequently:

- the end-to-next-start switch-off gap is exactly \(d\);
- the start-to-start interval is \(0.05+d\);
- the first frame is preceded by a fixed 0.0001 s light-on wait.

The eleven saved hold values are `0.1, 0.3, ..., 2.1 s`, with ten shots at
each value.

### Sequence 0050: bright wait

For saved bright wait \(w\), in seconds:

\[
\begin{aligned}
t_\mathrm{fluor} &= 0.6841116770321879+w,\\
t_\mathrm{fluor2} &= t_\mathrm{fluor}+0.06.
\end{aligned}
\]

Both triggers are 0.05 s, leaving a commanded 0.01 s switch-off gap. The ten
saved waits are `0.1, 0.3, ..., 1.9 s`, with ten shots at each value.

Across all 210 new shots, the compiled 100 MHz digital edges agree with the
saved exposure schedule to within 47.1 ns. This is agreement between two
representations of the commanded program, not an independent camera timestamp.

## Camera metadata conflict

The following sources agree on 50 ms for both new datasets:

- the configured camera exposure global;
- the device-level exposure schedule;
- the Hamamatsu group configuration;
- the compiled trigger pulse.

The leaf image datasets instead carry stale attributes claiming a 5 ms
exposure and a 256 x 16 subarray, although the stored arrays are 600 x 700 and
the group-level subarray is 700 x 600. The leaf attributes are therefore not
used for timing or shape.

The standardized data contract records:

- `exposure_s=0.05` as a corroborated **commanded** exposure;
- measured exposure as null;
- per-frame hardware timestamp as null;
- a run-level explanation of the metadata conflict.

The same conflict exists in the paired 100 ms reference: its compiled schedule
and device-level exposure are 100 ms while the stale leaf attribute says 5 ms.

## Switch and DDS command state

The embedded connection metadata and compiled digital program map two
science-imaging switch outputs and the camera trigger. The embedded sequence
semantics define low as imaging light commanded on and high as the optical
switches commanded off.

For sequence 0044:

- both imaging switches are commanded on during every 50 ms exposure;
- both are commanded off throughout every swept inter-frame hold;
- the DDS frequency and amplitude commands remain unchanged throughout.

For sequence 0050:

- both imaging switches are commanded on during the swept pre-first-frame wait
  and the first exposure;
- both are commanded off for the 10 ms inter-frame gap;
- both return on for the second exposure;
- the DDS frequency and amplitude commands remain unchanged throughout.

The saved DDS command arrays are constant:

| device | channel frequencies (MHz) | amplitude commands |
|---|---|---|
| DDS 0 | 80, 80, 67.5, 100 | 0.65, 0.60, 1.00, 0.60 |
| DDS 1 | 80, 80, 80, 80 | 0.20, 0.00, 0.14, 1.00 |

These amplitudes are control commands, not calibrated optical powers. No
photodiode, extinction measurement, leakage monitor or independent switch
readback is present. The defensible public term is therefore **switch-off
hold**, not fully dark hold or intrinsic dark lifetime.

Under the additional physical assumption that residual leakage can only add
loss, an operational switch-off lifetime would be a lower bound on a fully
dark lifetime (and its rate an upper bound). The data do not test that
assumption.

## Globals and sequence provenance

Each new shot stores the same 141 evaluated global names.

- In 0044, only `SECOND_HAMAMATSU_FLUOR_DELAY` changes.
- In 0050, only `WAIT_BEFORE_FIRST_FLUOR` changes.
- Between the two sequences, the recorded differences are the number of
  fluorescence frames, the inter-frame hold, and the pre-first-frame wait.

The new sequences embed byte-identical sequence source and connection-table
source. Their hashes are recorded in the ignored audit manifest. No labscript
Git commit or package version is stored in the HDF5 files, so those acquisition
provenance fields remain null with an explanation. The analysis dependency
commit is recorded separately from the read-only laboratory-analysis checkout.

## Acquisition order and confounding

Neither sweep was randomized.

- 0044 repeats an ascending eleven-point cycle ten times:
  `condition_position = shot_order mod 11`.
- 0050 repeats an ascending ten-point cycle ten times:
  `condition_position = shot_order mod 10`.

Condition is perfectly confounded with position inside a cycle. It is not
perfectly confounded with whole-run chronology because the cycle repeats:
the linear correlations of condition with full-run shot order are about 0.094
and 0.096. Nevertheless, every cycle starts at the shortest condition and ends
at the longest, and longer conditions also make the shot itself longer.

The primary held-out split therefore keeps complete acquisition cycles intact.
The analysis also reports chronological/interleaved-cycle and within-cycle
drift sensitivities. No statistical model can remove an unmeasured systematic
effect that repeats at the same within-cycle phase as the condition.

The stored `REPEAT` global is constant. `repetition_index` is therefore an
explicitly code-derived cycle index, not direct metadata.

## Run-specific geometry validation

This section is **code-derived image-space inference**, not HDF5 metadata.
Geometry was fitted independently for each run from training cycles only:
66 shots for 0044 and 60 shots for 0050. The 100 modelled sites form one
10 x 10 grid in each run. Peak proximity is a diagnostic, not a requirement
for retaining a lattice-predicted site.

| run | pitch (px) | tilt (deg) | centre \(y,x\) (px) | detected variance peaks | detected-peak residual, median / max (px) |
|---|---:|---:|---:|---:|---:|
| 0044 | 11.205451 | -8.686355 | 331.076923, 380.923077 | 51 / 100 | 0.492548 / 2.899704 |
| 0050 | 11.233032 | -8.833857 | 342.037690, 382.587581 | 30 / 100 | 0.712770 / 2.786487 |

The lower detected-peak counts do not mean that only 51 or 30 sites were
analysed. The full training-fitted lattice predicts all 100 positions. They do
mean that a trap-light reference is still needed to verify whether every
modelled position is a physical trap.

Subset gates compare all 100 lattice positions. The table reports the
registered representation actually used by each gate. Ten-shot endpoint
subsets use a rigid translation of the training-frozen pitch and tilt when
their free fit cannot identify the periodic index origin.

| run | comparison | gate representation | median (px) | p90 (px) | p99 (px) | maximum (px) |
|---|---|---|---:|---:|---:|---:|
| 0044 | early vs late | registered free grids | 0.000000115 | 0.000000152 | 0.000000172 | 0.000000179 |
| 0044 | shortest vs longest hold | frozen-shape rigid translation | 0.007338 | 0.007338 | 0.007338 | 0.007338 |
| 0044 | early vs late within training | registered free grids | 0.000000748 | 0.000001093 | 0.000001260 | 0.000001320 |
| 0044 | training vs all-data unsupervised | registered free grids | 0.000000084 | 0.000000134 | 0.000000157 | 0.000000168 |
| 0050 | early vs late | registered free grids | 0.144703 | 0.222326 | 0.247733 | 0.258097 |
| 0050 | shortest vs longest wait | frozen-shape rigid translation | 0.087901 | 0.087901 | 0.087901 | 0.087901 |
| 0050 | early vs late within training | registered free grids | 0.065300 | 0.106685 | 0.120697 | 0.127407 |
| 0050 | training vs all-data unsupervised | registered free grids | 0.120620 | 0.194851 | 0.216854 | 0.228295 |

All comparisons match 100/100 positions with no unmatched site and pass the
predeclared limits: median below 0.5 px, p90 below 0.8 px, p99 below 1.2 px
and maximum below 1.5 px. The nearly zero 0044 values reflect repeated
selection of the same discrete pitch/tilt template after registration; they
must not be interpreted as nanometre-scale localization precision.

Free sparse-subset fits in 0050 exhibit periodic aliases: shortest versus
longest selects a three-row index displacement, and training versus all-data
selects a one-row displacement. Both disappear when the training-frozen
lattice shape is translated directly. This is why the geometry is frozen per
run and why endpoint subsets are not allowed to redefine site indices.

Across runs, the fitted-centroid displacement is
\((\Delta y,\Delta x)=(10.960767,1.664504)\) px, with magnitude 11.086432 px.
The nearest 0044 lattice vector is \((11.076923,1.692308)\) px; the residual
is 0.119438 px and the corresponding index step is one row. Recorded camera
shape, crop and offset metadata are identical. The image data therefore show
a one-pitch displacement but do not identify whether it is a real array
translation, an unrecorded state change or a one-index alias.

## Site-free background validation

This section is also **code-derived inference**. Every eligible estimator
excludes a 5 px radius around every modelled site. Method selection uses only
site-free residual structure and background coupling; count separation is
explicitly excluded from the selection criterion.

| run | method | eligible as primary | median / p90 block-median residual (counts) | maximum absolute per-frame coupling |
|---|---|---|---:|---:|
| 0044 | global site-free median | yes | 19.2358 / 24.7110 | 0.3690 |
| 0044 | robust spatial surface | yes | 7.5183 / 8.9276 | 0.2752 |
| 0044 | fixed template + frame offset | **selected** | **5.1739 / 8.4259** | 0.3432 |
| 0050 | global site-free median | yes | 19.5212 / 19.7830 | 0.1436 |
| 0050 | robust spatial surface | yes | 8.5056 / 8.7720 | 0.0875 |
| 0050 | fixed template + frame offset | **selected** | **3.9608 / 4.7070** | 0.1195 |

Raw counts are ineligible because they do not estimate background. The local
annulus is ineligible because it intersects neighbouring-site masks:

| run | sites with overlap | median / p90 / max annulus overlap | median / max neighbouring masks intersected |
|---|---:|---:|---:|
| 0044 | 100 / 100 | 66.03% / 67.50% / 68.15% | 11 / 12 |
| 0050 | 100 / 100 | 66.20% / 67.17% / 67.72% | 11 / 12 |

Thus the annulus's weak row-level correlation after subtraction is not
evidence that it is physically valid: roughly two thirds of its pixels lie
inside neighbouring-site masks.

### First-frame pedestal in 0044

The notebook's two-component empty-trap tail placed frame 1 near 10,478.5 raw
ROI counts and later frames near 9,365--9,430, a spread of about 1,114 counts.
That estimator is endogenous to occupancy and emission overlap. Direct
site-free medians instead give:

| frame | site-free median in one 5 x 5 ROI (counts) | mean across shots | shot-to-shot SD |
|---:|---:|---:|---:|
| 1 | 9,125 | 9,120.00 | 48.58 |
| 2 | 8,900 | 8,895.45 | 41.22 |
| 3 | 8,900 | 8,894.55 | 43.29 |
| 4 | 8,900 | 8,895.91 | 42.91 |
| 5 | 8,900 | 8,892.50 | 41.69 |

The site-free first-versus-later pedestal is therefore 225 counts per ROI,
not 1,114 counts. The selected template estimator, which also represents the
fixed spatial pattern under each ROI, gives 9,541.08 counts for frame 1 and
9,318.11--9,320.85 for frames 2--5. It removes the global and spatial
pedestal while retaining frame-specific emission parameters; it does not
reinterpret the remaining frame difference as physical loss.

### Sweep-dependent background in 0050

The site-free median is 9,050 counts per ROI in frame 1 and 8,975 in frame 2,
a 75-count frame offset. From the shortest to longest wait, the estimated
background changes are:

| estimator | frame 1 change (counts/ROI) | frame 2 change (counts/ROI) |
|---|---:|---:|
| global site-free median | +20.00 | +12.50 |
| robust spatial surface | +16.93 | +7.79 |
| fixed template + frame offset | +20.27 | +12.13 |
| contaminated local annulus | -152.85 | -137.29 |

The notebook's low-tail trap estimator reported approximately -137 and -106
counts over the sweep. The sign and scale do not agree with the site-free
estimators, while the contaminated annulus reproduces a large negative drift.
This supports the interpretation that the trap-tail/annulus drift is coupled
to declining apparent occupancy rather than measuring only camera
background.

The selected template's background-versus-apparent-occupancy correlations are
-0.282 and -0.135 in the two bright-wait frames, compared with +0.851 and
+0.817 for the contaminated annulus. In 0044 the template correlations range
from -0.414 to +0.325 across frames. Because condition is fixed within cycle,
these correlations cannot distinguish optical background, loading and
within-cycle drift. They are carried into background-method sensitivity
analyses rather than treated as proof that correction is perfect.

## Cross-run camera coordinates

Sequences 0044 and 0050 store identical camera shape, subarray size, subarray
offset, sequence-source hash, and connection-source hash. A fitted
approximately one-pitch array displacement between the runs cannot be
explained by a recorded crop or coordinate change. It must be resolved by
independent image-space geometry fits; candidates include a real translation,
an unrecorded state change, or a one-index lattice-fit alias.

## Experimenter context not encoded in HDF5

- Cooling parameters were not optimized.
- Atom temperature and cooling performance may therefore be suboptimal.
- Results characterize this operating condition, not the apparatus's best
  achievable performance.

These are required interpretation caveats, not claims inferred from the
fluorescence arrays.

## Missing identifying controls

Neither new sequence contains:

- empirical occupancy labels;
- a matched empty Hamamatsu reference;
- a complete fluorescence-camera dark frame;
- a calibrated optical leakage measurement;
- a pre/post nondestructive reference;
- a natural-loss control with matched wall-clock timing;
- a direct temperature or heating measurement.

Accordingly, the analyses may report apparent occupancy, model-implied count
overlap, operational switch-off decay, effective bright-wait decay and
apparent inter-readout retention. They may not report empirical fidelity,
false-positive/false-negative rates, intrinsic dark lifetime, definitive
heating, or a proven fixed per-pulse loss.

## Prototype notebook assumptions

The reference notebooks are useful discrepancy checks, but they:

- fit geometry and per-site mixtures on the full data;
- use a neighbour-contaminated local annulus as a primary correction in the
  bright-wait analysis;
- treat many site observations as independent for uncertainty;
- use weighted least squares or local fit covariance for physical rates;
- compare model complexity without held-out predictive likelihood;
- use an additive loss budget where the exact survival is multiplicative.

V1 reproduces their raw grouping/count baselines only to identify why a number
changes. Prototype numerical conclusions are not regression targets.

## Audit conclusion

The raw engineering checks pass for a held-out **operational** analysis:
shots and frames are complete, commanded timing is internally corroborated,
and switch/DDS command states are recoverable. The dominant unresolved
hardware fact is the lack of an independent measurement of residual light
during switch-off holds. The dominant design limitation is deterministic
within-cycle sweep order. Both limitations must accompany every downstream
rate estimate.
