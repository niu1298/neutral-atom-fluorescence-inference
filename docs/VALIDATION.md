# Validation

Round 1 produced a site geometry and a background correction. Both were fit
results presented as facts. This document records how each was tested, and what
the tests changed.

**Regenerate:**

```bash
python scripts/validate_site_geometry.py      --config configs/paired_100ms.yaml
python scripts/compare_background_methods.py  --config configs/paired_100ms.yaml
```

Machine-readable: `reports/validation/site_geometry_validation.json`,
`reports/validation/background_method_comparison.json` (both gitignored).

---

## 1. Site geometry

### Half-run stability

The geometry is refitted from scratch on shots 0–49 and on shots 50–99 — two
independent variance maps, two independent lattice fits — and the two site sets
are matched by **globally optimal assignment**, not greedy nearest-neighbour.
Greedy matching is order dependent, so it could give two different answers for
the same data, which is precisely what a stability test must not do.

| Statistic | Value |
|---|---|
| Sites in each half | 200 / 200 |
| Matched within 4 px | **200**, 0 unmatched either way |
| Median displacement | **0.080 px** |
| 90th percentile | 0.117 px |
| 99th percentile | 0.136 px |
| Maximum | **0.149 px** |
| Bulk translation (row, column) | −0.069, +0.021 px |
| Residual after removing the bulk shift, median / max | 0.041 / 0.124 px |

Every site lands within a seventh of a pixel of where the other half of the run
puts it. There is no meaningful drift to correct for.

### Is the 10 × 10 window real, or imposed?

`ny = nx = 10` is written in the configuration, so it has to be justified from
the data. Two independent checks:

**Peak-count saturation.** Refitting with a 14 × 14 window captures at most
**2.0%** more detected peaks. If the occupied region extended past 10 × 10, a
larger window would find more sites. It does not.

**Boundary profile.** The fitted lattice is extrapolated outward and the
variance score is read at each modelled position:

| | grid A | grid B |
|---|---|---|
| Median score, interior 8 × 8 | 31,951 | 34,226 |
| Median score, edge ring of the window | 29,424 | 33,596 |
| Median score, first ring **outside** | 461 | 883 |
| Edge-to-outside ratio | **64×** | **38×** |

The edge row and column are as bright as the interior, and one lattice step
further out the score falls to the noise floor. A larger lattice loaded in its
middle by a Gaussian beam would taper off gradually; this is a step. Each block
really is a 10 × 10 structure.

> An earlier version of this check grew the fitting window and counted detected
> peaks in the added ring. That test was wrong and reported a failure: growing
> the window does not find new peaks, it only relabels the same peaks as
> "edge", so the check could never pass whatever the data looked like. It was
> replaced rather than retuned.

### Are the two blocks independent?

If one physical array were being imaged twice, the two blocks would share their
shot-to-shot occupancy: site (row, col) of one would track its counterpart in
the other almost perfectly. Comparing index-matched correlations against a
permuted null makes that testable.

| | Value |
|---|---|
| Index-matched pairs | 100 |
| Median matched correlation | **−0.003** |
| Median permuted-null correlation | −0.001 |
| Duplicate-image hypothesis | **rejected** |

For reference, the median pairwise correlation *within* each block is also
about zero (−0.003, −0.002), so there is no global shot-to-shot loading
fluctuation either. The blocks behave as independent measurement locations,
which is what licenses treating all 200 sites as separate observations.

### The two flagged sites

`site_not_detected` marks a modelled position with no variance peak within 3 px.
Both flagged sites were checked individually against every candidate cause:

| | site 29 (grid A, r2 c9) | site 127 (grid B, r2 c7) |
|---|---|---|
| Outside the valid image region | no | no |
| ROI clipped by the frame | no | no |
| Overlaps a neighbour | no (10.3 px away) | no (11.2 px away) |
| Low-signal site | **yes** — peak variance ranks 2nd lowest of 200 | no — ranks 26th of 200 |
| Lattice-fit outlier | yes, residual 9.7 px | borderline, residual 3.07 px |
| Intentionally absent trap | unknown: no trap-light reference exists |
| **Verdict** | a real but unusually faint site whose variance peak is too weak to be a local maximum | peak-detection ambiguity — a neighbouring diffuse feature merged with it; the modelled position itself is sound |

Neither is dropped. Both keep their modelled ROI and carry the flag, so any
downstream analysis excludes them deliberately or not at all.

### What the geometry gate does and does not certify

Certified: the site set is stable across the run, matched one-to-one between
halves, not clipped by the configured window, not overlapping, inside the valid
image region, and not a duplicated image of a single array.

**Not** certified: that each site is a physically verified trap. Sites are
localised from atom fluorescence, so a trap never loaded during these 100 shots
is invisible to this procedure. A trap-light reference image would settle it;
this run has none.

### Physical interpretation

Measured: two spatially disjoint square arrays in the same exposure, exactly
10 × 10 each with hard boundaries, differing in pitch (10.28 vs 11.21 px) and
in-plane rotation (16.7° vs 10.3°).

Supported by the data and the sequence: two independent crossed-beam 2D
lattices are configured and driven, one continuous lattice cannot produce two
pitches, and the blocks are not duplicate images.

Contradicted by the sequence: the second lattice's deflector channels are
ramped to zero amplitude before the imaging block, and that branch does execute
for this parameter set — so on the face of the sequence only one potential is
confining during the exposure.

**Status: partially resolved.** That the blocks are distinct, sharply bounded
and independent is established. Which potential holds each block during the
exposure is not, and one shot with either lattice disabled from the start would
settle it. No result in this repository depends on the answer.

---

## 2. Background method

### What was wrong

Round 1 used a 13–33 px local annulus as the primary background. At a 10–11 px
site pitch that ring contains roughly ten neighbouring sites.

### The four methods compared

| | Method | Estimated from |
|---|---|---|
| A | no correction | — |
| B | global site-free median, per frame | pixels outside a 5 px disk around every site |
| C | robust degree-4 spatial surface, refitted per frame | same, Huber IRLS on 80,000 pixels at fixed stride |
| **D** | fixed spatial template + per-frame common-mode offset | template from the 200-frame mean with site pixels filled from the surface; per frame, one scalar median offset |

Mask parameters: 5 px exclusion radius (ROI half-width 2 plus a 3 px margin,
more than two PSF widths beyond the ROI edge), hot pixels beyond 8 robust sigma
of the run mean also excluded. 404,259 usable pixels of 420,000; 3 hot pixels.

### Results

| Method | Residual structure (counts/px) | \|drift\| /100 shots | \|corr with own background\| | d′ frame 0 | d′ frame 1 |
|---|---|---|---|---|---|
| A raw | — | 126.9 | — | 2.80 | 3.29 |
| B global | 40.97 | 67.6 | 0.148 | 2.81 | 3.29 |
| C spatial | 15.03 | 82.2 | 0.216 | 2.98 | 3.42 |
| **D template + offset** | **8.74** | 72.7 | 0.171 | 2.97 | 3.41 |
| legacy annulus | — | 50.7 | **0.571** | 2.76 | 3.17 |

### Why D, and why not on separation

Selection is on residual spatial structure (double weight), drift with
acquisition order, and how strongly the corrected count still tracks its own
background. **d′ is reported and explicitly not used to rank.** Ranking on d′
rewards subtracting less, and would select "no correction" — which leaves the
entire frame-dependent background in the signal.

D wins on the criterion that actually speaks to validity: it leaves 8.74
counts/px of residual block-median structure, against 15.03 for C and 40.97 for
B. The static fringe pattern is measured once at 200-frame signal-to-noise
rather than re-estimated from each frame, which is both more accurate and more
stable.

The legacy annulus fails on its own terms: its corrected count still correlates
with its own background at |r| = 0.571, four times worse than any site-masked
method, and it produces the lowest separation of anything tested. Its apparent
advantage on drift (50.7) is an artefact of absorbing real signal.

### Effect on the reported frame-dependent shift

| Estimator | Frame 1 − frame 0 |
|---|---|
| Legacy annulus (Round 1 primary) | **−36.56 ± 9.94** counts/px |
| D, template + offset (current primary) | **−12.53 ± 1.48** counts/px |
| B, global site-free median | −12.92 ± 1.69 counts/px |
| Whole-frame median, independent of any model | −13.77 ± 1.74 counts/px |

The Round-1 number was about three times too large with seven times the spread.
The two site-masked estimators agree with each other to 0.4 counts/px and with
the model-free whole-frame median.

### Honest limitation

A 5 px exclusion radius at a 10–11 px pitch masks roughly 74% of each block, so
C and D partly interpolate the background underneath the array. The
interpolation is smooth by construction and cannot represent structure finer
than the mask spacing. Whether real fine structure exists under the sites is
not determinable from this data alone.

---

## 3. What changed as a result

| | Round 1 | Round 1.5 |
|---|---|---|
| Schema version | 1.0 | **2.0** |
| Primary background | local annulus | template + per-frame offset |
| Background methods stored | 3, one overwriting the notion of "the" correction | 4 + 1 diagnostic, all in separate columns |
| Frame 1 − frame 0 shift | −36.56 counts/px | −12.53 counts/px |
| Paired-readout agreement | 91.77% | 91.98% |
| Apparent bright-to-dark | 11.52% | 11.98% |
| Apparent dark-to-bright | 3.75% | 3.49% |
| Geometry | asserted | tested, with a gate |
