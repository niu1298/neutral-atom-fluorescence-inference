# Prototype discrepancy ledger

This ledger explains why the reproducible V1 loss-sweep analysis does not
numerically reproduce every output saved in `survival.ipynb` and
`bright_loss.ipynb`. The notebooks are useful loading and grouping checks.
Their fitted conclusions are neither ground truth nor regression targets.

The V1 values below come from the frozen held-out analysis artifact and its
deterministic 1,000-replicate complete-shot bootstrap. Each bootstrap draw
resamples shots within condition while retaining all sites and frames from a
selected shot.

This ledger compares the thresholded operational baselines. A separately
gated latent-state model has a different likelihood and parameter
decomposition; its transition parameters must not be substituted into these
rows without saying that the estimand changed.

## Raw-data agreement

The independent loader reproduces the notebook inventory and grouping before
any classifier is fitted:

| check | notebook | V1 audit | result |
|---|---|---|---|
| 0044 shots | 110 | 110 complete, 0 incomplete | agrees |
| 0044 frames | `fluor` through `fluor5` | same five names and order | agrees |
| 0044 grouping | 0.1--2.1 s, 11 points, 10 shots each | same values and counts | agrees |
| 0050 shots | 100 | 100 complete, 0 incomplete | agrees |
| 0050 frames | `fluor`, `fluor2` | same two names and order | agrees |
| 0050 grouping | 0.1--1.9 s, 10 points, 10 shots each | same values and counts | agrees |
| exposure | 50 ms | 50 ms commanded exposure corroborated by two program representations | agrees |
| 0050 inter-frame gap | 10 ms | 10 ms commanded switch-off gap | agrees |

Raw ROI totals agree when evaluated at the same site coordinates and with the
same 5 x 5 ROI. V1 nevertheless refits coordinates independently and does not
pin occupancy calls to notebook thresholds. The raw grouping agreement shows
that changed estimates are methodological, not the result of silently loading
different conditions.

## Geometry discrepancy

| run | notebook full-data fit | V1 training-only fit | explanation |
|---|---|---|---|
| 0044 | pitch about 11.205 px; tilt about -8.69 deg; centre about (331.1, 380.9) px | pitch 11.205451 px; tilt -8.686355 deg; centre (331.076923, 380.923077) px | numerical sanity check agrees; V1 freezes the fit using 66 training shots |
| 0050 | pitch about 11.227 px; tilt about -8.85 deg; centre about (342.1, 382.5) px | pitch 11.233032 px; tilt -8.833857 deg; centre (342.037690, 382.587581) px | small change follows from training-only fitting and independent validation |

The fitted cross-run displacement is 11.086432 px and is consistent with one
0044 lattice step to 0.119438 px. Identical recorded crop metadata rules out a
documented crop change, but the images cannot distinguish a real translation
from an unrecorded state change or an index alias. The two sequences are not
pooled.

## Analysis method changes

| stage | prototype notebooks | V1 method | consequence |
|---|---|---|---|
| geometry | full dataset | training cycles only; validate early/late and endpoint subsets | prevents validation/test geometry leakage |
| background | raw/per-image trap tails; local `bgsub` annulus in bright-wait analysis | site-free fixed spatial template plus per-frame common-mode offset | separates camera background diagnostics from changing occupancy |
| emissions | independent per-site, per-image two-Gaussian fits using all shots | pooled frame emissions with shrinkage site offsets, fitted on training shots | reduces weak-site overfitting and freezes the classifier before testing |
| split | none | complete cycles: 60% train, 20% validation, 20% test | every shot, site and frame stays in one subset |
| model choice | full-data fit quality or nested chi-square comparison | validation predictive likelihood plus complexity and boundary gates | test data do not select structure |
| physical fit | aggregated percentages and weighted least squares | constrained event likelihood | probabilities remain in their physical range |
| uncertainty | site-level/binomial-looking errors or local covariance | resample complete shots within condition, retaining every site and frame | recognizes 10 independent shots per condition rather than 1,000 independent sites |
| loss arithmetic | fixed loss plus rate-times-time approximation | \(1-q_j\exp(-\lambda t)\) | avoids additive-probability error |

For both runs, validation selected the template-corrected
`shrinkage_site_offsets_k5` emission model among eligible baselines. Its
held-out count diagnostics are:

| run | validation shots | test shots | test count NLL / row | model-implied overlap | mean separation \(d'\) |
|---|---:|---:|---:|---:|---:|
| 0044 | 22 | 22 | 8.08139 | 2.607% | 3.710 |
| 0050 | 20 | 20 | 7.90791 | 4.122% | 3.300 |

These overlaps describe Gaussian components under the fitted model. They are
not empirical readout fidelity, FPR or FNR. The notebook's quantities named
`F`--about 0.9458 for 0044 and 0.9390 for 0050--are likewise model-implied
separation summaries, not labelled accuracy, and use a different aggregation.

## The 0044 pedestal discrepancy

The notebook fits the low-count component of trap ROI histograms separately
for each frame:

- frame 1 empty-component level: about 10,478.5 raw ROI counts;
- frames 2--5: about 9,365--9,430 counts;
- maximum spread: about 1,114 counts.

That low-count component is not a site-free background measurement. It can
move when occupancy, weak-site composition, atom emission or component overlap
changes.

The direct site-free estimate gives a much smaller frame effect:

- frame 1 median: 9,125 counts per 5 x 5 ROI;
- frames 2--5 median: 8,900 counts;
- spread: 225 counts.

The selected spatial-template estimate is 9,541.08 counts in frame 1 and
9,318.11--9,320.85 in frames 2--5. The difference between 1,114 and 225
counts explains why separate full-data trap thresholds absorb substantially
more than the measured camera-background pedestal. V1 keeps frame-specific
emissions after site-free correction rather than assuming that background
correction alone makes all five histograms identical.

## Switch-off-hold estimates

The operational model is

\[
P(\mathrm{retained}_{j+1}\mid\mathrm{apparently\ occupied}_j,t)
 =q_j\exp(-\lambda_{\mathrm{switch\_off}}t).
\]

| quantity | prototype notebook | final V1 estimate | change and reason |
|---|---:|---:|---|
| shared rate | 0.051766 /s | 0.044868 /s (95% cluster CI 0.030169--0.064239) | 13.3% lower after site-free template correction, frozen emissions and event likelihood |
| operational time constant | 19.3179 s | 22.2877 s (95% cluster CI 15.5668--33.1472) | reciprocal of the revised operational rate; not an intrinsic dark lifetime |
| uncertainty | 0.006304 /s from the prototype fit | 1,000-replicate complete-shot bootstrap | sites are not treated as independent experimental repeats |

The interval factors also move:

| interval | prototype \(q_j\) | final V1 \(q_j\), 95% cluster CI | difference |
|---|---:|---:|---:|
| 1 to 2 | 0.795172 | 0.767797 (0.743183--0.793828) | -2.738 percentage points |
| 2 to 3 | 0.865716 | 0.859174 (0.841061--0.877952) | -0.654 percentage points |
| 3 to 4 | 0.891571 | 0.853840 (0.836925--0.874557) | -3.773 percentage points |
| 4 to 5 | 0.881367 | 0.889421 (0.868685--0.912400) | +0.805 percentage points |

The prototype imposed one shared slope with four intercepts. V1 compares three
structures on validation events:

| candidate | parameters | validation mean NLL / event | delta from selected |
|---|---:|---:|---:|
| shared rate | 5 | 0.509501 | 0 |
| first interval separate | 6 | 0.509725 | 0.000224 |
| interval-specific rates | 8 | 0.509823 | 0.000322 |

All converged without a boundary parameter. Validation therefore selects the
shared-rate model: the more complex slopes do not improve prediction. The low
first-interval factor remains mechanistically distinct in size, but it mixes
fixed physical loss, classification error, initial-state selection and
sequence transients. It does not prove pair loss.

## Bright-wait estimates

| quantity | prototype notebook | final V1 estimate | change and reason |
|---|---:|---:|---|
| effective rate | 0.390477 /s | 0.608361 /s (95% cluster CI 0.545625--0.666081) | 55.8% higher under the frozen template/shrinkage classifier |
| effective time constant | 2.561 s | 1.64376 s (95% cluster CI 1.50132--1.83276) | reciprocal of the revised effective rate |
| initial apparent occupancy \(\pi_0\) | not used as the headline | 0.354049 (95% cluster CI 0.333704--0.373463) | model parameter, not loading ground truth |
| floor | no floor selected | 0 | same simple structure, selected with held-out complexity checks |

The point estimate remains background-sensitive, but the defensible
site-free variants agree more closely with each other than with the notebook:

| count variant | effective rate (/s) | status |
|---|---:|---|
| raw | 0.578457 | sensitivity only; no correction |
| global site-free | 0.588301 | eligible |
| spatial site-free | 0.606124 | eligible |
| fixed template + offset | 0.608361 | selected |
| local annulus | 0.483031 | diagnostic only; neighbour contaminated |

The floor candidate improves validation NLL by only 0.000517 per site
observation, or 0.0517 per independent validation shot. That gain is not
stable under shot-level resampling, so the additional floor parameter fails
the complexity gate and the no-floor model is selected. This is more
conservative than choosing a floor from a bare reduction in chi-square.

## Bright-wait background drift

The notebook's low-tail trap estimate moves by approximately -137 counts in
frame 1 and -106 counts in frame 2 across the wait sweep. Site-free estimates
instead move slightly upward:

| estimator | frame 1, shortest to longest | frame 2, shortest to longest |
|---|---:|---:|
| global site-free | +20.00 counts | +12.50 counts |
| spatial site-free | +16.93 counts | +7.79 counts |
| fixed template + offset | +20.27 counts | +12.13 counts |
| local annulus | -152.85 counts | -137.29 counts |

The contaminated annulus is also strongly correlated with apparent occupancy
(\(r=0.851\) and \(0.817\)), whereas the selected template correlations are
\(-0.282\) and \(-0.135\). The notebook drift is therefore not carried
forward as evidence of camera-background drift.

## Image-1 to image-2 control

The notebook reports:

- mean apparent retention 0.769769;
- slope versus prior wait \(-0.061356\pm0.013600\) /s;
- a nominal 4.5-sigma departure from flat.

That uncertainty calculation lets many sites contribute as though they were
independent experiments. Under the V1 classifier and shot-grouped validation,
the flat control is selected. Its fixed retention is 0.881418 (95% cluster
CI 0.865506--0.898076);
the monotone candidate lands at zero trend and does not improve validation
prediction. The held-out control contains 393 apparent-retention events from
20 independent test shots. No wait-dependent post-wait trend is resolved, so
the notebook's significance claim is not repeated.

The change in mean retention is not itself proof that one classifier is
correct: there are no empirical occupancy labels. It records the sensitivity
of apparent transitions to background and emission modelling.

## Fixed-cost hypothesis

The prototype uses its effective bright rate to predict 1.93% loss during one
50 ms exposure, then contrasts that value with much larger inter-readout
losses. With the final V1 rate, the exact constant-rate
prediction is

\[
1-\exp[-(0.608361\ \mathrm{s}^{-1})(0.05\ \mathrm{s})]=2.996\%.
\]

The later V1 interval factors correspond to apparent fixed losses of 14.08%,
14.62% and 11.06% before applying the hold-time exponential. Their mean is
13.25% (95% cluster CI 11.60--14.70%), compared with a 3.00% bright-model
prediction (2.69--3.28%). The observed-minus-predicted gap is 10.26%
(8.62--11.79%). Thus the simple constant-rate bright-wait model still does
not explain the full apparent inter-readout loss.

That mismatch does **not** prove a fixed per-exposure or fixed per-pulse cost.
The two sequences have a one-pitch coordinate displacement, different
background trajectories, different state histories and no randomized
cross-sequence control. Other explanations include turn-on/off transients,
nonstationary heating, state selection, residual classification error,
different optical conditions and imperfect exposure accounting. A
pulse-count by total-light-time experiment is required to identify a fixed
per-pulse contribution.

## Replacement rule

The estimates in this ledger may be updated only from the generated
machine-readable result. Its uncertainty implementation:

1. resample complete shots within each condition;
2. retain every site and frame belonging to a selected shot;
3. report the independent-shot count;
4. keep validation-based model selection frozen;
5. score the test set once;
6. use exact multiplicative loss.

The completed public regeneration satisfies these requirements with 1,000
successful draws and no failed bootstrap fits. No site-binomial sigma claim is
carried forward.
