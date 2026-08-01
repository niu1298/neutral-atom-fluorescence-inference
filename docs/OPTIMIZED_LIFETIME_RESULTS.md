# Optimized lifetime and repeated-imaging results

This document is the detailed human-readable interpretation of
`reports/optimized_lifetime_results_20260731.json`. It covers the five
2026-07-31 acquisitions only. The 2026-07-28 loss sweep is retained separately
as a pre-optimization benchmark.

## Claim boundary

All occupancies are inferred latent quantities. The experiment has no
externally labelled empty/occupied truth and no direct atom-loss timestamp.
The public conclusions are therefore operational and conditional on the
validated observation and survival models.

| Public quantity | Operational definition | Data and selected model | Uncertainty | Allowed claim | Prohibited claim |
|---|---|---|---|---|---|
| apparent occupancy | posterior probability of the occupied latent count component | exposure-specific shrinkage model | complete-shot resampling for aggregates | held-out latent-class occupancy under the model | labelled occupancy or empirical fidelity |
| model-implied overlap | overlap integral of fitted empty/occupied count densities | final exposure-specific emission model | diagnostic, not a binomial error rate | component separability | measured classification error |
| dark lifetime | inverse common dark-hold hazard | 0094 shared-rate interval model | 1,000 complete-shot bootstrap replicates | operational survival under commanded fully-dark holds | intrinsic vacuum lifetime |
| bright lifetime | inverse illuminated-wait hazard | 0090 no-floor decay | 1,000 complete-shot bootstrap replicates | operational survival under recorded illumination | universal cooling or trap lifetime |
| prefix survival | multiplicative survival from fitted bright and dark hazards | selected repeated-imaging M0 | complete-shot and block sensitivities | predicted survival through a frame prefix | directly observed atom survival |
| loss budget | ordered bright, dark, and residual segment probabilities | selected M0; M1 as sensitivity | propagated clustered intervals | rate-based prior segment attribution | exact event time or count-conditioned location |

## Held-out empty/occupied inference

At each exposure, validation selects `shrinkage_site_offsets_k5`: a two-state
emission model with five-fold shrinkage of site offsets. No site meets the
predeclared weak-site exclusion rule.

| Exposure | Test shots | Test rows | Mean NLL | d-prime | Model-implied overlap | Posterior entropy |
|---:|---:|---:|---:|---:|---:|---:|
| 50 ms | 20 | 10,000 | 8.18036 | 4.80026 | 0.005891 | 0.01362 |
| 100 ms | 20 | 10,000 | 8.88601 | 5.10806 | 0.003270 | 0.01366 |
| 200 ms | 20 | 10,000 | 9.46087 | 5.42329 | 0.001961 | 0.01163 |

Validation-only extra-state and heavy-tail diagnostics pass their rejection
gates at every exposure. The largest ambiguous-count fraction is 1.9% against
a 5% threshold; the largest nearest-component residual beyond four standard
deviations is 0.05% against a 2% threshold. A third state and heavy-tail model
are therefore not fitted to test data.

A focused held-out posterior-predictive audit classifies the remaining density
mismatch as **B: mild misspecification, scientific results stable**. The public
histogram uses the same emission-adjusted coordinate as the frozen Gaussian
components: background-corrected counts after the training-fitted frame and
site offsets. Reconstructing that coordinate and the validation ranking gives
zero numerical discrepancy; component weights sum to one and empty plus
occupied density equals the scored predictive density to floating-point
precision. The displayed empirical density integrates to 1.000. The summed
frozen density integrates to 0.9950, 0.9971, and 0.9927 over the clipped 50,
100, and 200 ms display ranges; the omitted mass is the disclosed 0.25% tail
at each end, not an extra normalization.

Across exposures, integrated absolute density error is 0.159, 0.150, and
0.114, while Jensen-Shannon divergence is 0.0048, 0.0045, and 0.0035 nats.
Complete-shot bands show localized residuals near the two component modes and
their slopes, rather than a missing dominant state. Posterior-partitioned
empirical widths are within about 6% of the frozen component widths. The mode
offset is mildly negative overall, and the 0.25th–99.75th percentile display
clipping moves the empirical mode by less than one bin. These partitions are
model-conditioned diagnostics, not labelled empty/occupied truth.

The audit resamples all sites and frames within each of 20 complete test shots
(400 replicates) and also checks frame index, early/late shots, ordered site
quartiles, and exposure. Eligible frozen candidates change mean held-out
posterior occupancy by at most about two percentage points. Already-reported
emission sensitivities keep the dark and bright lifetimes within their stated
robustness ranges, and every repeated-imaging background sensitivity still
selects the continuous-only model. The lifetime and no-material-pulse-term
conclusions are therefore unchanged; no test-data refit is used.

## Dark lifetime

Validation selects the shared-rate model. The common dark hazard and its
inverse are:

| Quantity | Estimate | 95% complete-shot interval |
|---|---:|---:|
| `lambda_dark` | 0.027984 s⁻¹ | 0.024839–0.031163 s⁻¹ |
| `tau_dark` | 35.734 s | 32.089–40.260 s |

The final held-out score is NLL 1437.810 over 9,053 eligible transition events,
or 0.158821 per event. All 1,000 primary bootstrap replicates succeed.

The shared hazard acts across four separately estimated interval factors:

| Transition | `q` estimate | 95% complete-shot interval |
|---:|---:|---:|
| frame 1→2 | 0.986095 | 0.980507–0.991178 |
| frame 2→3 | 0.992673 | 0.988468–0.996469 |
| frame 3→4 | 0.994681 | 0.990957–0.998255 |
| frame 4→5 | 0.994974 | 0.991293–0.998300 |

These factors describe apparent retention conditional on the selected latent
state model. The physical time variable is the declared dark hold; exposure
duration and the 2 µs command overhead are not folded into it.

Robustness ranges for `tau_dark` are 33.36–39.66 s under endpoint removal,
34.33–36.21 s across background methods, and 33.50–35.73 s across eligible
emission variants. An alternate contiguous split gives 36.54 s. A whole-cycle
bootstrap gives 31.74–40.73 s. None reverses the resolved positive hazard.

## Bright lifetime and control

Validation selects a no-floor exponential decay. Adding a floor changes the
validation mean NLL from 0.679668 to 0.679678 and fails the complexity gate.

| Quantity | Estimate | 95% complete-shot interval |
|---|---:|---:|
| initial apparent occupancy `pi0` | 0.514117 | — |
| `lambda_bright` | 0.049250 s⁻¹ | 0.042443–0.055746 s⁻¹ |
| `tau_bright` | 20.304 s | 17.939–23.561 s |

The final held-out score is NLL 2706.208 over 4,000 frame-site rows, or
0.676552 per row. All 1,000 primary bootstrap replicates succeed.

For the frame-1→frame-2 post-wait control, validation selects a flat model:
`q0 = 0.988339` with interval 0.985688–0.990723. A monotone trend is not
resolved, so its slope is structurally fixed rather than reported as a
measured zero with a confidence interval.

Background sensitivity places `tau_bright` at 19.83–21.06 s; eligible emission
variants give 20.03–20.42 s; an alternate split gives 20.49 s; and the
whole-cycle interval is 17.49–23.41 s.

## Repeated imaging

Validation selects M0, the continuous-only model. It uses the independently
estimated bright and dark hazards and fits one initial loading probability for
each exposure run.

| Run | Initial apparent occupancy |
|---|---:|
| 50 ms | 0.523349 |
| 100 ms | 0.525824 |
| 200 ms | 0.520571 |

The final held-out M0 score is NLL 20764.239 over 30,000 rows, or 0.692141 per
row. All 1,000 complete-shot and all 1,000 contiguous-block bootstrap
replicates succeed.

### Model gate

| Model | Validation NLL | Mean NLL | Incremental improvement | Public gate |
|---|---:|---:|---:|---|
| M0 continuous-only | 20769.433 | 0.692314 | reference | selected |
| M1 plus common pulse factor | 20767.969 | 0.692266 | 0.0000488 per row | below 0.0001 materiality threshold |
| M2 exposure-specific pulse factors | — | — | 0.0000028 over M1 | clustered interval crosses zero |
| M3 diagnostic extension | — | — | diagnostic only | not public |

The clustered interval for the M1 improvement is positive,
0.0000213–0.0000793 NLL per row, but remains entirely below the predeclared
materiality threshold. Statistical detectability at this row count is not
enough to make the extra term scientifically material.

The retained
[matched-total-exposure figure](../assets/readme/exposure_segmentation_result.png)
shows the complete-shot intervals for the fixed-bright-time contrasts.

M1 is retained only as sensitivity. It estimates `q = 0.994281`, with
complete-shot interval 0.993009–0.995622 and block-bootstrap interval
0.992882–0.995674. Early and late shot-order estimates are 0.994558 and
0.993986. All background variants and the alternate split still select M0.

## Apparent-occupancy prefixes

| Exposure | Frame 1 | Frame 2 | Frame 3 | Frame 4 | Frame 5 |
|---:|---:|---:|---:|---:|---:|
| 50 ms | 0.5282 | 0.5248 | 0.5227 | 0.5185 | 0.5170 |
| 100 ms | 0.5336 | 0.5260 | 0.5211 | 0.5160 | 0.5108 |
| 200 ms | 0.5260 | 0.5145 | 0.5049 | 0.4960 | 0.4885 |

The corresponding selected-model cumulative survivals are:

| Exposure | Frame 1 | Frame 2 | Frame 3 | Frame 4 | Frame 5 |
|---:|---:|---:|---:|---:|---:|
| 50 ms | 0.99754 | 0.99481 | 0.99208 | 0.98937 | 0.98666 |
| 100 ms | 0.99509 | 0.98992 | 0.98478 | 0.97967 | 0.97458 |
| 200 ms | 0.99020 | 0.98022 | 0.97034 | 0.96056 | 0.95088 |

The first table describes all-shot latent-class occupancy. The second removes
run-specific initial loading and evaluates the selected rate model. Their
different purposes are kept explicit.

## Fixed-total-bright-time identification

Direct held-out, model-normalized contrasts are:

| Total bright time | Contrast | Difference | 95% complete-shot interval |
|---:|---|---:|---:|
| 100 ms | 2×50 − 1×100 | −0.01144 | −0.07044 to 0.04949 |
| 200 ms | 4×50 − 1×200 | −0.00673 | −0.06425 to 0.04953 |
| 200 ms | 2×100 − 1×200 | +0.00409 | −0.05337 to 0.06008 |
| 400 ms | 4×100 − 2×200 | +0.00926 | −0.05283 to 0.06835 |

Every interval crosses zero. The selected M0 rate model predicts much smaller
deterministic differences caused only by the unequal number of 10 ms gaps:
−0.000278 at 100 ms, −0.000831 and −0.000277 at 200 ms, and −0.000548 at
400 ms. The data do not resolve an additional segmentation penalty.

## Interpretation

The five acquisitions support latent-class occupancy inference, resolved dark
and bright operational hazards, and multiplicative bright/dark loss accounting.
They do not support empirical fidelity, a universal bright hazard, a selected
pulse-associated loss term, or exact loss timing. Those boundaries are part of
the result, not post-hoc disclaimers.
