# Repeated-imaging loss decomposition

This document defines the bright, dark-gap, and residual apparent-loss budget
for the five-frame 50, 100, and 200 ms acquisitions.

## Timing basis

For five frames, total bright time is five times the frame exposure and total
declared dark time is four times 10 ms. The compiled frame-start interval also
contains 2 µs of command-update overhead. That overhead is recorded in the
data contract but is not relabelled as physical dark exposure.

The selected continuous-only model, M0, applies independently estimated bright
and dark hazards:

`S = exp(-lambda_bright × T_bright) × exp(-lambda_dark × T_dark)`.

The loss budget orders the bright and dark segments and assigns the decrement
in survival produced by each factor. Thus bright contribution, dark
contribution, residual contribution, and final survival sum to one.

## Primary selected-model budget

| Exposure | Bright loss | 95% interval | Dark-gap loss | 95% interval | Residual | Final survival | 95% interval |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 ms | 0.012230 | 0.010497–0.013833 | 0.001112 | 0.000987–0.001238 | 0 by M0 | 0.986658 | 0.985058–0.988389 |
| 100 ms | 0.024311 | 0.021035–0.027420 | 0.001105 | 0.000987–0.001229 | 0 by M0 | 0.974584 | 0.971428–0.977785 |
| 200 ms | 0.048031 | 0.041835–0.054252 | 0.001092 | 0.000980–0.001211 | 0 by M0 | 0.950878 | 0.944635–0.957072 |

The dark contribution changes slightly with exposure because it is applied
after a different amount of surviving bright exposure. It does not imply a
different dark hazard.

Residual loss is zero by selected model structure, not a precisely measured
physical zero. The validation gate does not select an additional pulse factor.

## Unselected pulse-model sensitivity

M1 adds a common multiplicative factor per frame. It improves validation mean
NLL by 0.0000488 per row, below the predeclared 0.0001 materiality threshold.
It is therefore not the headline model.

Conditional on M1, the five-frame budget is:

| Exposure | Bright loss | Dark-gap loss | Residual pulse loss | Residual 95% interval | Final survival |
|---:|---:|---:|---:|---:|---:|
| 50 ms | 0.012092 | 0.001102 | 0.022527 | 0.017087–0.027506 | 0.964280 |
| 100 ms | 0.024036 | 0.001096 | 0.022389 | 0.017310–0.027341 | 0.952480 |
| 200 ms | 0.047490 | 0.001082 | 0.022117 | 0.017001–0.026987 | 0.929311 |

This table is model-structure sensitivity. Its apparently resolved residual
cannot override M1's failed held-out materiality gate.

## Prefix survival under M0

| Exposure | After frame 1 | After frame 2 | After frame 3 | After frame 4 | After frame 5 |
|---:|---:|---:|---:|---:|---:|
| 50 ms | 0.997541 | 0.994809 | 0.992084 | 0.989367 | 0.986658 |
| 100 ms | 0.995087 | 0.989921 | 0.984782 | 0.979670 | 0.974584 |
| 200 ms | 0.990198 | 0.980218 | 0.970339 | 0.960559 | 0.950878 |

These are survival multipliers. They exclude the run-specific initial loading
probability, enabling exposure and segmentation comparisons on a common scale.

## Matched-total-exposure evidence

| Total bright time | Direct normalized contrast | 95% interval | M0 timing-only contrast |
|---:|---:|---:|---:|
| 100 ms: 2×50 − 1×100 | −0.011442 | −0.070436 to 0.049488 | −0.000278 |
| 200 ms: 4×50 − 1×200 | −0.006726 | −0.064254 to 0.049531 | −0.000831 |
| 200 ms: 2×100 − 1×200 | +0.004089 | −0.053369 to 0.060076 | −0.000277 |
| 400 ms: 4×100 − 2×200 | +0.009256 | −0.052834 to 0.068353 | −0.000548 |

All direct intervals cross zero. The small M0 differences arise from the extra
10 ms dark gaps in more segmented prefixes. The matched comparisons therefore
do not identify a material pulse-associated penalty.

## What can be located

The sequence defines when bright and dark segments occurred. Fitted hazards
then give prior probabilities that apparent loss belongs to each segment.
That is the supported location statement.

The camera integrates over each exposure, occupancy is latent, and there is no
independent event timestamp. Consequently, this analysis does not provide:

- an exact time of loss;
- a directly observed bright-versus-dark label for an individual atom;
- a count-conditioned posterior over event location;
- a hardware-specific switching-transient measurement.

No latent loss-location model is promoted because those outputs would require
clustered uncertainty and an identifiable observation channel beyond the
integrated frames available here.
