# 2026-07-28 pre-optimization loss-sweep benchmark

> **Historical benchmark.** This document and
> `reports/loss_sweep_results.json` preserve the 2026-07-28 analysis and remain
> reproducible through `scripts/reproduce_all.py`. They are not the source of
> the optimized 2026-07-31 lifetimes, repeated-imaging model, or loss budget.
> See [optimized results](OPTIMIZED_LIFETIME_RESULTS.md) for the current result.

This document defines the public quantities before numerical interpretation.
Exact estimates and intervals are generated into `reports/readme_metrics.md`;
definitions and claim limits live here so a fitted number cannot silently
change its scientific meaning.

## Supporting generated figures

The README uses the four-panel overview as its primary quantitative result.
The standalone diagnostic views remain available for detailed inspection:

- [Operational switch-off-hold retention](../assets/readme/dark_hold_retention.png)
- [Effective bright-wait apparent occupancy decay](../assets/readme/bright_wait_decay.png)
- [Held-out per-site apparent retention](../assets/readme/per_site_retention_map.png)

The per-site map describes spatial heterogeneity; sites are not additional
independent experimental repeats.

## Claim table

| quantity | operational definition | dataset | assumptions | uncertainty method | what may be claimed | what may not be claimed |
|---|---|---|---|---|---|---|
| model-implied count overlap | overlap integral of fitted empty/occupied emission components; descriptive full-data fit for V0 and held-out emission fit for the sweeps | all three | selected component model is an adequate description; component labels fixed by ordered means | descriptive full-data scoring for V0; held-out predictive scoring for V1; shot-cluster summaries where reported | separation of the fitted count components under the stated model | empirical fidelity, FPR or FNR |
| apparent occupancy | V0 descriptive reference call, or V1 posterior probability/threshold call from a rule fitted without the evaluated shot | all three | two count components correspond operationally to lower/higher apparent occupancy | V0 descriptive shot-cluster summary; V1 complete-shot held-out split and shot-cluster interval | fraction called apparently occupied by the stated rule | true occupancy probability without labels |
| fixed interval retention \(q_j\) | zero-hold intercept factor in \(q_j\exp(-\lambda_\mathrm{switch\_off}t)\) | 0044 | selected transition/emission model; interval structure adequate | profile or complete-shot cluster bootstrap | interval-specific apparent fixed retention factor | a pure physical survival probability, pair-loss probability or classification accuracy |
| `lambda_switch_off` | shared or selected interval-specific slope of apparent retention against commanded switch-off hold | 0044 | commanded timing is correct; selected slope structure; background and calls do not generate the trend | condition-stratified complete-shot cluster bootstrap; profile likelihood where used | operational decay rate under the recorded switch-off configuration | intrinsic dark loss rate |
| `tau_switch_off` | reciprocal of a resolved positive `lambda_switch_off` | 0044 | same as rate; reciprocal is reported only when the rate is resolved away from zero | transform of the clustered rate distribution | operational switch-off time constant | fully dark lifetime or best apparatus lifetime |
| `lambda_bright_effective` | slope in the selected apparent image-1 occupancy model versus commanded illuminated wait | 0050 | initial loading/drift checks; selected no-floor/floor structure; background does not create the trend | condition-stratified complete-shot cluster bootstrap | effective apparent decay during the recorded bright-wait condition | intrinsic bright-state lifetime or a stationary microscopic hazard |
| `tau_bright_effective` | reciprocal of a resolved positive `lambda_bright_effective` | 0050 | same as bright rate | transform of the clustered rate distribution | effective bright-wait time constant | intrinsic lifetime |
| `post_wait_retention_trend` | dependence of apparent image-1 to image-2 retention on prior bright-wait duration | 0050 | frozen apparent-occupancy model; flat versus constrained monotone comparison | complete-shot cluster bootstrap and held-out likelihood | survivor-state or classification distribution changes with prior wait, if resolved | definitive heating without a temperature measurement |
| latent transition parameter | transition survival in the gated binary-state model | 0044 exploratory fit only; no accepted 0050 latent fit | timing/background/emission gates pass; parameters identifiable; held-out likelihood improves; complete-shot uncertainty is available | multiple starts, complete-shot cluster bootstrap, synthetic recovery and condition-resolved posterior checks | a predictive transition parameter only if every gate passes | any public transition estimate from the current working-independence exploratory profile; empirical fidelity; or a uniquely physical decomposition |

`q_j` and an emission error can trade off. Even when an apparent-retention
curve is precise, the interval intercept can mix fixed physical loss,
classification error, first-frame transients and state selection.

## Exact survival and loss accounting

For interval \(j\), the operational survival is

\[
S_j(t)=q_j\exp(-\lambda_\mathrm{switch\_off}t).
\]

The exact total operational loss is

\[
L_j(t)=1-q_j\exp(-\lambda_\mathrm{switch\_off}t).
\]

It is not `fixed_loss + dark_loss`. An additive decomposition may be shown
only as a labelled first-order approximation for small probabilities.

## Independent unit and uncertainty

The independent repeat is a shot. Sites and frames within a shot share loading,
illumination, background, timing, camera state and drift. Primary intervals
therefore resample complete shots within condition, preserving every site and
frame. The number of independent shots per sweep point is displayed on public
figures.

Site heterogeneity is reported separately. A supplementary two-way bootstrap
may also resample sites, but it is not the primary physical-repeat interval.
Naive site-binomial intervals are retained only to demonstrate how much they
understate uncertainty.

The reported operational intervals condition on the frozen classifier fitted
to the training shots. Each bootstrap replicate resamples complete physical
shots within condition, but it does not refit the emission model. Uncertainty
from choosing or estimating that classifier is addressed through predeclared
background/emission-model sensitivities, not folded into the primary
physical-repeat interval.

## Model selection and held-out use

The primary split keeps complete acquisition cycles intact:

- cycles 0–5 for training;
- cycles 6–7 for validation;
- cycles 8–9 for final testing.

Every condition is represented by six, two and two independent shots. Geometry,
fixed background templates, emission/threshold parameters, and shrinkage
parameters are fitted on training shots. Validation chooses
background/emission and physical model structure. The test set is scored once.
A separate alternate cycle split and within-cycle drift analysis assess time
sensitivity.

Held-out count likelihood selects predictive descriptions; it does not create
ground-truth occupancy labels.

The current five-frame latent fit is exploratory and is **not accepted**. Its
ordinary profile likelihood treats site trajectories as conditionally
independent, and the formal public gate fails because complete-shot clustered
transition-parameter uncertainty is absent. Synthetic recovery passed its
declared 35% criterion but retained 28.1% relative rate error, supporting weak
physical precision. The fitted model's ordinary held-out likelihood
improvement supports predictive sequence structure, not a precise physical
transition rate. No latent transition estimate or representative posterior
trajectory is promoted to a public result.

## Switch-off interpretation

During nominal holds, the two optical switches are commanded off while the
DDS frequency and amplitude commands remain configured. No optical extinction
measurement is available. The public terms are:

- operational switch-off hold decay;
- lifetime under the switch-off configuration;
- `lambda_switch_off` and `tau_switch_off`.

If residual leakage can only add loss, `tau_switch_off` is a lower bound on a
fully dark lifetime and `lambda_switch_off` an upper bound on its rate. That
one-direction assumption is physical context, not identified by these data.

### Endpoint sensitivity

The shared-slope structure was frozen on validation before endpoint checks.
All four refits succeeded:

| included switch-off holds | variant | `lambda_switch_off` (s^-1) | `tau_switch_off` (s) | later-interval apparent loss |
|---|---|---:|---:|---:|
| 0.1–2.1 s | all points | 0.04487 | 22.29 | 13.25% |
| 0.3–2.1 s | exclude shortest | 0.04642 | 21.54 | 13.27% |
| 0.1–1.9 s | exclude longest | 0.04861 | 20.57 | 13.02% |
| 0.3–1.9 s | exclude both | 0.05149 | 19.42 | 12.92% |

The endpoint choice changes the fitted operational rate but not its sign or the
qualitative apparent-loss conclusion. These are sensitivity refits, not four
independent estimates.

## Bright-wait interpretation

The imaging light is commanded on during the swept wait before image 1. The
public terms are:

- effective bright-wait decay;
- `lambda_bright_effective` and `tau_bright_effective`;
- post-wait retention trend.

A rate extrapolated over 50 ms is compared with apparent inter-readout loss
using a joint bootstrap. If it is smaller, the allowed conclusion is:

> The simple constant-rate bright-wait model does not explain the full
> apparent inter-readout loss.

The data do not prove a fixed per-exposure or fixed per-pulse mechanism.
Turn-on/off transients, nonstationary heating, state selection, classification
error, background, timing accounting and cross-sequence differences remain
alternatives.

### Selected-model sampling and structural sensitivity

Validation selected the no-floor bright-wait structure. Its apparent gap and
sampling interval are conditional on that selection:

| bright-wait structure | fit scope | predicted 50 ms apparent loss | observed − predicted apparent-loss gap |
|---|---|---:|---:|
| selected no-floor | selected on validation; final development/test workflow frozen | 3.00% | 10.26 percentage points (8.62–11.79 pp complete-shot sampling interval) |
| unresolved floor alternative | candidate declared on validation; refit on training + validation only | 6.68% | 6.57 percentage points |

The 6.57–10.26 percentage-point span is a model-structure sensitivity range,
not a confidence interval or total uncertainty. The correct floor-alternative
gap is 6.57 percentage points because it comes from the reviewed
development-only refit after the candidate structures were frozen; test data
were not used. An earlier scratch value of approximately 7.2 percentage points
came from a different exploratory calculation outside that reviewed fit scope
and is not a publication result. Both reviewed structures leave a positive
apparent gap.

### Whole-cycle bootstrap sensitivity

The primary bootstrap resamples complete shots within condition. The separate
whole-cycle bootstrap preserves dependence shared across conditions in each
repeated acquisition cycle. All 1,000 dark and 1,000 bright refits succeeded:

| quantity | whole-cycle estimate (95% interval) |
|---|---:|
| `lambda_switch_off` | 0.04487 s^-1 (0.02251–0.06514 s^-1) |
| `tau_switch_off` | 22.29 s (15.35–44.43 s) |
| `lambda_bright_effective` | 0.6084 s^-1 (0.5649–0.6602 s^-1) |
| `tau_bright_effective` | 1.64 s (1.51–1.77 s) |
| predicted 50 ms apparent loss | 3.00% (2.78–3.25%) |
| later-interval apparent loss | 13.25% (11.84–14.58%) |
| observed − predicted apparent-loss gap | 10.26 percentage points (8.81–11.61 pp) |

This sensitivity preserves a different dependence structure; it does not
replace the primary condition-stratified complete-shot sampling interval.
Sweep value is still perfectly confounded with fixed ascending within-cycle
position.

### Post-wait control and kappa

Validation selected a flat image-1-to-image-2 apparent-retention model. In that
selected structure, \(\kappa=0\) is **structurally fixed** rather than estimated
with a confidence interval. It therefore does not establish a precisely zero
physical trend. The unselected monotone sensitivity places \(\kappa\) on the
boundary at 0 with a 0–0.0330 complete-shot sensitivity interval and does not
resolve a monotone trend. Neither structure identifies heating.

## Prominent limitations

- The nominal dark hold is switch-off, not a fully de-energized DDS/RF state.
- Cooling was not optimized.
- Both new datasets come from one experimental date.
- There are only ten independent shots per condition, split 6/2/2.
- Sites within a shot are correlated.
- There is no empirical occupancy ground truth.
- There is no matched-empty fluorescence sequence or complete dark-frame
  control.
- There is no pre/post nondestructive reference.
- Sweep order is fixed ascending within every repeated cycle.
- Sequence 0044 has a large first-frame pedestal that must be treated with
  site-free pixels.
- Sequences 0044 and 0050 have an approximately one-pitch coordinate shift and
  need independent geometry.
- The two sequences may not be directly poolable even when their recorded
  imaging globals match.
- Interval intercepts mix fixed physical loss, classification error and
  sequence transients.
- The selected no-floor sampling interval is conditional on one model
  structure; the 6.57–10.26 percentage-point structural range is not a
  confidence interval.
- The flat post-wait structure fixes \(\kappa=0\); it does not measure an
  exactly zero physical trend.
- Commanded timing is internally corroborated, but no per-frame camera
  timestamp or independent realized-exposure readback is stored.

These limitations are stop conditions for causal or intrinsic-lifetime claims,
not reasons to suppress the operational measurements.
