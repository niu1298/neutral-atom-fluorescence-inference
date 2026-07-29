# Interpreting the loss-sweep quantities

This document defines the public quantities before numerical interpretation.
Exact estimates and intervals are generated into `reports/readme_metrics.md`;
definitions and claim limits live here so a fitted number cannot silently
change its scientific meaning.

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

- six cycles for training;
- two cycles for validation;
- two cycles for final testing.

Every condition is represented by six, two and two independent shots. Fixed
background templates, emission/threshold parameters and shrinkage parameters
are fitted on training shots. Validation chooses background/emission and
physical model structure. The test set is scored once. A separate alternate
cycle split and within-cycle drift analysis assess time sensitivity.

Held-out count likelihood selects predictive descriptions; it does not create
ground-truth occupancy labels.

The current five-frame latent fit is exploratory and is **not accepted**. Its
ordinary profile likelihood treats site trajectories as conditionally
independent and its posterior check does not establish complete-shot
uncertainty. The public gate therefore fails even though the fitted model
improves ordinary held-out count likelihood. No latent transition estimate or
representative posterior trajectory is promoted to a public result.

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

## Bright-wait interpretation

The imaging light is commanded on during the swept wait before image 1. The
public terms are:

- effective bright-wait decay;
- `lambda_bright_effective` and `tau_bright_effective`;
- post-wait retention trend.

A rate extrapolated over 50 ms is compared with apparent inter-readout loss
using a joint bootstrap. If it is smaller, the allowed conclusion is:

> The simple constant-rate bright-wait model does not explain the full
> inter-readout loss.

The data do not prove a fixed per-exposure or fixed per-pulse mechanism.
Turn-on/off transients, nonstationary heating, state selection, classification
error, background, timing accounting and cross-sequence differences remain
alternatives.

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
- Commanded timing is internally corroborated, but no per-frame camera
  timestamp or independent realized-exposure readback is stored.

These limitations are stop conditions for causal or intrinsic-lifetime claims,
not reasons to suppress the operational measurements.
