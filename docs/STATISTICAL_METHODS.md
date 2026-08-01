# Statistical methods for optimized lifetime inference

## Experimental unit and frozen splits

The complete experimental shot is the independent unit. Sites and frames
within a shot share preparation, illumination, geometry, and background and
are therefore never treated as independent bootstrap observations.

Bright- and dark-lifetime conditions repeat over 20 complete acquisition
cycles. Cycles 0–11 train, 12–15 validate, and 16–19 form the final test set.
Each repeated-imaging run uses chronological shots 0–59 for training, 60–79
for validation, and 80–99 for test. A seeded permutation of five contiguous
20-shot blocks is used only as a drift sensitivity.

Geometry and fixed background templates use training shots. Validation chooses
the background correction, emission structure, lifetime structure, and
repeated-imaging structure. The test split is then scored once.

## Observation model

For site `i`, frame `f`, and exposure `e`, the background-corrected count is
modelled as a two-component latent mixture. The latent state is empty or
occupied; the occupied component includes shrinkage-estimated site offsets.
Separate emission parameters are retained for 50, 100, and 200 ms exposures.

Candidate shrinkage strengths are compared by validation negative
log-likelihood. The selected five-fold shrinkage balances site heterogeneity
against overfitting. Posterior apparent occupancy follows from Bayes' rule
under the selected mixture. Model-implied overlap is the integral of the
smaller fitted component density and is not an observed error frequency.

Validation-only gates check whether ambiguous counts or extreme residuals
justify a third state or heavy-tail extension. Test observations are not used
to decide whether those models exist. Weak-site rules are likewise fixed from
training diagnostics before test evaluation.

## Dark-lifetime model

The dark run contains five 50 ms frames separated by one declared dark hold
per transition. For transition `j` and hold `t`, apparent retention is

`q_j × exp(-lambda_dark × t)`.

The shared-rate candidate uses one nonnegative hazard and four interval
factors. Alternatives relax the shared-rate structure. Validation NLL chooses
the model; the reported lifetime is `1 / lambda_dark`. The exposure itself and
the 2 µs command overhead do not enter `t`.

## Bright-lifetime model and control

For the first frame after illuminated wait `t`, apparent occupancy is

`pi0 × exp(-lambda_bright × t)`

under the no-floor candidate. A floor candidate adds a long-time occupancy
parameter and must improve validation likelihood enough to pass its complexity
gate. The post-wait frame-1→frame-2 control separately compares a flat
retention factor with a monotone wait-dependent alternative.

The bright lifetime is operational for the recorded sequence. Transfer into
repeated-imaging survival is explicitly checked because the bright-lifetime
run has different pre-heat commands.

## Repeated-imaging models

For a prefix ending after frame `N`, the exact timing covariates are:

- cumulative bright time: `N × exposure_s`;
- cumulative dark time: `(N − 1) × 0.010 s`;
- pulse count: `N`.

M0 is continuous-only. Its survival is

`exp(-lambda_bright × bright_time) × exp(-lambda_dark × dark_time)`.

Each exposure run has a separate initial loading probability. M1 multiplies M0
by a common pulse-associated factor. M2 allows exposure-specific pulse factors;
M3 is a diagnostic extension. A more complex model must improve held-out
validation NLL, exceed the predeclared materiality threshold, have stable
clustered uncertainty, and survive split/background/drift sensitivities.

The M1 improvement is statistically positive but below the materiality
threshold, so M0 remains primary. Conditional M1 pulse factors are sensitivity
parameters, not selected estimates.

## Matched-prefix contrasts

Segmentation is assessed at equal total bright exposure:

- 2×50 ms versus 1×100 ms;
- 4×50 ms versus 2×100 ms versus 1×200 ms;
- 4×100 ms versus 2×200 ms.

Run-specific initial loading is normalized out. The exact count of 10 ms dark
gaps remains in the model rather than being attributed to pulse segmentation.
Direct held-out differences are resampled by complete shot. Rate-model
contrasts are also reported to show the small difference expected from dark-gap
count alone.

## Clustered uncertainty

Primary confidence intervals use 1,000 bootstrap replicates. A replicate
resamples complete shots within condition and carries all of each shot's sites
and frames together. Lifetime models are refit within every replicate.

For repeated imaging, a second 1,000-replicate block bootstrap resamples
contiguous acquisition blocks to preserve slow drift. Whole-cycle bootstraps
for lifetime runs preserve dependence across the ordered condition grid. These
are sensitivities rather than replacements for the predeclared primary
intervals.

Intervals are percentile intervals from successful replicates. The reviewed
run completed all primary and block replicates without failure.

## Loss decomposition

Segment loss probabilities are computed sequentially from multiplicative
survival. For selected M0, residual pulse loss is structurally zero. Bright and
dark terms sum with their interaction accounted for by ordering, so the
reported contributions plus final survival equal one.

Uncertainty propagates the clustered dark and bright hazard draws. The M1
sensitivity adds pulse-factor draws. This is a prior allocation based on
sequence timing and fitted hazards; integrated frame counts do not identify an
exact event location.

## Model and claim gates

A public claim requires all applicable conditions:

1. source timing, geometry, and background gates pass;
2. model structure is selected on validation data;
3. held-out improvement exceeds its materiality threshold;
4. complete-shot uncertainty is stable and scientifically resolved;
5. block, alternate-split, background, and emission sensitivities do not
   reverse the conclusion;
6. wording stays within the observable information.

Failure of a gate is reported as a negative or unresolved result. It is not
replaced by a conditional parameter from an unselected model.
