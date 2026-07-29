# Prioritized next experiments

The current sweeps identify operational timing dependences under one
non-optimized cooling condition. They do not identify empirical readout error,
fully dark loss, heating, or a fixed per-pulse loss. The experiments below are
ordered by how directly they remove those ambiguities.

All sweep points should be randomized within balanced acquisition blocks.
Record the randomization seed, commanded and measured timing, switch state,
DDS/RF state, camera configuration and acquisition order. A block should
contain every condition once before any condition repeats.

## Minimal next dataset

Run a pulse-count by total-light-time factorial with a fully dark control and
pre/post reference labels:

- total commanded imaging-light time: 25, 50, 100 and 200 ms;
- pulse count: 1, 2 and 4;
- same total light time divided equally among pulses;
- matched wall-clock dark control at every total duration;
- current switch-off state and a verified DDS/RF-off state;
- pre-reference, test pulse(s), post-reference;
- randomized complete blocks.

A pilot of 8 independent shots per cell requires
`4 durations x 3 pulse counts x 2 optical-off states x 8 = 192` test shots,
plus matched dark/reference controls. Use it to verify timing, loading,
background and effect scale—not to publish a precise mechanism. A full run of
25-30 shots per cell requires 600-720 test shots and supports shot-cluster
uncertainty and a pulse-count by light-time interaction.

This one factorial separates:

- a continuous light-time hazard (effect changes with total light time);
- a fixed pulse cost (effect changes with pulse count at fixed light time);
- natural wall-clock loss (matched dark control);
- residual switch-off loss (current switch state versus verified DDS/RF off);
- classification error and reference-induced loss (pre/post references and
  explicit reference-only controls).

## 1. Fully dark control

Compare two inter-frame hold configurations:

1. the current optical-switch-off command with imaging DDS/RF commands retained;
2. optical switches off **and** DDS/RF amplitude disabled, or another hardware
   state independently shown to produce no imaging field.

Measure or bound optical extinction at the atoms with a suitable detector or
calibrated proxy. Use randomized holds spanning approximately
`0.1, 0.3, 0.7, 1.5, 3 and 5 s`.

- Pilot: 10 shots per hold and state, 120 shots total.
- Full run: 30 shots per hold and state, 360 shots total.

Include equal-duration reference-only and no-reference controls. This
experiment determines whether the present switch-off decay is compatible with
a fully dark decay; it is the minimum measurement needed before using the term
intrinsic dark lifetime.

## 2. Exposure-duration sweep

Hold cooling settings, detunings, powers, camera region and inter-frame delay
fixed. Sweep the commanded exposure duration through approximately
`10, 25, 50, 100 and 200 ms`. Add shorter points if the camera trigger supports
them and count separation remains measurable.

- Pilot: 10 shots per duration, 50 shots.
- Full run: 30 shots per duration, 150 shots.

Acquire matched empty-camera frames without atoms and matched light-off frames.
The short-duration points reveal an intercept-like cost; the longer points
constrain a proportional light-time hazard. Duration alone cannot distinguish
a pulse cost from a turn-on transient, which is why the factorial below is
required.

## 3. Pulse count versus total light time

At fixed total light-on time, compare one long pulse with several shorter
pulses. A compact design is:

| total light time | pulse counts |
|---:|---|
| 25 ms | 1, 2, 4 |
| 50 ms | 1, 2, 4 |
| 100 ms | 1, 2, 4 |
| 200 ms | 1, 2, 4 |

Keep cumulative light time, cooling state and total wall-clock duration
matched. Log each turn-on and turn-off edge.

- Pilot: 8 shots per cell, 96 shots.
- Full run: 25-30 shots per cell, 300-360 shots.

An effect of pulse count at fixed total light time supports a pulse-associated
cost. An effect only of total light time supports a continuous illuminated
hazard. Their interaction tests whether the cost per pulse changes as atoms
heat or the survivor population changes.

## 4. Reference-labelled sequence

Use:

1. a pre-reference readout;
2. the controlled test frame or pulse sequence;
3. a post-reference readout.

Add reference-only controls with the same number and duration of reference
pulses. Vary the order of control and test blocks across randomized cycles.

- Pilot: 20 shots per control/test condition.
- Full run: at least 50 shots per condition when estimating rare transitions.

No fluorescence reference is perfectly nondestructive. The controls are needed
to separate test-sequence loss from reference-induced loss and to define an
empirical label protocol.

## 5. Natural-loss control

For every principal wall-clock duration, run an otherwise identical sequence
with imaging light off and verified DDS/RF amplitude off. Preserve trap,
cooling and camera-trigger timing where possible.

- Pilot: 10 shots at each of 4 durations, 40 shots.
- Full run: 30 shots at each duration, 120 shots.

This is the comparison required to distinguish illuminated loss from natural
loss over the same elapsed time.

## 6. Optimized-cooling repeat

Repeat the benchmark only after cooling parameters are optimized and a
temperature or cooling-performance diagnostic is recorded. Do not overwrite
the present condition: label current and optimized datasets separately.

- Pilot: 20 shots at the shortest, middle and longest conditions.
- Full run: repeat the chosen primary sweep with at least 25 shots per point.

The comparison determines whether the present effective rates characterize a
suboptimal preparation rather than an apparatus limit.

## 7. Cross-day frozen evaluation

Fit geometry, background template, emission parameters, thresholds and model
structure on one or two development days. Freeze them before opening a later
day's outcomes.

- Development: at least 20 shots per condition on each of two days.
- Frozen test: at least 30 shots per condition on a third day.

Record any purely coordinate-registration transform separately from statistical
parameters. This tests whether the inference generalizes beyond one
experimental date.

## 8. Randomized and interleaved acquisition

The 2026-07-28 sweeps repeat ascending condition cycles. Future acquisition
must randomize condition order inside complete blocks. If hardware constraints
prevent full randomization:

- use a balanced Latin-square order across blocks;
- reverse alternating blocks;
- record block and within-block position;
- include periodic fixed-condition drift sentinels.

Use at least 10 complete blocks for a pilot and 25-30 for a full run. Split
train/validation/test by complete blocks, not by sites or frames.

## Decision sequence

1. Run the 192-shot minimal factorial pilot and inspect timing, geometry,
   site-free background and reference-label stability.
2. If the reference protocol is usable, run the 600-720-shot full factorial.
3. If current switch-off and verified DDS/RF-off states differ, prioritize the
   fully dark lifetime sweep.
4. If pulse count matters at fixed total time, refine turn-on/off timing and
   pulse count; otherwise expand exposure duration.
5. Repeat the frozen analysis after cooling optimization and on a later day.

No pilot significance threshold should be used as a mechanism-discovery
stopping rule. The pilot sets operating ranges and variance; the full-run
design and analysis should be frozen before acquisition.
