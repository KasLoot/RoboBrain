# Failure-detection experiment report

Date: 2026-09-02

## Scope and constraints

All experiment code and generated results are contained in
`tests/experiments/`. Production remained unchanged during the experiment
phase; the selected policy was implemented afterward.

Ground truth supplied for the failed attempt:

- sustained plug rotation: frames 5493--6020;
- plug loose on the table: frames 6030--6060;
- the preceding eight attempts are successful controls.

The benchmark also includes the known successful-attempt rotation confusion
around frames 1815--2041 and the previous drop confusion at frame 3602.

## Root cause confirmed

The supplied 1080p run did not miss all visual evidence:

- both rotation voters produced `R/R` repeatedly, but rotation terminal output
  was disabled and therefore only recorded as a shadow alert;
- drop returned `D` at frame 6049, but production required a second positive
  observation on a later frame;
- the next production observation was frame 6106, after the roughly one-second
  drop interval had ended.

The original five-request 1080p run had a 1.956 s monitoring median. In the
repeated fixed-frame reproduction, endpoint variability increased the measured
median to 2.351 s and p95 to 3.998 s. A two-future-frame confirmation rule is
therefore structurally unable to guarantee detection of a one-second event.

Enabling raw rotation alerts is also unsafe: the successful attempt around
frames 1919--1988 produces a short `R/R` burst.

## Endpoint availability

Both stale SSH forwarding processes were replaced with fresh tunnels. Port
8000 then responded and exposed a Gemma 4 26B A4B IT server configured with
`max_soft_tokens=1120`.

The port-9000 SSH tunnel is healthy, but remote port 9000 rejects connections.
Remote inspection found no listener for 9000 and only the 1120-token vLLM
process, which was using about 93 GB of GPU memory. The 560-token experiments
could not be run without stopping or replacing the working server, which was
not done.

## Fixed-frame results

Metrics count the repeated labeled frames. Rotation precision below excludes
frame 5457 because it is only 36 frames before the supplied onset and is a
boundary disagreement rather than the clear successful-attempt counterexample.
Drop metrics exclude the leaked positive frame 6040 where noted.

| Configuration | Rotation recall | Rotation precision | Drop recall | Drop precision | p50 round | p95 round |
|---|---:|---:|---:|---:|---:|---:|
| Production, 1080p/1120 | 56.5% | 79.6% | 100% | 80.0% | 2.351 s | 3.998 s |
| Production, 848x480/1120 | 60.9% | 76.4% | 100% | 100% | 1.995 s | 2.244 s |
| Current-focused four-way, 1080p/1120 | 95.7% | 91.7% | 100% | 100% | 1.598 s | 2.005 s |
| Current-focused four-way, 848x480/1120 | 4.35% | 27.3% | 100% | 60.0% | 1.563 s | 1.730 s |
| Rotation-only current-focused, 1080p/1120 | 95.7% | 91.7% | n/a | n/a | 0.898 s | 1.085 s |
| Primary three-way, 1080p/1120 | 95.7% | 91.7% | 100% | 100% | 1.293 s | 1.790 s |
| Production primary four-way, 1080p/1120 | 95.7% | 84.6% | 100% | 100% | 1.560 s | 1.753 s |

The specialized rotation-only cold first pass was 1.004 s p50 / 1.197 s p95.
Its lower aggregate values include server prefix-cache benefit on repeated
identical requests. Live replay was 0.948 s p50 / 1.442 s p95.

### Why 848x480 is rejected

The current 848x480/1120 production-equivalent run was not materially faster
at median latency and produced 41 false `I/I` insertion-consensus records over
three repeats. These included multiple frames inside the failed rotation
sequence. Since a single `I/I` result currently terminates an attempt as
complete, this resolution can convert a real failure into a false success.

The reduced current-focused layout was worse at 848x480: rotation recall fell
to 4.35% and same-frame drop precision to 60%. The 180x170 tight panel does not
retain enough detail for this layout.

### Current-focused image layout

The successful ablation uses three small references above one large, native
1080p `CURRENT` panel instead of six equal-sized panels. This increased the
effective current-frame detail while reducing request cost.

The two rotation banks retained independent normal and rotated examples. For
drop, bank A uses dropped, controlled, and inserted examples; bank B uses
dropped, stand, and human examples.

### Temporal rotation policy

The following policies were replayed independently over all three fixed-frame
repeats:

| Required consecutive `R/R` observations | Successful control | Failed attempt | First failure evidence |
|---:|---|---|---:|
| 1 | false alert in every repeat | detected | frame 5457 |
| 2 | false alert in every repeat | detected | frame 5493 |
| 3 | no false alert in any repeat | detected in every repeat | frame 5516 |
| 4 | no false alert in any repeat | detected in every repeat | frame 5546 |

Three consecutive observations are the selected threshold. Unlike the current
elapsed-source-frame rule, an unobserved gap does not count as positive
evidence.

## Live-cadence replay

The replay advances the source by the measured VLM inference time and does not
count offline HEVC seek/decode overhead. All eight successful attempts were
replayed as active monitoring intervals.

### Rotation-only finalist

- Successful attempts without terminal false alert: 8/8.
- Failed-attempt detection: 3/3 live-cadence repeats.
- Evidence frames: 5536, 5524, and 5548.
- Completed-alert source-equivalent frames: 5564, 5552, and 5590.
- Completed-alert delay from frame 5493: 2.0--3.2 s.
- Live p50/p95 round latency: 0.948/1.442 s.

### Primary three-way finalist

The continuous primary batch contains rotation A, rotation B, and conservative
drop A. Drop B is requested only when drop A is positive, using the same
retained source frame.

- Successful attempts without terminal false alert: 8/8.
- Failed attempt detected as rotation at evidence frame 5540.
- Alert completed around source-equivalent frame 5577, 2.8 s after onset.
- Live p50/p95 primary round latency: 1.350/2.282 s.
- A targeted replay starting at frame 6010, too late to accumulate rotation,
  sampled the drop at frame 6052 and confirmed `D/D` on that retained frame.
  It emitted `DROP_OR_CONTROL_LOSS`; the response completed around frame 6104.

The primary three-way design is the recommended production candidate because
it preserves continuous drop coverage. The rotation-only lane is the preferred
latency benchmark and could be used when a separate drop lane is available.

## Drop-reference leakage ablation

The current positive drop reference is source frame 6040, inside the evaluated
failure interval. A current-only, no-positive-reference detector was tested to
measure dependence on that reference:

- cold p50/p95: 0.593/0.902 s;
- drop recall excluding frame 6040: 75%;
- drop precision: 75%;
- it consistently misclassified successful insertion frame 4072 as a drop.

The leakage-free detector is therefore unsuitable as a terminal decision. It
can be a cheap trigger, but still requires the reference-based retained-frame
verification. The reference-based two-vote drop lane achieved 100% recall and
precision on the tested frames, including every sampled frame from 6030--6060,
but its generalization remains unproven because of the reference leakage.

## Implemented production policy

Use 1920x1080 input with the 1120-token endpoint and:

1. Replace equal-tile safety grids with the tested current-focused layout.
2. Continuously issue rotation A, rotation B, and conservative drop A in one
   primary batch.
3. Emit rotation only after three consecutive same-frame `R/R` consensuses.
4. On drop-A `D`, retain that exact frame and invoke drop B; emit drop only on
   same-frame `D/D`.
5. Never infer sustained evidence from the number of source frames skipped
   while inference was running.
6. Give confirmed safety failure priority over completion.
7. Keep insertion/completion and reset/attempt management outside the primary
   safety cadence; the 848x480 false-completion result shows why these paths
   need separate validation.
8. Preserve raw specialist outputs, sampled source frame, completion time, grid
   variant, and policy streak in the production log.

The production primary batch also includes insertion trigger A so successful
attempts retain same-frame completion coverage. Insertion B runs only after a
positive trigger and only after safety fusion for that frame. The exact
four-request production-primary benchmark retained 95.7% rotation recall,
100% tested drop-trigger recall/precision, and hit all 8/8 labeled insertion
frames. It measured 1.560 s p50 and 1.753 s p95 client-observed primary latency
on the final server run. Three false insertion triggers were safely routed to
verification rather than treated as completion.

## Limitations and next evidence needed

- Only one labeled failed attempt was supplied.
- The drop positive and likely some rotation failure references are derived
  from the same scene, so same-video performance is diagnostic rather than a
  generalization guarantee.
- The 560-token endpoint was unavailable and has no measured result.
- Before deployment, the selected policy should be evaluated on independent
  success, rotation, and drop videos with references sourced from other runs.

## Main artifacts

- `benchmark_frames.json`: fixed labels and intervals.
- `run_vlm_benchmark.py`: resumable source/endpoint/variant benchmark.
- `analyze_vlm_benchmark.py`: accuracy and latency summaries.
- `evaluate_fusion_policies.py`: temporal threshold comparison.
- `run_safety_replay.py`: live-cadence attempt replay.
- `results/summary_all_completed.json`: consolidated fixed-frame metrics.
- `results/replay_1920x1080_tokens-1120_primary-3way_*.summary.json`:
  primary-candidate replay outcomes.
- `results/replay_1920x1080_tokens-1120_rotation-only_*.summary.json`:
  fastest rotation-lane replay outcomes.
