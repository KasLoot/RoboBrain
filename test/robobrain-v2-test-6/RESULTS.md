# Experiment 6: evaluation results

## Updated episode 94

The user extended `traj-94.MOV` with more frames after insertion. Rerunning only that clip with experiment 6 and the same 1120-token endpoint produced **SUCCESS at 18.78 seconds of simulation**, with 10/10 success votes and 0/10 drop votes in its first verification round. All twenty verification responses completed before the decision. There were 50 completed requests, no request errors, and no scheduler audit violations. Visual inspection of the terminal evidence confirms the adapter is seated and the gripper is clear above it.

The implementation, full configuration, prompts, and reference images are byte-identical to the original run snapshots. Only the input clip changed; no production code was modified. The original run is preserved.

Replacing the old episode 94 result with this rerun gives **105/105 correct episode outcomes**. This combines the new episode 94 result with the previous 104 episode results; it is not a fresh run of all 105 episodes.

| Updated combined outcome | Count |
|---|---:|
| Correct episode outcomes | 105/105 |
| SUCCESS confirmations on successful episodes | 103/103 |
| DROP_FAILURE confirmations on failed episodes | 2/2 |
| False failure labels | 0 |
| False success labels on failed episodes | 0 |
| Unresolved episodes | 0 |

Rerun artifacts:

- [Single-episode result](runs/rerun-traj-94-updated/summary.json).
- [Updated 105-episode combined results and provenance](runs/rerun-traj-94-updated/updated_dataset_summary.json).
- [Rerun timing and scheduler audit](runs/rerun-traj-94-updated/audit.json).
- [Clip hash, unchanged-settings checks, and visual review](runs/rerun-traj-94-updated/validation.json).
- [Terminal evidence image](runs/rerun-traj-94-updated/review/traj-94.jpg).

The 18.78-second measurement is elapsed playback time, not insertion-to-decision latency. The development-benchmark limitations below still apply.

## Original complete run

The measurements in this section describe the original run before episode 94 was extended. That complete 105-episode run matched 104 annotations (99.05%): 102 of 103 successful episodes and both drop failures. Every emitted terminal label matched its episode annotation. The original `traj-94.MOV` remained ON_GOING until the 90-second benchmark watchdog.

This matches experiment 3's completed-run accuracy, while removing an unintended socket-ordinal requirement and resolving the earlier close-release and socket-number cases. It is not a perfect detector, and these results do not establish performance on independent recordings.

| Outcome | Count |
|---|---:|
| Episodes evaluated | 105 |
| Correct episode outcomes, including unresolved episodes in the denominator | 104/105 |
| SUCCESS confirmations on successful episodes | 102/103 |
| DROP_FAILURE confirmations on failed episodes | 2/2 |
| False failure labels | 0 |
| False success labels on failed episodes | 0 |
| Unresolved episodes | 1 |
| Completed requests | 6,495 |
| Request errors | 0 |
| Completed verification rounds | 160 |
| Scheduler audit violations | 0 |

The run used the supplied Gemma 4 26B A4B IT server at `http://127.0.0.1:9000/v1`, configured by the user for 1120 visual tokens. Temperature was zero and every request explicitly set `chat_template_kwargs.enable_thinking=false`.

The success crop is `(1000, 230, 1470, 630)`, producing 470 x 400 pixels. The drop crop is `(330, 100, 1620, 860)`, producing 1290 x 760 pixels. Coordinates are `(x1, y1, x2, y2)`, with exclusive upper bounds. Both are extracted from the original 1920 x 1080 frame and sent without resizing. Each specialist receives two labeled reference images and one current crop.

Normal monitoring sends the two specialists concurrently and starts the next pair when the preceding pair finishes. A positive observation starts one global verification round: ten paired observations over one second, with the next normal pair reused as observation zero. The original triggering observation is excluded. Each specialist requires strictly more than five positive responses. All twenty requests finish before resolving a round; errors and UNCERTAIN responses never contribute positive evidence. Both or neither specialist confirming leaves ON_GOING. Additional triggers are not queued while verification is active.

| Timing measurement | Result |
|---|---:|
| Median completed-request latency | 1.08 s |
| Completed-request latency, 95th percentile | 7.21 s |
| Median normal monitoring frequency across episodes | 0.69 Hz |
| Median verification duration, trigger to completed responses | 1.79 s |
| Median verification observation interval | 0.100 s |
| Verification observation interval, 95th percentile | 0.109 s |
| Episode 9: reviewed drop-onset-to-decision interval | 3.87–4.07 s |
| Episode 82: reviewed drop-onset-to-decision interval | 5.27–5.57 s |

Episode 9 stayed nonterminal during the rotation preceding its drop. The drop intervals come from separately reviewed source frames in `event_times.json`. Terminal evidence frames were visually reviewed; no premature terminal decisions were found in that review. Exact success-onset timestamps are not annotated, so no mean success-detection latency is claimed.

These are eventual episode outcomes, not a guaranteed response-time deadline. Some confirmations were delayed: episode 13's SUCCESS was published at 54.00 seconds of simulation. Request latency and verification duration should not be substituted for event-to-decision latency.

The original run's only miss was episode 94. The drop specialist consistently located the adapter in the socket, but the success specialist repeatedly reported gripper contact. The original recorded sequence showed withdrawal followed by a nearby gripper pose; the current-only classifier did not reliably recognize that final projection as released. The result remained ON_GOING rather than being converted to a failure or an assumed success. The diagnostic sequence is saved in `runs/all-105-1120/review/release-sequence-94.jpg`. The extended-clip rerun above resolves this episode without changing the classifier.

Validation included executable checks for strict thresholds, contradictory observations, unique evidence slots, native image dimensions, disabled thinking, malformed responses, camera cadence, concurrent normal monitoring, complete-response waits, fresh-trigger rearming, and cancellation/timeout cleanup. The final implementation also passed after copying only its own code, configuration, prompts, and images to `/tmp` and running in a fresh `uv` script environment. The final source, prompt/image snapshots, configuration, and 105 unique result records were checked for consistency. Production code and project dependency files were not modified.

The reference images and evaluated episodes come from the same recording. This is a development benchmark with only two failure episodes, not a held-out generalization test. Earlier incomplete runs are explicitly marked STOPPED and are not presented as complete evaluations.

Artifacts:

- `runs/all-105-1120/summary.json`: all 105 episode results.
- `runs/all-105-1120/audit.json`: timing and scheduler checks.
- `runs/all-105-1120/visual_review.json`: visual review scope.
- `runs/all-105-1120/traj-*.jsonl`: observations, verification rounds, and published statuses.
- `runs/all-105-1120/run_config.json`, `run.py`, `prompts/`, and `references/`: exact run snapshots.

From the repository root:

```sh
uv run --script test/robobrain-v2-test-6/run.py \
  --videos video/insert_plug-full_model-aug-28-105trail-1x \
  --base-url http://127.0.0.1:9000/v1
```

Use `--episodes 1 9 82 94` for a short selected run. Status events are written live to each episode's JSONL file, and the terminal result is printed to the console. The existing SSH forward must be available.
