# Experiment 6: additional dataset evaluation

The latest results are **93/93 correct episode outcomes**, with **85 SUCCESS decisions, 8 DROP_FAILURE decisions, and no remaining ON_GOING clips**. Visual review supports every terminal decision. The monitor, prompts, reference images, crop coordinates, and model settings were unchanged from the recommended experiment 6 baseline.

| Dataset | SUCCESS confirmed | DROP_FAILURE confirmed | ON_GOING | Correct outcomes |
|---|---:|---:|---:|---:|
| `insert_plug-no_force-aug-28-1x` | 41/41 | 7/7 | 0 | 48/48 |
| `insert_plug-full_model-aug-27-1x` | 44/44 | 1/1 | 0 | 45/45 |

There were no incorrect terminal labels. The August 27 result comes from one complete 45-episode run. The latest no-force result combines 47 unchanged results from its complete run with a rerun of the subsequently extended `traj-32.MOV`; it is not a second complete 48-episode run.

## Annotation review and clips needing more footage

**No clips currently need additional frames to resolve an ON_GOING result.**

The original no-force `traj-32.MOV` was annotated SUCCESS, but its footage ended while the gripper still held the adapter during insertion. The monitor remained ON_GOING through the benchmark watchdog, and visual inspection supported withholding a terminal decision. That original SUCCESS annotation was not established by the provided footage.

During testing, this clip was extended and its annotation changed to FAIL. The added footage clearly shows the adapter falling onto the table. The unchanged monitor then confirmed **FAIL / DROP_FAILURE at 20.36 seconds of simulation**, with **10/10 drop votes and 0/10 success votes**. Its decision arrived 3.19–3.39 seconds after the visually reviewed drop onset. The original run and its input hashes remain preserved.

- [Original clip's final insertion sequence](review-new-datasets/insert_plug-no_force-aug-28-1x/traj-32-release-sequence.jpg).
- [Extended clip's drop sequence](review-new-datasets/insert_plug-no_force-aug-28-1x/updated-traj-32/sequence.jpg).
- [Rerun terminal evidence](runs/rerun-no-force-traj-32-updated/review/traj-32.jpg).

The no-force drops are `traj-4`, `traj-14`, `traj-31`, **updated `traj-32`**, `traj-37`, `traj-42`, and `traj-47`. The August 27 drop is `traj-30`. All are visually supported. No other annotation disagreements were found in this review.

I inspected the final source frames of every clip and the actual source evidence used for every terminal decision. I also inspected temporal sequences for every drop and for ambiguous release cases, including the nearby-gripper cases. These showed the plug remaining seated after release. No premature terminal decisions were found in this review. Exact success-onset times are not annotated, so elapsed playback time is not reported as success-detection latency.

## Runtime measurements

Both datasets used the supplied Gemma 4 26B A4B IT endpoint at `http://127.0.0.1:9000/v1`, configured for 1120 visual tokens. Thinking was explicitly disabled and temperature was zero. The 470 × 400 success crop and 1290 × 760 drop crop were sent at their native dimensions without resizing.

The camera runs at 30 fps. Normal specialist pairs run as soon as the preceding pair completes. Verification uses ten paired observations over one second, waits for all twenty request outcomes, requires more than five positive votes per specialist, and allows only one active verification round.

| Measurement | No-force, latest combined results | August 27, complete run |
|---|---:|---:|
| Completed requests represented in these results | 2,781 | 2,625 |
| Request errors | 0 | 1 |
| Completed verification rounds | 67 | 75 |
| Scheduler audit violations | 0 | 0 |
| Median request latency | 1.09 s | 1.14 s |
| Request latency, 95th percentile | 7.19 s | 7.30 s |
| Median normal monitoring frequency across episodes | 0.87 Hz | 0.64 Hz |
| Median verification duration | 1.85 s | 4.93 s |
| Median verification observation interval | 0.100 s | 0.100 s |

The single error was a disconnected success-specialist request in August 27 `traj-45`, verification round 2, slot 2. It contributed UNCERTAIN and no positive vote. The episode still reached the correct SUCCESS decision.

The following latency intervals use separately reviewed drop-onset bounds from each dataset. They measure the visible drop to the published FAIL decision, including verification and response time.

| Dataset | Clip | Drop-to-decision latency |
|---|---|---:|
| No-force | `traj-4` | 3.80–4.40 s |
| No-force | `traj-14` | 3.54–4.14 s |
| No-force | `traj-31` | 3.84–4.44 s |
| No-force | Updated `traj-32` | 3.19–3.39 s |
| No-force | `traj-37` | 3.15–3.75 s |
| No-force | `traj-42` | 3.15–3.75 s |
| No-force | `traj-47` | 3.23–3.83 s |
| August 27 | `traj-30` | 3.64–4.24 s |

These are individual replay results on the supplied clips. No prompts or detector parameters were tuned on these datasets. Rotation-only failure detection remains outside this version's scope.

## Saved results and reproducibility

- [All 93 latest episode results, with dataset identifiers](new_datasets_summary.json).
- [Latest no-force episode results and provenance](runs/rerun-no-force-traj-32-updated/updated_dataset_summary.json), [combined audit](runs/rerun-no-force-traj-32-updated/updated_dataset_audit.json), and [visual assessment summary](runs/rerun-no-force-traj-32-updated/updated_dataset_reviewed_summary.json).
- [August 27 episode results](runs/full-model-aug27-1120/summary.json), [audit](runs/full-model-aug27-1120/audit.json), and [visual assessment summary](runs/full-model-aug27-1120/reviewed_summary.json).
- [Original no-force run](runs/no-force-aug28-1120/summary.json), [record of the dataset change](runs/no-force-aug28-1120/dataset_update.json), and [updated episode 32 rerun](runs/rerun-no-force-traj-32-updated/summary.json).

Each run retains its source, configuration, prompts, reference images, observation logs, input hashes, and validation records. Visual evidence is under each run's `review/` directory and under `review-new-datasets/`.

The evaluation helper `audit.py` now accepts `--event-times` explicitly, preventing timestamps for identically named episodes in another dataset from being reused. Its checks and a dataset-selection check passed. The monitoring implementation and production code were not modified.

From the repository root, a fresh complete run can be started with either command:

```sh
uv run --script test/robobrain-v2-test-6/run.py \
  --videos video/insert_plug-no_force-aug-28-1x \
  --base-url http://127.0.0.1:9000/v1

uv run --script test/robobrain-v2-test-6/run.py \
  --videos video/insert_plug-full_model-aug-27-1x \
  --base-url http://127.0.0.1:9000/v1
```
