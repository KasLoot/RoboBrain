# Experiment 3 results

The current-only 1120-token implementation completed all 105 episodes with 104 correct outcomes: 102 of 103 successes and both drops. There were no incorrect terminal labels. Episode 58 remained ON_GOING until the benchmark watchdog because the success specialist interpreted the final gripper projection as continued contact.

The 560-token comparison was stopped after 64 completed episodes: 54 correct, six unresolved successes, and four false drop alarms. Its GPU was reassigned to the next experiment. This is a partial comparison, not a 105-episode result.

| Metric | 1120-token complete run |
|---|---:|
| Episode accuracy | 104/105 (99.05%) |
| Successful episodes confirmed | 102/103 |
| Drop failures detected | 2/2 |
| Incorrect terminal labels | 0 |
| Unresolved episodes | 1 |
| Completed requests | 6,629 |
| Request errors | 19 |
| Median request latency | 1.08 s |
| Request latency, 95th percentile | 7.28 s |
| Median normal monitoring frequency | 0.65 Hz |
| Median verification duration | 2.31 s |
| Completed verification rounds | 173 |
| Scheduler audit violations | 0 |

The two reviewed drop-onset intervals give event-to-decision latencies of 3.17–3.37 seconds for episode 9 and 3.92–4.22 seconds for episode 82. Episode 9 was not declared failed during its preceding rotation. Exact success-onset annotations are unavailable; elapsed episode time is not presented as success detection latency.

The visual review includes terminal evidence frames and additional short sequences for close-release views. This recording, including its reference images, is a development benchmark. It does not establish performance on independent recordings.

Experiment 4 tested temporal inputs to resolve the close-release ambiguity but regressed in its 560-token screening. Experiment 5 instead tightens the native success crop while retaining this experiment's drop specialist and scheduler.

Raw results: `runs/all-105-1120/summary.json`, `runs/all-105-1120/audit.json`, and per-episode JSONL logs. The stopped comparison is documented in `runs/all-105-560/STOPPED.md`.
