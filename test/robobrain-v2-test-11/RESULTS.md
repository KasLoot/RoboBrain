# Experiment 11 results

User review update: Dataset 1 traj-93 remains in the evaluation and counts as a system error. The gripper occludes the plug, making this a difficult visual edge case. The original SUCCESS annotation and erroneous ROTATION_FAILURE prediction are preserved.

## Development dataset

The complete 105-episode replay matched **104/105 original labels (99.05%)**: 102 SUCCESS outcomes, both annotated failures detected, and one rotation decision on the SUCCESS-labeled traj-93. There were no unresolved episodes, request errors or scheduler violations.

Traj-9 confirmed ROTATION_FAILURE at **29.38 s**, approximately **3.2–3.4 s before its recorded drop**. Its reviewed rotation-onset-to-decision interval was **15.6–16.0 s**. Traj-82 confirmed DROP_FAILURE approximately **5.2–5.3 s after its reviewed drop onset**.

Traj-93 again produced ROTATION_FAILURE, at 28.16 s, using ten positive rotation votes from 16.03–16.93 s. The gripper obscures the plug during that interval. This is counted as a system error against its unchanged SUCCESS annotation, with the visual occlusion documented as an edge case.

| Measurement | Result |
|---|---:|
| Original labels matched | 104/105 |
| Correct SUCCESS labels | 102/103 |
| Annotated failures detected | 2/2 |
| Total rotation decisions | 2, including traj-93 |
| ON_GOING results | 0 |
| Completed requests | 6,762 |
| Request errors / scheduler violations | 0 / 0 |
| Median / p95 request latency | 1.63 / 2.72 s |
| Median normal sampling frequency | 0.63 Hz |
| Median verification completion time | 2.87 s |

Some moving-frame verification bursts still take about 12 s. Correct episode outcomes do not establish a low-latency robot-control guarantee.

All terminal evidence has visual-review provenance. Exact source-frame/crop matches reused the earlier visual review after checking unchanged video hashes; changed captures were inspected directly. The ambiguous close-gripper release sequences and traj-93's subsequent successful insertion remain documented.

## Stand correction

The additional stand reference and support clarification fix the known August 27 traj-36 regression in a targeted replay: the early views are STAND or UNKNOWN, and the clip finishes SUCCESS at 31.80 s. The four-episode development pilot also passed, including early rotation detection in traj-9 and drop detection in traj-82. The image screen matched 121/123 expected votes; its exceptions were the deliberate airborne-drop abstention and one blurred rotation miss.

The complete August 27 regression matched **45/45 labels**, including the corrected traj-36. There were no unresolved clips, request errors or scheduler violations. Its drop case traj-30 confirmed FAIL at 20.22 s, about 5.5–5.8 s after the reviewed drop onset. The August 27 recording informed this correction, so this is a regression test rather than an untouched holdout.

The no-force run exposed a new support-classification error: traj-4 visibly drops beside/behind the stand, but the added stand reference caused persistent STAND classifications. It remained ON_GOING through the 90-second watchdog. The clip already shows the landing; missing footage is not the cause. This partial run was stopped after 23 completed episodes, with 22 label matches and that one unresolved failure. It is not a complete 48-episode evaluation. All output is preserved, including `runs/no-force-48/STOPPED.json`.

Experiment 12 clarifies the difference between support on the block's top face and a dropped adapter lying beside or behind it. No production code was changed.

## Artifacts

- [Development results](runs/all-105/summary.json)
- [Timing and scheduler audit](runs/all-105/audit.json)
- [Visual-review provenance](runs/all-105/visual_review.json)
- [Traj-93 policy review](runs/all-105/rotation-policy-review.json)
- [Stand regression](runs/stand-regression/summary.json)
- [Standalone and workflow validation](validation.json)
- [Frozen candidate hashes](candidate-sha256.json)

No production code was changed. Recovery/reset handoffs are simulated log events; no real robot API was connected. FALLING remains an uncertain observation, so drop confirmation requires subsequent positive TABLE evidence in the visible landing region.
