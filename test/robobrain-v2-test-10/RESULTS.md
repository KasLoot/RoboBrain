# Experiment 10 results

User review update: Dataset 1 traj-93 remains in the evaluation and counts as a system error. The gripper occludes the plug, making this a difficult visual edge case. The original SUCCESS annotation and erroneous ROTATION_FAILURE prediction are preserved.

## Development dataset: all 105 episodes

The frozen candidate matched **104/105 original episode annotations (99.05%)**. It reported 102 SUCCESS outcomes and detected both annotated failures: traj-9 as ROTATION_FAILURE and traj-82 as DROP_FAILURE. No episodes remained ON_GOING. There were 6,729 completed requests, zero request errors and zero scheduler audit violations.

The sole incorrect outcome is traj-93, labeled SUCCESS but reported ROTATION_FAILURE. Verification sampled 15.83–16.73 s and reported FAIL at 28.36 s. The gripper obscures the plug during the relevant interval, so the apparent grasp orientation cannot be reliably judged. Per the user's clarification, this counts as a false failure; it is not excluded or relabeled.

| Measurement | Result |
|---|---:|
| Correct original episode labels | 104/105 |
| Correct SUCCESS labels | 102/103 |
| Detected annotated failures | 2/2 |
| Total rotation decisions | 2, including the traj-93 disagreement |
| Unresolved episodes | 0 |
| Request errors / scheduler violations | 0 / 0 |
| Median / p95 request latency | 1.61 / 3.11 s |
| Median normal monitoring rate | 0.63 Hz |
| Median verification completion time | 2.84 s |

Traj-9's rotation decision was published at **27.86 s**, approximately **4.7–4.9 s before its recorded drop** at 32.6–32.8 s. Its rotation was visually apparent around 13.4–13.8 s, so rotation-onset-to-decision latency was approximately 14.1–14.5 s. Traj-82's drop decision was published at 26.95 s, approximately 6.0 s after the reviewed drop onset. Moving-frame verification bursts can still take about 12 s on this endpoint; these results do not establish a low-latency deployment guarantee.

All 105 terminal evidence crops were visually inspected. Additional release sequences were reviewed for the close-gripper cases 13, 34, 43, 61, 64 and 67. Traj-93's occlusion and later successful insertion are saved as an edge-case record. Success-onset timestamps are not annotated, so no aggregate success-onset latency is claimed.

This candidate differs from experiment 8 only in its configured drop vote mapping: FALLING abstains; TABLE remains positive. That prevents visually ambiguous moving plugs from starting false drop-verification rounds. It also means an airborne-only drop, or one that leaves the visible landing region, cannot be confirmed without subsequent positive evidence. Success and rotation prompts, native images, thresholds and the monitor are unchanged.

## Additional datasets

The complete August 27 evaluation matched **44/45 original labels**. Traj-36 was a genuine false DROP_FAILURE: the adapter was still on its stand, partly hidden by the robot. Its 3.13–4.03 s verification window received seven erroneous TABLE votes and reported FAIL at 14.45 s. The pickup sequence confirms that no drop occurred. All other episode labels were correct and no episodes remained ON_GOING.

Traj-30's early rotation observation triggered verification, but its future window spanned the subsequent drop. Several retained-grasp observations were incorrectly called NOT_HELD by the rotation specialist, and only three TABLE votes were collected. A second round correctly confirmed DROP_FAILURE at 28.91 s, approximately 14.2–14.5 s after the drop. Correct eventual episode classification therefore does not establish reliable early rotation detection.

There were 2,960 completed requests, one transient disconnected rotation request at the start of traj-23 (mapped to UNCERTAIN), and no scheduler violations. Every terminal evidence image was reviewed. [August 27 results](runs/aug27-45/summary.json) and [visual findings](runs/aug27-45/visual_review.json) are preserved.

Experiment 10 was not run on the no-force dataset after this confirmed false alarm. Experiment 11 adds a stand reference drawn from Dataset 1 and clarifies partially obscured support. Because this correction was motivated by the August 27 error, subsequent tests on that recording are regression tests, not untouched held-out evaluations.

## Artifacts

- [105-episode raw results](runs/all-105/summary.json)
- [Timing and scheduler audit](runs/all-105/audit.json)
- [Visual review scope and findings](runs/all-105/visual_review.json)
- [Traj-93 policy review](runs/all-105/rotation-policy-review.json)
- [Frozen candidate hashes](candidate-sha256.json)
- [Development input hashes](dataset-manifest-105.json)
- [Standalone validation](validation.json)

The original prompts, native reference images, source and configuration are snapshotted in each run. Runtime code was not tuned using the additional datasets. Production code remains unchanged. Recovery/reset handoffs are simulated log events; no robot-control API is connected.
