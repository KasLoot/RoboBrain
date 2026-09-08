# RoboBrain rotation specialist — experiment 12

Experiment 12 achieved **197/198 correct episode outcomes (99.49%)** across three complete sequential replays. **Every clip is included.** Dataset 1 `traj-93.MOV` is the sole system error and is documented below as a gripper-occlusion edge case. Production code is unchanged.

| Dataset | Correct outcomes | Correct SUCCESS | Annotated failures detected | False FAIL | ON_GOING |
|---|---:|---:|---:|---:|---:|
| [Full model, August 28](runs/all-105/scored_summary.json) | 104/105 | 102 | 2/2 | 1 | 0 |
| [Full model, August 27](runs/aug27-45/scored_summary.json) | 45/45 | 44 | 1/1 | 0 | 0 |
| [No force, August 28](runs/no-force-48/scored_summary.json) | 48/48 | 41 | 7/7 | 0 | 0 |
| **Total** | **197/198** | **187** | **10/10** | **1** | **0** |

There were no false SUCCESS outcomes. Nine annotated failures ended as DROP_FAILURE; Dataset 1 traj-9 ended as ROTATION_FAILURE. The system also incorrectly reported ROTATION_FAILURE for traj-93. No clip needs additional frames to resolve an ON_GOING result in this run.

**Known system error — Dataset 1 traj-93.** Its annotation is SUCCESS, but the system reported FAIL / ROTATION_FAILURE at 29.00 seconds. Nine of ten rotation votes were positive, using frames from 15.87–16.77 seconds. The gripper obscures the plug during this interval, preventing a reliable assessment of its orientation. Repeated observations reinforced the same ambiguous view. Per the user's clarification, this is counted as an incorrect system result; the clip is neither omitted nor relabeled. The [edge-case record](runs/all-105/rotation-policy-review.json), [raw requests](runs/all-105/traj-93.jsonl), and [review images](runs/all-105/review/rotation-review-93-1.jpg) preserve the evidence.

The ten correctly classified failure clips are Dataset 1 traj-9 and traj-82, Dataset 2 traj-30, and Dataset 3 traj-4, 14, 31, 32, 37, 42 and 47. Visual review supports the current no-force annotations, including the previously extended traj-32 clip. No annotation was changed during these experiments.

The reusable `Specialist(system_prompt, config)` supplies each specialist's prompt, native crop, reference images, response schema, observation-to-vote mapping and threshold. Success, drop and rotation run concurrently. A positive observation starts one shared verification round: ten future timestamps across one second, three requests per timestamp, and all **30 responses** awaited. Confirmation requires **more than five** positive votes. The trigger does not vote; the next normal batch supplies timestamp zero. Normal monitoring continues while verification is pending, and another verification round cannot overlap it. Conflicting success/failure confirmations leave ON_GOING; an accepted terminal decision is latched for that attempt.

The simulator publishes at 30 fps. Application crops are never resized: success 470×400, drop 1290×760, rotation 520×540. Tests used `/workspace/models/gemma-4-26B-A4B-it` through `http://127.0.0.1:9000/v1`, temperature 0 and thinking disabled. Success requires a seated plug after release; rotation requires a visibly retained, rotated grasp. The final drop prompt distinguishes support on the stand's top face from a dropped plug lying beside or behind the stand. This fixed the August 27 traj-36 false alarm and no-force traj-4 miss found in earlier iterations.

| Runtime measurement | August 28 full model | August 27 full model | No force |
|---|---:|---:|---:|
| Completed requests | 6,833 | 2,962 | 3,252 |
| Request errors | 0 | 0 | 1 |
| Median / p95 request latency | 1.56 / 2.79 s | 1.65 / 10.97 s | 1.63 / 10.93 s |
| Median normal batch rate | 0.64 Hz | 0.62 Hz | 0.63 Hz |
| Median verification completion | 2.65 s | 2.96 s | 3.12 s |
| Longest verification completion | 13.12 s | 12.08 s | 11.97 s |
| Scheduler audit violations | 0 | 0 | 0 |

Dataset 1 traj-9 confirmed rotation at **27.73 seconds**, approximately **4.87–5.07 seconds before its recorded drop**. Its rotation was visible around 13.4–13.8 seconds, giving a **13.93–14.33 second** detection delay. Traj-82's drop delay was 5.04–5.14 seconds. Dataset 2 traj-30 confirmed DROP_FAILURE with a 12.58–12.85 second delay; its earlier rotation did not pass verification. No-force drop delays ranged from approximately 4.92 to 6.69 seconds. These timings expose a remaining limitation for prompt robot intervention despite good episode-level classification.

The single request error was malformed JSON from the rotation specialist during normal monitoring of no-force traj-4. It became UNCERTAIN and did not change the final result. Audits found no overlapping rounds, missing or duplicate verification votes, or decisions published before their verification responses completed.

Every terminal evidence crop has visual-review provenance. Exact frame/crop matches reused previous inspections after checking unchanged video and annotation hashes; changed captures were inspected directly. Ambiguous release sequences and failure onset sequences received additional temporal review. This is not frame-by-frame annotation of every video. No premature SUCCESS was observed in the reviewed evidence, but success-onset timestamps are not comprehensively labeled, so no aggregate success-onset latency is claimed.

The annotations primarily label terminal SUCCESS/FAIL, rather than every rotation event. These results therefore do not establish rotation precision or recall across all attempts. FALLING observations currently abstain: drop confirmation needs positive TABLE evidence within the visible landing region. Both additional datasets informed support-prompt corrections, making these final scores regression results rather than untouched holdout estimates. The prior production system's saved results remain separate; this rotation extension introduces the documented traj-93 false failure.

Confirmed outcomes produce simulated control handoff events: stop insertion, release and request recovery after failure, then reset the scene; success requests a reset. No real robot-control or recovery API was called, and the complete physical recovery loop was not tested.

Reproduction and evidence:

- [Standalone implementation and commands](../../test/robobrain-v2-test-12/README.md), [validation](validation.json), [all episode outcomes](runs/scored_combined_summary.json).
- Timing and scheduler audits: [Dataset 1](runs/all-105/scored_audit.json), [Dataset 2](runs/aug27-45/scored_audit.json), [Dataset 3](runs/no-force-48/scored_audit.json).
- Visual findings: [Dataset 1](runs/all-105/visual_review.json), [Dataset 2](runs/aug27-45/visual_review.json), [Dataset 3](runs/no-force-48/visual_review.json). Each run also contains its native evidence crops and raw JSONL request logs.
- Input hashes: [Dataset 1](dataset-manifest-105.json), [Dataset 2](dataset-manifest-aug27.json), [Dataset 3](dataset-manifest-no-force.json).
- [Candidate hashes at evaluation](runs/candidate-sha256-at-run.json), [original evaluation driver](runs/evaluate_all_at_run.txt), [current candidate hashes](candidate-sha256.json). Detector code, prompts, references and configuration were identical across the final three runs; later changes only affect evaluation helpers and reporting.
