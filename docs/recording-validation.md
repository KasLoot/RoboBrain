# Combined-system recording validation

Validated on 25 September 2026 with the Show-Harness insert-plug scene, OpenAI **gpt-6-sol** for planning/control, and OpenAI **gpt-6-luna** for RoboBrain outcome/progress monitoring. The live execution ran for about 15 minutes, finished, and RoboBrain reported **SUCCESS**. The recording requirements passed.

[Run manifest](../runs/recording-validation-20260924T235301Z/robobrain/runs/20260924T235302.866042Z-b29fb00d/run.json) · [Machine-readable verification](recording-validation-metrics.json) · [Recording guide](combined-system-recording.md)

## Recorded outputs

All requested files are in this one run folder:

```text
runs/recording-validation-20260924T235301Z/robobrain/runs/20260924T235302.866042Z-b29fb00d
```

| Output | Verified result |
| --- | --- |
| Show-Harness videos | Seven files: five controller views, original combined camera/decision layout, and 1920×1080 global. Each has **12,808 frames**. |
| RoboBrain videos | Four files: two individual views, 1920×1080 annotated global, and 2640×1080 combined. Each has **27,183 frames**. |
| Robot trajectory | **One initial measured pose and 66 paired controls**, each with its measured final pose. |
| Show-Harness VLM audit | **66 calls**: one planner and 65 controller calls. Every call and HTTP transport start has a terminal record. |
| RoboBrain VLM audit | **804 requests**: 307 success, 307 drop, and 190 progress requests. **797 responses and seven cancellations**, all paired. No HTTP error responses. |
| Request images | **1,769 referenced image assets** verified against their SHA-256 values. |
| Runtime health | No recording error, no dropped physics events, real-time factor **0.99994**. |
| Credential check | No configured OpenAI credential found in the run's JSON, JSONL, or text logs. |

Representative first, middle, and last frames decoded from all 11 videos. Video frame counts match each corresponding frame index. The final combined frames were inspected visually: RoboBrain's individual views are on the left, the global view is on the right, and task status/progress appears at the global view's bottom-right. Original Show-Harness camera titles, depth view, and decision panel are preserved.

Open the main videos and data:

- [RoboBrain combined](../runs/recording-validation-20260924T235301Z/robobrain/runs/20260924T235302.866042Z-b29fb00d/robobrain-video/combined.mp4), [annotated global](../runs/recording-validation-20260924T235301Z/robobrain/runs/20260924T235302.866042Z-b29fb00d/robobrain-video/global.mp4), [view 1](../runs/recording-validation-20260924T235301Z/robobrain/runs/20260924T235302.866042Z-b29fb00d/robobrain-video/view-camera1.mp4), [view 2](../runs/recording-validation-20260924T235301Z/robobrain/runs/20260924T235302.866042Z-b29fb00d/robobrain-video/view-camera2.mp4).
- [Show-Harness combined](../runs/recording-validation-20260924T235301Z/robobrain/runs/20260924T235302.866042Z-b29fb00d/show-harness-video/combined.mp4), [global](../runs/recording-validation-20260924T235301Z/robobrain/runs/20260924T235302.866042Z-b29fb00d/show-harness-video/global.mp4).
- [Robot trajectory](../runs/recording-validation-20260924T235301Z/robobrain/runs/20260924T235302.866042Z-b29fb00d/robot-trajectory.jsonl), [controller model events](../runs/recording-validation-20260924T235301Z/robobrain/runs/20260924T235302.866042Z-b29fb00d/show-harness-vlm/vlm.jsonl), [monitor model events](../runs/recording-validation-20260924T235301Z/robobrain/runs/20260924T235302.866042Z-b29fb00d/robobrain-vlm/20260924-235304-859531/calls.jsonl).
- [RoboBrain final preview](../runs/recording-validation-20260924T235301Z/robobrain/runs/20260924T235302.866042Z-b29fb00d/robobrain-final-preview.jpg), [Show-Harness final preview](../runs/recording-validation-20260924T235301Z/robobrain/runs/20260924T235302.866042Z-b29fb00d/show-harness-final-preview.jpg).

## Timing and retained trial details

The Show-Harness render/record worker captured approximately 14 frames per second while its presentation video is encoded at 30 fps. Its **426.93 seconds of video covers about 909.13 seconds of simulation activity**; use `show-harness-video/frames.jsonl` for actual timestamps. It reports 13,373 omitted recording intervals. RoboBrain recorded 906.10 seconds of video over 906.74 seconds, with 20 omitted sample slots and four visibly marked unavailable-image frames. Neither recording worker blocked physics. The complete run occupies about **6.24 GB**.

The first Show-Harness video frame in this retained live trial predates service identity assignment; its state sequence and simulation time are present, and its containing run manifest supplies scene provenance. Startup recording now waits for the identity, and the network/reset recording check was rerun successfully with an assertion that every frame has an episode and service ID. The trial's original metadata was preserved.

The live trial used offscreen rendering. An initial GUI launch exposed a mixed OpenGL-context failure. The native viewer now runs in a separate process; real GUI-plus-recording checks passed at 64 and 512 camera resolution, including the original 2624×1632 combined canvas, Full HD global video, continuing physics after stop, and complete viewer/recording shutdown.

The controller summary records `end_reason=task_success` but its subsequent physical success sample is false because angular rest was not satisfied at that later sample. The original summary and decision annotation preserve this observation. Insertion quality is outside the requested acceptance scope, and this validation does not claim a clean insertion. RoboBrain's SUCCESS is evaluated under the user's intended reset-after-success workflow; indefinite success stability is not required. No automatic success-triggered reset was added.

## Other verification

- **105 simulator tests and 184 parameterized checks passed.**
- Frontend, OpenAI adapter, monitor audit, UI, Builder, endpoints, progress, simulation UI/monitor/client/policy, and service checks passed.
- Real HTTP/WebSocket plus MuJoCo recording check passed: two coherent movable views and fixed global feed, original/monitor video exports, camera revision change, stop/resume in one run, and reset into a second run without mixed-scene frames.
- Recorder checks passed for synchronized encoding, native individual-view dimensions, bottom-right overlays, unavailable/gap annotation, overwrite prevention, and stopping before a new episode enters the old video.
- Trajectory tests passed for cancellation followed immediately by reset: the interrupted control retains its old execution-boundary pose.
- Provider tests passed for unchanged audited request payloads, progress independence, prompt/image/response capture, redaction, HTTP failures, and cancellation pairing.
- The original `/home/yuxin/workspace/Show-Harness` checkout remained read-only: **175 source hashes matched** and its Git status was clean.

The temporary live-validation services were shut down after finalizing the recordings. Code, tests, documentation, caches, and artifacts were written only inside the RoboBrain workspace.
