# Combined-system recordings and run boundaries

RoboBrain and Show-Harness MuJoCo write their outputs into one shared directory for each simulation run. They remain independent processes with separate model clients and recording workers. This document describes the current recording contract; the new [recording validation](recording-validation.md) reports measured results. Historical test results remain in the [implementation validation](combined-system-validation.md) and [live insert-plug report](live-insert-plug-validation.md).

## Choose the shared output location

The simulator defaults to `<working directory>/.robobrain/runs`; launching from the RoboBrain workspace gives the standard shared location. For the default RoboBrain UI data directory, start both applications from the repository as described in [the runbook](combined-system-runbook.md).

If RoboBrain uses a different data directory, explicitly give the simulator that directory's `runs` subdirectory. For example, these commands keep all output inside the current workspace:

```bash
# Terminal 1; run from the RoboBrain repository.
OPENAI_MODEL=gpt-6-sol uv run --project simulator --python 3.13 show-harness-mujoco \
  --gui --record --record-global \
  --vlm-backend openai \
  --robot-config configs/robot_mujoco_plug.yaml \
  --run-root /home/yuxin/workspace/RoboBrain/.robobrain/runs

# Terminal 2.
uv run --python 3.13 robobrain ui \
  --data /home/yuxin/workspace/RoboBrain/.robobrain
```

Supply model credentials through each process's environment. The MuJoCo plug task preset selects OpenAI `gpt-6-luna` for RoboBrain outcome and progress monitoring. The simulator reports its current run directory over the API. Its optional native GUI uses a separate viewer process; `gui_error` is independent of `recording_error`. RoboBrain checks that this directory lies within its configured data directory's `runs` tree before writing into it.

**A shared filesystem is required for this one-folder layout.** Camera and command transport still uses the network API. Merely connecting to a remote simulator does not mount or copy its recording directory; both processes must see the configured shared path.

## What creates a new run

| Event | Recording behavior |
| --- | --- |
| Simulator initializes its scene | Creates a timestamp-and-UUID run directory and its manifest. |
| Start robot | Uses the current run and records the initial measured robot pose before its first control. |
| Stop or resume robot | Keeps the same run and attempt; events distinguish execution generations. Physics and enabled simulator video continue. |
| Stop monitoring | Finalizes that RoboBrain monitoring session. It does not stop robot execution or simulator recording. |
| Start monitoring again without scene reset | Keeps the same run, creates separate monitoring logs/audits, and preserves existing videos in a separate session directory. |
| Reset scene | Closes the previous run's recording segment and creates a new run for the reset scene. RoboBrain ends the previous monitoring/video session when it observes the episode change. |
| Simulator restarts | Creates a new service identity, scene, and run. |

After reset, explicitly start the next monitoring session and robot attempt. A confirmed SUCCESS does **not** automatically reset the scene or start another run. Success-triggered reset is a future production workflow, not an implemented behavior.

The simulator's `--record` flag enables its video recording. `--record-global` remains accepted and enables recording too. Enabling the RoboBrain Record toggle for a MuJoCo monitoring session also enables the simulator's video recording and records RoboBrain's videos. Stopping monitoring does not turn off the simulator's enabled recording.

Simulator trajectory and model-call audits are written even when video is disabled. RoboBrain also writes its monitoring events, reports, and model-call audits for MuJoCo sessions with the Record toggle off.

## Directory layout

Files are created as the corresponding work occurs. A run without model calls or video recording will not contain every file shown here.

```text
runs/<timestamp>-<uuid>/
├── run.json
├── lifecycle.jsonl
├── physics-events.jsonl
├── robot-trajectory.jsonl
├── robobrain-sessions.jsonl
├── show-harness-video/
│   ├── side.mp4
│   ├── wrist.mp4
│   ├── front.mp4
│   ├── wrist_insert.mp4
│   ├── wrist_depth.mp4
│   ├── combined.mp4
│   ├── global.mp4
│   ├── frames.jsonl
│   └── annotations.jsonl
├── show-harness-logs/
│   └── <attempt>/generation_<n>/.../
│       ├── metadata.json
│       ├── steps.jsonl
│       └── summary.json
├── show-harness-vlm/
│   ├── vlm.jsonl
│   ├── assets/
│   └── vlm/<call-id>/
├── robobrain-video/
│   ├── view-camera1.mp4
│   ├── view-camera2.mp4
│   ├── global.mp4
│   ├── combined.mp4
│   ├── frames.jsonl
│   └── sessions/<monitor-session-id>/...
├── robobrain-logs/<monitor-session-id>/
│   ├── events.jsonl
│   └── output-<monitor-session-id>.log / .json
└── robobrain-vlm/<monitor-session-id>/
    ├── calls.jsonl
    └── assets/
```

`run.json` identifies the scene, run, and output locations. `robobrain-sessions.jsonl` maps monitoring session IDs and tasks to their logs. The first RoboBrain video session uses `robobrain-video/`; later sessions use `robobrain-video/sessions/<monitor-session-id>/` so restarting monitoring cannot overwrite the earlier recording. Individual monitor filenames follow their configured logical source IDs; a one-source task produces one individual view video.

The original standalone `scripts/run_mujoco.py` baseline retains its own episode logging and output conventions. The shared layout above applies to the continuous simulation service connected to RoboBrain.

## Videos and camera identity

Show-Harness writes the original controller-camera videos and uses the original combined-canvas layout. For the plug scene, this includes the side, wrist, front, angled-wrist, and grayscale wrist-depth views, along with the controller decision panel and square global overview. The combined overview is rendered with its original square projection; it is not a stretched Full HD image. The separate `global.mp4` is the fixed `lab_overview` camera at **1920×1080**.

RoboBrain writes its individual requested camera views, a **1920×1080 global video with task-status/progress annotations**, and a combined video with monitor views on the left and the annotated global view on the right. The individual monitor views retain their native resolution, subject to encoder-required even dimensions. Retained images during input gaps are visibly marked unavailable.

RoboBrain's global image uses the same fixed `lab_overview` camera and projection as the simulator's separate global video. It arrives in the same simulation-state bundle as RoboBrain's free-camera views. Changing a monitor viewpoint does not change the fixed global camera or the controller's calibrated cameras.

Within each simulator capture, all individual views, the combined canvas, and the global view derive from one pinned physics snapshot. Within each RoboBrain encoded sample, its individual, global, and combined outputs derive from one received bundle. **The two applications sample independently:** matching video frame numbers between their recordings does not imply matching simulation states. Use their frame metadata to correlate them.

## Measured robot trajectory

`robot-trajectory.jsonl` records the initial robot pose and a start/end pair around each high-level control. Records include control IDs, action tokens, stage/step information, and the measured robot state. Completion, interruption, and failure remain distinguishable; commanded targets are not substituted for measured movement.

| Pose field | Interpretation |
| --- | --- |
| `tcp_pose` | `[x, y, z, qx, qy, qz, qw]`; world position in metres, orientation as an XYZW quaternion. |
| `arm_qpos_rad` | Measured `joint1` through `joint7` positions in radians. |
| `gripper_position_m` | Measured gripper opening in metres. |
| `actuator_ctrl` | Native actuator control values; these are not all positions or SI quantities. |
| Scene/generation, `state_seq`, simulation and capture times | Identify when and where the measured pose was sampled. |

When a control is interrupted, its boundary pose is retained even if the old execution worker finishes cleanup after a scene reset. This prevents a reset pose from being recorded as that control's final position. Stop retains the sampled gripper control and holds the measured arm position while physics continues.

`physics-events.jsonl` supplements this trajectory with applied transitions and low-level ramp/settle timing, target controls, and partial execution. One high-level action can contain several such intervals. These events are written by a separate worker; disk writes do not run on the physics thread.

## Model requests and responses

`show-harness-vlm/vlm.jsonl` records planner/controller model calls and their transport requests, responses, errors, and cancellations. Entries carry call IDs, role/stage context, model, attempt, execution generation, and observation identity where available. A response that arrives after stop remains in the audit trail, but its obsolete generation prevents it from controlling the robot.

`robobrain-vlm/<monitor-session-id>/calls.jsonl` records the monitoring model interactions, including outcome specialists, verification, and independent progress requests. Request image payloads are saved as referenced assets rather than repeated base64 text. Keep the asset directories with the JSONL files when copying a run. Configured credentials are redacted from these audit files.

The task-authoring workflow is separate from an individual monitoring run; this layout does not imply that Task Builder conversations appear in a run's monitoring audit.

## Timing, gaps, and finalization

The recordings use a presentation frame rate: RoboBrain encodes at 30 fps, while Show-Harness uses the configured recording rate, normally 30 fps. Their frame metadata preserves actual sampling times and native simulation identities.

Encoding uses bounded/latest-sample handling. Under load, intermediate samples can be omitted while physics continues. **A video can therefore be shorter than the elapsed wall-clock run.** It must not be used as the sole clock for action timing, model latency, or playback synchronization.

- Show-Harness `frames.jsonl` maps encoded frames to `state_seq`, simulation time, capture time, execution generation, and omitted intervals; `annotations.jsonl` supplies controller context.
- RoboBrain `frames.jsonl` maps encoded frames to wall elapsed time, sample slots, simulation/source identities, status/progress, repeated snapshots, unavailable intervals, and skipped samples.
- Read status fields such as recording errors and gap counters when diagnosing missing or slow output. Compare native identities and times, not just MP4 frame numbers.

Stop monitoring and allow finalization before treating RoboBrain's final video filenames and reports as complete. Scene reset rotates simulator recording; service shutdown flushes its remaining output. Interrupted or incomplete recording files should be inspected alongside their error and metadata logs.

## Acceptance scope

The user accepts post-success status variation for a workflow that ends the completed episode at a reset boundary. Stable indefinite display on an unchanged successful scene is not a requirement of this recording work. Automatic reset on success still requires separate implementation.

Optimizing insertion quality is also outside this task's scope. The earlier live report retains the observed recovery action, physical-success measurements, and outcome/progress variation as historical facts. They do not change the recording requirements: complete run organization, measured trajectory, model-call auditability, camera identity, and honest timing metadata. The [recording validation](recording-validation.md) documents the completed live trial and revised recording checks.
