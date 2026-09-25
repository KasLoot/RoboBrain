# Running RoboBrain with Show-Harness MuJoCo

The integration lives entirely in this workspace. `simulator/` is an independent, installable MuJoCo-only application extracted from Show-Harness `8c89d3b`; the original `/home/yuxin/workspace/Show-Harness` is not a runtime dependency and was not edited. RoboBrain and the simulator use separate processes and can use separate Python environments.

## Start the applications

From the RoboBrain repository, start the simulator in one terminal:

```bash
# OPENAI_API_KEY must already be set in this terminal.
export OPENAI_MODEL=gpt-6-sol
uv run --project simulator --python 3.13 show-harness-mujoco \
  --gui --record --record-global \
  --vlm-backend openai \
  --robot-config configs/robot_mujoco_plug.yaml
```

It listens on `http://127.0.0.1:8766`, with a ready scene and robot execution idle. It does not make controller VLM requests until **Start robot**. Credentials from the original Show-Harness checkout were not copied; supply them in the simulator environment. Existing model overrides such as `OPENAI_MODEL` remain supported by the extracted configuration loader.

Start RoboBrain in a second terminal:

```bash
uv run --python 3.13 robobrain ui
```

Open `http://127.0.0.1:8765`. On Tasks, choose **MuJoCo plug task** for the editable starter, or **Create with AI** for another task. The original camera/video tasks and authoring workflow remain available.

RoboBrain's monitoring endpoint and Task Builder endpoint are configured separately from the simulator's controller. The MuJoCo starter now selects the official OpenAI endpoint and `gpt-6-luna` with low reasoning; the UI process needs `OPENAI_API_KEY`. Local model-server tasks remain supported. The integration does not launch a monitoring model server. Set the task's **Model & settings** to your working endpoint and model, as for existing RoboBrain tasks.

## Supervised workflow

1. Open the task's Monitor page. Select **MuJoCo simulation** and the service address.
2. **Connect inputs**. Check both views and adjust their world target, distance, azimuth and elevation. **Apply live view** affects the runtime camera; **Save pose preset**, then **Save task**, retains it for future connections.
3. Review the outcome specialists and Progress criteria. Camera/source assignments and full-frame crops use the existing Prompt Designer.
4. Optionally enable recording, then **Start monitoring**. Both outcome and progress workers consume the same coherent camera bundles at their independent sampling rates.
5. **Start robot** begins VLM-driven manipulation in Show-Harness.
6. **Stop robot** cancels the current trajectory, holds the arm's measured joint positions, preserves the gripper actuator control value at interruption, and blocks further task actions. Physics, contacts, gravity, cameras and monitoring continue.
7. **Resume robot** uses fresh observations and replanning in the same attempt. It retains the plan cursor and measured stage-travel anchor, and does not repeat the original runner's initial release. An outstanding provider request must return or time out before the single execution worker can begin the new generation; the old result cannot execute.
8. **Stop monitoring** ends inference and finalizes RoboBrain outputs without changing robot execution. **Reset scene** is separate: it returns a stopped/finished scene to its initial state, creates a new shared run folder, and ends the previous RoboBrain monitoring/video session. Start the next monitoring session and robot attempt explicitly.

A camera pose can change while monitoring. Saved task configuration, dimensions, crops, and source assignments remain locked. New camera revisions invalidate pending outcome evidence and progress confirmation/history. Temporary stream gaps are marked unavailable; recordings visibly label any retained last frame.

Only two simultaneous monitor views are supported initially. Both are runtime free cameras; neither needs to be predefined in XML. The controller's calibrated wrist/front/side cameras are separate and remain unchanged by monitor-camera controls.

## Task Builder and visual criteria

In **Create with AI**, choose MuJoCo inputs, connect the virtual cameras, and capture scene images. Captured images become explicit saved assets with view/episode provenance. Select which images can be used as specialist references, then generate and review the draft. The builder can generate both outcome specialists and independent progress nodes.

The supplied plug starter includes labeled initial/completed simulator references and describes the orange handle, silver cylindrical pin, blue socket and gray stand. Its success criterion includes seating, release, and more than 50 mm of retreat clearance. The 117 mm completed reference is an example, not the minimum. [Live validation](live-insert-plug-validation.md) observed post-success outcome/progress variation; the user accepts post-success variation for a workflow ending at reset. Automatic success-triggered reset is not implemented. Slip detection is supplied disabled for review. Progress stages carry no invented completion percentage. Reaching RETREATED does not itself declare SUCCESS.

**The starter is not an empirically calibrated detector.** Review camera placement, visible scale, reference examples and model performance before enabling automatic robot stopping. Automatic stop defaults to **Observe only**. Optional policies stop on a current FAIL or on SUCCESS/FAIL, using the configured verification mode. They never automatically resume, release the gripper, reset the scene, or start recovery.

## Offline plumbing check

For physics, rendering and command testing without a controller API call:

```bash
MUJOCO_GL=egl uv run --project simulator --python 3.13 show-harness-mujoco \
  --no-controller --record --record-global
```

Use this headless command on Linux with a working EGL setup. `--no-controller` allows start/stop/resume/reset to exercise the lifecycle while the arm holds position; it is not a task-solving controller. Keep RoboBrain monitoring stopped unless its model endpoint is available or deliberately replaced with a local test endpoint.

The original decision-synchronous CLI is also retained:

```bash
cd simulator
uv run --python 3.13 python scripts/run_mujoco.py \
  --gui --record --record-global \
  --vlm-backend openai \
  --robot-config configs/robot_mujoco_plug.yaml
```

Add `--smoke-test` to this baseline command for scripted camera/motion checks without controller inference. The service's continuous physics is a material change from the baseline: objects can settle or slip while a VLM is reasoning.

## Outputs and operational details

The [recording guide](combined-system-recording.md) describes the directory tree, video layouts, measured trajectory fields, model audits, and timing metadata.

- Each initialized/reset scene has one timestamp-and-UUID folder under `.robobrain/runs/`. Robot stop/resume keeps that folder; reset creates the next. Configure simulator `--run-root <RoboBrain data directory>/runs` when using a nondefault UI `--data` directory. Both processes must share this filesystem path.
- `show-harness-video/` contains the original individual camera videos, combined controller canvas, and fixed 1920×1080 global video. `robobrain-video/` contains individual monitor views, the global video with status/progress, and the combined monitor canvas. Later monitoring sessions in the same run use `robobrain-video/sessions/<session-id>/`.
- `robot-trajectory.jsonl` contains the initial and after-control measured TCP/joint/gripper poses, including interrupted-control boundaries. Both applications keep model request/response audits, referenced images, logs, and reports under the same run, even when MuJoCo video recording is off.
- The global feed uses the same fixed camera projection in both applications. Each application's views are coherent within a sample; their recordings are independently sampled. Video frame counts are presentation time. Consult `frames.jsonl` for native simulation/wall times and encoder gaps; under overload, video duration can be shorter than elapsed run time while physics continues.
- A controller-session lease prevents multiple clients from dispatching commands. Lease expiry does not stop the independently running simulator. Commands are idempotent and reject obsolete scene/generation identities. Read-only observers can own separate monitor cameras.
- Camera stream loss does not automatically stop the robot. RoboBrain marks evidence unavailable and rejects stale automatic interventions. Reconnection never starts the robot.
- A failed controller request produces an execution error and hold while healthy physics continues. Reset is available after the hold has applied. A fatal physics error requires restarting the simulator.

See [the design](combined-system-design.md), [validation notes](combined-system-validation.md), and [simulator README](../simulator/README.md) for implementation boundaries and checks.
