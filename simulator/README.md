# Show-Harness MuJoCo

This is an independent application extracted into RoboBrain's `simulator/` directory. It owns MuJoCo, robot planning, execution, rendering and recordings; RoboBrain connects over HTTP/WebSocket. It does not import RoboBrain. The original Show-Harness checkout is a read-only source and is not needed at runtime.

## Install and run

Use a separate environment from RoboBrain, from this directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
# Set OPENAI_API_KEY in this environment.
export OPENAI_MODEL=gpt-6-sol
.venv/bin/python -m show_harness_mujoco \
  --gui --record --record-global --vlm-backend openai \
  --robot-config configs/robot_mujoco_plug.yaml --run-root ../.robobrain/runs
```

The service binds `127.0.0.1:8766` and waits for an explicit robot start. Camera requests do not start robot execution or call a model. Headless Linux deployments can set `MUJOCO_GL=egl` before launch. Use `--no-controller` to exercise physics, arbitrary virtual cameras, recording and the control protocol without provider requests. The optional GUI runs in an isolated viewer process fed with coherent states, so its native graphics context cannot invalidate EGL recording contexts. Omit `--gui` for headless operation. `--port` is configurable.

In RoboBrain, select MuJoCo input, connect to the service, configure camera views, and start monitoring and robot execution with their separate buttons. Robot stop cancels trajectories, holds measured arm joints, preserves the sampled gripper actuator command, and leaves gravity, contacts, rendering and monitoring running. Reset restores the settled initial scene and does not start execution.

Shared per-run artifacts are described in the [recording guide](../docs/combined-system-recording.md). Match `--run-root` to RoboBrain's data directory plus `/runs`; stop/resume shares one folder, while scene reset rotates to a new folder. `gui_error` and `recording_error` are reported separately.

The original decision-synchronous baseline remains independently runnable:

```bash
.venv/bin/python scripts/run_mujoco.py \
  --gui --record --record-global --vlm-backend openai \
  --robot-config configs/robot_mujoco_plug.yaml
```

Add `--smoke-test` for scripted baseline camera/Cartesian/gripper checks without model requests. The baseline intentionally advances simulation only during actions. Continuous service mode changes physical behavior during cloud waits and requires supervised task-quality validation before relying on its results.

## Runtime and API

`core/sim/service_runtime.py` owns a single wall-clock-paced physics thread. Controllers use private coherent observation snapshots and submit generation-checked trajectories. A separate render thread produces atomic two-camera bundles from one copied state. Rendering, video encoding, VLM calls, GUI synchronization and IK never execute on the physics thread.

Start/stop/reset invalidate stale decisions at both the execution and physics boundaries. Resume retains the parsed plan, stage cursor, total/stage budgets, measured stage-entry TCP anchor, and attempt identity. It takes fresh observations, clears pending actions, and skips the original runner's initial release. Cumulative stage travel is reported without treating robot movement as proof of object movement. One persistent execution worker bounds provider concurrency to one. After a stop, a pending resume may wait for the previous HTTP request to return or time out (up to the configured timeout, capped at 120 seconds); its obsolete result cannot move the robot. Physics and stop acknowledgement continue during that wait.

The API exposes `/v1/health`, `/v1/status`, exclusive renewable controller claims, idempotent generation-checked commands, independently owned dynamic monitor cameras, snapshots, atomic WebSocket frame bundles, and lifecycle events. See [the combined design](../docs/combined-system-design.md) and `show_harness_mujoco/service.py` for the contract. The built-in service is for local loopback use; remote authentication/TLS is not supplied.

Runtime free cameras accept world-space look-at coordinates, distance, azimuth and elevation. They are separate from the controller's fixed calibrated cameras and viewer camera. Controller observations contain the original RGB/proprioception/depth contract. Ground-truth success diagnostics remain simulator logs rather than RoboBrain visual evidence.

## Recording and tests

Baseline recording preserves the upstream annotated recordings. Service recordings use a separate render/encode worker and capture actual continuous activity, including VLM waits and stopped-robot intervals. Files live under the configured `log_dir` in a `continuous_*` directory. `frames.jsonl` records native physics sequence, simulation time, capture clock, execution generation, and dropped intervals; a slow encoder drops intermediate snapshots rather than delaying physics. Recordings span the service lifetime; resets and control transitions are identified in frame metadata. `--record-global` adds the fixed 1920×1080 overview. Inspect `recording_error` and `recording_gaps` in status for encoding failures or overload.

```bash
MUJOCO_GL=egl .venv/bin/python -m pytest tests
```

Tests include real MuJoCo stop during a gripper ramp, held-control preservation, stale planner replies, resume serialization, continuous idle/stopped physics, coherent controller observations, Cartesian motion, scene/depth geometry, baseline recording and offline replay. No test needs a paid VLM request. Rendering throughput and successful VLM task completion must be measured on the deployment machine.

## Extraction and attribution

`UPSTREAM_MANIFEST.json` records the source revision and SHA-256 for every copied upstream file. The verified source is Show-Harness `8c89d3bdcf1c7d28bc5050621dabf7cdde31008e`. The extraction retains only the MuJoCo dependency closure, single-arm controller/planner infrastructure, relevant prompts/configurations, recording/replay utilities/tests, scene assets and approximately 33 MB of Panda meshes. Hardware transports, ROS cameras, other simulators and training code are excluded.

`core/launch.py` contains only simulator runner wiring. Hardware session typing and an image resize dependency were isolated from the shared runner. Files adapted from upstream carry change notices. `scripts/extract_source.py` documents the read-only extraction procedure; rerunning it refreshes copied baseline files and must be done with care after local edits.

Show-Harness is provided under the [Apache-2.0 license](LICENSE). The Panda model and meshes retain their [MuJoCo Menagerie license](third_party/mujoco_menagerie/franka_emika_panda/LICENSE). New integration modules are separate from the unchanged source checkout.
