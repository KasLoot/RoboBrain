# Combined-system implementation validation

This page records the earlier implementation-stage smoke checks. See [the subsequent live insert-plug validation](live-insert-plug-validation.md) for real OpenAI runs and their remaining limitations.

Date: 2026-09-24. RoboBrain baseline `9f80dcf`; extracted Show-Harness baseline `8c89d3b`.

## Implemented scope

- Independent MuJoCo application under `simulator/`, with its own packaging and dependencies; the original Show-Harness checkout is not imported at runtime.
- Continuous physics with a single state owner, cancellable trajectories, measured arm hold, preserved gripper control, fresh resume planning, and generation checks at action application.
- HTTP lifecycle/control APIs, exclusive controller leases, idempotent command receipts, camera ownership/revisions, atomic snapshots and WebSocket camera bundles.
- RoboBrain simulation inputs, separate robot/monitor controls, live camera updates, Task Builder inputs and reference capture, a simulation plug starter, independent outcome/progress processing, and correlated recordings.
- Freshness/duplicate/context checks, bounded input buffering, temporary-camera-gap handling, and optional current-outcome automatic stopping.

The later workspace restriction changed the proposed sibling-project location to `simulator/` inside RoboBrain. At this implementation stage, service recordings spanned the service lifetime with reset/generation metadata. The current [recording layout](combined-system-recording.md) uses one shared folder per initialized/reset scene; the standalone baseline retains its original episode/combined recording behavior. The existing token-based replay utility remains a baseline-mode tool. Continuous action/control timing is logged separately; token-only replay does not reproduce continuous VLM wait intervals.

## Automated results

All of the following checks passed on a workspace-local Python 3.13 environment:

| Check | Coverage |
| --- | --- |
| `python -m pytest -q simulator/tests --disable-warnings` | **89 tests, 184 subtests**: original actions/depth/geometry/Cartesian integration/recording/replay plus continuous physics, gripper interruption, stale planner responses, resume serialization and coherent observations |
| `tests/check_simulator_service.py` | Controller leases, command idempotency, acknowledgement loss, pending/late callbacks, error/reset transitions, owned camera changes, same-snapshot bundles and event gaps |
| `tests/check_simulation_client.py` | Real TCP/WebSocket transport with a deterministic test runtime, two-view coherence, frame decoding, camera revisions, references, lifecycle commands, and commands after monitor detach |
| `tests/check_simulation_real.py --record` | Actual MuJoCo with two runtime cameras, real network client, start/stop/resume/reset, physics after stop, PNG capture, original camera recordings and decodable 1920×1080 global video |
| `tests/check_simulation_monitor.py` | One normal request per distinct snapshot, duplicate/stale abstention, cancellation on reset, recording through temporary camera gaps, and complete JSON reports |
| `tests/check_simulation_policy.py` | Synthetic current FAIL stops robot once while monitoring/progress continue; monitoring teardown preserves robot state |
| `tests/check_simulation_ui.py` | Input/schema bounds, authoring round-trip, reference provenance, narrow run-lock exemptions and separate robot/monitor commands |
| `tests/check_frontend.mjs` | Rendering, escaped text, task readiness, robot button states, live-camera versus saved-configuration locks, and independent progress display |
| Existing `check_ui`, `check_builder`, `check_endpoints`, `check_progress`, `check_robobrain` | Existing authoring, backend compatibility, status/verification, progress and recording behavior |

The simulator source distribution and wheel build successfully. A wheel extracted into an isolated workspace directory loaded its own packaged scene and Panda assets and started healthy physics; it did not depend on the original source checkout.

The inherited recording-layout test used a font-sensitive pixel probe. Its extracted copy now checks reserved panel padding while preserving the background-layout assertion. Original Show-Harness tests were not modified.

## Live browser workflow

The browser check used actual MuJoCo images and a local deterministic inference endpoint. It verified:

- Two independent camera previews and editable free-camera poses.
- Both outcome and progress displays during a monitoring run.
- Separate robot start/stop controls while monitoring remained active.
- Live camera changes while the saved task remained locked.
- Continued recording across a temporary camera-revision gap.
- Clean monitor stop/finalization with all MOV, LOG and JSON links present.

The final reviewed run produced 1,166 video frames, marked one unavailable interval, and recorded 1,132 synthetic model requests with matching response counts. The model replies explicitly identified themselves as synthetic UI-test evidence. They do not establish visual detector accuracy.

## Measurements and limits

A short headless real-MuJoCo transport smoke, with two 640×480 views requested at 15 fps and recording enabled at a test-configured 10 fps, observed command application on the order of 1–2 ms and initial capture-to-client age around 65 ms. This is a local smoke measurement, not a latency guarantee or a throughput benchmark with live VLM inference. Slow rendering/encoding is reported and intermediate frames may be dropped.

These implementation-stage checks made no paid or external VLM calls. Subsequent real-model results are documented in the [live insert-plug report](live-insert-plug-validation.md); they are separate from the smoke measurements here. The starter's success/retreat and failure criteria require visual calibration. The native MuJoCo GUI window was not exercised by the headless checks; the browser UI was exercised. A physical-camera check could not run because `/dev/video0` was unavailable; mocked local-camera regression checks passed.

The temporary browser-review services were stopped after verification. Runtime caches, environments, test outputs and recordings used for these checks were kept in ignored directories inside RoboBrain. All 175 copied upstream file hashes still match the original Show-Harness files, and the original Show-Harness Git working tree is clean.

## Reproduce

Use the project environments described in [the runbook](combined-system-runbook.md). The checks make no provider requests. For headless MuJoCo, use `MUJOCO_GL=egl` where supported. Set `TMPDIR`, `XDG_CACHE_HOME`, `MESA_SHADER_CACHE_DIR`, and any graphics-driver cache directory inside the workspace when reproducing the same write boundary.

The automated check commands are the paths in the table above. Run Python checks with Python 3.13, `PYTHONDONTWRITEBYTECODE=1`, and a workspace-local temporary directory. `check_simulation_real.py` writes its measurements and videos under `.tmp/simulation-real-*`. It deliberately uses `--no-controller` for lifecycle and transport validation.
