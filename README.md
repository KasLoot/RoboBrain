# RoboBrain

## Attempt-scoped terminal monitoring

Each robot attempt is monitored by focused terminal specialists:

- two inserted/seated reference-bank votes;
- one loose-on-table drop specialist;
- two tight-view rotation verification votes.

The five compact one-image requests run concurrently while an attempt is active.
The deterministic attempt manager emits exactly one terminal `COMPLETE` or
`FAIL`, then stops terminal inference. During simulated continuous-video replay,
the state/reset specialist provides a one-observation reset fallback and starts a
fresh attempt after the robot leaves that scene. Real deployment should instead
create the fresh manager from the external reset-complete handshake.

Terminal rules:

- inserted requires both inserted banks to vote positive on the same frame;
- drop requires two consecutive loose-on-table positives;
- rotation requires both tight-view verifiers and the configured source-frame
  duration of continuous evidence; it is logged in shadow mode by default;
- contradictory inserted and drop votes abstain and continue monitoring.

Rotation terminal events remain disabled until an independent positive failure
attempt is available. Set `ROBOBRAIN_ENABLE_ROTATION_TERMINAL=1` only for
explicit experimental runs; shadow evidence remains present in `latency.jsonl`.

Specialist demonstrations are packaged as lossless crops under
`src/robobrain/prompt/reference_frames`. They are independent of
`VIDEO_INPUT_PATH`; changing the monitored video changes only the live
`CURRENT` panel. To intentionally rebuild those assets from the canonical
videos, run:

```powershell
uv run python test/vlm_latency_experiments/materialize_production_reference_assets.py
```

Start the SSH port forward, then run the application as usual:

```powershell
ssh -N -L 8000:127.0.0.1:8000 -p 25521 -i ~/.ssh/id_ed25519 root@82.221.170.234

# Enable Rotation detection
$env:ROBOBRAIN_ENABLE_ROTATION_TERMINAL = "1"

uv run robobrain
```

Each sampled timestamp writes one JSON object to `latency.jsonl`. The file is
also copied into the timestamped `runs/` archive at shutdown.
`parallel_requests_ms` and `fast_parallel_requests_ms` measure the active phase
for that frame. `requests` contains per-specialist timing and `attempt` contains
the attempt ID, lifecycle phase, terminal result, and temporal evidence. Every
completed request has these main values:

- `round_trip_ms`: client-observed duration of the complete call.
- `generation_latency_ms`: remote vLLM queue + prefill/first-token + decode.
- `network_latency_estimate_ms`: round trip minus generation latency.

The vLLM sub-phases are retained as `queue_ms`,
`time_to_first_token_ms`, and `decode_ms`. In formula form:

```text
generation_latency_ms = queue_ms + time_to_first_token_ms + decode_ms
network_latency_estimate_ms = round_trip_ms - generation_latency_ms
```

`network_latency_estimate_ms` is a residual, not a pure wire RTT. It includes
SSH and HTTP transfer, SDK serialization/parsing, and remote work outside the
vLLM engine timer (for example some request or image preprocessing). This is
the finest per-request split available from the server's existing timing
response without adding HTTP-boundary instrumentation to the remote server.

Rotation confirmation is measured in source-video frames, not request count, so
faster inference does not silently weaken the threshold.
