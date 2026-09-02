# RoboBrain

## Supported input resolutions

RoboBrain selects an explicit crop calibration from the input video's detected
resolution before it initializes the VLM pipeline. Assets exist for these
configurations:

| Source resolution | Main crop | Main panel | Tight rotation crop | Tight panel |
| --- | --- | --- | --- | --- |
| `848x480` | `(350, 50) -> (650, 300)` | `300x250` | `(420, 90) -> (600, 260)` | `180x170` |
| `1920x1080` | `(792, 112) -> (1472, 675)` | `680x563` | `(951, 202) -> (1358, 585)` | `407x383` |

The 1080p coordinates map the calibrated 848p field of view back onto the
original video used to create the resized input. Any other resolution prints an
`Unsupported resolution` message and exits before sending a VLM request.
The current production safety policy is validated only for `1920x1080` and
fails fast on the experimental `848x480` profile.

The runtime never resizes the live `CURRENT` panel. It directly slices the
configured main or tight crop from each live frame and loads matching
pre-cropped references from
`src/robobrain/prompt/reference_frames/<resolution>`. Safety grids render three
small reference thumbnails above the native-size current panel. Startup fails
if a reference does not have the exact native size required by its profile.

## Attempt-scoped terminal monitoring

Each active attempt runs four primary requests:

- an inserted/seated trigger;
- a conservative loose-on-table drop trigger;
- two tight-view rotation verification votes.

The primary requests run concurrently. A positive insertion or drop trigger
launches its alternate verifier on the exact same retained source frame. The
deterministic attempt manager evaluates safety before completion, emits exactly
one terminal `COMPLETE` or `FAIL`, then stops terminal inference. During
simulated continuous-video replay, the state/reset specialist provides a
one-observation reset fallback and starts a fresh attempt after the robot leaves
that scene. Real deployment should instead create the fresh manager from the
external reset-complete handshake.

Terminal rules:

- inserted requires trigger A and verifier B on the same retained frame;
- drop requires trigger A and verifier B on the same retained frame;
- rotation requires three consecutive sampled observations in which both
  tight-view verifiers vote `R`;
- skipped source frames never count as unseen rotation evidence;
- confirmed drop or rotation has priority over completion.

Rotation terminal events are enabled by default. Set
`ROBOBRAIN_ENABLE_ROTATION_TERMINAL=0` only as an emergency kill switch; shadow
evidence remains present in `latency.jsonl` while disabled.

Specialist demonstrations are packaged as lossless native crops under the two
resolution-specific `src/robobrain/prompt/reference_frames` directories. They
are independent of `VIDEO_INPUT_PATH`; changing the monitored video changes
only the live `CURRENT` panel. To intentionally rebuild those assets from the
canonical videos and unannotated stills, run:

```powershell
uv run python scripts/materialize_resolution_reference_assets.py
```

Start the SSH port forward, then run the application as usual:

```powershell
ssh -N -L 8000:127.0.0.1:8000 -p 14315 -i ~/.ssh/id_ed25519 root@157.157.221.177

uv run robobrain
```

Each sampled timestamp writes one JSON object to `latency.jsonl`. The file is
also copied into the timestamped `runs/` archive at shutdown.
`parallel_requests_ms` measures the whole cycle, `primary_requests_ms` (also
available under the legacy `fast_parallel_requests_ms` key) measures the
concurrent primary batch, and `verification_requests_ms` measures triggered
same-frame verification. `requests` contains per-specialist timing and
`attempt` contains the attempt ID, lifecycle phase, terminal result, and
observation evidence. Every completed request has these main values:

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

Rotation confirmation is measured in actual consensus observations, not source
frame distance. Faster inference therefore increases the opportunity to detect
failure without allowing skipped, unobserved frames to satisfy the threshold.
