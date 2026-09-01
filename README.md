# RoboBrain

## Failure monitoring and latency

The latency-critical path now sends two compact one-image requests per sampled
timestamp:

- `state`: stand/reset, controlled, dropped, or uncertain;
- `rotation`: normal, rotated, or uncertain.

They run concurrently. The larger general `scene` request remains available for
completion detection, but runs asynchronously every 120 source frames and does
not block failure monitoring. After failure is latched, rotation requests stop
until the state specialist sees the plug back on its stand.

Start the SSH port forward, then run the application as usual:

```powershell
ssh -N -L 8000:127.0.0.1:8000 -p 13316 -i ~/.ssh/id_ed25519 root@216.243.220.242
uv run robobrain
```

Each completed fast-path timestamp writes one JSON object to `latency.jsonl`.
The file is also copied into the timestamped `runs/` archive at shutdown.
`parallel_requests_ms` and `fast_parallel_requests_ms` both measure the compact
failure path; `requests.scene_completed` is populated only on the cycle that an
asynchronous scene request finishes. Every completed request has these main
values:

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

Rotation confirmation is measured in source-video time, not request count. A
rotation must remain observed for 0.6 seconds before it is latched. This keeps
the threshold stable when faster inference causes more frames to be sampled.
