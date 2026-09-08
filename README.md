# RoboBrain

RoboBrain monitors plug insertion using separate SUCCESS and DROP specialists, followed by a bounded verification round.

<!-- GitHub removes custom video elements. Linked GIFs keep both demos visible in the README. -->

| Failure case — traj-30: DROP_FAILURE | Success case — traj-31: SUCCESS |
|:---:|:---:|
| [<img src="asset/traj-30-failure.gif" alt="Animated demo of RoboBrain detecting a dropped plug" width="400">](asset/traj-30-failure.mp4) | [<img src="asset/traj-31-success.gif" alt="Animated demo of RoboBrain confirming successful plug insertion" width="400">](asset/traj-31-success.mp4) |

Click either animated preview to open the full MP4 video with audio.

## Run

From this repository:

```sh
uv sync
uv run robobrain --record --canvas
```

Or activate the environment created by `uv`:

```sh
source .venv/bin/activate
robobrain --record --canvas
```

The default input is `video/insert_plug-full_model-aug-28-105trail-1x`. Episodes run in numeric order. The default model endpoint is `http://127.0.0.1:9000/v1`, using `/workspace/models/gemma-4-26B-A4B-it`. Keep the vLLM server and its SSH forward running. The recommended server has a 1120-token visual budget.

Recording requires `ffmpeg` on PATH. The live canvas requires a desktop and GUI-enabled OpenCV; omit `--canvas` for headless operation. FFmpeg and a working OpenCV GUI are available on the development machine.

Select another dataset, particular episodes, or a single video:

```sh
uv run robobrain --videos video/insert_plug-no_force-aug-28-1x --record --canvas
uv run robobrain --episodes 82 94 --record --canvas
uv run robobrain --videos video/insert_plug-no_force-aug-28-1x/traj-32.MOV --record
```

The two flags are independent. `--record` saves files, `--canvas` shows the live feed, and omitting both prints each episode's final status. Press **Q**, **Esc**, or close the canvas to stop the run. **Ctrl+C** also stops it. The current episode's logs and a playable recording are finalized; an interrupted recording can be shorter than the full source.

Existing episode output folders are protected. Use `--overwrite` to replace selected recordings, or `--output another-directory` to choose a different output root. `uv run robobrain --help` lists all options.

## Recordings

Each completed recording creates exactly these three files:

```text
output/
  dataset_name/
    traj-x/
      output-traj-x.MOV
      output-traj-x.log
      output-traj-x.json
```

- **MOV:** the full source video at its original resolution, encoded as H.264 at 30 fps, with its audio retained. When a decision arrives after the source ends, the final view continues so the published decision is visible for at least one second. Status changes appear at their actual publication times. Encoding happens after inference for that episode.
- **LOG:** a streaming JSON-lines event log containing every request start, every returned output, errors, cancellations, verification rounds and status updates. Full endpoint response bodies are retained; image payloads are not duplicated into the log.
- **JSON:** an indented report with the final result, configuration, requests sorted by ID, parsed model outputs, raw responses, timestamps and verification events.

The canvas and recording use the same overlay. The current source frame appears in the top-right corner. The bottom-right panel has exactly three lines:

```text
Operating Status (frame x):
Status: ON_GOING | SUCCESS | FAIL
Failure type: NONE | DROP_FAILURE
```

The entire heading is blue; only “Operating Status” is bold. Frame numbers in the overlay and readable JSON are **one-based source-frame numbers**. The heading identifies the most recently completed normal observation or the evidence used for a verified decision. Before the first completed batch it shows frame 0. The raw LOG keeps zero-based source indices. The current source-frame number stays on the last source frame after EOF.

## Monitoring behavior

The camera publishes to its buffer at 30 fps independently of model latency, including after EOF. Both specialists receive separate simultaneous requests. Normal pairs run as soon as the preceding pair finishes. Thinking is disabled and temperature is zero.

A positive observation starts one verification round with ten paired observations over one second. The next normal pair is reused as observation zero; the original triggering observation is excluded. Both specialists participate. All twenty request outcomes finish before the round resolves. Each specialist needs **more than five positive votes**; conflicting or insufficient evidence keeps ON_GOING. Normal monitoring continues during verification, and additional rounds cannot overlap or queue up.

SUCCESS requires a seated plug after gripper release. DROP_FAILURE requires the plug to be on the table or falling, rather than on the stand, in the gripper or in the socket. Rotation-only failure detection is outside this version's scope. Terminal decisions remain latched until the next episode.

The default episode watchdog is 90 seconds. An unresolved episode stays ON_GOING, never an inferred failure. Override it with `--episode-timeout SECONDS`. Request errors and cancellations are UNCERTAIN and do not supply positive evidence; five consecutive failed normal pairs stop the run with an endpoint error.

Default settings are in `src/robobrain/config.json`. Use `--config custom-config.json` or `--base-url URL` to override them. The SUCCESS crop is 470 × 400 pixels and the DROP crop is 1290 × 760 pixels; application-side model images are never resized or annotated. Dataset annotations are not required and are not used by the production monitor.

## Checks

```sh
uv run python tests/check_robobrain.py
```

These checks exercise verification thresholds and conflicts, concurrency and rearming, disabled thinking, native image dimensions, malformed responses, cancellation, 30 fps continuation, overlay formatting, complete source preservation, late status timing, audio retention, and raw/readable logs. Recording checks require `ffmpeg` and `ffprobe`.

## Note

This is the sixth iteration, from `test/robobrain-v2-test-6`
