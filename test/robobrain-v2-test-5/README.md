# RoboBrain v2, experiment 5

This variant keeps experiment 3's runtime and drop specialist, and tightens the success crop to 470 x 400 native pixels around the socket and gripping fingers. The two success reference crops use the same bounds. No image is resized. No production Python modules or sibling experiment code are imported. The only external input is the episode dataset.

From this directory:

```sh
uv run --script check.py
uv run --script probe.py --videos /absolute/path/to/episodes --output probe-results-new
uv run --script run.py --videos /absolute/path/to/episodes --base-url http://127.0.0.1:9000/v1 --output runs/new-run
uv run --script audit.py runs/new-run --videos /absolute/path/to/episodes
```

The scripts declare their own dependencies for `uv`; an existing environment containing OpenCV and httpx can also run them using `uv run --no-sync python run.py ...`. `--episodes 1 9 82 105` selects a pilot. Omitting it evaluates all annotated episodes. Output directories must be new.

The user-supplied servers use Gemma 4 26B A4B IT: port 8000 has a 560-token vision budget and port 9000 has 1120. Both requests explicitly disable thinking and use temperature zero. Application-side crops retain their native dimensions.

Normal requests run as fast as each preceding pair completes. A positive starts ten future paired observations over one second, with the next normal pair reused as observation zero. The triggering observation is excluded. Normal monitoring continues during verification, but cannot start or queue another round. All twenty requests finish before decision; timeouts/errors contribute no positive evidence. Each specialist needs strictly more than five positives. Both or neither confirming leaves ON_GOING. Only a fresh observation captured after the preceding round finished can retrigger verification. Terminal status is latched per episode.

The simulator keeps publishing the final decoded frame at 30 fps until monitoring stops. The benchmark watchdog stops an unresolved episode after 90 seconds; this remains ON_GOING, never an inferred failure. There are no padding metrics. Endpoint failure in five consecutive normal batches aborts the run instead of silently treating infrastructure failure as task failure.

`summary.json` is saved after each episode. Per-episode JSONL files contain request observations, timestamps, response latency, verification counts, and published statuses. Prompt and source snapshots accompany each run. Episode labels are used only by the evaluator after monitoring; they never enter a model request. `decision_time_s` is elapsed playback time, not detection latency: computing event-to-decision latency needs independently reviewed event timestamps.

`probe.py` tests 26 visually reviewed states, including grasping, insertion before release, rotation while still held, two drops, released successes, earlier false positives, and the close-release cases that were unresolved in earlier experiments. Probe accuracy measures individual observations, not episode monitoring accuracy. Reference images come from this recording; the 105 episodes form a development benchmark, not an independent generalization test.

`audit.py` checks complete verification responses, absence of overlapping rounds, request latency, achieved observation spacing, and the two visually reviewed drop-onset intervals in `event_times.json`. It can produce contact sheets of the actual source frames used for terminal evidence. Source images retain native crop dimensions. Reported event-to-decision latency is available only for the two reviewed drop events; elapsed episode time is not substituted for event latency.

The implementation passed `check.py` after copying only this folder's source, configuration, prompts, and reference images to `/tmp` and running in a fresh `uv` script environment. Checks cover thresholds, contradictory evidence, per-slot uniqueness, native image dimensions, disabled thinking, malformed responses, EOF camera cadence, concurrent normal monitoring, complete-response waits, fresh-trigger rearming, timeout cleanup, and episode isolation through fresh monitor state.

Protocol reference: https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html
