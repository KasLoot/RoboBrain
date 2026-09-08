# RoboBrain v2, experiment 2

Two visual specialists with labeled reference images, native-resolution crops, a 30 fps simulated camera, and one globally exclusive verification round. No production Python modules or sibling experiment code are imported. The only external input is the episode dataset. See RESULTS.md for the measured outcomes and the reason experiment 3 was needed.

From this directory:

```sh
uv run --script check.py
uv run --script probe.py --videos /absolute/path/to/episodes --output probe-results-new
uv run --script run.py --videos /absolute/path/to/episodes --base-url http://127.0.0.1:9000/v1 --output runs/new-run
```

The scripts declare their own dependencies for `uv`; an existing environment containing OpenCV and httpx can also run them using `uv run --no-sync python run.py ...`. `--episodes 1 9 82 105` selects a pilot. Omitting it evaluates all annotated episodes. Output directories must be new.

The user-supplied servers use Gemma 4 26B A4B IT: port 8000 has a 560-token vision budget and port 9000 has 1120. Both requests explicitly disable thinking and use temperature zero. Application-side crops retain their native dimensions.

Normal requests run as fast as each preceding pair completes. A positive starts ten future paired observations over one second, with the next normal pair reused as observation zero. The triggering observation is excluded. Normal monitoring continues during verification, but cannot start or queue another round. All twenty requests finish before decision; timeouts/errors contribute no positive evidence. Each specialist needs strictly more than five positives. Both or neither confirming leaves ON_GOING. Only a fresh observation captured after the preceding round finished can retrigger verification. Terminal status is latched per episode.

The simulator keeps publishing the final decoded frame at 30 fps until monitoring stops. The benchmark watchdog stops an unresolved episode after 90 seconds; this remains ON_GOING, never an inferred failure. There are no padding metrics. Endpoint failure in five consecutive normal batches aborts the run instead of silently treating infrastructure failure as task failure.

`summary.json` is saved after each episode. Per-episode JSONL files contain request observations, timestamps, response latency, verification counts, and published statuses. Prompt and source snapshots accompany each run. Episode labels are used only by the evaluator after monitoring; they never enter a model request. `decision_time_s` is elapsed playback time, not detection latency: computing event-to-decision latency needs independently reviewed event timestamps.

`probe.py` tests fifteen manually selected visual states, including grasping, insertion before release, rotation while still held, two drops, and released successes. Probe accuracy measures individual observations, not episode monitoring accuracy.

Protocol reference: https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html
