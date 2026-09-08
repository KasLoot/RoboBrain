# RoboBrain v2, experiment 6

This variant keeps experiment 3's runtime and drop specialist, and uses a 470 x 400 native success crop around the socket and gripping fingers. It removes the unintended requirement that the model call the occupied socket "rightmost": success depends on seating and release. The two success reference crops use the same bounds. No image is resized. No production Python modules or sibling experiment code are imported. The only external input is the episode dataset.

The updated 1120-token results match 105/105 episode annotations, including both drop failures, with no incorrect terminal labels or unresolved episodes. This combines the previous 104 episode results with a successful rerun of the user-extended episode 94; the unchanged original full run scored 104/105. See [RESULTS.md](RESULTS.md) for the measurements, provenance, and limitations.

Testing on two additional datasets produced 93/93 correct latest outcomes, including all eight drops, with no remaining ON_GOING clips. The no-force result includes a rerun of the extended and relabeled episode 32. See [NEW_DATASETS_RESULTS.md](NEW_DATASETS_RESULTS.md) for the visual review, input-version provenance, and runtime measurements.

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

`probe.py` tests 27 visually reviewed states, including grasping, insertion before release, rotation while still held, two drops, released successes, earlier false positives, close-release cases, and the socket-ordinal misclassification. Probe accuracy measures individual observations, not episode monitoring accuracy. Reference images come from this recording; the 105 episodes form a development benchmark, not an independent generalization test.

`audit.py` checks complete verification responses, absence of overlapping rounds, request latency, and achieved observation spacing. Pass `--event-times /path/to/event_times.json` to report event latency using timestamps reviewed for that specific dataset; omitting it skips event latency. This folder's `event_times.json` belongs only to the original 105-episode dataset. Episode names repeat across datasets, so those timestamps must not be reused for another recording. The audit can produce contact sheets of the actual source frames used for terminal evidence. Source images retain native crop dimensions; elapsed episode time is not substituted for event latency.

The implementation passed `check.py` after copying only this folder's source, configuration, prompts, and reference images to `/tmp` and running in a fresh `uv` script environment. Checks cover thresholds, contradictory evidence, per-slot uniqueness, native image dimensions, disabled thinking, malformed responses, EOF camera cadence, concurrent normal monitoring, complete-response waits, fresh-trigger rearming, timeout cleanup, and episode isolation through fresh monitor state.

Protocol reference: https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html
