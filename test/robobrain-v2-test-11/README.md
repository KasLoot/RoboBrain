# RoboBrain experiment 11

Standalone successor to experiment 10. The drop specialist receives an additional native reference of the adapter on its stand, taken from Dataset 1, traj-1 frame 0. Its prompt explicitly distinguishes the supporting white block from the adapter even when partly occluded. This addresses a false TABLE classification discovered in the August 27 evaluation. That recording is consequently a regression dataset for this iteration, not an untouched holdout.

The monitor, success and rotation prompts, existing reference images, crops and voting thresholds are unchanged. FALLING still abstains; TABLE is the positive drop observation. The failure mode where a drop leaves the visible landing region remains outside that conservative evidence rule.

```sh
uv run --script test/robobrain-v2-test-11/check.py
uv run --script test/robobrain-v2-test-11/run.py \
  --videos video/insert_plug-full_model-aug-28-105trail-1x \
  --episodes 1 9 82 93 \
  --output test/robobrain-v2-test-11/runs/my-run
```

Omit `--episodes` for the full dataset. `--specialists success drop` selects the two-specialist control. The vLLM endpoint is configured as `http://127.0.0.1:9000/v1`; `--base-url` overrides it. Output folders must be new. All code, prompts and images required by this experiment live in this folder.

`Specialist(system_prompt, config)` provides the schema, raw-observation-to-vote mapping, native image crop, references, strict parser and evidence threshold. One shared monitor schedules future verification and arbitrates outcomes. The default round collects ten slots over one second for each of the three specialists, waits for all 30 responses, and requires strictly more than five positives per specialist. The next normal batch is reused as slot zero; the original trigger does not vote. Normal monitoring continues, but verification rounds cannot overlap or queue. Conflicting success/failure confirmations remain ON_GOING.

The camera publishes at 30 fps, continuing with its final frame after EOF. A 90-second benchmark watchdog leaves unresolved attempts ON_GOING. Thinking is disabled and the application never resizes its native crops.

Confirmed decisions are latched per attempt. Controller handoffs are simulated log events only: stop insertion, release/recover on failure, then reset. A new recorded episode represents the next ready scene. No real robot API, recovery acknowledgment or physical reset is exercised. Production code has not been changed.

Use `audit.py RUN --videos DATASET` for scheduling/timing checks and terminal evidence crops. Optional reviewed timing files are dataset-specific: `event_times.json` for Dataset 1, `event_times_aug27.json` for August 27, and `event_times_no_force.json` for no-force. All episode-label scores must retain the original annotations; the rotation/recovery case in Dataset 1 traj-93 requires separate policy interpretation.
