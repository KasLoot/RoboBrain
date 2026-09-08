# RoboBrain experiment 12

Results: **197/198 correct episode outcomes**. Development traj-93 is counted as a system error and documented as a gripper-occlusion edge case. See [RESULTS.md](RESULTS.md) and the [collected reports](../../results/rotation-specialist-v12/README.md).

Standalone successor to experiment 11. The drop prompt classifies actual support: the adapter must stand on the block's top face to be STAND. An adapter lying beside/behind the block is TABLE, including partial occlusion; a visible empty top face is a useful distinguishing cue. Gripper/socket support still takes priority. This addresses the missed no-force traj-4 drop without discarding the stand reference that fixed August 27 traj-36.

The shared monitor, success/rotation prompts, reference images, native crops and thresholds are unchanged. FALLING abstains, while TABLE is positive drop evidence. This still requires positive evidence in the visible landing region.

```sh
uv run --script test/robobrain-v2-test-12/check.py
uv run --script test/robobrain-v2-test-12/run.py \
  --videos video/insert_plug-full_model-aug-28-105trail-1x \
  --episodes 1 9 82 \
  --output test/robobrain-v2-test-12/runs/my-run
```

Omit `--episodes` for a complete dataset. `--specialists success drop` selects the control. `--base-url` overrides the configured vLLM endpoint at `http://127.0.0.1:9000/v1`. Each output directory must be new. All runtime code, prompts and reference images are inside this folder; no other experiment or production module is imported.

`Specialist(system_prompt, config)` handles its schema, observation-to-vote mapping, native crop, references, parsing and strict evidence threshold. The shared monitor schedules one verification round at a time: ten slots over one second, all three specialists at each slot, and all 30 responses awaited. More than five positives confirms a specialist. The next normal batch supplies slot zero; the trigger itself does not vote. Normal monitoring continues while verification is pending. Conflicting success/failure confirmations leave ON_GOING; confirmed terminal results are latched per attempt.

The camera publishes at 30 fps and continues publishing its final frame after EOF. A 90-second experiment watchdog leaves unresolved attempts ON_GOING. Thinking is disabled. The application never resizes its crops.

Recovery/reset handoffs are simulated log events only. No real robot-control API is called. This detector has now been [integrated into production](../../results/production-v12/README.md); this experiment remains standalone, with its original evaluation artifacts retained.

`probe.py` checks development images. `regression_probe.py` tests the two support errors using 30 requests per frozen image; it is an image-classification check, not episode playback. `audit.py RUN --videos DATASET` audits scheduling and generates terminal evidence crops. Reviewed timing files are dataset-specific: `event_times.json`, `event_times_aug27.json`, and `event_times_no_force.json`.

Both additional recordings have now informed support-prompt corrections, so subsequent scores are regression results, not untouched held-out estimates. Original episode annotations remain unchanged. All 198 clips count in the evaluation, including Dataset 1 traj-93: the system reported ROTATION_FAILURE for a SUCCESS clip while the gripper obscured the plug. This is counted as a false failure, not excused or relabeled.

To repeat the complete benchmark:

```sh
uv run --script test/robobrain-v2-test-12/evaluate_all.py --dry-run
uv run --script test/robobrain-v2-test-12/evaluate_all.py \
  --output-root test/robobrain-v2-test-12/runs/repeat
uv run --script test/robobrain-v2-test-12/report_results.py \
  --runs test/robobrain-v2-test-12/runs/repeat
```

The benchmark helper runs all 105, 45 and 48 clips. `report_results.py --self-check` validates scoring, including rejection of a missing traj-93; running it without arguments regenerates the saved scored reports. Raw summaries and logs remain intact. The evaluator used for the measured run and its hash manifest are archived in `runs/evaluate_all_at_run.txt` and `runs/candidate-sha256-at-run.json`; only evaluation/reporting helpers changed after that run.
