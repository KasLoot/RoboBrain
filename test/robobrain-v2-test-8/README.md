# RoboBrain experiment 8

Standalone successor to experiment 7. Rotation inspection classifies the current grasp as `HELD_NORMAL`, `HELD_ROTATED`, `NOT_HELD`, `NOT_INSERTING`, or `UNKNOWN`. A third reference distinguishes a seated adapter below an open, nearby gripper. Success and drop observations retain experiment 6's prompts and images.

```sh
uv run --script test/robobrain-v2-test-8/run.py \
  --videos video/insert_plug-full_model-aug-28-105trail-1x \
  --episodes 1 2 9 11 15 44 58 82 94 105 \
  --output test/robobrain-v2-test-8/runs/my-run
```

The configured OpenAI-compatible vLLM endpoint is `http://127.0.0.1:9000/v1`. Output directories must be new. Omit `--episodes` to replay the complete annotation list. `--specialists success drop` disables rotation for a control run. No imports depend on production or another experiment.

`Specialist(system_prompt, config)` owns the output schema, raw-observation-to-vote mapping, native crop, labeled references, request parsing and strict evidence threshold. The shared monitor owns scheduling and arbitration. Native image crops are never resized by the application; thinking is disabled.

A positive normal observation starts one global future verification round: ten capture slots over one second, all three specialists at each slot. The next normal batch is reused as slot zero; the triggering batch does not vote. Normal monitoring continues while the round is pending. All 30 responses must settle before another round can start or a terminal result is published. More than five positives confirms a specialist. Concurrent success and failure confirmations leave `ON_GOING`; concurrent failure types produce one `FAIL`, with drop as the primary type and both types recorded.

Confirmed outcomes are latched per insertion attempt. `ReplayController` only logs simulated handoffs: stop insertion, release/recover after failure, then reset; the next recorded episode represents a new ready scene. It calls no robot API and does not claim to test physical recovery, reset acknowledgments, or cross-process API idempotency. Real integration needs the controller API contract and a reset-ready acknowledgment before insertion monitoring restarts.

The simulator publishes frames at 30 fps and continues publishing the final frame after EOF until the attempt ends. A 90-second benchmark watchdog leaves unresolved attempts `ON_GOING`.

Checks and log audit:

```sh
uv run --script test/robobrain-v2-test-8/check.py
uv run --script test/robobrain-v2-test-8/audit.py --self-check
uv run --script test/robobrain-v2-test-8/audit.py \
  test/robobrain-v2-test-8/runs/my-run \
  --event-times test/robobrain-v2-test-8/event_times.json \
  --videos video/insert_plug-full_model-aug-28-105trail-1x
```

The event sidecar is only for Dataset 1. Evaluation compares terminal labels with annotations and separately reviews failure type, event timing and visible terminal evidence. The annotations have no rotation labels. Results on these episodes are development measurements, not proof of rotation generalization.
