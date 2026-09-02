# RoboBrain failure-detection experiments

This directory contains offline and replay experiments for the missed failure
around source frames 5493--6060. Production files under `src/` are deliberately
not modified.

## Reproducible inputs

- `benchmark_frames.json` is the fixed labeled frame set.
- `run_vlm_benchmark.py` sends production-equivalent or experimental safety
  requests to an OpenAI-compatible VLM endpoint and writes resumable JSONL.
- `analyze_vlm_benchmark.py` calculates specialist accuracy, stability, and
  latency from those JSONL files.

Always run Python through `uv` from the repository root. Example:

```bash
uv run python tests/experiments/run_vlm_benchmark.py \
  --source 1920x1080 --port 8000 --variant production --repeats 3

uv run python tests/experiments/analyze_vlm_benchmark.py \
  tests/experiments/results/*.jsonl
```

The production variant issues the same five monitoring requests in parallel:
two insertion banks, one drop bank, and two rotation banks. The
`current_focused` variant is an experimental four-request safety batch with two
independent drop banks and two reduced rotation banks. It does not change the
runtime pipeline.

`drop_no_reference` sends two differently worded current-only drop checks. It
is the leakage ablation: neither request contains frame 6040 or any other
positive failure reference.

`rotation_current_focused` and `drop_reference_focused` run the two halves of
the current-focused batch independently to measure a decoupled safety
scheduler and the second stage of a fast-trigger/verified-drop cascade.

`safety_primary` continuously runs both rotation voters plus the conservative
drop-A voter. A positive drop-A result is intended to trigger drop-B on the
same retained frame instead of paying for both drop requests on every round.

`production_primary` adds the insertion-A trigger to that three-way safety
batch. Production invokes insertion-B only on a positive retained-frame
trigger, after the safety result has been fused.

## Ground-truth caveat

The current production `DROPPED` reference is source frame 6040, which lies
inside the evaluated drop interval. Reports therefore show metrics both with
and without frame 6040. Same-video results are diagnostic and cannot establish
generalization to unseen scenes.
