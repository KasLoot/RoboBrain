# RoboBrain v2 experiments

Each numbered directory contains its own implementation, configuration, prompts, reference images, and runnable checks. No experiment imports production code or another experiment's code. The episode dataset is supplied separately with `--videos`.

Recommended next baseline: [experiment 6](robobrain-v2-test-6/RESULTS.md), using the 1120-token endpoint. Its updated combined results match 105/105 episode annotations, with both drops detected, no incorrect terminal labels, and no unresolved episodes. Only the user-extended `traj-94.MOV` was rerun; the other 104 results are retained from the original complete run, which scored 104/105. The code, prompts, images, and configuration were unchanged for the rerun.

[Additional dataset evaluation](robobrain-v2-test-6/NEW_DATASETS_RESULTS.md): 93/93 correct latest outcomes across the no-force August 28 and full-model August 27 datasets, with eight drops detected and no remaining ON_GOING clips. All terminal decisions were visually reviewed. The no-force score includes the rerun of its extended and relabeled `traj-32.MOV`.

| Experiment | Change | Measured outcome |
|---|---|---|
| [1](robobrain-v2-test-1/RESULTS.md) | Zero-shot specialists | Live pilot exposed false and premature drop decisions |
| [2](robobrain-v2-test-2/RESULTS.md) | Labeled reference images and clearer contact rules | Better success recognition; abstract drop verdict could contradict observed stand location |
| [3](robobrain-v2-test-3/RESULTS.md) | Derive drop votes from physical location | 104/105 on the complete 1120-token run; no incorrect terminal labels, one unresolved success |
| [4](robobrain-v2-test-4/RESULTS.md) | Add an earlier real frame to each request | Regressed in the 560-token screening; not selected |
| [5](robobrain-v2-test-5/RESULTS.md) | Tighter native success crop | Eight-episode pilot passed; a longer run exposed an unintended socket-ordinal constraint |
| [6](robobrain-v2-test-6/RESULTS.md) | Remove socket-ordinal constraint | Updated combined score: 105/105 after rerunning the extended episode 94; original complete run: 104/105 |

Runs marked `STOPPED.md` are partial experiments and must not be reported as complete 105-episode evaluations. Episode-label accuracy and individual-observation probe accuracy are distinct measurements. Drop timing is checked against separately reviewed onset intervals; exact success-onset timestamps are not available.

From the repository root, run a standalone experiment with:

```sh
uv run --script test/robobrain-v2-test-6/run.py \
  --videos video/insert_plug-full_model-aug-28-105trail-1x \
  --base-url http://127.0.0.1:9000/v1
```

The 1120-token endpoint is used for accuracy-focused evaluation. The simulator publishes at 30 fps independently of request latency. A single global verification round contains ten paired observations; all twenty responses finish before another round can start. The next normal pair is reused as the first verification observation. Each specialist requires more than five positives, and errors never contribute positive evidence.

The repository's existing `.gitignore` excludes `test/`; these artifacts are present on disk without changes to that rule or production files.
