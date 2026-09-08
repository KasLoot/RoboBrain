# RoboBrain test results

Latest saved results for the three datasets, evaluated with experiment 6 (the version promoted into production) on the 1120-token Gemma 4 endpoint. The collection includes summaries, raw request logs, visual evidence, configuration snapshots, validation and rerun history.

| Dataset | Correct outcomes | SUCCESS | DROP_FAILURE | ON_GOING |
|---|---:|---:|---:|---:|
| [insert_plug-full_model-aug-28-105trail-1x](insert_plug-full_model-aug-28-105trail-1x/README.md) | 105/105 | 103 | 2 | 0 |
| [insert_plug-no_force-aug-28-1x](insert_plug-no_force-aug-28-1x/README.md) | 48/48 | 41 | 7 | 0 |
| [insert_plug-full_model-aug-27-1x](insert_plug-full_model-aug-27-1x/README.md) | 45/45 | 44 | 1 | 0 |
| **Total** | **198/198** | **188** | **10** | **0** |

- [All dataset and episode results](summary.json).
- [Detailed original 105-episode evaluation and episode 94 update](insert_plug-full_model-aug-28-105trail-1x/REPORT.md).
- [Detailed additional-dataset evaluation and episode 32 update](ADDITIONAL_DATASETS_REPORT.md).

The original 105-episode score retains 104 results and replaces episode 94 with its extended-clip rerun. The no-force score retains 47 results and replaces episode 32 with its extended-clip rerun; that annotation changed from SUCCESS to FAIL. The August 27 score is one complete 45-episode run. These are previously saved results, not a new run of all 198 clips.

No selected episode is unresolved. All terminal decisions were visually reviewed. One request error occurred in the August 27 run and was handled as UNCERTAIN; the final episode outcome was correct. The detailed reports describe the review scope and timing limits.
