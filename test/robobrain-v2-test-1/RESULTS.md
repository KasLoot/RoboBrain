# Experiment 1 results

The zero-shot baseline is not suitable for the task.

The 15-frame probe produced 28/30 correct specialist decisions at both 560 and 1120 visual tokens. Neither endpoint returned request errors in that probe.

The real-time 1120-token pilot was more revealing:

| Episode | Expected | Published | Assessment |
|---|---|---|---|
| 1 | SUCCESS | FAIL at 18.74 s | Incorrect drop classification |
| 9 | FAIL after the drop around 32.7 s | FAIL at 16.67 s | Premature; rotation/held state must remain ON_GOING |
| 82 | FAIL | FAIL at 26.11 s | Correct drop detection |
| 105 | SUCCESS | ON_GOING at the 90 s watchdog | Missed success |

Matching episode labels alone gives 2/4, but only one of the four outcomes was correctly classified and timed. Repeated voting amplified a persistent visual mistake. Experiment 2 therefore adds labeled reference images and clarifies gripper contact versus proximity.

Evidence: `probe-results/`, `runs/pilot-1120/`. Production files were not changed.
