# Experiment 5 results

Tightened the native success crop from 610 x 600 to 470 x 400 pixels, retaining the scheduler and location-based drop specialist. The success references were cropped to the same bounds. No images were resized.

The probe scored 50/52 specialist observations at 560 visual tokens and 51/52 at 1120. The 560-token live pilot completed all eight selected episodes correctly, including the prior close-release misses and both drops at valid times.

The 1120-token full evaluation was stopped after 58 completed episodes: 57 correct, one unresolved, and no incorrect terminal labels. The tighter crop resolved episode 58. Episode 44 exposed a different issue: the success model correctly observed an inserted adapter but rejected it because it called the socket "middle" rather than "rightmost". Socket ordinal was an unintended prompt constraint, not a requirement of the user's task.

Experiment 6 removes that constraint while keeping insertion, release, the native crops, and the global twenty-response verification policy.

Evidence: `probe-results-560/`, `probe-results-1120/`, `runs/pilot-560/`, and `runs/all-105-1120/`. The stopped run is explicitly marked in `STOPPED.md` inside its run directory; it is not a complete 105-episode result.
