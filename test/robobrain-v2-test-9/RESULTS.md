# Experiment 9 results

This prompt-only attempt did not fix the response delay. All four pilot episode labels were correct, but traj-9 ultimately confirmed DROP_FAILURE at 41.31 s; it did not confirm rotation before the drop. An approach-frame FALLING false alarm started the first round. A later rotation round sampled a blurred interval and received only three positive rotation votes, below the required strict threshold.

The static screen likewise retained motion-blur mistakes. The drop prompt refinement is therefore rejected. Experiment 10 returns to experiment 8's prompts and changes only the drop vote mapping so FALLING abstains and TABLE remains positive.

There were no request errors or scheduler violations. These four correct eventual outcomes do not demonstrate a useful rotation intervention. See [pilot results](runs/pilot-105/summary.json), [audit](runs/pilot-105/audit.json), and [image probe](probe-results/port-9000.json). Production and robot APIs were not changed.
