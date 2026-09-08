# Experiment 2 results

Added two native-resolution reference images to each specialist and clarified the seated adapter's protruding shape and the gripper's release gap.

The 560-token static probe improved to 30/30 correct specialist decisions. Both live pilots matched all four final episode labels. However, the 560-token run declared traj-9 a drop at 5.24 s, before pickup and long before its actual drop. Its explanations correctly described the adapter as on the stand while its POSITIVE verdict was inconsistent with that description.

| Endpoint | Correct final labels | Correct labels with valid drop timing | Request errors |
|---|---:|---:|---:|
| 560 tokens | 4/4 | 3/4 | 3 |
| 1120 tokens | 4/4 | 4/4 | 11 |

The 1120-token pilot declared traj-9 FAIL at 35.85 s and traj-82 FAIL at 28.70 s, both after the visually reviewed drops. The 560-token pilot was faster but unreliable before pickup.

Experiment 3 changes drop output to an explicit physical-location enum and derives the vote in Python. This removes the model's need to translate a correctly observed stand/gripper/socket relation into an abstract positive/negative failure flag.

Evidence: `probe-results-560/`, `runs/pilot-560/`, `runs/pilot-1120/`. All implementation code and reference images are local to this experiment folder.
