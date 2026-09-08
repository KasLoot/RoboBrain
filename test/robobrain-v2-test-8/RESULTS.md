# Experiment 8 results

The ten-episode streaming pilot matched all ten episode annotations: eight successes, traj-9 as ROTATION_FAILURE, and traj-82 as DROP_FAILURE. There were no request errors, false terminal labels or unresolved episodes. All verification rounds awaited 30 unique responses and the scheduler audit found no violations.

This candidate is not selected yet: traj-9's rotation decision was published at 39.91 s, after the recorded drop at approximately 32.6–32.8 s. Its rotation evidence was genuinely captured earlier, at approximately 28 s, but this is too late to claim a pre-drop intervention. A false drop trigger during the 9.67 s approach started an expensive round that blocked a new rotation verification; each moving-frame burst took about 12 s on this endpoint. Experiment 9 refines the drop prompt to require visible separation from the fingers.

The rotation-specific image screen removed all three released-plug false positives found in experiment 7. One blurred rotated frame was misclassified as not held. The raw image probe includes one inherited incorrect expected drop label; its independent correction is documented in [probe review](probe-results/REVIEW.md), without altering the recorded requests or responses.

Artifacts: [pilot results](runs/pilot-105/summary.json), [timing and scheduling audit](runs/pilot-105/audit.json), [reviewed event intervals](event_times.json), and [raw probe output](probe-results/port-9000.json). Full-dataset testing was deferred pending the delay fix. Production code and actual robot APIs were untouched.
