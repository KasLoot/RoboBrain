# Visual review of probe labels

Raw probe responses and their original expected labels are preserved. These are development image checks, not episode-level accuracy measurements.

The revised rotation prompt removes the three released-plug false positives seen in experiment 7 (traj-2 at 14.7 s, traj-11 at 17.1 s, traj-15 at 15.7 s). It identifies five of the six sampled physically rotated states in traj-9. At 24 s motion blur causes it to mistake the adapter for a falling object; this is a missed rotation vote, not a verified terminal decision.

Independent review of surrounding frames corrects one old probe label: traj-82 at 21 s is already falling, so its positive drop observation is correct. The original probe incorrectly expected no positive specialist at that timestamp. See `../event-review/traj-82-drop-fine.jpg` and `../event_times.json`. No model prompt was changed for this correction.

The drop outputs on traj-9 at 9.6 s, 24 s and 32.4 s are individual false positives: the adapter is still retained. The streaming evidence horizon must reject these or produce a correct independently verified failure type. Source-frame seeking was checked against sequential decoding for four frames and produced identical pixels.
