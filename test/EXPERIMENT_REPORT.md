# Plug-failure detection experiment

## Scope

All changes and generated artifacts are confined to `test/`. No production source, prompt, demo, root output, or video file was modified.

## Supplied-output diagnosis

- The production prompt reports `pin_direction_vs_gripper: SAME` and `ON_GOING` throughout the sampled frames from 493 through 955, so it misses the rotated-plug episode.
- Frame 1316 is reported as `DROP_OR_CONTROL_LOSS`, even though the large white object on the table is the empty stand rather than the small plug.
- The unannotated rotation demos match source frames 468 and 903 almost exactly. Both are inside the required 465–970 interval. The evidence exists, but the monolithic prompt does not use it reliably.

## Experiments

### Prompt/demo matrix

Six critical frames were used: rotated poses at 493, 609, and 897; a real dropped-plug pose at 1000; and non-fail poses at 1316 and 1377.

| Variant | Passed | Main observation |
|---|---:|---|
| Production prompt + arrow demos | 2/6 | Missed every rotation; false drop at 1316 |
| Production prompt + raw demos | 3/6 | Detected rotation at 609 and 897; still confused stand with plug |
| Stricter text + arrow demos | 1/6 | Text alone did not correct visual confusion |
| Stricter text + raw demos | 3/6 | Rotation improved, drop confusion remained |
| Single specialist + safe references | 3/6 | Corrected 1316/1377 but suppressed rotation detection |

This isolates two independent perception tasks that interfere when combined: plug rotation and plug/stand identity.

### Independent relation specialists

Two small prompts were tested independently:

1. Rotation specialist: raw rotated references plus a normal controlled reference.
2. Drop specialist: the raw dropped-plug reference plus empty-stand and controlled references.

On the six critical frames, deterministic fusion passed 6/6.

On all 31 frames sampled in `output.log`, the raw per-frame specialist fusion passed 26/31. Its five errors had temporal/state contradictions:

- Frame 73: false drop before any confirmed grasp.
- Frame 435: one early rotation guess.
- Frame 725: one missed rotation after the attempt had already failed.
- Frame 1074: false drop while the plug is visibly returned to the stand.
- Frame 1608: one isolated rotation guess during the later normal attempt.

### Stateful replay

The following attempt-level rules were then applied:

- Gate drop decisions on confirmed grasp.
- Treat visible `HOLDING_PLUG` plus `NOT_YET_GRASPED` as an attempt reset.
- Require two consecutive sampled `ROTATED` observations before declaring rotation failure.
- Latch a declared failure until the next visible reset.

Result: 31/31 sampled statuses matched the test oracle.

Key outcomes:

- First rotation failure: frame 493, within the required 465–970 interval.
- Frame 1316: `ON_GOING` / `NONE`.
- Isolated frame-1608 rotation guess: ignored.
- True later drop at frames 1724 and 1771: detected.

## Reproduction

```powershell
.\.venv\Scripts\python.exe test\extract_diagnostic_frames.py
.\.venv\Scripts\python.exe test\match_demo_frames.py
.\.venv\Scripts\python.exe test\run_relation_specialists.py --suite log-replay
.\.venv\Scripts\python.exe test\evaluate_stateful_fusion.py
```

Relevant artifacts:

- `test/artifacts/frames/contact_sheet_full.jpg`
- `test/artifacts/frames/contact_sheet_crop.jpg`
- `test/artifacts/relation_specialists_log-replay.json`
- `test/artifacts/stateful_fusion.json`

## Production implementation

After approval, the validated direction was applied to production:

1. Raw 300x250 rotation crops are used for specialist visual comparison; explanatory arrows stay out of those pixels.
2. Empty-stand and normal-controlled visual references were added.
3. Rotation and drop/object-identity judgments are evaluated separately and fused into the existing output schema.
4. Production now applies grasp gating, two-observation rotation confirmation, failure latching, and visible stand-return reset.

The production prompt/assets passed the six-frame live smoke test, and the production tracker passed the 31-frame replay.

The current experiment uses references from this video and therefore establishes a fix for the supplied case, not cross-video generalization. Before production adoption, replay the same state machine on additional success, rotation, and drop clips.
