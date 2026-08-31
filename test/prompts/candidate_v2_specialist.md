# Role

Classify the current robot-plug relation by direct visual comparison with the labeled reference images.

# Reference meanings

- `FAIL_DEMO_1`: the small white plug is visibly separate from the gripper on the table. This is `DROP`.
- `FAIL_DEMO_2` and `FAIL_DEMO_3`: the small white plug remains held but has rotated relative to the gripper. Any matching held pose is `ROTATED`.
- `SAFE_OCCLUSION_DEMO`: the large white block in the lower foreground is the empty stand, not the plug. The plug is held by or hidden under the gripper. This is `NO_FAILURE`.
- `CONTROLLED_NORMAL_DEMO`: the plug is held by the gripper in a normal state. The empty stand may still be visible elsewhere. This is `NO_FAILURE`.

Judge only `FULL_WORKSPACE_VIEW` and `ZOOM_IN_VIEW` as current. Do not classify a reference image.

# Decision

Use visual identity, not mere whiteness:

1. If the small plug is visibly separate from the gripper and matches `FAIL_DEMO_1`, choose `DROP`.
2. Otherwise, if the held plug pose matches `FAIL_DEMO_2` or `FAIL_DEMO_3`, choose `ROTATED`.
3. If the scene matches either safe reference, choose `NO_FAILURE`.
4. If the plug is hidden and there is no positive evidence that the small plug is separate, choose `NO_FAILURE`. Non-visibility alone is never a drop.

The large rectangular stand is never a dropped plug. A stand visible far from the gripper is normal.

# Output

Return only this JSON object:

```json
{
  "visual_match": "DROP | ROTATED | NO_FAILURE",
  "small_plug_separate_from_gripper": "YES | NO | UNCERTAIN",
  "large_white_stand_visible": "YES | NO | UNCERTAIN",
  "status": "ON_GOING | FAIL",
  "failure_type": "NONE | ROTATED_OR_MISALIGNED | DROP_OR_CONTROL_LOSS",
  "failure_description": "none | a short description"
}
```

Map `DROP` to `FAIL` / `DROP_OR_CONTROL_LOSS`, `ROTATED` to `FAIL` / `ROTATED_OR_MISALIGNED`, and `NO_FAILURE` to `ON_GOING` / `NONE`.
