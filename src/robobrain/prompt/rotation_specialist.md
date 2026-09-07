# Task

Decide only whether the small white plug in the current image matches the rotated-in-gripper failure references.

# Labeled references

- `CONTROLLED_NORMAL_DEMO` shows the plug held in a normal recoverable pose. It is `NOT_ROTATED`.
- `FAIL_DEMO_2` and `FAIL_DEMO_3` show the exact plug rotated relative to the gripper. Both are `ROTATED`.

Judge only `FULL_WORKSPACE_VIEW` and `ZOOM_IN_VIEW` as current. Compare the current held-plug pose directly with the reference poses. Camera and robot-arm movement do not make a rotated pose normal. Do not use proximity to the socket as the deciding cue.

If the plug is not visible well enough to compare, return `UNCERTAIN`, not `ROTATED`.

Return only:

```json
{
  "rotation_match": "ROTATED | NOT_ROTATED | UNCERTAIN",
  "matched_reference": "FAIL_DEMO_2 | FAIL_DEMO_3 | CONTROLLED_NORMAL_DEMO | NONE",
  "reason": "a short visual reason"
}
```
