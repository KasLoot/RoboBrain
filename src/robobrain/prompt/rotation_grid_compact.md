# Task

Decide only whether the small white plug in the current image matches the
rotated-in-gripper failure references.

# References

- `NORMAL` is a normal controlled plug. Its pin direction and the gripper
  direction agree.
- `ROTATED_A` and `ROTATED_B` are failures. Their pin direction and gripper
  direction disagree.

Judge only the grid panel labeled `CURRENT`. First identify the closest
reference pose, then compare the pin axis and gripper/finger axis inside the
current panel. Compare this internal relation, not the absolute arm, camera,
power-strip, or socket angle. Camera and robot motion cannot make a rotated
grasp normal. A small current pose that closely matches a rotated reference is
still rotated.

Return only this compact JSON, with no Markdown and no explanation:

{"visible":"Y|N","closest":"NORMAL|ROTATED_A|ROTATED_B|NONE","rotation":"R|N|U"}

Use `U` only when the plug is not held or its plug-to-gripper relation is truly
not visible.
