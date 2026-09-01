# Task

Decide whether the small white plug in the panel labeled `CURRENT` is clearly
seated in a socket of the power strip.

- `I`: visibly inserted/seated. The gripper may still touch it, be open above
  it, or withdraw after release.
- `N`: not fully seated. This includes stand/reset, reaching, grasping,
  transport, above-socket approach, contact without clear seating, loose table
  drop, and human handling.
- `U`: genuinely impossible to determine.

Panels labeled `INSERTED` are positive references. Panels labeled
`NOT_INSERTED`, `STAND`, and `DROPPED` are hard negatives. Judge only `CURRENT`.

Require positive visual evidence that the plug occupies a socket. Gripper
release is not drop when the plug remains seated. Do not infer insertion merely
from proximity to the power strip.

Return exactly one character: `I`, `N`, or `U`. Do not explain.
