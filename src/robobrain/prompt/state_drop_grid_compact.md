# Task

Classify the current small white plug state as exactly one letter:

- `S` — RESET/STAND: the plug is upright on its large white pedestal/stand and
  is not controlled by the gripper.
- `C` — CONTROLLED: the plug is held by, directly under, or safely occluded by
  the gripper. An empty white stand elsewhere does not change this class.
- `D` — DROPPED: the small plug itself is positively visible lying or standing
  separately on the table, away from the gripper and not seated on its stand.
- `U` — UNCERTAIN: none of the relations above is visually supportable.

The grid panels whose headings begin with `RESET`, `CONTROLLED`, `SAFE`, and
`DROPPED` are labeled references. Judge only `CURRENT`.

Use object identity, not white color. The large rectangular pedestal is the
stand, not a dropped plug. Non-visibility alone is not a drop. Distinguish a
plug seated upright on its pedestal (`S`) from a plug separated on the table
(`D`).

Return exactly one character: `S`, `C`, `D`, or `U`. Do not explain.
