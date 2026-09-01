# Preconditions

The attempt is active. Judge only the panel labeled `CURRENT`.

# Task

Classify the internal plug-pin direction versus gripper/finger direction:

- `R`: the pin axis points in a different direction from the gripper/finger axis.
- `N`: the pin axis agrees with the gripper/finger axis.
- `U`: the axes truly cannot be compared.

Panels labeled `ROTATED` are positive failures. Panels labeled `NORMAL` are
hard negative examples from normal transport, grasp, and successful insertion.
Socket proximity, arm pose, camera angle, and the absolute diagonal appearance
must not decide the answer. Compare the two axes inside `CURRENT`.

Return only compact JSON, without Markdown or explanation:

{"closest":"ROTATED_A|ROTATED_B|NORMAL_A|NORMAL_B|NORMAL_C|NONE","rotation":"R|N|U"}
