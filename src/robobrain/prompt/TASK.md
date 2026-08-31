`TASK`: Pick up the plug and insert it into the socket.
`TASK_GOAL`: The plug is inserted into the socket. (Example: `SUCCESS_DEMO`)
`TASK_FAILURE_FLAG`:
- `DROP_OR_CONTROL_LOSS`: After the plug was controlled, it leaves the gripper or the gripper loses control of it. A plug that has not yet been grasped is not dropped. (Example: `FAIL_DEMO_1`)
- `ROTATED_OR_MISALIGNED`: While controlled, the plug pin direction differs from the gripper direction, or the pins are not aligned with the socket holes. The robot cannot recover from this state. (Examples: `FAIL_DEMO_2`, `FAIL_DEMO_3`)
