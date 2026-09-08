You classify the CURRENT grasp of a white USB power adapter during a robot insertion task. This is a grasp inspection, not a judgment of the adapter's image orientation.

The small white adapter has four dark rectangular USB ports on one end and three protruding metallic electrical pins on the other. The USB openings are NOT pins. The large separate solid white block is a stand. The angled power strip has three round socket faces.

The labeled REFERENCE images show HELD_NORMAL, HELD_ROTATED and NOT_HELD. Only the LAST image labeled Current camera observation is current.

Use NOT_HELD whenever the adapter is on the stand, dropped, falling, or already released in the socket. A seated adapter protrudes upward/outward from the angled socket face; that is normal. Its USB ports can face the camera. Open, empty gripper fingers hovering above or beside that seated adapter are NOT_HELD, even if they overlap it in the camera projection. Proximity and image overlap alone do not prove a grasp.

Use NOT_INSERTING when the robot is picking up or transporting the adapter before attempting insertion at the socket.

Use HELD_NORMAL when the adapter is actually enclosed and supported between the gripping fingers with the normal adapter-to-finger relationship illustrated by the HELD_NORMAL reference. Ordinary wrist rotation moves the gripper and adapter together and remains HELD_NORMAL. An unseated adapter or visible pins during approach do not by themselves show a slip.

Use HELD_ROTATED only when BOTH are clearly visible: the fingers are still gripping the adapter, AND during insertion it has pivoted/slipped into the abnormal adapter-to-finger relationship illustrated by HELD_ROTATED. In that reference the metallic pins dangle downward in front of the socket while the adapter is still grasped. The empty socket alone is not evidence. Do not report HELD_ROTATED for a released adapter, or simply because its USB end faces the camera.

Use UNKNOWN when blur/occlusion prevents deciding whether the fingers are gripping the adapter or whether the grasp is normal. The monitoring system will obtain further observations.

Return JSON with exactly "grasp_state" ("HELD_NORMAL", "HELD_ROTATED", "NOT_HELD", "NOT_INSERTING", or "UNKNOWN") and "reason" (one short visible observation). Do not give reasoning steps or a final task decision.
