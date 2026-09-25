"""Measured robot-stage travel, independent of object ground truth."""
import numpy as np


def stage_progress_text(motion, anchor_xyz, current_pose, nominal_lift_mm=None):
    anchor = np.asarray(anchor_xyz, dtype=float)[:3]
    current = np.asarray(current_pose, dtype=float)[:3]
    delta = (current - anchor) * 1000.
    text = ('MEASURED STAGE PROGRESS (retained across stop/resume): '
            f'stage={motion}; stage-entry TCP XYZ mm={np.round(anchor * 1000, 2).tolist()}; '
            f'current measured TCP XYZ mm={np.round(current * 1000, 2).tolist()}; '
            f'NET displacement since this stage began mm={np.round(delta, 2).tolist()}. '
            'This is cumulative stage travel, not the displacement of only the last atomic action. ')
    if str(motion).upper() == 'LIFT' and nominal_lift_mm is not None:
        remaining = max(0., float(nominal_lift_mm) - float(delta[2]))
        text += (f'The nominal post-grasp upward travel is {nominal_lift_mm:g} mm; remaining nominal TCP travel={remaining:.2f} mm. '
                 'If travel is already met, do not repeat a 50 mm lift merely because execution resumed '
                 'or the last correction was smaller than 50 mm. If short, correct only the remaining '
                 'distance with appropriate available steps. ')
    return text + ('TCP travel does not prove plug travel or task completion. Use CURRENT images to '
                   'confirm secure carry, clearance, and all stage conditions before DONE. Genuine '
                   'regrasp/recovery starts a new stage visit; pausing alone does not.')
