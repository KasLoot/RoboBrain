"""Scripted contact-physics fixtures. Never imported by deployed robot control.

Known object coordinates initialize repeatable tests only. Objects remain free
bodies and are moved by the original Panda actuators and contact dynamics.
"""
from pathlib import Path

import mujoco
import numpy as np

from core.config import load_yaml
from core.sim.continuous_physics import configure_continuous_physics
from core.sim.mujoco_session import MujocoSession
from scripts.run_mujoco import make_controller

ROOT = Path(__file__).resolve().parents[1]


def make_plug_session(noslip_iterations=0, *, cone='pyramidal', impratio=1):
    cfg = load_yaml(ROOT / 'configs/robot_mujoco_plug.yaml')
    # Isolate the test's explicit option from the deployed plug profile default.
    cfg['continuous_physics'] = {'noslip_iterations': noslip_iterations, 'cone': cone, 'impratio': impratio}
    session = MujocoSession(cfg)
    configure_continuous_physics(session.model, cfg)
    make_controller(session, cfg)  # Retain verified speed scaling/Cartesian sizes.
    return session


def move_tcp(session, position):
    pose = session.get_ee_pose()
    pose[:3] = position
    session.update_desired_ee_pose(pose)


def grasp_and_lift(session, *, height_offset_m=0.):
    move_tcp(session, [.48, -.10, .18])
    move_tcp(session, [.48, -.10, .106 + height_offset_m])
    session.control_gripper(True)
    move_tcp(session, [.48, -.10, .20 + height_offset_m])


def loaded_finger_contacts(session):
    """Audit actual loaded handle contacts on each finger, not aperture alone."""
    handle = session.model.geom('plug_handle').id
    sides = {int(session.model.body(name).id): name for name in ('left_finger', 'right_finger')}
    normals = {name: 0. for name in sides.values()}
    tangents = {name: 0. for name in sides.values()}
    counts = {name: 0 for name in sides.values()}
    for index, contact in enumerate(session.data.contact):
        left, right = map(int, contact.geom)
        if handle not in (left, right):
            continue
        body = int(session.model.geom_bodyid[right if left == handle else left])
        if body not in sides:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(session.model, session.data, index, force)
        name = sides[body]
        normals[name] += max(0., float(force[0]))
        tangents[name] += float(np.linalg.norm(force[1:3]))
        counts[name] += int(force[0] > 1e-4)
    return {'normal_force_N': normals, 'tangential_force_N': tangents,
            'loaded_contact_count': counts, 'bilateral_loaded_grasp': all(value > .01 for value in normals.values())}


def insertion_sample(session):
    """Audit-only geometry in the socket frame; not a controller observation."""
    entry = session.data.site('socket_entry')
    rotation = entry.xmat.reshape(3, 3)
    tip = rotation.T @ (session.data.site('plug_tip').xpos - entry.xpos)
    length = float(2 * session.model.geom('plug_peg').size[1])
    contacts = loaded_finger_contacts(session)
    return {'sim_time_s': float(session.data.time), 'tip_penetration_m': float(-tip[2]),
            'pin_length_m': length, 'tip_offset_xy_m': float(np.linalg.norm(tip[:2])),
            'fraction_of_pin_below_entry': float(np.clip(-tip[2] / length, 0, 1)),
            'gripper_width_m': float(session.get_gripper_position()[0]), **contacts}


class InsertionTrace:
    """Capture a test's physical path, including the instant release is requested."""
    def __init__(self, session):
        self.session = session
        self.rows = []
        self.release_at_sim_time = None

    def capture(self):
        # Match the service's post-step kinematic refresh without integrating time.
        mujoco.mj_kinematics(self.session.model, self.session.data)
        self.rows.append(insertion_sample(self.session))

    def gripper_command(self, close):
        if not close and self.release_at_sim_time is None:
            self.release_at_sim_time = float(self.session.data.time)
        self.capture()

    def qualified_before_release(self):
        return any(row['bilateral_loaded_grasp'] and row['tip_penetration_m'] >= .025
                   and row['tip_offset_xy_m'] < .003
                   and (self.release_at_sim_time is None or row['sim_time_s'] < self.release_at_sim_time)
                   for row in self.rows)

    def close(self):
        pass


def observe_seated_geometry(session, seconds=5.):
    """Quantify seating and residual motion; keep original success criteria intact."""
    positions, angular, tilts, depth_errors = [], [], [], []
    geometry_failures = []
    success_count = 0
    class Probe:
        def capture(self):
            nonlocal success_count
            mujoco.mj_kinematics(session.model, session.data)
            diagnostic = session.success_diagnostics()
            positions.append(session.data.body('plug').xpos.copy())
            angular.append(float(np.linalg.norm(session.data.joint('plug_joint').qvel[3:])))
            tilts.append(float(np.rad2deg(np.arccos(np.clip(diagnostic['criteria']['upright']['value'], -1, 1)))))
            depth_errors.append(diagnostic['criteria']['insertion_depth']['value'])
            geometric = [name for name in diagnostic['failed_criteria'] if name not in ('linear_rest', 'angular_rest')]
            if geometric: geometry_failures.append(geometric)
            success_count += int(diagnostic['success'])
        def close(self): pass
    old_recorder = session.recorder
    session.recorder = Probe()
    try:
        session._advance(seconds)
    finally:
        session.recorder = old_recorder
    positions = np.asarray(positions)
    return {'observed_sim_s': seconds, 'samples': len(positions),
            'position_peak_to_peak_m': np.ptp(positions, axis=0).tolist(),
            'max_tilt_deg': max(tilts), 'max_depth_error_m': max(depth_errors),
            'angular_speed_min_rad_s': min(angular), 'angular_speed_max_rad_s': max(angular),
            'angular_rest_failed_samples': sum(value >= .1 for value in angular),
            'original_success_samples': success_count, 'geometry_failures': geometry_failures}
