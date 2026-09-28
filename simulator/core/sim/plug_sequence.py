"""Ground-truth process evaluation for logs/success only, never VLM observations.

The controller must choose motions and release from camera evidence. This evaluator
prevents a final seated pose from retroactively certifying an accidental free fall.
It neither moves objects nor supplies target coordinates to the controller.
"""
from copy import deepcopy
import threading

import mujoco
import numpy as np

from core.sim.mujoco_success import PLUG_MIN_INSERTION_M, PLUG_PIN_LENGTH_M


CARRY_STAGES = {'LIFT', 'MOVE', 'INSERT', 'RELEASE', 'RETREAT', 'PLACE'}


def measure_engagement(model, data, *, empty_width_m=.005):
    """Measure actual bore penetration and loaded contacts on both fingertip pads."""
    entry = data.site('socket_entry')
    rotation = entry.xmat.reshape(3, 3)
    tip = rotation.T @ (data.site('plug_tip').xpos - entry.xpos)
    axis = rotation.T @ data.site('plug_tip').xmat.reshape(3, 3)[:, 2]
    peg = model.geom('plug_peg')
    pin_length = float(2 * peg.size[1])
    if not np.isclose(pin_length, PLUG_PIN_LENGTH_M, atol=1e-6):
        raise ValueError('Plug process criterion requires the configured 50 mm pin')
    depth = float(-tip[2])
    # Only the shaft segment actually below the entry plane must fit the bore.
    entered = min(pin_length, max(0., depth / axis[2])) if axis[2] > 0 else 0.
    ends = np.array([tip, tip + entered * axis])
    radial_extent = float(np.max(np.linalg.norm(ends[:, :2], axis=1)) + peg.size[0])
    aperture = float(model.site('socket_entry').size[0])
    handle = model.geom('plug_handle').id
    fingers = {model.body(name).id: name for name in ('left_finger', 'right_finger')}
    forces = {name: 0. for name in fingers.values()}
    counts = {name: 0 for name in fingers.values()}
    force = np.zeros(6)
    for index, contact in enumerate(data.contact):
        if handle not in contact.geom:
            continue
        other = int(contact.geom[1] if contact.geom[0] == handle else contact.geom[0])
        name = fingers.get(int(model.geom_bodyid[other]))
        if name is None or model.geom_type[other] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        mujoco.mj_contactForce(model, data, index, force)
        normal = max(0., float(force[0]))
        if normal > 1e-4:
            forces[name] += normal
            counts[name] += 1
    width = sum(float(data.joint(name).qpos[0]) for name in ('finger_joint1', 'finger_joint2'))
    closed_command = float(data.ctrl[model.actuator('actuator8').id]) < 1.
    held = bool(closed_command and width > empty_width_m and min(forces.values()) > .01)
    qualified = bool(held and depth >= PLUG_MIN_INSERTION_M and radial_extent < aperture
                     and axis[2] > np.cos(np.deg2rad(6)))
    return {'sim_time_s': float(data.time), 'pin_length_m': pin_length,
            'required_insertion_m': PLUG_MIN_INSERTION_M, 'tip_penetration_m': depth,
            'inserted_fraction': max(0., min(1., depth / pin_length)),
            'radial_extent_m': radial_extent, 'bore_radius_m': aperture,
            'upright_cosine': float(axis[2]), 'gripper_width_m': width,
            'closed_command': closed_command, 'loaded_pad_contacts': counts,
            'pad_normal_forces_n': forces, 'held_by_both_pads': held,
            'half_inserted_while_held': qualified,
            'tcp_to_handle_top_z_m': float(data.site('tcp').xpos[2] - data.site('plug_top').xpos[2])}


class PlugInsertionSequence:
    def __init__(self, artifacts, *, empty_width_m=.005):
        self.artifacts = artifacts
        self.empty_width_m = float(empty_width_m)
        self.lock = threading.RLock()
        self.carry_started = False
        self.half_inserted = None
        self.qualified_release = None
        self.pending_release = None
        self.violation = None
        self.last = None

    def observe(self, model, data, identity, *, stage='', token=None, boundary='observation'):
        sample = {**identity, **measure_engagement(model, data, empty_width_m=self.empty_width_m)}
        with self.lock:
            if stage.upper() in CARRY_STAGES and sample['held_by_both_pads']:
                self.carry_started = True
            if sample['half_inserted_while_held'] and self.violation is None and self.half_inserted is None:
                self.half_inserted = deepcopy(sample)
                self.artifacts.append('plug-insertion-sequence.jsonl', {
                    'event': 'half_inserted_while_held', 'sample': sample})
            if self.carry_started and self.half_inserted is None and self.violation is None:
                if sample['closed_command'] and sample['gripper_width_m'] <= self.empty_width_m:
                    self.violation = {'reason': 'grasp_lost_before_half_insertion', 'sample': deepcopy(sample)}
                elif token == 'RELEASE' and boundary == 'before_gripper_open':
                    self.violation = {'reason': 'release_before_half_insertion', 'sample': deepcopy(sample)}
                if self.violation:
                    self.artifacts.append('plug-insertion-sequence.jsonl', {'event': 'sequence_violation', **self.violation})
            if token == 'RELEASE' and boundary == 'before_gripper_open' and self.half_inserted and not self.violation:
                # This is the actuation boundary, after visual authorization. A
                # rejected high-level request must not certify a release or violation.
                if sample['half_inserted_while_held']:
                    self.pending_release = deepcopy(sample)
                elif sample['closed_command'] or self.qualified_release is None:
                    self.violation = {'reason': 'release_without_current_held_half_insertion', 'sample': deepcopy(sample)}
                    self.artifacts.append('plug-insertion-sequence.jsonl', {'event': 'sequence_violation', **self.violation})
            if token == 'RELEASE' and boundary == 'after_gripper_open':
                if self.pending_release and not self.violation and not sample['closed_command'] and sample['gripper_width_m'] > .06:
                    self.qualified_release = {**self.pending_release, 'opening_completed_at': deepcopy(identity)}
                    self.artifacts.append('plug-insertion-sequence.jsonl', {
                        'event': 'qualified_release', 'sample': self.qualified_release})
                self.pending_release = None
            self.last = sample
            if boundary != 'observation':
                self.artifacts.append('plug-insertion-sequence.jsonl', {
                    'event': boundary, 'token': token, 'stage': stage, 'sample': sample})
            return {'current': deepcopy(sample), **self.state()}

    def cancel_release(self, reason):
        with self.lock:
            if self.pending_release is not None:
                self.artifacts.append('plug-insertion-sequence.jsonl', {
                    'event': 'opening_interrupted', 'reason': str(reason), 'pre_open_sample': self.pending_release})
            self.pending_release = None

    def state(self):
        with self.lock:
            return {'required_insertion_m': PLUG_MIN_INSERTION_M,
                    'half_inserted_while_held': deepcopy(self.half_inserted),
                    'qualified_release': deepcopy(self.qualified_release),
                    'violation': deepcopy(self.violation), 'release_pending': self.pending_release is not None,
                    'passed': bool(self.half_inserted and self.qualified_release and not self.violation)}

    def supplement(self, diagnostics):
        state = self.state()
        criteria = dict(diagnostics['criteria'])
        criteria['held_half_insertion_before_release'] = {
            'value': int(state['passed']), 'operator': '>=', 'threshold': 1,
            'unit': 'qualified_sequence', 'passed': state['passed']}
        failed = [name for name, item in criteria.items() if not item['passed']]
        return {**diagnostics, 'criteria': criteria, 'failed_criteria': failed,
                'success': not failed, 'insertion_sequence': state}
