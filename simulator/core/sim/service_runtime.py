"""Continuous MuJoCo execution with one physics owner and generation cancellation.

The original decision-synchronous CLI remains unchanged. Service controllers operate
on private snapshots and submit trajectories; only the physics thread writes live
MjData. Provider calls, IK, rendering, encoding, GUI and recording never run there.
"""
from __future__ import annotations

from collections import deque
from copy import copy, deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
import json
import queue
from pathlib import Path
import threading
import time
from typing import Callable
from uuid import uuid4

import mujoco
import numpy as np
from PIL import Image

from core.config import load_yaml, resolve_vlm_config
from core.sim.mujoco_session import MujocoSession
from core.sim.mujoco_success import PLUG_MIN_INSERTION_M

ROOT = Path(__file__).resolve().parents[2]


class ExecutionCancelled(RuntimeError):
    """The command belongs to a stopped or superseded execution generation."""
    execution_cancelled = True


@dataclass
class _Command:
    kind: str
    generation: int
    attempt_id: str | None = None
    seconds: float = 0
    target: np.ndarray | None = None
    applied: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None
    started: float | None = None
    start_ctrl: np.ndarray | None = None


class _SnapshotSession(MujocoSession):
    """Existing controller/session interface backed by private observation data."""
    skip_initial_release = True

    def __init__(self, runtime, generation=None, attempt_id=None):
        self.__dict__.update(copy(runtime._session.__dict__))
        self.runtime = runtime
        self.generation = generation
        self.attempt_id = attempt_id
        self.data = mujoco.MjData(self.model)
        self._ik = mujoco.MjData(self.model)
        self._jac_pos = np.zeros((3, self.model.nv))
        self._jac_rot = np.zeros((3, self.model.nv))
        self.robot = self
        self.recorder = self._renderer = self._viewer = None
        self._reading = False
        self.observation_identity = None
        self.artifacts = runtime._run_artifacts
        self.insertion_sequence = runtime._plug_sequence
        self.audit_context = {}
        self._release_guard = None
        self._release_approval = None
        self._current_control_id = None
        self._control_context = {'stage': 'INITIALIZATION', 'source': 'system', 'carry_established': False}
        self._refresh()

    def _check(self):
        if self.generation is not None:
            self.runtime.check_execution(self.generation, self.attempt_id)

    def _refresh(self):
        self._check()
        if not self._reading:
            self.observation_identity = self.runtime.snapshot(self.data)

    def get_ee_pose(self):
        self._refresh()
        return super().get_ee_pose()

    def get_gripper_position(self):
        self._refresh()
        return super().get_gripper_position()

    def get_observation(self):
        self._refresh()
        self._reading = True
        try:
            obs = super().get_observation()
            obs['simulation_snapshot'] = dict(self.observation_identity)
            self._check()
            return obs
        finally:
            self._reading = False

    def success_diagnostics(self):
        self._refresh()
        self._reading = True
        try:
            diagnostics = super().success_diagnostics()
            if self.insertion_sequence is not None:
                self._observe_insertion()
                diagnostics = self.insertion_sequence.supplement(diagnostics)
            return diagnostics
        finally:
            self._reading = False

    def check_success(self):
        self._refresh()
        self._reading = True
        try:
            return super().check_success()
        finally:
            self._reading = False

    def set_control_context(self, *, stage, source, carry_established):
        self._control_context = {'stage': stage, 'source': source, 'carry_established': carry_established}

    def authorize_release(self, *, stage, source, carry_established):
        if self._release_guard is None:
            return
        self._release_approval = None
        self.set_control_context(stage=stage, source=source, carry_established=carry_established)
        decision = self._release_guard(self, stage=stage, source=source, carry_established=carry_established)
        self._check()
        from core.sim.release_guard import ReleaseNotQualified
        if not isinstance(decision, dict) or decision.get('verdict') not in ('HALF_INSERTED_HELD', 'SUPPORTED_SOURCE_REGRASP'):
            raise ReleaseNotQualified('UNCERTAIN', 'Release verifier did not provide an explicit qualification')
        if decision['verdict'] == 'SUPPORTED_SOURCE_REGRASP' and (str(stage).upper() != 'GRASP' or carry_established):
            raise ReleaseNotQualified('UNSAFE', 'Source regrasp exemption is unavailable after carrying')
        self._release_approval = {'identity': (self.generation, self.attempt_id, self._current_control_id),
                                  'decision': decision}

    def _check_release_grasp(self):
        self._refresh()
        if self._control_context['carry_established']:
            from core.runners.real import GraspLost
            width = float(super().get_gripper_position()[0])
            threshold = float(self.runtime.config.get('empty_width_m', .005))
            if float(self.data.ctrl[self._gripper]) < 1 and width <= threshold:
                raise GraspLost(token='RELEASE', stage=self._control_context['stage'],
                    source=self._control_context['source'], measured_width_m=width, empty_width_m=threshold)

    def control_gripper(self, close):
        self._refresh()
        if not close and self._release_guard is not None:
            self._check_release_grasp()
            identity = (self.generation, self.attempt_id, self._current_control_id)
            if not self._release_approval or self._release_approval['identity'] != identity:
                # Includes the controller's internal empty-GRASP reopening path.
                self.authorize_release(**self._control_context)
            self._check_release_grasp()
            decision = self._release_approval['decision']
            if decision.get('verdict') == 'HALF_INSERTED_HELD':
                # Same-generation physics continues during the provider request.
                # Re-measure observed RGB-D at dispatch; old depth cannot authorize
                # a release after withdrawal/slip, even if width still looks held.
                from core.sim.visual_insertion import estimate_insertion
                from core.sim.release_guard import ReleaseNotQualified
                observation = self.get_observation()
                measured = estimate_insertion(observation)
                self.artifacts.append('release-verification.jsonl', {
                    'event': 'release_dispatch_depth_checked', 'check_id': decision.get('check_id'),
                    'observation': observation['simulation_snapshot'], 'measurement': measured})
                if not measured['qualified']:
                    self._release_approval = None
                    raise ReleaseNotQualified('UNCERTAIN', 'Fresh dispatch RGB-D does not confirm 25 mm insertion',
                        check_id=decision.get('check_id'), observation=observation['simulation_snapshot'])
            self._check_release_grasp()
            self._release_approval = None
            self._observe_insertion(token='RELEASE', stage=self._control_context['stage'], boundary='before_gripper_open')
        try:
            result = super().control_gripper(close)
            if not close and self._release_guard is not None:
                self._refresh()
                self._observe_insertion(token='RELEASE', stage=self._control_context['stage'], boundary='after_gripper_open')
            return result
        except BaseException as exc:
            if not close and self.insertion_sequence is not None:
                self.insertion_sequence.cancel_release(exc)
            raise

    def update_desired_ee_pose(self, pose):
        self._refresh()
        return super().update_desired_ee_pose(pose)

    def set_audit_context(self, **fields):
        self.audit_context.update(fields)

    def record_initial_pose(self):
        self._check()
        self.runtime.record_initial_pose()

    def record_decision(self, **fields):
        self._check()
        with self.runtime._artifact_lock:
            if self.artifacts is not self.runtime._run_artifacts:
                raise ExecutionCancelled('Decision belongs to a previous scene')
            self.runtime._record_context = {'step': fields.get('step_idx'), 'stage': fields.get('stage', ''),
                'decision': fields.get('token', ''), 'executing': fields.get('token', ''),
                'source': fields.get('source', 'vlm'), 'annotation': fields.get('annotation', ''),
                'vlm_output': fields.get('output', ''), 'status': 'Executing'}
        self.artifacts.append('show-harness-decisions.jsonl', {'event': 'decision', **fields,
            'execution_generation': self.generation, 'attempt_id': self.attempt_id})

    def record_step_result(self, record):
        from core.record.episode_logger import status_flags
        with self.runtime._artifact_lock:
            if self.artifacts is self.runtime._run_artifacts:
                self.runtime._record_context['status'] = '; '.join(label for label, _ in status_flags(record)) or 'Action completed'
        self.artifacts.append('show-harness-decisions.jsonl', {'event': 'step_complete', 'result': record,
            'execution_generation': self.generation, 'attempt_id': self.attempt_id})

    def record_run_finished(self, success, end_reason):
        with self.runtime._artifact_lock:
            if self.artifacts is self.runtime._run_artifacts:
                self.runtime._record_context['status'] = f'{"SUCCESS" if success else "STOPPED"}: {end_reason}'
        self.artifacts.append('show-harness-decisions.jsonl', {'event': 'execution_finished',
            'success': success, 'end_reason': end_reason, 'execution_generation': self.generation,
            'attempt_id': self.attempt_id})

    def _observe_insertion(self, *, token=None, stage=None, boundary='observation'):
        if self.insertion_sequence is None:
            return None
        return self.insertion_sequence.observe(self.model, self.data, self.observation_identity,
            stage=stage or self.audit_context.get('stage', ''), token=token, boundary=boundary)

    def record_control_start(self, **fields):
        self._refresh()
        control_id = uuid4().hex
        self._current_control_id = control_id
        self._release_approval = None
        process = self._observe_insertion(token=fields.get('token'), stage=fields.get('stage'), boundary='before_control')
        self.artifacts.append('robot-trajectory.jsonl', {'event': 'control_started', 'control_id': control_id,
                              **fields, 'pose': self.runtime._robot_pose(), 'insertion_process': process})
        with self.runtime._artifact_lock:
            if self.artifacts is self.runtime._run_artifacts:
                self.runtime._record_context.update(executing=str(fields.get('token', '')), status='Executing')
        return control_id

    def record_control_end(self, control_id, **fields):
        # Do not generation-check: a stopped/interrupted control still needs its final measured pose.
        process = None
        try:
            self._refresh()
            process = self._observe_insertion(token=fields.get('token'), stage=fields.get('stage'), boundary='after_control')
        except ExecutionCancelled:
            pass  # The old robot boundary pose below remains valid; never sample a reset scene.
        self.artifacts.append('robot-trajectory.jsonl', {'event': 'control_finished', 'control_id': control_id,
                              **fields, 'execution_generation': self.generation, 'attempt_id': self.attempt_id,
                              'pose': self.runtime.final_control_pose(self.generation), 'insertion_process': process})

    def load_runner_checkpoint(self):
        return self.runtime.load_runner_checkpoint(self.generation, self.attempt_id)

    def save_runner_checkpoint(self, checkpoint):
        self.runtime.save_runner_checkpoint(self.generation, self.attempt_id, checkpoint)

    def _advance(self, seconds, target_ctrl=None):
        self._check()
        self.runtime.submit_target(self.generation, self.attempt_id, seconds, target_ctrl)
        self._refresh()


class _GuardedClient:
    """One synchronous request at a time, guarded before and after every VLM call."""
    def __init__(self, client, check):
        self._client, self._check = client, check

    def __getattr__(self, name):
        value = getattr(self._client, name)
        if callable(value) and (name.startswith('complete') or name == 'health_check'):
            def guarded(*args, **kwargs):
                self._check()
                result = value(*args, **kwargs)
                self._check()
                return result
            return guarded
        return value


class ServiceRuntime:
    def __init__(self, config_path=None, *, vlm_backend='openai', gui=False,
                 record=False, record_global=False, config=None, runner_factory=None, no_controller=False, run_root=None):
        self.config = deepcopy(config) if config is not None else load_yaml(
            config_path or ROOT / 'configs/robot_mujoco_plug.yaml')
        self.vlm_backend = vlm_backend
        self._session = MujocoSession(self.config, gui=False)
        from core.sim.continuous_physics import configure_continuous_physics
        self._contact_settings = configure_continuous_physics(self._session.model, self.config)
        self._session.model.vis.global_.offwidth = max(1920, 3 * (self._session.resolution + 32) - 32)
        self._session.model.vis.global_.offheight = max(1080, 3 * (self._session.resolution + 32) - 32)
        self.model = self._session.model
        self._initial = mujoco.MjData(self.model)
        mujoco.mj_copyData(self._initial, self.model, self._session.data)
        self._lock = threading.RLock()
        self._commands = deque()
        self._active = None
        self._generation = 0
        self._applied_generation = 0
        self._attempt_id = None
        self._runner_checkpoint = None
        self._identities = {}
        self._enabled = False
        self._hold_applied = True
        self._healthy = True
        self._physics_error = None
        self._closed = threading.Event()
        self._execution_wake = threading.Event()
        self._pending_execution = None
        self._runner_factory = runner_factory
        self._no_controller = no_controller
        self._state_seq = 0
        self._captured_ns = time.monotonic_ns()
        self._clock_wall = time.monotonic()
        self._clock_sim = float(self._session.data.time)
        self._real_time_factor = 1.0
        self._monitor_renderers = {}
        self._monitor_data = None
        self._render_owner = None
        self._actions = deque(maxlen=100)
        self._boundary_poses = {}
        self._events = queue.Queue(maxsize=4096)
        self._event_seq = 0
        self._events_dropped = 0
        self._telemetry_error = None
        self._telemetry_thread = threading.Thread(target=self._telemetry_loop, name='mujoco-telemetry', daemon=True)
        self._record_error = None
        self._gui_error = None
        self._recorded_frames = 0
        self._recording_gaps = 0
        self._artifact_lock = threading.RLock()
        self._run_root = Path(run_root or self.config.get('run_root', Path.cwd() / '.robobrain' / 'runs')).resolve()
        self._run_artifacts = None
        self._run_info = None
        self._run_directories = {}
        self._vlm_artifacts = {}
        self._run_min_generation = 0
        self._record_enabled = bool(record or record_global)
        self._record_context = {}
        self._initial_pose_recorded = False
        self._new_artifact_run(0)
        self._physics = threading.Thread(target=self._physics_loop, name='mujoco-physics', daemon=True)
        self._executor = threading.Thread(target=self._execution_loop, name='mujoco-executor', daemon=True)
        self._physics.start()
        self._executor.start()
        self._telemetry_thread.start()
        self._record_thread = self._gui_thread = None
        self._record_thread = threading.Thread(target=self._record_loop, args=(True,),
                                               name='mujoco-record', daemon=True)
        self._record_thread.start()
        if gui:
            self._gui_thread = threading.Thread(target=self._gui_loop, name='mujoco-gui', daemon=True)
            self._gui_thread.start()

    def _new_artifact_run(self, generation):
        from core.record.run_artifacts import RunArtifacts
        with self._artifact_lock:
            if self._run_artifacts:
                self._run_artifacts.append('lifecycle.jsonl', {'event': 'run_closed', 'reason': 'scene_reset'})
            run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '-' + uuid4().hex[:8]
            directory = self._run_root / run_id
            self._run_artifacts = RunArtifacts(directory)
            from core.sim.plug_sequence import PlugInsertionSequence
            self._plug_sequence = (PlugInsertionSequence(self._run_artifacts,
                                  empty_width_m=self.config.get('empty_width_m', .005))
                                  if self.config.get('scene') == 'plug_insert' else None)
            eligible = [g for g in self._identities if g <= generation]
            identity = self._identities[max(eligible)] if eligible else {}
            self._run_info = {**identity, 'run_id': run_id, 'directory': str(directory), 'minimum_generation': generation,
                              'video_enabled': self._record_enabled, 'created_at': datetime.now(timezone.utc).isoformat()}
            self._run_min_generation = generation
            self._run_directories[generation] = str(directory)
            self._initial_pose_recorded = False
            self._record_context = {}
            self._run_artifacts.write_json('run.json', {**self._run_info, 'scene': self.config.get('scene'),
                'show_harness_video': 'show-harness-video', 'robot_trajectory': 'robot-trajectory.jsonl',
                'show_harness_vlm': 'show-harness-vlm', 'timing_mode': 'continuous',
                'global_camera': 'lab_overview', 'global_resolution': [1920, 1080],
                'continuous_physics': self._contact_settings,
                'plug_insertion_process': ({'minimum_held_insertion_m': PLUG_MIN_INSERTION_M,
                    'audit': 'plug-insertion-sequence.jsonl', 'ground_truth_used_for': 'evaluation_only'}
                    if self._plug_sequence is not None else None),
                'trajectory_schema': {'tcp_frame': 'world', 'tcp_pose': ['x_m', 'y_m', 'z_m', 'qx', 'qy', 'qz', 'qw'],
                    'arm_joint_order': [f'joint{i}' for i in range(1, 8)], 'arm_qpos_units': 'radians',
                    'gripper_position_units': 'metres'}})
            return dict(self._run_info)

    def recording_run(self, enable=False):
        with self._artifact_lock:
            if enable:
                self._record_enabled = True
                self._run_info['video_enabled'] = True
                self._run_artifacts.append('lifecycle.jsonl', {'event': 'video_recording_enabled'})
                manifest = json.loads((self._run_artifacts.directory / 'run.json').read_text())
                self._run_artifacts.write_json('run.json', {**manifest, **self._run_info})
            return dict(self._run_info)

    def _robot_pose(self):
        # Short, allocation-light measured snapshot, also safe on the physics owner.
        with self._lock:
            s = self._session
            eligible = [g for g in self._identities if g <= self._applied_generation]
            identity = self._identities[max(eligible)] if eligible else {}
            return {**identity, 'attempt_id': self._attempt_id, 'execution_generation': self._applied_generation,
                    'state_seq': self._state_seq, 'sim_time_s': float(s.data.time),
                    'captured_at_server_monotonic_ns': time.monotonic_ns(),
                    'arm_qpos_rad': s.data.qpos[s._qpos].tolist(),
                    'tcp_pose': s.get_ee_pose().tolist(), 'tcp_pose_units': 'xyz_metres_quaternion_xyzw',
                    'gripper_position_m': s.get_gripper_position().tolist(),
                    'actuator_ctrl': s.data.ctrl.tolist()}

    def final_control_pose(self, generation):
        with self._lock:
            if generation != self._applied_generation:
                pose = self._boundary_poses.get(generation)
                if pose is not None:
                    return {**deepcopy(pose), 'captured_at_execution_boundary': True}
                return {'execution_generation': generation, 'unavailable': 'Execution boundary pose expired'}
            return self._robot_pose()

    def record_initial_pose(self):
        with self._artifact_lock:
            if not self._initial_pose_recorded:
                self._run_artifacts.append('robot-trajectory.jsonl', {'event': 'initial_pose', 'pose': self._robot_pose()})
                self._initial_pose_recorded = True

    def load_runner_checkpoint(self, generation, attempt_id):
        with self._lock:
            self.check_execution(generation, attempt_id)
            return deepcopy(self._runner_checkpoint)

    def save_runner_checkpoint(self, generation, attempt_id, checkpoint):
        with self._lock:
            self.check_execution(generation, attempt_id)
            self._runner_checkpoint = deepcopy(checkpoint)

    def check_execution(self, generation, attempt_id):
        with self._lock:
            if (self._closed.is_set() or not self._healthy or not self._enabled
                    or generation != self._generation or attempt_id != self._attempt_id):
                raise ExecutionCancelled('Execution superseded or stopped')

    def _transition(self, kind, generation, attempt_id=None, expected_generation=None):
        command = _Command(kind, generation, attempt_id)
        with self._lock:
            if self._closed.is_set() or not self._healthy:
                raise RuntimeError(self._physics_error or 'Physics unavailable')
            if expected_generation is not None and expected_generation != self._generation:
                raise ExecutionCancelled("Transition superseded")
            if generation < self._generation:
                raise ExecutionCancelled('Old transition generation')
            self._generation = generation
            self._enabled = False
            # Urgent transitions are processed before trajectories on the next tick.
            self._commands.appendleft(command)
        if not command.applied.wait(5):
            raise TimeoutError('Physics did not acknowledge transition within 5 seconds')
        if command.error:
            raise command.error
        return command

    def start_execution(self, generation, attempt_id, on_finished: Callable):
        self.record_initial_pose()
        self._transition('start', generation, attempt_id)
        with self._lock:
            self._pending_execution = (generation, attempt_id, on_finished)
        self._execution_wake.set()

    def stop_execution(self, generation):
        self._transition('stop', generation)
        with self._lock:
            self._pending_execution = None

    def reset(self, generation):
        self._new_artifact_run(generation)
        self._transition('reset', generation)
        self.record_initial_pose()
        with self._lock:
            self._pending_execution = None

    def submit_target(self, generation, attempt_id, seconds, target):
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError('Action duration must be finite and positive')
        if target is not None:
            target = np.array(target, dtype=float, copy=True)
            if target.shape != (self.model.nu,) or not np.isfinite(target).all():
                raise ValueError('Actuator target shape or values invalid')
        command = _Command('target', generation, attempt_id, float(seconds), target)
        with self._lock:
            self.check_execution(generation, attempt_id)
            self._commands.append(command)
        while not command.applied.wait(0.05):
            self.check_execution(generation, attempt_id)
        if command.error:
            raise command.error
        self.check_execution(generation, attempt_id)

    def _cancel_active(self):
        if self._active is not None:
            command = self._active
            self._record_action({'generation': command.generation, 'attempt_id': command.attempt_id, 'interrupted': True,
                                  'start_ctrl': command.start_ctrl.tolist(),
                                  'target_ctrl': command.target.tolist() if command.target is not None else None,
                                  'started_sim_time_s': command.started,
                                  'ended_sim_time_s': float(self._session.data.time),
                                  'actual_arm_qpos_rad': self._session.data.qpos[self._session._qpos].tolist()})
            command.error = ExecutionCancelled('Active trajectory interrupted by hold')
            command.applied.set()
            self._active = None

    def _record_action(self, event):
        self._actions.append(event)
        self._emit_physics_event({'kind': 'trajectory', **event})

    def _emit_physics_event(self, event):
        # Called by the sole physics owner under its state lock. No disk I/O here.
        generation = event.get('generation', self._applied_generation)
        eligible = [g for g in self._identities if g <= generation]
        identity = self._identities[max(eligible)] if eligible else {}
        self._event_seq += 1
        item = {**identity, 'attempt_id': self._attempt_id, 'event_seq': self._event_seq,
                'state_seq': self._state_seq, 'sim_time_s': float(self._session.data.time),
                'server_monotonic_ns': time.monotonic_ns(), **event,
                '_run_directory': self._run_directories[max(g for g in self._run_directories if g <= generation)]}
        try:
            self._events.put_nowait(item)
        except queue.Full:
            self._events_dropped += 1

    def _telemetry_loop(self):
        try:
            while not self._closed.is_set() or not self._events.empty():
                try:
                    event = self._events.get(timeout=.1)
                except queue.Empty:
                    continue
                directory = Path(event.pop('_run_directory'))
                with (directory / 'physics-events.jsonl').open('a') as stream:
                    stream.write(json.dumps(event) + '\n')
        except Exception as exc:
            self._telemetry_error = str(exc)

    def _install_hold(self):
        self._cancel_active()
        s = self._session
        # Preserve gripper ctrl exactly, including a partly completed open/close ramp.
        s.data.ctrl[s._actuators] = s.data.qpos[s._qpos]
        self._hold_applied = True

    def _physics_loop(self):
        dt = float(self.model.opt.timestep)
        deadline = time.monotonic()
        try:
            while not self._closed.is_set():
                with self._lock:
                    while self._commands:
                        command = self._commands[0]
                        if command.kind == 'target' and self._active is not None:
                            break
                        self._commands.popleft()
                        if command.generation != self._generation:
                            command.error = ExecutionCancelled('Obsolete queued command')
                            command.applied.set()
                            continue
                        if command.kind != 'target':
                            self._boundary_poses[self._applied_generation] = self._robot_pose()
                            self._boundary_poses = {g: self._boundary_poses[g] for g in sorted(self._boundary_poses)[-32:]}
                            self._install_hold()
                            if command.kind == 'reset':
                                mujoco.mj_copyData(self._session.data, self.model, self._initial)
                                self._attempt_id = None
                                self._actions.clear()
                                self._runner_checkpoint = None
                                self._clock_wall = time.monotonic()
                                self._clock_sim = float(self._session.data.time)
                            if command.kind == 'start':
                                self._attempt_id = command.attempt_id
                                self._enabled = True
                            self._applied_generation = command.generation
                            self._emit_physics_event({'kind': command.kind, 'generation': command.generation,
                                                      'ctrl': self._session.data.ctrl.tolist()})
                            command.applied.set()
                        elif self._enabled and command.attempt_id == self._attempt_id:
                            command.started = float(self._session.data.time)
                            command.start_ctrl = self._session.data.ctrl.copy()
                            self._active = command
                            self._hold_applied = False
                        else:
                            command.error = ExecutionCancelled('Execution no longer enabled')
                            command.applied.set()
                    s = self._session
                    if self._active:
                        action = self._active
                        if action.generation != self._generation or not self._enabled:
                            self._install_hold()
                        else:
                            fraction = min(1., (float(s.data.time) + dt - action.started) / action.seconds)
                            if action.target is not None:
                                blend = fraction**3 * (10 + fraction * (-15 + 6 * fraction))
                                s.data.ctrl[:] = action.start_ctrl + blend * (action.target - action.start_ctrl)
                    s.data.qfrc_applied[s._dofs] = s.data.qfrc_bias[s._dofs]
                    mujoco.mj_step(self.model, s.data)
                    mujoco.mj_kinematics(self.model, s.data)
                    mujoco.mj_camlight(self.model, s.data)
                    if not np.isfinite(s.data.qpos).all():
                        raise RuntimeError('MuJoCo state became non-finite')
                    if self._active and float(s.data.time) >= self._active.started + self._active.seconds:
                        action = self._active
                        self._record_action({'generation': action.generation, 'attempt_id': action.attempt_id, 'interrupted': False,
                                              'start_ctrl': action.start_ctrl.tolist(),
                                              'target_ctrl': action.target.tolist() if action.target is not None else None,
                                              'started_sim_time_s': action.started,
                                              'ended_sim_time_s': float(s.data.time),
                                              'actual_arm_qpos_rad': s.data.qpos[s._qpos].tolist()})
                        action.applied.set()
                        self._active = None
                    self._state_seq += 1
                    self._captured_ns = time.monotonic_ns()
                    elapsed = time.monotonic() - self._clock_wall
                    if elapsed > 0.1:
                        self._real_time_factor = (float(s.data.time) - self._clock_sim) / elapsed
                deadline += dt
                now = time.monotonic()
                # Preserve timestep; bound catch-up so commands cannot starve.
                if deadline < now - 0.05:
                    deadline = now
                self._closed.wait(max(0, deadline - now))
        except BaseException as exc:
            with self._lock:
                self._healthy = False
                self._physics_error = f'{type(exc).__name__}: {exc}'
                self._enabled = False
                self._cancel_active()
                for command in self._commands:
                    command.error = RuntimeError(self._physics_error)
                    command.applied.set()
                self._commands.clear()

    def set_identity(self, server_instance_id, episode_id, generation=0):
        """Bind episode identity to the physics generation that actually applies it."""
        with self._lock:
            self._identities[int(generation)] = {"server_instance_id": server_instance_id, "episode_id": episode_id}
            self._identities = {g: self._identities[g] for g in sorted(self._identities)[-16:]}
        if generation == 0:
            with self._artifact_lock:
                self._run_info.update(server_instance_id=server_instance_id, episode_id=episode_id)
                manifest = json.loads((self._run_artifacts.directory / 'run.json').read_text())
                self._run_artifacts.write_json('run.json', {**manifest, **self._run_info})
            self.record_initial_pose()

    def snapshot(self, destination=None):
        """Copy all state under a short lock; render/forward only on the private copy."""
        if destination is None:
            destination = mujoco.MjData(self.model)
        with self._lock:
            mujoco.mj_copyData(destination, self.model, self._session.data)
            eligible = [g for g in self._identities if g <= self._applied_generation]
            identity = self._identities[max(eligible)] if eligible else {}
            return {**identity, 'attempt_id': self._attempt_id,
                    'state_seq': self._state_seq, 'sim_time_s': float(destination.time),
                    'captured_at_server_monotonic_ns': self._captured_ns,
                    'execution_generation': self._applied_generation}

    def status(self):
        with self._lock:
            return {'physics_healthy': self._healthy, 'hold_applied': self._hold_applied,
                    'state_seq': self._state_seq, 'sim_time_s': float(self._session.data.time),
                    'real_time_factor': self._real_time_factor,
                    'physics_error': self._physics_error, 'recording_error': self._record_error, 'gui_error': self._gui_error,
                    'telemetry_error': self._telemetry_error, 'physics_events_dropped': self._events_dropped,
                    'recorded_frames': self._recorded_frames, 'recording_gaps': self._recording_gaps,
                    'execution_generation': self._applied_generation, 'recording_run': dict(self._run_info)}

    def render_bundle(self, views, encoding='jpeg', include_global=False):
        owner = threading.get_ident()
        if self._render_owner is not None and self._render_owner != owner:
            raise RuntimeError('Monitor rendering must stay on its dedicated context thread')
        self._render_owner = owner
        if self._monitor_data is None:
            self._monitor_data = mujoco.MjData(self.model)
        meta = self.snapshot(self._monitor_data)
        views = list(views)
        if include_global:
            views.append({'view_id': '__global__', 'width': 1920, 'height': 1080,
                          'camera': {'fixed': 'lab_overview'}})
        used_sizes = {(int(v['width']), int(v['height'])) for v in views}
        for size in list(self._monitor_renderers):
            if size not in used_sizes:
                self._monitor_renderers.pop(size).close()
        images = []
        for view in views:
            width, height = int(view['width']), int(view['height'])
            renderer = self._monitor_renderers.get((width, height))
            if renderer is None:
                renderer = self._monitor_renderers[(width, height)] = mujoco.Renderer(
                    self.model, height=height, width=width)
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            pose = view['camera']
            if pose.get('fixed'):
                camera = pose['fixed']
            else:
                camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                camera.lookat[:] = pose['lookat_m']
                camera.distance = float(pose['distance_m'])
                camera.azimuth = float(pose['azimuth_deg'])
                camera.elevation = float(pose['elevation_deg'])
            renderer.update_scene(self._monitor_data, camera=camera)
            rgb = renderer.render()
            buffer = BytesIO()
            fmt = 'PNG' if encoding.lower() == 'png' else 'JPEG'
            Image.fromarray(rgb).save(buffer, format=fmt, **({'quality': 85} if fmt == 'JPEG' else {}))
            images.append({'view_id': view['view_id'], 'width': width, 'height': height,
                           'data': buffer.getvalue(), 'projection': {'type': 'perspective',
                             'vertical_fov_deg': float(self.model.camera(pose['fixed']).fovy[0] if pose.get('fixed') else self.model.vis.global_.fovy)}})
        return {**meta, 'images': images}

    def close_renderer(self):
        """Call on the render owner thread before stopping its executor."""
        for renderer in self._monitor_renderers.values():
            renderer.close()
        self._monitor_renderers.clear()
        self._render_owner = None

    def _execution_loop(self):
        while not self._closed.is_set():
            self._execution_wake.wait(0.1)
            self._execution_wake.clear()
            with self._lock:
                pending, self._pending_execution = self._pending_execution, None
            if pending is None:
                continue
            generation, attempt, callback = pending
            session = None
            error = None
            try:
                self.check_execution(generation, attempt)
                session = _SnapshotSession(self, generation, attempt)
                if self._no_controller:
                    while not self._closed.wait(0.05):
                        self.check_execution(generation, attempt)
                elif self._runner_factory is not None:
                    self._runner_factory(session, generation, attempt)
                else:
                    self._run_episode(session, generation, attempt)
            except ExecutionCancelled:
                continue
            except BaseException as exc:
                error = f'{type(exc).__name__}: {exc}'
            finally:
                if session is not None:
                    session.close()
            with self._lock:
                current = generation == self._generation and self._enabled and not self._closed.is_set()
            if current:
                # Hold at completion/error too; callback publishes the lifecycle transition.
                try:
                    self._transition('stop', generation + 1, expected_generation=generation)
                except ExecutionCancelled:
                    continue
                callback(error)

    def _run_episode(self, session, generation, attempt):
        from core.launch import make_runner
        from core.prompting.prompt_loader import load_prompt_dir
        from core.record.episode_logger import EpisodeLogger
        from core.sim.launch import make_vlm_client
        from core.sim.mujoco_success import plug_success_prompt
        from scripts.run_mujoco import make_controller, render_plug_retreat_text
        cfg = deepcopy(self.config)
        cfg['vlm'] = resolve_vlm_config(cfg, backend=self.vlm_backend)
        cfg['vlm_backend'] = cfg['vlm']['backend']
        # One persistent executor serializes requests across stop/resume. No abandoned
        # workers accumulate; a resumed request waits for the old bounded call to exit.
        cfg['vlm']['timeout_s'] = min(120., float(cfg['vlm'].get('timeout_s', 120)))
        cfg['vlm']['max_retries'] = 0
        if cfg['vlm'].get('api_key') in (None, '', 'EMPTY'):
            raise RuntimeError(f"Set {cfg['vlm'].get('api_key_env', 'OPENAI_API_KEY')} to start VLM execution")
        if cfg.get('scene') == 'plug_insert':
            cfg['task'] = render_plug_retreat_text(cfg['task'], cfg)
        with self._lock:
            previous = list(self._actions)[-12:]
        if previous:
            cfg['task'] += ('\nResume this same interrupted attempt from CURRENT observations. '
                            'Preserve an existing grasp. Do not restart completed lift/transport '
                            'milestones or release merely because planning restarted. Recent '
                            'physical action timing (interrupted means PARTIAL, not completed): '
                            + json.dumps(previous))
        controller = make_controller(session, cfg)
        client = make_vlm_client(None, cfg)
        from core.record.run_artifacts import AuditedClient, RunArtifacts
        artifacts = session.artifacts
        if artifacts.directory not in self._vlm_artifacts:
            self._vlm_artifacts[artifacts.directory] = RunArtifacts(artifacts.directory / 'show-harness-vlm')
        vlm_artifacts = self._vlm_artifacts[artifacts.directory]
        vlm_artifacts.add_secrets(cfg['vlm'].get('api_key', ''))
        guarded = AuditedClient(client, vlm_artifacts,
                                context=lambda: {**session.audit_context, 'attempt_id': attempt,
                                                 'execution_generation': generation,
                                                 'observation': session.audit_context.get('observation', session.observation_identity)},
                                check=lambda: self.check_execution(generation, attempt))
        log_root = session.artifacts.directory / 'show-harness-logs'
        if cfg.get('scene') == 'plug_insert':
            from core.sim.release_guard import VisualReleaseGuard
            session._release_guard = VisualReleaseGuard(guarded, artifacts, max_tokens=4096)
        logger = EpisodeLogger(log_root / attempt / f'generation_{generation}', task_id=0,
                               variant=cfg['vlm_backend'], video_fps=2,
                               record_video=False, primary_camera='front')
        metadata_cfg = deepcopy(cfg)
        metadata_cfg['vlm']['api_key'] = '[redacted]'
        logger.write_metadata({'config': metadata_cfg, 'simulator': 'mujoco', 'timing_mode': 'continuous',
                               'attempt_id': attempt, 'execution_generation': generation})
        prompts = load_prompt_dir(ROOT / 'prompts')
        name = ('controller_mujoco_plug_prompt' if cfg.get('scene') == 'plug_insert' else
                'controller_mujoco_cartesian_prompt' if getattr(controller, 'cartesian_actions', None)
                else 'controller_mujoco_prompt')
        prompts['controller_prompt'] = render_plug_retreat_text(prompts[name], cfg)
        prompts[f'controller_{cfg["vlm_backend"]}_prompt'] = prompts['controller_prompt']
        context = cfg.get('cartesian_observation_context', cfg.get('observation_context', ''))
        prompts['common_context'] += '\n' + render_plug_retreat_text(context, cfg)
        if cfg.get('scene') == 'plug_insert':
            prompts['common_context'] += '\n' + plug_success_prompt(cfg)
        prompts['common_context'] += ('\nPhysics continues during reasoning. CURRENT images identify '
                                     'one coherent past snapshot. Recheck measured pose before actions; '
                                     'do not assume an object remained stationary during reasoning.')
        try:
            runner = make_runner(cfg, prompts, guarded, session, controller, logger, False)
            runner.run()
        finally:
            guarded.close()
            client.session.close()

    def _record_loop(self, global_video):
        from core.record.continuous_recorder import ContinuousRecorder
        session = recorder = None
        active_directory = None
        try:
            session = _SnapshotSession(self)
            fps = min(60., max(1., float(self.config.get('recording', {}).get('fps', 30))))
            last_sim = None
            while not self._closed.is_set():
                wall = time.monotonic()
                with self._artifact_lock:
                    directory = self._run_info['directory']
                    enabled, minimum = self._record_enabled, self._run_min_generation
                    context = dict(self._record_context)
                if directory != active_directory:
                    if recorder:
                        recorder.close()
                    recorder, active_directory, last_sim = None, directory, None
                if not enabled:
                    self._closed.wait(.1)
                    continue
                session._refresh()
                identity = session.observation_identity
                with self._artifact_lock:
                    if directory != self._run_info['directory'] or minimum != self._run_min_generation:
                        continue
                if not identity.get('episode_id') or identity['execution_generation'] < minimum:
                    self._closed.wait(.01)
                    continue
                session._reading = True
                try:
                    if recorder is None:
                        recorder = ContinuousRecorder(session, Path(directory) / 'show-harness-video', fps=fps,
                            model=resolve_vlm_config(self.config, backend=self.vlm_backend).get('model', ''),
                            task=self.config.get('task', ''), global_video=global_video)
                    sim_time = identity['sim_time_s']
                    gap = 0 if last_sim is None else max(0, round((sim_time-last_sim)*fps)-1)
                    last_sim = sim_time
                    self._recording_gaps += gap
                    if recorder.capture({**identity, 'dropped_intervals': gap}, context=context or None):
                        self._recorded_frames += 1
                finally:
                    session._reading = False
                self._closed.wait(max(0, 1/fps - (time.monotonic()-wall)))
        except BaseException as exc:
            self._record_error = f'{type(exc).__name__}: {exc}'
        finally:
            try:
                if recorder:
                    recorder.close()
            except BaseException as exc:
                self._record_error = f'Recording finalization failed: {exc}'
            finally:
                if session:
                    session.close()

    def _gui_loop(self):
        from core.sim.viewer_process import run_viewer
        try:
            run_viewer(self)
        except BaseException as exc:
            self._gui_error = f'{type(exc).__name__}: {exc}'

    def close(self):
        if self._closed.is_set(): return
        if self._healthy:
            self.stop_execution(self._generation + 1)
        self._closed.set()
        self._execution_wake.set()
        self._physics.join(timeout=5)
        self._telemetry_thread.join(timeout=5)
        for thread in (self._record_thread, self._gui_thread):
            if thread is not None:
                thread.join(timeout=30)
                if thread.is_alive():
                    if thread is self._gui_thread:
                        self._gui_error = 'Viewer finalization timed out'
                    else:
                        self._record_error = f'{thread.name} finalization timed out; outputs may be incomplete'
        self._run_artifacts.append('lifecycle.jsonl', {'event': 'service_closed', 'recording_error': self._record_error})
        # A paid HTTP request may still be returning; its daemon executor is guarded
        # and owns its private context. It cannot touch live physics after close.
        self._executor.join(timeout=0.2)
