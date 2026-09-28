"""Fresh robot telemetry blocks obsolete plug actions; no provider or renderer calls."""
from types import SimpleNamespace

import numpy as np
import pytest

from core.runners.real import GraspLost, RealEpisodeRunner
from core.sim.service_runtime import ExecutionCancelled
from core.v0_types import Subgoal
from core.vlm.vlm_client import VLMResponse
from plugins.recovery import RecoveryPlugin
from test_runner_checkpoint import Session, Planner, runner


class MeasuredSession(Session):
    def __init__(self):
        super().__init__()
        self.width = .032
        self.audit = []
    def get_gripper_position(self):
        return np.array([self.width])
    def get_observation(self):
        return super().get_observation() | {'gripper_width': self.width}
    def record_control_start(self, **fields):
        self.audit.append(('start', fields))
        return str(len(self.audit))
    def record_control_end(self, control_id, **fields):
        self.audit.append(('end', fields | {'control_id': control_id}))


@pytest.mark.parametrize('late_token', ['MV_DOWN_SMALL', 'RELEASE', 'DONE'])
def test_grasp_lost_during_provider_wait_rejects_pending_action(tmp_path, late_token):
    session = MeasuredSession()
    def decide(ctx, **kwargs):
        if ctx.subgoal.motion == 'GRASP':
            return VLMResponse('DONE', 'Visually confirmed grip', {})
        assert ctx.obs['gripper_width'] == .032
        session.width = 0.0  # Passive loss after image capture, during a model wait.
        return VLMResponse(late_token, 'Reply based on the old image', {})
    instance, controller = runner(session, Planner(), decide, tmp_path / 'run')
    with pytest.raises(GraspLost) as caught:
        instance.run()
    assert controller.actions == ['DONE'], 'A stale model action reached the controller'
    assert caught.value.execution_cancelled is False
    assert caught.value.details['measured_width_m'] == 0
    assert caught.value.details['source'] == 'vlm'
    final = session.audit[-1][1]
    assert final['status'] == 'failed' and final['guard'] == 'grasp_lost'
    assert final['guard_evidence']['empty_width_m'] == .005
    assert session.checkpoint['carry_established'] is True


def simple_runner(width=.032, *, closed=True, carried=True, scene='plug_insert'):
    session = MeasuredSession()
    session.width, session.scene = width, scene
    calls = []
    controller = SimpleNamespace(gripper_closed=closed, grasp_min_width_m=.005,
        gripper_close_threshold_m=.065,
        step=lambda token, **kw: calls.append(token) or SimpleNamespace(token=token, note=''))
    instance = RealEpisodeRunner.__new__(RealEpisodeRunner)
    instance.session, instance.controller = session, controller
    instance.recovery_plugin = RecoveryPlugin(empty_width_m=.005, open_width_m=.065)
    instance._carry_established = carried
    return instance, session, calls


def test_post_action_recovery_uses_fresh_width_and_cannot_auto_release():
    instance, session, calls = simple_runner(width=.001)
    subgoals = [Subgoal(stage, 'plug', 'handle', stage, 'Move', 'Observe') for stage in ('GRASP', 'INSERT')]
    result = SimpleNamespace(gripper_closed=True, grasp_empty=False)
    recovery = instance._recovery_after_step(token='MV_DOWN_SMALL', result=result, current_index=1,
        subgoals=subgoals, obs={'gripper_width': .032}, subgoal_done=False)
    assert recovery.event == 'lost_grasp' and recovery.release
    with pytest.raises(GraspLost) as caught:
        instance._control_step('RELEASE', step_idx=64, stage='INSERT', source='recovery')
    assert caught.value.details['source'] == 'recovery' and not calls
    # Conversely, a stale empty reading must not reopen a currently held object.
    session.width = .032
    assert instance._recovery_after_step(token='GRASP', result=result, current_index=0,
        subgoals=subgoals, obs={'gripper_width': 0.0}, subgoal_done=True) is None


def test_guard_preserves_initial_regrasp_held_release_and_open_retreat():
    instance, session, calls = simple_runner(width=0., carried=False)
    instance._control_step('RELEASE', step_idx=0, stage='GRASP', source='recovery')
    assert calls == ['RELEASE'], 'Initial source-stand empty-grasp recovery was changed'
    session.width = .032
    # Visual half-depth readiness is a separate condition; this telemetry check
    # must permit its RELEASE when the gripper still measurably holds the handle.
    instance._control_step('RELEASE', step_idx=30, stage='RELEASE', source='vlm')
    assert calls[-1] == 'RELEASE' and instance._carry_established
    instance.controller.gripper_closed = False
    session.width = .08
    instance._control_step('MV_UP_LARGE', step_idx=31, stage='RETREAT', source='vlm')
    assert calls[-1] == 'MV_UP_LARGE'
    instance.session.scene = 'pick_place'
    instance.controller.gripper_closed = True
    session.width = 0.
    instance._control_step('DONE', step_idx=32, stage='MOVE', source='vlm')
    assert calls[-1] == 'DONE', 'Plug-specific protection changed other tasks'


def test_resume_after_grasp_rollback_keeps_carry_history(tmp_path):
    session, planner = MeasuredSession(), Planner()
    def before_stop(ctx, **kwargs):
        if ctx.step_idx == 2:
            raise ExecutionCancelled('operator stopped')
        return VLMResponse('DONE' if ctx.subgoal.motion == 'GRASP' else 'MV_UP_LARGE', '', {})
    instance, _ = runner(session, planner, before_stop, tmp_path / 'first')
    with pytest.raises(ExecutionCancelled):
        instance.run()
    assert session.checkpoint['carry_established'] is True
    session.checkpoint['current_index'] = 0  # A recovery rewind must not erase physical history.
    session.width = 0.
    resumed, controller = runner(session, planner,
        lambda **kw: VLMResponse('RELEASE', 'retry grasp', {}), tmp_path / 'resumed')
    with pytest.raises(GraspLost):
        resumed.run()
    assert controller.actions == [] and planner.calls == 1


@pytest.mark.parametrize('width,expected', [(0., 'EMPTY_CLOSED'), (.005, 'EMPTY_CLOSED'),
    (.032, 'CLOSED'), (.08, 'OPEN'), (float('nan'), 'UNKNOWN'), (float('inf'), 'UNKNOWN'), (None, 'UNKNOWN')])
def test_prompt_gripper_state_does_not_call_empty_grip_holding(width, expected):
    instance, _, _ = simple_runner()
    assert instance._observed_gripper_state({'gripper_width': width}) == expected
    assert instance._observed_gripper_state({}) == 'UNKNOWN'
