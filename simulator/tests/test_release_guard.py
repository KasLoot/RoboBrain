"""Release protocol checks with controlled visual-model replies, never paid calls.

The 18/30 markers select mock visual judgments; these tests verify authorization
and dispatch plumbing, not the accuracy of any real model's depth estimates.
"""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from core.record.run_artifacts import AuditedClient, RunArtifacts
from core.runners.real import GraspLost
from core.sim.mujoco_depth import depth_to_text
from core.sim.release_guard import ReleaseNotQualified, VisualReleaseGuard
from core.vlm.vlm_client import VLMResponse
from test_runner_grasp_safeguard import simple_runner


def observation(marker=30):
    image = np.empty((64, 64, 3), dtype=np.uint8)
    image[:] = [20, 60, 200]
    image[16:48, 16:48] = [220, 85, 20]
    image[0, 0, 0] = marker  # Mock model category marker, outside eroded measurement surfaces.
    calibration = {'near_m': 0., 'far_m': .3, 'tcp_depth_m': .1, 'representation': 'text'}
    depth = np.full((64, 64), .1 + .082 - marker / 1000.)
    depth[16:48, 16:48] = .1
    text = depth_to_text(depth, calibration, grid_rows=8, grid_cols=8)
    return {'agentview': image.copy(), 'wrist': image.copy(), 'primary_camera': 'Front', 'wrist_first': True,
        'extra_views': {'side': image.copy(), 'wrist_insert': image.copy(), 'wrist_depth': image.copy()},
        'wrist_depth_text': text, 'wrist_depth_calibration': calibration, 'wrist_depth_m': depth,
        'wrist_camera_axes': {'sightline': [0, 0, -1]},
        'simulation_snapshot': {'server_instance_id': 'server', 'episode_id': 'episode', 'attempt_id': 'attempt',
                                'execution_generation': 7, 'state_seq': marker},
        'forbidden_object_ground_truth': 'DO_NOT_FEED_THIS_TO_THE_MODEL',
        'success_diagnostics': 'DO_NOT_FEED_THIS_TO_THE_MODEL'}


class VisualClient:
    model = 'gpt-6-sol'
    max_tokens = 8192
    def __init__(self, session, *, verdict=None, on_call=None, error=None):
        self.session, self.verdict, self.on_call, self.error = session, verdict, on_call, error
        self.calls = []
    def complete_json(self, prompt, agentview_image, wrist_image=None, **kwargs):
        self.calls.append({'prompt': prompt, 'primary': agentview_image.copy(), 'extra': wrist_image,
                           'context': deepcopy(self.session.audit_context), 'kwargs': kwargs})
        if self.error:
            raise self.error
        if self.on_call:
            self.on_call()
        marker = int(agentview_image[0, 0, 0])
        verdict = self.verdict or ('HALF_INSERTED_HELD' if marker >= 25 else 'UNSAFE')
        result = {'verdict': verdict, 'evidence': f'Test visual judgment: {marker} mm inside the bore; handle visibly held.'}
        return VLMResponse('', json.dumps(result), {'json': result})


def session_for(marker):
    class Session:
        def __init__(self):
            self.audit_context = {'role': 'controller', 'phase': 'control_decision', 'observation': {'state_seq': -1}}
            self.reads = 0
        def get_observation(self):
            self.reads += 1
            return observation(marker)
    return Session()


@pytest.mark.parametrize('marker,approved', [(18, False), (30, True)])
def test_visual_depth_category_controls_release_and_context_restores(tmp_path, marker, approved):
    session = session_for(marker)
    original = deepcopy(session.audit_context)
    artifacts = RunArtifacts(tmp_path)
    raw = VisualClient(session)
    client = AuditedClient(raw, artifacts, context=lambda: session.audit_context)
    guard = VisualReleaseGuard(client, artifacts)
    if approved:
        result = guard(session, stage='RELEASE', source='vlm', carry_established=True)
        assert result['verdict'] == 'HALF_INSERTED_HELD'
        assert result['observation']['state_seq'] == marker
    else:
        with pytest.raises(ReleaseNotQualified) as caught:
            guard(session, stage='RELEASE', source='vlm', carry_established=True)
        assert caught.value.details['verdict'] == 'UNSAFE'
    assert session.reads == 1 and session.audit_context == original
    assert len(raw.calls) == 1 and len(raw.calls[0]['extra']) == 3
    assert raw.calls[0]['kwargs']['max_tokens'] == 8192
    assert raw.calls[0]['context']['role'] == 'release_verification'
    assert raw.calls[0]['context']['observation']['state_seq'] == marker
    assert 'DO_NOT_FEED_THIS_TO_THE_MODEL' not in raw.calls[0]['prompt']
    requests = [json.loads(line) for line in (tmp_path / 'vlm.jsonl').read_text().splitlines()]
    assert [r['event'] for r in requests] == ['vlm_call_start', 'vlm_call_end']
    assert all(r['role'] == 'release_verification' for r in requests)
    assert len(requests[0]['images']['wrist_image']) == 3
    assert 'base64,' not in (tmp_path / 'vlm.jsonl').read_text()


@pytest.mark.parametrize('stage,carried,allowed', [('GRASP', False, True), ('GRASP', True, False), ('INSERT', False, False)])
def test_source_supported_exemption_is_limited_to_uncarried_grasp(stage, carried, allowed):
    session = session_for(0)
    guard = VisualReleaseGuard(VisualClient(session, verdict='SUPPORTED_SOURCE_REGRASP'))
    if allowed:
        assert guard(session, stage=stage, source='recovery', carry_established=carried)['verdict'] == 'SUPPORTED_SOURCE_REGRASP'
    else:
        with pytest.raises(ReleaseNotQualified):
            guard(session, stage=stage, source='recovery', carry_established=carried)


@pytest.mark.parametrize('verdict', ['UNSAFE', 'UNCERTAIN', 'MALFORMED'])
def test_denials_and_malformed_categories_never_authorize(verdict):
    session = session_for(30)
    original = deepcopy(session.audit_context)
    guard = VisualReleaseGuard(VisualClient(session, verdict=verdict))
    with pytest.raises(ReleaseNotQualified):
        guard(session, stage='RELEASE', source='vlm', carry_established=True)
    assert session.audit_context == original


def test_provider_error_fails_closed_and_restores_context():
    session = session_for(30)
    original = deepcopy(session.audit_context)
    guard = VisualReleaseGuard(VisualClient(session, error=RuntimeError('provider unavailable')))
    with pytest.raises(ReleaseNotQualified) as caught:
        guard(session, stage='RELEASE', source='vlm', carry_established=True)
    assert caught.value.execution_cancelled is False
    assert caught.value.details['error_type'] == 'RuntimeError'
    assert session.audit_context == original


def test_missing_camera_or_mismatched_depth_never_calls_provider():
    session = session_for(30)
    client = VisualClient(session)
    guard = VisualReleaseGuard(client)
    incomplete = observation()
    del incomplete['extra_views']['side']
    with pytest.raises(ReleaseNotQualified):
        guard(session, stage='RELEASE', source='vlm', carry_established=True, observation=incomplete)
    mismatched = observation()
    depth = json.loads(mismatched['wrist_depth_text'])
    depth['source_width_px'] += 1
    mismatched['wrist_depth_text'] = json.dumps(depth)
    with pytest.raises(ReleaseNotQualified):
        guard(session, stage='RELEASE', source='vlm', carry_established=True, observation=mismatched)
    assert not client.calls


@pytest.mark.parametrize('marker,slip,expected', [(18, False, ReleaseNotQualified), (30, True, GraspLost), (30, False, None)])
def test_runner_dispatch_rechecks_grip_after_visual_wait(marker, slip, expected):
    instance, session, controller_calls = simple_runner(carried=False)
    session.audit_context = {'role': 'controller'}
    session.get_observation = lambda: observation(marker)
    controls = []
    session.set_control_context = lambda **fields: controls.append(fields)
    client = VisualClient(session, on_call=(lambda: setattr(session, 'width', 0.)) if slip else None)
    guard = VisualReleaseGuard(client)
    session.authorize_release = lambda **fields: guard(session, **fields)
    if expected:
        with pytest.raises(expected):
            instance._control_step('RELEASE', step_idx=40, stage='RELEASE', source='vlm')
        assert not controller_calls
    else:
        instance._control_step('RELEASE', step_idx=40, stage='RELEASE', source='vlm')
        assert controller_calls == ['RELEASE']
    assert controls == [{'stage': 'RELEASE', 'source': 'vlm', 'carry_established': True}]
    assert len(client.calls) == 1


def test_recovery_rollback_hook_preserves_historical_carry():
    instance, session, calls = simple_runner(carried=True)
    session.audit_context = {}
    session.get_observation = lambda: observation(0)
    controls = []
    session.set_control_context = lambda **fields: controls.append(fields)
    guard = VisualReleaseGuard(VisualClient(session, verdict='SUPPORTED_SOURCE_REGRASP'))
    session.authorize_release = lambda **fields: guard(session, **fields)
    with pytest.raises(ReleaseNotQualified):
        instance._control_step('RELEASE', step_idx=41, stage='GRASP', source='recovery')
    assert controls[0]['carry_established'] is True and not calls


def test_model_false_positive_cannot_override_observed_depth_bound():
    session = session_for(12)
    guard = VisualReleaseGuard(VisualClient(session, verdict='HALF_INSERTED_HELD'))
    with pytest.raises(ReleaseNotQualified) as caught:
        guard(session, stage='RELEASE', source='vlm', carry_established=True)
    assert caught.value.details['verdict'] == 'UNSAFE'
    assert caught.value.details['measurement']['estimate_m'] == pytest.approx(.012)
    assert caught.value.details['measurement']['lower_bound_m'] == pytest.approx(.009)


def test_missing_dense_depth_blocks_destination_but_not_supported_source():
    session = session_for(30)
    obs = observation(30)
    del obs['wrist_depth_m']
    guard = VisualReleaseGuard(VisualClient(session, verdict='HALF_INSERTED_HELD'))
    with pytest.raises(ReleaseNotQualified) as caught:
        guard(session, stage='RELEASE', source='vlm', carry_established=True, observation=obs)
    assert caught.value.details['verdict'] == 'UNCERTAIN'
    source = VisualReleaseGuard(VisualClient(session, verdict='SUPPORTED_SOURCE_REGRASP'))
    assert source(session, stage='GRASP', source='recovery', carry_established=False,
                  observation=obs)['verdict'] == 'SUPPORTED_SOURCE_REGRASP'
