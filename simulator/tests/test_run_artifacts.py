"""Audit boundary checks: no provider calls or simulator rendering required."""
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from core.record.run_artifacts import AuditedClient, RunArtifacts
from core.runners.real import RealEpisodeRunner


class ExecutionCancelled(RuntimeError):
    pass


class FakeClient:
    model = 'audit-test-model'
    api_key = 'test-private-key-never-log'

    def __init__(self):
        self.failure = None
        self.after_response = None

    def _post_chat(self, payload):
        if self.failure:
            raise self.failure
        if self.after_response:
            self.after_response()
        return {'choices': [{'message': {'content': '{"token":"GRASP"}'}}], 'api_key': self.api_key}

    def complete_json(self, prompt, agentview_image, wrist_image=None, debug=False):
        out = BytesIO()
        Image.fromarray(agentview_image).save(out, 'PNG')
        raw = self._post_chat({'model': self.model, 'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': prompt},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(out.getvalue()).decode()}},
        ]}]})
        return SimpleNamespace(raw_text=raw['choices'][0]['message']['content'], token='GRASP', payload={'json': {'token': 'GRASP'}})


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_jsonl_thread_safety_redaction_and_assets(tmp_path):
    audit = RunArtifacts(tmp_path, secrets=['test-private-key-never-log'])
    assert audit.directory == tmp_path.resolve()
    @dataclass
    class Result:
        position: object
        token: str
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: audit.append('events', {'event': 'sample', 'i': i,
            'result': Result(np.array([i, 1.0]), 'GRASP'), 'api_key': 'hidden',
            'error': RuntimeError('Authorization: Bearer test-private-key-never-log'),
            'url': 'https://person:secret@example.org/api?api_key=hidden&normal=ok'}), range(80)))
    events = rows(tmp_path / 'events.jsonl')
    assert len(events) == 80
    assert [e['audit_seq'] for e in events] == list(range(1, 81))
    text = (tmp_path / 'events.jsonl').read_text()
    assert 'test-private-key-never-log' not in text and 'person:secret' not in text
    assert all(e['result']['token'] == 'GRASP' for e in events)
    assert all(e['api_key'] == '[REDACTED]' for e in events)
    assert all('normal=ok' in e['url'] for e in events)
    with pytest.raises(ValueError):
        audit.save_bytes('../outside.bin', b'no')
    with pytest.raises(ValueError):
        audit.write_json('/tmp/not-a-run-file.json', {})
    ref = audit.save_image('images/one.png', np.zeros((8, 9, 3), np.uint8))
    assert ref['width'] == 9 and ref['height'] == 8
    assert (tmp_path / ref['asset']).is_file()


def test_call_and_raw_transport_audit(tmp_path):
    audit = RunArtifacts(tmp_path)
    client = FakeClient()
    audited = AuditedClient(client, audit, context=lambda: {'role': 'planner', 'execution_generation': 3})
    response = audited.complete_json('Inspect the plug', np.zeros((8, 9, 3), np.uint8))
    assert response.token == 'GRASP'
    events = rows(tmp_path / 'vlm.jsonl')
    assert [e['event'] for e in events] == ['vlm_call_start', 'vlm_transport_start', 'vlm_transport_end', 'vlm_call_end']
    assert all(e['role'] == 'planner' and e['model'] == client.model for e in events)
    assert len({e['call_id'] for e in events}) == 1
    assert events[0]['parameters']['prompt'] == 'Inspect the plug'
    assert events[-1]['raw_text'] == '{"token":"GRASP"}'
    raw_request = events[1]['request']['messages'][0]['content'][1]['image_url']['url']
    assert (tmp_path / raw_request['asset']).is_file()
    text = (tmp_path / 'vlm.jsonl').read_text()
    assert 'base64,' not in text and client.api_key not in text
    assert events[-2]['response']['choices'][0]['message']['content'] == '{"token":"GRASP"}'
    audited.close()
    assert client._post_chat.__self__ is client


def test_provider_errors_and_late_cancelled_response(tmp_path):
    audit = RunArtifacts(tmp_path)
    client = FakeClient()
    active = [True]
    def check():
        if not active[0]:
            raise ExecutionCancelled('generation superseded')
    audited = AuditedClient(client, audit, check=check, role='controller')
    client.failure = RuntimeError('failed with api_key=test-private-key-never-log')
    with pytest.raises(RuntimeError):
        audited.complete_json('request one', np.zeros((8, 9, 3), np.uint8))
    assert [e['event'] for e in rows(tmp_path/'vlm.jsonl')][-2:] == ['vlm_transport_error', 'vlm_call_error']
    client.failure = None
    client.after_response = lambda: active.__setitem__(0, False)
    with pytest.raises(ExecutionCancelled):
        audited.complete_json('request two', np.zeros((8, 9, 3), np.uint8))
    events = rows(tmp_path / 'vlm.jsonl')
    assert [e['event'] for e in events][-3:] == ['vlm_transport_end', 'vlm_transport_cancel', 'vlm_call_cancel']
    assert client.api_key not in (tmp_path / 'vlm.jsonl').read_text()


def test_every_high_level_control_records_end_even_if_interrupted():
    events = []
    class Session:
        def record_control_start(self, **fields):
            events.append(('start', fields))
            return 'control-123'
        def record_control_end(self, control_id, **fields):
            events.append(('end', {**fields, 'control_id': control_id}))
    class Controller:
        def step(self, token, **kwargs):
            if token == 'INTERRUPT':
                raise ExecutionCancelled('stop')
            if token == 'FAIL':
                raise ValueError('failed action')
            return SimpleNamespace(token=token, note='(noop)' if token == 'GRASP' else '', done=token == 'DONE')
    runner = object.__new__(RealEpisodeRunner)
    runner.session, runner.controller = Session(), Controller()
    fields = dict(step_idx=7, stage='GRASP', source='vlm')
    for token in ('GRASP', 'RELEASE', 'DONE'):
        runner._control_step(token, **fields)
    runner._control_step('STILL', **fields, execute=False)
    for token, kind in (('INTERRUPT', ExecutionCancelled), ('FAIL', ValueError)):
        with pytest.raises(kind):
            runner._control_step(token, **fields)
    assert [event[0] for event in events] == ['start', 'end'] * 6
    assert [event[1]['status'] for event in events if event[0] == 'end'] == ['noop', 'completed', 'completed', 'noop', 'interrupted', 'failed']
    assert all(event[1]['control_id'] == 'control-123' for event in events if event[0] == 'end')


def test_full_runner_hooks_initial_grasp_noop_recovery_done_and_cancel(tmp_path):
    from test_runner_checkpoint import Session, Planner, runner
    from core.vlm.vlm_client import VLMResponse
    from core.sim.service_runtime import ExecutionCancelled as RuntimeCancelled
    events = []
    class AuditedSession(Session):
        skip_initial_release = False
        state_seq = 0
        def get_observation(self):
            self.state_seq += 1
            obs = super().get_observation()
            obs['simulation_snapshot'] = {'state_seq': self.state_seq}
            # Later proprio reads may update this; audit must retain the image's identity.
            self.observation_identity = {'state_seq': -999}
            return obs
        def record_initial_pose(self):
            events.append(('initial_pose', self.pose.copy()))
        def set_audit_context(self, **fields):
            events.append(('context', fields))
        def record_control_start(self, **fields):
            events.append(('start', fields))
            return str(len(events))
        def record_control_end(self, control_id, **fields):
            events.append(('end', fields | {'control_id': control_id, 'pose': self.pose.copy()}))
        def record_decision(self, **fields):
            events.append(('decision', fields))
        def record_step_result(self, result):
            events.append(('step_result', result))
        def record_run_finished(self, success, reason):
            events.append(('finished', {'success': success, 'reason': reason}))
    tokens = iter(['GRASP', 'STILL', 'DONE', 'CANCEL'])
    def decide(**kwargs):
        token = next(tokens)
        return VLMResponse(token, 'Visible explanation for ' + token, {})
    session = AuditedSession()
    instance, controller = runner(session, Planner(), decide, tmp_path / 'runner')
    original_step = controller.step
    def step(token, **kwargs):
        if token == 'CANCEL':
            raise RuntimeCancelled('stopped by user')
        result = original_step(token, **kwargs)
        result.note = '(noop)' if token == 'GRASP' else ''
        return result
    controller.step = step
    instance._recovery_after_step = lambda **kw: SimpleNamespace(release=True, block_done=False, reset_history=False,
        prompt_note='', rollback_index=None) if kw['token'] == 'GRASP' else None
    with pytest.raises(RuntimeCancelled):
        instance.run()
    assert events[0][0] == 'initial_pose'
    starts = [fields for kind, fields in events if kind == 'start']
    ends = [fields for kind, fields in events if kind == 'end']
    assert [f['token'] for f in starts] == ['RELEASE', 'GRASP', 'RELEASE', 'STILL', 'DONE', 'CANCEL']
    assert [f['token'] for f in ends] == [f['token'] for f in starts]
    assert [f['status'] for f in ends] == ['completed', 'noop', 'completed', 'noop', 'completed', 'interrupted']
    assert starts[0]['source'] == 'startup' and starts[2]['source'] == 'recovery'
    assert all(f['pose'].shape == (7,) for f in ends)
    assert [f['output'] for kind, f in events if kind == 'decision'] == [
        'Visible explanation for GRASP', 'Visible explanation for STILL', 'Visible explanation for DONE', 'Visible explanation for CANCEL']
    assert len([e for e in events if e[0] == 'step_result']) == 3
    assert events[-1] == ('finished', {'success': False, 'reason': 'interrupted'})
    assert {f['role'] for kind, f in events if kind == 'context'} == {'planner', 'controller'}
    assert all(f['observation']['state_seq'] > 0 for kind, f in events if kind == 'context')


def test_actual_openai_client_model_and_private_key_are_audited(tmp_path):
    from core.vlm.vlm_client import VLMClient
    client = VLMClient(base_url='https://api.openai.com/v1', model='gpt-6-sol', api_key='private-config-test-key',
                       timeout_s=1, max_tokens=100, temperature=0, provider='openai', max_retries=0)
    # Replace transport before attaching audit: this test makes zero provider requests.
    client._post_chat = lambda payload: {'choices': [{'message': {'content': '{"token":"DONE"}'}}], 'model': payload['model']}
    audit = RunArtifacts(tmp_path)
    wrapped = AuditedClient(client, audit, role='controller')
    try:
        wrapped.complete_json('Check private-config-test-key', np.zeros((8, 9, 3), np.uint8))
        events = rows(tmp_path / 'vlm.jsonl')
        assert len(events) == 4
        assert all(event['model'] == 'gpt-6-sol' for event in events)
        assert events[1]['request']['model'] == 'gpt-6-sol'
        assert 'private-config-test-key' not in (tmp_path / 'vlm.jsonl').read_text()
    finally:
        wrapped.close()
        client.session.close()
