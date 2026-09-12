"""Task Builder checks: uv run python tests/check_builder.py (no cloud requests)."""

import asyncio
import copy
from collections import deque
import io
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import cv2
import httpx
import numpy as np
from fastapi.testclient import TestClient

from robobrain.monitor import Frame
from robobrain.runtime import prepare
from robobrain.task_builder import TaskBuilder
from robobrain.web import create_app


def checks(root):
    app = create_app(root)
    store, builder = app.state.store, app.state.builder
    original = copy.deepcopy(store.get('insert_plug'))
    replies, payloads = deque(), []
    control = {'hang': False, 'vision': True, 'status': 200}
    real_client = httpx.AsyncClient

    async def endpoint(request):
        assert request.headers.get('Authorization') == 'Bearer builder-test-key'
        if request.url.path.endswith('/tags'):
            return httpx.Response(200, json={'models': [{'name': 'kimi-k3'}]})
        if request.url.path.endswith('/show'):
            return httpx.Response(200, json={'capabilities': ['thinking'] + (['vision'] if control['vision'] else [])})
        payload = json.loads(request.content)
        payloads.append(payload)
        assert payload['model'] == 'kimi-k3' and payload['stream'] and payload['think']
        assert 'format' not in payload and 'response_format' not in payload
        await asyncio.sleep(.03)
        if control['hang']:
            await asyncio.Event().wait()
        if control['status'] != 200:
            return httpx.Response(control['status'])
        content = replies.popleft()
        chunks = [{'message': {'thinking': 'Private model history.'}},
                  {'message': {'content': content[:len(content)//2]}},
                  {'message': {'content': content[len(content)//2:]}, 'done': True,
                   'done_reason': 'stop', 'eval_count': 100, 'prompt_eval_count': 250}]
        return httpx.Response(200, text='\n'.join(json.dumps(c) for c in chunks))

    def client_factory(self):
        return real_client(base_url=self.settings.base_url+'/', headers={'Authorization':'Bearer builder-test-key'},
                           transport=httpx.MockTransport(endpoint))

    with patch.dict('os.environ', {'OLLAMA_API_KEY': 'builder-test-key'}), patch.object(TaskBuilder, 'client', client_factory):
        with TestClient(app, base_url='http://127.0.0.1', headers={'X-RoboBrain':'1'}) as client:
            def wait():
                deadline = time.monotonic()+5
                while time.monotonic() < deadline:
                    state = client.get('/api/builder').json()
                    if not state['busy']:
                        return state
                    time.sleep(.01)
                raise AssertionError('Builder did not finish')

            state = client.get('/api/builder').json()
            assert state['settings']['model'] == 'kimi-k3:cloud' and state['key_configured']
            assert 'builder-test-key' not in json.dumps(state)
            assert client.post('/api/builder/check').json()['connection']['vision']
            png = cv2.imencode('.png', np.full((100,120,3), 127, np.uint8))[1].tobytes()
            asset = client.post('/api/assets/image', files={'file':('scene.png',io.BytesIO(png),'image/png')}).json()['asset']
            inputs = state['inputs']
            inputs['sources'][1]['enabled'] = True
            base = client.post('/api/tasks', json={'name':'Zero horizon', 'copy_from':'insert_plug'}).json()
            base['config']['evidence_horizon'] = 0
            assert client.put('/api/tasks/'+base['id'], json=base).status_code == 200
            inputs['base_task_id'] = base['id']
            message = {'action':'chat','text':'Move black wires to a blue mat. Fail on a deposited distractor.',
                       'attachments':[{'asset':asset,'label':'Initial scene','reference':True}], 'inputs':inputs}
            replies.append('I will create one success specialist and one failure specialist.')
            assert client.post('/api/builder/respond', json=message).status_code == 200
            assert client.post('/api/run', json={'task_id':'insert_plug'}).status_code == 409
            assert client.post('/api/tasks', json={'name':'Race'}).status_code == 409
            state = wait()
            assert state['phase'] == 'idle' and len(state['messages']) == 2
            assert 'Private model history.' not in json.dumps(state)
            assert payloads[0]['messages'][1]['images'] and 'Initial scene' in payloads[0]['messages'][1]['content']

            draft = {'name':'Select black wires','summary':'Move only black wires onto the blue mat.', 'specialists':[
                {'name':'Success','prompt':'Report POSITIVE when all black wires are on the blue mat and no distractors are on it. Otherwise NEGATIVE or UNCERTAIN.',
                 'failure_type':None,'camera_sources':['camera1','camera2'],'reference_assets':[asset]},
                {'name':'Wrong object','prompt':'Report POSITIVE when a non-black wire or a brick is deposited on the blue mat. Otherwise NEGATIVE or UNCERTAIN.',
                 'failure_type':'WRONG_OBJECT_ON_MAT','camera_sources':['camera2'],'reference_assets':[]}]}
            replies.extend(['not valid JSON', json.dumps(draft)])
            assert client.post('/api/builder/respond', json={**message,'action':'generate','text':'','attachments':[]}).status_code == 200
            state = wait()
            assert state['phase'] == 'ready' and state['draft'] == {**draft, 'progress': None}, state
            assert len(payloads) == 3 and len(store.tasks) == 2
            assert any(m.get('thinking') == 'Private model history.' for m in payloads[-1]['messages'])
            assert not any('not valid JSON' in m['content'] for m in state['messages'])
            bad = copy.deepcopy(draft)
            bad['specialists'][0]['reference_assets'] = ['not-uploaded.png']
            assert client.post('/api/builder/create', json={'revision':state['revision'],'draft':bad}).status_code == 422
            bad = copy.deepcopy(draft)
            bad['specialists'][1]['failure_type'] = None
            assert client.post('/api/builder/create', json={'revision':state['revision'],'draft':bad}).status_code == 422
            assert client.post('/api/builder/create', json={'revision':-1,'draft':draft}).status_code == 422
            assert len(store.tasks) == 2
            draft['specialists'][0]['prompt'] += ' Return only verdict and reason as JSON.'
            body = {'revision':state['revision'],'draft':draft}
            created = client.post('/api/builder/create', json=body)
            assert created.status_code == 200, created.text
            task = created.json()
            assert client.post('/api/builder/create', json=body).json()['id'] == task['id']
            assert len(store.tasks) == 3 and task['config']['evidence_horizon'] == 0
            assert task['config']['model'] == original['config']['model'] and task['config']['base_url'] == original['config']['base_url']
            views = {s['id']:Frame(np.zeros((240,320,3),np.uint8),0,0,time.monotonic()) for s in task['sources']}
            config, specialists = prepare(task, store, views)
            assert len(specialists) == 2 and all(s.threshold == 5 for s in specialists.values())
            assert store.get('insert_plug') == original
            assert TaskBuilder(store).state()['created']

            # A cancelled, refused, malformed or text-only-model request must never create a task.
            client.post('/api/builder/reset')
            control['hang'] = True
            assert client.post('/api/builder/respond', json=message).status_code == 200
            assert client.post('/api/builder/cancel').status_code == 200
            assert wait()['phase'] == 'cancelled' and len(store.tasks) == 3
            control['hang'] = False
            control['status'] = 401
            client.post('/api/builder/respond', json={**message,'attachments':[]})
            assert 'HTTP 401' in wait()['error']
            control['status'] = 200
            replies.extend(['{}','{}'])
            client.post('/api/builder/respond', json={**message,'action':'generate','attachments':[]})
            state = wait()
            assert state['phase'] == 'error' and state['draft'] is None and len(store.tasks) == 3
            replies.append(json.dumps({'name':'More information needed','summary':'What should count as failure?','specialists':[]}))
            client.post('/api/builder/respond', json={**message,'action':'generate','attachments':[]})
            state = wait()
            assert state['draft'] is None and state['messages'][-1]['content'] == 'What should count as failure?'
            control['vision'] = False
            before = len(payloads)
            client.post('/api/builder/respond', json=message)
            assert 'does not support image' in wait()['error'] and len(payloads) == before
            app.state.runtime.phase = 'running'
            assert client.post('/api/builder/respond', json=message).status_code == 409
            assert client.post('/api/builder/create', json=body).status_code == 409
            app.state.runtime.phase = 'idle'
            settings = state['settings']
            assert client.put('/api/builder/settings', json={**settings,'base_url':'https://attacker.example/api'}).status_code == 422
            assert client.put('/api/builder/settings', json={**settings,'model':'another-model'}).status_code == 200
            state = client.get('/api/builder').json()
            assert not state['messages'] and state['draft'] is None
            assert client.post('/api/builder/check').status_code == 422
            assert store.get('insert_plug') == original


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as root:
        checks(Path(root))
    print('Builder checks passed: image chat, thinking history, repair, validation, task creation, persistence, idempotency, cancellation, model errors and run locks.')
