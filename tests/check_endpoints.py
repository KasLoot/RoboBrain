"""Monitor transport checks: uv run python tests/check_endpoints.py. No cloud requests."""

import asyncio
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import httpx
import numpy as np
from fastapi.testclient import TestClient

from check_ui import FakeCamera
from robobrain.endpoints import cloud_model, endpoint_error, model_info, ollama_headers
from robobrain.monitor import Backend, Frame, load_config
from robobrain.web import create_app


def reply(request, mode='valid', positive=False):
    path = request.url.path
    if path.endswith('/tags'):
        return httpx.Response(200, json={'models':[{'name':'gemma4:31b'}]})
    if path.endswith('/show'):
        return httpx.Response(200, json={'capabilities':['vision','thinking']})
    assert path.endswith('/chat'), path
    payload = json.loads(request.content)
    assert payload['model'] == 'gemma4:31b' and payload['think'] is False and payload['stream'] is False
    assert 'response_format' not in payload and payload['options']['temperature'] == 0
    schema = payload.get('format') or json.loads(payload['messages'][-1]['content'].split('without Markdown:\n')[-1])
    field = next(k for k in schema['properties'] if k != 'reason')
    value = {'verdict':'POSITIVE' if positive else 'NEGATIVE','location':'UNKNOWN','grasp_state':'UNKNOWN'}[field]
    content = {field:value,'reason':'Visible test evidence'}
    if mode == 'extra':
        content['status'] = 'FAIL'
    body = {'message':{'role':'assistant','content':'not JSON' if mode == 'malformed' else json.dumps(content)},
            'done':mode != 'incomplete','done_reason':'length' if mode == 'length' else 'stop',
            'prompt_eval_count':42,'eval_count':12}
    if mode == 'thinking':
        body['message']['thinking'] = 'Unexpected reasoning'
    return httpx.Response(200, json=body)


async def transport_checks():
    base = load_config()
    image = np.zeros((1080,1920,3),np.uint8)
    frame = Frame(image,0,0,time.monotonic())
    for url in ('https://ollama.com/api','http://localhost:11434/api'):
        config = {**base,'base_url':url,'model':'gemma4:31b-cloud'}
        requests, active, peak = [], 0, 0
        async def handler(request):
            nonlocal active, peak
            requests.append(request)
            assert request.headers.get('Authorization') == ('Bearer test-ollama-key' if request.url.host == 'ollama.com' else None)
            if request.url.path.endswith('/chat'):
                active += 1
                peak = max(active,peak)
                await asyncio.sleep(.01)
                active -= 1
            return reply(request)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            backend = Backend(config,client)
            info = await backend.check_model()
            assert info['model'] == 'gemma4:31b' and config['model'] == 'gemma4:31b-cloud'
            observations = await backend.batch(frame,frame.observed_at,lambda e:None,normal=True)
        assert peak == 3 and all('error' not in o for o in observations)
        assert all(o['usage'] == {'prompt_tokens':42,'completion_tokens':12} for o in observations)
        assert [r.url.path for r in requests[:2]] == ['/api/tags','/api/show']
        for request, specialist in zip(requests[2:],backend.specialists.values()):
            payload = json.loads(request.content)
            assert payload['messages'][0]['content'] == specialist.prompt
            original = [*specialist.references,*specialist.current_content(frame)]
            images = [p['image_url']['url'].split(',',1)[1] for p in original if p['type'] == 'image_url']
            assert payload['messages'][1]['images'] == images, 'Image bytes/order must be preserved'
            assert all(p['text'] in payload['messages'][1]['content'] for p in original if p['type'] == 'text')
            assert ('format' in payload) == (request.url.host == 'localhost')
            assert payload['options']['num_predict'] == config['max_tokens']

    for mode in ('malformed','extra','thinking','incomplete','length'):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:reply(r,mode))) as client:
            backend = Backend({**base,'base_url':'http://localhost:11434/api','model':'gemma4:31b'},client)
            observations = await backend.batch(frame,frame.observed_at,lambda e:None,normal=True)
        assert all(o['verdict'] == 'UNCERTAIN' and 'error' in o for o in observations), mode
    assert cloud_model('http://localhost:11434/api','gemma4:31b-cloud')
    assert ollama_headers('https://example.com/api') == {}
    with patch.dict('os.environ', {'OLLAMA_API_KEY':''}):
        try:
            ollama_headers('https://ollama.com/api')
        except ValueError as exc:
            assert 'OLLAMA_API_KEY' in str(exc)
        else:
            raise AssertionError('Missing cloud key accepted')
    try:
        ollama_headers('http://ollama.com/api')
    except ValueError:
        pass
    else:
        raise AssertionError('Cloud credential would be sent without HTTPS')
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:httpx.Response(200,json={'models':[{'name':'gemma4:31b'}]}))) as client:
        try:
            await model_info(client,'https://ollama.com/api','missing-model')
        except ValueError as exc:
            assert 'not available' in str(exc)
        else:
            raise AssertionError('Unknown model accepted')
    assert 'SSH' not in endpoint_error('https://ollama.com/api',ValueError('test'))


def api_checks(root):
    app = create_app(root)
    real_client = httpx.AsyncClient
    paths = []
    async def native_endpoint(request):
        paths.append(request.url.path)
        assert request.headers['Authorization'] == 'Bearer test-ollama-key'
        await asyncio.sleep(.02)
        return reply(request,positive=True)
    def client_factory(**kwargs):
        return real_client(**kwargs,transport=httpx.MockTransport(native_endpoint))
    with patch('robobrain.runtime.LiveCamera',FakeCamera), patch('robobrain.runtime.httpx.AsyncClient',client_factory):
        with TestClient(app,base_url='http://127.0.0.1',headers={'X-RoboBrain':'1'}) as client:
            task=client.get('/api/tasks').json()[0]
            task['config'].update(base_url='https://ollama.com/api',model='gemma4:31b-cloud',evidence_horizon=0)
            for spec in task['specialists']:
                for block in spec['blocks']:
                    if block['type']=='camera': block['crop']=None
            assert client.put('/api/tasks/'+task['id'],json=task).status_code == 200
            checked=client.post('/api/model/check',json={'task_id':task['id']})
            assert checked.status_code == 200 and checked.json()['model']=='gemma4:31b',checked.text
            assert client.post('/api/run',json={'task_id':task['id'],'record':True}).status_code == 200
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                state=client.get('/api/state').json()
                if state['result']['status']=='SUCCESS':break
                time.sleep(.02)
            assert state['phase']=='running' and state['result']['status']=='SUCCESS',state
            assert client.post('/api/run/stop').status_code==200
            deadline=time.monotonic()+10
            while time.monotonic()<deadline:
                state=client.get('/api/state').json()
                if not state['busy']:break
                time.sleep(.02)
            assert state['phase']=='finished' and not state['error'],state
            report=json.loads(next(root/p for p in state['output'] if p.endswith('.json')).read_text())
            assert report['requests'][0]['response']['message']['content']
            assert all(p in {'/api/tags','/api/show','/api/chat'} for p in paths)


if __name__=='__main__':
    with patch.dict('os.environ',{'OLLAMA_API_KEY':'test-ollama-key'}):
        asyncio.run(transport_checks())
        with tempfile.TemporaryDirectory() as root:
            api_checks(Path(root))
    print('Endpoint checks passed: cloud/local Ollama, aliases, auth, image preservation, concurrency, malformed replies, UI startup and recording.')
