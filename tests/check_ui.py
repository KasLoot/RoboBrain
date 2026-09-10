"""UI/input adapter checks: uv run python tests/check_ui.py (no cameras or VLM needed)."""
import asyncio
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from unittest.mock import patch

import cv2
import httpx
import numpy as np
from fastapi.testclient import TestClient

from robobrain.monitor import Backend, Frame, HERE, load_config
from robobrain.runtime import prepare
from robobrain.sources import CompositeVideo, Snapshot
from robobrain.web import create_app
from robobrain.workspace import Workspace


def response(request):
    if request.url.path.endswith('/models'):
        return httpx.Response(200, json={'data': [{'id': load_config()['model']}]})
    payload = json.loads(request.content)
    properties = payload['response_format']['json_schema']['schema']['properties']
    field = next(k for k in properties if k != 'reason')
    value = {'verdict': 'NEGATIVE', 'location': 'UNKNOWN', 'grasp_state': 'UNKNOWN'}[field]
    return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps({field: value, 'reason': 'Test observation'})}}]})


async def unchanged_requests(store):
    task = store.get('insert_plug')
    assert task['name'] == 'Insert plug'
    assert task['config'] == json.loads((HERE / 'config.json').read_text())
    image = np.zeros((1080, 1920, 3), np.uint8)
    frame = Frame(image, 0, 0, time.monotonic())
    snapshot = Snapshot(image, 0, 0, frame.observed_at, {'camera1': frame})
    config, specialists = prepare(task, store, snapshot.views)
    for spec in task['specialists']:
        assert spec['blocks'][0]['text'] == (HERE / 'prompts' / (spec['id'] + '.md')).read_text()
        refs = [b for b in spec['blocks'] if b['type'] == 'image']
        for block, original in zip(refs, task['config']['references'][spec['id']]):
            assert store.asset_path(block['asset']).read_bytes() == (HERE / 'references' / original['file']).read_bytes()
    payloads = []
    def capture(request):
        payloads.append(json.loads(request.content))
        return response(request)
    async with httpx.AsyncClient(transport=httpx.MockTransport(capture)) as client:
        legacy = Backend(load_config(), client)
        adapted = Backend(config, client, specialists=specialists)
        await legacy.batch(frame, frame.observed_at, lambda e: None, normal=True)
        await adapted.batch(snapshot, frame.observed_at, lambda e: None, normal=True)
    assert payloads[:3] == payloads[3:], 'UI import must produce identical legacy VLM requests'
    spec = task['specialists'][0]
    spec['blocks'].append({**spec['blocks'][-1], 'id': 'second', 'source': 'camera2', 'crop': None, 'alias': 'Second camera'})
    task['sources'][1]['enabled'] = True
    second = Frame(np.full((240, 320, 3), 127, np.uint8), 19, 19, frame.observed_at)
    config, specs = prepare(task, store, {'camera1': frame, 'camera2': second})
    snapshot = Snapshot(image, 0, 0, frame.observed_at, {'camera1': frame, 'camera2': second})
    content = specs['success'].current_content(snapshot)
    assert len([p for p in content if p['type'] == 'image_url']) == 4
    assert content[-2]['text'] == 'Second camera'
    assert specs['success'].prompt == legacy.specialists['success'].prompt


class FakeCamera:
    closed = []
    def __init__(self, source):
        self.source = source
        self.ended = threading.Event()
        self.index = 0
        shape = (240, 320, 3) if source['id'] == 'camera1' else (360, 480, 3)
        self.image = np.full(shape, 60 if source['id'] == 'camera1' else 180, np.uint8)
    def start(self):
        self.started_at = time.monotonic()
    def get(self):
        self.index += 1
        return Frame(self.image, self.index, self.index, time.monotonic())
    def stop(self):
        self.closed.append(self.source['id'])


def api_checks(root):
    app = create_app(root)
    async_client = httpx.AsyncClient
    positive = False
    async def slow_response(request):
        if request.url.path.endswith('/chat/completions'):
            await asyncio.sleep(.05)
            if positive:
                payload = json.loads(request.content)
                if 'verdict' in payload['response_format']['json_schema']['schema']['properties']:
                    return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps({'verdict': 'POSITIVE', 'reason': 'Test evidence'})}}]})
        return response(request)
    def client_factory(**kwargs):
        return async_client(**kwargs, transport=httpx.MockTransport(slow_response))
    with patch('robobrain.runtime.LiveCamera', FakeCamera), patch('robobrain.runtime.httpx.AsyncClient', client_factory):
        with TestClient(app, base_url='http://127.0.0.1', headers={'X-RoboBrain': '1'}) as client:
            assert client.get('/').status_code == 200
            assert client.get('/', headers={'Host': 'testserver'}).status_code == 403
            assert client.post('/api/tasks', headers={'X-RoboBrain': '', 'Origin': 'https://untrusted.example'}, json={'name': 'Bad'}).status_code == 403
            task = client.post('/api/tasks', json={'name': 'Checks', 'copy_from': 'insert_plug'}).json()
            task['sources'][1]['enabled'] = True
            task['config']['episode_timeout'] = .3
            for spec in task['specialists']:
                for block in spec['blocks']:
                    if block['type'] == 'camera':
                        block['crop'] = None
            assert client.put('/api/tasks/' + task['id'], json=task).status_code == 200
            bad = copy.deepcopy(task)
            bad['id'] = '../escape'
            assert client.put('/api/tasks/' + task['id'], json=bad).status_code == 422
            image = (HERE / 'references' / 'success_released.jpg').read_bytes()
            uploaded = client.post('/api/assets/image', files={'file': ('reference.jpg', image, 'image/jpeg')})
            assert uploaded.status_code == 200 and uploaded.json()['width'] > 0
            assert client.post('/api/assets/image', files={'file': ('bad.jpg', b'not an image', 'image/jpeg')}).status_code == 422
            assert client.get('/api/assets/unknown.jpg').status_code == 422
            assert client.post(f"/api/tasks/{task['id']}/sources/camera1/connect").status_code == 200
            assert client.get('/api/preview/camera1').headers['content-type'] == 'image/jpeg'
            assert client.get(f"/api/tasks/{task['id']}/request/success").status_code == 200
            for record in (False, True):
                started = client.post('/api/run', json={'task_id': task['id'], 'record': record})
                assert started.status_code == 200 and started.json()['busy']
                assert client.put('/api/tasks/' + task['id'], json=task).status_code == 409
                assert client.post('/api/run', json={'task_id': task['id']}).status_code == 409
                time.sleep(.45)
                assert client.get('/api/state').json()['busy'], 'UI run stopped at the old episode timeout'
                assert client.post('/api/run/stop').status_code == 200
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    state = client.get('/api/state').json()
                    if not state['busy']:
                        break
                    time.sleep(.05)
                assert state['phase'] == 'finished', state
                assert state['result']['status'] == 'ON_GOING' and state['result']['stop_reason'] == 'user_stop'
                assert not state['error'], state
                if not record:
                    assert not state['output'] and not (root / 'runs').exists()
                else:
                    assert sorted(Path(p).suffix for p in state['output']) == ['.MOV', '.json', '.log']
                    movie = next(root / p for p in state['output'] if p.endswith('.MOV'))
                    stream = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(movie)]))['streams'][0]
                    assert (stream['width'], stream['height'], stream['r_frame_rate']) == (800, 360, '30/1')
                    report = json.loads(next(root / p for p in state['output'] if p.endswith('.json')).read_text())
                    assert report['task_template']['id'] == task['id']
                    assert set(report['requests'][0]['source_frames']) == {'camera1', 'camera2'}
                    assert report['requests'][0]['response']['choices']
            with patch('robobrain.runtime.subprocess.Popen', side_effect=OSError('Test encoder unavailable')):
                client.post('/api/run', json={'task_id': task['id'], 'record': True})
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    state = client.get('/api/state').json()
                    if not state['busy']:
                        break
                    time.sleep(.02)
                assert state['phase'] == 'error' and 'Test encoder unavailable' in state['error']
                assert state['result']['stop_reason'] == 'recording_error' and not state['sources']
            task['config']['episode_timeout'] = 10
            positive = True
            client.put('/api/tasks/' + task['id'], json=task)
            client.post('/api/run', json={'task_id': task['id'], 'record': False})
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                state = client.get('/api/state').json()
                if state['result']['status'] == 'SUCCESS':
                    break
                time.sleep(.02)
            assert state['result']['status'] == 'SUCCESS' and state['phase'] == 'running' and state['busy']
            assert client.put('/api/tasks/' + task['id'], json=task).status_code == 409
            positive = False
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                state = client.get('/api/state').json()
                if state['result']['status'] == 'ON_GOING':
                    break
                time.sleep(.02)
            assert state['result']['status'] == 'ON_GOING' and state['busy']
            positive = True
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                state = client.get('/api/state').json()
                if state['verification'] and state['verification']['active']:
                    break
                time.sleep(.02)
            assert state['verification']['active']
            assert client.post('/api/run/stop').status_code == 200
            deadline = time.monotonic() + 5
            while client.get('/api/state').json()['busy'] and time.monotonic() < deadline:
                time.sleep(.05)
            assert client.get('/api/state').json()['result']['stop_reason'] == 'user_stop'
            assert client.get('/api/state').json()['verification']['cancelled']
            positive = False
            task['input_mode'] = 'video'
            task['config']['episode_timeout'] = .3
            paths = {}
            for index, source in enumerate(task['sources']):
                video = root / 'assets' / f'clip{index}.MOV'
                subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                    f'testsrc2=size={320+160*index}x{240+120*index}:rate=30',
                    '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000', '-t', str(.4+.2*index),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(video)], check=True)
                source['video'] = video.name
                paths[source['id']] = video
            composite = CompositeVideo(paths, {s['id']: s['name'] for s in task['sources']})
            indexes = []
            try:
                while composite.read()[0]:
                    indexes.append([f.source_index for f in composite.frames.values()])
            finally:
                composite.release()
            assert indexes[0] == [0, 0] and indexes[-1] == [11, 17] and len(indexes) == 18
            client.put('/api/tasks/' + task['id'], json=task)
            client.post('/api/run', json={'task_id': task['id'], 'record': True})
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                state = client.get('/api/state').json()
                if state['elapsed_s'] >= .8:
                    break
                time.sleep(.02)
            assert state['phase'] == 'running', 'Continuous replay stopped automatically'
            assert client.post('/api/run/stop').status_code == 200
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                state = client.get('/api/state').json()
                if not state['busy']:
                    break
                time.sleep(.05)
            assert state['phase'] == 'finished', state
            movie = next(root / p for p in state['output'] if p.endswith('.MOV'))
            streams = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(movie)]))['streams']
            assert len([s for s in streams if s['codec_type'] == 'audio']) == 2
            video = next(s for s in streams if s['codec_type'] == 'video')
            assert (video['width'], video['height']) == (800, 360) and int(video['nb_frames']) >= 24
            assert client.delete('/api/tasks/' + task['id']).status_code == 200
            assert len(client.get('/api/tasks').json()) == 1
    assert 'camera1' in FakeCamera.closed and 'camera2' in FakeCamera.closed
    assert len(Workspace(root).tasks) == 1


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        asyncio.run(unchanged_requests(Workspace(root)))
        api_checks(root)
    print('UI checks passed: exact legacy requests/assets/config, two-view assembly, uploads, persistence, backend locks, recording artifacts, stop and cleanup.')
