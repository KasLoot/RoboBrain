"""Remote evidence cadence, cancellation, and recording gap checks without models."""
import asyncio
import copy
import json
from unittest.mock import patch
import httpx
from pathlib import Path
import shutil
import tempfile
import threading
import time
from types import SimpleNamespace

import numpy as np

from robobrain.monitor import Frame, input_batch, current_observations, monitor
from robobrain.sources import Snapshot
from robobrain.runtime import Recorder
from robobrain.presentation import LiveView, EpisodeReport
from robobrain.progress import ProgressRunner


class Camera:
    mode = 'mujoco'
    def __init__(self):
        self.started_at = time.monotonic()
        self.number = 1
        self.context = ('instance', 'episode', 'attempt', 1)
        self.unavailable = False
        self.image = np.zeros((96, 128, 3), np.uint8)
    def get(self):
        if self.unavailable:
            raise RuntimeError('Waiting for camera revision')
        identity = (*self.context, self.number)
        meta = {'identity': identity, 'context': self.context,
                'simulation': {'state_seq': self.number, 'sim_time_s': self.number / 15}}
        frame = Frame(self.image, self.number, self.number, time.monotonic(), metadata=meta)
        return Snapshot(frame.image, frame.sequence, frame.source_index, frame.observed_at,
                        {'camera1': frame}, metadata=meta)
    def evidence_context(self): return self.context
    def accepts_evidence(self, evidence):
        return not self.unavailable and evidence and evidence['context'] == self.context


class Backend:
    specialists = {'success': SimpleNamespace(name='success', failure_type=None, threshold=0,
                    evidence_count=lambda obs: sum(o['verdict']=='POSITIVE' for o in obs),
                    confirmed=lambda obs: any(o['verdict']=='POSITIVE' for o in obs))}
    def __init__(self): self.calls = 0; self.release = None
    async def batch(self, frame, started_at, log, **tags):
        self.calls += 1
        if self.release: await self.release.wait()
        return [{'specialist':'success','verdict':'POSITIVE','reason':'synthetic',
                 'captured_time_s':frame.observed_at-started_at, 'source_time_s':0.,
                 'evidence':{'context':frame.metadata['context'], 'identity':frame.metadata['identity'],
                             'observed_at':frame.observed_at}, **tags}]


async def checks():
    camera, backend = Camera(), Backend()
    first = await input_batch(camera,backend,camera.started_at,lambda e:None,normal=True)
    second = asyncio.create_task(input_batch(camera,backend,camera.started_at,lambda e:None,normal=True))
    await asyncio.sleep(.05)
    assert backend.calls == 1 and not second.done(), 'Repeated frame sent to VLM again'
    camera.number += 1
    await asyncio.wait_for(second,.5)
    assert backend.calls == 2
    duplicate = current_observations(camera,[first[0],first[0]])
    assert [o['verdict'] for o in duplicate] == ['POSITIVE','UNCERTAIN']
    camera.context = ('instance','episode-reset','attempt',2)
    assert current_observations(camera,first)[0]['verdict']=='UNCERTAIN'
    camera.unavailable = True
    unavailable = await input_batch(camera,backend,camera.started_at,lambda e:None,normal=False,slot=1)
    assert unavailable[0]['slot']==1 and unavailable[0]['verdict']=='UNCERTAIN'
    assert backend.calls == 2

    camera, backend = Camera(), Backend()
    backend.release = asyncio.Event()
    stop = threading.Event()
    events=[]
    job=asyncio.create_task(monitor(camera,backend,{'evidence_horizon':0},events.append,stop,continuous=True))
    await asyncio.sleep(.03)
    camera.context=('instance','different-episode','attempt',2)
    await asyncio.sleep(.08)
    assert not any(e.get('status')=='SUCCESS' for e in events), 'Old inference escaped reset cancellation'
    backend.release.set()
    await asyncio.sleep(.05)
    stop.set()
    await asyncio.wait_for(job,1)
    successes=[e for e in events if e.get('status')=='SUCCESS']
    assert successes and all(e['evidence']['context']==camera.context for e in successes)

    # Runtime's episode-bound log callback stops the old run synchronously.
    # monitor must not queue a new-episode call between that callback and its next loop.
    camera, backend = Camera(), Backend()
    backend.release = asyncio.Event()
    stop = threading.Event()
    events = []
    def end_episode(event):
        events.append(copy.deepcopy(event))
        if event['type'] == 'simulation_context_changed':
            stop.set()
    job = asyncio.create_task(monitor(camera, backend, {'evidence_horizon': 0}, end_episode, stop, continuous=True))
    await asyncio.sleep(.03)
    assert backend.calls == 1
    camera.context = ('instance', 'next-episode', None, 2)
    result = await asyncio.wait_for(job, 1)
    assert result['stop_reason'] == 'simulation_episode_changed'
    assert backend.calls == 1, 'Reset scheduled a new-scene request in the old audit run'



async def progress_audit_checks():
    camera = Camera()
    events, calls = [], []
    view = LiveView()
    def log(event):
        # Mimic immediate JSONL serialization: later mutation must not repair this copy.
        events.append(json.loads(json.dumps(event)))
        view.update(event)
    definition = {'enabled': True, 'nodes': [{'id': 'HELD', 'label': 'Held', 'criteria': 'Handle visibly clamped.'}],
        'blocks': [{'id': 'context', 'type': 'text', 'text': 'Track visible grasp progress.'},
                   {'id': 'view', 'type': 'camera', 'source': 'camera1', 'crop': None}],
        'interval_s': .05, 'confirmations': 1, 'stale_after_s': 1., 'history_frames': 2, 'max_tokens': 128}
    outcome = SimpleNamespace(config={'max_tokens': 128}, model='fake', info={}, audit=None, request_gate=None)
    class FakeProgressBackend:
        native = False
        def __init__(self, *args, **kwargs):
            pass
        async def complete(self, observer, history):
            calls.append(history[-1][1].metadata['context'])
            if len(calls) == 1:
                # Camera generation changed while this model request was in flight.
                camera.context = ('instance', 'episode', 'attempt', 2)
                camera.unavailable = True
            return httpx.Response(200, json={'choices': [{'message': {'content':
                json.dumps({'node_id': 'HELD', 'reason': 'Visible clamp'})}}]},
                request=httpx.Request('POST', 'http://test.invalid'))
    async def until(predicate):
        deadline = time.monotonic() + 2
        while not predicate():
            assert time.monotonic() < deadline, (events, calls)
            await asyncio.sleep(.01)
    with patch('robobrain.progress.Backend', FakeProgressBackend):
        progress = ProgressRunner(definition, None, camera, outcome, log)
        progress.start()
        try:
            await until(lambda: any(e['type'] == 'progress_state' and e.get('quality') == 'unavailable' for e in events))
            observation = next(e for e in events if e['type'] == 'progress_observation')
            assert observation['discarded'] and observation['node_id'] == 'UNKNOWN'
            assert observation['discarded_node_id'] == 'HELD'
            assert any(e['type'] == 'progress_state' and e.get('quality') == 'stale' for e in events)
            assert view.progress['quality'] == 'unavailable', 'Recorded overlay retained confirmed progress'
            camera.unavailable = False
            camera.number += 1
            await until(lambda: progress.state()['quality'] == 'confirmed')
            assert len(calls) == 2, 'Generation change should allow fresh progress within the same episode'
            camera.context = ('instance', 'reset-episode', None, 3)
            camera.number += 1
            await asyncio.wait_for(progress.job, 1)
            assert len(calls) == 2, 'Progress called the provider for a new episode in the old audit folder'
            assert any(e['type'] == 'progress_episode_ended' for e in events)
            assert view.progress['quality'] == 'unavailable'
        finally:
            await progress.close()

    camera = Camera()
    events.clear()
    calls.clear()
    class BlockedProgressBackend(FakeProgressBackend):
        async def complete(self, observer, history):
            calls.append(history[-1][1].metadata['context'])
            await asyncio.Event().wait()
    with patch('robobrain.progress.Backend', BlockedProgressBackend):
        progress = ProgressRunner(definition, None, camera, outcome, log)
        progress.start()
        await until(lambda: bool(calls))
        camera.context = ('instance', 'cancel-reset', None, 4)
        await progress.close()
        cancelled = next(e for e in events if e['type'] == 'progress_observation')
        assert cancelled['cancelled'] and cancelled['discarded']
        assert view.progress['quality'] == 'stale', 'Reset cancellation failed to invalidate the recorded overlay'
        assert len(calls) == 1


def recording():
    ffmpeg=shutil.which('ffmpeg')
    assert ffmpeg, 'ffmpeg required for recording regression'
    camera=Camera(); stopped=threading.Event(); events=[]
    view=LiveView(); view.status={'status':'ON_GOING','failure_type':None}
    with tempfile.TemporaryDirectory() as directory:
        recorder=Recorder(camera,[{'id':'camera1','name':'Camera'}],Path(directory)/'test.MOV',view,ffmpeg,stopped,on_event=events.append)
        time.sleep(.12)
        camera.unavailable=True
        time.sleep(.14)
        camera.unavailable=False
        camera.number+=1
        time.sleep(.1)
        result=recorder.finish(False)
        assert not stopped.is_set(), 'Temporary view gap stopped monitoring'
        assert result['unavailable_frames'] > 0 and result['frames'] > result['unavailable_frames']
        assert any(e['unavailable'] for e in events)
        assert (Path(directory)/'test.MOV').stat().st_size>0
        report=EpisodeReport(Path('gap'), 'simulation-test', {'specialists': {}},Path(directory),view)
        camera.unavailable=True
        asyncio.run(input_batch(camera, Backend(), camera.started_at, report.log, normal=False, slot=0))
        report.finish({'status':'ON_GOING','source_frame':0},result)
        assert (Path(directory)/'output-gap.json').is_file(), 'Unavailable stream broke report finalization'


if __name__=='__main__':
    asyncio.run(checks())
    asyncio.run(progress_audit_checks())
    recording()
    print('Simulation monitoring checks passed: distinct frame cadence, stale/duplicate abstention, reset cancellation, and recorded camera gaps without stopping inference.')
