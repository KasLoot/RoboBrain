"""Confirmed outcome stops robot while monitoring remains active. No real model."""
import asyncio
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import httpx

from check_simulation_client import FakeRuntime, server
from robobrain.runtime import Runtime
from robobrain.templates import mujoco_plug_task
from robobrain.workspace import Workspace


async def check(url, physics, directory):
    store=Workspace(directory)
    task=mujoco_plug_task()
    task['simulation'].update(base_url=url,auto_stop='fail')
    task['config'].update(base_url='http://model.test/v1',model='test',evidence_horizon=0)
    task['progress']['interval_s']=.05
    runtime=Runtime(store)
    true_client=httpx.AsyncClient
    async def reply(request):
        if request.method=='GET': return httpx.Response(200,json={'data':[{'id':'test'}]})
        prompt=json.loads(request.content)['messages'][0]['content']
        if 'predefined node' in prompt:
            content={'reason':'synthetic progress', 'node_id':'ON_STAND'}
        else:
            content={'reason':'synthetic outcome','verdict':'POSITIVE' if 'off its source stand' in prompt else 'NEGATIVE'}
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(content)}}]})
    transport=httpx.MockTransport(reply)
    def client(*args,**kwargs):
        kwargs['transport']=transport
        return true_client(*args,**kwargs)
    try:
        for source in task['sources']:
            await runtime.connect(task,source['id'])
        await runtime.robot_command('start','start-policy-test')
        with patch('httpx.AsyncClient',client):
            runtime.start(task,False)
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                if physics.stops: break
                await asyncio.sleep(.03)
            assert len(physics.stops)==1, (physics.stops,runtime.error,runtime.state())
            assert runtime.phase=='running', runtime.state()
            await asyncio.sleep(.15)
            assert len(physics.stops)==1, 'Held visual failure sent repeated robot stops'
            runtime.stop_event.set()
            await asyncio.wait_for(runtime.job,3)
            assert runtime.error is None, runtime.error
        assert runtime.simulation.state()['execution_state']=='STOPPED'
        assert runtime.progress is not None
        assert runtime.simulation is not None, 'Monitoring teardown destroyed command connection'
    finally:
        await runtime.close()


def run():
    physics=FakeRuntime()
    with server(physics) as (url,app), tempfile.TemporaryDirectory() as directory:
        asyncio.run(check(url,physics,Path(directory)))
    print('Automatic outcome policy passed: current FAIL stops once, progress stays independent, monitoring continues, and teardown preserves robot state.')

if __name__=='__main__': run()
