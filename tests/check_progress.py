"""Progress checks: uv run python tests/check_progress.py. No cameras or cloud required."""

import asyncio
import copy
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import cv2
import httpx
import numpy as np
from fastapi.testclient import TestClient

from check_ui import FakeCamera, response
from check_endpoints import reply as native_reply
from robobrain.monitor import Frame
from robobrain.presentation import annotate_frame, progress_snapshot
from robobrain.progress import ProgressObserver, ProgressTracker
from robobrain.task_builder import Draft, TaskBuilder
from robobrain.web import create_app
from robobrain.workspace import ProgressDefinition, Workspace


def definition():
    return {'enabled': True, 'nodes': [
        {'id':'P0','label':'Neither group sorted','criteria':'Neither category is completely sorted.','completed':0,'total':2},
        {'id':'BRICKS','label':'Bricks sorted','criteria':'Only all bricks are inside the box.','completed':1,'total':2},
        {'id':'WIRES','label':'Wires sorted','criteria':'Only all wires are on the mat.','completed':1,'total':2},
        {'id':'BOTH','label':'Both groups sorted','criteria':'All bricks are in the box and all wires are on the mat.','completed':2,'total':2}],
        'blocks':[{'id':'context','type':'text','text':'Sort brown bricks into the white box and black wires onto the yellow mat.'},
                  {'id':'view','type':'camera','source':'camera1','crop':None}],
        'unit_label':'groups','interval_s':.05,'confirmations':2,'stale_after_s':1.,'history_frames':3,'max_tokens':192}


def tracker_checks(root):
    config=ProgressDefinition.model_validate(definition())
    tracker=ProgressTracker(config)
    def observe(node,key,captured,now=None):
        return tracker.observe({'node_id':node,'reason':'Visible evidence','captured_time_s':captured},key,captured if now is None else now)
    assert observe('P0',1,0)['quality']=='pending'
    assert observe('P0',1,.1) is None
    assert observe('P0',2,.2)['node_id']=='P0'
    assert observe('BOTH',3,.3)['quality']=='pending'
    assert observe('BOTH',4,.4)['node_id']=='BOTH', 'Skipped nodes must be allowed'
    assert progress_snapshot(tracker.state,.5)['percent']==100
    assert observe('BRICKS',5,.5)['quality']=='pending'
    assert observe('BRICKS',6,.6)['completed']==1, 'Progress must allow backward movement'
    assert observe('UNKNOWN',7,.7)['quality']=='unknown'
    assert tracker.state['node_id']=='BRICKS' and progress_snapshot(tracker.state,.8)['percent'] is None
    assert observe('WIRES',8,.8)['matches']==1
    assert observe('WIRES',9,.9)['node_id']=='WIRES'
    assert progress_snapshot(tracker.state,2.)['quality']=='stale'
    assert observe('BOTH',10,1.,3.)['quality']=='stale' and tracker.state['node_id']=='WIRES'
    for key,node in enumerate(('BRICKS','BOTH','BRICKS'),11):
        now=3.+key/10
        assert observe(node,key,now)['quality']=='pending'
        current=progress_snapshot(tracker.state,now)
        assert current['node_id']=='WIRES' and current['quality']=='stale' and current['percent'] is None, 'Fresh conflicting candidates must not keep old confirmed evidence fresh'

    for change in ({'nodes':[config.nodes[0].model_dump()]*2}, {'nodes':[{'id':'UNKNOWN','label':'Unknown','criteria':'Unclear'}]},
                   {'nodes':[{'id':'X','label':'X','criteria':'X','completed':3,'total':2}]}, {'confirmations':0}):
        try: ProgressDefinition.model_validate({**definition(),**change})
        except ValueError: pass
        else: raise AssertionError('Invalid progress configuration accepted')
    store=Workspace(root)
    frame=Frame(np.full((240,320,3),170,np.uint8),0,0,0.)
    observer=ProgressObserver(config,store,{'camera1':frame})
    assert observer.parse('{"node_id":"UNKNOWN","reason":"Occluded"}')['node_id']=='UNKNOWN'
    for raw in ('{"node_id":"invented","reason":"X"}', '{"node_id":"BOTH","reason":"X","status":"SUCCESS"}'):
        try: observer.parse(raw)
        except ValueError: pass
        else: raise AssertionError('Invalid progress output accepted')
    snapshot=type('Snapshot',(),{'views':{'camera1':frame}})()
    content=observer.current_content([(0.,snapshot),(.1,snapshot)])
    assert sum(p['type']=='image_url' for p in content)==2
    assert 'oldest to newest' in observer.prompt
    tracker.state.update(node_id='BOTH',label='Both groups sorted',quality='confirmed',completed=2,total=2,confirmed_captured_time_s=0.)
    image=np.full((1080,1920,3),170,np.uint8)
    status={'status':'FAIL','failure_type':'WRONG_POSITIONING'}
    with patch('robobrain.presentation.cv2.putText'):
        background=annotate_frame(image,0,status,show_frame_numbers=False,progress=tracker.state,elapsed_s=.5)
    annotated=annotate_frame(image,0,status,show_frame_numbers=False,progress=tracker.state,elapsed_s=.5)
    glyphs=np.any(annotated!=background,axis=2)
    assert glyphs.any() and np.all(np.any(background!=image,axis=2)[glyphs])
    assert np.all(image==170)

    # Builder's progress object must compile outside the outcome-specialist collection.
    builder=TaskBuilder(store)
    base=store.get('insert_plug')
    builder.data.update(config=base['config'],inputs={k:base[k] for k in ('input_mode','sources')})
    draft=Draft.model_validate({'name':'Progress draft','summary':'A task with progress','specialists':[{
        'name':'Success','prompt':'Detect successful completion from the visible task scene.','failure_type':None,
        'camera_sources':['camera1'],'reference_assets':[]}], 'progress':{
        'prompt':'Track how many groups are sorted.','nodes':definition()['nodes'],
        'camera_sources':['camera1'],'reference_assets':[],'unit_label':'groups','history_frames':3}})
    built=builder.compile(draft)
    assert len(built['specialists'])==1 and built['progress']['enabled'] and len(built['progress']['nodes'])==4


def runtime_checks(root, native=False):
    app=create_app(root)
    original=app.state.store.get('insert_plug')
    client_type=httpx.AsyncClient
    control={'node':'BOTH','positive':False,'delay':.12,'invalid':False,'hang':False}
    normal,progress=[],[]
    active,peak=0,0
    def schema(payload):
        return payload['format'] if native else payload['response_format']['json_schema']['schema']
    async def model(request):
        nonlocal active,peak
        if request.url.path.endswith('/models'): return response(request)
        if request.url.path.endswith(('/tags','/show')): return native_reply(request)
        payload=json.loads(request.content)
        if 'node_id' in schema(payload)['properties']:
            progress.append(payload)
            active+=1
            peak=max(peak,active)
            try:
                await asyncio.sleep(control['delay'])
                if control['hang']: await asyncio.Event().wait()
                raw='not JSON' if control['invalid'] and not native else json.dumps({'node_id':control['node'],'reason':'Test scene evidence'})
                if native:
                    return httpx.Response(200,json={'message':{'content':raw},'done':not control['invalid'],'done_reason':'stop'})
                return httpx.Response(200,json={'choices':[{'message':{'content':raw}}]})
            finally: active-=1
        normal.append(payload)
        await asyncio.sleep(.02)
        if native: return native_reply(request,positive=control['positive'])
        if control['positive'] and 'verdict' in payload['response_format']['json_schema']['schema']['properties']:
            return httpx.Response(200,json={'choices':[{'message':{'content':'{"verdict":"POSITIVE","reason":"Test completion"}'}}]})
        return response(request)
    def factory(**kwargs): return client_type(**kwargs,transport=httpx.MockTransport(model))
    with patch('robobrain.runtime.LiveCamera',FakeCamera),patch('robobrain.runtime.httpx.AsyncClient',factory):
        with TestClient(app,base_url='http://127.0.0.1',headers={'X-RoboBrain':'1'}) as client:
            task=copy.deepcopy(original)
            task['config']['evidence_horizon']=0
            if native: task['config'].update(base_url='http://localhost:11434/api',model='gemma4:31b')
            for spec in task['specialists']:
                for block in spec['blocks']:
                    if block['type']=='camera':block['crop']=None
            task['progress']=definition()
            assert client.put('/api/tasks/'+task['id'],json=task).status_code==200
            assert client.post('/api/tasks/'+task['id']+'/sources/camera1/connect').status_code==200
            preview=client.get('/api/tasks/'+task['id']+'/progress/request')
            assert preview.status_code==200 and 'node_id' in preview.json()['schema']['properties']
            assert client.post('/api/run',json={'task_id':task['id'],'record':True}).status_code==200
            def wait(predicate):
                deadline=time.monotonic()+6
                while time.monotonic()<deadline:
                    state=client.get('/api/state').json()
                    if predicate(state):return state
                    time.sleep(.02)
                raise AssertionError(state)
            state=wait(lambda s:s.get('progress') and s['progress']['node_id']=='BOTH')
            assert state['result']['status']=='ON_GOING', 'Final progress node must not cause SUCCESS'
            assert len(normal)>len(progress)*3 and peak==1, 'Progress must neither serialize outcomes nor overlap itself'
            assert client.put('/api/tasks/'+task['id'],json=task).status_code==409
            control['node']='BRICKS'
            state=wait(lambda s:s['progress']['node_id']=='BRICKS')
            assert state['progress']['completed']==1
            control['invalid']=True
            state=wait(lambda s:s['progress']['quality']=='unknown')
            assert state['busy'] and state['result']['status']=='ON_GOING'
            control['positive']=True
            state=wait(lambda s:s['result']['status']=='SUCCESS')
            assert state['progress']['quality']!='confirmed'
            control['hang']=True
            state=wait(lambda s:s['progress']['quality']=='stale')
            before=len(normal)
            time.sleep(.15)
            assert len(normal)>before
            stopped=time.monotonic()
            client.post('/api/run/stop')
            state=wait(lambda s:not s['busy'])
            assert state['phase']=='finished' and time.monotonic()-stopped<3 and active==0,state
            assert state['result']['status']=='SUCCESS'
            assert sorted(Path(p).suffix for p in state['output'])==['.MOV','.json','.log']
            report=json.loads(next(root/p for p in state['output'] if p.endswith('.json')).read_text())
            assert report['progress']['requests'] and report['progress']['request_count']==len(progress)
            assert all('verdict' in r for r in report['requests'])
            assert all('node_id' in r for r in report['progress']['requests'])
            assert not any('node_id' in schema(p)['properties'] for p in normal)
            raw_log=[json.loads(line) for line in next(root/p for p in state['output'] if p.endswith('.log')).read_text().splitlines()]
            raw=next(e for e in raw_log if e['type']=='progress_state' and e['quality']=='confirmed')
            readable=next(e for e in report['events'] if e['type']=='progress_state' and e['quality']=='confirmed')
            for key in ('source_frames','confirmed_source_frames'):
                assert readable[key]['camera1']['source_frame']==raw[key]['camera1']['source_frame']+1


if __name__=='__main__':
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        tracker_checks(root/'unit')
        runtime_checks(root/'runtime')
        runtime_checks(root/'native',native=True)
    print('Progress checks passed: node validation, unknown/stale evidence, duplicate frames, skips/regressions, history, builder compilation, independent outcomes, cancellation and recordings.')
