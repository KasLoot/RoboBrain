"""Opt-in VLM check on labelled, unannotated images; sends requests to the task's model.

uv run python tests/check_progress_model.py WORKSPACE CASES_JSON
CASES_JSON: {"task_id":"...", "cases":[{"image":"relative.jpg","node_id":"EXPECTED"}]}
Use static arrangements with one camera. Images must not contain recorded status overlays.
"""

import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import httpx

from robobrain.monitor import Backend, Frame
from robobrain.progress import ProgressObserver
from robobrain.workspace import ProgressDefinition, Workspace


async def check(workspace, manifest_path):
    store=Workspace(workspace)
    manifest=json.loads(manifest_path.read_text())
    assert manifest['cases'], 'Provide at least one labelled image'
    task=store.get(manifest['task_id'])
    definition=ProgressDefinition.model_validate(task['progress'])
    sources={b.source for b in definition.blocks if b.enabled and b.type=='camera'}
    assert len(sources)==1 and definition.history_frames==1, 'Use single-camera, single-frame progress definitions'
    source=next(iter(sources))
    results=[]
    async with httpx.AsyncClient(trust_env=False) as client:
        backend=Backend({**task['config'],'max_tokens':definition.max_tokens},client,specialists={})
        await backend.check_model()
        for case in manifest['cases']:
            image=cv2.imread(str(manifest_path.parent/case['image']))
            assert image is not None,case['image']
            frame=Frame(image,0,0,0.)
            observer=ProgressObserver(definition,store,{source:frame})
            response=await backend.complete(observer,[(0.,SimpleNamespace(views={source:frame}))])
            response.raise_for_status()
            body=response.json()
            message=body['message'] if backend.native else body['choices'][0]['message']
            assert not any(message.get(k) for k in ('thinking','reasoning','reasoning_content')), 'Unexpected reasoning output'
            if backend.native:
                assert body.get('done') is True and body.get('done_reason')!='length', 'Incomplete model output'
            observed=observer.parse(message['content'])
            result={'image':case['image'],'expected':case['node_id'],**observed}
            results.append(result)
            print(json.dumps(result),flush=True)
    failures=[r for r in results if r['node_id']!=r['expected']]
    assert not failures,failures
    print(f'Progress VLM checks passed: {len(results)} labelled camera scenes.')


if __name__=='__main__':
    asyncio.run(check(sys.argv[1],Path(sys.argv[2])))
