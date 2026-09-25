from copy import deepcopy
from types import SimpleNamespace
import threading
import numpy as np
import pytest
from core.runners.real import RealEpisodeRunner
from core.v0_types import Subgoal,V0Config
from core.vlm.vlm_client import VLMResponse
from core.sim.service_runtime import ExecutionCancelled,ServiceRuntime
from core.sim.stage_progress import stage_progress_text

class Session:
    skip_initial_release=True
    control_mode='mujoco'
    scene='plug_insert'
    recorder=None
    def __init__(self): self.pose=np.array([.48,-.1,.106,1,0,0,0]); self.checkpoint=None
    def get_observation(self): return {'agentview':np.zeros((32,32,3),np.uint8),'ee_pose':self.pose.copy(),'gripper_width':.032}
    def load_runner_checkpoint(self): return deepcopy(self.checkpoint)
    def save_runner_checkpoint(self,c): self.checkpoint=deepcopy(c)
    def check_success(self): return False

class Logger:
    def __init__(self,path): self.run_dir=path;path.mkdir()
    def __getattr__(self,name): return lambda *a,**k:None

class Controller:
    gripper_closed=True
    z_floor_m=.025
    def __init__(self,session):self.session=session;self.actions=[]
    def step(self,token,**kw):
        self.actions.append(token)
        if token=='MV_UP_LARGE':self.session.pose[2]+=.0919
        return SimpleNamespace(done=token=='DONE',kind='move' if token.startswith('MV_') else 'done')

class Planner:
    def __init__(self):self.calls=0
    def plan(self,*a,**kw):
        self.calls+=1
        return [Subgoal(x,x,'handle',x,'Observe stage','Confirm visually') for x in ('GRASP','LIFT','RETREAT')],'test plan'


def runner(session,planner,decide,path):
    controller=Controller(session)
    r=RealEpisodeRunner(session=session,controller=controller,planner=planner,controls=SimpleNamespace(controller=SimpleNamespace(decide=decide)),logger=Logger(path),config=V0Config(70,1,2,controller_prompt_log_every=0),task='Insert plug',gripper_color='black',max_steps=10,loop_period_s=0,use_wrist_image=False,debug=False)
    r._show_live=lambda *a,**k:None;r._print_plan=lambda *a:None;r._print_step=lambda **k:None
    r._record=lambda **k:{};r._write_action_table=lambda *a:None;r._success_diagnostics=lambda:None
    return r,controller


def test_resume_retains_cursor_anchor_and_budget(tmp_path):
    session=Session();planner=Planner();calls=[]
    def first(ctx,**kw):
        calls.append(ctx)
        if len(calls)==3:raise ExecutionCancelled('paused during inference')
        return VLMResponse('DONE' if ctx.subgoal.motion=='GRASP' else 'MV_UP_LARGE','',{})
    r,_=runner(session,planner,first,tmp_path/'first')
    with pytest.raises(ExecutionCancelled):r.run()
    assert planner.calls==1
    assert session.checkpoint['current_index']==1 and session.checkpoint['next_step']==2
    assert session.checkpoint['stage_anchor'][2]==pytest.approx(.106)
    seen=[]
    def resumed(ctx,**kw):
        seen.append(ctx)
        return VLMResponse('DONE','',{})
    r,controller=runner(session,planner,resumed,tmp_path/'resumed')
    result=r.run()
    assert planner.calls==1,'Resume replanned completed milestones'
    assert [ctx.subgoal.motion for ctx in seen]==['LIFT','RETREAT']
    assert seen[0].step_idx==2 and '91.9' in seen[0].task
    assert 'remaining nominal TCP travel=0.00 mm' in seen[0].task
    assert controller.actions==['DONE','DONE']
    assert result.steps==4
    assert not result.success,'TCP movement must not establish object/task success'


def test_stale_runner_cannot_rewind_checkpoint():
    r=ServiceRuntime.__new__(ServiceRuntime);r._lock=threading.RLock();r._closed=threading.Event();r._enabled=r._healthy=True;r._generation=3;r._attempt_id='attempt';r._runner_checkpoint={'current_index':2}
    with pytest.raises(ExecutionCancelled):r.save_runner_checkpoint(1,'attempt',{'current_index':0})
    assert r.load_runner_checkpoint(3,'attempt')['current_index']==2


def test_stage_travel_is_not_assumed_for_other_tasks():
    text=stage_progress_text('LIFT',[0,0,0],[0,0,.092])
    assert 'nominal' not in text
    assert 'does not prove plug travel' in text


def test_cancelled_generation_can_finalize_without_reading_new_state():
    def obsolete(): raise ExecutionCancelled('old generation')
    r=RealEpisodeRunner.__new__(RealEpisodeRunner)
    r.session=SimpleNamespace(success_diagnostics=obsolete)
    assert r._success_diagnostics() is None
