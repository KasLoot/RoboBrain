// node tests/check_frontend.mjs — exercise task rendering without a browser or dependencies.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const source = readFileSync(new URL('../src/robobrain/static/app.js', import.meta.url), 'utf8');
const boundary = source.indexOf("app.addEventListener('input'");
assert.ok(boundary > 0, 'Cannot locate the frontend startup boundary');
const app = { innerHTML: '', querySelector: () => null };
const context = vm.createContext({
  document: { querySelector: selector => selector === '#app' ? app : {querySelectorAll:()=>[]}, querySelectorAll:()=>[] }
});
vm.runInContext(source.slice(0, boundary), context);
const run = code => vm.runInContext(code, context);

run(`task = {
  id: 'test', name: 'Test', specialists: [{
    id: 'gesture', name: 'Gesture', enabled: true,
    definition: {field: 'verdict', values: {POSITIVE: 'POSITIVE'}, failure_type: null},
    blocks: [{type: 'text', enabled: true, text: 'Detect a gesture'},
             {type: 'camera', enabled: true, source: 'camera1'}]
  }]
}; renderSpecialists();`);
assert.match(app.innerHTML, /specialist-card/);
assert.match(app.innerHTML, /Configured/);
assert.equal(run('readyToRun()'), true);

run('task.specialists[0].definition = null; renderSpecialists();');
assert.match(app.innerHTML, /Draft/);
assert.match(app.innerHTML, /Missing configuration JSON/);
assert.equal(run('readyToRun()'), false);
run('task.specialists[0].definition = {}; renderSpecialists();');
assert.match(app.innerHTML, /Missing configuration JSON/);

run(`task.specialists[0].definition = {field: 'verdict', values: {UNCERTAIN: 'UNCERTAIN'}, failure_type: null};
task.specialists[0].blocks.push({type: 'image', enabled: true, asset: null}); renderSpecialists();`);
assert.match(app.innerHTML, /Reference image not uploaded/);
assert.equal(run('readyToRun()'), false);
run('task.specialists[0].blocks[2].enabled = false;');
assert.equal(run('readyToRun()'), true);
run(`task.specialists.push({id: 'draft', name: 'Draft', enabled: false, definition: null, blocks: []}); renderSpecialists();`);
assert.equal(run('readyToRun()'), true, 'Disabled drafts must not block configured specialists');
run('task.specialists = []; renderSpecialists();');
assert.match(app.innerHTML, /Add your first specialist/);
assert.equal(run('readyToRun()'), false);
run(`page = 'builder'; builder = {
  settings:{model:'kimi-k3:cloud',base_url:'https://ollama.com/api'},key_configured:true,
  inputs:{base_task_id:null,input_mode:'camera',sources:[{id:'camera1',name:'Camera 1',device:'0',enabled:true}]},
  draft:null,messages:[],attachments:[],usage:[],phase:'idle',busy:false
}; renderBuilder();`);
assert.match(app.innerHTML, /Your specialists will appear here/);
assert.doesNotMatch(app.innerHTML, /data-action="builder-create"/);
run(`builder.draft = {name:'<img src=x onerror=alert(1)>',summary:'A generated task',specialists:[{
  name:'Success',prompt:'Detect complete placement from visible evidence.',failure_type:null,
  camera_sources:['camera1'],reference_assets:[]
}]}; renderBuilder();`);
assert.match(app.innerHTML, /data-action="builder-create"/);
assert.match(app.innerHTML, /&lt;img src=x onerror=alert\(1\)&gt;/);
assert.doesNotMatch(app.innerHTML, /<img src=x/);
assert.match(app.innerHTML, /Configuration JSON/);
run('builder.created = true; renderBuilder();');
assert.doesNotMatch(app.innerHTML, /data-action="builder-create"/);
assert.match(app.innerHTML, /Open created task/);

run(`page='progress'; task.sources=[{id:'camera1'}]; task.progress=newProgress();
task.progress.nodes=[{id:'PLACED',label:'<Placed>',criteria:'The object is resting inside its box.',completed:1,total:2}];`);
let editor=run('progressEditor(task.progress)');
assert.match(editor, /Enable progress/);
assert.match(editor, /&lt;Placed&gt;/);
assert.match(editor, /data-progress-field="completed" value="1"/);
assert.match(editor, /data-progress-setting="confirmations"/);
run('state.busy=true;');
assert.match(run('progressEditor(task.progress)'), /data-edit disabled/);

let generatedProgress='';
app.querySelector=selector=>selector==='.builder-preview > .hint'
  ? {insertAdjacentHTML:(_,html)=>generatedProgress=html}
  : selector==='#progress-enabled' ? {closest:()=>({remove(){}})} : null;
run(`page='builder'; state.busy=false; builder.created=false;
builder.draft.progress={prompt:'Track visible placements.',nodes:task.progress.nodes,camera_sources:['camera1'],reference_assets:[],unit_label:'groups',history_frames:1};
renderBuilder();`);
assert.match(generatedProgress,/Progress task context/);
assert.match(generatedProgress,/data-builder-progress-list="camera_sources"/);
assert.match(generatedProgress,/&lt;Placed&gt;/);
run('builder.created=true; renderBuilder();');
assert.doesNotMatch(app.innerHTML,/data-action="builder-create"/);
app.querySelector=()=>null;

// The progress panel must not change the independently supplied outcome.
const nodes=new Map();
const element=selector=>{
  if (!nodes.has(selector)) nodes.set(selector,{innerHTML:'',dataset:{},querySelectorAll:()=>[]});
  return nodes.get(selector);
};
context.document.querySelector=element;
run(`page='monitor'; state.busy=false; task.config={evidence_horizon:0}; task.progress.enabled=true;
state.task_id=task.id; state.result={status:'FAIL',failure_type:'WRONG_POSITIONING'};
state.progress={node_id:'PLACED',label:'<Placed>',quality:'confirmed',completed:1,total:2,unit_label:'groups',percent:50,
reason:'<visible evidence>',age_s:.5,requests:3,responses:3,history:[]}; updateMonitor();`);
assert.equal(element('#task-status').textContent,'FAIL');
assert.match(element('#progress-panel').innerHTML,/1\/2 groups · 50%/);
assert.match(element('#progress-panel').innerHTML,/&lt;visible evidence&gt;/);
run(`state.progress.quality='stale'; state.progress.percent=null; updateMonitor();`);
assert.match(element('#progress-panel').innerHTML,/last known/);
assert.doesNotMatch(element('#progress-panel').innerHTML,/50%|value="1"/);
run(`state.progress=null; task.progress=null; updateMonitor();`);
assert.equal(element('#progress-panel').hidden,true);
// Robot lifecycle controls remain independent of monitoring and progress state.
run(`task.input_mode='mujoco'; task.simulation=defaultSimulation();
task.sources=[{id:'camera1',name:'Camera 1',width:640,height:480,fps:15,camera:freeCamera(),enabled:true}];
state.busy=true; state.simulation={connected:true,service_state:'READY',executor_state:'RUNNING',physics_healthy:true,hold_applied:false,
views:{camera1:{camera_revision:3,camera:freeCamera()}},sim_time_s:12.25,episode_id:'scene-1'};
updateSimulationUI();`);
assert.equal(element('#robot-stop').disabled,false,'Robot stop must remain usable during monitoring');
assert.equal(element('#robot-start').disabled,true);
assert.equal(element('#robot-reset').disabled,true,'A running robot cannot be reset');
run(`state.simulation.executor_state='STOPPING';updateSimulationUI();`);
assert.match(element('#simulation-detail').textContent,/awaiting applied stop/);
run(`state.simulation.executor_state='STOPPED';updateSimulationUI();`);
assert.equal(element('#robot-start').disabled,false,'Monitoring must not prevent resume');
assert.equal(element('#robot-start').textContent,'Resume robot');
assert.equal(element('#robot-reset').disabled,false);
run(`state.busy=false; state.simulation.executor_state='ERROR';updateSimulationUI();`);
assert.equal(element('#robot-reset').disabled,true,'Error reset requires applied hold');
run(`state.simulation.hold_applied=true;updateSimulationUI();`);
assert.equal(element('#robot-reset').disabled,false);
run(`state.simulation.connected=false;updateSimulationUI();`);
assert.equal(element('#robot-start').disabled,true);
assert.equal(element('#robot-stop').disabled,true);
run(`state.busy=true;`);
const monitorCamera=run("cameraFields(task.sources[0])");
assert.match(monitorCamera,/Apply live view/);
assert.match(monitorCamera,/data-action="simulation-preset"[^>]*data-edit disabled/);
assert.doesNotMatch(monitorCamera,/data-camera-field="[^"]+"[^>]*data-edit/,'Pose fields must remain live while the task is locked');
assert.match(run("cameraFields(task.sources[0],'builder')"),/data-camera-field="x"[^>]*data-edit disabled/);
run(`state.busy=false; page='builder'; builder.draft=null; builder.created=false;
setInputMode(builder.inputs,'mujoco'); renderBuilder();`);
assert.match(app.innerHTML,/MuJoCo simulation/);
assert.match(app.innerHTML,/builder-simulation-connect/);
assert.match(app.innerHTML,/Capture scene reference/);
assert.equal(run('builder.inputs.sources[0].width'),640);
assert.equal(run('builder.inputs.sources[0].height'),480);
assert.equal(run('builder.inputs.simulation.auto_stop'),'off');
console.log('Frontend checks passed: rendering, escaped content, builder drafts, independent progress, robot lifecycle states and live-camera/configuration locks.');
