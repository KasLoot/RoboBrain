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
console.log('Frontend checks passed: task readiness, builder draft rendering, escaped model content and created-task state.');
