const app = document.querySelector('#app'),
  nav = document.querySelector('#nav'),
  dialog = document.querySelector('#modal');
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;'
} [c]));
const clone = v => structuredClone(v),
  uid = p => p + '_' + crypto.randomUUID().replaceAll('-', '').slice(0, 12);
let tasks = [],
  task = null,
  specialist = null,
  selectedBlock = null,
  dirty = false,
  record = false,
  pending = false,
  page = 'tasks',
  lastHash = '#tasks';
let state = {
    busy: false,
    phase: 'idle',
    sources: {},
    specialists: {},
    result: {},
    output: []
  },
  toastTimer;

function toast(message) {
  const el = document.querySelector('#toast');
  el.textContent = message;
  el.style.display = 'block';
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.style.display = 'none', 6500);
}
async function api(path, method = 'GET', body) {
  const options = {
    method,
    headers: {
      'X-RoboBrain': '1'
    }
  };
  if (body instanceof FormData) options.body = body;
  else if (body !== undefined) {
    options.headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(body);
  }
  const r = await fetch('/api' + path, options),
    v = await r.json();
  if (!r.ok) throw new Error(typeof v.detail === 'string' ? v.detail : JSON.stringify(v.detail || v));
  return v;
}

function ask(title, body, action, label = 'Save', danger = false) {
  dialog.innerHTML = `<form><div class="dialog-head"><h2>${esc(title)}</h2><button type="button" data-close aria-label="Close dialog">×</button></div>${body}<div class="dialog-actions"><button type="button" data-close>Cancel</button><button type="submit" class="${danger?'danger':'primary'}">${esc(label)}</button></div></form>`;
  dialog.querySelectorAll('[data-close]').forEach(b => b.onclick = () => dialog.close());
  dialog.querySelector('form').onsubmit = async e => {
    e.preventDefault();
    e.submitter.disabled = true;
    try {
      await action(new FormData(e.target));
      dialog.close();
    } catch (error) {
      toast(error.message);
    } finally {
      e.submitter.disabled = false;
    }
  };
  dialog.showModal();
}

function confirmDelete(name, action) {
  ask('Delete ' + name + '?', `<p>This removes ${esc(name)}. Existing recordings and shared images are kept.</p>`, action, 'Delete', true);
}

function validateEditor() {
  const invalid = app.querySelector(':invalid');
  if (invalid) {
    invalid.reportValidity();
    throw new Error('Correct the highlighted field first.');
  }
}

function specialistIssues(s) {
  const blocks = s.blocks.filter(b => b.enabled);
  return [
    (!s.definition || !Object.keys(s.definition).length) && 'Missing configuration JSON',
    !blocks.some(b => b.type === 'text' && b.text.trim()) && 'Missing system prompt',
    !blocks.some(b => b.type === 'camera') && 'Missing camera/video block',
    blocks.some(b => b.type === 'image' && !b.asset) && 'Reference image not uploaded'
  ].filter(Boolean);
}

function readyToRun() {
  const active = task?.specialists.filter(s => s.enabled) || [];
  return active.length > 0 && active.every(s => specialistIssues(s).length === 0);
}

function markDirty() {
  dirty = true;
  updateLocks();
}
async function saveTask() {
  validateEditor();
  if (state.busy) throw new Error('Configuration is locked during a run.');
  const sid = specialist?.id;
  pending = true;
  updateLocks();
  try {
    task = await api('/tasks/' + task.id, 'PUT', task);
  } finally {
    pending = false;
  }
  tasks = await api('/tasks');
  dirty = false;
  specialist = sid ? task.specialists.find(s => s.id === sid) : null;
  updateLocks();
  toast('Task saved');
}
const link = (view, id = task?.id) => '#task/' + id + '/' + view;
const edit = () => `data-edit ${state.busy?'disabled':''}`;

function badges(s) {
  const ready = specialistIssues(s).length === 0;
  return `<span class="badge ${!s.enabled?'neutral':ready?'good':'warning'}">${!s.enabled?'Disabled':ready?'Configured':'Draft'}</span>`;
}

function header(title, description, actions = '') {
  return `<div class="page-head"><div><a class="breadcrumb" href="#tasks">Tasks</a><span class="breadcrumb"> / ${esc(task.name)}</span><h1>${esc(title)}</h1>${description?`<p>${esc(description)}</p>`:''}</div><div class="actions">${actions}</div></div><div id="lock-banner" class="lock-banner" ${state.busy?'':'hidden'}>Configuration is read-only while monitoring or finalizing a recording.</div>`;
}

function menu(kind, id, name) {
  return `<details class="menu"><summary aria-label="Actions for ${esc(name)}">⋯</summary><div>${['rename','duplicate','delete'].map(a=>`<button data-action="${a}-${kind}" data-id="${esc(id)}" ${edit()}>${a[0].toUpperCase()+a.slice(1)}</button>`).join('')}</div></details>`;
}

function renderTasks() {
  nav.innerHTML = '';
  app.innerHTML = `<div class="page-head"><div><div class="eyebrow">Workspace</div><h1>Tasks</h1><p>Saved setups, ready for another run.</p></div><button class="primary" data-action="new-task" ${edit()}>＋ New task</button></div><div class="grid">${tasks.map(t=>`<article class="card task-card"><div class="card-top"><span class="badge">Task template</span>${menu('task',t.id,t.name)}</div><h2>${esc(t.name)}</h2><p>${t.specialists.length} specialists · ${t.specialists.filter(s=>s.enabled).length} enabled</p><a class="button task-open" href="${link('specialists',t.id)}">Open task <span>→</span></a></article>`).join('')||'<div class="empty card"><h2>No tasks yet</h2><p>Create a task to start building its specialists.</p></div>'}</div>`;
}

function renderSpecialists() {
  app.innerHTML = header('Specialists', 'Each specialist keeps its own prompt and visual evidence.', `<span id="save-state"></span><button data-action="save" ${edit()}>Save task</button><button class="primary" data-action="new-specialist" ${edit()}>＋ New specialist</button>`) + `<div class="grid">${task.specialists.map(s=>`<article class="card specialist-card" data-specialist="${esc(s.id)}"><div class="card-top">${badges(s)}${menu('specialist',s.id,s.name)}</div>${s.enabled&&specialistIssues(s).length?`<p class="draft-reason">${esc(specialistIssues(s).join('; '))}.</p>`:''}<h2><a href="${link('designer/'+s.id)}">${esc(s.name)}</a></h2><p>${s.blocks.filter(b=>b.type==='image').length} reference images · ${s.blocks.filter(b=>b.type==='camera').length} views</p><div class="reference-strip">${s.blocks.filter(b=>b.type==='image'&&b.asset).slice(0,3).map(b=>`<img src="/api/assets/${encodeURIComponent(b.asset)}" alt="${esc(b.alias)}">`).join('')}</div><div class="card-bottom"><label class="switch"><input type="checkbox" data-specialist-enabled="${esc(s.id)}" ${s.enabled?'checked':''} ${edit()}> Enabled</label><a class="text-link" href="${link('designer/'+s.id)}">Edit prompt →</a></div></article>`).join('')||'<div class="empty card"><h2>Add your first specialist</h2><p>Start empty or duplicate a configured specialist.</p></div>'}</div>`;
}

function renderDesigner() {
  if (!specialist) {
    location.hash = link('specialists');
    return;
  }
  const block = specialist.blocks.find(b => b.id === selectedBlock) || specialist.blocks[0];
  selectedBlock = block?.id;
  app.innerHTML = header(specialist.name, 'Prompt Designer', `<span id="save-state"></span><button data-action="preview-request" id="preview-request">Preview model request</button><button class="primary" data-action="save" ${edit()}>Save specialist</button>`) + `<div class="designer"><aside class="block-list"><div class="section-head"><h2>Prompt blocks</h2><button data-action="new-block" ${edit()}>＋ New block</button></div><p class="hint">Text forms the system prompt. Images and current views follow the block order.</p>${specialist.blocks.map((b,i)=>`<article class="block ${b.id===selectedBlock?'selected':''} ${b.enabled?'':'muted'}"><button class="block-select" data-action="select-block" data-id="${esc(b.id)}"><span class="block-type">${b.type==='text'?'T':b.type==='image'?'▧':'◉'}</span><span><strong>${b.type==='camera'?'Camera / video':b.type[0].toUpperCase()+b.type.slice(1)}</strong><small>${esc(b.alias||'Untitled block')}</small></span></button><div class="block-preview">${b.type==='text'?esc(b.text.slice(0,150)||'Write your system prompt…'):b.type==='image'&&b.asset?`<img src="/api/assets/${encodeURIComponent(b.asset)}" alt="${esc(b.alias)}">`:b.type==='camera'?esc(task.sources.find(s=>s.id===b.source)?.name||'Select source'):'No image uploaded'}</div><div class="block-actions"><label class="switch"><input type="checkbox" data-block-enabled="${esc(b.id)}" ${b.enabled?'checked':''} ${edit()}> Enabled</label><div><button aria-label="Move block up" data-action="move-up" data-id="${esc(b.id)}" ${edit()} ${i===0?'disabled':''}>↑</button><button aria-label="Move block down" data-action="move-down" data-id="${esc(b.id)}" ${edit()} ${i===specialist.blocks.length-1?'disabled':''}>↓</button><button aria-label="Delete block" data-action="delete-block" data-id="${esc(b.id)}" ${edit()}>×</button></div></div></article>`).join('')}<details class="detector-config" ${!specialist.definition||!Object.keys(specialist.definition).length?'open':''}><summary>Existing detector configuration</summary><p class="hint">The same field, values and failure_type JSON used by RoboBrain. No decision rules are inferred from the name.</p><label>Configuration JSON<textarea id="definition" rows="10" spellcheck="false" ${edit()}>${specialist.definition?esc(JSON.stringify(specialist.definition,null,2)):''}</textarea></label><label>Existing evidence threshold<input type="number" min="0" id="threshold" value="${specialist.threshold}" ${edit()}></label></details></aside><section class="card inspector">${block?blockEditor(block):'<div class="empty"><h2>No block selected</h2><p>Add a text, image, or camera block.</p></div>'}</section></div>`;
  installCrop();
}

function blockEditor(b) {
  const image = b.type === 'image';
  return `<div class="section-head"><h2>${b.type==='text'?'System prompt':image?'Reference image':'Current view'}</h2><span class="badge">${b.enabled?'Enabled':'Disabled'}</span></div><label>Alias${b.type==='text'?'':' / label sent to the model'}<input data-block-field="alias" value="${esc(b.alias)}" ${edit()}></label>${b.type==='text'?`<label>Prompt text<textarea class="prompt-text" data-block-field="text" spellcheck="false" ${edit()}>${esc(b.text)}</textarea></label>`:image?`<label class="upload-zone" data-upload-kind="image"><input type="file" accept="image/jpeg,image/png" id="image-file" ${edit()}><strong>Choose an image or drop it here</strong><span>JPEG or PNG · up to 50 MB</span></label>${b.asset?`<div class="image-stage" id="crop-stage"><img id="crop-image" src="/api/assets/${encodeURIComponent(b.asset)}" alt="${esc(b.alias)}"><div id="crop-box" hidden></div></div>`:'<div class="empty preview-empty">No reference image</div>'}`:`<label>Source<select data-block-field="source" ${edit()}>${task.sources.map(s=>`<option value="${esc(s.id)}" ${s.id===b.source?'selected':''}>${esc(s.name)}</option>`).join('')}</select></label><div class="source-link"><span>Configure devices and videos on Monitor.</span><a class="text-link" href="${link('monitor')}">Open Monitor →</a></div><div class="image-stage" id="crop-stage"><img id="crop-image" data-source="${esc(b.source)}" alt="Camera preview"><div id="crop-box" hidden></div><span class="preview-notice">Connect this source on Monitor to see its preview.</span></div>`}${b.type!=='text'?`<div class="section-head crop-head"><h2>Crop</h2><button data-action="reset-crop" ${edit()}>Use full image</button></div><p class="hint">Drag on the image or enter original-image pixel coordinates. Right and bottom edges are exclusive.</p><div class="crop-fields">${['Left (x)','Top (y)','Right (x)','Bottom (y)'].map((label,i)=>`<label>${label}<input type="number" min="0" data-crop="${i}" value="${b.crop?.[i]??''}" placeholder="Full image" ${edit()}></label>`).join('')}</div><div class="section-head"><span id="crop-size" class="hint"></span><button data-action="apply-crop" ${edit()}>Apply crop</button></div>`:''}`;
}

function renderMonitor() {
  app.innerHTML = header('Monitor', task.name, `<button data-action="settings" ${edit()}>Model & settings</button>`) + `<div class="monitor-toolbar card"><div class="mode-picker"><label><input type="radio" name="mode" value="camera" ${task.input_mode==='camera'?'checked':''} ${edit()}> Cameras</label><label><input type="radio" name="mode" value="video" ${task.input_mode==='video'?'checked':''} ${edit()}> Video simulation</label></div><div class="actions"><span id="save-state"></span><button data-action="save" ${edit()}>Save task</button><button data-action="connect-all" ${edit()}>Connect inputs</button></div></div><div class="camera-grid">${task.sources.map(s=>`<section class="card source-card"><div class="section-head"><h2>${esc(s.name)}</h2><label class="switch"><input type="checkbox" data-source-enabled="${esc(s.id)}" ${s.enabled?'checked':''} ${edit()}> Enabled</label></div><div class="source-controls">${task.input_mode==='camera'?`<label>Device<input data-source-id="${esc(s.id)}" data-source-field="device" value="${esc(s.device)}" placeholder="0 or /dev/video0" ${edit()}></label><button data-action="connect" data-id="${esc(s.id)}" ${edit()}>Connect</button>`:`<label class="upload-zone compact" data-upload-kind="video" data-source-id="${esc(s.id)}"><input type="file" accept=".mov,.mp4,.avi,.mkv" data-video-source="${esc(s.id)}" ${edit()}><strong>${s.video?'Replace video':'Choose or drop a video'}</strong><span>${esc(s.video||'MOV, MP4, AVI or MKV · up to 2 GB')}</span></label><button data-action="connect" data-id="${esc(s.id)}" ${edit()}>Preview</button>`}</div><div class="camera-preview"><img data-source="${esc(s.id)}" alt="${esc(s.name)} live view"><span class="preview-notice">${s.enabled?'Not connected':'Source disabled'}</span></div><div class="source-health" data-health="${esc(s.id)}">Not connected</div>${task.input_mode==='camera'?`<details class="capture-settings"><summary>Capture settings</summary><div class="capture-fields">${['width','height','fps'].map(k=>`<label>${k==='fps'?'FPS':k[0].toUpperCase()+k.slice(1)}<input type="number" min="1" data-source-id="${esc(s.id)}" data-source-field="${k}" value="${s[k]}" ${edit()}></label>`).join('')}</div></details>`:''}</section>`).join('')}</div><section class="card operating"><div><div class="eyebrow">Operating status</div><div id="task-status" class="task-status">ON_GOING</div><p id="failure-type">Failure type: NONE</p><p id="run-detail" class="hint">Ready to start</p></div><div class="run-actions"><label class="record-choice"><input type="checkbox" id="record" ${record?'checked':''} ${edit()}> Record <span>MOV · LOG · JSON</span></label><div class="actions"><button class="primary start" id="start" data-action="start">▶ Start</button><button class="danger" id="stop" data-action="stop" disabled>■ Stop</button></div><span id="record-status" class="hint"></span><p id="start-reason" class="start-reason" role="status" hidden></p></div></section><div id="run-error" class="error-box" role="alert" hidden></div><div id="verification" class="verification card" hidden></div><section class="card observations"><div class="section-head"><h2>Specialists</h2><span id="request-count" class="hint">No requests yet</span></div><div class="table-wrap"><table><thead><tr><th>Specialist</th><th>Observation</th><th>Visible evidence</th><th>Latency</th></tr></thead><tbody>${task.specialists.filter(s=>s.enabled).map(s=>`<tr data-observation="${esc(s.id)}"><td>${esc(s.name)}</td><td>Waiting</td><td>—</td><td>—</td></tr>`).join('')}</tbody></table></div></section><section id="outputs" class="card" hidden></section>`;
  document.querySelector('.camera-grid').before(document.querySelector('.operating'));
  updateMonitor();
}
async function route() {
  if (pending) {
    toast('Wait for the current operation to finish.');
    location.hash = lastHash;
    return;
  }
  try {
    validateEditor();
  } catch (e) {
    toast(e.message);
    location.hash = lastHash;
    return;
  }
  document.querySelectorAll('img[data-object-url]').forEach(img => URL.revokeObjectURL(img.dataset.objectUrl));
  const parts = location.hash.slice(1).split('/'),
    nextId = parts[0] === 'task' ? parts[1] : null;
  if (state.busy && nextId && nextId !== state.task_id) {
    toast('Task switching is locked during a run.');
    location.hash = lastHash;
    return;
  }
  if (task && nextId !== task.id && dirty) {
    location.hash = lastHash;
    ask('Unsaved changes', '<p>Save your task before switching, or discard your edits.</p>', async () => {
      dirty = false;
      task = null;
      location.hash = parts.join('/');
    }, 'Discard changes', true);
    return;
  }
  if (nextId && task?.id !== nextId) task = clone(tasks.find(t => t.id === nextId));
  if (nextId && !task) {
    location.hash = '#tasks';
    return;
  }
  page = nextId ? (parts[2] || 'specialists') : 'tasks';
  specialist = page === 'designer' ? task.specialists.find(s => s.id === parts[3]) : null;
  lastHash = location.hash || '#tasks';
  nav.innerHTML = task && page !== 'tasks' ? `<a class="${page!=='monitor'?'active':''}" href="${link('specialists')}">Specialists</a><a class="${page==='monitor'?'active':''}" href="${link('monitor')}">Monitor</a>` : '';
  if (page === 'tasks') renderTasks();
  else if (page === 'designer') renderDesigner();
  else if (page === 'monitor') renderMonitor();
  else renderSpecialists();
  updateLocks();
  refreshPreviews();
}

function updateLocks() {
  document.querySelectorAll('[data-edit]').forEach(el => el.disabled = state.busy || pending);
  dialog.querySelectorAll('input,textarea,select,button[type="submit"]').forEach(el => el.disabled = state.busy || pending);
  document.querySelectorAll('#save-state').forEach(el => {
    el.textContent = dirty ? 'Unsaved changes' : 'Saved';
    el.className = dirty ? 'unsaved' : 'hint';
  });
  const banner = document.querySelector('#lock-banner');
  if (banner) banner.hidden = !state.busy;
  const preview = document.querySelector('#preview-request');
  if (preview) {
    preview.disabled = dirty;
    preview.title = dirty ? 'Save before previewing' : '';
  }
  document.querySelectorAll('.task-open').forEach(el => el.setAttribute('aria-disabled', String(state.busy && !el.hash.includes('/' + state.task_id + '/'))));
  if (page === 'monitor') {
    document.querySelector('#start').disabled = state.busy || dirty || pending || !readyToRun();
    document.querySelector('#stop').disabled = !state.busy || state.phase === 'stopping';
    document.querySelector('#record').checked = state.busy ? state.recording : record;
  }
}

function updateMonitor() {
  if (page !== 'monitor') return;
  const relevant = state.task_id === task.id,
    result = relevant ? state.result : {},
    status = document.querySelector('#task-status');
  status.textContent = result.status || 'ON_GOING';
  status.dataset.status = result.status || 'ON_GOING';
  document.querySelector('#failure-type').textContent = 'Failure type: ' + (result.failure_type || 'NONE');
  const phases = {
    idle: 'Ready to start',
    starting: 'Connecting inputs and checking the model…',
    running: 'Monitoring continuously',
    stopping: 'Stopping monitoring…',
    finalizing: 'Finalizing run…',
    finished: 'Run finished',
    error: 'Run ended with an error'
  };
  document.querySelector('#run-detail').textContent = relevant ? `${phases[state.phase]||state.phase} · ${(state.elapsed_s||0).toFixed(1)} s${result.stop_reason==='episode_timeout'?' · Episode timeout':''}` : 'Ready to start';
  document.querySelector('#record-status').textContent = relevant && state.busy && state.recording ? (state.phase === 'finalizing' ? 'Finalizing recording…' : state.phase === 'starting' ? 'Recording enabled' : '● Recording') : '';
  const error = document.querySelector('#run-error');
  error.hidden = !(relevant && state.error);
  error.textContent = state.error || '';
  document.querySelector('#request-count').textContent = relevant ? `${state.responses} responses / ${state.requests} requests` : 'No requests yet';
  for (const row of document.querySelectorAll('[data-observation]')) {
    const o = relevant ? state.specialists[row.dataset.observation] : null,
      c = row.querySelectorAll('td');
    c[1].textContent = o?.verdict || 'Waiting';
    c[1].className = (o?.verdict || '').toLowerCase();
    c[2].textContent = o?.error || o?.reason || '—';
    if (o?.captured_time_s != null) {
      const age = document.createElement('small');
      age.className = 'evidence-age';
      age.textContent = state.busy ? `Evidence captured ${Math.max(0, state.elapsed_s - o.captured_time_s).toFixed(1)} s ago` : `Evidence captured at ${Math.max(0, o.captured_time_s).toFixed(1)} s`;
      c[2].append(age);
    }
    c[3].textContent = o?.latency_s != null ? o.latency_s.toFixed(2) + ' s' : '—';
  }
  const v = relevant ? state.verification : null,
    box = document.querySelector('#verification');
  box.hidden = !v;
  if (v) box.innerHTML = `<div class="section-head"><h2>${v.active?'Verification active':v.cancelled?'Verification cancelled':'Verification finished'}</h2><span class="badge">Round ${v.round_id}</span></div><p>${v.slots.length}/${task.config.evidence_horizon} observation slots · ${v.responses}/${task.config.evidence_horizon*task.specialists.filter(s=>s.enabled).length} responses${v.active?' · Normal monitoring continues':''}</p>${v.counts?`<p>${Object.entries(v.counts).map(([id,count])=>`${esc(task.specialists.find(s=>s.id===id)?.name||id)}: ${count} positive`).join(' · ')}</p>`:''}`;
  const out = document.querySelector('#outputs');
  out.hidden = !(relevant && state.output.length);
  if (!out.hidden) out.innerHTML = `<h2>Run artifacts</h2><div class="artifact-links">${state.output.map(p=>`<a href="/api/output/${p.split('/').map(encodeURIComponent).join('/')}" download>${esc(p.split('/').pop())}</a>`).join('')}</div>`;
  for (const el of document.querySelectorAll('[data-health]')) {
    const info = state.preview_task_id === task.id ? state.sources[el.dataset.health] : null;
    el.textContent = info?.connected ? `Connected · ${info.width} × ${info.height} · latest image ${info.age_s.toFixed(1)} s ago` : info?.error || 'Not connected';
    el.classList.toggle('connected', !!info?.connected);
    const button = el.closest('.source-card').querySelector('[data-action="connect"]');
    if (button && !pending) button.textContent = info?.connected && !dirty ? 'Disconnect' : task.input_mode === 'video' ? 'Preview' : 'Connect';
  }
  updateLocks();
}
async function refreshPreviews() {
  for (const image of document.querySelectorAll('img[data-source]')) {
    const source = state.preview_task_id === task?.id ? state.sources[image.dataset.source] : null,
      notice = image.parentElement.querySelector('.preview-notice');
    if (!source?.connected) {
      image.hidden = true;
      if (notice) notice.hidden = false;
      continue;
    }
    if (image.dataset.loading === '1') continue;
    image.dataset.loading = '1';
    fetch('/api/preview/' + encodeURIComponent(image.dataset.source)).then(async r => {
      if (!r.ok) throw new Error('Preview unavailable');
      const blob = await r.blob();
      if (!image.isConnected) return;
      image.dataset.nativeWidth = r.headers.get('X-Image-Width');
      image.dataset.nativeHeight = r.headers.get('X-Image-Height');
      const url = URL.createObjectURL(blob),
        previous = image.dataset.objectUrl;
      image.onload = () => {
        if (previous) URL.revokeObjectURL(previous);
        drawCrop();
      };
      image.src = url;
      image.dataset.objectUrl = url;
      image.hidden = false;
      if (notice) notice.hidden = true;
    }).catch(() => {}).finally(() => image.dataset.loading = '0');
  }
}

function drawCrop() {
  const block = specialist?.blocks.find(b => b.id === selectedBlock),
    image = document.querySelector('#crop-image'),
    box = document.querySelector('#crop-box');
  if (!block || !image || !box) return;
  const w = Number(image.dataset.nativeWidth) || image.naturalWidth,
    h = Number(image.dataset.nativeHeight) || image.naturalHeight;
  box.hidden = !block.crop || !w;
  if (block.crop && w) {
    const [x1, y1, x2, y2] = block.crop;
    Object.assign(box.style, {
      left: 100 * x1 / w + '%',
      top: 100 * y1 / h + '%',
      width: 100 * (x2 - x1) / w + '%',
      height: 100 * (y2 - y1) / h + '%'
    });
  }
  const size = document.querySelector('#crop-size');
  if (size) size.textContent = block.crop ? `${block.crop[2]-block.crop[0]} × ${block.crop[3]-block.crop[1]} pixels` : (w ? `Full image · ${w} × ${h}` : '');
}

function installCrop() {
  const stage = document.querySelector('#crop-stage'),
    image = document.querySelector('#crop-image');
  if (!stage || !image) return;
  image.addEventListener('load', drawCrop);
  let start = null;

  function position(e) {
    const r = image.getBoundingClientRect(),
      w = Number(image.dataset.nativeWidth) || image.naturalWidth,
      h = Number(image.dataset.nativeHeight) || image.naturalHeight;
    return [Math.max(0, Math.min(w, Math.round((e.clientX - r.left) / r.width * w))), Math.max(0, Math.min(h, Math.round((e.clientY - r.top) / r.height * h)))];
  }
  stage.onpointerdown = e => {
    if (state.busy || image.hidden || !image.naturalWidth) return;
    e.preventDefault();
    start = position(e);
    stage.setPointerCapture(e.pointerId);
  };
  stage.onpointermove = e => {
    if (!start) return;
    const end = position(e),
      b = specialist.blocks.find(b => b.id === selectedBlock);
    b.crop = [Math.min(start[0], end[0]), Math.min(start[1], end[1]), Math.max(start[0], end[0]), Math.max(start[1], end[1])];
    drawCrop();
  };
  stage.onpointerup = () => {
    if (!start) return;
    start = null;
    const b = specialist.blocks.find(b => b.id === selectedBlock);
    if (b.crop && (b.crop[0] === b.crop[2] || b.crop[1] === b.crop[3])) b.crop = null;
    document.querySelectorAll('[data-crop]').forEach(el => el.value = b.crop?.[Number(el.dataset.crop)] ?? '');
    markDirty();
    drawCrop();
  };
}
async function uploadFile(file, kind, sourceId) {
  if (!file || state.busy || pending) return;
  const target = kind === 'image' ? specialist.blocks.find(b => b.id === selectedBlock) : task.sources.find(s => s.id === sourceId);
  const data = new FormData();
  data.append('file', file);
  pending = true;
  updateLocks();
  toast('Uploading…');
  try {
    const uploaded = await api('/assets/' + kind, 'POST', data);
    if (kind === 'image') {
      target.asset = uploaded.asset;
      target.crop = null;
    } else target.video = uploaded.asset;
  } finally {
    pending = false;
    updateLocks();
  }
  markDirty();
  await route();
  toast('Uploaded. Save the task to keep the assignment.');
}
app.addEventListener('input', e => {
  const el = e.target;
  if (state.busy || pending) return;
  if (el.id === 'definition') {
    try {
      specialist.definition = el.value.trim() ? JSON.parse(el.value) : null;
      el.setCustomValidity('');
    } catch {
      el.setCustomValidity('Enter valid JSON.');
    }
    markDirty();
  }
  if (el.dataset.blockField) {
    specialist.blocks.find(b => b.id === selectedBlock)[el.dataset.blockField] = el.value;
    markDirty();
  }
  if (el.dataset.sourceField) {
    task.sources.find(s => s.id === el.dataset.sourceId)[el.dataset.sourceField] = el.dataset.sourceField === 'device' ? el.value : Number(el.value);
    markDirty();
  }
});
app.addEventListener('change', async e => {
  const el = e.target;
  try {
    if (el.id === 'record') {
      record = el.checked;
      return;
    }
    if (state.busy) return;
    if (el.dataset.specialistEnabled) {
      task.specialists.find(s => s.id === el.dataset.specialistEnabled).enabled = el.checked;
      markDirty();
    }
    if (el.dataset.blockEnabled) {
      specialist.blocks.find(b => b.id === el.dataset.blockEnabled).enabled = el.checked;
      markDirty();
      renderDesigner();
      updateLocks();
    }
    if (el.dataset.sourceEnabled) {
      task.sources.find(s => s.id === el.dataset.sourceEnabled).enabled = el.checked;
      markDirty();
    }
    if (el.name === 'mode') {
      task.input_mode = el.value;
      markDirty();
      renderMonitor();
    }
    if (el.dataset.blockField === 'source') {
      renderDesigner();
      updateLocks();
      refreshPreviews();
    }
    if (el.id === 'threshold') {
      specialist.threshold = Number(el.value);
      markDirty();
    }
    if (el.id === 'image-file') await uploadFile(el.files[0], 'image');
    if (el.dataset.videoSource) await uploadFile(el.files[0], 'video', el.dataset.videoSource);
  } catch (error) {
    toast(error.message);
  }
});
app.addEventListener('dragover', e => {
  if (e.target.closest('[data-upload-kind]')) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  }
});
app.addEventListener('drop', e => {
  const zone = e.target.closest('[data-upload-kind]');
  if (zone) {
    e.preventDefault();
    uploadFile(e.dataTransfer.files[0], zone.dataset.uploadKind, zone.dataset.sourceId).catch(error => toast(error.message));
  }
});
app.addEventListener('contextmenu', e => {
  const card = e.target.closest('[data-specialist]');
  if (card) {
    e.preventDefault();
    card.querySelector('details').open = true;
  }
});
app.addEventListener('click', async e => {
  const button = e.target.closest('[data-action]');
  if (!button) return;
  const action = button.dataset.action,
    id = button.dataset.id;
  if (state.busy && button.hasAttribute('data-edit')) return;
  try {
    if (action === 'save') {
      await saveTask();
      return;
    }
    if (['new-task', 'duplicate-task', 'rename-task'].includes(action)) {
      const current = tasks.find(t => t.id === id);
      ask(action === 'rename-task' ? 'Rename task' : action === 'duplicate-task' ? 'Duplicate task' : 'New task', `<label>Task name<input name="name" required maxlength="120" value="${esc(action==='rename-task'?current.name:action==='duplicate-task'?current.name+' copy':'')}" autofocus></label>`, async f => {
        if (action === 'rename-task') await api('/tasks/' + id, 'PUT', {
          ...current,
          name: f.get('name')
        });
        else {
          const created = await api('/tasks', 'POST', {
            name: f.get('name'),
            copy_from: action === 'duplicate-task' ? id : null
          });
          tasks = await api('/tasks');
          task = null;
          dirty = false;
          location.hash = link('specialists', created.id);
        }
        tasks = await api('/tasks');
        if (action === 'rename-task') {
          task = null;
          await route();
        }
      }, action === 'rename-task' ? 'Rename' : 'Create task');
      return;
    }
    if (action === 'delete-task') {
      const current = tasks.find(t => t.id === id);
      confirmDelete(current.name, async () => {
        await api('/tasks/' + id, 'DELETE');
        tasks = await api('/tasks');
        task = null;
        dirty = false;
        location.hash = '#tasks';
        await route();
      });
      return;
    }
    if (['new-specialist', 'rename-specialist', 'duplicate-specialist'].includes(action)) {
      const current = task.specialists.find(s => s.id === id);
      ask(action === 'rename-specialist' ? 'Rename specialist' : 'New specialist', `<label>Specialist name<input name="name" required maxlength="120" value="${esc(action==='rename-specialist'?current.name:action==='duplicate-specialist'?current.name+' copy':'')}" autofocus></label>`, async f => {
        if (action === 'rename-specialist') current.name = f.get('name');
        else {
          const s = action === 'duplicate-specialist' ? clone(current) : {
            enabled: true,
            definition: null,
            threshold: 5,
            blocks: [{
              id: uid('block'),
              type: 'text',
              alias: 'System prompt',
              enabled: true,
              text: '',
              asset: null,
              source: task.sources[0].id,
              crop: null
            }]
          };
          s.id = uid('specialist');
          s.name = f.get('name');
          task.specialists.push(s);
        }
        markDirty();
        await saveTask();
        await route();
      }, action === 'rename-specialist' ? 'Rename' : 'Create specialist');
      return;
    }
    if (action === 'delete-specialist') {
      const current = task.specialists.find(s => s.id === id);
      confirmDelete(current.name, async () => {
        task.specialists = task.specialists.filter(s => s.id !== id);
        markDirty();
        await saveTask();
        await route();
      });
      return;
    }
    if (action === 'select-block') {
      validateEditor();
      selectedBlock = id;
      renderDesigner();
      updateLocks();
      refreshPreviews();
      return;
    }
    if (action === 'new-block') {
      ask('New block', '<label>Type<select name="type"><option value="text">Text / system prompt</option><option value="image">Reference image</option><option value="camera">Camera / video view</option></select></label><label>Alias<input name="alias" maxlength="1000" placeholder="Label for this block"></label>', async f => {
        const type = f.get('type'),
          b = {
            id: uid('block'),
            type,
            enabled: true,
            alias: f.get('alias') || (type === 'camera' ? 'Current camera observation:' : type === 'text' ? 'System prompt' : 'Reference image'),
            text: '',
            asset: null,
            source: task.sources[0].id,
            crop: null
          };
        specialist.blocks.push(b);
        selectedBlock = b.id;
        markDirty();
        renderDesigner();
        updateLocks();
      }, 'Add block');
      return;
    }
    if (action === 'delete-block') {
      const b = specialist.blocks.find(b => b.id === id);
      confirmDelete(b.alias || 'block', async () => {
        specialist.blocks = specialist.blocks.filter(b => b.id !== id);
        selectedBlock = null;
        markDirty();
        renderDesigner();
        updateLocks();
      });
      return;
    }
    if (action === 'move-up' || action === 'move-down') {
      const i = specialist.blocks.findIndex(b => b.id === id),
        j = i + (action === 'move-up' ? -1 : 1);
      if (j >= 0 && j < specialist.blocks.length) {
        [specialist.blocks[i], specialist.blocks[j]] = [specialist.blocks[j], specialist.blocks[i]];
        markDirty();
        renderDesigner();
        updateLocks();
      }
      return;
    }
    if (action === 'reset-crop') {
      specialist.blocks.find(b => b.id === selectedBlock).crop = null;
      markDirty();
      renderDesigner();
      updateLocks();
      return;
    }
    if (action === 'apply-crop') {
      const values = [...document.querySelectorAll('[data-crop]')].map(el => el.value);
      let crop = null;
      if (values.some(Boolean)) {
        crop = values.map(Number);
        if (values.some(v => v === '') || crop.some(v => !Number.isInteger(v) || v < 0) || crop[0] >= crop[2] || crop[1] >= crop[3]) throw new Error('Enter four valid crop coordinates.');
        const image = document.querySelector('#crop-image'),
          w = Number(image?.dataset.nativeWidth) || image?.naturalWidth,
          h = Number(image?.dataset.nativeHeight) || image?.naturalHeight;
        if (w && (crop[2] > w || crop[3] > h)) throw new Error(`Crop must fit ${w} × ${h} pixels.`);
      }
      specialist.blocks.find(b => b.id === selectedBlock).crop = crop;
      markDirty();
      drawCrop();
      toast('Crop applied to draft. Save the specialist to keep it.');
      return;
    }
    if (action === 'settings') {
      const keys = ['base_url', 'model', 'fps', 'evidence_horizon', 'horizon_duration', 'request_timeout', 'max_tokens', 'max_connections'];
      ask('Model & monitoring settings', `<p class="hint">Existing RoboBrain configuration. Values change only when you save them.</p>${keys.map(k=>`<label>${esc(k.replaceAll('_',' '))}<input name="${k}" value="${esc(task.config[k])}" ${['base_url','model'].includes(k)?'':'type="number" step="any"'} required></label>`).join('')}<button type="button" id="check-model">Check saved model connection</button>`, async f => {
        keys.forEach(k => task.config[k] = ['base_url', 'model'].includes(k) ? f.get(k) : Number(f.get(k)));
        markDirty();
        await saveTask();
      }, 'Save settings');
      dialog.querySelector('#check-model').onclick = () => api('/model/check', 'POST', {
        task_id: task.id
      }).then(() => toast('Model connection is ready')).catch(error => toast(error.message));
      return;
    }
    if (action === 'preview-request') {
      const p = await api(`/tasks/${task.id}/request/${specialist.id}`);
      ask('Model request preview', `<h3>System prompt</h3><pre>${esc(p.system)}</pre><h3>User message</h3>${p.content.map(part=>part.type==='text'?`<p>${esc(part.text)}</p>`:`<img class="request-image" src="${esc(part.image_url.url)}" alt="Image sent to the model">`).join('')}<details><summary>Existing response schema</summary><pre>${esc(JSON.stringify(p.schema,null,2))}</pre></details>`, async () => {}, 'Close');
      return;
    }
    if (action === 'connect' || action === 'connect-all') {
      if (action === 'connect' && !dirty && state.preview_task_id === task.id && state.sources[id]?.connected) {
        state = await api('/sources/' + id + '/disconnect', 'POST');
        await route();
        return;
      }
      pending = true;
      button.disabled = true;
      button.textContent = 'Connecting…';
      updateLocks();
      if (dirty) await saveTask();
      for (const s of task.sources.filter(s => s.enabled && (action === 'connect-all' || s.id === id))) state = await api(`/tasks/${task.id}/sources/${s.id}/connect`, 'POST');
      toast('Inputs connected');
      pending = false;
      await route();
      return;
    }
    if (action === 'start') {
      pending = true;
      updateLocks();
      if (dirty) throw new Error('Save the task before starting.');
      state = await api('/run', 'POST', {
        task_id: task.id,
        record
      });
      updateLocks();
      updateMonitor();
      return;
    }
    if (action === 'stop') {
      state = await api('/run/stop', 'POST');
      updateLocks();
      updateMonitor();
      return;
    }
  } catch (error) {
    toast(error.message);
  } finally {
    pending = false;
    updateLocks();
    if (action === 'connect' || action === 'connect-all') {
      button.disabled = state.busy;
      button.textContent = action === 'connect-all' ? 'Connect inputs' : 'Connect';
    }
  }
});
window.addEventListener('hashchange', () => route().catch(error => toast(error.message)));
window.addEventListener('beforeunload', e => {
  if (dirty) {
    e.preventDefault();
    e.returnValue = '';
  }
});
window.addEventListener('keydown', e => {
  if ((e.metaKey || e.ctrlKey) && e.key === 's' && task && page !== 'tasks') {
    e.preventDefault();
    if (!state.busy) saveTask().catch(error => toast(error.message));
  }
});
tasks = await api('/tasks');
state = await api('/state');
await route();
const events = new EventSource('/api/events');
events.onmessage = e => {
  const oldBusy = state.busy;
  state = JSON.parse(e.data);
  document.querySelector('#connection').textContent = state.busy ? 'Run in progress' : 'Local workspace';
  if (oldBusy !== state.busy) updateLocks();
  updateMonitor();
};
events.onerror = () => {
  document.querySelector('#connection').textContent = 'Reconnecting…';
};
// Camera previews run independently of the slower status updates and inference.
setInterval(refreshPreviews, 1000 / 30);
const context = document.modelContext;
if (context?.registerTool) {
  const lifecycle = new AbortController();
  window.addEventListener('pagehide', () => lifecycle.abort(), {
    once: true
  });
  try {
    Promise.resolve(context.registerTool({
      name: 'open_robobrain_task',
      description: 'Open a saved task. Does not start monitoring or modify configuration.',
      inputSchema: {
        type: 'object',
        properties: {
          task_id: {
            type: 'string'
          }
        },
        required: ['task_id'],
        additionalProperties: false
      },
      annotations: {
        readOnlyHint: false,
        untrustedContentHint: true
      },
      async execute(input) {
        if (!input || typeof input.task_id !== 'string' || Object.keys(input).length !== 1 || !tasks.some(t => t.id === input.task_id)) throw new Error('Unknown task');
        if (dirty || state.busy) throw new Error('Finish editing or stop the run before switching tasks');
        location.hash = link('specialists', input.task_id);
        await route();
        return {
          task_id: input.task_id,
          name: task.name
        };
      }
    }, {
      signal: lifecycle.signal
    })).catch(() => {});
  } catch {}
}
