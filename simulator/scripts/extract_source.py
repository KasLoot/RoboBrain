"""Reproduce the read-only upstream extraction into this simulator directory."""
from pathlib import Path
import ast
import hashlib
import json
import shutil
import subprocess

UPSTREAM = Path('/home/yuxin/workspace/Show-Harness')
DEST = Path(__file__).resolve().parents[1]
modified = {}
source = (UPSTREAM / 'core/launch.py').read_text()
tree = ast.parse(source)
keep = {'make_runner', 'recovery_empty_width_m', 'recovery_open_width_m', 'high_above_table_m', 'coarse_step_m', 'action_chunk_step_num', 'mem_text_len'}
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in keep]
names = {n.id for f in functions for n in ast.walk(f) if isinstance(n, ast.Name)}
imports = []
for node in tree.body:
    if isinstance(node, (ast.ImportFrom, ast.Import)):
        module = getattr(node, 'module', '') or ''
        if module.startswith(('core.franka', 'core.piper', 'interpreters.franka', 'interpreters.piper')):
            continue
        aliases = [a for a in node.names if (a.asname or a.name.split('.')[0]) in names or module == '__future__']
        if aliases:
            node.names = aliases
            imports.append(ast.unparse(node))
modified['core/launch.py'] = '# Modified for the MuJoCo-only extraction: hardware launch paths removed.\n' + '\n'.join(imports) + '\n\nEMPTY_GRASP_WIDTH_M = 0.005\nTABLE_CONTACT_Z_M = 0.025\n\n' + '\n\n'.join(ast.get_source_segment(source, f) for f in functions) + '\n'
source = (UPSTREAM / 'core/runners/real.py').read_text()
source = source.replace('from interpreters.franka_atomic_controller import AtomicStepResult, FrankaAtomicController', 'from interpreters.real_atomic_controller import AtomicStepResult, RealAtomicController as FrankaAtomicController')
source = source.replace('from core.franka.franka_session import FrankaSession', 'FrankaSession = Any  # Structural session interface; no physical-robot transport.')
source = source.replace('            self.controller.step(RELEASE_TOKEN)\n            obs =', '            if not getattr(self.session, "skip_initial_release", False):\n                self.controller.step(RELEASE_TOKEN)\n            obs =', 1)
modified['core/runners/real.py'] = '# Modified: transport-free session typing and service resume without startup release.\n' + source
image_source = (UPSTREAM / 'core/record/images.py').read_text()
helper_source = (UPSTREAM / 'core/franka/camera_utils.py').read_text()
helper = next(n for n in ast.parse(helper_source).body if isinstance(n, ast.FunctionDef) and n.name == 'resize_with_pad')
image_source = image_source.replace('import imageio.v2 as imageio', 'import cv2\nimport imageio.v2 as imageio')
image_source = image_source.replace('        from core.franka.camera_utils import resize_with_pad  # noqa: PLC0415', '')
modified['core/record/images.py'] = '# Modified: inline shared resize helper; omit hardware camera dependencies.\n' + image_source + '\n\n' + ast.get_source_segment(helper_source, helper) + '\n'
modified['plugins/subgoal/__init__.py'] = '# Modified: retain only the single-arm planner dependency.\nfrom .agent import SUBGOAL_PLAN_SCHEMA, SubgoalPlannerAgent, load_prompt\nfrom .plugin import SubgoalPlanner\n'
queue = ['scripts/run_mujoco.py', 'scripts/mujoco/replay.py', 'core/sim/mujoco_replay.py', *modified]
seen = set()
manifest = []
def resolve(module):
    p = module.replace('.', '/')
    for rel in (p + '.py', p + '/__init__.py'):
        if (UPSTREAM / rel).is_file():
            return rel

def enqueue(module):
    rel = resolve(module)
    if rel:
        queue.append(rel)
        parts = module.split('.')
        for i in range(1, len(parts)):
            init = '/'.join(parts[:i]) + '/__init__.py'
            if (UPSTREAM / init).is_file():
                queue.append(init)

while queue:
    rel = queue.pop()
    if rel in seen: continue
    seen.add(rel)
    assert not any(part in rel for part in ('core/franka/', 'core/piper/', 'maniskill', 'robolab', 'train/')), rel
    raw = (UPSTREAM / rel).read_bytes()
    text = modified.get(rel, raw.decode())
    target = DEST / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    manifest.append({'path': rel, 'upstream_sha256': hashlib.sha256(raw).hexdigest(), 'adapted': rel in modified})
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            for a in node.names: enqueue(a.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            if node.level:
                parent = rel.split('/')[:-1]
                module = '.'.join(parent[:len(parent) - node.level + 1] + ([module] if module else []))
            enqueue(module)
            for a in node.names: enqueue(module + '.' + a.name)
# Co-located prompt assets are discovered dynamically by their owning plugins.
for directory in {str(Path(rel).parent) for rel in seen}:
    for path in (UPSTREAM / directory).glob('*.txt'):
        rel = path.relative_to(UPSTREAM)
        shutil.copy2(path, DEST / rel)
        manifest.append({'path': str(rel), 'upstream_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'adapted': False})
extra = [UPSTREAM / p for p in ('LICENSE', 'configs/robot_mujoco.yaml', 'configs/robot_mujoco_plug.yaml', 'configs/primitives_franka.yaml')]
extra += [UPSTREAM / 'prompts' / p for p in ('common_context.txt', 'controller.txt', 'controller_mujoco.txt', 'controller_mujoco_cartesian.txt', 'controller_mujoco_plug.txt')]
extra += list((UPSTREAM / 'assets/mujoco').rglob('*'))
extra += list((UPSTREAM / 'third_party/mujoco_menagerie/franka_emika_panda').rglob('*'))
extra += list((UPSTREAM / 'tests').glob('test_mujoco*.py'))
for path in extra:
    if not path.is_file(): continue
    rel = path.relative_to(UPSTREAM)
    target = DEST / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    adapted = str(rel) == 'tests/test_mujoco_recording.py'
    if adapted:
        text = target.read_text().replace('self.assertLess(int(frame[470, 240].max()), 80)', 'self.assertLess(int(frame[470, 130].max()), 80)')
        target.write_text('# Modified: recording background probe uses reserved padding, not font-dependent footer glyphs.\n' + text)
    manifest.append({'path': str(rel), 'upstream_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'adapted': adapted})
revision = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
(DEST / 'UPSTREAM_MANIFEST.json').write_text(json.dumps({'upstream': str(UPSTREAM), 'revision': revision, 'files': sorted(manifest, key=lambda item: item['path'])}, indent=2) + '\n')
print(f'Extracted {len(manifest)} files from {revision}; upstream read only.')
