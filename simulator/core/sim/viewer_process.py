"""Isolated native MuJoCo viewer fed with replaceable, coherent physics snapshots.

EGL recording/render workers stay in the service process. GLFW and its native
window live in a fresh child process, avoiding mixed graphics-library contexts and
concurrent GLFW initialization. Neither the child nor this worker advances physics.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
MAX_PACKET_BYTES = 4 * 1024 * 1024
MODEL_KEYS = ('scene', 'plug_success', 'wrist_depth', 'camera_resolution', 'motion_s',
              'settle_s', 'ik_damping', 'reset_qpos', 'gripper_color', 'continuous_physics')


def _packet(value):
    encoded = json.dumps(value, separators=(',', ':'), allow_nan=False).encode() + b'\n'
    if len(encoded) > MAX_PACKET_BYTES:
        raise ValueError('Native viewer state exceeds the bounded transport size')
    return encoded


def _viewer_environment(cache_root):
    environment = os.environ.copy()
    # The viewer has no reason to possess model/provider credentials.
    for name in list(environment):
        if any(part in name.upper() for part in ('API_KEY', 'TOKEN', 'SECRET', 'PASSWORD')):
            environment.pop(name)
    environment['MUJOCO_GL'] = 'glfw'
    environment.pop('PYOPENGL_PLATFORM', None)
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    environment['PYTHONPATH'] = str(ROOT) + (os.pathsep + environment['PYTHONPATH'] if environment.get('PYTHONPATH') else '')
    for name, suffix in (('TMPDIR', 'tmp'), ('XDG_CACHE_HOME', 'xdg'), ('XDG_CONFIG_HOME', 'config'),
                         ('XDG_STATE_HOME', 'state'), ('MESA_SHADER_CACHE_DIR', 'mesa'),
                         ('__GL_SHADER_DISK_CACHE_PATH', 'nvidia')):
        path = cache_root / suffix
        path.mkdir(parents=True, exist_ok=True)
        environment[name] = str(path)
    return environment


def run_viewer(runtime, *, fps=30, startup_timeout_s=20):
    """Run in the runtime's GUI worker; closing the runtime terminates only its child.

    A partially written packet is completed; at most one additional latest packet
    waits behind it. Pipe backpressure never reaches the physics thread. The native
    window can be closed independently without stopping robot execution.
    """
    import mujoco
    import numpy as np

    if not 1 <= float(fps) <= 60:
        raise ValueError('Viewer fps must be in [1,60]')
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    data = mujoco.MjData(runtime.model)
    state = np.empty(mujoco.mj_stateSize(runtime.model, state_spec), dtype=np.float64)

    def state_packet(kind='state'):
        identity = runtime.snapshot(data)
        mujoco.mj_getState(runtime.model, data, state, state_spec)
        value = {'type': kind, 'state': state.tolist(), 'state_seq': identity['state_seq']}
        if kind == 'init':
            # An explicit scene-building whitelist excludes task text, VLM profiles,
            # endpoints and secrets; the viewer never imports controller/runner code.
            value['config'] = {key: runtime.config[key] for key in MODEL_KEYS if key in runtime.config}
        return _packet(value)

    cache_root = Path(runtime._run_root).resolve().parent / '.viewer-cache'
    environment = _viewer_environment(cache_root)
    initial_packet = state_packet('init')
    process = subprocess.Popen([sys.executable, '-B', '-m', 'core.sim.viewer_process', '--child'],
                               cwd=ROOT, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, bufsize=0)
    runtime._viewer_process = process
    runtime._viewer_ready = False
    selector = selectors.DefaultSelector()
    output_tail = bytearray()
    received = bytearray()
    pending = initial_packet
    offset = 0
    latest = None
    next_sample = time.monotonic() + 1 / fps
    started = time.monotonic()
    try:
        os.set_blocking(process.stdin.fileno(), False)
        os.set_blocking(process.stdout.fileno(), False)
        selector.register(process.stdout, selectors.EVENT_READ, 'output')
        selector.register(process.stdin, selectors.EVENT_WRITE, 'input')
        while not runtime._closed.is_set():
            code = process.poll()
            if code is not None:
                if code:
                    detail = output_tail.decode(errors='replace').strip()[-4000:]
                    raise RuntimeError(f'Native viewer exited with code {code}: {detail}')
                if not runtime._viewer_ready:
                    raise RuntimeError('Native viewer closed before initialization completed')
                return
            now = time.monotonic()
            if not runtime._viewer_ready and now - started > startup_timeout_s:
                raise TimeoutError('Native viewer did not initialize within the startup timeout')
            if now >= next_sample:
                latest = state_packet()
                next_sample = now + 1 / fps
            for key, _ in selector.select(timeout=min(.05, max(0., next_sample - time.monotonic()))):
                if key.data == 'output':
                    try:
                        chunk = os.read(process.stdout.fileno(), 16384)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(process.stdout)
                        continue
                    output_tail.extend(chunk)
                    if len(output_tail) > 65536:
                        del output_tail[:-65536]
                    received.extend(chunk)
                    while b'\n' in received:
                        line, _, remaining = received.partition(b'\n')
                        received[:] = remaining
                        try:
                            message = json.loads(line)
                        except (ValueError, UnicodeError):
                            continue
                        if message.get('viewer_event') == 'ready':
                            runtime._viewer_ready = True
                        elif message.get('viewer_event') == 'error':
                            raise RuntimeError('Native viewer: ' + str(message.get('error', 'unknown error')))
                    if len(received) > 65536:
                        received.clear()
                else:
                    if pending is None:
                        pending, latest, offset = latest, None, 0
                    if pending is None:
                        # Writable pipes are always ready. Avoid a busy loop until
                        # the next replaceable sample is due.
                        runtime._closed.wait(min(.01, max(0., next_sample - time.monotonic())))
                        continue
                    try:
                        written = os.write(process.stdin.fileno(), pending[offset:])
                    except BlockingIOError:
                        continue
                    except BrokenPipeError:
                        # Let poll classify an ordinary window close vs a crash.
                        runtime._closed.wait(.01)
                        continue
                    offset += written
                    if offset == len(pending):
                        pending, offset = None, 0
    finally:
        selector.close()
        process.stdin.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        process.stdout.close()
        runtime._viewer_process = None
        runtime._viewer_ready = False


def _child_main():
    import mujoco
    import numpy as np
    from mujoco import viewer
    from core.sim.mujoco_session import MujocoSession

    def report(event, **fields):
        print(json.dumps({'viewer_event': event, **fields}), flush=True)

    class ViewOnlySession(MujocoSession):
        def _advance(self, *args, **kwargs):
            # The normal constructor settles its own scene with mj_step. The
            # display-only child instead receives the parent's integrated state.
            mujoco.mj_forward(self.model, self.data)

    def read_message():
        line = sys.stdin.buffer.readline(MAX_PACKET_BYTES + 1)
        if not line:
            return None
        if len(line) > MAX_PACKET_BYTES or not line.endswith(b'\n'):
            raise ValueError('Native viewer received an oversized or incomplete packet')
        message = json.loads(line)
        if not isinstance(message, dict):
            raise ValueError('Native viewer packet must be an object')
        return message

    session = None
    closed = threading.Event()
    lock = threading.Lock()
    latest = None
    try:
        initial = read_message()
        if initial is None or initial.get('type') != 'init':
            raise ValueError('Native viewer requires its scene initialization packet')
        config = {key: initial['config'][key] for key in MODEL_KEYS if key in initial['config']}
        session = ViewOnlySession(config, gui=False)
        from core.sim.continuous_physics import configure_continuous_physics
        configure_continuous_physics(session.model, config)
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state_size = mujoco.mj_stateSize(session.model, state_spec)

        def decode_state(message):
            value = np.asarray(message.get('state'), dtype=np.float64)
            if value.shape != (state_size,) or not np.isfinite(value).all():
                raise ValueError('Native viewer received invalid integration state')
            return value

        initial_state = decode_state(initial)
        mujoco.mj_setState(session.model, session.data, initial_state, state_spec)
        mujoco.mj_forward(session.model, session.data)

        def read_states():
            nonlocal latest
            try:
                while not closed.is_set():
                    message = read_message()
                    if message is None:
                        break
                    state = decode_state(message)
                    with lock:
                        latest = state
            except BaseException as exc:
                report('error', error=f'{type(exc).__name__}: {exc}')
            finally:
                closed.set()

        receiver = threading.Thread(target=read_states, name='viewer-state-reader', daemon=True)
        receiver.start()
        with viewer.launch_passive(session.model, session.data) as window:
            window.cam.lookat[:] = [.3, 0, .05]
            window.cam.distance, window.cam.azimuth, window.cam.elevation = 2.6, 35, -25
            report('ready')
            while not closed.is_set() and window.is_running():
                with lock:
                    state, latest = latest, None
                with window.lock():
                    if state is not None:
                        mujoco.mj_setState(session.model, session.data, state, state_spec)
                        mujoco.mj_forward(session.model, session.data)
                window.sync()
                closed.wait(1 / 60)
        closed.set()
        report('closed')
        return 0
    except BaseException as exc:
        report('error', error=f'{type(exc).__name__}: {exc}')
        return 1
    finally:
        closed.set()
        if session is not None:
            session.close()


if __name__ == '__main__':
    if sys.argv[1:] != ['--child']:
        raise SystemExit('This module is launched by the simulator GUI worker.')
    raise SystemExit(_child_main())
