"""Unpaid native-viewer + EGL recording regression; requires a working DISPLAY.

Run from the repository: python -B tests/check_simulation_gui.py --resolution 512
Creates/closes only its own passive MuJoCo window and writes below .tmp/.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

WORKSPACE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolution', type=int, choices=(64, 512, 640), default=512)
    args = parser.parse_args()
    if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
        raise SystemExit('This check requires a working native display; use the headless checks otherwise.')
    directory = WORKSPACE / '.tmp' / ('simulation-gui-' + uuid4().hex[:12])
    directory.mkdir(parents=True)
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    for variable, name in (('TMPDIR', 'tmp'), ('XDG_CACHE_HOME', 'cache'),
                           ('MESA_SHADER_CACHE_DIR', 'mesa'), ('__GL_SHADER_DISK_CACHE_PATH', 'nvidia')):
        path = directory / name
        path.mkdir()
        os.environ[variable] = str(path)
    sys.path.insert(0, str(WORKSPACE / 'simulator'))
    from core.config import load_yaml
    from core.sim.service_runtime import ServiceRuntime
    import imageio.v2 as imageio

    cfg = load_yaml(WORKSPACE / 'simulator/configs/robot_mujoco_plug.yaml')
    cfg['camera_resolution'] = args.resolution
    cfg['recording']['fps'] = 2
    runtime = ServiceRuntime(config=cfg, gui=True, record=True, record_global=True,
                             no_controller=True, run_root=directory / 'runs')
    runtime.set_identity('gui-smoke-server', 'gui-smoke-episode')
    child = None
    try:
        deadline = time.monotonic() + 20
        while not getattr(runtime, '_viewer_ready', False) or runtime.status()['recorded_frames'] < 1:
            status = runtime.status()
            assert not status['recording_error'], status
            assert not status.get('gui_error'), status
            assert time.monotonic() < deadline, status
            time.sleep(.05)
        child = runtime._viewer_process
        views = [{'view_id': name, 'width': 160, 'height': 120,
                  'camera': {'lookat_m': [.45, 0, .15], 'distance_m': .8,
                             'azimuth_deg': azimuth, 'elevation_deg': -25}}
                 for name, azimuth in (('a', 45), ('b', -45))]
        first = runtime.render_bundle(views, include_global=True)
        runtime.start_execution(1, 'unpaid-gui-smoke', lambda error: None)
        runtime.stop_execution(2)
        before = runtime.status()['sim_time_s']
        time.sleep(.12)
        second = runtime.render_bundle(views, include_global=True)
        assert runtime.status()['sim_time_s'] > before
        assert second['state_seq'] > first['state_seq']
        assert {item['view_id'] for item in second['images']} == {'a', 'b', '__global__'}
    finally:
        runtime.close_renderer()
        runtime.close()
    status = runtime.status()
    assert not status['recording_error'], status
    assert not status.get('gui_error'), status
    assert not runtime._gui_thread.is_alive()
    assert not runtime._record_thread.is_alive()
    assert child is not None and child.poll() is not None
    videos = {}
    for name in ('side', 'wrist', 'front', 'wrist_insert', 'wrist_depth', 'combined', 'global'):
        path = Path(status['recording_run']['directory']) / 'show-harness-video' / f'{name}.mp4'
        with imageio.get_reader(path) as reader:
            videos[name] = {'frames': reader.count_frames(), 'shape': reader.get_data(0).shape}
    assert len({item['frames'] for item in videos.values()}) == 1, videos
    assert videos['global']['shape'] == (1080, 1920, 3), videos
    expected_height = 3 * (args.resolution + 32)
    assert videos['combined']['shape'] == (expected_height, 2 * args.resolution + expected_height - 32, 3), videos
    result = {'resolution': args.resolution, 'passed': True, 'viewer_child_exit': child.returncode,
              'videos': videos, 'status': status}
    (directory / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    print('Artifacts:', directory)


if __name__ == '__main__':
    main()
