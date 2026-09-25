"""Loopback client, coherent frame, and obsolete-evidence checks. No VLM calls."""
import asyncio
from contextlib import contextmanager
import copy
import socket
import threading
import time

import uvicorn

from check_simulator_service import FakeRuntime, create_app
from robobrain.simulation import SimulationConnection, SimulationReader, decode_bundle
from robobrain.sources import Inputs
from robobrain.monitor import current_observations, decide, Specialist, load_config, input_batch


@contextmanager
def server(runtime):
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    app = create_app(runtime)
    process = uvicorn.Server(uvicorn.Config(app, log_level='error', lifespan='on'))
    worker = threading.Thread(target=process.run, kwargs={'sockets': [sock]}, daemon=True)
    worker.start()
    deadline = time.monotonic() + 5
    while not process.started and worker.is_alive() and time.monotonic() < deadline:
        time.sleep(.01)
    assert process.started, 'Local API did not start'
    try:
        yield f'http://127.0.0.1:{port}', app
    finally:
        process.should_exit = True
        worker.join(10)
        sock.close()
        assert not worker.is_alive(), 'API failed to close'


def until(fn, timeout=10):
    deadline = time.monotonic() + timeout
    error = None
    while time.monotonic() < deadline:
        try:
            return fn()
        except (RuntimeError, AssertionError) as exc:
            error = exc
            time.sleep(.03)
    raise AssertionError(str(error))


def run():
    fake = FakeRuntime()
    with server(fake) as (url, app):
        connection = SimulationConnection({'base_url': url, 'stale_after_s': 2})
        camera = {'type': 'free', 'lookat_m': [.4, 0, .15], 'distance_m': .8,
                  'azimuth_deg': 45., 'elevation_deg': -25.}
        sources = [{'id': f'camera{i}', 'name': f'Camera {i}', 'camera': camera,
                    'width': 64, 'height': 64, 'fps': 20} for i in (1, 2)]
        readers = {s['id']: SimulationReader(connection, s) for s in sources}
        try:
            for reader in readers.values():
                reader.start()
            inputs = Inputs(readers, 'mujoco')
            frame = until(inputs.get)
            assert len(frame.views) == 2
            assert len({v.source_index for v in frame.views.values()}) == 1
            evidence = {'context': frame.metadata['context'], 'identity': frame.metadata['identity'],
                        'observed_at': frame.observed_at, 'simulation': frame.metadata['simulation']}
            assert inputs.accepts_evidence(evidence)
            assert 'view_token' not in str(connection.state())
            positive = {'specialist': 'success', 'verdict': 'POSITIVE', 'evidence': evidence, 'slot': 0}
            votes = current_observations(inputs, [positive, {**positive, 'slot': 1}])
            assert [v['verdict'] for v in votes] == ['POSITIVE', 'UNCERTAIN'], votes
            changed = connection.update_view('camera1', {**camera, 'azimuth_deg': 100})
            assert changed['camera_revision'] == 2
            assert not inputs.accepts_evidence(evidence), 'Old viewpoint remains valid'
            moved = until(inputs.get)
            assert moved.views['camera1'].image[0, 0, 2] > frame.views['camera1'].image[0, 0, 2]
            png, provenance = connection.reference('camera1')
            assert png.startswith(b'\x89PNG') and provenance['views'][0]['camera_revision'] == 2
            result = connection.command('start', 'start-once')
            assert result['status'] == 'applied' and len(fake.starts) == 1
            running = until(inputs.get)
            attempt = running.metadata['simulation']['attempt_id']
            before = fake.status()['sim_time_s']
            stopped = connection.command('stop', 'stop-once')
            assert stopped['status'] == 'applied'
            time.sleep(.1)
            assert fake.status()['sim_time_s'] > before and fake.hold_applied
            after_stop = until(inputs.get)
            assert after_stop.metadata['simulation']['attempt_id'] == attempt
            connection.command('start', 'resume')
            assert connection.status['attempt_id'] == attempt and len(fake.starts) == 2
            connection.command('stop', 'stop-again')
            old_episode = connection.status['episode_id']
            connection.command('reset', 'reset')
            assert connection.status['episode_id'] != old_episode
            assert not inputs.accepts_evidence(evidence)
            until(inputs.get)
            for reader in readers.values():
                reader.stop()
            assert connection.command('start', 'no-monitor')['status'] == 'applied'
            assert connection.state()['execution_state'] == 'RUNNING', 'Reader teardown removed robot controls'
            connection.command('stop', 'final-stop')
            for bad in (b'', b'abc', b'\x00\x00\x10\x00{}'):
                try:
                    decode_bundle(bad)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Malformed packet accepted')
        finally:
            connection.close()
    print('Simulation client checks passed: live two-view transport, coherent snapshots, camera revisions, '
          'duplicate evidence, PNG references, start/stop/resume/reset, physics after stop, control after monitor detach.')


if __name__ == '__main__':
    run()
