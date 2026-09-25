"""Network-only MuJoCo client. No simulator imports or shared mutable physics state."""

import json
import struct
import threading
import time
from uuid import uuid4
from urllib.parse import urlsplit

import cv2
import httpx
import numpy as np
from websockets.sync.client import connect as websocket_connect

from .monitor import Frame

IDENTITY_FIELDS = ('server_instance_id', 'episode_id', 'attempt_id', 'execution_generation')
MAX_MESSAGE_BYTES = 32 * 1024 * 1024


def decode_bundle(packet):
    """Validate the entire bundle before exposing any view to consumers."""
    if not isinstance(packet, bytes) or not 4 < len(packet) <= MAX_MESSAGE_BYTES:
        raise ValueError('Invalid simulation frame message')
    size = struct.unpack('!I', packet[:4])[0]
    if not 0 < size <= min(65536, len(packet) - 4):
        raise ValueError('Invalid simulation frame header length')
    header = json.loads(packet[4:4 + size])
    if not isinstance(header, dict) or str(header.get('protocol_version')) != '1':
        raise ValueError('Unsupported simulation protocol')
    for name in (*IDENTITY_FIELDS, 'state_seq', 'bundle_id', 'sim_time_s',
                 'captured_at_server_monotonic_ns', 'sent_at_server_monotonic_ns'):
        if name not in header:
            raise ValueError('Missing simulation frame field: ' + name)
    if (type(header['state_seq']) is not int or header['state_seq'] < 0
            or type(header['execution_generation']) is not int
            or not isinstance(header['sim_time_s'], (float, int))
            or not np.isfinite(header['sim_time_s'])):
        raise ValueError('Invalid simulation frame identity/time')
    for name in ('captured_at_server_monotonic_ns', 'sent_at_server_monotonic_ns'):
        if type(header[name]) is not int or header[name] < 0:
            raise ValueError('Invalid simulation clock')
    if header['sent_at_server_monotonic_ns'] < header['captured_at_server_monotonic_ns']:
        raise ValueError('Simulation capture time exceeds send time')
    views = header.get('views')
    if not isinstance(views, list) or not 1 <= len(views) <= 3:
        raise ValueError('Expected one or two monitor views and optional global view')
    regular = [v for v in views if v.get('view_id') != '__global__']
    if not 1 <= len(regular) <= 2 or len(views) - len(regular) > 1:
        raise ValueError('Invalid monitor/global view count')
    payload = packet[4 + size:]
    images, end = {}, 0
    for view in views:
        name = view.get('view_id')
        width, height = view.get('width'), view.get('height')
        offset, length = view.get('offset'), view.get('length')
        if (not isinstance(name, str) or not name or name in images
                or type(width) is not int or type(height) is not int
                or not 16 <= width <= 2048 or not 16 <= height <= 2048
                or type(offset) is not int or type(length) is not int
                or offset != end or length <= 0 or offset + length > len(payload)
                or type(view.get('camera_revision')) is not int
                or view.get('encoding') not in ('jpeg', 'png')):
            raise ValueError('Invalid simulation image descriptor')
        encoded = np.frombuffer(payload[offset:offset + length], np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None or image.shape != (height, width, 3):
            raise ValueError('Simulation image dimensions do not match its descriptor')
        image.setflags(write=False)
        images[name] = image
        end = offset + length
    if end != len(payload):
        raise ValueError('Unexpected bytes after simulation images')
    return header, images


class SimulationConnection:
    """One latest-bundle stream, with a separate persistent command/status client."""
    def __init__(self, settings, *, client=None):
        self.settings = dict(settings)
        self.base_url = settings.get('base_url', 'http://127.0.0.1:8766').rstrip('/')
        url = urlsplit(self.base_url)
        if url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost') or url.username or url.password:
            raise ValueError('Use a local HTTP MuJoCo service address')
        self.client = client or httpx.Client(base_url=self.base_url, timeout=5, trust_env=False)
        self.lock = threading.RLock()
        self.command_lock = threading.Lock()
        self.view_lock = threading.RLock()
        self.pending_cleanup = []
        self.command_requests = {}
        self.changed = threading.Event()
        self.closed = threading.Event()
        self.thread = None
        self.sources = {}
        self.latest = None
        self.status = {}
        self.error = None
        self.control_token = None
        self.clock_offset = None
        self.clock_uncertainty = None
        self.clock_checked_at = 0.
        self.local_generation = 0
        self.include_global = False
        self.client_id = 'robobrain-' + uuid4().hex

    def request(self, method, path, body=None):
        try:
            response = self.client.request(method, path, json=body)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            try:
                detail = exc.response.json().get('detail', exc.response.text)
            except ValueError:
                detail = exc.response.text
            raise ValueError(f'Simulation service: {detail}') from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f'Simulation connection unavailable: {exc}') from exc

    def refresh_status(self):
        before = time.monotonic()
        health = self.request('GET', '/v1/health').json()
        if str(health.get('protocol_version')) != '1':
            raise ValueError('Unsupported simulation service protocol')
        after = time.monotonic()
        clock = health.get('server_sent_monotonic_ns', health.get('server_monotonic_ns'))
        status = self.request('GET', '/v1/status').json()
        with self.lock:
            previous_instance = self.status.get('server_instance_id')
            old_identity = tuple(self.status.get(k) for k in IDENTITY_FIELDS)
            self.status = status
            if self.control_token and status.get('control_owner') != self.client_id:
                self.control_token = None
            self.status.setdefault('server_instance_id', health.get('server_instance_id'))
            if clock is not None:
                self.clock_offset = (before + after) / 2 - clock / 1e9
                self.clock_uncertainty = (after - before) / 2
                self.clock_checked_at = after
            if previous_instance and previous_instance != self.status['server_instance_id']:
                self.control_token = None
                self.latest = None
                self.local_generation += 1
                for source in self.sources.values():
                    source['remote'] = None
                self.changed.set()
            elif old_identity != tuple(self.status.get(k) for k in IDENTITY_FIELDS):
                self.latest = None
                self.local_generation += 1
            self.error = None
            return dict(self.status)

    def state(self):
        with self.lock:
            return {**self.status, 'connected': bool(self.status) and not self.error,
                    'error': self.error, 'base_url': self.base_url,
                    'views': {k: {n: v for n, v in (s.get('remote') or {}).items() if n != 'view_token'}
                              for k, s in self.sources.items()}}

    def _create_view(self, source):
        return self.request('POST', '/v1/views', {
            'server_instance_id': self.status['server_instance_id'],
            'camera': source['camera'], 'width': source['width'], 'height': source['height'],
            'requested_fps': source['fps'],
        }).json()

    def add_source(self, source):
        with self.view_lock:
            self._add_source(source)
        # The stream worker also cleans up views; never hold the mutation lock
        # while waiting for that worker to publish the first frame.
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                self.bundle([source['id']])
                return
            except RuntimeError:
                if self.closed.wait(.05):
                    break
        self.remove_source(source['id'])
        raise RuntimeError(self.error or 'MuJoCo did not deliver a fresh camera frame')

    def _add_source(self, source):
        self.refresh_status()
        self.remove_source(source['id'])
        remote = self._create_view(source)
        with self.lock:
            self.sources[source['id']] = {'definition': dict(source), 'remote': remote}
            self.latest = None
            self.local_generation += 1
            self.changed.set()
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, name='mujoco-input', daemon=True)
                self.thread.start()
    def remove_source(self, source_id):
        with self.view_lock:
            self._remove_source(source_id)

    def _remove_source(self, source_id):
        with self.lock:
            item = self.sources.pop(source_id, None)
            if item:
                self.latest = None
                self.local_generation += 1
                self.changed.set()
        if item and item.get('remote'):
            with self.lock:
                self.pending_cleanup.append((self.status.get('server_instance_id'), item['remote']))
            self._cleanup_views()

    def _cleanup_views(self):
        with self.view_lock:
            with self.lock:
                pending, self.pending_cleanup = self.pending_cleanup, []
            for instance, view in pending:
                if instance != self.status.get('server_instance_id'):
                    continue
                try:
                    self.request('DELETE', '/v1/views/' + view['view_id'], {
                        'server_instance_id': instance, 'view_token': view['view_token'],
                        'camera_revision': view['camera_revision']})
                except (ValueError, RuntimeError):
                    with self.lock:
                        self.pending_cleanup.append((instance, view))

    def context(self, source_ids):
        with self.lock:
            revisions = []
            for name in sorted(source_ids):
                item = self.sources.get(name, {}).get('remote') or {}
                revisions.append((name, item.get('view_id'), item.get('camera_revision')))
            return (*[self.status.get(k) for k in IDENTITY_FIELDS], self.local_generation, tuple(revisions))

    def bundle(self, source_ids, *, with_global=False):
        with self.lock:
            if self.latest is None or self.error:
                raise RuntimeError(self.error or 'Waiting for a fresh simulation bundle')
            header, images, observed_at = self.latest
            if time.monotonic() - observed_at > self.settings.get('stale_after_s', 5):
                raise RuntimeError('Simulation frames are stale')
            if tuple(header.get(k) for k in IDENTITY_FIELDS) != tuple(self.status.get(k) for k in IDENTITY_FIELDS):
                raise RuntimeError('Waiting for the current simulation execution context')
            descriptors = {v['view_id']: v for v in header['views']}
            context = self.context(source_ids)
            result = {}
            for name in source_ids:
                item = self.sources.get(name, {}).get('remote') or {}
                view_id = item.get('view_id')
                view = descriptors.get(view_id)
                if not view or view['camera_revision'] != item.get('camera_revision'):
                    raise RuntimeError('Waiting for the requested camera revision')
                metadata = {'simulation': header, 'context': context,
                            'identity': (*context, header['state_seq']),
                            'view_id': view_id, 'camera_revision': view['camera_revision']}
                result[name] = Frame(images[view_id], header['state_seq'], header['state_seq'], observed_at,
                                     metadata=metadata)
            if with_global:
                return result, images.get('__global__')
            return result

    def _publish(self, packet):
        header, images = decode_bundle(packet)
        received = time.monotonic()
        with self.lock:
            if header['server_instance_id'] != self.status.get('server_instance_id'):
                return
            if self.clock_offset is None or received - self.clock_checked_at > 10:
                raise RuntimeError('Simulation clock freshness is unavailable')
            # Earliest possible local capture time gives a conservative age.
            observed = header['captured_at_server_monotonic_ns'] / 1e9 + self.clock_offset - self.clock_uncertainty
            if observed > received + .1:
                raise ValueError('Simulation clock is inconsistent')
            if self.latest and header['episode_id'] == self.latest[0]['episode_id'] and header['state_seq'] < self.latest[0]['state_seq']:
                return
            self.latest = (header, images, min(observed, received))
            self.error = None

    def _run(self):
        next_status = next_renew = 0.
        while not self.closed.is_set():
            try:
                self.refresh_status()
                self._cleanup_views()
                with self.lock:
                    missing = [name for name, item in self.sources.items() if item['remote'] is None]
                for name in missing:
                    with self.lock:
                        definition = self.sources[name]['definition']
                    remote = self._create_view(definition)
                    with self.lock:
                        if name in self.sources:
                            self.sources[name]['remote'] = remote
                with self.lock:
                    view_ids = [s['remote']['view_id'] for s in self.sources.values() if s['remote']]
                    subscription = {'server_instance_id': self.status['server_instance_id'],
                                    'view_ids': view_ids, 'encoding': 'jpeg', 'include_global': self.include_global}
                    self.changed.clear()
                if not view_ids:
                    self._renew_control()
                    self.closed.wait(.5)
                    continue
                address = 'ws' + self.base_url[4:] + '/v1/frames'
                with websocket_connect(address, open_timeout=5, close_timeout=1,
                                       max_size=MAX_MESSAGE_BYTES, max_queue=1, proxy=None) as stream:
                    stream.send(json.dumps(subscription))
                    while not self.closed.is_set() and not self.changed.is_set():
                        now = time.monotonic()
                        if now >= next_status:
                            self.refresh_status()
                            next_status = now + .5
                        if now >= next_renew:
                            self._renew_control()
                            next_renew = now + 5
                        try:
                            packet = stream.recv(timeout=.25)
                        except TimeoutError:
                            continue
                        if isinstance(packet, str):
                            raise RuntimeError('Simulation stream: ' + packet[:300])
                        self._publish(packet)
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)
                    self.latest = None
                    self.local_generation += 1
                self.closed.wait(.5)

    def _renew_control(self):
        if self.control_token:
            try:
                self.request('POST', '/v1/control-session/renew', {
                    'server_instance_id': self.status['server_instance_id'], 'control_token': self.control_token})
            except (ValueError, RuntimeError):
                self.control_token = None

    def command(self, command, request_id=None, expected_context=None):
        if command not in ('start', 'stop', 'reset'):
            raise ValueError('Unknown robot command')
        with self.command_lock:
            self.refresh_status()
            if request_id and request_id in self.command_requests:
                previous = self.command_requests[request_id]
                if previous['command'] != command:
                    raise ValueError('Request ID already belongs to a different robot command')
                if previous['server_instance_id'] != self.status['server_instance_id']:
                    raise ValueError('Command belongs to a previous service instance')
                try:
                    return self.request('GET', '/v1/commands/' + request_id).json()
                except ValueError:
                    # A rejected/unreceived request has no command receipt.
                    self.command_requests.pop(request_id, None)
            if expected_context is not None:
                names = [v[0] for v in expected_context[-1]]
                if self.context(names) != expected_context:
                    raise ValueError('The monitoring decision belongs to an obsolete simulation context')
                self.bundle(names)  # Reject stale evidence even if the context is unchanged.
            if not self.control_token:
                claim = self.request('POST', '/v1/control-session', {
                    'server_instance_id': self.status['server_instance_id'],
                    'client_id': self.client_id, 'lease_seconds': 30}).json()
                self.control_token = claim['control_token']
            body = {k: self.status.get(k) for k in IDENTITY_FIELDS}
            body.update(command=command, request_id=request_id or uuid4().hex,
                        control_token=self.control_token, reason='visual_outcome' if expected_context else 'operator')
            self.command_requests[body['request_id']] = dict(body)
            while len(self.command_requests) > 128:
                self.command_requests.pop(next(iter(self.command_requests)))
            # The same body/request ID is used if the receipt was lost.
            try:
                result = self.request('POST', '/v1/commands', body).json()
            except RuntimeError:
                result = self.request('POST', '/v1/commands', body).json()
            deadline = time.monotonic() + 3
            while result.get('status') == 'accepted' and time.monotonic() < deadline:
                self.closed.wait(.025)
                result = self.request('GET', '/v1/commands/' + body['request_id']).json()
            self.refresh_status()
            with self.lock:
                if self.thread is None:
                    self.thread = threading.Thread(target=self._run, name='mujoco-input', daemon=True)
                    self.thread.start()
            return result

    def update_view(self, source_id, camera):
        with self.view_lock:
            return self._update_view(source_id, camera)

    def _update_view(self, source_id, camera):
        with self.lock:
            item = self.sources.get(source_id)
            if not item or not item.get('remote'):
                raise ValueError('Connect this simulation source first')
            remote = dict(item['remote'])
        result = self.request('PATCH', '/v1/views/' + remote['view_id'], {
            'server_instance_id': self.status['server_instance_id'], 'view_token': remote['view_token'],
            'camera_revision': remote['camera_revision'], 'camera': camera}).json()
        with self.lock:
            item['remote'] = result
            item['definition']['camera'] = dict(camera)
            self.latest = None
            self.local_generation += 1
            self.changed.set()
        return {k: v for k, v in result.items() if k != 'view_token'}

    def recording_run(self, enable=True):
        self.refresh_status()
        if enable:
            result = self.request('POST', '/v1/recordings', {
                'server_instance_id': self.status['server_instance_id']}).json()
            with self.lock:
                self.include_global = True
                self.changed.set()
            return result
        return self.status.get('recording_run')

    def stop_recording_feed(self):
        with self.lock:
            self.include_global = False
            self.changed.set()

    def reference(self, source_id):
        with self.lock:
            item = self.sources.get(source_id, {}).get('remote')
            if not item:
                raise ValueError('Connect the simulation camera first')
            view_id = item['view_id']
        packet = self.request('POST', '/v1/snapshots', {
            'server_instance_id': self.status['server_instance_id'], 'view_ids': [view_id], 'encoding': 'png'}).content
        header, images = decode_bundle(packet)
        ok, png = cv2.imencode('.png', images[view_id])
        if not ok:
            raise RuntimeError('Could not encode simulation reference')
        return png.tobytes(), header

    def close(self):
        self.closed.set()
        self.changed.set()
        if self.thread:
            self.thread.join(12)
        for source_id in list(self.sources):
            self.remove_source(source_id)
        if self.control_token:
            try:
                self.request('DELETE', '/v1/control-session', {
                    'server_instance_id': self.status.get('server_instance_id'), 'control_token': self.control_token})
            except (ValueError, RuntimeError):
                pass
        self.client.close()


class SimulationReader:
    def __init__(self, connection, source):
        self.connection, self.source = connection, source
        self.ended = threading.Event()
        self.started_at = 0.

    def start(self):
        self.started_at = time.monotonic()
        self.connection.add_source(self.source)

    def get(self):
        return self.connection.bundle([self.source['id']])[self.source['id']]

    def stop(self):
        self.connection.remove_source(self.source['id'])
