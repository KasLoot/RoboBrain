"""Original Show-Harness recording layout for continuous, snapshot-owned physics.

The caller pins one private session snapshot for each capture and schedules samples
on its recording worker. This module never advances physics, resamples the session,
changes model settings, or fabricates decision-hold frames.
"""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import threading

from core.record.episode_logger import _font, _jsonable, status_flags
from core.record.images import StreamingVideoWriter
from core.record.mujoco_recorder import CAMERAS, GLOBAL_CAMERA, GLOBAL_VIDEO_SIZE, MujocoRecorder


class ContinuousRecorder(MujocoRecorder):
    """Reuse the upstream canvas with independently owned writers and GL contexts.

    Construct, capture and close on the same recording thread. ``session.data``
    must remain pinned throughout ``capture`` (including proprioception getters).
    The shared model framebuffer must be allocated before any runtime workers:
    at least max(1920, combined_height - 32) by max(1080, combined_height - 32).
    """

    def __init__(self, session, directory, fps, model='', task='', global_video=True):
        # Do not invoke the baseline initializer: it captures immediately and its
        # render path temporarily changes the shared model's framebuffer settings.
        self.fps = float(fps)
        if not math.isfinite(self.fps) or not 1 <= self.fps <= 60:
            raise ValueError('recording.fps must be between 1 and 60')
        self.session = session
        self.cameras = tuple(getattr(session, 'cameras', CAMERAS))
        self.global_video = bool(global_video)
        self.global_camera = None
        if getattr(session, 'model', None) is not None:
            try:
                session.model.camera(GLOBAL_CAMERA)
            except KeyError:
                pass
            else:
                self.global_camera = GLOBAL_CAMERA
        if self.global_video and self.global_camera is None:
            raise ValueError(f'Separate global video requires the {GLOBAL_CAMERA!r} camera')
        if self.global_camera is not None:
            size = int(session.resolution)
            combined_height = (3 if 'wrist_insert' in self.cameras else 2) * (size + 32)
            square = combined_height - 32
            needed_width = max(square, GLOBAL_VIDEO_SIZE[0] if self.global_video else 0)
            needed_height = max(square, GLOBAL_VIDEO_SIZE[1] if self.global_video else 0)
            settings = session.model.vis.global_
            if settings.offwidth < needed_width or settings.offheight < needed_height:
                raise ValueError('Preallocate MuJoCo framebuffer before starting workers: '
                                 f'need at least {needed_width}x{needed_height}; '
                                 f'got {settings.offwidth}x{settings.offheight}')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        names = (*self.cameras, 'combined') + (('global',) if self.global_video else ())
        self.paths = {name: self.directory / f'{name}.mp4' for name in names}
        self.paths.update(annotations=self.directory / 'annotations.jsonl',
                          frames=self.directory / 'frames.jsonl')
        self._resources = ExitStack()
        self._thread_id = threading.get_ident()
        self._closed = False
        self._global_renderers = {}
        self.last_global_frame = None
        self.frame_count = 0
        self.dropped_intervals = 0
        self._start_time = None
        self._last_identity = None
        self._identity = {}
        self._step_start = None
        self.hold_frames = 0
        self.context = {'stage': 'Preparing', 'decision': 'WAIT', 'executing': '-',
                        'source': 'system', 'annotation': 'Waiting for the first decision.',
                        'vlm_output': ''}
        self.status = ''
        size = int(session.resolution)
        self.font = _font(max(9, round(size * 18 / 512)))
        self.title_font = _font(max(9, round(size * 16 / 512)), bold=True)
        self.action_font = _font(max(12, round(size * 36 / 512)), bold=True)
        try:
            self._log = self._resources.enter_context(self.paths['annotations'].open('w', encoding='utf-8'))
            self._frames_log = self._resources.enter_context(self.paths['frames'].open('w', encoding='utf-8'))
            # Preserve exact dimensions at any supported resolution, including
            # Full HD's 1080 height. No encoder-added padding or image stretching.
            self.writers = {name: StreamingVideoWriter(self.paths[name], self.fps, macro_block_size=1)
                            for name in names}
            for writer in self.writers.values():
                self._resources.callback(writer.close)
            self.event('recording_started', timing_mode='continuous', synthetic_decision_holds=False,
                       fps=self.fps, model=model, task=task, cameras=self.cameras,
                       combined_only_cameras=[self.global_camera] if self.global_camera and not self.global_video else [],
                       global_video_camera=self.global_camera if self.global_video else None,
                       global_video_resolution=GLOBAL_VIDEO_SIZE if self.global_video else None,
                       video_files={name: str(self.paths[name]) for name in names})
        except BaseException:
            self._resources.close()
            self._closed = True
            raise

    def _check_owner(self):
        if threading.get_ident() != self._thread_id:
            raise RuntimeError('ContinuousRecorder must stay on its recording/context thread')

    def event(self, name, **fields):
        self._check_owner()
        if self._closed:
            return
        now = float(self.session.data.time)
        entry = {**self._identity, 'event': name, 'utc': datetime.now(timezone.utc).isoformat(),
                 'sim_time_s': now, 'video_time_s': self.frame_count / self.fps,
                 'frame': self.frame_count, **self.context, **fields}
        self._log.write(json.dumps(_jsonable(entry), ensure_ascii=False) + '\n')
        self._log.flush()

    def _set_context(self, context):
        if context is None:
            return
        incoming = dict(context)
        if 'token' in incoming and 'decision' not in incoming:
            incoming['decision'] = incoming.pop('token')
        if 'step_idx' in incoming and 'step' not in incoming:
            incoming['step'] = incoming.pop('step_idx')
        old = json.dumps(_jsonable({'context': self.context, 'status': self.status}), sort_keys=True)
        self.status = str(incoming.pop('status', self.status))
        self.context.update(incoming)
        for key in ('stage', 'decision', 'executing', 'source', 'annotation', 'vlm_output'):
            self.context[key] = str(self.context.get(key) or '')
        new = json.dumps(_jsonable({'context': self.context, 'status': self.status}), sort_keys=True)
        if old != new:
            self.event('context_changed', status=self.status)

    def capture(self, identity, context=None):
        """Append exactly one coherent sample; return False for a repeated snapshot."""
        self._check_owner()
        if self._closed:
            return False
        identity = dict(identity)
        now = float(self.session.data.time)
        if 'sim_time_s' in identity and not math.isclose(float(identity['sim_time_s']), now, abs_tol=1e-8):
            raise ValueError('Recording identity does not match the supplied private snapshot')
        identity['sim_time_s'] = now
        self._identity = identity
        self._set_context(context)
        previous = self._last_identity
        if previous is not None and 'state_seq' in identity and all(
            identity.get(key) == previous.get(key)
            for key in ('server_instance_id', 'episode_id', 'execution_generation', 'state_seq')
        ):
            return False
        if self._start_time is None:
            self._start_time = now
        frames = {name: self.session.render(name) for name in self.cameras}
        canvas = self._canvas(frames, now - self._start_time)
        if self.global_camera is not None:
            square = canvas.shape[0] - 32
            canvas = self._append_global(canvas, self._render_global(square))
        frames['combined'] = canvas
        if self.global_video:
            self.last_global_frame = self._render_global(*GLOBAL_VIDEO_SIZE)
            frames['global'] = self.last_global_frame
        completed = []
        try:
            for name, frame in frames.items():
                self.writers[name].append(frame)
                completed.append(name)
        except BaseException as exc:
            self.event('frame_append_failed', completed_videos=completed,
                       error=f'{type(exc).__name__}: {exc}')
            raise
        dropped = 0
        if previous is not None and identity.get('episode_id') == previous.get('episode_id'):
            delta = now - previous['sim_time_s']
            if delta >= 0:
                dropped = max(0, round(delta * self.fps) - 1)
        self.dropped_intervals += dropped
        entry = {**identity, 'recorded_frame': self.frame_count, 'video_time_s': self.frame_count / self.fps,
                 'dropped_intervals': dropped, 'context': dict(self.context), 'status': self.status,
                 'dimensions': {name: {'width': int(frame.shape[1]), 'height': int(frame.shape[0])}
                                for name, frame in frames.items()}}
        self._frames_log.write(json.dumps(_jsonable(entry), ensure_ascii=False) + '\n')
        self._frames_log.flush()
        self.frame_count += 1
        self._last_identity = identity
        return True

    def _render_global(self, width, height=None):
        import mujoco
        height = width if height is None else height
        renderer = self._global_renderers.get((width, height))
        if renderer is None:
            # Allocation sizes were checked at construction. The live shared
            # MjModel remains immutable while physics and other renderers use it.
            renderer = mujoco.Renderer(self.session.model, height=height, width=width)
            self._resources.callback(renderer.close)
            self._global_renderers[width, height] = renderer
        renderer.update_scene(self.session.data, camera=self.global_camera)
        return renderer.render().copy()

    def begin_step(self, *, step_idx, stage, token, annotation, output='', source='vlm'):
        self._check_owner()
        self._step_start = self.frame_count
        self._set_context({'step': step_idx, 'stage': stage, 'decision': token, 'executing': token,
                           'source': source, 'annotation': annotation, 'vlm_output': output,
                           'status': 'Executing'})
        self.event('decision', frame_start=self._step_start)

    def gripper_command(self, close):
        self._check_owner()
        self.context['executing'] = 'GRASP' if close else 'RELEASE'
        self.event('gripper_command', annotation='Physical gripper close' if close else 'Physical gripper open')

    def end_step(self, record):
        self._check_owner()
        flags = [label for label, _ in status_flags(record)]
        self.status = '; '.join(flags) or 'Action completed'
        self.event('step_complete', frame_start=self._step_start, frame_end=self.frame_count,
                   annotation=self.status, result=record)
        self._step_start = None

    def finish(self, success=None, end_reason='closed'):
        self._check_owner()
        if self._closed:
            return
        self.status = f'{"SUCCESS" if success else "STOPPED"}: {end_reason}'
        try:
            self.event('episode_complete', success=success, end_reason=end_reason, annotation=self.status)
        finally:
            self.close()

    def close(self):
        self._check_owner()
        if self._closed:
            return
        try:
            if self._step_start is not None:
                self.event('step_interrupted', frame_start=self._step_start, frame_end=self.frame_count,
                           annotation='Recording ended before this action completed.')
            self.event('recording_closed', frames=self.frame_count, duration_s=self.frame_count / self.fps,
                       dropped_intervals=self.dropped_intervals)
        finally:
            self._closed = True
            self._resources.close()
