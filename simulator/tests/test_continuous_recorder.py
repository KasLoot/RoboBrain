"""Continuous recordings retain the original layout without synthetic samples."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import imageio.v2 as imageio
import numpy as np

from core.record.continuous_recorder import ContinuousRecorder
from core.record.mujoco_recorder import GLOBAL_CAMERA


class Scene:
    resolution = 64
    cameras = ('side', 'wrist', 'front', 'wrist_insert', 'wrist_depth')

    def __init__(self):
        self.data = SimpleNamespace(time=2.0)
        self.model = SimpleNamespace(camera=Mock(return_value=SimpleNamespace(id=0)),
                                     vis=SimpleNamespace(global_=SimpleNamespace(offwidth=1920, offheight=1080)))
        self.render_times = []

    def render(self, camera):
        self.render_times.append((camera, self.data.time))
        return np.full((64, 64, 3), 30 + self.cameras.index(camera) * 35, np.uint8)

    def get_gripper_position(self):
        return np.array([0.04])


class GlobalRenderer:
    def __init__(self, model, *, height, width):
        self.shape = (height, width, 3)
        self.model = model
        self.captures = []
        self.close = Mock()

    def update_scene(self, data, *, camera):
        self.captures.append((data.time, camera))

    def render(self):
        return np.broadcast_to(np.array([65, 105, 175], np.uint8), self.shape).copy()


class ContinuousRecorderTests(unittest.TestCase):
    def test_original_video_layout_and_native_identity_without_extra_hold_frames(self):
        with TemporaryDirectory() as directory:
            scene = Scene()
            renderer_factory = Mock(side_effect=GlobalRenderer)
            with patch.dict('sys.modules', {'mujoco': SimpleNamespace(Renderer=renderer_factory)}):
                recorder = ContinuousRecorder(scene, directory, fps=10, model='test-model', task='Insert plug')
                self.addCleanup(recorder.close)
                self.assertEqual(recorder.frame_count, 0)
                recorder.begin_step(step_idx=0, stage='LIFT', token='MV_UP_LARGE', annotation='Lift the held plug.')
                self.assertEqual(recorder.frame_count, 0)
                identity = {'server_instance_id': 's', 'episode_id': 'e', 'attempt_id': 'a',
                            'execution_generation': 1, 'state_seq': 100, 'sim_time_s': 2.0,
                            'captured_at_server_monotonic_ns': 100_000_000}
                self.assertTrue(recorder.capture(identity))
                self.assertFalse(recorder.capture(identity, {'status': 'Waiting for model'}))
                scene.data.time = 2.3
                identity.update(state_seq=250, sim_time_s=2.3, captured_at_server_monotonic_ns=400_000_000)
                self.assertTrue(recorder.capture(identity, {'status': 'Executed', 'annotation': 'Actual upward motion.'}))
                recorder.end_step({'act': 'MV_UP_LARGE'})
                recorder.finish(True, 'task_success')
                recorder.close()
            self.assertEqual(recorder.frame_count, 2)
            self.assertEqual(recorder.dropped_intervals, 2)
            self.assertEqual(set(recorder.paths), {*scene.cameras, 'combined', 'global', 'frames', 'annotations'})
            # Preserve the original square global projection and separate16:9 export.
            renderer_factory.assert_any_call(scene.model, height=256, width=256)
            renderer_factory.assert_any_call(scene.model, height=1080, width=1920)
            self.assertEqual(renderer_factory.call_count, 2)
            for renderer in recorder._global_renderers.values():
                self.assertEqual(renderer.captures, [(2.0, GLOBAL_CAMERA), (2.3, GLOBAL_CAMERA)])
                renderer.close.assert_called_once_with()
            self.assertEqual((scene.model.vis.global_.offwidth, scene.model.vis.global_.offheight), (1920, 1080))
            self.assertEqual(scene.render_times, [(camera, timestamp) for timestamp in (2.0, 2.3) for camera in scene.cameras])
            rows = [json.loads(line) for line in recorder.paths['frames'].read_text().splitlines()]
            self.assertEqual([row['state_seq'] for row in rows], [100, 250])
            self.assertEqual([row['sim_time_s'] for row in rows], [2.0, 2.3])
            self.assertEqual(rows[1]['context']['annotation'], 'Actual upward motion.')
            self.assertEqual(rows[1]['dropped_intervals'], 2)
            events = [json.loads(line) for line in recorder.paths['annotations'].read_text().splitlines()]
            self.assertFalse(events[0]['synthetic_decision_holds'])
            self.assertEqual(events[-1]['frames'], 2)
            self.assertIn('episode_complete', [event['event'] for event in events])
            for name in (*scene.cameras, 'combined', 'global'):
                with imageio.get_reader(recorder.paths[name]) as reader:
                    self.assertEqual(reader.count_frames(), 2)
                    frame = reader.get_data(0)
                    shape = (1080, 1920, 3) if name == 'global' else (288, 384, 3) if name == 'combined' else (64, 64, 3)
                    self.assertEqual(frame.shape, shape)
                    if name == 'combined':
                        # Fixed camera cells remain in the unchanged upstream layout.
                        for camera, (y, x) in {'wrist': (64, 32), 'front': (64, 96),
                                               'side': (160, 32), 'wrist_insert': (160, 96),
                                               'wrist_depth': (256, 32)}.items():
                            np.testing.assert_allclose(frame[y, x], 30 + scene.cameras.index(camera)*35, atol=8)
                        np.testing.assert_allclose(frame[150, 250], [65, 105, 175], atol=8)

    def test_rejects_insufficient_preallocation_without_mutating_model_or_writing_files(self):
        with TemporaryDirectory() as directory:
            scene = Scene()
            scene.resolution = 512
            with self.assertRaisesRegex(ValueError, '1920x1600'):
                ContinuousRecorder(scene, directory, fps=10)
            self.assertEqual((scene.model.vis.global_.offwidth, scene.model.vis.global_.offheight), (1920, 1080))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_rejects_identity_from_a_different_snapshot(self):
        with TemporaryDirectory() as directory:
            scene = Scene()
            recorder = ContinuousRecorder(scene, directory, fps=10)
            self.addCleanup(recorder.close)
            with self.assertRaisesRegex(ValueError, 'identity does not match'):
                recorder.capture({'sim_time_s': 3.0, 'state_seq': 100})
            self.assertEqual(recorder.frame_count, 0)
