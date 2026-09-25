"""Actual MuJoCo lifecycle checks; these never call a VLM provider."""
import threading
import time
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from core.config import load_yaml
from core.sim.service_runtime import ExecutionCancelled, ROOT, ServiceRuntime, _SnapshotSession


def wait_for(predicate, seconds=3):
    end = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() >= end:
            raise AssertionError('Condition did not become true')
        time.sleep(.005)


class ServiceRuntimeTests(unittest.TestCase):
    def make_runtime(self, **kwargs):
        config = load_yaml(ROOT / 'configs/robot_mujoco_plug.yaml')
        config['camera_resolution'] = 64
        runtime = ServiceRuntime(config=config, **kwargs)
        self.addCleanup(runtime.close)
        return runtime

    def test_idle_and_stop_keep_physics_running_and_old_targets_fail(self):
        runtime = self.make_runtime(no_controller=True)
        before = runtime.status()['sim_time_s']
        wait_for(lambda: runtime.status()['sim_time_s'] > before + .04)
        runtime.start_execution(1, 'a', lambda error: self.fail(error))
        runtime.stop_execution(2)
        stopped = runtime.status()['sim_time_s']
        wait_for(lambda: runtime.status()['sim_time_s'] > stopped + .04)
        self.assertTrue(runtime.status()['hold_applied'])
        with self.assertRaises(ExecutionCancelled):
            runtime.submit_target(1, 'a', .1, None)
        runtime.reset(3)
        self.assertEqual(runtime.status()['execution_generation'], 3)
        self.assertLess(runtime.status()['sim_time_s'], 1.03)

    def test_stop_cancels_active_gripper_ramp_at_sampled_control(self):
        runtime = self.make_runtime(no_controller=True)
        runtime.start_execution(1, 'a', lambda error: None)
        target = runtime._session.data.ctrl.copy()
        target[runtime._session._gripper] = 0
        errors = []
        def move():
            try: runtime.submit_target(1, 'a', .8, target)
            except ExecutionCancelled as exc: errors.append(exc)
        worker = threading.Thread(target=move)
        worker.start()
        wait_for(lambda: runtime._active is not None)
        time.sleep(.15)
        began = time.monotonic()
        runtime.stop_execution(2)
        self.assertLess(time.monotonic() - began, .25)
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        data = mujoco.MjData(runtime.model)
        runtime.snapshot(data)
        gripper = float(data.ctrl[runtime._session._gripper])
        self.assertGreater(gripper, 0)
        self.assertLess(gripper, 255)
        held = data.ctrl[runtime._session._actuators].copy()
        np.testing.assert_allclose(held, data.qpos[runtime._session._qpos], atol=.002)
        time.sleep(.12)
        runtime.snapshot(data)
        self.assertEqual(gripper, float(data.ctrl[runtime._session._gripper]))
        np.testing.assert_array_equal(held, data.ctrl[runtime._session._actuators])
        self.assertTrue(runtime._actions[-1]['interrupted'])

    def test_stopped_planner_cannot_act_and_resume_does_not_release(self):
        entered = threading.Event()
        unblock = threading.Event()
        resumed = threading.Event()
        generations = []
        errors = []
        def runner(session, generation, attempt):
            generations.append(generation)
            if generation == 1:
                entered.set()
                unblock.wait(3)
                session.control_gripper(False)
            else:
                self.assertTrue(session.skip_initial_release)
                self.assertEqual(session.attempt_id, 'same-attempt')
                resumed.set()
        runtime = self.make_runtime(runner_factory=runner)
        runtime.start_execution(1, 'same-attempt', errors.append)
        self.assertTrue(entered.wait(1))
        runtime.stop_execution(2)
        runtime.start_execution(3, 'same-attempt', errors.append)
        self.assertEqual(generations, [1])
        unblock.set()
        self.assertTrue(resumed.wait(2))
        wait_for(lambda: runtime.status()['execution_generation'] == 4)
        self.assertEqual(generations, [1, 3])
        self.assertEqual(errors, [None])
        self.assertEqual(len(runtime._actions), 0)

    def test_controller_observation_uses_one_private_snapshot(self):
        runtime = self.make_runtime(no_controller=True)
        session = _SnapshotSession(runtime)
        self.addCleanup(session.close)
        times = []
        def render(camera):
            times.append(float(session.data.time))
            time.sleep(.01)
            return np.zeros((64, 64, 3), dtype=np.uint8)
        def depth(camera):
            times.append(float(session.data.time))
            return np.full((64, 64), .1, np.float32)
        with patch.object(session, 'render', side_effect=render), patch.object(session, 'render_depth', side_effect=depth):
            obs = session.get_observation()
        self.assertEqual(len(set(times)), 1)
        self.assertEqual(times[0], obs['simulation_snapshot']['sim_time_s'])
        self.assertGreater(runtime.status()['sim_time_s'], times[0])

    def test_cartesian_motion_advances_only_on_physics_owner(self):
        runtime = self.make_runtime(no_controller=True)
        runtime.start_execution(1, 'a', lambda error: None)
        session = _SnapshotSession(runtime, 1, 'a')
        before = session.get_ee_pose()
        target = before.copy()
        target[2] += .01
        session.update_desired_ee_pose(target)
        np.testing.assert_allclose(session.get_ee_pose()[:3], target[:3], atol=.003)
        self.assertEqual(len(runtime._actions), 2)
        runtime.stop_execution(2)
