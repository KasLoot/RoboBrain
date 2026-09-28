"""All service gripper-opening routes consume a current visual authorization."""
import tempfile
import unittest
from pathlib import Path

from core.config import load_yaml
from core.runners.real import GraspLost
from core.sim.release_guard import ReleaseNotQualified
from core.sim.service_runtime import ROOT, ServiceRuntime, _SnapshotSession, ExecutionCancelled


class ServiceReleaseGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        config = load_yaml(ROOT / 'configs/robot_mujoco_plug.yaml')
        config['camera_resolution'] = 64
        self.runtime = ServiceRuntime(config=config, no_controller=True, run_root=Path(self.tmp.name))
        self.addCleanup(self.runtime.close)
        self.runtime.set_identity('test-server', 'test-episode')
        self.runtime.start_execution(1, 'attempt', lambda error: None)
        self.session = _SnapshotSession(self.runtime, 1, 'attempt')
        self.addCleanup(self.session.close)
        self.session.control_gripper(True)
        self.session.set_control_context(stage='GRASP', source='controller_internal', carry_established=False)
        self.session.record_control_start(token='GRASP', step_idx=0, stage='GRASP', source='vlm')

    def test_internal_empty_grasp_reopen_cannot_bypass_visual_guard(self):
        calls = []
        def deny(session, **context):
            calls.append(context)
            raise ReleaseNotQualified('UNSAFE', 'fixture refuses source qualification')
        self.session._release_guard = deny
        with self.assertRaises(ReleaseNotQualified):
            self.session.control_gripper(False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(float(self.runtime._session.data.ctrl[self.runtime._session._gripper]), 0.)

    def test_explicit_approval_is_consumed_once_without_duplicate_request(self):
        calls = []
        self.session._release_guard = lambda session, **context: (calls.append(context) or {'verdict': 'SUPPORTED_SOURCE_REGRASP'})
        self.session.authorize_release(stage='GRASP', source='vlm', carry_established=False)
        self.session.control_gripper(False)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(self.session._release_approval)
        self.session.control_gripper(True)
        self.session.record_control_start(token='RELEASE', step_idx=1, stage='GRASP', source='vlm')
        self.session.control_gripper(False)
        self.assertEqual(len(calls), 2)

    def test_generation_change_during_verification_cannot_open(self):
        def stop(session, **context):
            self.runtime.stop_execution(2)
        self.session._release_guard = stop
        with self.assertRaises(ExecutionCancelled):
            self.session.control_gripper(False)
        self.assertEqual(float(self.runtime._session.data.ctrl[self.runtime._session._gripper]), 0.)

    def test_stop_after_open_returns_clears_pending_process_qualification(self):
        from unittest.mock import patch
        from core.sim.mujoco_session import MujocoSession
        original = MujocoSession.control_gripper
        self.session._release_guard = lambda session, **context: {'verdict': 'SUPPORTED_SOURCE_REGRASP'}
        self.session.insertion_sequence.pending_release = {'control_id': 'pending-test'}
        def stop_on_return(session, close):
            result = original(session, close)
            self.runtime.stop_execution(2)
            return result
        with patch.object(MujocoSession, 'control_gripper', stop_on_return):
            with self.assertRaises(ExecutionCancelled):
                self.session.control_gripper(False)
        self.assertIsNone(self.session.insertion_sequence.pending_release)
        self.assertIsNone(self.session.insertion_sequence.qualified_release)

    def test_same_generation_visual_approval_rechecks_current_depth(self):
        from unittest.mock import patch
        self.session._release_guard = lambda session, **context: {'verdict': 'HALF_INSERTED_HELD'}
        self.session.authorize_release(stage='GRASP', source='vlm', carry_established=False)
        fresh = {'simulation_snapshot': {'state_seq': 100, 'execution_generation': 1}}
        with patch.object(self.session, 'get_observation', return_value=fresh), patch(
                'core.sim.visual_insertion.estimate_insertion', return_value={'qualified': False, 'lower_bound_m': .012}):
            with self.assertRaises(ReleaseNotQualified):
                self.session.control_gripper(False)
        self.assertEqual(float(self.runtime._session.data.ctrl[self.runtime._session._gripper]), 0.)
        self.assertIsNone(self.session._release_approval)

    def test_positive_visual_reply_cannot_override_fresh_empty_gripper(self):
        self.session.set_control_context(stage='INSERT', source='vlm', carry_established=True)
        self.session._release_guard = lambda session, **context: {'verdict': 'HALF_INSERTED_HELD'}
        with self.assertRaises(GraspLost):
            self.session.control_gripper(False)
        self.assertEqual(float(self.runtime._session.data.ctrl[self.runtime._session._gripper]), 0.)


if __name__ == '__main__':
    unittest.main()
