"""A seated final pose cannot replace proof of controlled insertion while held."""
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from core.record.run_artifacts import RunArtifacts
from core.sim.plug_sequence import PlugInsertionSequence, measure_engagement
from plug_physics_fixtures import make_plug_session, grasp_and_lift, move_tcp


class PlugSequenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # Original contacts suffice for short process-classification fixtures.
        self.session = make_plug_session(0)
        self.addCleanup(self.session.close)
        self.sequence = PlugInsertionSequence(RunArtifacts(Path(self.tmp.name)))
        self.tick = 0

    def sample(self, **kwargs):
        self.tick += 1
        return self.sequence.observe(self.session.model, self.session.data,
            {'state_seq': self.tick, 'episode_id': 'test', 'execution_generation': 1}, **kwargs)

    def carry_over_socket(self):
        grasp_and_lift(self.session)
        result = self.sample(stage='LIFT')
        self.assertTrue(result['current']['held_by_both_pads'])
        move_tcp(self.session, [.48, .16, .20])
        self.sample(stage='MOVE')

    def finish(self):
        self.session.control_gripper(False)
        self.sample(stage='RELEASE', token='RELEASE', boundary='after_gripper_open')
        self.session._advance(2.)
        move_tcp(self.session, [.48, .16, .22])
        self.session._advance(.5)
        self.sample(stage='RETREAT')
        return self.sequence.supplement(self.session.success_diagnostics())

    def test_half_depth_held_then_release_satisfies_temporal_requirement(self):
        self.carry_over_socket()
        move_tcp(self.session, [.48, .16, .104])
        result = self.sample(stage='INSERT')
        self.assertTrue(result['current']['half_inserted_while_held'], result)
        self.assertGreaterEqual(result['current']['tip_penetration_m'], .025)
        self.sample(stage='RELEASE', token='RELEASE', boundary='before_gripper_open')
        final = self.finish()
        self.assertTrue(final['insertion_sequence']['passed'])
        self.assertTrue(final['success'], final['failed_criteria'])

    def test_release_above_hole_and_gravity_seating_is_not_process_success(self):
        self.carry_over_socket()
        self.sample(stage='INSERT', token='RELEASE', boundary='before_gripper_open')
        final = self.finish()
        self.assertTrue(self.session.success_diagnostics()['success'], 'fixture must end seated and open')
        self.assertFalse(final['success'])
        self.assertEqual(final['insertion_sequence']['violation']['reason'], 'release_before_half_insertion')
        self.assertIsNone(final['insertion_sequence']['half_inserted_while_held'])

    def test_shallow_entry_does_not_authorize_release(self):
        self.carry_over_socket()
        move_tcp(self.session, [.48, .16, .12])
        result = self.sample(stage='INSERT', token='RELEASE', boundary='before_gripper_open')
        self.assertGreater(result['current']['tip_penetration_m'], 0)
        self.assertLess(result['current']['tip_penetration_m'], .025)
        self.assertFalse(result['passed'])
        self.assertEqual(result['violation']['reason'], 'release_before_half_insertion')

    def test_previous_half_depth_does_not_authorize_release_after_withdrawal(self):
        self.carry_over_socket()
        move_tcp(self.session, [.48, .16, .104])
        self.assertTrue(self.sample(stage='INSERT')['current']['half_inserted_while_held'])
        move_tcp(self.session, [.48, .16, .20])
        result = self.sample(stage='RELEASE', token='RELEASE', boundary='before_gripper_open')
        self.assertEqual(result['violation']['reason'], 'release_without_current_held_half_insertion')
        self.assertFalse(result['passed'])

    def test_blocked_intent_does_not_certify_release_or_physical_violation(self):
        self.carry_over_socket()
        intent = self.sample(stage='RELEASE', token='RELEASE', boundary='before_control')
        self.assertIsNone(intent['violation'])
        self.assertIsNone(intent['qualified_release'])
        self.assertFalse(intent['passed'])
        move_tcp(self.session, [.48, .16, .104])
        intent = self.sample(stage='RELEASE', token='RELEASE', boundary='before_control')
        self.assertIsNotNone(intent['half_inserted_while_held'])
        self.assertIsNone(intent['qualified_release'])
        self.assertFalse(intent['release_pending'])
        dispatch = self.sample(stage='RELEASE', token='RELEASE', boundary='before_gripper_open')
        self.assertTrue(dispatch['release_pending'])
        self.assertFalse(dispatch['passed'])
        self.assertTrue(self.finish()['insertion_sequence']['passed'])

    def test_seated_empty_closed_gripper_cannot_qualify_as_held(self):
        self.carry_over_socket()
        self.session.control_gripper(False)
        self.session._advance(2.)
        self.session.control_gripper(True)
        result = self.sample(stage='INSERT')
        self.assertGreater(result['current']['tip_penetration_m'], .04)
        self.assertFalse(result['current']['held_by_both_pads'])
        self.assertFalse(result['current']['half_inserted_while_held'])
        self.assertEqual(result['violation']['reason'], 'grasp_lost_before_half_insertion')


if __name__ == '__main__':
    unittest.main()
