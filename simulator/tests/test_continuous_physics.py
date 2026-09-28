"""Actual continuous holds and temporal insertion checks; never call a VLM."""
import json
import unittest
from types import SimpleNamespace

import mujoco
import numpy as np

from core.sim.continuous_physics import configure_continuous_physics
from plug_physics_fixtures import (InsertionTrace, grasp_and_lift, insertion_sample,
                                   loaded_finger_contacts, make_plug_session, move_tcp,
                                   observe_seated_geometry)


class ContinuousPhysicsSettingsTests(unittest.TestCase):
    def model(self):
        return SimpleNamespace(opt=SimpleNamespace(noslip_iterations=0, noslip_tolerance=1e-6,
                                                   timestep=.002, cone=0, impratio=1.))

    def test_default_and_explicit_settings_preserve_timestep(self):
        model = self.model()
        expected = {'cone': 'pyramidal', 'impratio': 1., 'noslip_iterations': 0,
                    'noslip_tolerance': 1e-6, 'timestep_s': .002}
        self.assertEqual(configure_continuous_physics(model, {}), expected)
        report = configure_continuous_physics(model, {'continuous_physics': {
            'cone': 'elliptic', 'impratio': 100, 'noslip_iterations': 0}})
        self.assertEqual(report, {**expected, 'cone': 'elliptic', 'impratio': 100.})
        for value in (0, 5, 10, 20):
            with self.subTest(value=value):
                report = configure_continuous_physics(model, {'continuous_physics': {'noslip_iterations': value}})
                self.assertEqual(report['noslip_iterations'], value)
                self.assertEqual(report['timestep_s'], .002)

    def test_bad_options_fail_before_mutating_model(self):
        invalid = [{'noslip_iterations': value} for value in (True, False, -1, 21, 5.0, '5', None, [], {})]
        invalid += [{'cone': value} for value in (1, True, None, 'ELLIPTIC', 'unknown', [])]
        invalid += [{'impratio': value} for value in (True, False, 0, -1, '100', None, float('inf'), float('nan'))]
        for section in invalid:
            model = self.model()
            before = vars(model.opt).copy()
            with self.subTest(section=section), self.assertRaises(ValueError):
                configure_continuous_physics(model, {'continuous_physics': section})
            self.assertEqual(vars(model.opt), before)
        for section in (None, [], 5, 'settings'):
            with self.subTest(section=section), self.assertRaises(ValueError):
                configure_continuous_physics(self.model(), {'continuous_physics': section})


class ContinuousGripPhysicsTests(unittest.TestCase):
    def session(self, *, selected_profile=True):
        session = make_plug_session(0, cone='elliptic' if selected_profile else 'pyramidal',
                                   impratio=100 if selected_profile else 1)
        self.addCleanup(session.close)
        return session

    def test_center_and_edge_grasps_survive_120_sim_seconds_without_binding_object(self):
        for offset in (0., .022):
            with self.subTest(grasp_height_offset_m=offset):
                session = self.session()
                friction = session.model.geom_friction.copy()
                gains = session.model.actuator_gainprm.copy()
                equality_count = session.model.neq
                self.assertEqual(session.model.joint('plug_joint').type[0], mujoco.mjtJoint.mjJNT_FREE)
                grasp_and_lift(session, height_offset_m=offset)
                initial = session.data.body('plug').xpos - session.get_ee_pose()[:3]
                session._advance(120.)
                final = session.data.body('plug').xpos - session.get_ee_pose()[:3]
                drift = float(np.linalg.norm(final - initial))
                self.assertLess(drift, .0002, 'Continuous model latency must not consume pad overlap')
                self.assertTrue(loaded_finger_contacts(session)['bilateral_loaded_grasp'])
                self.assertGreater(session.get_gripper_position()[0], .02)
                height = float(session.data.body('plug').xpos[2])
                session.control_gripper(False)
                session._advance(1.)
                self.assertLess(session.data.body('plug').xpos[2], height - .05)
                self.assertGreater(session.get_gripper_position()[0], .07)
                self.assertFalse(loaded_finger_contacts(session)['bilateral_loaded_grasp'])
                self.assertFalse(any(warning.number for warning in session.data.warning))
                np.testing.assert_array_equal(session.model.geom_friction, friction)
                np.testing.assert_array_equal(session.model.actuator_gainprm, gains)
                self.assertEqual(session.model.neq, equality_count)
                self.assertEqual(session.model.opt.timestep, .002)
                self.assertEqual(session.model.opt.noslip_iterations, 0)

    def test_unconfigured_baseline_keeps_original_solver_and_free_release(self):
        session = self.session(selected_profile=False)
        self.assertEqual(session.model.opt.cone, mujoco.mjtCone.mjCONE_PYRAMIDAL)
        self.assertEqual(session.model.opt.impratio, 1.)
        self.assertEqual(session.model.opt.noslip_iterations, 0)
        grasp_and_lift(session)
        initial = session.data.body('plug').xpos.copy()
        session._advance(120.)
        self.assertTrue(loaded_finger_contacts(session)['bilateral_loaded_grasp'])
        self.assertLess(session.data.body('plug').xpos[2], initial[2])
        session.control_gripper(False)
        session._advance(1.)
        self.assertLess(session.data.body('plug').xpos[2], initial[2] - .05)
        self.assertFalse(any(warning.number for warning in session.data.warning))

    def assert_bounded_seating(self, session):
        metrics = observe_seated_geometry(session, seconds=5.)
        # This task requires real engagement before release and seated geometry.
        # The original angular-rest success predicate remains unchanged: retain
        # its measured failures explicitly instead of declaring every sample success.
        print('SEATING_METRICS', json.dumps(metrics))
        self.assertFalse(metrics['geometry_failures'], metrics)
        self.assertLess(metrics['max_depth_error_m'], .002, metrics)
        self.assertLess(max(metrics['position_peak_to_peak_m']), .002, metrics)
        self.assertLess(metrics['max_tilt_deg'], 6., metrics)
        self.assertFalse(any(warning.number for warning in session.data.warning))
        self.assertTrue(np.isfinite(session.data.qpos).all())
        return metrics

    def test_half_inserted_while_held_can_release_and_remain_in_socket(self):
        session = self.session()
        grasp_and_lift(session)
        trace = session.recorder = InsertionTrace(session)
        move_tcp(session, [.48, .16, .20])
        move_tcp(session, [.48, .16, .104])
        before = insertion_sample(session)
        self.assertGreaterEqual(before['tip_penetration_m'], .025)
        self.assertLess(before['tip_penetration_m'], .035)
        self.assertTrue(before['bilateral_loaded_grasp'])
        self.assertTrue(trace.qualified_before_release())
        session.control_gripper(False)
        move_tcp(session, [.48, .16, .20])
        self.assertTrue(trace.qualified_before_release())
        self.assert_bounded_seating(session)

    def test_baseline_freefall_can_seat_but_cannot_satisfy_held_half_insertion(self):
        # The original physics reproduces the logged failure: an early drop can
        # satisfy final-state success while never entering halfway under grasp.
        session = self.session(selected_profile=False)
        grasp_and_lift(session)
        trace = session.recorder = InsertionTrace(session)
        move_tcp(session, [.48, .16, .20])
        move_tcp(session, [.48, .16, .162])
        before = insertion_sample(session)
        self.assertLess(before['tip_penetration_m'], -.025)
        self.assertTrue(before['bilateral_loaded_grasp'])
        session.control_gripper(False)
        move_tcp(session, [.48, .16, .20])
        self.assertFalse(trace.qualified_before_release(), 'Later freefall through half-depth must never qualify')
        self.assertTrue(any(row['tip_penetration_m'] >= .025 and not row['bilateral_loaded_grasp'] for row in trace.rows))
        self.assert_bounded_seating(session)
