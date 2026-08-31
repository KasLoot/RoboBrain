"""Unit tests for the production attempt-level failure fusion."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robobrain.__main__ import AttemptFailureTracker, fuse_detection_response


def scene(
    *,
    stand="EMPTY",
    control="CONTROLLED",
    history="GRASP_CONFIRMED",
    status="ON_GOING",
    direction="SAME",
):
    return {
        "white_stand_state": stand,
        "gripper_control": control,
        "pin_direction_vs_gripper": direction,
        "pin_alignment_with_socket": "NOT_AT_SOCKET",
        "insertion_state": "NOT_INSERTED",
        "grasp_history": history,
        "status": status,
        "failure_type": "NONE",
        "failure_description": "none",
    }


def rotation(value):
    return {"rotation_match": value}


def drop(value):
    return {"drop_match": value}


class AttemptFailureTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = AttemptFailureTracker(rotation_confirmation_count=2)

    def update_and_fuse(self, observation, rotation_value, drop_value):
        state = self.tracker.update(
            observation,
            rotation(rotation_value),
            drop(drop_value),
        )
        return state, fuse_detection_response(observation, state)

    def test_drop_is_ignored_before_first_grasp(self):
        state, fused = self.update_and_fuse(
            scene(
                control="NOT_YET_GRASPED",
                history="NOT_YET_GRASPED",
            ),
            "UNCERTAIN",
            "DROP",
        )

        self.assertFalse(state["grasp_confirmed"])
        self.assertIsNone(state["failure_type"])
        self.assertEqual(fused["status"], "ON_GOING")
        self.assertEqual(fused["failure_type"], "NONE")

    def test_rotation_requires_two_observations_then_latches(self):
        first_state, first = self.update_and_fuse(
            scene(), "ROTATED", "NO_DROP"
        )
        second_state, second = self.update_and_fuse(
            scene(), "ROTATED", "NO_DROP"
        )
        third_state, third = self.update_and_fuse(
            scene(), "NOT_ROTATED", "NO_DROP"
        )

        self.assertEqual(first_state["rotation_streak"], 1)
        self.assertEqual(first["status"], "ON_GOING")
        self.assertEqual(first["pin_direction_vs_gripper"], "UNCERTAIN")
        self.assertEqual(second_state["rotation_streak"], 2)
        self.assertEqual(second["status"], "FAIL")
        self.assertEqual(
            second["failure_type"], "ROTATED_OR_MISALIGNED"
        )
        self.assertEqual(second["pin_direction_vs_gripper"], "DIFFERENT")
        self.assertEqual(
            third_state["failure_type"], "ROTATED_OR_MISALIGNED"
        )
        self.assertEqual(third["status"], "FAIL")

    def test_visible_return_to_stand_resets_even_if_drop_model_disagrees(self):
        self.update_and_fuse(scene(), "ROTATED", "NO_DROP")
        self.update_and_fuse(scene(), "ROTATED", "NO_DROP")

        reset_state, reset_result = self.update_and_fuse(
            scene(
                stand="HOLDING_PLUG",
                control="NOT_YET_GRASPED",
                history="GRASP_CONFIRMED",
            ),
            "UNCERTAIN",
            "DROP",
        )

        self.assertTrue(reset_state["reset_visible"])
        self.assertFalse(reset_state["grasp_confirmed"])
        self.assertIsNone(reset_state["failure_type"])
        self.assertEqual(reset_result["status"], "ON_GOING")
        self.assertEqual(reset_result["grasp_history"], "NOT_YET_GRASPED")

    def test_isolated_rotation_after_reset_does_not_fail(self):
        self.update_and_fuse(
            scene(
                stand="HOLDING_PLUG",
                control="NOT_YET_GRASPED",
                history="NOT_YET_GRASPED",
            ),
            "UNCERTAIN",
            "NO_DROP",
        )
        self.update_and_fuse(scene(), "NOT_ROTATED", "NO_DROP")
        _, isolated = self.update_and_fuse(
            scene(), "ROTATED", "NO_DROP"
        )
        state, recovered = self.update_and_fuse(
            scene(), "NOT_ROTATED", "NO_DROP"
        )

        self.assertEqual(isolated["status"], "ON_GOING")
        self.assertEqual(state["rotation_streak"], 0)
        self.assertEqual(recovered["status"], "ON_GOING")

    def test_validated_drop_after_grasp_fails_immediately(self):
        self.update_and_fuse(scene(), "NOT_ROTATED", "NO_DROP")
        state, fused = self.update_and_fuse(
            scene(control="LOST", direction="NOT_VISIBLE"),
            "UNCERTAIN",
            "DROP",
        )

        self.assertEqual(state["failure_type"], "DROP_OR_CONTROL_LOSS")
        self.assertEqual(fused["status"], "FAIL")
        self.assertEqual(fused["failure_type"], "DROP_OR_CONTROL_LOSS")
        self.assertEqual(fused["gripper_control"], "LOST")

    def test_frame_1316_style_false_drop_is_vetoed(self):
        self.update_and_fuse(scene(), "NOT_ROTATED", "NO_DROP")
        state, fused = self.update_and_fuse(
            scene(control="LOST", direction="NOT_VISIBLE"),
            "NOT_ROTATED",
            "NO_DROP",
        )

        self.assertIsNone(state["failure_type"])
        self.assertEqual(fused["status"], "ON_GOING")
        self.assertEqual(fused["failure_type"], "NONE")
        self.assertEqual(fused["gripper_control"], "UNCERTAIN")


if __name__ == "__main__":
    unittest.main()
