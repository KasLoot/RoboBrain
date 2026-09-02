import unittest

from fusion_policies import ConsecutiveSafetyFusion


class ConsecutiveSafetyFusionTests(unittest.TestCase):
    def test_three_rotation_consensus_observations_are_required(self):
        fusion = ConsecutiveSafetyFusion(rotation_required=3)
        self.assertIsNone(fusion.update("R", "R", "N", "N"))
        self.assertIsNone(fusion.update("R", "R", "N", "N"))
        self.assertEqual(
            fusion.update("R", "R", "N", "N"),
            "ROTATED_OR_MISALIGNED",
        )

    def test_split_rotation_vote_resets_the_streak(self):
        fusion = ConsecutiveSafetyFusion(rotation_required=3)
        fusion.update("R", "R", "N", "N")
        fusion.update("R", "R", "N", "N")
        self.assertIsNone(fusion.update("R", "N", "N", "N"))
        self.assertEqual(fusion.rotation_streak, 0)

    def test_drop_requires_same_frame_consensus(self):
        fusion = ConsecutiveSafetyFusion(rotation_required=3)
        self.assertIsNone(fusion.update("U", "U", "D", "N"))
        self.assertEqual(
            fusion.update("U", "U", "D", "D"),
            "DROP_OR_CONTROL_LOSS",
        )

    def test_failure_is_latched(self):
        fusion = ConsecutiveSafetyFusion(rotation_required=3)
        fusion.update("U", "U", "D", "D")
        self.assertEqual(
            fusion.update("N", "N", "N", "N"),
            "DROP_OR_CONTROL_LOSS",
        )


if __name__ == "__main__":
    unittest.main()
