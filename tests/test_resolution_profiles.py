import unittest
from unittest.mock import patch

import cv2
import numpy as np

from robobrain.__main__ import (
    RESOLUTION_CONFIGS,
    SimulatedCamera,
    UnsupportedResolutionError,
    crop_frame,
    load_terminal_reference_banks,
    main,
    make_comparison_grid,
    resolution_config_for,
    tight_current_crop,
)


class ResolutionProfileTests(unittest.TestCase):
    def test_supported_profiles_have_explicit_crop_boxes(self):
        low_resolution = resolution_config_for(848, 480)
        self.assertEqual(low_resolution.crop_top_left, (350, 50))
        self.assertEqual(low_resolution.crop_bottom_right, (650, 300))
        self.assertEqual(low_resolution.crop_size, (300, 250))
        self.assertEqual(low_resolution.tight_crop_size, (180, 170))

        high_resolution = resolution_config_for(1920, 1080)
        self.assertEqual(high_resolution.crop_top_left, (792, 112))
        self.assertEqual(high_resolution.crop_bottom_right, (1472, 675))
        self.assertEqual(high_resolution.crop_size, (680, 563))
        self.assertEqual(high_resolution.tight_crop_size, (407, 383))

    def test_unsupported_resolution_is_rejected(self):
        with self.assertRaisesRegex(
            UnsupportedResolutionError,
            r"Unsupported resolution: 1280x720",
        ):
            resolution_config_for(1280, 720)

    def test_main_quits_before_model_setup_for_unsupported_resolution(self):
        error = UnsupportedResolutionError(
            "Unsupported resolution: 1280x720."
        )
        with (
            patch("robobrain.__main__.SimulatedCamera", side_effect=error),
            patch("builtins.print") as print_mock,
        ):
            self.assertEqual(main(), 1)
        print_mock.assert_called_once_with(error)

    def test_live_and_reference_panels_match_for_every_profile(self):
        for profile in RESOLUTION_CONFIGS.values():
            with self.subTest(source_size=profile.source_size):
                source_width, source_height = profile.source_size
                frame = np.zeros(
                    (source_height, source_width, 3),
                    dtype=np.uint8,
                )
                current = crop_frame(
                    frame,
                    profile.crop_box,
                    expected_source_size=profile.source_size,
                )
                tight_current = tight_current_crop(frame, profile)
                expected_standard_shape = (
                    profile.crop_size[1],
                    profile.crop_size[0],
                    3,
                )
                expected_tight_shape = (
                    profile.tight_crop_size[1],
                    profile.tight_crop_size[0],
                    3,
                )
                self.assertEqual(current.shape, expected_standard_shape)
                self.assertEqual(tight_current.shape, expected_tight_shape)

                reference_banks = load_terminal_reference_banks(profile)
                for bank_name, references in reference_banks.items():
                    expected_shape = (
                        expected_tight_shape
                        if bank_name.startswith("rotation_")
                        else expected_standard_shape
                    )
                    for label, reference in references:
                        with self.subTest(bank=bank_name, label=label):
                            self.assertEqual(reference.shape, expected_shape)

    def test_runtime_pipeline_does_not_resize_live_or_reference_images(self):
        profile = resolution_config_for(1920, 1080)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        with patch(
            "robobrain.__main__.cv2.resize",
            side_effect=AssertionError("runtime resize is forbidden"),
        ):
            current = crop_frame(
                frame,
                profile.crop_box,
                expected_source_size=profile.source_size,
            )
            tight_current = tight_current_crop(frame, profile)
            reference_banks = load_terminal_reference_banks(profile)

        self.assertEqual(
            (current.shape[1], current.shape[0]),
            profile.crop_size,
        )
        self.assertEqual(
            (tight_current.shape[1], tight_current.shape[0]),
            profile.tight_crop_size,
        )
        self.assertTrue(reference_banks)

    def test_grid_rejects_mixed_panel_resolutions(self):
        reference = np.zeros((250, 300, 3), dtype=np.uint8)
        current = np.zeros((563, 680, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "panel resolution mismatch"):
            make_comparison_grid([("REFERENCE", reference)], current)

    def test_camera_detects_both_supported_video_resolutions(self):
        class FakeCapture:
            def __init__(self, resolution):
                self.resolution = resolution
                self.released = False

            def isOpened(self):
                return True

            def get(self, property_id):
                values = {
                    cv2.CAP_PROP_FRAME_WIDTH: self.resolution[0],
                    cv2.CAP_PROP_FRAME_HEIGHT: self.resolution[1],
                    cv2.CAP_PROP_FPS: 30,
                }
                return values[property_id]

            def release(self):
                self.released = True

        for expected_resolution in RESOLUTION_CONFIGS:
            with self.subTest(source_size=expected_resolution):
                capture = FakeCapture(expected_resolution)
                with patch(
                    "robobrain.__main__.cv2.VideoCapture",
                    return_value=capture,
                ):
                    camera = SimulatedCamera("input.mov", loop=False)
                try:
                    self.assertEqual(
                        camera.resolution_config.source_size,
                        expected_resolution,
                    )
                finally:
                    camera.stop()
                self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
