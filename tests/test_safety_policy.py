import unittest
import json
import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from robobrain.__main__ import (
    TerminalAttemptManager,
    LatestInferenceResult,
    load_terminal_reference_banks,
    main,
    make_current_focused_grid,
    resolution_config_for,
    run_inference,
)


def monitoring_manager(**kwargs):
    manager = TerminalAttemptManager(**kwargs)
    manager.observe_initial_state("S")
    manager.observe_initial_state("C")
    if manager.phase != "MONITORING":
        raise AssertionError("Test manager did not enter MONITORING")
    return manager


class TerminalSafetyPolicyTests(unittest.TestCase):
    def test_rotation_requires_three_observations_not_skipped_frames(self):
        manager = monitoring_manager(rotation_confirmation_count=3)

        first = manager.update_safety_specialists(100, "N", None, "R", "R")
        second = manager.update_safety_specialists(1000, "N", None, "R", "R")
        third = manager.update_safety_specialists(1001, "N", None, "R", "R")

        self.assertFalse(first["terminal_emitted"])
        self.assertFalse(second["terminal_emitted"])
        self.assertTrue(third["terminal_emitted"])
        self.assertEqual(third["failure_type"], "ROTATED_OR_MISALIGNED")
        self.assertEqual(third["rotation_consensus_streak"], 3)

    def test_split_rotation_vote_resets_consensus(self):
        manager = monitoring_manager(rotation_confirmation_count=3)
        manager.update_safety_specialists(100, "N", None, "R", "R")
        manager.update_safety_specialists(130, "N", None, "R", "R")
        update = manager.update_safety_specialists(160, "N", None, "R", "N")

        self.assertFalse(update["terminal_emitted"])
        self.assertEqual(update["rotation_consensus_streak"], 0)

    def test_drop_requires_two_votes_on_the_same_retained_frame(self):
        manager = monitoring_manager()
        trigger_only = manager.update_safety_specialists(
            6049,
            "D",
            None,
            "U",
            "U",
        )
        confirmed = manager.update_safety_specialists(
            6052,
            "D",
            "D",
            "U",
            "U",
        )

        self.assertFalse(trigger_only["terminal_emitted"])
        self.assertTrue(confirmed["terminal_emitted"])
        self.assertEqual(confirmed["failure_type"], "DROP_OR_CONTROL_LOSS")

    def test_safety_has_priority_over_completion(self):
        manager = monitoring_manager(rotation_confirmation_count=3)
        manager.update_safety_specialists(100, "N", None, "R", "R")
        manager.update_safety_specialists(130, "N", None, "R", "R")

        update = manager.update_terminal_specialists(
            160,
            "I",
            "I",
            "N",
            "R",
            "R",
        )

        self.assertEqual(update["terminal_status"], "FAIL")
        self.assertEqual(update["failure_type"], "ROTATED_OR_MISALIGNED")

    def test_completion_requires_trigger_and_verifier(self):
        manager = monitoring_manager()
        trigger_only = manager.update_completion_specialists(500, "I", None)
        confirmed = manager.update_completion_specialists(500, "I", "I")

        self.assertFalse(trigger_only["terminal_emitted"])
        self.assertTrue(confirmed["terminal_emitted"])
        self.assertEqual(confirmed["terminal_status"], "COMPLETE")


class CurrentFocusedGridTests(unittest.TestCase):
    def test_current_panel_keeps_native_pixels_and_dominates_grid(self):
        current = np.full((120, 180, 3), 77, dtype=np.uint8)
        references = [
            ("A", np.full_like(current, 10)),
            ("B", np.full_like(current, 20)),
            ("C", np.full_like(current, 30)),
        ]

        grid = make_current_focused_grid(references, current)

        label_height = max(22, round(current.shape[0] * 0.08))
        reference_width = current.shape[1] // 3
        reference_height = round(
            current.shape[0] * reference_width / current.shape[1]
        )
        current_y = label_height + reference_height + label_height
        self.assertEqual(grid.shape[1], current.shape[1])
        np.testing.assert_array_equal(grid[current_y:], current)

    def test_production_reference_subsets_match_validated_banks(self):
        banks = load_terminal_reference_banks(
            resolution_config_for(1920, 1080)
        )

        self.assertEqual(
            [label for label, _ in banks["drop_primary"]],
            ["DROPPED", "CONTROLLED", "INSERTED"],
        )
        self.assertEqual(
            [label for label, _ in banks["drop_confirmation"]],
            ["DROPPED", "STAND_RESET", "HUMAN"],
        )
        self.assertEqual(
            [label for label, _ in banks["rotation_primary_a"]],
            ["NORMAL_A", "NORMAL_B", "ROTATED_A"],
        )
        self.assertEqual(
            [label for label, _ in banks["rotation_primary_b"]],
            ["NORMAL_A", "NORMAL_C", "ROTATED_B"],
        )

    def test_main_rejects_unvalidated_low_resolution_profile(self):
        camera = Mock()
        camera.resolution_config = resolution_config_for(848, 480)
        with (
            patch("robobrain.__main__.SimulatedCamera", return_value=camera),
            patch("builtins.print") as print_mock,
        ):
            self.assertEqual(main(), 1)

        camera.stop.assert_called_once_with()
        self.assertIn(
            "validated only for 1920x1080",
            print_mock.call_args.args[0],
        )


class ProductionInferenceIntegrationTests(unittest.TestCase):
    class StopEvent:
        def __init__(self):
            self.value = False

        def is_set(self):
            return self.value

        def set(self):
            self.value = True

        def wait(self, timeout):
            del timeout

    class Camera:
        def __init__(self, stop_event, frame_indices):
            self.stop_event = stop_event
            self.frame_indices = iter(frame_indices)
            self.fps = 30.0
            self.resolution_config = resolution_config_for(1920, 1080)
            full = np.zeros((1080, 1920, 3), dtype=np.uint8)
            crop = np.zeros((563, 680, 3), dtype=np.uint8)
            self.frame_chunk = [[full, crop]]

        def get(self):
            try:
                return self.frame_chunk, next(self.frame_indices)
            except StopIteration:
                self.stop_event.set()
                return None, None

    def test_runtime_emits_rotation_after_three_sampled_consensuses(self):
        def run_with_contents():
            stop_event = self.StopEvent()
            camera = self.Camera(stop_event, (0, 1, 100, 1000, 1001))
            latest = LatestInferenceResult()
            banks = load_terminal_reference_banks(camera.resolution_config)
            state_vlm = object()
            inserted_vlm = object()
            drop_vlm = object()
            rotation_vlm = object()
            state_choices = iter(("S", "C"))

            def fake_invoke(model, messages):
                del messages
                if model is state_vlm:
                    content = next(state_choices)
                elif model is inserted_vlm:
                    content = "N"
                elif model is drop_vlm:
                    content = "N"
                else:
                    content = '{"closest":"ROTATED_A","rotation":"R"}'
                return SimpleNamespace(content=content), {"round_trip_ms": 1.0}

            with tempfile.TemporaryDirectory() as temporary_directory:
                temporary_path = Path(temporary_directory)
                with (
                    patch(
                        "robobrain.__main__.LOG_OUTPUT_PATH",
                        temporary_path / "output.log",
                    ),
                    patch(
                        "robobrain.__main__.LATENCY_OUTPUT_PATH",
                        temporary_path / "latency.jsonl",
                    ),
                    patch(
                        "robobrain.__main__.invoke_with_latency",
                        side_effect=fake_invoke,
                    ),
                    redirect_stdout(io.StringIO()),
                ):
                    run_inference(
                        state_vlm,
                        inserted_vlm,
                        drop_vlm,
                        rotation_vlm,
                        "state prompt",
                        "inserted prompt",
                        "drop prompt",
                        "rotation prompt",
                        banks["state"],
                        banks["inserted_a"],
                        banks["inserted_b"],
                        banks["drop_primary"],
                        banks["drop_confirmation"],
                        banks["rotation_primary_a"],
                        banks["rotation_primary_b"],
                        camera,
                        stop_event,
                        latest,
                    )
                latency_records = [
                    json.loads(line)
                    for line in (temporary_path / "latency.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
            return latest.snapshot(), latency_records

        (content, frame_index, _), records = run_with_contents()
        self.assertEqual(frame_index, 1001)
        self.assertEqual(json.loads(content)["status"], "FAIL")
        self.assertEqual(
            records[-1]["attempt"]["rotation_consensus_streak"],
            3,
        )
        self.assertIsNone(records[-1]["requests"]["inserted_b"])

    def test_runtime_confirms_drop_on_the_triggering_frame(self):
        stop_event = self.StopEvent()
        camera = self.Camera(stop_event, (0, 1, 6049))
        latest = LatestInferenceResult()
        banks = load_terminal_reference_banks(camera.resolution_config)
        state_vlm = object()
        inserted_vlm = object()
        drop_vlm = object()
        rotation_vlm = object()
        state_choices = iter(("S", "C"))

        def fake_invoke(model, messages):
            del messages
            if model is state_vlm:
                content = next(state_choices)
            elif model is inserted_vlm:
                content = "N"
            elif model is drop_vlm:
                content = "D"
            else:
                content = '{"closest":"NONE","rotation":"U"}'
            return SimpleNamespace(content=content), {"round_trip_ms": 1.0}

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            with (
                patch(
                    "robobrain.__main__.LOG_OUTPUT_PATH",
                    temporary_path / "output.log",
                ),
                patch(
                    "robobrain.__main__.LATENCY_OUTPUT_PATH",
                    temporary_path / "latency.jsonl",
                ),
                patch(
                    "robobrain.__main__.invoke_with_latency",
                    side_effect=fake_invoke,
                ),
                redirect_stdout(io.StringIO()),
            ):
                run_inference(
                    state_vlm,
                    inserted_vlm,
                    drop_vlm,
                    rotation_vlm,
                    "state prompt",
                    "inserted prompt",
                    "drop prompt",
                    "rotation prompt",
                    banks["state"],
                    banks["inserted_a"],
                    banks["inserted_b"],
                    banks["drop_primary"],
                    banks["drop_confirmation"],
                    banks["rotation_primary_a"],
                    banks["rotation_primary_b"],
                    camera,
                    stop_event,
                    latest,
                )
            last_record = json.loads(
                (temporary_path / "latency.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[-1]
            )

        content, frame_index, _ = latest.snapshot()
        self.assertEqual(frame_index, 6049)
        self.assertEqual(json.loads(content)["failure_type"], "DROP_OR_CONTROL_LOSS")
        self.assertEqual(last_record["observations"]["drop_a"], "D")
        self.assertEqual(last_record["observations"]["drop_b"], "D")
        self.assertIsNotNone(last_record["requests"]["drop_b"])

    def test_runtime_verifies_completion_after_safety(self):
        stop_event = self.StopEvent()
        camera = self.Camera(stop_event, (0, 1, 530))
        latest = LatestInferenceResult()
        banks = load_terminal_reference_banks(camera.resolution_config)
        state_vlm = object()
        inserted_vlm = object()
        drop_vlm = object()
        rotation_vlm = object()
        state_choices = iter(("S", "C"))

        def fake_invoke(model, messages):
            del messages
            if model is state_vlm:
                content = next(state_choices)
            elif model is inserted_vlm:
                content = "I"
            elif model is drop_vlm:
                content = "N"
            else:
                content = '{"closest":"NORMAL_A","rotation":"N"}'
            return SimpleNamespace(content=content), {"round_trip_ms": 1.0}

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            with (
                patch(
                    "robobrain.__main__.LOG_OUTPUT_PATH",
                    temporary_path / "output.log",
                ),
                patch(
                    "robobrain.__main__.LATENCY_OUTPUT_PATH",
                    temporary_path / "latency.jsonl",
                ),
                patch(
                    "robobrain.__main__.invoke_with_latency",
                    side_effect=fake_invoke,
                ),
                redirect_stdout(io.StringIO()),
            ):
                run_inference(
                    state_vlm,
                    inserted_vlm,
                    drop_vlm,
                    rotation_vlm,
                    "state prompt",
                    "inserted prompt",
                    "drop prompt",
                    "rotation prompt",
                    banks["state"],
                    banks["inserted_a"],
                    banks["inserted_b"],
                    banks["drop_primary"],
                    banks["drop_confirmation"],
                    banks["rotation_primary_a"],
                    banks["rotation_primary_b"],
                    camera,
                    stop_event,
                    latest,
                )
            last_record = json.loads(
                (temporary_path / "latency.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[-1]
            )

        content, frame_index, _ = latest.snapshot()
        self.assertEqual(frame_index, 530)
        self.assertEqual(json.loads(content)["status"], "COMPLETE")
        self.assertEqual(last_record["observations"]["inserted_a"], "I")
        self.assertEqual(last_record["observations"]["inserted_b"], "I")
        self.assertIsNotNone(last_record["requests"]["inserted_b"])


if __name__ == "__main__":
    unittest.main()
