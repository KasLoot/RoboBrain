"""Replay specialist outputs through an attempt-level temporal state machine."""

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from robobrain.__main__ import AttemptFailureTracker, fuse_detection_response


BASELINE_LOG_PATH = ROOT / "output.log"
SPECIALIST_RESULTS_PATH = (
    TEST_DIRECTORY / "artifacts" / "relation_specialists_log-replay.json"
)
OUTPUT_PATH = TEST_DIRECTORY / "artifacts" / "stateful_fusion.json"


def parse_baseline_log():
    text = BASELINE_LOG_PATH.read_text(encoding="utf-8")
    matches = list(re.finditer(r"Frame \[(\d+)\]\s+", text))
    results = {}
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        block = text[match.end() : end]
        object_start = block.find("{")
        object_end = block.rfind("}")
        if object_start < 0 or object_end <= object_start:
            continue
        results[int(match.group(1))] = json.loads(
            block[object_start : object_end + 1]
        )
    return results


def expected_status(frame_index):
    if 493 <= frame_index <= 1012:
        return "FAIL"
    if frame_index in {1724, 1771}:
        return "FAIL"
    return "ON_GOING"


def main():
    baseline = parse_baseline_log()
    specialist_rows = json.loads(
        SPECIALIST_RESULTS_PATH.read_text(encoding="utf-8")
    )

    tracker = AttemptFailureTracker(rotation_confirmation_count=2)
    replay = []

    for specialist_row in specialist_rows:
        frame_index = specialist_row["frame"]
        baseline_result = baseline[frame_index]
        failure_state = tracker.update(
            baseline_result,
            specialist_row["rotation"],
            specialist_row["drop"],
        )
        fused = fuse_detection_response(baseline_result, failure_state)
        actual_status = fused["status"]
        wanted_status = expected_status(frame_index)
        replay.append(
            {
                "frame": frame_index,
                "baseline_stand": baseline_result.get("white_stand_state"),
                "baseline_gripper": baseline_result.get("gripper_control"),
                **failure_state,
                "status": actual_status,
                "failure_type": fused["failure_type"],
                "expected_status": wanted_status,
                "pass": actual_status == wanted_status,
            }
        )

    OUTPUT_PATH.write_text(
        json.dumps(replay, indent=2),
        encoding="utf-8",
    )
    passed = sum(row["pass"] for row in replay)
    print(f"summary={passed}/{len(replay)}")
    for row in replay:
        print(
            f"frame={row['frame']} rotation={row['rotation_match']} "
            f"streak={row['rotation_streak']} drop={row['drop_match']} "
            f"reset={row['reset_visible']} status={row['status']} "
            f"pass={row['pass']}"
        )
    print(f"results={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
