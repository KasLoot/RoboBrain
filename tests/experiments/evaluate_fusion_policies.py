"""Evaluate temporal policies on repeated fixed-frame predictions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from fusion_policies import ConsecutiveSafetyFusion


NORMAL_GROUPS = {"rotation_hard_negative", "drop_hard_negative"}
FAILURE_GROUPS = {
    "pre_rotation_boundary",
    "rotation_positive",
    "drop_positive",
    "drop_positive_reference_leak",
}
ROTATION_ONSET = 5493


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_records(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate_sequence(records, rotation_required):
    policy = ConsecutiveSafetyFusion(rotation_required=rotation_required)
    event = None
    for record in sorted(records, key=lambda item: item["frame"]):
        predictions = record["predictions"]
        failure_type = policy.update(
            predictions.get("rotation_a"),
            predictions.get("rotation_b"),
            predictions.get("drop_a"),
            predictions.get("drop_b"),
        )
        if failure_type is not None:
            event = {
                "frame": record["frame"],
                "failure_type": failure_type,
                "rotation_streak": policy.rotation_streak,
            }
            break
    return event


def main():
    arguments = parse_arguments()
    records = [record for record in read_records(arguments.input) if not record["error"]]
    by_repeat = defaultdict(list)
    for record in records:
        by_repeat[record["repeat"]].append(record)

    results = []
    for rotation_required in (1, 2, 3, 4):
        repeat_results = []
        for repeat, repeat_records in sorted(by_repeat.items()):
            normal = [
                record for record in repeat_records if record["group"] in NORMAL_GROUPS
            ]
            failure = [
                record for record in repeat_records if record["group"] in FAILURE_GROUPS
            ]
            normal_event = evaluate_sequence(normal, rotation_required)
            failure_event = evaluate_sequence(failure, rotation_required)
            repeat_results.append(
                {
                    "repeat": repeat,
                    "normal_false_event": normal_event,
                    "failure_event": failure_event,
                    "rotation_evidence_delay_frames": (
                        failure_event["frame"] - ROTATION_ONSET
                        if failure_event
                        and failure_event["failure_type"] == "ROTATED_OR_MISALIGNED"
                        else None
                    ),
                }
            )
        results.append(
            {
                "rotation_required": rotation_required,
                "normal_sequences_without_false_event": sum(
                    item["normal_false_event"] is None for item in repeat_results
                ),
                "failure_sequences_detected": sum(
                    item["failure_event"] is not None for item in repeat_results
                ),
                "repeat_results": repeat_results,
            }
        )

    summary = {
        "input": str(arguments.input),
        "policy_results": results,
    }
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
