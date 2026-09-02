"""Summarize fixed-frame VLM benchmark JSONL files."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)
    return round(ordered[index], 3)


def ratio(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else None


def load_records(path):
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
    return records


def consensus(record, task):
    predictions = record.get("predictions", {})
    if task == "rotation":
        return (
            "R"
            if predictions.get("rotation_a") == predictions.get("rotation_b") == "R"
            else "N"
        )
    if task == "drop":
        if "drop_b" in predictions:
            return (
                "D"
                if predictions.get("drop_a") == predictions.get("drop_b") == "D"
                else "N"
            )
        return "D" if predictions.get("drop_a") == "D" else "N"
    if task == "inserted":
        return (
            "I"
            if predictions.get("inserted_a") == predictions.get("inserted_b") == "I"
            else "N"
        )
    raise ValueError(task)


def task_metrics(
    records,
    task,
    positive,
    include_reference_leak=True,
    include_rotation_boundary=True,
):
    relevant = []
    for record in records:
        predictions = record.get("predictions", {})
        if task == "inserted" and "inserted_a" not in predictions:
            continue
        expected = record.get("expected", {}).get(task)
        if expected not in {positive, "N"}:
            continue
        if (
            not include_reference_leak
            and record.get("group") == "drop_positive_reference_leak"
        ):
            continue
        if (
            not include_rotation_boundary
            and record.get("group") == "pre_rotation_boundary"
        ):
            continue
        relevant.append((expected, consensus(record, task)))
    true_positive = sum(e == positive and p == positive for e, p in relevant)
    false_negative = sum(e == positive and p != positive for e, p in relevant)
    false_positive = sum(e == "N" and p == positive for e, p in relevant)
    true_negative = sum(e == "N" and p != positive for e, p in relevant)
    return {
        "tp": true_positive,
        "fn": false_negative,
        "fp": false_positive,
        "tn": true_negative,
        "recall": ratio(true_positive, true_positive + false_negative),
        "precision": ratio(true_positive, true_positive + false_positive),
        "false_positive_rate": ratio(false_positive, false_positive + true_negative),
    }


def summarize(path):
    records = load_records(path)
    completed = [record for record in records if record.get("error") is None]
    errors = [record for record in records if record.get("error") is not None]
    round_times = [record["parallel_round_ms"] for record in completed]
    repeat_votes = {}
    for record in completed:
        key = record["frame"]
        repeat_votes.setdefault(key, []).append(
            (
                consensus(record, "rotation"),
                consensus(record, "drop"),
                consensus(record, "inserted")
                if "inserted_a" in record.get("predictions", {})
                else None,
            )
        )
    unstable_frames = {
        frame: Counter(votes)
        for frame, votes in repeat_votes.items()
        if len(set(votes)) > 1
    }
    configuration = {
        key: completed[0].get(key) if completed else None
        for key in ("source", "port", "visual_token_label", "variant")
    }
    return {
        "path": str(path),
        "configuration": configuration,
        "records": len(records),
        "completed": len(completed),
        "errors": len(errors),
        "latency_ms": {
            "mean": round(statistics.fmean(round_times), 3) if round_times else None,
            "p50": percentile(round_times, 0.50),
            "p95": percentile(round_times, 0.95),
        },
        "rotation_strict_boundary": task_metrics(completed, "rotation", "R"),
        "rotation_without_boundary": task_metrics(
            completed,
            "rotation",
            "R",
            include_rotation_boundary=False,
        ),
        "drop_with_reference_frame": task_metrics(completed, "drop", "D"),
        "drop_without_reference_frame": task_metrics(
            completed,
            "drop",
            "D",
            include_reference_leak=False,
        ),
        "inserted": task_metrics(completed, "inserted", "I"),
        "unstable_frames": {
            str(frame): {str(vote): count for vote, count in counter.items()}
            for frame, counter in unstable_frames.items()
        },
    }


def main():
    arguments = parse_arguments()
    summaries = [summarize(path) for path in arguments.inputs]
    rendered = json.dumps(summaries, indent=2)
    print(rendered)
    if arguments.json_output:
        arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
