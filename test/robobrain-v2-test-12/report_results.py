# /// script
# requires-python = ">=3.13"
# dependencies = ["httpx>=0.28,<1", "opencv-python>=4.11,<6"]
# ///
"""Score every annotated episode in completed runs, preserving raw predictions."""
import argparse
import json
from pathlib import Path

from audit import audit
from evaluate_all import DATASETS
from run import HERE, summary, write_json

MANIFESTS = ("dataset-manifest-105.json", "dataset-manifest-aug27.json", "dataset-manifest-no-force.json")
TIMINGS = ("event_times.json", "event_times_aug27.json", "event_times_no_force.json")


def complete_rows(rows, all_names):
    names = [r["video"] for r in rows]
    if len(names) != len(set(names)) or not set(names) <= all_names:
        raise ValueError("Duplicate or unexpected episode results")
    if all_names - set(names):
        raise ValueError("Run is incomplete: a scored episode is missing")
    return rows


def collect(runs):
    combined, datasets = [], []
    for (name, dataset), manifest_name, timing_name in zip(DATASETS, MANIFESTS, TIMINGS):
        run = runs / name
        if (run / "STOPPED.json").exists():
            raise ValueError(f"Refusing to score a stopped run: {run}")
        raw = json.loads((run / "summary.json").read_text())
        manifest = json.loads((HERE / manifest_name).read_text())
        all_names = {r["video"] for r in manifest["episodes"]}
        rows = complete_rows(raw["results"], all_names)
        scored = {**summary(rows), "dataset": dataset, "total_clips": len(all_names),
                  "raw_replayed_clips": len(raw["results"]),
                  "raw_results": "summary.json"}
        write_json(run / "scored_summary.json", scored)
        report = audit(run, json.loads((HERE / timing_name).read_text()))
        report.update(dataset=dataset)
        write_json(run / "scored_audit.json", report)
        datasets.append({k: v for k, v in scored.items() if k != "results"})
        combined.extend({"dataset": dataset, **row} for row in rows)
    result = {**summary(combined), "total_clips": sum(d["total_clips"] for d in datasets),
              "datasets": datasets}
    write_json(runs / "scored_combined_summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("results", "datasets")}, indent=2))


def check():
    rows = [{"video": "traj-1.MOV"}, {"video": "traj-93.MOV"}]
    names = {r["video"] for r in rows}
    assert complete_rows(rows, names) == rows
    for bad in (rows[:1], rows + rows[:1], rows + [{"video": "missing.MOV"}]):
        try:
            complete_rows(bad, names)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid or incomplete evaluation accepted")
    print("Scoring checks passed: every episode included, incomplete/duplicate/unexpected results rejected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=HERE / "runs")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    check() if args.self_check else collect(args.runs)
