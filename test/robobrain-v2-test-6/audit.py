# /// script
# requires-python = ">=3.13"
# dependencies = ["opencv-python>=4.11,<6"]
# ///
"""Audit completed episode logs; optionally create native-size evidence contact sheets."""
import argparse
import json
from pathlib import Path
import statistics

def inspect_events(events, n):
    active = None
    errors, finished = [], []
    for event in events:
        if event["type"] == "verification_started":
            if active is not None:
                errors.append("Overlapping verification rounds")
            active = event["round_id"]
        elif event["type"] == "verification_finished":
            if event["round_id"] != active:
                errors.append("Verification completed without matching start")
            observations = [e for e in events if e["type"] == "observation" and e.get("round_id") == active]
            expected = {(i, s) for i in range(n) for s in ("success", "drop")}
            keys = [(o["slot"], o["specialist"]) for o in observations]
            if len(keys) != 2*n or set(keys) != expected:
                errors.append("Incomplete or duplicated verification responses")
            if any(o["received_time_s"] > event["time_s"] + .001 for o in observations):
                errors.append("Decision preceded response completion")
            finished.append(event)
            active = None
    return errors, finished


def percentile(values, fraction):
    return sorted(values)[min(len(values)-1, int(fraction*len(values)))] if values else None


def audit(run, event_times=None):
    data = json.loads((run / "summary.json").read_text())
    config = json.loads((run / "run_config.json").read_text())
    # Episode names repeat across datasets; timestamps must be selected explicitly.
    event_times = {} if event_times is None else event_times
    latencies, normal_hz, verification_durations, sample_intervals, violations = [], [], [], [], []
    response_errors, completed_requests, rounds = 0, 0, 0
    onset_results, evidence = [], []
    for row in data["results"]:
        events = [json.loads(line) for line in (run / (Path(row["video"]).stem + ".jsonl")).read_text().splitlines()]
        issues, finished = inspect_events(events, config["evidence_horizon"])
        violations.extend({"video": row["video"], "issue": issue} for issue in issues)
        starts = {e["round_id"]: e for e in events if e["type"] == "verification_started"}
        rounds += len(finished)
        for end in finished:
            verification_durations.append(end["time_s"] - starts[end["round_id"]]["time_s"])
            samples = sorted((e for e in events if e["type"] == "observation" and e.get("round_id") == end["round_id"] and e["specialist"] == "success"), key=lambda e: e["slot"])
            sample_intervals.extend(b["captured_time_s"] - a["captured_time_s"] for a, b in zip(samples, samples[1:]))
        observations = [e for e in events if e["type"] == "observation"]
        completed_requests += len(observations)
        response_errors += sum("error" in e for e in observations)
        latencies.extend(e["latency_s"] for e in observations)
        if row.get("normal_hz") is not None:
            normal_hz.append(row["normal_hz"])
        if row["video"] in event_times and row["status"] == "FAIL":
            low, high = event_times[row["video"]]["onset_interval_s"]
            onset_results.append({"video": row["video"], "premature": row["decision_time_s"] < low,
                                  "latency_interval_s": [row["decision_time_s"] - high, row["decision_time_s"] - low]})
        specialist = "success" if row["status"] == "SUCCESS" else "drop"
        positives = [e for e in observations if e.get("round_id") == row.get("verification_round") and e["specialist"] == specialist and e["verdict"] == "POSITIVE"]
        if positives and row["status"] != "ON_GOING":
            selected = max(positives, key=lambda e: e["slot"])
            evidence.append({"video": row["video"], "status": row["status"], "source_frame": selected["source_frame"],
                             "source_time_s": selected["source_time_s"], "decision_time_s": row["decision_time_s"],
                             "crop": config["crops"][specialist]})
    report = {
        "episodes": data["episodes"], "correct_episode_labels": data["correct"],
        "success_correct": data["success_correct"], "drop_detected": data["drop_detected"],
        "false_failures": data["false_failures"], "false_successes": data["false_successes"],
        "unresolved": data["unresolved"], "completed_requests": completed_requests,
        "request_errors": response_errors, "request_latency_median_s": percentile(latencies, .5),
        "request_latency_p95_s": percentile(latencies, .95),
        "normal_hz_median": statistics.median(normal_hz) if normal_hz else None,
        "verification_rounds": rounds, "verification_duration_median_s": percentile(verification_durations, .5),
        "verification_sample_interval_median_s": percentile(sample_intervals, .5),
        "verification_sample_interval_p95_s": percentile(sample_intervals, .95),
        "scheduler_violations": violations, "reviewed_drop_timing": onset_results,
        "evidence": evidence,
    }
    (run / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def review_images(report, videos, output):
    import cv2
    import numpy as np
    output.mkdir(parents=True, exist_ok=True)
    successes = []
    for row in report["evidence"]:
        destination = output / (Path(row["video"]).stem + ".jpg")
        if not destination.exists():
            cap = cv2.VideoCapture(str(videos / row["video"]))
            cap.set(cv2.CAP_PROP_POS_FRAMES, row["source_frame"])
            ok, frame = cap.read()
            if not ok:
                cap.release()
                cap = cv2.VideoCapture(str(videos / row["video"]))
                for _ in range(row["source_frame"] + 1):
                    ok, frame = cap.read()
                    if not ok:
                        break
            cap.release()
            if not ok:
                raise ValueError(f"Cannot recover evidence frame for {row['video']}")
            x1, y1, x2, y2 = row["crop"]
            crop = frame[y1:y2, x1:x2]
            tile = np.zeros((crop.shape[0] + 40, crop.shape[1], 3), np.uint8)
            tile[40:] = crop
            text = f"{row['video']} {row['status']} evidence={row['source_time_s']:.2f}s decision={row['decision_time_s']:.2f}s"
            cv2.putText(tile, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1, cv2.LINE_AA)
            if not cv2.imwrite(str(destination), tile, [cv2.IMWRITE_JPEG_QUALITY,95]):
                raise ValueError("Cannot write review image")
        if row["status"] == "SUCCESS":
            successes.append(destination)
    for index in range(0, len(successes), 6):
        paths = successes[index:index+6]
        tiles = [cv2.imread(str(path)) for path in paths]
        tiles.extend(np.zeros_like(tiles[0]) for _ in range(6-len(tiles)))
        sheet = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
        if not cv2.imwrite(str(output / f"page-{index//6+1:02d}.jpg"), sheet):
            raise ValueError("Cannot write review sheet")
    print(f"Review images: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", type=Path)
    parser.add_argument("--videos", type=Path)
    parser.add_argument("--event-times", type=Path, help="Reviewed onset timestamps for this dataset; omit to skip event latency reporting")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        assert inspect_events([], 10) == ([], [])
        events = [{"type":"verification_started","round_id":1}, {"type":"verification_started","round_id":2}]
        assert "Overlapping verification rounds" in inspect_events(events,10)[0]
        assert percentile([1,3,2],.5) == 2
        print("Audit checks passed")
    else:
        if not args.run:
            parser.error("Specify a run directory")
        event_times = json.loads(args.event_times.read_text()) if args.event_times else None
        report = audit(args.run, event_times)
        print(json.dumps({k:v for k,v in report.items() if k != "evidence"}, indent=2))
        if args.videos:
            review_images(report,args.videos,args.run/"review")
