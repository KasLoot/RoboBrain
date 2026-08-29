"""Probe RoboBrain failure detection without changing production code.

The script has two useful modes:

1. Offline audit of an existing ``output.log``.
2. Live comparison of the current single-frame payload with temporal and
   cropped-temporal payloads sent to the OpenAI-compatible VLM endpoint.

Examples:

    uv run python tests/probe_failure_detection.py --dry-run --log output.log
    uv run python tests/probe_failure_detection.py --run-live

The live probe expects the same local endpoint and model configured by
``src/robobrain/__main__.py`` unless they are overridden on the command line.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import cv2
from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


DEFAULT_VIDEO = Path("video/failure_part.mp4")
DEFAULT_LOG = Path("output.log")
DEFAULT_SYSTEM_PROMPT = Path("src/robobrain/prompt/brain_prompt.md")
DEFAULT_TASK_PROMPT = Path("src/robobrain/prompt/TASK.md")
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "/workspace/models/gemma-4-26B-A4B-it"

ROTATION_RANGE = range(460, 976)
DROP_RANGE = range(984, 1024)


TEMPORAL_ADDENDUM = """

# Temporal evidence supplied by the failure-detection probe

- You may receive multiple labelled frames. They are ordered OLDEST to
  NEWEST. Judge the newest frame, using all earlier frames as evidence of
  motion and state transitions.
- A failure flag is an episode-level event. Once the supplied frames show a
  failure flag, return FAIL even if the newest frame is static or the robot
  could retry.
- "plug rotates in the gripper" means that the plug changes orientation
  relative to the gripper jaws; rigid motion of the gripper and plug together
  is not this failure.
- "plugging action misses the target socket pinhole" means that the plug tip
  passes beside the intended opening or contacts the socket housing without
  entering the opening.
- "plug drops on the table" means a visible transition from a plug constrained
  by the gripper to an unsupported/falling plug or a plug resting on the table
  while the gripper is empty.
"""


@dataclass(frozen=True)
class ProbeCase:
    name: str
    event: str
    frame_indices: tuple[int, ...]
    crop_to_action: bool
    temporal_prompt: bool
    expect_failure: bool


CASES = (
    ProbeCase(
        name="nominal_single_full",
        event="nominal_control",
        frame_indices=(440,),
        crop_to_action=False,
        temporal_prompt=False,
        expect_failure=False,
    ),
    ProbeCase(
        name="nominal_temporal_action_crop",
        event="nominal_control",
        frame_indices=(280, 320, 360, 400, 440),
        crop_to_action=True,
        temporal_prompt=True,
        expect_failure=False,
    ),
    ProbeCase(
        name="rotation_single_full",
        event="rotation_or_socket_miss",
        frame_indices=(956,),
        crop_to_action=False,
        temporal_prompt=False,
        expect_failure=True,
    ),
    ProbeCase(
        name="rotation_temporal_full",
        event="rotation_or_socket_miss",
        frame_indices=(460, 590, 720, 850, 975),
        crop_to_action=False,
        temporal_prompt=True,
        expect_failure=True,
    ),
    ProbeCase(
        name="rotation_temporal_action_crop",
        event="rotation_or_socket_miss",
        frame_indices=(460, 590, 720, 850, 975),
        crop_to_action=True,
        temporal_prompt=True,
        expect_failure=True,
    ),
    ProbeCase(
        name="drop_single_full",
        event="drop",
        frame_indices=(1001,),
        crop_to_action=False,
        temporal_prompt=False,
        expect_failure=True,
    ),
    ProbeCase(
        name="drop_temporal_full",
        event="drop",
        frame_indices=(975, 984, 990, 996, 1002, 1008, 1023),
        crop_to_action=False,
        temporal_prompt=True,
        expect_failure=True,
    ),
    ProbeCase(
        name="drop_temporal_action_crop",
        event="drop",
        frame_indices=(975, 984, 990, 996, 1002, 1008, 1023),
        crop_to_action=True,
        temporal_prompt=True,
        expect_failure=True,
    ),
)


@dataclass(frozen=True)
class LogRecord:
    run: int
    frame_index: int
    status: str | None
    failure_description: str | None


def parse_json_object(content: Any) -> dict[str, Any] | None:
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return None

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_log(path: Path) -> list[LogRecord]:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^Frame \[(?P<frame>\d+)\]\s*(?P<body>.*?)"
        r"(?=^Frame \[|\Z)",
        flags=re.MULTILINE | re.DOTALL,
    )

    records: list[LogRecord] = []
    run = 1
    previous_frame: int | None = None
    for match in pattern.finditer(text):
        frame_index = int(match.group("frame"))
        if previous_frame is not None and frame_index < previous_frame:
            run += 1
        previous_frame = frame_index

        parsed = parse_json_object(match.group("body"))
        records.append(
            LogRecord(
                run=run,
                frame_index=frame_index,
                status=parsed.get("status") if parsed else None,
                failure_description=(
                    parsed.get("failure_description") if parsed else None
                ),
            )
        )
    return records


def event_log_summary(
    records: Iterable[LogRecord],
    frame_range: range,
    include_samples: bool,
) -> dict[str, Any]:
    samples_by_run: dict[int, list[LogRecord]] = {}
    for record in records:
        if record.frame_index in frame_range:
            samples_by_run.setdefault(record.run, []).append(record)

    detected_runs = [
        run
        for run, samples in samples_by_run.items()
        if any(sample.status == "FAIL" for sample in samples)
    ]
    eligible_runs = len(samples_by_run)
    result = {
        "frame_range": [frame_range.start, frame_range.stop - 1],
        "runs_with_samples": eligible_runs,
        "runs_with_fail": len(detected_runs),
        "run_level_recall": (
            len(detected_runs) / eligible_runs if eligible_runs else None
        ),
        "detected_runs": detected_runs,
        "sample_counts": {
            str(run): len(samples) for run, samples in samples_by_run.items()
        },
    }
    if include_samples:
        result["samples"] = {
            str(run): [
                {
                    "frame": sample.frame_index,
                    "status": sample.status,
                    "failure_description": sample.failure_description,
                }
                for sample in samples
            ]
            for run, samples in samples_by_run.items()
        }
    return result


def audit_log(path: Path, include_samples: bool = False) -> dict[str, Any]:
    records = parse_log(path)
    all_failures = []
    for position, record in enumerate(records):
        if record.status != "FAIL":
            continue
        next_record = records[position + 1] if position + 1 < len(records) else None
        if next_record is not None and next_record.run != record.run:
            next_record = None
        all_failures.append(
            {
                "run": record.run,
                "frame": record.frame_index,
                "failure_description": record.failure_description,
                "next_frame": next_record.frame_index if next_record else None,
                "next_status": next_record.status if next_record else None,
            }
        )
    return {
        "path": str(path),
        "runs": max((record.run for record in records), default=0),
        "records": len(records),
        "rotation_or_socket_miss": event_log_summary(
            records,
            ROTATION_RANGE,
            include_samples,
        ),
        "drop": event_log_summary(records, DROP_RANGE, include_samples),
        "all_failures": all_failures,
    }


def read_video_frames(
    video_path: Path,
    wanted_indices: Iterable[int],
) -> tuple[dict[int, Any], dict[str, Any]]:
    wanted = set(wanted_indices)
    if not wanted:
        raise ValueError("At least one frame index is required")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    metadata = {
        "fps": capture.get(cv2.CAP_PROP_FPS),
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    frames: dict[int, Any] = {}
    frame_index = 0
    final_index = max(wanted)
    try:
        while frame_index <= final_index:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in wanted:
                frames[frame_index] = frame.copy()
            frame_index += 1
    finally:
        capture.release()

    missing = sorted(wanted.difference(frames))
    if missing:
        raise RuntimeError(f"Could not read requested video frames: {missing}")
    return frames, metadata


def action_crop(frame: Any) -> Any:
    """Crop the gripper, plug, socket, and nearby table from this video."""
    height, width = frame.shape[:2]
    left = round(width * 0.365)
    right = round(width * 0.835)
    bottom = round(height * 0.65)
    return frame[0:bottom, left:right].copy()


def encode_image(frame: Any) -> str:
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
    )
    if not ok:
        raise RuntimeError("Could not JPEG-encode a probe frame")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def build_probe_messages(
    case: ProbeCase,
    all_frames: dict[int, Any],
    system_prompt: str,
    task_prompt: str,
) -> list[Any]:
    if not case.temporal_prompt:
        frame = all_frames[case.frame_indices[0]]
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                "data:image/jpeg;base64," + encode_image(frame)
                            )
                        },
                    },
                    {"type": "text", "text": task_prompt},
                ]
            ),
        ]

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "The labelled observations below are ordered oldest to "
                "newest. Compare the plug pose and support state across them."
            ),
        }
    ]
    for position, frame_index in enumerate(case.frame_indices):
        frame = all_frames[frame_index]
        if case.crop_to_action:
            frame = action_crop(frame)
        age_label = "OLDEST" if position == 0 else "NEWEST" if position == len(
            case.frame_indices
        ) - 1 else "INTERMEDIATE"
        content.extend(
            (
                {
                    "type": "text",
                    "text": f"Source video frame {frame_index} ({age_label})",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + encode_image(frame)
                    },
                },
            )
        )
    content.append({"type": "text", "text": task_prompt})
    return [
        SystemMessage(content=system_prompt + TEMPORAL_ADDENDUM),
        HumanMessage(content=content),
    ]


def save_case_frames(
    output_directory: Path,
    frames: dict[int, Any],
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        case_directory = output_directory / case.name
        case_directory.mkdir(parents=True, exist_ok=True)
        for frame_index in case.frame_indices:
            frame = frames[frame_index]
            if case.crop_to_action:
                frame = action_crop(frame)
            output_path = case_directory / f"frame_{frame_index:04d}.jpg"
            if not cv2.imwrite(str(output_path), frame):
                raise RuntimeError(f"Could not write {output_path}")


def live_probe(
    frames: dict[int, Any],
    system_prompt: str,
    task_prompt: str,
    base_url: str,
    model: str,
    timeout: float,
    repeats: int,
) -> dict[str, Any]:
    model_client = ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key="EMPTY",
        temperature=0,
        timeout=timeout,
        max_retries=0,
    )

    results: list[dict[str, Any]] = []
    for case in CASES:
        messages = build_probe_messages(
            case,
            frames,
            system_prompt,
            task_prompt,
        )
        for repeat in range(1, repeats + 1):
            response = model_client.invoke(messages)
            parsed = parse_json_object(response.content)
            status = parsed.get("status") if parsed else None
            result = {
                "case": case.name,
                "event": case.event,
                "repeat": repeat,
                "frame_indices": list(case.frame_indices),
                "crop_to_action": case.crop_to_action,
                "temporal_prompt": case.temporal_prompt,
                "expect_failure": case.expect_failure,
                "status": status,
                "detected": status == "FAIL",
                "passed": (status == "FAIL") == case.expect_failure,
                "parsed_response": parsed,
                "raw_response": response.content,
            }
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

    by_case: dict[str, dict[str, Any]] = {}
    for case in CASES:
        case_results = [item for item in results if item["case"] == case.name]
        detections = sum(bool(item["detected"]) for item in case_results)
        passes = sum(bool(item["passed"]) for item in case_results)
        by_case[case.name] = {
            "detections": detections,
            "attempts": len(case_results),
            "detection_rate": detections / len(case_results),
            "passes": passes,
            "pass_rate": passes / len(case_results),
        }
    return {"results": results, "summary": by_case}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument(
        "--skip-log",
        action="store_true",
        help="Skip the historical log audit (useful for live-only probes).",
    )
    parser.add_argument(
        "--system-prompt",
        type=Path,
        default=DEFAULT_SYSTEM_PROMPT,
    )
    parser.add_argument("--task-prompt", type=Path, default=DEFAULT_TASK_PROMPT)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--run-live",
        action="store_true",
        help="Send all probe cases to the configured VLM endpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate video extraction and payload dimensions without inference.",
    )
    parser.add_argument(
        "--save-frames",
        type=Path,
        help="Optionally save the exact full/cropped probe images for inspection.",
    )
    parser.add_argument(
        "--include-log-samples",
        action="store_true",
        help="Include every event-window log record instead of concise counts.",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")

    output: dict[str, Any] = {}
    if not args.skip_log and args.log.exists():
        output["log_audit"] = audit_log(
            args.log,
            include_samples=args.include_log_samples,
        )

    wanted_indices = {
        frame_index for case in CASES for frame_index in case.frame_indices
    }
    frames, metadata = read_video_frames(args.video, wanted_indices)
    output["video"] = {"path": str(args.video), **metadata}
    output["cases"] = [
        {
            "name": case.name,
            "event": case.event,
            "frame_indices": list(case.frame_indices),
            "crop_to_action": case.crop_to_action,
            "temporal_prompt": case.temporal_prompt,
            "expect_failure": case.expect_failure,
            "image_shapes": [
                list(
                    (
                        action_crop(frames[index])
                        if case.crop_to_action
                        else frames[index]
                    ).shape
                )
                for index in case.frame_indices
            ],
        }
        for case in CASES
    ]

    if args.save_frames:
        save_case_frames(args.save_frames, frames)
        output["saved_frames"] = str(args.save_frames)

    if args.run_live:
        system_prompt = args.system_prompt.read_text(encoding="utf-8")
        task_prompt = args.task_prompt.read_text(encoding="utf-8")
        output["live_probe"] = live_probe(
            frames=frames,
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            base_url=args.base_url,
            model=args.model,
            timeout=args.timeout,
            repeats=args.repeats,
        )
    elif not args.dry_run:
        print(
            "No live inference requested; pass --run-live to contact the VLM "
            "or --dry-run for an explicit offline audit.",
            file=sys.stderr,
        )

    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
