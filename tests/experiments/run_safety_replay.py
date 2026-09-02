"""Replay active-attempt segments at measured real-time VLM cadence."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2


EXPERIMENT_DIRECTORY = Path(__file__).resolve().parent
if str(EXPERIMENT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIRECTORY))

from fusion_policies import ConsecutiveSafetyFusion  # noqa: E402
from run_vlm_benchmark import (  # noqa: E402
    DROP_BINARY_PROMPT_PATH,
    MODEL_ID,
    ROTATION_VERIFY_PROMPT_PATH,
    SOURCE_PATHS,
    VLLMTimingChatOpenAI,
    compact_enum_choice,
    compact_rotation_vote,
    crop_frame,
    current_focused_requests,
    rotation_current_focused_requests,
    safety_primary_requests,
    load_terminal_reference_banks,
    resolution_config_for,
    tight_current_crop,
    invoke_request,
)


FPS = 30.0
ATTEMPTS = (
    {"attempt": 1, "start": 137, "end": 530, "expected": "SUCCESS"},
    {"attempt": 2, "start": 604, "end": 1097, "expected": "SUCCESS"},
    {"attempt": 3, "start": 1285, "end": 1668, "expected": "SUCCESS"},
    {"attempt": 4, "start": 1788, "end": 2288, "expected": "SUCCESS"},
    {"attempt": 5, "start": 2379, "end": 2869, "expected": "SUCCESS"},
    {"attempt": 6, "start": 3061, "end": 3491, "expected": "SUCCESS"},
    {"attempt": 7, "start": 3580, "end": 4072, "expected": "SUCCESS"},
    {"attempt": 8, "start": 4178, "end": 4856, "expected": "SUCCESS"},
    {"attempt": 9, "start": 4967, "end": 6060, "expected": "FAIL"},
    # Targeted fallback replay: start too late to accumulate three rotation
    # votes, so successful detection must come from the retained-frame drop
    # confirmation path.
    {"attempt": 10, "start": 6010, "end": 6060, "expected": "FAIL"},
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=tuple(SOURCE_PATHS), default="1920x1080")
    parser.add_argument("--port", type=int, choices=(8000, 9000), default=8000)
    parser.add_argument("--rotation-required", type=int, default=3)
    parser.add_argument(
        "--lane",
        choices=("combined", "rotation_only", "primary_3way"),
        default="combined",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempts", nargs="*", type=int)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    return parser.parse_args()


def read_frame(capture, frame_index):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index}")
    return frame


def main():
    arguments = parse_arguments()
    selected_attempts = set(arguments.attempts or ())
    attempts = [
        attempt
        for attempt in ATTEMPTS
        if not selected_attempts or attempt["attempt"] in selected_attempts
    ]
    if not attempts:
        raise ValueError("No attempts selected")

    width, height = (int(value) for value in arguments.source.split("x"))
    resolution = resolution_config_for(width, height)
    banks = load_terminal_reference_banks(resolution)
    prompts = {
        "drop": DROP_BINARY_PROMPT_PATH.read_text(encoding="utf-8"),
        "rotation": ROTATION_VERIFY_PROMPT_PATH.read_text(encoding="utf-8"),
        "inserted": "unused",
    }
    base_model = VLLMTimingChatOpenAI(
        model=MODEL_ID,
        base_url=f"http://127.0.0.1:{arguments.port}/v1",
        api_key="EMPTY",
        temperature=0,
        timeout=arguments.request_timeout,
        max_retries=1,
    )
    models = {
        "drop_a": base_model.bind(max_tokens=8),
        "drop_b": base_model.bind(max_tokens=8),
        "rotation_a": base_model.bind(max_tokens=24),
        "rotation_b": base_model.bind(max_tokens=24),
    }

    capture = cv2.VideoCapture(str(SOURCE_PATHS[arguments.source]))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {SOURCE_PATHS[arguments.source]}")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    attempt_summaries = []
    request_builder = {
        "combined": current_focused_requests,
        "rotation_only": rotation_current_focused_requests,
        "primary_3way": safety_primary_requests,
    }[arguments.lane]

    try:
        with (
            arguments.output.open("w", encoding="utf-8", buffering=1) as output_file,
            ThreadPoolExecutor(max_workers=4) as pool,
        ):
            for attempt in attempts:
                policy = ConsecutiveSafetyFusion(arguments.rotation_required)
                simulated_elapsed = 0.0
                frame_index = attempt["start"]
                sampled_frames = []
                terminal_event = None

                while frame_index <= attempt["end"]:
                    frame = read_frame(capture, frame_index)
                    current_crop = crop_frame(
                        frame,
                        resolution.crop_box,
                        expected_source_size=resolution.source_size,
                    )
                    tight_crop = tight_current_crop(frame, resolution)
                    messages = request_builder(
                        frame_index,
                        current_crop,
                        tight_crop,
                        prompts,
                        banks,
                    )
                    request_started = time.perf_counter()
                    futures = {
                        name: pool.submit(invoke_request, models[name], message)
                        for name, message in messages.items()
                    }
                    raw = {}
                    latencies = {}
                    predictions = {}
                    for name, future in futures.items():
                        content, latency = future.result()
                        raw[name] = content
                        latencies[name] = latency
                        predictions[name] = (
                            compact_rotation_vote(content)
                            if name.startswith("rotation_")
                            else compact_enum_choice(content, {"D", "N", "U"})
                        )
                    if (
                        arguments.lane == "primary_3way"
                        and predictions.get("drop_a") == "D"
                    ):
                        confirmation_message = current_focused_requests(
                            frame_index,
                            current_crop,
                            tight_crop,
                            prompts,
                            banks,
                        )["drop_b"]
                        content, latency = invoke_request(
                            models["drop_b"],
                            confirmation_message,
                        )
                        raw["drop_b"] = content
                        latencies["drop_b"] = latency
                        predictions["drop_b"] = compact_enum_choice(
                            content,
                            {"D", "N", "U"},
                        )
                    round_ms = (time.perf_counter() - request_started) * 1000
                    failure_type = policy.update(
                        predictions.get("rotation_a"),
                        predictions.get("rotation_b"),
                        predictions.get("drop_a"),
                        predictions.get("drop_b"),
                    )
                    sampled_frames.append(frame_index)
                    record = {
                        "attempt": attempt["attempt"],
                        "expected": attempt["expected"],
                        "frame": frame_index,
                        "predictions": predictions,
                        "raw": raw,
                        "latencies": latencies,
                        "parallel_round_ms": round(round_ms, 3),
                        "rotation_streak": policy.rotation_streak,
                        "failure_type": failure_type,
                    }
                    print(json.dumps(record, separators=(",", ":")), file=output_file)
                    print(
                        f"attempt={attempt['attempt']} expected={attempt['expected']} "
                        f"frame={frame_index} pred={predictions} "
                        f"streak={policy.rotation_streak} failure={failure_type} "
                        f"round={round_ms:.1f}ms",
                        flush=True,
                    )
                    if failure_type is not None:
                        terminal_event = {
                            "frame": frame_index,
                            "alert_completed_source_equivalent_frame": round(
                                frame_index + round_ms * FPS / 1000
                            ),
                            "failure_type": failure_type,
                        }
                        break

                    # Random access into the HEVC fixture is much slower than a
                    # live sequential camera. Advance the simulated source only
                    # by measured model time, not by offline seek/decode time.
                    simulated_elapsed += round_ms / 1000
                    next_frame = attempt["start"] + round(
                        simulated_elapsed * FPS
                    )
                    frame_index = max(frame_index + 1, next_frame)

                attempt_summaries.append(
                    {
                        **attempt,
                        "sampled_frames": sampled_frames,
                        "terminal_event": terminal_event,
                        "correct": (
                            terminal_event is None
                            if attempt["expected"] == "SUCCESS"
                            else terminal_event is not None
                        ),
                    }
                )
    finally:
        capture.release()

    summary_path = arguments.output.with_suffix(".summary.json")
    summary = {
        "source": arguments.source,
        "port": arguments.port,
        "lane": arguments.lane,
        "rotation_required": arguments.rotation_required,
        "attempts": attempt_summaries,
        "successful_attempt_false_alerts": sum(
            item["expected"] == "SUCCESS" and item["terminal_event"] is not None
            for item in attempt_summaries
        ),
        "failed_attempts_detected": sum(
            item["expected"] == "FAIL" and item["terminal_event"] is not None
            for item in attempt_summaries
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {arguments.output} and {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
