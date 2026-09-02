"""Run fixed-frame VLM safety benchmarks without changing production code."""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from langchain.messages import HumanMessage, SystemMessage


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robobrain.__main__ import (  # noqa: E402
    DROP_BINARY_PROMPT_PATH,
    INSERTED_BINARY_PROMPT_PATH,
    ROTATION_VERIFY_PROMPT_PATH,
    VLLMTimingChatOpenAI,
    build_compact_grid_message,
    compact_enum_choice,
    compact_rotation_vote,
    crop_frame,
    load_terminal_reference_banks,
    resolution_config_for,
    tight_current_crop,
)
from robobrain.latency import invoke_with_latency  # noqa: E402


MODEL_ID = "/workspace/models/gemma-4-26B-A4B-it"
MANIFEST_PATH = Path(__file__).with_name("benchmark_frames.json")
RESULT_DIRECTORY = Path(__file__).with_name("results")
SOURCE_PATHS = {
    "1920x1080": REPOSITORY_ROOT
    / "video/insert_plug-full_model-aug-28-105trail-1x.MOV",
    "848x480": REPOSITORY_ROOT
    / "video/insert_plug-full_model-aug-28-105trail-1x-w848h480.MOV",
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=tuple(SOURCE_PATHS), required=True)
    parser.add_argument("--port", type=int, choices=(8000, 9000), required=True)
    parser.add_argument(
        "--variant",
        choices=(
            "production",
            "current_focused",
            "safety_primary",
            "production_primary",
            "rotation_current_focused",
            "drop_reference_focused",
            "drop_no_reference",
        ),
        default="production",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--groups",
        nargs="*",
        help="Optional benchmark groups; all groups run when omitted.",
    )
    parser.add_argument(
        "--frames",
        nargs="*",
        type=int,
        help="Optional explicit source-frame subset.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=1)
    return parser.parse_args()


def read_manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def selected_cases(manifest, groups=None, frames=None):
    groups = set(groups or ())
    frames = set(frames or ())
    cases = []
    for case in manifest["frames"]:
        if groups and case["group"] not in groups:
            continue
        if frames and case["frame"] not in frames:
            continue
        cases.append(case)
    return sorted(cases, key=lambda case: case["frame"])


def read_frames(video_path, frame_indices):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    frames = {}
    try:
        for frame_index in sorted(frame_indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Could not read source frame {frame_index} from {video_path}"
                )
            frames[frame_index] = frame
    finally:
        capture.release()
    return frames


def compact_reference_subset(references, labels):
    by_label = {}
    for label, image in references:
        by_label.setdefault(label, []).append(image)
    selected = []
    for label in labels:
        images = by_label.get(label)
        if not images:
            raise KeyError(f"Reference label {label!r} not present")
        selected.append((label, images.pop(0)))
    return selected


def current_focused_grid(reference_images, current_image):
    """Give CURRENT most of the grid area while retaining three references."""
    if len(reference_images) != 3:
        raise ValueError("The current-focused grid expects exactly three references")
    height, width = current_image.shape[:2]
    label_height = max(22, round(height * 0.08))
    reference_width = width // 3
    reference_height = max(1, round(height * reference_width / width))
    grid_height = label_height + reference_height + label_height + height
    grid = np.full((grid_height, width, 3), 245, dtype=np.uint8)

    for index, (label, image) in enumerate(reference_images):
        x1 = index * reference_width
        x2 = width if index == 2 else (index + 1) * reference_width
        tile_width = x2 - x1
        resized = cv2.resize(
            image,
            (tile_width, reference_height),
            interpolation=cv2.INTER_AREA,
        )
        grid[label_height:label_height + reference_height, x1:x2] = resized
        cv2.putText(
            grid,
            label,
            (x1 + 3, max(16, label_height - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (15, 15, 15),
            1,
            cv2.LINE_AA,
        )

    current_y = label_height + reference_height + label_height
    grid[current_y:current_y + height, :width] = current_image
    cv2.putText(
        grid,
        "CURRENT",
        (4, current_y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (15, 15, 15),
        2,
        cv2.LINE_AA,
    )
    return grid


def build_current_focused_message(
    system_prompt,
    reference_images,
    current_image,
    current_frame_index,
    grid_meaning,
):
    grid = current_focused_grid(reference_images, current_image)
    ok, encoded = cv2.imencode(
        ".jpg",
        grid,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    if not ok:
        raise RuntimeError("Could not encode current-focused grid")
    image_base64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            f"Image label: COMPARISON_GRID. Meaning: {grid_meaning}. "
                            f"Source frame: {current_frame_index}. Classify only CURRENT."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                ]
            ),
        ],
        "grid_shape": list(grid.shape[:2]),
        "jpeg_bytes": len(encoded),
    }


def build_single_image_message(system_prompt, current_image, current_frame_index):
    ok, encoded = cv2.imencode(
        ".jpg",
        current_image,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    if not ok:
        raise RuntimeError("Could not encode current-only image")
    image_base64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            f"Source frame: {current_frame_index}. "
                            "Classify only this current image."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                ]
            ),
        ],
        "grid_shape": list(current_image.shape[:2]),
        "jpeg_bytes": len(encoded),
    }


def production_requests(
    frame_index,
    current_crop,
    tight_crop,
    prompts,
    banks,
):
    specifications = {
        "inserted_a": (
            prompts["inserted"],
            banks["inserted_a"],
            current_crop,
            "inserted positives, not-inserted hard negatives, and CURRENT",
        ),
        "inserted_b": (
            prompts["inserted"],
            banks["inserted_b"],
            current_crop,
            "alternate inserted positives, hard negatives, and CURRENT",
        ),
        "drop_a": (
            prompts["drop"],
            banks["drop"],
            current_crop,
            "dropped positive and controlled, inserted, stand, human hard negatives plus CURRENT",
        ),
        "rotation_a": (
            prompts["rotation"],
            banks["rotation_a"],
            tight_crop,
            "tight normal hard negatives, rotation positives, and CURRENT",
        ),
        "rotation_b": (
            prompts["rotation"],
            banks["rotation_b"],
            tight_crop,
            "alternate tight normal hard negatives, rotation positives, and CURRENT",
        ),
    }
    return {
        name: build_compact_grid_message(
            prompt,
            references,
            image,
            current_frame_index=frame_index,
            grid_meaning=meaning,
        )
        for name, (prompt, references, image, meaning) in specifications.items()
    }


def current_focused_requests(
    frame_index,
    current_crop,
    tight_crop,
    prompts,
    banks,
):
    rotation_a = compact_reference_subset(
        banks["rotation_a"],
        ("NORMAL_A", "NORMAL_B", "ROTATED_A"),
    )
    rotation_b = compact_reference_subset(
        banks["rotation_b"],
        ("NORMAL_A", "NORMAL_C", "ROTATED_B"),
    )
    drop_a = compact_reference_subset(
        banks["drop"],
        ("DROPPED", "CONTROLLED", "INSERTED"),
    )
    drop_b = compact_reference_subset(
        banks["drop"],
        ("DROPPED", "STAND_RESET", "HUMAN"),
    )
    specifications = {
        "drop_a": (
            prompts["drop"],
            drop_a,
            current_crop,
            "one dropped positive, two hard negatives, and an enlarged CURRENT",
        ),
        "drop_b": (
            prompts["drop"],
            drop_b,
            current_crop,
            "an independent dropped comparison with stand and human hard negatives",
        ),
        "rotation_a": (
            prompts["rotation"],
            rotation_a,
            tight_crop,
            "normal and rotated references with an enlarged CURRENT",
        ),
        "rotation_b": (
            prompts["rotation"],
            rotation_b,
            tight_crop,
            "an alternate normal/rotated bank with an enlarged CURRENT",
        ),
    }
    return {
        name: build_current_focused_message(
            prompt,
            references,
            image,
            current_frame_index=frame_index,
            grid_meaning=meaning,
        )
        for name, (prompt, references, image, meaning) in specifications.items()
    }


DROP_NO_REFERENCE_PROMPTS = (
    """You monitor a robot plug-insertion task. Decide whether the small white
plug is visibly loose on the tabletop. Return exactly D when the plug itself is
positively visible away from the gripper, its pedestal, and every power-strip
socket. Return N for held, inserted, pedestal/reset, human-handled, occluded, or
otherwise non-dropped states. The large rectangular white pedestal is not the
plug. Non-visibility is never a drop. Return U only when the scene is genuinely
impossible to classify. Output exactly one character: D, N, or U.""",
    """Classify the current robot scene using object identity and contact.
D means the small two-pin white plug is clearly separate from the gripper and
lying or standing loose on the table. N means no confirmed loose plug: it may
be grasped, near the gripper, seated in a socket, on its starting stand, hidden,
or handled by a person. Do not confuse the large white stand with the plug.
Require visible positive evidence for D. Reply with only D, N, or U.""",
)


def drop_no_reference_requests(
    frame_index,
    current_crop,
    tight_crop,
    prompts,
    banks,
):
    del tight_crop, prompts, banks
    return {
        "drop_a": build_single_image_message(
            DROP_NO_REFERENCE_PROMPTS[0],
            current_crop,
            frame_index,
        ),
        "drop_b": build_single_image_message(
            DROP_NO_REFERENCE_PROMPTS[1],
            current_crop,
            frame_index,
        ),
    }


def rotation_current_focused_requests(*args, **kwargs):
    requests = current_focused_requests(*args, **kwargs)
    return {
        name: message
        for name, message in requests.items()
        if name.startswith("rotation_")
    }


def safety_primary_requests(*args, **kwargs):
    requests = current_focused_requests(*args, **kwargs)
    return {
        name: message
        for name, message in requests.items()
        if name in {"rotation_a", "rotation_b", "drop_a"}
    }


def production_primary_requests(
    frame_index,
    current_crop,
    tight_crop,
    prompts,
    banks,
):
    requests = safety_primary_requests(
        frame_index,
        current_crop,
        tight_crop,
        prompts,
        banks,
    )
    requests["inserted_a"] = build_compact_grid_message(
        prompts["inserted"],
        banks["inserted_a"],
        current_crop,
        current_frame_index=frame_index,
        grid_meaning=(
            "inserted positives, not-inserted hard negatives, and CURRENT"
        ),
    )
    return requests


def drop_reference_focused_requests(*args, **kwargs):
    requests = current_focused_requests(*args, **kwargs)
    return {
        name: message
        for name, message in requests.items()
        if name.startswith("drop_")
    }


def parse_prediction(name, content):
    if name.startswith("rotation_"):
        return compact_rotation_vote(content)
    if name.startswith("drop_"):
        return compact_enum_choice(content, {"D", "N", "U"})
    if name.startswith("inserted_"):
        return compact_enum_choice(content, {"I", "N", "U"})
    return "U"


def output_path_for(arguments):
    if arguments.output:
        return arguments.output
    token_label = "1120" if arguments.port == 8000 else "560"
    filename = (
        f"{arguments.source}_tokens-{token_label}_{arguments.variant}"
        f"_r{arguments.repeats}.jsonl"
    )
    return RESULT_DIRECTORY / filename


def completed_keys(output_path):
    keys = set()
    if not output_path.exists():
        return keys
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("error") is None:
            keys.add((record["frame"], record["repeat"]))
    return keys


def invoke_request(model, message):
    response, latency = invoke_with_latency(model, message["messages"])
    return response.content, latency


def main():
    arguments = parse_arguments()
    if arguments.repeats < 1:
        raise ValueError("--repeats must be positive")

    manifest = read_manifest()
    cases = selected_cases(manifest, arguments.groups, arguments.frames)
    if not cases:
        raise ValueError("No benchmark frames matched the selection")

    width, height = (int(value) for value in arguments.source.split("x"))
    resolution = resolution_config_for(width, height)
    video_path = SOURCE_PATHS[arguments.source]
    frames = read_frames(video_path, [case["frame"] for case in cases])
    banks = load_terminal_reference_banks(resolution)
    prompts = {
        "inserted": INSERTED_BINARY_PROMPT_PATH.read_text(encoding="utf-8"),
        "drop": DROP_BINARY_PROMPT_PATH.read_text(encoding="utf-8"),
        "rotation": ROTATION_VERIFY_PROMPT_PATH.read_text(encoding="utf-8"),
    }

    base_model = VLLMTimingChatOpenAI(
        model=MODEL_ID,
        base_url=f"http://127.0.0.1:{arguments.port}/v1",
        api_key="EMPTY",
        temperature=0,
        timeout=arguments.request_timeout,
        max_retries=arguments.max_retries,
    )
    models = {
        "inserted_a": base_model.bind(max_tokens=8),
        "inserted_b": base_model.bind(max_tokens=8),
        "drop_a": base_model.bind(max_tokens=8),
        "drop_b": base_model.bind(max_tokens=8),
        "rotation_a": base_model.bind(max_tokens=24),
        "rotation_b": base_model.bind(max_tokens=24),
    }

    output_path = output_path_for(arguments)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(output_path)
    request_builder = {
        "production": production_requests,
        "current_focused": current_focused_requests,
        "safety_primary": safety_primary_requests,
        "production_primary": production_primary_requests,
        "rotation_current_focused": rotation_current_focused_requests,
        "drop_reference_focused": drop_reference_focused_requests,
        "drop_no_reference": drop_no_reference_requests,
    }[arguments.variant]
    total = len(cases) * arguments.repeats
    finished = len(done)
    wall_times = []

    with (
        output_path.open("a", encoding="utf-8", buffering=1) as output_file,
        ThreadPoolExecutor(max_workers=5) as pool,
    ):
        for repeat in range(arguments.repeats):
            for case in cases:
                key = (case["frame"], repeat)
                if key in done:
                    continue
                frame = frames[case["frame"]]
                current_crop = crop_frame(
                    frame,
                    resolution.crop_box,
                    expected_source_size=resolution.source_size,
                )
                tight_crop = tight_current_crop(frame, resolution)
                messages = request_builder(
                    case["frame"],
                    current_crop,
                    tight_crop,
                    prompts,
                    banks,
                )

                started_at = time.perf_counter()
                futures = {
                    name: pool.submit(invoke_request, models[name], message)
                    for name, message in messages.items()
                }
                raw = {}
                predictions = {}
                latencies = {}
                error = None
                try:
                    for name, future in futures.items():
                        content, latency = future.result()
                        raw[name] = content
                        predictions[name] = parse_prediction(name, content)
                        latencies[name] = latency
                except Exception as exception:  # preserve partial diagnostics
                    error = f"{type(exception).__name__}: {exception}"
                    for future in futures.values():
                        future.cancel()
                round_ms = (time.perf_counter() - started_at) * 1000
                wall_times.append(round_ms)
                record = {
                    "schema_version": 1,
                    "source": arguments.source,
                    "video": str(video_path.relative_to(REPOSITORY_ROOT)),
                    "port": arguments.port,
                    "visual_token_label": 1120 if arguments.port == 8000 else 560,
                    "variant": arguments.variant,
                    "frame": case["frame"],
                    "repeat": repeat,
                    "group": case["group"],
                    "expected": {
                        "rotation": case["rotation"],
                        "drop": case["drop"],
                        "inserted": case["inserted"],
                    },
                    "predictions": predictions,
                    "raw": raw,
                    "latencies": latencies,
                    "parallel_round_ms": round(round_ms, 3),
                    "request_payloads": {
                        name: {
                            key: value
                            for key, value in message.items()
                            if key != "messages"
                        }
                        for name, message in messages.items()
                    },
                    "error": error,
                }
                print(json.dumps(record, separators=(",", ":")), file=output_file)
                finished += 1
                median_ms = statistics.median(wall_times)
                print(
                    f"[{finished}/{total}] {arguments.source} port={arguments.port} "
                    f"variant={arguments.variant} frame={case['frame']} repeat={repeat} "
                    f"pred={predictions} round={round_ms:.1f}ms "
                    f"running_median={median_ms:.1f}ms"
                    + (f" ERROR={error}" if error else ""),
                    flush=True,
                )

    print(f"Saved {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
