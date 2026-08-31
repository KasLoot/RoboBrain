"""Live smoke test using the production prompts, references, and builder."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from langchain_openai import ChatOpenAI


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robobrain.__main__ import (
    BRAIN_PROMPT_PATH,
    CROP_BOTTOM_RIGHT,
    CROP_TOP_LEFT,
    DROP_REFERENCE_IMAGE_PATHS,
    DROP_SPECIALIST_PROMPT_PATH,
    ROTATION_REFERENCE_IMAGE_PATHS,
    ROTATION_SPECIALIST_PROMPT_PATH,
    TASK_PROMPT_PATH,
    build_message,
    build_relation_specialist_message,
    load_demonstration_images,
    load_reference_images,
    parse_response_content,
)


VIDEO_PATH = ROOT / "video" / "failure_part.mp4"
EXPECTATIONS = {
    493: ("ROTATED", "NO_DROP"),
    609: ("ROTATED", "NO_DROP"),
    897: ("ROTATED", "NO_DROP"),
    1000: ("UNCERTAIN", "DROP"),
    1316: ("NOT_ROTATED", "NO_DROP"),
    1377: ("NOT_ROTATED", "NO_DROP"),
}


def read_camera_images(frame_index):
    capture = cv2.VideoCapture(str(VIDEO_PATH))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {VIDEO_PATH}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index}")

    x1, y1 = CROP_TOP_LEFT
    x2, y2 = CROP_BOTTOM_RIGHT
    return [[frame, frame[y1:y2, x1:x2]]]


def main():
    model = ChatOpenAI(
        model="/workspace/models/gemma-4-26B-A4B-it",
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        temperature=0,
        timeout=120,
        max_retries=0,
    )
    rotation_prompt = ROTATION_SPECIALIST_PROMPT_PATH.read_text(
        encoding="utf-8"
    )
    drop_prompt = DROP_SPECIALIST_PROMPT_PATH.read_text(encoding="utf-8")
    rotation_references = load_reference_images(
        ROTATION_REFERENCE_IMAGE_PATHS
    )
    drop_references = load_reference_images(DROP_REFERENCE_IMAGE_PATHS)
    scene_prompt = BRAIN_PROMPT_PATH.read_text(encoding="utf-8")
    task_prompt = TASK_PROMPT_PATH.read_text(encoding="utf-8")
    scene_demos = load_demonstration_images()

    results = []
    with ThreadPoolExecutor(max_workers=3) as request_pool:
        for frame_index, expected in EXPECTATIONS.items():
            camera_images = read_camera_images(frame_index)
            scene_message = build_message(
                scene_prompt,
                task_prompt,
                scene_demos,
                camera_images,
                current_frame_index=frame_index,
            )
            rotation_message = build_relation_specialist_message(
                rotation_prompt,
                rotation_references,
                camera_images,
                current_frame_index=frame_index,
            )
            drop_message = build_relation_specialist_message(
                drop_prompt,
                drop_references,
                camera_images,
                current_frame_index=frame_index,
            )
            scene_future = request_pool.submit(
                model.invoke,
                scene_message["messages"],
            )
            rotation_future = request_pool.submit(
                model.invoke,
                rotation_message["messages"],
            )
            drop_future = request_pool.submit(
                model.invoke,
                drop_message["messages"],
            )
            scene_result = parse_response_content(scene_future.result().content)
            rotation = parse_response_content(
                rotation_future.result().content
            )
            drop = parse_response_content(drop_future.result().content)
            actual = (
                rotation.get("rotation_match") if rotation else None,
                drop.get("drop_match") if drop else None,
            )
            row = {
                "frame": frame_index,
                "expected": expected,
                "actual": actual,
                "pass": actual == expected and scene_result is not None,
                "scene": scene_result,
                "rotation": rotation,
                "drop": drop,
            }
            results.append(row)
            print(
                f"frame={frame_index} rotation={actual[0]} "
                f"drop={actual[1]} pass={row['pass']}",
                flush=True,
            )

    output_path = ROOT / "test" / "artifacts" / "production_smoke.json"
    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print(f"summary={sum(row['pass'] for row in results)}/{len(results)}")
    print(f"results={output_path}")


if __name__ == "__main__":
    main()
