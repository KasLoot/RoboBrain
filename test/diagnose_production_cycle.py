"""Time one production-shaped inference cycle without changing production."""

import os

os.environ["LANGSMITH_TRACING"] = "false"

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


VIDEO_PATH = ROOT / "video" / "insert_plug-no_force-aug-28-1x.MOV"
FRAME_INDEX = 29


def elapsed(start):
    return round(time.perf_counter() - start, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    started = time.perf_counter()
    capture = cv2.VideoCapture(str(VIDEO_PATH))
    capture.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {FRAME_INDEX}")
    x1, y1 = CROP_TOP_LEFT
    x2, y2 = CROP_BOTTOM_RIGHT
    camera_images = [[frame, frame[y1:y2, x1:x2]]]
    print(f"frame_read_s={elapsed(started)}", flush=True)

    scene_prompt = BRAIN_PROMPT_PATH.read_text(encoding="utf-8")
    rotation_prompt = ROTATION_SPECIALIST_PROMPT_PATH.read_text(
        encoding="utf-8"
    )
    drop_prompt = DROP_SPECIALIST_PROMPT_PATH.read_text(encoding="utf-8")
    task = TASK_PROMPT_PATH.read_text(encoding="utf-8")
    scene_demos = load_demonstration_images()
    rotation_references = load_reference_images(
        ROTATION_REFERENCE_IMAGE_PATHS
    )
    drop_references = load_reference_images(DROP_REFERENCE_IMAGE_PATHS)

    scene_message = build_message(
        scene_prompt,
        task,
        scene_demos,
        camera_images,
        current_frame_index=FRAME_INDEX,
    )
    rotation_message = build_relation_specialist_message(
        rotation_prompt,
        rotation_references,
        camera_images,
        current_frame_index=FRAME_INDEX,
    )
    drop_message = build_relation_specialist_message(
        drop_prompt,
        drop_references,
        camera_images,
        current_frame_index=FRAME_INDEX,
    )
    print(f"messages_built_s={elapsed(started)}", flush=True)

    model = ChatOpenAI(
        model="/workspace/models/gemma-4-26B-A4B-it",
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        temperature=0,
        timeout=20,
        max_retries=0,
    )
    requests = {
        "scene": scene_message,
        "rotation": rotation_message,
        "drop": drop_message,
    }
    responses = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(model.invoke, request["messages"]): name
            for name, request in requests.items()
        }
        print(f"requests_submitted_s={elapsed(started)}", flush=True)
        for future in as_completed(futures):
            name = futures[future]
            try:
                responses[name] = future.result()
                print(f"{name}_completed_s={elapsed(started)}", flush=True)
            except Exception as exc:
                print(
                    f"{name}_failed_s={elapsed(started)} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )

    for name, response in responses.items():
        parsed = parse_response_content(response.content)
        print(
            f"{name}_parsed={isinstance(parsed, dict)} "
            f"content_chars={len(str(response.content))}",
            flush=True,
        )
    print(f"total_s={elapsed(started)}", flush=True)


if __name__ == "__main__":
    main()
