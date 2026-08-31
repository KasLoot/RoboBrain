"""Compare raw requests transport with the OpenAI SDK for one image prompt."""

import os

os.environ["LANGSMITH_TRACING"] = "false"

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import requests


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robobrain.__main__ import (
    BRAIN_PROMPT_PATH,
    CROP_BOTTOM_RIGHT,
    CROP_TOP_LEFT,
    ROTATION_REFERENCE_IMAGE_PATHS,
    ROTATION_SPECIALIST_PROMPT_PATH,
    TASK_PROMPT_PATH,
    build_message,
    build_relation_specialist_message,
    load_demonstration_images,
    load_reference_images,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-width", type=int, default=0)
    parser.add_argument("--tokenize-only", action="store_true")
    parser.add_argument(
        "--kind", choices=("scene", "rotation"), default="rotation"
    )
    args = parser.parse_args()
    capture = cv2.VideoCapture(
        str(ROOT / "video" / "insert_plug-no_force-aug-28-1x.MOV")
    )
    capture.set(cv2.CAP_PROP_POS_FRAMES, 29)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError("Could not read frame")

    if args.full_width:
        scale = args.full_width / frame.shape[1]
        frame = cv2.resize(
            frame,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    x1, y1 = CROP_TOP_LEFT
    x2, y2 = CROP_BOTTOM_RIGHT
    camera_images = [[frame, frame[y1:y2, x1:x2]]]
    if args.kind == "scene":
        request = build_message(
            BRAIN_PROMPT_PATH.read_text(encoding="utf-8"),
            TASK_PROMPT_PATH.read_text(encoding="utf-8"),
            load_demonstration_images(),
            camera_images,
            current_frame_index=29,
        )
    else:
        request = build_relation_specialist_message(
            ROTATION_SPECIALIST_PROMPT_PATH.read_text(encoding="utf-8"),
            load_reference_images(ROTATION_REFERENCE_IMAGE_PATHS),
            camera_images,
            current_frame_index=29,
        )
    system_message, human_message = request["messages"]
    payload = {
        "model": "/workspace/models/gemma-4-26B-A4B-it",
        "messages": [
            {"role": "system", "content": system_message.content},
            {"role": "user", "content": human_message.content},
        ],
        "temperature": 0,
        "max_tokens": 256,
    }
    serialized = json.dumps(payload)
    print(f"request_bytes={len(serialized.encode('utf-8'))}", flush=True)
    started = time.perf_counter()
    endpoint = (
        "http://127.0.0.1:8000/tokenize"
        if args.tokenize_only
        else "http://127.0.0.1:8000/v1/chat/completions"
    )
    response = requests.post(
        endpoint,
        data=serialized,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    print(
        f"elapsed_s={time.perf_counter() - started:.3f} "
        f"status={response.status_code} bytes={len(response.content)}",
        flush=True,
    )
    response.raise_for_status()
    if args.tokenize_only:
        result = response.json()
        print(
            f"token_count={result.get('count')} "
            f"max_model_len={result.get('max_model_len')}",
            flush=True,
        )
    else:
        print(response.json()["choices"][0]["message"]["content"], flush=True)


if __name__ == "__main__":
    main()
