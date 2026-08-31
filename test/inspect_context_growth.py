"""Show whether production request payloads grow with source frame index."""

import sys
from pathlib import Path

import cv2


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
)


VIDEO_PATH = ROOT / "video" / "insert_plug-no_force-aug-28-1x.MOV"
FRAME_INDICES = (29, 4941, 8513, 12110)


def read_camera_images(capture, frame_index):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index}")
    x1, y1 = CROP_TOP_LEFT
    x2, y2 = CROP_BOTTOM_RIGHT
    return [[frame, frame[y1:y2, x1:x2]]]


def stats(message):
    messages = message["messages"]
    blocks = messages[1].content
    text_characters = sum(
        len(block["text"]) for block in blocks if block["type"] == "text"
    )
    image_characters = sum(
        len(block["image_url"]["url"])
        for block in blocks
        if block["type"] == "image_url"
    )
    return {
        "messages": len(messages),
        "text_blocks": sum(block["type"] == "text" for block in blocks),
        "image_blocks": sum(
            block["type"] == "image_url" for block in blocks
        ),
        "text_characters": text_characters,
        "image_url_characters": image_characters,
    }


def main():
    capture = cv2.VideoCapture(str(VIDEO_PATH))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {VIDEO_PATH}")

    scene_prompt = BRAIN_PROMPT_PATH.read_text(encoding="utf-8")
    task = TASK_PROMPT_PATH.read_text(encoding="utf-8")
    rotation_prompt = ROTATION_SPECIALIST_PROMPT_PATH.read_text(
        encoding="utf-8"
    )
    drop_prompt = DROP_SPECIALIST_PROMPT_PATH.read_text(encoding="utf-8")
    scene_demos = load_demonstration_images()
    rotation_references = load_reference_images(
        ROTATION_REFERENCE_IMAGE_PATHS
    )
    drop_references = load_reference_images(DROP_REFERENCE_IMAGE_PATHS)
    previous_context = {
        "source_frame": 1,
        "grasp_history": "GRASP_CONFIRMED",
        "status": "ON_GOING",
        "failure_type": "NONE",
    }

    try:
        for frame_index in FRAME_INDICES:
            camera_images = read_camera_images(capture, frame_index)
            requests = {
                "scene": build_message(
                    scene_prompt,
                    task,
                    scene_demos,
                    camera_images,
                    current_frame_index=frame_index,
                    previous_detection_context=previous_context,
                ),
                "rotation": build_relation_specialist_message(
                    rotation_prompt,
                    rotation_references,
                    camera_images,
                    current_frame_index=frame_index,
                ),
                "drop": build_relation_specialist_message(
                    drop_prompt,
                    drop_references,
                    camera_images,
                    current_frame_index=frame_index,
                ),
            }
            print(f"frame={frame_index}")
            for name, request in requests.items():
                print(f"  {name}: {stats(request)}")
    finally:
        capture.release()


if __name__ == "__main__":
    main()
