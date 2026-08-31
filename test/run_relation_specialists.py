"""Evaluate independent rotation/drop visual specialists and fuse their output."""

import argparse
import base64
import json
from pathlib import Path

import cv2
from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


ROOT = Path(__file__).resolve().parents[1]
TEST_DIRECTORY = Path(__file__).resolve().parent
PROMPT_DIRECTORY = ROOT / "src" / "robobrain" / "prompt"
VIDEO_PATH = ROOT / "video" / "failure_part.mp4"
CROP = (350, 50, 650, 300)
SMOKE_EXPECTED = {
    493: ("FAIL", "ROTATED_OR_MISALIGNED"),
    609: ("FAIL", "ROTATED_OR_MISALIGNED"),
    897: ("FAIL", "ROTATED_OR_MISALIGNED"),
    1000: ("FAIL", "DROP_OR_CONTROL_LOSS"),
    1316: ("ON_GOING", "NONE"),
    1377: ("ON_GOING", "NONE"),
}
LOG_REPLAY_FRAMES = (
    29,
    73,
    134,
    197,
    258,
    318,
    377,
    435,
    493,
    551,
    609,
    667,
    725,
    783,
    840,
    897,
    955,
    1012,
    1074,
    1135,
    1197,
    1258,
    1316,
    1377,
    1435,
    1493,
    1551,
    1608,
    1666,
    1724,
    1771,
)
LOG_REPLAY_EXPECTED = {
    frame_index: (
        ("FAIL", "ROTATED_OR_MISALIGNED")
        if 465 <= frame_index <= 970
        else ("FAIL", "DROP_OR_CONTROL_LOSS")
        if frame_index in {1012, 1724, 1771}
        else ("ON_GOING", "NONE")
    )
    for frame_index in LOG_REPLAY_FRAMES
}


def read_frame(frame_index):
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
    return frame


def crop(frame):
    x1, y1, x2, y2 = CROP
    return frame[y1:y2, x1:x2]


def encode(image):
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Could not encode image")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def parse(content):
    text = str(content)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def image_pair(label, meaning, image):
    return [
        {
            "type": "text",
            "text": f"Image label: {label}. Meaning: {meaning}",
        },
        {"type": "image_url", "image_url": {"url": encode(image)}},
    ]


def current_image_content(frame_index):
    frame = read_frame(frame_index)
    return [
        *image_pair(
            "FULL_WORKSPACE_VIEW",
            f"current full scene at source frame {frame_index}",
            frame,
        ),
        *image_pair(
            "ZOOM_IN_VIEW",
            f"current close scene at source frame {frame_index}",
            crop(frame),
        ),
    ]


def rotation_content(frame_index):
    normal = crop(read_frame(1377))
    rotated_2 = cv2.imread(str(PROMPT_DIRECTORY / "FAIL_DEMO_2.jpg"))
    rotated_3 = cv2.imread(str(PROMPT_DIRECTORY / "FAIL_DEMO_3.jpg"))
    return [
        *image_pair(
            "CONTROLLED_NORMAL_DEMO",
            "normal held plug; NOT_ROTATED",
            normal,
        ),
        *image_pair(
            "FAIL_DEMO_2",
            "definitive rotated-in-gripper example; ROTATED",
            rotated_2,
        ),
        *image_pair(
            "FAIL_DEMO_3",
            "definitive rotated-in-gripper example; ROTATED",
            rotated_3,
        ),
        *current_image_content(frame_index),
    ]


def drop_content(frame_index):
    dropped = cv2.imread(str(PROMPT_DIRECTORY / "FAIL_DEMO_1.jpg"))
    safe = crop(read_frame(1316))
    controlled = crop(read_frame(1377))
    return [
        *image_pair(
            "FAIL_DEMO_1",
            "small plug visible separately on table; DROP",
            dropped,
        ),
        *image_pair(
            "SAFE_OCCLUSION_DEMO",
            "large white block is empty stand, not plug; NO_DROP",
            safe,
        ),
        *image_pair(
            "CONTROLLED_NORMAL_DEMO",
            "plug controlled and empty stand elsewhere; NO_DROP",
            controlled,
        ),
        *current_image_content(frame_index),
    ]


def fuse(rotation, drop):
    if rotation.get("rotation_match") == "ROTATED":
        return "FAIL", "ROTATED_OR_MISALIGNED"
    if drop.get("drop_match") == "DROP":
        return "FAIL", "DROP_OR_CONTROL_LOSS"
    return "ON_GOING", "NONE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=("smoke", "log-replay"),
        default="smoke",
    )
    args = parser.parse_args()
    expected_results = (
        SMOKE_EXPECTED
        if args.suite == "smoke"
        else LOG_REPLAY_EXPECTED
    )
    model = ChatOpenAI(
        model="/workspace/models/gemma-4-26B-A4B-it",
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        temperature=0,
        timeout=120,
        max_retries=0,
    )
    rotation_prompt = (
        TEST_DIRECTORY / "prompts" / "rotation_specialist.md"
    ).read_text(encoding="utf-8")
    drop_prompt = (
        TEST_DIRECTORY / "prompts" / "drop_specialist.md"
    ).read_text(encoding="utf-8")

    rows = []
    for frame_index, expected in expected_results.items():
        print(f"running frame={frame_index} rotation", flush=True)
        rotation_response = model.invoke(
            [
                SystemMessage(content=rotation_prompt),
                HumanMessage(content=rotation_content(frame_index)),
            ]
        )
        rotation = parse(rotation_response.content)
        print(
            f"rotation frame={frame_index} "
            f"match={rotation.get('rotation_match')}",
            flush=True,
        )
        print(f"running frame={frame_index} drop", flush=True)
        drop_response = model.invoke(
            [
                SystemMessage(content=drop_prompt),
                HumanMessage(content=drop_content(frame_index)),
            ]
        )
        drop = parse(drop_response.content)
        actual = fuse(rotation, drop)
        row = {
            "frame": frame_index,
            "expected": expected,
            "rotation": rotation,
            "drop": drop,
            "actual": actual,
            "pass": actual == expected,
        }
        rows.append(row)
        print(
            f"fused frame={frame_index} actual={actual[0]}/{actual[1]} "
            f"pass={row['pass']}",
            flush=True,
        )

    artifact_directory = TEST_DIRECTORY / "artifacts"
    artifact_directory.mkdir(parents=True, exist_ok=True)
    output_path = (
        artifact_directory / f"relation_specialists_{args.suite}.json"
    )
    output_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"summary={sum(row['pass'] for row in rows)}/{len(rows)}")
    print(f"results={output_path}")


if __name__ == "__main__":
    main()
