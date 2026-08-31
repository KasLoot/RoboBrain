"""Compare prompt/demo variants on selected frames; never writes production files."""

import argparse
import base64
import json
from pathlib import Path

import cv2
from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIRECTORY = ROOT / "src" / "robobrain" / "prompt"
TEST_DIRECTORY = Path(__file__).resolve().parent
VIDEO_PATH = ROOT / "video" / "failure_part.mp4"
MODEL_NAME = "/workspace/models/gemma-4-26B-A4B-it"
MODEL_BASE_URL = "http://127.0.0.1:8000/v1"
CROP = (350, 50, 650, 300)

FRAME_EXPECTATIONS = {
    493: ("FAIL", "ROTATED_OR_MISALIGNED"),
    609: ("FAIL", "ROTATED_OR_MISALIGNED"),
    897: ("FAIL", "ROTATED_OR_MISALIGNED"),
    1000: ("FAIL", "DROP_OR_CONTROL_LOSS"),
    1316: ("ON_GOING", "NONE"),
    1377: ("ON_GOING", "NONE"),
}

ANNOTATED_DEMOS = (
    ("SUCCESS_DEMO", "SUCCESS_DEMO_WITH_ARROW.png"),
    ("FAIL_DEMO_1", "FAIL_DEMO_1.jpg"),
    ("FAIL_DEMO_2", "FAIL_DEMO_2_WITH_ARROW.png"),
    ("FAIL_DEMO_3", "FAIL_DEMO_3_WITH_ARROW.png"),
)
RAW_DEMOS = (
    ("SUCCESS_DEMO", "SUCCESS_DEMO.png"),
    ("FAIL_DEMO_1", "FAIL_DEMO_1.jpg"),
    ("FAIL_DEMO_2", "FAIL_DEMO_2.jpg"),
    ("FAIL_DEMO_3", "FAIL_DEMO_3.jpg"),
)
SPECIALIST_DEMOS = (
    ("FAIL_DEMO_1", "FAIL_DEMO_1.jpg"),
    ("FAIL_DEMO_2", "FAIL_DEMO_2.jpg"),
    ("FAIL_DEMO_3", "FAIL_DEMO_3.jpg"),
    ("SAFE_OCCLUSION_DEMO", "video-frame:1316:crop"),
    ("CONTROLLED_NORMAL_DEMO", "video-frame:1377:crop"),
)
SPECIALIST_DEMOS_SAFE_FIRST = (
    ("SAFE_OCCLUSION_DEMO", "video-frame:1316:crop"),
    ("CONTROLLED_NORMAL_DEMO", "video-frame:1377:crop"),
    ("FAIL_DEMO_1", "FAIL_DEMO_1.jpg"),
    ("FAIL_DEMO_2", "FAIL_DEMO_2.jpg"),
    ("FAIL_DEMO_3", "FAIL_DEMO_3.jpg"),
)

LABEL_DESCRIPTIONS = {
    "SUCCESS_DEMO": "Reference example of a fully inserted plug.",
    "FAIL_DEMO_1": "Reference example of the small plug visibly dropped away from the gripper.",
    "FAIL_DEMO_2": "Definitive reference example of the plug rotated in the gripper.",
    "FAIL_DEMO_3": "Definitive reference example of the plug rotated in the gripper.",
    "SAFE_OCCLUSION_DEMO": (
        "Safe reference: the large white block is the empty stand, while the "
        "small plug is held by or occluded by the gripper. This is not a drop."
    ),
    "CONTROLLED_NORMAL_DEMO": (
        "Safe reference: the small plug is controlled by the gripper. The "
        "large white stand elsewhere in view is not a dropped plug."
    ),
    "FULL_WORKSPACE_VIEW": "Current full workspace view.",
    "ZOOM_IN_VIEW": "Current close view of the manipulation.",
}


def parse_json(content):
    text = str(content).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def encode_image(image):
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Could not encode image")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


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


def read_demo(filename):
    if filename.startswith("video-frame:"):
        _, frame_text, view = filename.split(":")
        frame = read_frame(int(frame_text))
        if view == "full":
            return frame
        if view == "crop":
            x1, y1, x2, y2 = CROP
            return frame[y1:y2, x1:x2]
        raise ValueError(f"Unsupported demo view: {view}")

    image = cv2.imread(str(PROMPT_DIRECTORY / filename))
    if image is None:
        raise RuntimeError(f"Could not read demo {filename}")
    return image


def build_human_message(frame_index, demos):
    frame = read_frame(frame_index)
    x1, y1, x2, y2 = CROP
    labeled_images = []
    for label, filename in demos:
        image = read_demo(filename)
        labeled_images.append((label, image))
    labeled_images.extend(
        (
            ("FULL_WORKSPACE_VIEW", frame),
            ("ZOOM_IN_VIEW", frame[y1:y2, x1:x2]),
        )
    )

    content = []
    for label, image in labeled_images:
        content.extend(
            (
                {
                    "type": "text",
                    "text": (
                        f"Image label: {label}. "
                        f"Meaning: {LABEL_DESCRIPTIONS[label]}"
                        + (
                            f" Source frame: {frame_index}."
                            if label in {"FULL_WORKSPACE_VIEW", "ZOOM_IN_VIEW"}
                            else ""
                        )
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": encode_image(image)},
                },
            )
        )

    content.extend(
        (
            {
                "type": "text",
                "text": (
                    "TASK\nPick up the plug and insert it into the socket."
                ),
            },
            {
                "type": "text",
                "text": (
                    "PREVIOUS_CONTEXT\n"
                    + json.dumps(
                        {
                            "available": True,
                            "source_frame": frame_index - 58,
                            "grasp_history": "GRASP_CONFIRMED",
                            "previous_status": "ON_GOING",
                            "previous_failure_type": "NONE",
                        },
                        separators=(",", ":"),
                    )
                    + "\nPrevious grasp history is authoritative until the "
                    "small plug is visibly returned to the stand."
                ),
            },
        )
    )
    return HumanMessage(content=content)


def variants():
    production_prompt = (
        PROMPT_DIRECTORY / "brain_prompt.md"
    ).read_text(encoding="utf-8")
    candidate_prompt = (
        TEST_DIRECTORY / "prompts" / "candidate_v1.md"
    ).read_text(encoding="utf-8")
    specialist_prompt = (
        TEST_DIRECTORY / "prompts" / "candidate_v2_specialist.md"
    ).read_text(encoding="utf-8")
    return {
        "baseline_annotated": (production_prompt, ANNOTATED_DEMOS),
        "baseline_raw": (production_prompt, RAW_DEMOS),
        "candidate_annotated": (candidate_prompt, ANNOTATED_DEMOS),
        "candidate_raw": (candidate_prompt, RAW_DEMOS),
        "specialist_raw_safe": (specialist_prompt, SPECIALIST_DEMOS),
        "specialist_safe_first": (
            specialist_prompt,
            SPECIALIST_DEMOS_SAFE_FIRST,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(variants()),
        default=list(variants()),
    )
    args = parser.parse_args()

    model = ChatOpenAI(
        model=MODEL_NAME,
        base_url=MODEL_BASE_URL,
        api_key="EMPTY",
        temperature=0,
        timeout=120,
        max_retries=0,
    )
    artifact_directory = TEST_DIRECTORY / "artifacts"
    artifact_directory.mkdir(parents=True, exist_ok=True)
    output_path = artifact_directory / "prompt_matrix.jsonl"

    rows = []
    for variant_name in args.variants:
        system_prompt, demos = variants()[variant_name]
        for frame_index, expected in FRAME_EXPECTATIONS.items():
            print(
                f"running variant={variant_name} frame={frame_index}",
                flush=True,
            )
            response = model.invoke(
                [
                    SystemMessage(content=system_prompt),
                    build_human_message(frame_index, demos),
                ]
            )
            parsed = parse_json(response.content)
            actual = (
                str(parsed.get("status", "")) if parsed else "PARSE_ERROR",
                str(parsed.get("failure_type", "")) if parsed else "PARSE_ERROR",
            )
            row = {
                "variant": variant_name,
                "frame": frame_index,
                "expected_status": expected[0],
                "expected_failure_type": expected[1],
                "actual_status": actual[0],
                "actual_failure_type": actual[1],
                "pass": actual == expected,
                "parsed": parsed,
                "raw": response.content,
            }
            rows.append(row)
            print(
                f"result variant={variant_name} frame={frame_index} "
                f"actual={actual[0]}/{actual[1]} pass={row['pass']}",
                flush=True,
            )

    with output_path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nsummary")
    for variant_name in args.variants:
        variant_rows = [row for row in rows if row["variant"] == variant_name]
        passed = sum(row["pass"] for row in variant_rows)
        print(f"{variant_name}: {passed}/{len(variant_rows)}")
    print(f"results={output_path}")


if __name__ == "__main__":
    main()
