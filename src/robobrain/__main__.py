import os
import json
import shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from langchain_openai import ChatOpenAI


# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_ENDPOINT"] = "https://eu.api.smith.langchain.com"
# os.environ["LANGSMITH_PROJECT"] = "RoboBrain"


from typing import Literal
from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState
from langchain.messages import SystemMessage



import cv2
import threading
import time
import base64


VIDEO_WINDOW_NAME = "RoboBrain video stream"
VIDEO_OUTPUT_PATH = "output.mp4"
LOG_OUTPUT_PATH = "output.log"
RUN_ARCHIVE_ROOT = Path("runs")
WINDOW_SIZE = 1
CROP_TOP_LEFT = (350, 50)
CROP_BOTTOM_RIGHT = (650, 300)
PROMPT_DIRECTORY = Path(__file__).resolve().parent / "prompt"
BRAIN_PROMPT_PATH = PROMPT_DIRECTORY / "brain_prompt.md"
TASK_PROMPT_PATH = PROMPT_DIRECTORY / "TASK.md"
ROTATION_SPECIALIST_PROMPT_PATH = (
    PROMPT_DIRECTORY / "rotation_specialist.md"
)
DROP_SPECIALIST_PROMPT_PATH = PROMPT_DIRECTORY / "drop_specialist.md"
ROTATION_CONFIRMATION_COUNT = 2
DEMONSTRATION_IMAGE_PATHS = (
    ("SUCCESS_DEMO", PROMPT_DIRECTORY / "SUCCESS_DEMO_WITH_ARROW.png"),
    ("FAIL_DEMO_1", PROMPT_DIRECTORY / "FAIL_DEMO_1.jpg"),
    (
        "FAIL_DEMO_2",
        PROMPT_DIRECTORY / "FAIL_DEMO_2_WITH_ARROW.png",
    ),
    (
        "FAIL_DEMO_3",
        PROMPT_DIRECTORY / "FAIL_DEMO_3_WITH_ARROW.png",
    ),
)
ROTATION_REFERENCE_IMAGE_PATHS = (
    (
        "CONTROLLED_NORMAL_DEMO",
        PROMPT_DIRECTORY / "CONTROLLED_NORMAL_DEMO.jpg",
    ),
    ("FAIL_DEMO_2", PROMPT_DIRECTORY / "FAIL_DEMO_2.jpg"),
    ("FAIL_DEMO_3", PROMPT_DIRECTORY / "FAIL_DEMO_3.jpg"),
)
DROP_REFERENCE_IMAGE_PATHS = (
    ("FAIL_DEMO_1", PROMPT_DIRECTORY / "FAIL_DEMO_1.jpg"),
    (
        "SAFE_OCCLUSION_DEMO",
        PROMPT_DIRECTORY / "SAFE_OCCLUSION_DEMO.jpg",
    ),
    (
        "CONTROLLED_NORMAL_DEMO",
        PROMPT_DIRECTORY / "CONTROLLED_NORMAL_DEMO.jpg",
    ),
)
IMAGE_LABEL_DESCRIPTIONS = {
    "SUCCESS_DEMO": "Reference example of a fully inserted plug",
    "FAIL_DEMO_1": "Reference example of a dropped or uncontrolled plug",
    "FAIL_DEMO_2": (
        "Rotation failure: yellow is gripper direction and green is pin direction"
    ),
    "FAIL_DEMO_3": (
        "Rotation failure: yellow is gripper direction and green is pin direction"
    ),
    "FULL_WORKSPACE_VIEW": "Current full workspace view",
    "ZOOM_IN_VIEW": "Current close view of the manipulation",
}
RELATION_IMAGE_LABEL_DESCRIPTIONS = {
    "CONTROLLED_NORMAL_DEMO": (
        "Normal held plug; the empty stand elsewhere is not a dropped plug"
    ),
    "FAIL_DEMO_1": "Small plug visibly separate on the table; DROP",
    "FAIL_DEMO_2": "Definitive rotated-in-gripper example; ROTATED",
    "FAIL_DEMO_3": "Definitive rotated-in-gripper example; ROTATED",
    "SAFE_OCCLUSION_DEMO": (
        "Large white block is the empty stand, not the plug; NO_DROP"
    ),
    "FULL_WORKSPACE_VIEW": "Current full workspace view",
    "ZOOM_IN_VIEW": "Current close view of the manipulation",
}
DETECTION_CONTEXT_FIELDS = (
    "white_stand_state",
    "gripper_control",
    "insertion_state",
    "failure_type",
    "failure_description",
)
VALID_DETECTION_STATUSES = {"ON_GOING", "COMPLETE", "FAIL"}
VALID_GRASP_HISTORY = {"NOT_YET_GRASPED", "GRASP_CONFIRMED"}
VALID_ROTATION_MATCHES = {"ROTATED", "NOT_ROTATED", "UNCERTAIN"}
VALID_DROP_MATCHES = {"DROP", "NO_DROP", "UNCERTAIN"}


class SimulatedCamera:
    def __init__(self, video_path, loop=True, window_size=1):
        if not isinstance(window_size, int) or isinstance(window_size, bool):
            raise TypeError("window_size must be an integer")
        if window_size < 0:
            raise ValueError("window_size must be non-negative")

        self.cap = cv2.VideoCapture(video_path)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_interval = 1.0 / self.fps

        self.loop = loop
        self.window_size = window_size

        # window_size is the number of frames before t, so the inclusive
        # interval [t - window_size, ..., t] contains window_size + 1 frames.
        self.frame_buffer = deque(maxlen=window_size)
        self.lock = threading.Lock()

        self.running = False
        self.thread = None

    def start(self):
        self.running = True

        self.thread = threading.Thread(
            target=self._run,
            daemon=True
        )

        self.thread.start()
        time.sleep(1)  # warm up the camera

    def _run(self):
        next_frame_time = time.monotonic()
        frame_index = 0

        while self.running:
            ok, frame = self.cap.read()

            if not ok:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_index = 0
                    with self.lock:
                        self.frame_buffer.clear()
                    continue
                break

            with self.lock:
                self.frame_buffer.append((frame_index, frame))

            frame_index += 1

            next_frame_time += self.frame_interval
            delay = next_frame_time - time.monotonic()

            if delay > 0:
                time.sleep(delay)

    def get(self):
        with self.lock:
            if not self.frame_buffer:
                return None, None

            current_frame_index = self.frame_buffer[-1][0]
            x1, y1 = CROP_TOP_LEFT
            x2, y2 = CROP_BOTTOM_RIGHT
            frame_chunk = []

            for _, frame in self.frame_buffer:
                full_frame = frame.copy()
                cropped_frame = full_frame[y1:y2, x1:x2].copy()
                frame_chunk.append([full_frame, cropped_frame])

            return frame_chunk, current_frame_index

    def stop(self):
        self.running = False

        if self.thread:
            self.thread.join()

        self.cap.release()


class LatestInferenceResult:
    """Thread-safe storage for the most recent model response."""

    def __init__(self):
        self.lock = threading.Lock()
        self.content = None
        self.frame_index = None
        self.version = 0

    def update(self, frame_index, content):
        with self.lock:
            self.content = content
            self.frame_index = frame_index
            self.version += 1

    def snapshot(self):
        with self.lock:
            return self.content, self.frame_index, self.version


def parse_response_content(content):
    """Parse JSON responses, including JSON wrapped in a Markdown code fence."""
    if isinstance(content, dict):
        return content

    if not isinstance(content, str):
        return None

    content = content.strip()
    object_start = content.find("{")
    object_end = content.rfind("}")

    if object_start == -1 or object_end <= object_start:
        return None

    try:
        return json.loads(content[object_start:object_end + 1])
    except json.JSONDecodeError:
        return None


def detection_context_from_response(
    frame_index,
    content,
    previous_detection_context=None,
):
    """Keep a compact, validated context for the next inference call."""
    parsed_content = parse_response_content(content)
    if not isinstance(parsed_content, dict):
        return None

    status = str(parsed_content.get("status", "")).strip().upper()
    if status not in VALID_DETECTION_STATUSES:
        return None

    context = {"source_frame": frame_index}
    for field_name in DETECTION_CONTEXT_FIELDS:
        field_value = parsed_content.get(field_name)
        if isinstance(field_value, (str, int, float, bool)):
            context[field_name] = field_value

    reported_grasp_history = str(
        parsed_content.get("grasp_history", "")
    ).strip().upper()
    previous_grasp_history = ""
    if isinstance(previous_detection_context, dict):
        previous_grasp_history = str(
            previous_detection_context.get("grasp_history", "")
        ).strip().upper()

    gripper_control = str(
        parsed_content.get("gripper_control", "")
    ).strip().upper()
    white_stand_state = str(
        parsed_content.get("white_stand_state", "")
    ).strip().upper()
    plug_returned_to_stand = (
        white_stand_state == "HOLDING_PLUG"
        and gripper_control == "NOT_YET_GRASPED"
    )

    if gripper_control == "CONTROLLED":
        grasp_history = "GRASP_CONFIRMED"
    elif plug_returned_to_stand:
        grasp_history = "NOT_YET_GRASPED"
    elif previous_grasp_history == "GRASP_CONFIRMED":
        grasp_history = "GRASP_CONFIRMED"
    elif reported_grasp_history in VALID_GRASP_HISTORY:
        grasp_history = reported_grasp_history
    else:
        grasp_history = None

    if grasp_history is not None:
        context["grasp_history"] = grasp_history
    context["status"] = status
    return context


def response_choice(content, field_name, valid_choices, default="UNCERTAIN"):
    """Read one normalized enum value from a structured model response."""
    parsed_content = parse_response_content(content)
    if not isinstance(parsed_content, dict):
        return default

    choice = str(parsed_content.get(field_name, "")).strip().upper()
    return choice if choice in valid_choices else default


class AttemptFailureTracker:
    """Fuse relation observations into stable, attempt-level failure state."""

    def __init__(self, rotation_confirmation_count=2):
        if not isinstance(rotation_confirmation_count, int):
            raise TypeError("rotation_confirmation_count must be an integer")
        if rotation_confirmation_count < 1:
            raise ValueError(
                "rotation_confirmation_count must be at least one"
            )
        self.rotation_confirmation_count = rotation_confirmation_count
        self.reset()

    def reset(self):
        self.grasp_confirmed = False
        self.rotation_streak = 0
        self.latched_failure = None

    def update(self, scene_content, rotation_content, drop_content):
        scene = parse_response_content(scene_content)
        if not isinstance(scene, dict):
            scene = {}

        stand_state = str(
            scene.get("white_stand_state", "")
        ).strip().upper()
        gripper_control = str(
            scene.get("gripper_control", "")
        ).strip().upper()
        reported_grasp_history = str(
            scene.get("grasp_history", "")
        ).strip().upper()
        reset_visible = (
            stand_state == "HOLDING_PLUG"
            and gripper_control == "NOT_YET_GRASPED"
        )

        if reset_visible:
            self.reset()
        elif (
            gripper_control == "CONTROLLED"
            or reported_grasp_history == "GRASP_CONFIRMED"
        ):
            self.grasp_confirmed = True

        rotation_match = response_choice(
            rotation_content,
            "rotation_match",
            VALID_ROTATION_MATCHES,
        )
        drop_match = response_choice(
            drop_content,
            "drop_match",
            VALID_DROP_MATCHES,
        )

        if (
            not reset_visible
            and self.grasp_confirmed
            and rotation_match == "ROTATED"
        ):
            self.rotation_streak += 1
        else:
            self.rotation_streak = 0

        if self.latched_failure is None:
            if self.rotation_streak >= self.rotation_confirmation_count:
                self.latched_failure = "ROTATED_OR_MISALIGNED"
            elif (
                not reset_visible
                and self.grasp_confirmed
                and drop_match == "DROP"
            ):
                self.latched_failure = "DROP_OR_CONTROL_LOSS"

        return {
            "reset_visible": reset_visible,
            "grasp_confirmed": self.grasp_confirmed,
            "rotation_match": rotation_match,
            "rotation_streak": self.rotation_streak,
            "drop_match": drop_match,
            "failure_type": self.latched_failure,
        }


def fuse_detection_response(scene_content, failure_state):
    """Apply validated temporal failure state to the general scene result."""
    scene = parse_response_content(scene_content)
    if not isinstance(scene, dict):
        scene = {}

    fused = {
        "white_stand_state": "UNCERTAIN",
        "gripper_control": "UNCERTAIN",
        "pin_direction_vs_gripper": "UNCERTAIN",
        "pin_alignment_with_socket": "UNCERTAIN",
        "insertion_state": "UNCERTAIN",
        "grasp_history": "NOT_YET_GRASPED",
        "status": "ON_GOING",
        "failure_type": "NONE",
        "failure_description": "none",
    }
    for field_name in fused:
        field_value = scene.get(field_name)
        if isinstance(field_value, str) and field_value.strip():
            fused[field_name] = field_value.strip()

    reset_visible = bool(failure_state.get("reset_visible"))
    grasp_confirmed = bool(failure_state.get("grasp_confirmed"))
    rotation_match = failure_state.get("rotation_match", "UNCERTAIN")
    drop_match = failure_state.get("drop_match", "UNCERTAIN")
    failure_type = failure_state.get("failure_type")

    fused["grasp_history"] = (
        "GRASP_CONFIRMED" if grasp_confirmed else "NOT_YET_GRASPED"
    )

    if (
        str(fused["gripper_control"]).upper() == "LOST"
        and drop_match != "DROP"
    ):
        fused["gripper_control"] = "UNCERTAIN"

    if rotation_match == "ROTATED":
        fused["pin_direction_vs_gripper"] = (
            "DIFFERENT"
            if failure_type == "ROTATED_OR_MISALIGNED"
            else "UNCERTAIN"
        )
    elif (
        failure_type is None
        and str(fused["pin_direction_vs_gripper"]).upper() == "DIFFERENT"
    ):
        fused["pin_direction_vs_gripper"] = "UNCERTAIN"

    if failure_type == "ROTATED_OR_MISALIGNED":
        fused["status"] = "FAIL"
        fused["failure_type"] = failure_type
        fused["failure_description"] = (
            "the plug is rotated relative to the gripper"
            if rotation_match == "ROTATED"
            else "plug rotation was confirmed earlier in the current attempt"
        )
    elif failure_type == "DROP_OR_CONTROL_LOSS":
        fused["status"] = "FAIL"
        fused["failure_type"] = failure_type
        fused["failure_description"] = (
            "the plug is visibly separate from the gripper"
            if drop_match == "DROP"
            else "plug control loss was confirmed earlier in the current attempt"
        )
    else:
        scene_status = str(scene.get("status", "")).strip().upper()
        fused["status"] = (
            "COMPLETE"
            if scene_status == "COMPLETE" and not reset_visible
            else "ON_GOING"
        )
        fused["failure_type"] = "NONE"
        fused["failure_description"] = "none"

    return fused


def wrap_text(text, max_width, font, font_scale, thickness):
    """Wrap text to a pixel width supported by cv2.putText."""
    words = str(text).split()
    if not words:
        return [""]

    lines = []
    current_line = words[0]

    for word in words[1:]:
        candidate = f"{current_line} {word}"
        candidate_width = cv2.getTextSize(
            candidate,
            font,
            font_scale,
            thickness,
        )[0][0]

        if candidate_width <= max_width:
            current_line = candidate
        else:
            lines.append(current_line)
            current_line = word

    lines.append(current_line)
    return lines


def response_overlay_lines(content, response_frame_index):
    if content is None:
        return ["Model output: waiting..."]

    parsed_content = parse_response_content(content)
    header = f"Model output (frame {response_frame_index})"

    if parsed_content is not None:
        status = parsed_content.get("status", "unknown")
        progress_stage = parsed_content.get("progress_stage")
        failure_type = parsed_content.get("failure_type")
        failure_description = parsed_content.get(
            "failure_description",
            "none",
        )
        lines = [header]
        if progress_stage is not None:
            lines.append(f"Progress: {progress_stage}")
        lines.append(f"Status: {status}")
        if failure_type is not None:
            lines.append(f"Failure type: {failure_type}")
        lines.append(f"Failure: {failure_description}")
        return lines

    content_lines = [
        line.strip()
        for line in str(content).splitlines()
        if line.strip()
    ]
    return [header, *content_lines]


def draw_translucent_panel(frame, top_left, bottom_right, opacity=0.65):
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 0), -1)
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)


def annotate_frame(frame, frame_index, response_content, response_frame_index):
    """Add the frame number and latest model response to a video frame."""
    annotated_frame = frame.copy()
    frame_height, frame_width = annotated_frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(1.0, frame_width / 1600))
    thickness = max(1, round(font_scale * 2))
    margin = max(10, round(min(frame_width, frame_height) * 0.015))
    padding = max(8, round(margin * 0.75))
    line_gap = max(6, round(margin * 0.5))

    frame_label = f"Frame {frame_index}"
    (label_width, label_height), label_baseline = cv2.getTextSize(
        frame_label,
        font,
        font_scale,
        thickness,
    )
    label_left = frame_width - margin - label_width - (2 * padding)
    label_top = margin
    label_right = frame_width - margin
    label_bottom = label_top + label_height + label_baseline + (2 * padding)
    draw_translucent_panel(
        annotated_frame,
        (label_left, label_top),
        (label_right, label_bottom),
    )
    cv2.putText(
        annotated_frame,
        frame_label,
        (label_left + padding, label_top + padding + label_height),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    response_lines = response_overlay_lines(
        response_content,
        response_frame_index,
    )
    maximum_text_width = max(1, round(frame_width * 0.65) - (2 * padding))
    wrapped_lines = []
    for line in response_lines:
        wrapped_lines.extend(
            wrap_text(
                line,
                maximum_text_width,
                font,
                font_scale,
                thickness,
            )
        )

    maximum_lines = 8
    if len(wrapped_lines) > maximum_lines:
        wrapped_lines = wrapped_lines[:maximum_lines]
        wrapped_lines[-1] = f"{wrapped_lines[-1]}..."

    text_sizes = [
        cv2.getTextSize(line, font, font_scale, thickness)[0]
        for line in wrapped_lines
    ]
    line_height = max(height for _, height in text_sizes)
    panel_width = min(
        round(frame_width * 0.65),
        max(width for width, _ in text_sizes) + (2 * padding),
    )
    panel_height = (
        (len(wrapped_lines) * line_height)
        + ((len(wrapped_lines) - 1) * line_gap)
        + (2 * padding)
    )
    panel_left = margin
    panel_top = frame_height - margin - panel_height
    panel_right = panel_left + panel_width
    panel_bottom = frame_height - margin
    draw_translucent_panel(
        annotated_frame,
        (panel_left, panel_top),
        (panel_right, panel_bottom),
    )

    text_y = panel_top + padding + line_height
    for line_number, line in enumerate(wrapped_lines):
        text_color = (255, 220, 170) if line_number == 0 else (255, 255, 255)
        cv2.putText(
            annotated_frame,
            line,
            (panel_left + padding, text_y),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )
        text_y += line_height + line_gap

    return annotated_frame




def load_demonstration_images():
    demonstration_images = []

    for demonstration_name, image_path in DEMONSTRATION_IMAGE_PATHS:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(
                f"Could not load {demonstration_name} from {image_path}"
            )
        demonstration_images.append(image)

    return demonstration_images


def load_reference_images(reference_image_paths):
    """Load a labeled image set used by one visual relation specialist."""
    reference_images = []
    for image_label, image_path in reference_image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(
                f"Could not load {image_label} from {image_path}"
            )
        reference_images.append((image_label, image))
    return reference_images


def build_message(
    system_prompt,
    query,
    demonstration_images,
    camera_images,
    current_frame_index=None,
    previous_detection_context=None,
):
    if len(demonstration_images) != len(DEMONSTRATION_IMAGE_PATHS):
        raise ValueError(
            "Expected exactly four demonstration images in prompt order"
        )
    if len(camera_images) != 1 or len(camera_images[0]) != 2:
        raise ValueError(
            "Expected one timestamp containing a full and cropped frame"
        )

    labeled_images = [
        *[
            (demonstration_name, image)
            for (demonstration_name, _), image in zip(
                DEMONSTRATION_IMAGE_PATHS,
                demonstration_images,
            )
        ],
        ("FULL_WORKSPACE_VIEW", camera_images[0][0]),
        ("ZOOM_IN_VIEW", camera_images[0][1]),
    ]
    human_content = []
    for image_label, image in labeled_images:
        frame_text = (
            f" Source frame: {current_frame_index}."
            if image_label in {"FULL_WORKSPACE_VIEW", "ZOOM_IN_VIEW"}
            and current_frame_index is not None
            else ""
        )
        human_content.append(
            {
                "type": "text",
                "text": (
                    f"Image label: {image_label}. "
                    f"Meaning: {IMAGE_LABEL_DESCRIPTIONS[image_label]}."
                    f"{frame_text}"
                ),
            }
        )

        ok, encoded_image = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError("Could not encode model input image")

        image_base64 = base64.b64encode(
            encoded_image.tobytes()
        ).decode("ascii")
        human_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}",
                },
            }
        )

    previous_context_payload = {
        "available": previous_detection_context is not None,
    }
    if previous_detection_context is not None:
        previous_context_payload.update(
            {
                "source_frame": previous_detection_context.get(
                    "source_frame"
                ),
                "grasp_history": previous_detection_context.get(
                    "grasp_history"
                ),
                "previous_status": previous_detection_context.get("status"),
                "previous_failure_type": previous_detection_context.get(
                    "failure_type"
                ),
            }
        )

    human_content.extend(
        [
            {
                "type": "text",
                "text": f"TASK\n{query}",
            },
            {
                "type": "text",
                "text": (
                    "PREVIOUS_CONTEXT\n"
                    f"{json.dumps(previous_context_payload, separators=(',', ':'))}\n"
                    "Previous grasp_history is authoritative until the plug "
                    "is visibly back on the white stand. Previous status and "
                    "failure type are secondary."
                ),
            },
        ]
    )

    return {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content),
        ]
    }


def build_relation_specialist_message(
    system_prompt,
    reference_images,
    camera_images,
    current_frame_index=None,
):
    """Build a focused visual-comparison request for one relation."""
    if len(camera_images) != 1 or len(camera_images[0]) != 2:
        raise ValueError(
            "Expected one timestamp containing a full and cropped frame"
        )

    labeled_images = [
        *reference_images,
        ("FULL_WORKSPACE_VIEW", camera_images[0][0]),
        ("ZOOM_IN_VIEW", camera_images[0][1]),
    ]
    human_content = []
    for image_label, image in labeled_images:
        frame_text = (
            f" Source frame: {current_frame_index}."
            if image_label in {"FULL_WORKSPACE_VIEW", "ZOOM_IN_VIEW"}
            and current_frame_index is not None
            else ""
        )
        human_content.append(
            {
                "type": "text",
                "text": (
                    f"Image label: {image_label}. "
                    "Meaning: "
                    f"{RELATION_IMAGE_LABEL_DESCRIPTIONS[image_label]}."
                    f"{frame_text}"
                ),
            }
        )

        ok, encoded_image = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError("Could not encode specialist input image")
        image_base64 = base64.b64encode(
            encoded_image.tobytes()
        ).decode("ascii")
        human_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}",
                },
            }
        )

    return {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content),
        ]
    }


def run_inference(
    vlm,
    system_prompt,
    rotation_specialist_prompt,
    drop_specialist_prompt,
    task,
    demonstration_images,
    rotation_reference_images,
    drop_reference_images,
    camera,
    stop_event,
    latest_inference_result,
):
    """Send snapshots to the model without blocking the video canvas."""
    last_frame_index = None
    previous_detection_context = None
    failure_tracker = AttemptFailureTracker(
        rotation_confirmation_count=ROTATION_CONFIRMATION_COUNT
    )

    with (
        open(
            LOG_OUTPUT_PATH,
            "w",
            encoding="utf-8",
            buffering=1,
        ) as output_log,
        ThreadPoolExecutor(max_workers=3) as model_request_pool,
    ):
        while not stop_event.is_set():
            frame_chunk, current_frame_index = camera.get()

            if frame_chunk is None or current_frame_index == last_frame_index:
                stop_event.wait(0.01)
                continue

            if (
                last_frame_index is not None
                and current_frame_index < last_frame_index
            ):
                previous_detection_context = None
                failure_tracker.reset()

            last_frame_index = current_frame_index
            scene_message = build_message(
                system_prompt,
                task,
                demonstration_images,
                frame_chunk,
                current_frame_index=current_frame_index,
                previous_detection_context=previous_detection_context,
            )
            rotation_message = build_relation_specialist_message(
                rotation_specialist_prompt,
                rotation_reference_images,
                frame_chunk,
                current_frame_index=current_frame_index,
            )
            drop_message = build_relation_specialist_message(
                drop_specialist_prompt,
                drop_reference_images,
                frame_chunk,
                current_frame_index=current_frame_index,
            )

            scene_future = model_request_pool.submit(
                vlm.invoke,
                scene_message["messages"],
            )
            rotation_future = model_request_pool.submit(
                vlm.invoke,
                rotation_message["messages"],
            )
            drop_future = model_request_pool.submit(
                vlm.invoke,
                drop_message["messages"],
            )
            scene_response = scene_future.result()
            rotation_response = rotation_future.result()
            drop_response = drop_future.result()

            failure_state = failure_tracker.update(
                scene_response.content,
                rotation_response.content,
                drop_response.content,
            )
            fused_response = fuse_detection_response(
                scene_response.content,
                failure_state,
            )
            response_content = json.dumps(fused_response, indent=2)
            previous_detection_context = detection_context_from_response(
                current_frame_index,
                response_content,
                previous_detection_context=previous_detection_context,
            )
            latest_inference_result.update(
                current_frame_index,
                response_content,
            )

            print(f"Frame [{current_frame_index}] ", response_content)
            print(
                f"Frame [{current_frame_index}] ",
                response_content,
                file=output_log,
                flush=True,
            )


def archive_run_outputs(run_id):
    """Copy the latest root outputs into a unique per-run directory."""
    run_directory = RUN_ARCHIVE_ROOT / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    archived_paths = {}
    for output_path in (Path(LOG_OUTPUT_PATH), Path(VIDEO_OUTPUT_PATH)):
        if output_path.is_file():
            destination = run_directory / output_path.name
            shutil.copy2(output_path, destination)
            archived_paths[output_path.name] = destination
    return archived_paths


def show_video_canvas(camera, stop_event, latest_inference_result):
    """Display and record annotated frames until the canvas is closed."""
    cv2.namedWindow(VIDEO_WINDOW_NAME, cv2.WINDOW_NORMAL)
    print(
        "Video canvas opened. Press q or Escape to stop and save "
        f"{VIDEO_OUTPUT_PATH}."
    )

    video_writer = None
    written_frame_index = None
    rendered_frame_key = None
    annotated_frame = None
    written_frame_count = 0

    try:
        while not stop_event.is_set():
            frame_chunk, current_frame_index = camera.get()
            (
                response_content,
                response_frame_index,
                response_version,
            ) = latest_inference_result.snapshot()

            if frame_chunk is not None:
                current_frame_or_group = frame_chunk[-1]
                current_frame = (
                    current_frame_or_group[0]
                    if isinstance(current_frame_or_group, (list, tuple))
                    else current_frame_or_group
                )
                rendered_key = (current_frame_index, response_version)
                if rendered_key != rendered_frame_key:
                    annotated_frame = annotate_frame(
                        current_frame,
                        current_frame_index,
                        response_content,
                        response_frame_index,
                    )
                    rendered_frame_key = rendered_key

                if video_writer is None:
                    frame_height, frame_width = annotated_frame.shape[:2]
                    codec = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        VIDEO_OUTPUT_PATH,
                        codec,
                        camera.fps,
                        (frame_width, frame_height),
                    )
                    if not video_writer.isOpened():
                        raise RuntimeError(
                            f"Could not create {VIDEO_OUTPUT_PATH}"
                        )

                cv2.imshow(VIDEO_WINDOW_NAME, annotated_frame)

                if current_frame_index != written_frame_index:
                    video_writer.write(annotated_frame)
                    written_frame_index = current_frame_index
                    written_frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            try:
                window_visible = cv2.getWindowProperty(
                    VIDEO_WINDOW_NAME,
                    cv2.WND_PROP_VISIBLE,
                )
            except cv2.error:
                break

            if window_visible < 1:
                break
    finally:
        stop_event.set()
        if video_writer is not None:
            video_writer.release()
            print(
                f"Saved {written_frame_count} annotated frames to "
                f"{VIDEO_OUTPUT_PATH}."
            )

def main():
    if WINDOW_SIZE != 1:
        raise ValueError(
            "brain_prompt.md currently requires WINDOW_SIZE to be 1"
        )

    # vlm = ChatOllama(
    #     model="gemma4:31b-cloud",
    #     base_url="https://ollama.com",
    #     temperature=0,
    # )

    vlm = ChatOpenAI(
            model="/workspace/models/gemma-4-26B-A4B-it",
            base_url="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            temperature=0,
        )

    system_prompt = BRAIN_PROMPT_PATH.read_text(encoding="utf-8")
    rotation_specialist_prompt = (
        ROTATION_SPECIALIST_PROMPT_PATH.read_text(encoding="utf-8")
    )
    drop_specialist_prompt = DROP_SPECIALIST_PROMPT_PATH.read_text(
        encoding="utf-8"
    )
    TASK = TASK_PROMPT_PATH.read_text(encoding="utf-8")
    demonstration_images = load_demonstration_images()
    rotation_reference_images = load_reference_images(
        ROTATION_REFERENCE_IMAGE_PATHS
    )
    drop_reference_images = load_reference_images(
        DROP_REFERENCE_IMAGE_PATHS
    )

    # camera = SimulatedCamera("video/insert_plug-aug-28-105trail.MOV", loop=False)
    camera = SimulatedCamera(
        "video/failure_part.mp4",
        loop=False,
        window_size=WINDOW_SIZE,
    )
    camera.start()
    stop_event = threading.Event()
    latest_inference_result = LatestInferenceResult()
    inference_thread = threading.Thread(
        target=run_inference,
        args=(
            vlm,
            system_prompt,
            rotation_specialist_prompt,
            drop_specialist_prompt,
            TASK,
            demonstration_images,
            rotation_reference_images,
            drop_reference_images,
            camera,
            stop_event,
            latest_inference_result,
        ),
        daemon=True,
        name="robobrain-inference",
    )
    inference_thread.start()
    run_id = datetime.now(timezone.utc).strftime(
        "run_%Y%m%dT%H%M%S_%fZ"
    )

    try:
        show_video_canvas(camera, stop_event, latest_inference_result)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        camera.stop()
        inference_thread.join()
        cv2.destroyAllWindows()
        archived_paths = archive_run_outputs(run_id)
        if archived_paths:
            print(
                "Archived run outputs: "
                + ", ".join(str(path) for path in archived_paths.values())
            )
