import os
import json
import shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from robobrain.latency import VLLMTimingChatOpenAI, invoke_with_latency


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
import numpy as np
import threading
import time
import base64


VIDEO_WINDOW_NAME = "RoboBrain video stream"
VIDEO_INPUT_PATH = Path("video/insert_plug-full_model-aug-28-105trail-1x.MOV")
VIDEO_OUTPUT_PATH = f"output-{VIDEO_INPUT_PATH.stem}.mp4"
LOG_OUTPUT_PATH = f"output-{VIDEO_INPUT_PATH.stem}.log"
LATENCY_OUTPUT_PATH = f"output-{VIDEO_INPUT_PATH.stem}-latency.jsonl"
RUN_ARCHIVE_ROOT = Path("runs")
WINDOW_SIZE = 1
PROMPT_DIRECTORY = Path(__file__).resolve().parent / "prompt"
REFERENCE_FRAME_DIRECTORY = PROMPT_DIRECTORY / "reference_frames"
BRAIN_PROMPT_PATH = PROMPT_DIRECTORY / "brain_prompt.md"
TASK_PROMPT_PATH = PROMPT_DIRECTORY / "TASK.md"
ROTATION_SPECIALIST_PROMPT_PATH = (
    PROMPT_DIRECTORY / "rotation_specialist.md"
)
DROP_SPECIALIST_PROMPT_PATH = PROMPT_DIRECTORY / "drop_specialist.md"
ROTATION_GRID_PROMPT_PATH = (
    PROMPT_DIRECTORY / "rotation_grid_compact.md"
)
STATE_DROP_GRID_PROMPT_PATH = (
    PROMPT_DIRECTORY / "state_drop_grid_compact.md"
)
INSERTED_BINARY_PROMPT_PATH = PROMPT_DIRECTORY / "inserted_binary.md"
DROP_BINARY_PROMPT_PATH = PROMPT_DIRECTORY / "drop_binary.md"
ROTATION_VERIFY_PROMPT_PATH = (
    PROMPT_DIRECTORY / "rotation_hard_negatives.md"
)
# Legacy tracker compatibility; the active terminal manager below uses the
# observation count instead of elapsed source frames.
ROTATION_CONFIRMATION_SECONDS = 0.6
# Production fusion is observation-based: skipped source frames are never
# counted as unseen positive evidence.
TERMINAL_ROTATION_CONFIRMATION_COUNT = 3
ROTATION_TERMINAL_ENABLED = (
    os.environ.get("ROBOBRAIN_ENABLE_ROTATION_TERMINAL", "1") == "1"
)
VALIDATED_PRODUCTION_SOURCE_SIZE = (1920, 1080)
# Visual reset is only a continuous-video simulation fallback. Real deployment
# should create a fresh manager from the external reset-complete handshake.
INITIAL_STATE_CONFIRMATION_COUNT = 1
SCENE_CHECK_INTERVAL_FRAMES = 120
SCENE_RESULT_MAX_AGE_FRAMES = 120
GRID_JPEG_QUALITY = 95
# Legacy build_message() assets. The active compact-grid pipeline does not load
# these arrow-annotated images; load_terminal_reference_banks() uses the
# unannotated references declared below.
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
ROTATION_GRID_REFERENCE_IMAGE_PATHS = (
    (
        "NORMAL",
        PROMPT_DIRECTORY / "CONTROLLED_NORMAL_DEMO.jpg",
    ),
    ("ROTATED_A", PROMPT_DIRECTORY / "FAIL_DEMO_2.jpg"),
    ("ROTATED_B", PROMPT_DIRECTORY / "FAIL_DEMO_3.jpg"),
)
STATE_GRID_RESET_REFERENCE_FRAMES = (
    ("RESET_A", 29),
    ("RESET_B", 1074),
)
INSERTED_BANK_A_POSITIVE_FRAMES = (
    ("INSERTED_A", 4073),
    ("INSERTED_B", 10176),
)
INSERTED_BANK_B_POSITIVE_FRAMES = (
    ("INSERTED_A", 9612),
    ("INSERTED_B", 17125),
)
INSERTED_NEGATIVE_FRAMES = (
    ("NOT_INSERTED_TRANSPORT", 4600),
    ("NOT_INSERTED_APPROACH", 15850),
)
DROP_HUMAN_REFERENCE_FRAME = (("HUMAN", 15950),)
ROTATION_BANK_A_NORMAL_FRAMES = (
    ("NORMAL_B", 7569),
    ("NORMAL_C", 9447),
)
ROTATION_BANK_B_NORMAL_FRAMES = (
    ("NORMAL_B", 7301),
    ("NORMAL_C", 17074),
)
# Frame-number tuples above are retained for offline experiment compatibility.
# Production resolves these filenames inside the detected resolution's
# pre-materialized reference directory and never resizes or seeks at runtime.
STATE_GRID_RESET_REFERENCE_FILES = (
    ("RESET_A", "full_f000029_crop.png"),
    ("RESET_B", "full_f001074_crop.png"),
)
STATE_GRID_STATIC_REFERENCE_FILES = (
    ("CONTROLLED", "full_f004600_crop.png"),
    ("SAFE_OCCLUDED", "full_f009447_crop.png"),
    ("DROPPED", "full_f006040_crop.png"),
)
INSERTED_BANK_A_POSITIVE_FILES = (
    ("INSERTED_A", "full_f004073_crop.png"),
    ("INSERTED_B", "full_f010176_crop.png"),
)
INSERTED_BANK_B_POSITIVE_FILES = (
    ("INSERTED_A", "full_f009612_crop.png"),
    ("INSERTED_B", "full_f017125_crop.png"),
)
INSERTED_NEGATIVE_FILES = (
    ("NOT_INSERTED_TRANSPORT", "full_f004600_crop.png"),
    ("NOT_INSERTED_APPROACH", "full_f015850_crop.png"),
)
STAND_RESET_REFERENCE_FILES = (
    ("STAND_RESET", "full_f000029_crop.png"),
)
DROPPED_REFERENCE_FILES = (
    ("DROPPED", "full_f006040_crop.png"),
)
DROP_REFERENCE_FILES = (
    ("DROPPED", "full_f006040_crop.png"),
    ("CONTROLLED", "full_f004600_crop.png"),
    ("INSERTED", "full_f004073_crop.png"),
    ("STAND_RESET", "full_f000029_crop.png"),
    ("HUMAN", "full_f015950_crop.png"),
)
ROTATION_STATIC_REFERENCE_FILES = (
    ("NORMAL_A", "full_f004600_tight.png"),
    ("ROTATED_A", "rotated_a_tight.png"),
    ("ROTATED_B", "rotated_b_tight.png"),
)
ROTATION_BANK_A_NORMAL_FILES = (
    ("NORMAL_B", "full_f007569_tight.png"),
    ("NORMAL_C", "full_f009447_tight.png"),
)
ROTATION_BANK_B_NORMAL_FILES = (
    ("NORMAL_B", "full_f007301_tight.png"),
    ("NORMAL_C", "full_f017074_tight.png"),
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
VALID_STATE_MATCHES = {"S", "C", "D", "U"}


class UnsupportedResolutionError(ValueError):
    """Raised when no calibrated crop configuration exists for a source."""


@dataclass(frozen=True)
class ResolutionConfig:
    """Native crop calibration and references for one source resolution."""

    source_size: tuple[int, int]
    crop_top_left: tuple[int, int]
    crop_bottom_right: tuple[int, int]
    tight_crop_top_left: tuple[int, int]
    tight_crop_bottom_right: tuple[int, int]
    reference_directory: Path

    def __post_init__(self):
        source_width, source_height = self.source_size
        crop_x1, crop_y1 = self.crop_top_left
        crop_x2, crop_y2 = self.crop_bottom_right
        tight_x1, tight_y1 = self.tight_crop_top_left
        tight_x2, tight_y2 = self.tight_crop_bottom_right

        if not (
            0 <= crop_x1 < crop_x2 <= source_width
            and 0 <= crop_y1 < crop_y2 <= source_height
        ):
            raise ValueError("Configured crop must be inside the source frame")
        if not (
            crop_x1 <= tight_x1 < tight_x2 <= crop_x2
            and crop_y1 <= tight_y1 < tight_y2 <= crop_y2
        ):
            raise ValueError("Configured tight crop must be inside the main crop")

    @property
    def crop_size(self):
        return (
            self.crop_bottom_right[0] - self.crop_top_left[0],
            self.crop_bottom_right[1] - self.crop_top_left[1],
        )

    @property
    def crop_box(self):
        return (*self.crop_top_left, *self.crop_bottom_right)

    @property
    def tight_crop_box(self):
        return (*self.tight_crop_top_left, *self.tight_crop_bottom_right)

    @property
    def tight_crop_size(self):
        return (
            self.tight_crop_bottom_right[0] - self.tight_crop_top_left[0],
            self.tight_crop_bottom_right[1] - self.tight_crop_top_left[1],
        )


# The 1920x1080 boxes map the calibrated 848x480 field of view back onto the
# original video before it was resized by resize_video.sh. Keeping these values
# explicit makes each camera/source calibration independently adjustable.
RESOLUTION_CONFIGS = {
    (848, 480): ResolutionConfig(
        source_size=(848, 480),
        crop_top_left=(350, 50),
        crop_bottom_right=(650, 300),
        tight_crop_top_left=(420, 90),
        tight_crop_bottom_right=(600, 260),
        reference_directory=REFERENCE_FRAME_DIRECTORY / "848x480",
    ),
    (1920, 1080): ResolutionConfig(
        source_size=(1920, 1080),
        crop_top_left=(792, 112),
        crop_bottom_right=(1472, 675),
        tight_crop_top_left=(951, 202),
        tight_crop_bottom_right=(1358, 585),
        reference_directory=REFERENCE_FRAME_DIRECTORY / "1920x1080",
    ),
}


def resolution_config_for(width, height):
    """Return the calibrated configuration or reject the source resolution."""
    resolution = (int(width), int(height))
    try:
        return RESOLUTION_CONFIGS[resolution]
    except KeyError as error:
        supported = ", ".join(
            f"{supported_width}x{supported_height}"
            for supported_width, supported_height in RESOLUTION_CONFIGS
        )
        raise UnsupportedResolutionError(
            f"Unsupported resolution: {resolution[0]}x{resolution[1]}. "
            f"Supported resolutions: {supported}."
        ) from error


def detect_resolution(capture):
    """Detect a capture's dimensions and return its calibrated configuration."""
    width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return resolution_config_for(width, height)


def crop_frame(frame, crop_box, expected_source_size=None):
    """Crop a frame after optionally enforcing its configured source size."""
    if not isinstance(frame, np.ndarray) or frame.ndim != 3:
        raise ValueError("Expected a three-channel video frame")

    frame_height, frame_width = frame.shape[:2]
    if (
        expected_source_size is not None
        and (frame_width, frame_height) != expected_source_size
    ):
        resolution_config_for(frame_width, frame_height)
        raise RuntimeError(
            "Video resolution changed during capture: expected "
            f"{expected_source_size[0]}x{expected_source_size[1]}, got "
            f"{frame_width}x{frame_height}."
        )

    x1, y1, x2, y2 = crop_box
    if not (0 <= x1 < x2 <= frame_width and 0 <= y1 < y2 <= frame_height):
        raise ValueError(
            f"Crop box {(x1, y1, x2, y2)} is outside "
            f"{frame_width}x{frame_height}."
        )
    return frame[y1:y2, x1:x2].copy()


class SimulatedCamera:
    def __init__(self, video_path, loop=True, window_size=1):
        if not isinstance(window_size, int) or isinstance(window_size, bool):
            raise TypeError("window_size must be an integer")
        if window_size < 0:
            raise ValueError("window_size must be non-negative")

        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        try:
            self.resolution_config = detect_resolution(self.cap)
        except Exception:
            self.cap.release()
            raise

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(self.fps) or self.fps <= 0:
            self.cap.release()
            raise RuntimeError(
                f"Could not detect a valid frame rate for {self.video_path}"
            )
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
            frame_chunk = []

            for _, frame in self.frame_buffer:
                full_frame = frame.copy()
                cropped_frame = crop_frame(
                    full_frame,
                    self.resolution_config.crop_box,
                    expected_source_size=self.resolution_config.source_size,
                )
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


def compact_state_choice(content):
    """Parse the compact stand/control/drop specialist response."""
    if isinstance(content, str):
        choice = content.strip().strip("`* .\n\r\t").upper()
        if choice in VALID_STATE_MATCHES:
            return choice

    parsed_content = parse_response_content(content)
    if isinstance(parsed_content, dict):
        choice = str(parsed_content.get("state", "")).strip().upper()
        if choice in VALID_STATE_MATCHES:
            return choice
    return "U"


def compact_rotation_choice(content):
    """Parse the compact rotation-grid response into the production enum."""
    parsed_content = parse_response_content(content)
    if not isinstance(parsed_content, dict):
        return "UNCERTAIN"
    choice = str(parsed_content.get("rotation", "")).strip().upper()
    return {
        "R": "ROTATED",
        "N": "NOT_ROTATED",
        "U": "UNCERTAIN",
    }.get(choice, "UNCERTAIN")


class AttemptFailureTracker:
    """Fuse compact observations into stable, cadence-independent state."""

    def __init__(
        self,
        fps,
        rotation_confirmation_seconds=ROTATION_CONFIRMATION_SECONDS,
    ):
        if (
            isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or fps <= 0
        ):
            raise ValueError("fps must be a positive number")
        if (
            isinstance(rotation_confirmation_seconds, bool)
            or not isinstance(rotation_confirmation_seconds, (int, float))
            or rotation_confirmation_seconds <= 0
        ):
            raise ValueError(
                "rotation_confirmation_seconds must be a positive number"
            )
        self.fps = float(fps)
        self.rotation_confirmation_seconds = float(
            rotation_confirmation_seconds
        )
        self.rotation_confirmation_frames = max(
            1,
            round(self.fps * self.rotation_confirmation_seconds),
        )
        self.reset()

    def reset(self):
        self.grasp_confirmed = False
        self.rotation_started_frame = None
        self.latched_failure = None

    def update(self, frame_index, state_content, rotation_content):
        if not isinstance(frame_index, int) or isinstance(frame_index, bool):
            raise TypeError("frame_index must be an integer")
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative")

        state_match = compact_state_choice(state_content)
        rotation_match = compact_rotation_choice(rotation_content)
        reset_visible = state_match == "S"

        if reset_visible:
            self.reset()
        elif state_match == "C":
            self.grasp_confirmed = True

        if (
            not reset_visible
            and self.grasp_confirmed
            and state_match != "D"
            and rotation_match == "ROTATED"
        ):
            if (
                self.rotation_started_frame is None
                or frame_index < self.rotation_started_frame
            ):
                self.rotation_started_frame = frame_index
        else:
            self.rotation_started_frame = None

        rotation_evidence_frames = 0
        if self.rotation_started_frame is not None:
            rotation_evidence_frames = (
                frame_index - self.rotation_started_frame
            )

        drop_match = (
            "DROP"
            if state_match == "D"
            else "NO_DROP"
            if state_match in {"S", "C"}
            else "UNCERTAIN"
        )

        if self.latched_failure is None:
            if (
                rotation_evidence_frames
                >= self.rotation_confirmation_frames
            ):
                self.latched_failure = "ROTATED_OR_MISALIGNED"
            elif (
                not reset_visible
                and self.grasp_confirmed
                and state_match == "D"
            ):
                self.latched_failure = "DROP_OR_CONTROL_LOSS"

        return {
            "reset_visible": reset_visible,
            "state_match": state_match,
            "grasp_confirmed": self.grasp_confirmed,
            "rotation_match": rotation_match,
            "rotation_evidence_frames": rotation_evidence_frames,
            "rotation_evidence_seconds": round(
                rotation_evidence_frames / self.fps,
                3,
            ),
            "drop_match": drop_match,
            "failure_type": self.latched_failure,
        }


class TerminalAttemptManager:
    """Deterministically fuse terminal specialists for one attempt at a time."""

    def __init__(
        self,
        rotation_confirmation_count=TERMINAL_ROTATION_CONFIRMATION_COUNT,
        initial_confirmation_count=INITIAL_STATE_CONFIRMATION_COUNT,
        rotation_terminal_enabled=ROTATION_TERMINAL_ENABLED,
    ):
        if (
            not isinstance(rotation_confirmation_count, int)
            or isinstance(rotation_confirmation_count, bool)
            or rotation_confirmation_count < 1
        ):
            raise ValueError("rotation_confirmation_count must be positive")
        self.rotation_confirmation_count = rotation_confirmation_count
        self.initial_confirmation_count = initial_confirmation_count
        self.rotation_terminal_enabled = rotation_terminal_enabled
        self.attempt_id = 0
        self.hard_reset()

    def hard_reset(self):
        self.phase = "WAIT_INITIAL"
        self.initial_streak = 0
        self.departure_streak = 0
        self.terminal_status = None
        self.failure_type = None
        self.terminal_frame = None
        self.terminal_event_emitted = False
        self._clear_evidence()

    def _clear_evidence(self):
        self.drop_consensus = False
        self.rotation_consensus_streak = 0
        self.rotation_started_frame = None
        self.inserted_consensus = False

    def observe_initial_state(self, state_choice):
        """Advance WAIT_INITIAL/READY/TERMINAL using the reset specialist."""
        reset_completed = False
        monitoring_started = False
        if self.phase in {"WAIT_INITIAL", "TERMINAL"}:
            if state_choice == "S":
                self.initial_streak += 1
            else:
                self.initial_streak = 0
            if self.initial_streak >= self.initial_confirmation_count:
                self.phase = "READY"
                self.attempt_id += 1
                self.initial_streak = 0
                self.departure_streak = 0
                self.terminal_status = None
                self.failure_type = None
                self.terminal_frame = None
                self.terminal_event_emitted = False
                self._clear_evidence()
                reset_completed = True
        elif self.phase == "READY":
            if state_choice == "S":
                self.departure_streak = 0
            elif state_choice in {"C", "D"}:
                self.departure_streak += 1
            else:
                self.departure_streak = 0
            if self.departure_streak >= self.initial_confirmation_count:
                self.phase = "MONITORING"
                self.departure_streak = 0
                self._clear_evidence()
                monitoring_started = True
        return {
            "reset_completed": reset_completed,
            "monitoring_started": monitoring_started,
        }

    def _emit_terminal(self, frame_index, terminal_status, failure_type=None):
        self.phase = "TERMINAL"
        self.terminal_status = terminal_status
        self.failure_type = failure_type
        self.terminal_frame = frame_index
        self.terminal_event_emitted = True

    def update_safety_specialists(
        self,
        frame_index,
        drop_a,
        drop_b,
        rotation_a,
        rotation_b,
    ):
        """Fuse one primary safety observation and same-frame drop verification."""
        if self.phase != "MONITORING":
            return {"terminal_emitted": False, **self.snapshot()}

        self.drop_consensus = drop_a == "D" and drop_b == "D"
        rotation_positive = rotation_a == "R" and rotation_b == "R"

        if rotation_positive:
            if self.rotation_started_frame is None:
                self.rotation_started_frame = frame_index
            self.rotation_consensus_streak += 1
        else:
            self.rotation_started_frame = None
            self.rotation_consensus_streak = 0

        failure_type = None
        if self.drop_consensus:
            failure_type = "DROP_OR_CONTROL_LOSS"
        elif (
            self.rotation_terminal_enabled
            and self.rotation_consensus_streak
            >= self.rotation_confirmation_count
        ):
            failure_type = "ROTATED_OR_MISALIGNED"

        if failure_type is not None:
            self._emit_terminal(frame_index, "FAIL", failure_type)
            return {"terminal_emitted": True, **self.snapshot(frame_index)}

        return {"terminal_emitted": False, **self.snapshot(frame_index)}

    def update_completion_specialists(self, frame_index, inserted_a, inserted_b):
        """Accept completion only after the current frame's safety decision."""
        if self.phase != "MONITORING":
            return {"terminal_emitted": False, **self.snapshot(frame_index)}

        self.inserted_consensus = inserted_a == "I" and inserted_b == "I"
        if self.inserted_consensus:
            self._emit_terminal(frame_index, "COMPLETE")
            return {"terminal_emitted": True, **self.snapshot(frame_index)}

        return {"terminal_emitted": False, **self.snapshot(frame_index)}

    def update_terminal_specialists(
        self,
        frame_index,
        inserted_a,
        inserted_b,
        drop_a,
        rotation_a,
        rotation_b,
        drop_b=None,
    ):
        """Compatibility wrapper that always evaluates safety before success."""
        safety_update = self.update_safety_specialists(
            frame_index,
            drop_a,
            drop_b,
            rotation_a,
            rotation_b,
        )
        if safety_update["terminal_emitted"]:
            return safety_update
        return self.update_completion_specialists(
            frame_index,
            inserted_a,
            inserted_b,
        )

    def rotation_evidence_frames(self, current_frame=None):
        if self.rotation_started_frame is None or current_frame is None:
            return 0
        return max(0, current_frame - self.rotation_started_frame)

    def snapshot(self, current_frame=None):
        return {
            "attempt_id": self.attempt_id,
            "phase": self.phase,
            "terminal_status": self.terminal_status,
            "failure_type": self.failure_type,
            "terminal_frame": self.terminal_frame,
            "drop_consensus": self.drop_consensus,
            # Kept for log consumers; temporal drop streaks are no longer used.
            "drop_streak": 1 if self.drop_consensus else 0,
            "rotation_consensus_streak": self.rotation_consensus_streak,
            "rotation_confirmation_count": self.rotation_confirmation_count,
            "rotation_evidence_frames": self.rotation_evidence_frames(
                current_frame
            ),
            "rotation_shadow_alert": bool(
                not self.rotation_terminal_enabled
                and self.rotation_consensus_streak
                >= self.rotation_confirmation_count
            ),
            "inserted_consensus": self.inserted_consensus,
        }


def terminal_response(manager_state, observations, initial_state=None):
    """Map attempt-manager state back to the established production schema."""
    phase = manager_state["phase"]
    status = manager_state.get("terminal_status") or "ON_GOING"
    failure_type = manager_state.get("failure_type") or "NONE"
    response = {
        "white_stand_state": "UNCERTAIN",
        "gripper_control": "UNCERTAIN",
        "pin_direction_vs_gripper": "UNCERTAIN",
        "pin_alignment_with_socket": "UNCERTAIN",
        "insertion_state": "UNCERTAIN",
        "grasp_history": "NOT_YET_GRASPED"
        if phase in {"WAIT_INITIAL", "READY"}
        else "GRASP_CONFIRMED",
        "status": status,
        "failure_type": failure_type,
        "failure_description": "none",
    }

    if initial_state == "S" or phase == "READY":
        response.update(
            {
                "white_stand_state": "HOLDING_PLUG",
                "gripper_control": "NOT_YET_GRASPED",
                "pin_direction_vs_gripper": "NOT_VISIBLE",
                "pin_alignment_with_socket": "NOT_AT_SOCKET",
                "insertion_state": "NOT_INSERTED",
                "grasp_history": "NOT_YET_GRASPED",
            }
        )

    if status == "COMPLETE":
        response.update(
            {
                "white_stand_state": "EMPTY",
                "gripper_control": "UNCERTAIN",
                "pin_direction_vs_gripper": "SAME",
                "pin_alignment_with_socket": "ALIGNED",
                "insertion_state": "FULLY_SEATED",
                "grasp_history": "GRASP_CONFIRMED",
                "failure_description": "none",
            }
        )
    elif failure_type == "DROP_OR_CONTROL_LOSS":
        response.update(
            {
                "white_stand_state": "EMPTY",
                "gripper_control": "LOST",
                "pin_direction_vs_gripper": "NOT_VISIBLE",
                "pin_alignment_with_socket": "NOT_AT_SOCKET",
                "insertion_state": "NOT_INSERTED",
                "failure_description": "the plug is visibly loose on the table",
            }
        )
    elif failure_type == "ROTATED_OR_MISALIGNED":
        response.update(
            {
                "white_stand_state": "EMPTY",
                "gripper_control": "CONTROLLED",
                "pin_direction_vs_gripper": "DIFFERENT",
                "insertion_state": "NOT_INSERTED",
                "failure_description": (
                    "the plug pin direction differs from the gripper direction"
                ),
            }
        )
    elif phase == "MONITORING":
        response["gripper_control"] = "CONTROLLED"
        if observations.get("rotation_a") == observations.get("rotation_b") == "N":
            response["pin_direction_vs_gripper"] = "SAME"
    return response


def fuse_detection_response(
    scene_content,
    failure_state,
    completion_latched=False,
):
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
    state_match = failure_state.get("state_match", "U")
    grasp_confirmed = bool(failure_state.get("grasp_confirmed"))
    rotation_match = failure_state.get("rotation_match", "UNCERTAIN")
    drop_match = failure_state.get("drop_match", "UNCERTAIN")
    failure_type = failure_state.get("failure_type")

    fused["grasp_history"] = (
        "GRASP_CONFIRMED" if grasp_confirmed else "NOT_YET_GRASPED"
    )

    if state_match == "S":
        fused.update(
            {
                "white_stand_state": "HOLDING_PLUG",
                "gripper_control": "NOT_YET_GRASPED",
                "pin_direction_vs_gripper": "NOT_VISIBLE",
                "pin_alignment_with_socket": "NOT_AT_SOCKET",
                "insertion_state": "NOT_INSERTED",
            }
        )
    elif state_match == "C":
        fused["white_stand_state"] = "EMPTY"
        fused["gripper_control"] = "CONTROLLED"
    elif state_match == "D":
        fused.update(
            {
                "white_stand_state": "EMPTY",
                "gripper_control": "LOST",
                "pin_direction_vs_gripper": "NOT_VISIBLE",
                "pin_alignment_with_socket": "NOT_AT_SOCKET",
                "insertion_state": "NOT_INSERTED",
            }
        )

    if (
        str(fused["gripper_control"]).upper() == "LOST"
        and drop_match != "DROP"
    ):
        fused["gripper_control"] = "UNCERTAIN"

    if (
        failure_type == "ROTATED_OR_MISALIGNED"
        and state_match == "C"
    ):
        fused["pin_direction_vs_gripper"] = "DIFFERENT"
    elif rotation_match == "ROTATED":
        fused["pin_direction_vs_gripper"] = (
            "DIFFERENT"
            if failure_type == "ROTATED_OR_MISALIGNED"
            else "UNCERTAIN"
        )
    elif rotation_match == "NOT_ROTATED" and state_match == "C":
        fused["pin_direction_vs_gripper"] = "SAME"
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
            if (
                (scene_status == "COMPLETE" or completion_latched)
                and not reset_visible
                and state_match == "C"
                and rotation_match != "ROTATED"
            )
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


def load_reference_images(reference_image_paths, expected_size=None):
    """Load pre-cropped references and enforce their native pixel size."""
    reference_images = []
    for image_label, image_path in reference_image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(
                f"Could not load {image_label} from {image_path}"
            )
        actual_size = image.shape[1], image.shape[0]
        if expected_size is not None and actual_size != expected_size:
            raise RuntimeError(
                f"Reference {image_label} from {image_path} is "
                f"{actual_size[0]}x{actual_size[1]}, expected the pre-cropped "
                f"size {expected_size[0]}x{expected_size[1]}."
            )
        reference_images.append((image_label, image))
    return reference_images


def reference_paths(resolution_config, labeled_filenames):
    """Resolve labeled assets within one resolution-specific directory."""
    return tuple(
        (label, resolution_config.reference_directory / filename)
        for label, filename in labeled_filenames
    )


def load_video_crop_reference_images(
    video_path,
    labeled_frame_indices,
    resolution_config=None,
):
    """Offline experiment helper; production startup uses packaged assets."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open reference video {video_path}")

    reference_images = []
    try:
        if resolution_config is None:
            resolution_config = detect_resolution(capture)
        for image_label, frame_index in labeled_frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Could not read {image_label} from frame {frame_index}"
                )
            reference_images.append(
                (
                    image_label,
                    crop_frame(
                        frame,
                        resolution_config.crop_box,
                        expected_source_size=resolution_config.source_size,
                    ),
                )
            )
    finally:
        capture.release()
    return reference_images


def make_comparison_grid(reference_images, current_image, columns=2):
    """Combine labeled references and current view into one VLM image item."""
    if not isinstance(columns, int) or isinstance(columns, bool):
        raise TypeError("columns must be an integer")
    if columns < 1:
        raise ValueError("columns must be at least one")
    panels = [*reference_images, ("CURRENT", current_image)]
    if not panels:
        raise ValueError("At least one grid panel is required")
    expected_shape = None
    for image_label, image in panels:
        if not isinstance(image_label, str) or not image_label:
            raise ValueError("Every grid panel needs a non-empty label")
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            raise ValueError(f"Invalid image for grid panel {image_label}")
        if expected_shape is None:
            expected_shape = image.shape
        elif image.shape != expected_shape:
            raise ValueError(
                "Comparison-grid panel resolution mismatch: "
                f"expected {expected_shape[1]}x{expected_shape[0]}, got "
                f"{image.shape[1]}x{image.shape[0]} for {image_label}."
            )

    rows = (len(panels) + columns - 1) // columns
    tile_width = expected_shape[1]
    tile_height = expected_shape[0]
    label_height = 34
    grid = np.full(
        (rows * (tile_height + label_height), columns * tile_width, 3),
        245,
        dtype=np.uint8,
    )
    for panel_index, (image_label, image) in enumerate(panels):
        row, column = divmod(panel_index, columns)
        x = column * tile_width
        y = row * (tile_height + label_height)
        grid[
            y + label_height:y + label_height + image.shape[0],
            x:x + image.shape[1],
        ] = image
        cv2.putText(
            grid,
            image_label,
            (x + 6, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (15, 15, 15),
            2 if image_label == "CURRENT" else 1,
            cv2.LINE_AA,
        )
    return grid


def select_reference_images(reference_images, labels):
    """Select one image per label while preserving the requested order."""
    reference_by_label = {}
    for label, image in reference_images:
        if label in reference_by_label:
            raise ValueError(f"Duplicate reference label: {label}")
        reference_by_label[label] = image

    missing_labels = [label for label in labels if label not in reference_by_label]
    if missing_labels:
        raise ValueError(
            "Missing reference labels: " + ", ".join(missing_labels)
        )
    return [(label, reference_by_label[label]) for label in labels]


def make_current_focused_grid(reference_images, current_image):
    """Place three reference thumbnails above one native-size CURRENT panel."""
    if len(reference_images) != 3:
        raise ValueError("Current-focused grids require exactly three references")
    if not isinstance(current_image, np.ndarray) or current_image.ndim != 3:
        raise ValueError("Invalid CURRENT image")

    current_height, current_width = current_image.shape[:2]
    for image_label, image in reference_images:
        if not isinstance(image_label, str) or not image_label:
            raise ValueError("Every grid panel needs a non-empty label")
        if not isinstance(image, np.ndarray) or image.shape != current_image.shape:
            raise ValueError(
                "Current-focused reference resolution mismatch for "
                f"{image_label}: expected {current_width}x{current_height}."
            )

    label_height = max(22, round(current_height * 0.08))
    reference_width = current_width // 3
    reference_height = max(
        1,
        round(current_height * reference_width / current_width),
    )
    current_y = label_height + reference_height + label_height
    grid = np.full(
        (current_y + current_height, current_width, 3),
        245,
        dtype=np.uint8,
    )

    for index, (image_label, image) in enumerate(reference_images):
        x1 = index * reference_width
        x2 = current_width if index == 2 else (index + 1) * reference_width
        resized = cv2.resize(
            image,
            (x2 - x1, reference_height),
            interpolation=cv2.INTER_AREA,
        )
        grid[label_height:label_height + reference_height, x1:x2] = resized
        cv2.putText(
            grid,
            image_label,
            (x1 + 3, max(16, label_height - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (15, 15, 15),
            1,
            cv2.LINE_AA,
        )

    grid[current_y:current_y + current_height, :current_width] = current_image
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


def build_compact_grid_message(
    system_prompt,
    reference_images,
    current_image,
    current_frame_index,
    grid_meaning,
):
    """Build a focused request containing one labeled comparison-grid image."""
    comparison_grid = make_comparison_grid(
        reference_images,
        current_image,
    )
    ok, encoded_image = cv2.imencode(
        ".jpg",
        comparison_grid,
        [cv2.IMWRITE_JPEG_QUALITY, GRID_JPEG_QUALITY],
    )
    if not ok:
        raise ValueError("Could not encode compact comparison grid")
    image_base64 = base64.b64encode(encoded_image.tobytes()).decode("ascii")
    human_content = [
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
                "url": f"data:image/jpeg;base64,{image_base64}",
            },
        },
    ]
    return {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content),
        ]
    }


def build_current_focused_grid_message(
    system_prompt,
    reference_images,
    current_image,
    current_frame_index,
    grid_meaning,
):
    """Build the validated safety request with an enlarged CURRENT panel."""
    comparison_grid = make_current_focused_grid(
        reference_images,
        current_image,
    )
    ok, encoded_image = cv2.imencode(
        ".jpg",
        comparison_grid,
        [cv2.IMWRITE_JPEG_QUALITY, GRID_JPEG_QUALITY],
    )
    if not ok:
        raise ValueError("Could not encode current-focused comparison grid")
    image_base64 = base64.b64encode(encoded_image.tobytes()).decode("ascii")
    return {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Image label: COMPARISON_GRID. Meaning: "
                            f"{grid_meaning}. Source frame: "
                            f"{current_frame_index}. Classify only CURRENT."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                        },
                    },
                ]
            ),
        ]
    }


def tight_current_crop(frame, resolution_config):
    """Return the native tight plug/gripper crop without interpolation."""
    return crop_frame(
        frame,
        resolution_config.tight_crop_box,
        expected_source_size=resolution_config.source_size,
    )


def load_terminal_reference_banks(resolution_config):
    """Load exact-size pre-cropped assets for the detected resolution."""
    standard_size = resolution_config.crop_size
    tight_size = resolution_config.tight_crop_size
    state_references = [
        *load_reference_images(
            reference_paths(
                resolution_config,
                STATE_GRID_RESET_REFERENCE_FILES,
            ),
            expected_size=standard_size,
        ),
        *load_reference_images(
            reference_paths(
                resolution_config,
                STATE_GRID_STATIC_REFERENCE_FILES,
            ),
            expected_size=standard_size,
        ),
    ]

    inserted_negatives = [
        *load_reference_images(
            reference_paths(resolution_config, INSERTED_NEGATIVE_FILES),
            expected_size=standard_size,
        ),
        *load_reference_images(
            reference_paths(
                resolution_config,
                STAND_RESET_REFERENCE_FILES,
            ),
            expected_size=standard_size,
        ),
        *load_reference_images(
            reference_paths(resolution_config, DROPPED_REFERENCE_FILES),
            expected_size=standard_size,
        ),
    ]
    inserted_references_a = [
        *load_reference_images(
            reference_paths(
                resolution_config,
                INSERTED_BANK_A_POSITIVE_FILES,
            ),
            expected_size=standard_size,
        ),
        *inserted_negatives,
    ]
    inserted_references_b = [
        *load_reference_images(
            reference_paths(
                resolution_config,
                INSERTED_BANK_B_POSITIVE_FILES,
            ),
            expected_size=standard_size,
        ),
        *inserted_negatives,
    ]

    drop_references = load_reference_images(
        reference_paths(resolution_config, DROP_REFERENCE_FILES),
        expected_size=standard_size,
    )
    rotation_static_references = load_reference_images(
        reference_paths(
            resolution_config,
            ROTATION_STATIC_REFERENCE_FILES,
        ),
        expected_size=tight_size,
    )

    def rotation_reference_bank(reference_files):
        normal_references = load_reference_images(
            reference_paths(resolution_config, reference_files),
            expected_size=tight_size,
        )
        return [
            rotation_static_references[0],
            *normal_references,
            *rotation_static_references[1:],
        ]

    rotation_references_a = rotation_reference_bank(
        ROTATION_BANK_A_NORMAL_FILES
    )
    rotation_references_b = rotation_reference_bank(
        ROTATION_BANK_B_NORMAL_FILES
    )
    return {
        "state": state_references,
        "inserted_a": inserted_references_a,
        "inserted_b": inserted_references_b,
        "drop": drop_references,
        "drop_primary": select_reference_images(
            drop_references,
            ("DROPPED", "CONTROLLED", "INSERTED"),
        ),
        "drop_confirmation": select_reference_images(
            drop_references,
            ("DROPPED", "STAND_RESET", "HUMAN"),
        ),
        "rotation_a": rotation_references_a,
        "rotation_b": rotation_references_b,
        "rotation_primary_a": select_reference_images(
            rotation_references_a,
            ("NORMAL_A", "NORMAL_B", "ROTATED_A"),
        ),
        "rotation_primary_b": select_reference_images(
            rotation_references_b,
            ("NORMAL_A", "NORMAL_C", "ROTATED_B"),
        ),
    }


def compact_enum_choice(content, valid_choices, default="U"):
    """Parse and validate a one-character specialist response."""
    if isinstance(content, str):
        value = content.strip().strip("`* .\n\r\t").upper()
        if value in valid_choices:
            return value
    return default


def compact_rotation_vote(content):
    """Parse an R/N/U rotation verifier response."""
    parsed = parse_response_content(content)
    if not isinstance(parsed, dict):
        return "U"
    value = str(parsed.get("rotation", "")).strip().upper()
    return value if value in {"R", "N", "U"} else "U"


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
    state_vlm,
    inserted_vlm,
    drop_vlm,
    rotation_vlm,
    state_prompt,
    inserted_prompt,
    drop_prompt,
    rotation_prompt,
    state_references,
    inserted_references_a,
    inserted_references_b,
    drop_primary_references,
    drop_confirmation_references,
    rotation_primary_references_a,
    rotation_primary_references_b,
    camera,
    stop_event,
    latest_inference_result,
):
    """Run primary safety checks with retained-frame terminal verification."""
    last_frame_index = None
    attempt_manager = TerminalAttemptManager()

    with (
        open(
            LOG_OUTPUT_PATH,
            "w",
            encoding="utf-8",
            buffering=1,
        ) as output_log,
        open(
            LATENCY_OUTPUT_PATH,
            "w",
            encoding="utf-8",
            buffering=1,
        ) as latency_log,
        ThreadPoolExecutor(max_workers=4) as model_request_pool,
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
                attempt_manager.hard_reset()

            last_frame_index = current_frame_index
            full_frame = frame_chunk[0][0]
            current_crop = frame_chunk[0][1]
            cycle_started_at = time.perf_counter()
            request_latencies = {
                "state": None,
                "inserted_a": None,
                "inserted_b": None,
                # "drop" remains as an alias for older latency consumers.
                "drop": None,
                "drop_a": None,
                "drop_b": None,
                "rotation_a": None,
                "rotation_b": None,
            }
            observations = {
                "state": None,
                "inserted_a": None,
                "inserted_b": None,
                "drop": None,
                "drop_a": None,
                "drop_b": None,
                "rotation_a": None,
                "rotation_b": None,
            }
            primary_requests_ms = None
            verification_requests_ms = 0.0

            if attempt_manager.phase == "MONITORING":
                primary_requests_started_at = time.perf_counter()
                inserted_message_a = build_compact_grid_message(
                    inserted_prompt,
                    inserted_references_a,
                    current_crop,
                    current_frame_index=current_frame_index,
                    grid_meaning=(
                        "inserted positives, not-inserted hard negatives, and CURRENT"
                    ),
                )
                drop_message_a = build_current_focused_grid_message(
                    drop_prompt,
                    drop_primary_references,
                    current_crop,
                    current_frame_index=current_frame_index,
                    grid_meaning=(
                        "one dropped positive, controlled and inserted hard "
                        "negatives, and an enlarged CURRENT"
                    ),
                )
                tight_crop = tight_current_crop(
                    full_frame,
                    camera.resolution_config,
                )
                rotation_message_a = build_current_focused_grid_message(
                    rotation_prompt,
                    rotation_primary_references_a,
                    tight_crop,
                    current_frame_index=current_frame_index,
                    grid_meaning=(
                        "normal and rotated references with an enlarged CURRENT"
                    ),
                )
                rotation_message_b = build_current_focused_grid_message(
                    rotation_prompt,
                    rotation_primary_references_b,
                    tight_crop,
                    current_frame_index=current_frame_index,
                    grid_meaning=(
                        "an alternate normal and rotated bank with an enlarged "
                        "CURRENT"
                    ),
                )
                futures = {
                    "inserted_a": model_request_pool.submit(
                        invoke_with_latency,
                        inserted_vlm,
                        inserted_message_a["messages"],
                    ),
                    "drop_a": model_request_pool.submit(
                        invoke_with_latency,
                        drop_vlm,
                        drop_message_a["messages"],
                    ),
                    "rotation_a": model_request_pool.submit(
                        invoke_with_latency,
                        rotation_vlm,
                        rotation_message_a["messages"],
                    ),
                    "rotation_b": model_request_pool.submit(
                        invoke_with_latency,
                        rotation_vlm,
                        rotation_message_b["messages"],
                    ),
                }
                contents = {}
                for name, future in futures.items():
                    response, latency = future.result()
                    contents[name] = response.content
                    request_latencies[name] = latency
                primary_requests_ms = (
                    time.perf_counter() - primary_requests_started_at
                ) * 1000
                observations.update(
                    {
                        "inserted_a": compact_enum_choice(
                            contents["inserted_a"],
                            {"I", "N", "U"},
                        ),
                        "drop_a": compact_enum_choice(
                            contents["drop_a"],
                            {"D", "N", "U"},
                        ),
                        "rotation_a": compact_rotation_vote(
                            contents["rotation_a"]
                        ),
                        "rotation_b": compact_rotation_vote(
                            contents["rotation_b"]
                        ),
                    }
                )
                observations["drop"] = observations["drop_a"]
                request_latencies["drop"] = request_latencies["drop_a"]

                if observations["drop_a"] == "D":
                    verification_started_at = time.perf_counter()
                    drop_message_b = build_current_focused_grid_message(
                        drop_prompt,
                        drop_confirmation_references,
                        current_crop,
                        current_frame_index=current_frame_index,
                        grid_meaning=(
                            "an independent dropped comparison with stand and "
                            "human hard negatives and an enlarged CURRENT"
                        ),
                    )
                    drop_response_b, drop_latency_b = invoke_with_latency(
                        drop_vlm,
                        drop_message_b["messages"],
                    )
                    request_latencies["drop_b"] = drop_latency_b
                    observations["drop_b"] = compact_enum_choice(
                        drop_response_b.content,
                        {"D", "N", "U"},
                    )
                    verification_requests_ms += (
                        time.perf_counter() - verification_started_at
                    ) * 1000

                manager_update = attempt_manager.update_safety_specialists(
                    current_frame_index,
                    observations["drop_a"],
                    observations["drop_b"],
                    observations["rotation_a"],
                    observations["rotation_b"],
                )

                if (
                    not manager_update["terminal_emitted"]
                    and observations["inserted_a"] == "I"
                ):
                    verification_started_at = time.perf_counter()
                    inserted_message_b = build_compact_grid_message(
                        inserted_prompt,
                        inserted_references_b,
                        current_crop,
                        current_frame_index=current_frame_index,
                        grid_meaning=(
                            "alternate inserted positives, hard negatives, and "
                            "CURRENT"
                        ),
                    )
                    inserted_response_b, inserted_latency_b = invoke_with_latency(
                        inserted_vlm,
                        inserted_message_b["messages"],
                    )
                    request_latencies["inserted_b"] = inserted_latency_b
                    observations["inserted_b"] = compact_enum_choice(
                        inserted_response_b.content,
                        {"I", "N", "U"},
                    )
                    verification_requests_ms += (
                        time.perf_counter() - verification_started_at
                    ) * 1000
                    manager_update = (
                        attempt_manager.update_completion_specialists(
                            current_frame_index,
                            observations["inserted_a"],
                            observations["inserted_b"],
                        )
                    )

                if manager_update["terminal_emitted"]:
                    terminal_event = {
                        "attempt_id": manager_update["attempt_id"],
                        "source_frame": current_frame_index,
                        "status": manager_update["terminal_status"],
                        "failure_type": manager_update["failure_type"] or "NONE",
                    }
                    print(f"Terminal {json.dumps(terminal_event)}")
            else:
                state_message = build_compact_grid_message(
                    state_prompt,
                    state_references,
                    current_crop,
                    current_frame_index=current_frame_index,
                    grid_meaning=(
                        "labeled initial/reset, controlled, safe, and dropped "
                        "references plus CURRENT"
                    ),
                )
                state_response, state_latency = invoke_with_latency(
                    state_vlm,
                    state_message["messages"],
                )
                request_latencies["state"] = state_latency
                observations["state"] = compact_state_choice(
                    state_response.content
                )
                attempt_manager.observe_initial_state(observations["state"])
                primary_requests_ms = (
                    time.perf_counter() - cycle_started_at
                ) * 1000

            cycle_requests_ms = (time.perf_counter() - cycle_started_at) * 1000
            manager_state = attempt_manager.snapshot(current_frame_index)
            fused_response = terminal_response(
                manager_state,
                observations,
                initial_state=observations["state"],
            )
            response_content = json.dumps(fused_response, indent=2)
            latest_inference_result.update(
                current_frame_index,
                response_content,
            )

            latency_record = {
                "frame_index": current_frame_index,
                "video_timestamp_s": round(
                    current_frame_index / camera.fps,
                    3,
                ),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "parallel_requests_ms": round(cycle_requests_ms, 3),
                "fast_parallel_requests_ms": round(primary_requests_ms, 3),
                "primary_requests_ms": round(primary_requests_ms, 3),
                "verification_requests_ms": round(
                    verification_requests_ms,
                    3,
                ),
                "requests": request_latencies,
                "observations": observations,
                "attempt": manager_state,
            }
            latency_line = json.dumps(latency_record, separators=(",", ":"))
            print(f"Latency {latency_line}")
            print(latency_line, file=latency_log, flush=True)

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
    for output_path in (
        Path(LOG_OUTPUT_PATH),
        Path(LATENCY_OUTPUT_PATH),
        Path(VIDEO_OUTPUT_PATH),
    ):
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

    try:
        camera = SimulatedCamera(
            VIDEO_INPUT_PATH,
            loop=False,
            window_size=WINDOW_SIZE,
        )
    except UnsupportedResolutionError as error:
        print(error)
        return 1

    resolution_config = camera.resolution_config
    if resolution_config.source_size != VALIDATED_PRODUCTION_SOURCE_SIZE:
        print(
            "The production safety policy is validated only for "
            f"{VALIDATED_PRODUCTION_SOURCE_SIZE[0]}x"
            f"{VALIDATED_PRODUCTION_SOURCE_SIZE[1]}; got "
            f"{resolution_config.source_size[0]}x"
            f"{resolution_config.source_size[1]}."
        )
        camera.stop()
        return 1
    print(
        "Using resolution configuration "
        f"{resolution_config.source_size[0]}x"
        f"{resolution_config.source_size[1]}: crop "
        f"{resolution_config.crop_top_left} -> "
        f"{resolution_config.crop_bottom_right} "
        f"({resolution_config.crop_size[0]}x"
        f"{resolution_config.crop_size[1]} main panels), tight crop "
        f"{resolution_config.tight_crop_top_left} -> "
        f"{resolution_config.tight_crop_bottom_right} "
        f"({resolution_config.tight_crop_size[0]}x"
        f"{resolution_config.tight_crop_size[1]} rotation panels), "
        f"references {resolution_config.reference_directory}."
    )

    # vlm = ChatOllama(
    #     model="gemma4:31b-cloud",
    #     base_url="https://ollama.com",
    #     temperature=0,
    # )

    base_vlm = VLLMTimingChatOpenAI(
        model="/workspace/models/gemma-4-26B-A4B-it",
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        temperature=0,
    )
    state_vlm = base_vlm.bind(max_tokens=8)
    inserted_vlm = base_vlm.bind(max_tokens=8)
    drop_vlm = base_vlm.bind(max_tokens=8)
    rotation_vlm = base_vlm.bind(max_tokens=24)

    state_prompt = STATE_DROP_GRID_PROMPT_PATH.read_text(
        encoding="utf-8"
    )
    inserted_prompt = INSERTED_BINARY_PROMPT_PATH.read_text(encoding="utf-8")
    drop_prompt = DROP_BINARY_PROMPT_PATH.read_text(encoding="utf-8")
    rotation_prompt = ROTATION_VERIFY_PROMPT_PATH.read_text(encoding="utf-8")

    reference_banks = load_terminal_reference_banks(resolution_config)
    state_references = reference_banks["state"]
    inserted_references_a = reference_banks["inserted_a"]
    inserted_references_b = reference_banks["inserted_b"]
    drop_primary_references = reference_banks["drop_primary"]
    drop_confirmation_references = reference_banks["drop_confirmation"]
    rotation_primary_references_a = reference_banks["rotation_primary_a"]
    rotation_primary_references_b = reference_banks["rotation_primary_b"]

    camera.start()
    stop_event = threading.Event()
    latest_inference_result = LatestInferenceResult()
    inference_thread = threading.Thread(
        target=run_inference,
        args=(
            state_vlm,
            inserted_vlm,
            drop_vlm,
            rotation_vlm,
            state_prompt,
            inserted_prompt,
            drop_prompt,
            rotation_prompt,
            state_references,
            inserted_references_a,
            inserted_references_b,
            drop_primary_references,
            drop_confirmation_references,
            rotation_primary_references_a,
            rotation_primary_references_b,
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
    return 0
