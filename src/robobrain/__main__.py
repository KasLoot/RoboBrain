import os
import json
from collections import deque

from langchain_openai import ChatOpenAI


os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://eu.api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "RoboBrain"


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
WINDOW_SIZE = 1


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
        """Return frames [t - chunk_num, ..., t] and the current index t."""
        with self.lock:
            if not self.frame_buffer:
                return None, None

            current_frame_index = self.frame_buffer[-1][0]
            frame_chunk = [
                frame.copy()
                for _, frame in self.frame_buffer
            ]
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
        failure_description = parsed_content.get(
            "failure_description",
            "none",
        )
        return [
            header,
            f"Status: {status}",
            f"Failure: {failure_description}",
        ]

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




def build_message(system_prompt, query, camera_images):
    image_content = []
    for camera_image in camera_images:
        ok, encoded_image = cv2.imencode(".jpg", camera_image)
        if not ok:
            raise ValueError("Could not encode camera image")

        image_base64 = base64.b64encode(
            encoded_image.tobytes()
        ).decode("ascii")
        image_content.append(
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
            HumanMessage(
                content=[
                    *image_content,
                    {
                        "type": "text",
                        "text": query,
                    },
                ]
            ),
        ]
    }


def run_inference(
    vlm,
    system_prompt,
    task,
    camera,
    stop_event,
    latest_inference_result,
):
    """Send snapshots to the model without blocking the video canvas."""
    last_frame_index = None

    with open(LOG_OUTPUT_PATH, "a", encoding="utf-8", buffering=1) as output_log:
        while not stop_event.is_set():
            frame_chunk, current_frame_index = camera.get()

            if frame_chunk is None or current_frame_index == last_frame_index:
                stop_event.wait(0.01)
                continue

            last_frame_index = current_frame_index
            message = build_message(system_prompt, task, frame_chunk)
            response = vlm.invoke(message["messages"])
            latest_inference_result.update(
                current_frame_index,
                response.content,
            )

            print(f"Frame [{current_frame_index}] ", response.content)
            print(
                f"Frame [{current_frame_index}] ",
                response.content,
                file=output_log,
                flush=True,
            )


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
                current_frame = frame_chunk[-1]
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
    vlm = ChatOllama(
        model="gemma4:31b-cloud",
        base_url="https://ollama.com",
        temperature=0,
    )

    # vlm = ChatOpenAI(
    #         model="/workspace/models/gemma-4-26B-A4B-it",
    #         base_url="http://127.0.0.1:8000/v1",
    #         api_key="EMPTY",
    #         temperature=0,
    #     )

    with open("src/robobrain/prompt/brain_prompt.md", "r") as f:
        system_prompt = f.read()

    with open("src/robobrain/prompt/TASK.md", "r") as f:
        TASK = f.read()

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
            TASK,
            camera,
            stop_event,
            latest_inference_result,
        ),
        daemon=True,
        name="robobrain-inference",
    )
    inference_thread.start()

    try:
        show_video_canvas(camera, stop_event, latest_inference_result)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        camera.stop()
        cv2.destroyAllWindows()
