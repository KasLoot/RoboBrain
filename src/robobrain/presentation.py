"""Shared video overlay, live-view state, and per-episode recordings."""

import json
import math
from pathlib import Path
import subprocess
import tempfile
import threading

import cv2


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def annotate_frame(image, source_frame, status):
    """Use one-based source frames; zero means no completed observation yet."""
    frame = image.copy()  # Overlays must never reach the inference buffer.
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(.35, min(1., width / 1920))
    regular = max(1, round(scale * 2))
    bold = regular + 2
    margin, padding = max(8, round(20 * scale)), max(6, round(14 * scale))
    blue, white = (255, 145, 50), (255, 255, 255)

    def size(text, thickness=regular):
        return cv2.getTextSize(text, font, scale, thickness)[0]

    def panel(left, top, right, bottom):
        region = frame[max(0, top):min(height, bottom), max(0, left):min(width, right)]
        cv2.convertScaleAbs(region, alpha=.3, dst=region)

    label = f"Frame {source_frame + 1}"
    label_width, text_height = size(label)
    left = width - margin - label_width - 2 * padding
    panel(left, margin, width - margin, margin + text_height + 2 * padding + regular)
    cv2.putText(frame, label, (left + padding, margin + padding + text_height),
                font, scale, white, regular, cv2.LINE_AA)
    title = "Operating Status"
    suffix = f" (frame {status.get('source_frame', -1) + 1}):"
    lines = [f"Status: {status['status']}", f"Failure type: {status.get('failure_type') or 'NONE'}"]
    title_width = size(title, bold)[0]
    text_width = max(title_width + size(suffix, 1)[0], *(size(line)[0] for line in lines))
    line_height = max(text_height, size(title, bold)[1]) + max(8, round(18 * scale))
    left = max(0, width - margin - text_width - 2 * padding)
    top = height - margin - 3 * line_height - padding
    panel(left, top, width - margin, height - margin)
    x, y = left + padding, top + padding + text_height
    cv2.putText(frame, title, (x, y), font, scale, blue, bold, cv2.LINE_AA)
    cv2.putText(frame, suffix, (x + title_width, y), font, scale, blue, 1, cv2.LINE_AA)
    for index, line in enumerate(lines, 1):
        cv2.putText(frame, line, (x, y + index * line_height), font, scale, white, regular, cv2.LINE_AA)
    return frame


class LiveView:
    """The GUI reads snapshots; inference and decoding never call GUI functions."""

    def __init__(self):
        self.lock = threading.Lock()
        self.camera = None
        self.name = ""
        self.status = {"status": "ON_GOING", "failure_type": None, "source_frame": -1}

    def show(self, camera, name):
        with self.lock:
            self.camera, self.name = camera, name
            self.status = {"status": "ON_GOING", "failure_type": None, "source_frame": -1}

    def update(self, event):
        if event["type"] == "status":
            with self.lock:
                self.status = dict(event)

    def snapshot(self):
        with self.lock:
            camera, name, status = self.camera, self.name, dict(self.status)
        return (camera.get(), name, status) if camera else None


class EpisodeReport:
    def __init__(self, video, dataset, config, directory, view):
        self.video, self.dataset, self.config = video, dataset, config
        self.directory, self.view = directory, view
        self.events = []
        self.stream = None
        if directory:
            directory.mkdir(parents=True, exist_ok=True)
            self.stream = (directory / f"output-{video.stem}.log").open("w", encoding="utf-8", buffering=1)

    def log(self, event):
        self.view.update(event)
        if self.stream:
            self.events.append(event)
            self.stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def finish(self, result, recording=None):
        if not self.stream:
            return
        requests, other_events = [], []
        for event in self.events:
            if event["type"] == "observation":
                request = {
                    "request_id": event["request_id"], "specialist": event["specialist"],
                    "source_frame": event["source_frame"] + 1,
                    "captured_at_seconds": event["captured_time_s"],
                    "sent_at_seconds": event["sent_time_s"], "received_at_seconds": event["received_time_s"],
                    "latency_seconds": event["latency_s"], "normal_monitoring": event["normal"],
                    "verification_round": event.get("round_id"),
                    "verification_sample": event["slot"] + 1 if event.get("slot") is not None else None,
                    "verdict": event["verdict"], "reason": event["reason"],
                }
                for key in ("location", "cancelled", "error", "http_status", "response", "response_text"):
                    if key in event:
                        request[key] = event[key]
                if event.get("raw") is not None:
                    try:
                        request["model_output"] = json.loads(event["raw"])
                    except (ValueError, TypeError):
                        request["model_output"] = event["raw"]
                requests.append(request)
            elif event["type"] != "request_started":
                readable = dict(event)
                if "source_frame" in readable:
                    readable["source_frame"] += 1
                if "failure_type" in readable:
                    readable["failure_type"] = readable["failure_type"] or "NONE"
                other_events.append(readable)
        document = {
            "dataset": self.dataset, "episode": self.video.name, "input_video": str(self.video),
            "frame_numbering": "One-based source frames; 0 means no completed observation. The .log retains zero-based raw indices.",
            "result": {**result, "source_frame": result.get("source_frame", -1) + 1,
                       "failure_type": result.get("failure_type") or "NONE"},
            "configuration": self.config, "recording": recording,
            "request_count": len(requests), "request_errors": sum("error" in r for r in requests),
            "cancelled_requests": sum(r.get("cancelled", False) for r in requests),
            "requests": sorted(requests, key=lambda r: r["request_id"]), "events": other_events,
        }
        try:
            write_json(self.directory / f"output-{self.video.stem}.json", document)
        finally:
            self.stream.close()


def render_recording(video, destination, events, result, ffmpeg, stop_event):
    """Render all source frames and late decisions at 30 fps; retain original audio."""
    cap = cv2.VideoCapture(str(video))
    temporary = process = None
    written, fps, position = 0, 30, 0
    states = sorted((e for e in events if e["type"] == "status"), key=lambda e: e["decision_time_s"])
    state = {"status": "ON_GOING", "failure_type": None, "source_frame": -1}
    interrupted_at_start = stop_event.is_set()
    limit = max(1, math.ceil(result["decision_time_s"] * fps)) if interrupted_at_start else None
    try:
        ok, image = cap.read()
        if not ok:
            raise ValueError(f"Cannot decode recording input: {video}")
        height, width = image.shape[:2]
        with tempfile.NamedTemporaryFile(prefix=".robobrain-", suffix=".MOV", dir=destination.parent, delete=False) as file:
            temporary = Path(file.name)
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                   "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{width}x{height}",
                   "-framerate", str(fps), "-i", "pipe:0", "-i", str(video),
                   "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast",
                   "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                   "-af", "apad", "-shortest",
                   "-movflags", "+faststart", str(temporary)]
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)

            def write_frame(current, source_index):
                nonlocal position, state, written
                if current.shape[:2] != (height, width):
                    raise ValueError("Source video resolution changed during recording")
                while position < len(states) and states[position]["decision_time_s"] <= written / fps:
                    state = states[position]
                    position += 1
                process.stdin.write(annotate_frame(current, source_index, state).tobytes())
                written += 1

            index = 0
            try:
                while ok:
                    if written and ((limit is not None and written >= limit) or
                                    (not interrupted_at_start and stop_event.is_set())):
                        break
                    write_frame(image, index)
                    ok, following = cap.read()
                    if ok:
                        image = following
                        index += 1
                end = math.ceil((result["decision_time_s"] + 1.) * fps)
                while written < end and not stop_event.is_set():
                    write_frame(image, index)
                process.stdin.close()
                return_code = process.wait(timeout=60)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired) as exc:
                process.kill()
                process.wait()
                errors.seek(0)
                raise RuntimeError(f"Recording failed ({exc}): " + errors.read().decode(errors="replace")) from exc
            if return_code:
                errors.seek(0)
                raise RuntimeError("Recording failed: " + errors.read().decode(errors="replace"))
        temporary.replace(destination)
        return {"file": destination.name, "fps": fps, "width": width, "height": height,
                "codec": "H.264", "original_audio": "preserved", "interrupted": stop_event.is_set()}
    finally:
        cap.release()
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if process is not None and process.stdin is not None and not process.stdin.closed:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
        if temporary is not None:
            temporary.unlink(missing_ok=True)
