"""Shared video overlay, live-view state, and per-episode recordings."""

from collections import deque
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


def progress_snapshot(state, now_s):
    if not state:
        return None
    state = dict(state)
    captured = state.get("confirmed_captured_time_s")
    if captured is None:
        captured = state.get("captured_time_s")
    state["age_s"] = max(0., now_s - captured) if captured is not None else None
    if captured is not None and now_s - captured > state["stale_after_s"] and state["quality"] != "unavailable":
        state["quality"] = "stale"
    state["percent"] = (round(100 * state["completed"] / state["total"], 1)
                        if state.get("total") and state["quality"] == "confirmed" else None)
    return state


def annotate_frame(image, source_frame, status, *, show_frame_numbers=True, progress=None, elapsed_s=0.):
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

    label = f"Frame {source_frame + 1}" if show_frame_numbers else ""
    label_width, text_height = size(label)
    left = width - margin - label_width - 2 * padding
    if show_frame_numbers:
        panel(left, margin, width - margin, margin + text_height + 2 * padding + regular)
        cv2.putText(frame, label, (left + padding, margin + padding + text_height),
                    font, scale, white, regular, cv2.LINE_AA)
    title = "Operating Status"
    suffix = f" (frame {status.get('source_frame', -1) + 1}):" if show_frame_numbers else ":"
    lines = [f"Status: {status['status']}", f"Failure type: {status.get('failure_type') or 'NONE'}"]
    if progress:
        current = progress_snapshot(progress, elapsed_s)
        label = current["label"] if current["node_id"] != "UNKNOWN" else "Unknown"
        if current["quality"] != "confirmed" and current["node_id"] != "UNKNOWN":
            label = "Last known: " + label
        if current.get("total"):
            label += f" ({current['completed']}/{current['total']} {current['unit_label']})"
        for line in ("Progress: " + label, "Evidence: " + current["quality"] + (f" | {current['age_s']:.1f}s old" if current["age_s"] is not None else "")):
            while len(line) > 4 and size(line)[0] > width - 2 * (margin + padding):
                line = line[:-4] + "..."
            lines.append(line)
    title_width, title_height = size(title, bold)
    text_width = max(title_width + size(suffix, 1)[0], *(size(line)[0] for line in lines))
    line_height = title_height + max(8, round(18 * scale))
    left = max(0, width - margin - text_width - 2 * padding)
    top = height - margin - (len(lines) + 1) * line_height - padding
    panel(left, top, width - margin, height - margin)
    x, y = left + padding, top + padding + title_height
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
        self.progress = None

    def show(self, camera, name):
        with self.lock:
            self.camera, self.name = camera, name
            self.status = {"status": "ON_GOING", "failure_type": None, "source_frame": -1}
            self.progress = None

    def update(self, event):
        if event["type"] == "progress_state":
            with self.lock:
                self.progress = dict(event)
        if event["type"] == "status":
            with self.lock:
                self.status = dict(event)

    def snapshot(self):
        with self.lock:
            camera, name, status = self.camera, self.name, dict(self.status)
        return (camera.get(), name, status) if camera else None


class EpisodeReport:
    def __init__(self, video, dataset, config, directory, view, *, input_info=None, task_template=None):
        self.video, self.dataset, self.config = video, dataset, config
        self.directory, self.view = directory, view
        self.simulation_report = bool(input_info and input_info.get('mode') == 'mujoco')
        self.events = deque(maxlen=10000) if self.simulation_report else []
        self.summary_omitted_events = 0
        self.stream = None
        self.input_info, self.task_template = input_info, task_template
        if directory:
            directory.mkdir(parents=True, exist_ok=True)
            self.stream = (directory / f"output-{video.stem}.log").open("w", encoding="utf-8", buffering=1)

    def log(self, event):
        self.view.update(event)
        if self.stream:
            if self.simulation_report and event.get('type') == 'simulation_frame':
                self.summary_omitted_events += 1
            else:
                if self.simulation_report and len(self.events) == self.events.maxlen:
                    self.summary_omitted_events += 1
                self.events.append(event)
            self.stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def finish(self, result, recording=None):
        if not self.stream:
            return
        def readable_progress(event):
            value = dict(event)
            if "source_frame" in value:
                value["source_frame"] += 1
            for key in ("source_frames", "confirmed_source_frames"):
                if key in value:
                    value[key] = {name: {**frame, "source_frame": frame["source_frame"] + 1}
                                  for name, frame in value[key].items()}
            return value

        requests, other_events, progress_requests = [], [], []
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
                field = self.config["specialists"][event["specialist"]]["field"]
                for key in (field, "cancelled", "error", "http_status", "response", "response_text"):
                    if key in event:
                        request[key] = event[key]
                if event.get("raw") is not None:
                    try:
                        request["model_output"] = json.loads(event["raw"])
                    except (ValueError, TypeError):
                        request["model_output"] = event["raw"]
                if "source_frames" in event:
                    request["source_frames"] = {name: {**value, "source_frame": value["source_frame"] + 1}
                                                for name, value in event["source_frames"].items()}
                if "evidence" in event:
                    request["evidence"] = event["evidence"]
                requests.append(request)
            elif event["type"] == "progress_observation":
                progress_requests.append(readable_progress(event))
            elif event["type"] == "progress_state":
                other_events.append(readable_progress(event))
            elif event["type"] not in {"request_started", "progress_request_started"}:
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
        if self.simulation_report:
            document['summary_omitted_events'] = self.summary_omitted_events
            document['full_event_archive'] = f'output-{self.video.stem}.log'
            document['summary_scope'] = 'Latest 10000 non-frame events; full lossless history in event archive'
        if self.input_info is not None:
            document.pop("input_video")
            document["sources"] = self.input_info
            document["task_template"] = self.task_template
            document["frame_numbering"] += " Live camera source_frame uses the 30 fps run timeline; source_frames contains physical per-camera indices."
        if self.view.progress is not None:
            document["progress"] = {"result": readable_progress(progress_snapshot(self.view.progress, result.get("decision_time_s", 0))),
                                    "requests": progress_requests, "request_count": len(progress_requests),
                                    "request_errors": sum("error" in r for r in progress_requests)}
        try:
            write_json(self.directory / f"output-{self.video.stem}.json", document)
        finally:
            self.stream.close()


def render_recording(video, destination, events, result, ffmpeg, stop_event, *, capture=None, audio_sources=None, show_frame_numbers=True):
    """Render all source frames and late decisions at 30 fps; retain original audio."""
    cap = capture if capture is not None else cv2.VideoCapture(str(video))
    temporary = process = None
    written, fps, position = 0, 30, 0
    states = sorted((e for e in events if e["type"] == "status"), key=lambda e: e["decision_time_s"])
    progress_events = sorted((e for e in events if e["type"] == "progress_state"), key=lambda e: e["time_s"])
    progress, progress_position = None, 0
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
                   "-framerate", str(fps), "-i", "pipe:0"]
        audio_sources = [video] if audio_sources is None else audio_sources
        for source in audio_sources:
            command.extend(["-i", str(source)])
        command.extend(["-map", "0:v:0"])
        for index in range(len(audio_sources)):
            command.extend(["-map", f"{index + 1}:a?"])
        command.extend(["-c:v", "libx264", "-preset", "veryfast",
                   "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                   "-af", "apad", "-shortest",
                   "-movflags", "+faststart", str(temporary)])
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)

            def write_frame(current, source_index):
                nonlocal position, state, written, progress, progress_position
                if current.shape[:2] != (height, width):
                    raise ValueError("Source video resolution changed during recording")
                while position < len(states) and states[position]["decision_time_s"] <= written / fps:
                    state = states[position]
                    position += 1
                while progress_position < len(progress_events) and progress_events[progress_position]["time_s"] <= written / fps:
                    progress = progress_events[progress_position]
                    progress_position += 1
                options = {} if show_frame_numbers else {"show_frame_numbers": False}
                if progress:
                    options.update(progress=progress, elapsed_s=written / fps)
                annotated = annotate_frame(current, source_index, state, **options)
                process.stdin.write(annotated.tobytes())
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
                end = limit if interrupted_at_start else math.ceil((result["decision_time_s"] + 1.) * fps)
                while written < end and (interrupted_at_start or not stop_event.is_set()):
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
