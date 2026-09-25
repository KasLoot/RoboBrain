"""Synchronized MuJoCo camera, global and composite recordings.

Sampling follows wall time and keeps only the latest pending sample. An overloaded
encoder drops intermediate samples instead of delaying supervision; frames.jsonl
maps every encoded frame to its real sample time and records all resulting gaps.
"""

from dataclasses import dataclass
import copy
import json
import math
from pathlib import Path
import subprocess
import threading
import time

import cv2
import numpy as np

from .presentation import annotate_frame, progress_snapshot


FPS = 30
GLOBAL_SIZE = (1920, 1080)


def _image(image, label):
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"{label} must be a uint8 BGR image")
    return image


def _gap_banner(image, text):
    """Mark held/unavailable images without changing the evidence image buffer."""
    width = image.shape[1]
    scale = max(.32, min(.7, width / 1500))
    while len(text) > 4 and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > width - 16:
        text = text[:-4] + "..."
    height = max(24, round(44 * scale / .7))
    cv2.rectangle(image, (0, 0), (width - 1, height), (30, 35, 145), -1)
    cv2.putText(image, text, (8, height - 8), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


class SimulationLayout:
    """A fixed packed canvas; FFmpeg crops all outputs from the same input frame."""

    def __init__(self, images):
        if not 1 <= len(images) <= 2:
            raise ValueError("Simulation recording requires one or two source views")
        self.native_sizes = {}
        for source_id, image in images.items():
            if not isinstance(source_id, str) or not source_id.isidentifier():
                raise ValueError("Recording source IDs must be identifiers")
            image = _image(image, source_id)
            height, width = image.shape[:2]
            if not (16 <= width <= 1920 and 16 <= height <= 1080):
                raise ValueError("Source recording dimensions must be within 16–1920 by 16–1080")
            self.native_sizes[source_id] = width, height
        self.source_ids = tuple(images)
        self.slot_height = GLOBAL_SIZE[1] // len(images)
        wanted = max(round(width * self.slot_height / height) for width, height in self.native_sizes.values())
        self.left_width = min(960, max(160, wanted + wanted % 2))
        self.combined_size = self.left_width + GLOBAL_SIZE[0], GLOBAL_SIZE[1]
        self.crops = {
            "combined": (0, 0, *self.combined_size),
            "global": (self.left_width, 0, *GLOBAL_SIZE),
        }
        x = 0
        for source_id, (width, height) in self.native_sizes.items():
            width, height = width + width % 2, height + height % 2
            self.crops["view/" + source_id] = (x, GLOBAL_SIZE[1], width, height)
            x += width
        self.transport_size = max(x, self.combined_size[0]), GLOBAL_SIZE[1] + max(crop[3] for name, crop in self.crops.items() if name not in {"combined", "global"})

    def pack(self, images, global_image, status, *, progress=None, elapsed_s=0., gap=None, names=None):
        global_image = _image(global_image, "Global camera")
        if global_image.shape[:2] != (GLOBAL_SIZE[1], GLOBAL_SIZE[0]):
            raise ValueError("Global recording requires the exact 1920×1080 service camera image")
        if tuple(images) != self.source_ids:
            raise ValueError("Recording source IDs changed during recording")
        packed = np.zeros((self.transport_size[1], self.transport_size[0], 3), np.uint8)
        global_view = annotate_frame(global_image, -1, {"status": "ON_GOING", **status},
                                     show_frame_numbers=False, progress=progress, elapsed_s=elapsed_s)
        if gap:
            _gap_banner(global_view, gap)
        packed[:GLOBAL_SIZE[1], self.left_width:self.left_width + GLOBAL_SIZE[0]] = global_view
        for index, source_id in enumerate(self.source_ids):
            image = _image(images[source_id], source_id)
            width, height = self.native_sizes[source_id]
            if image.shape[:2] != (height, width):
                raise ValueError(f"{source_id} dimensions changed during recording")
            x, y, _, _ = self.crops["view/" + source_id]
            packed[y:y + height, x:x + width] = image
            if gap:
                _gap_banner(packed[y:y + height, x:x + width], gap)
            scale = min(self.left_width / width, self.slot_height / height)
            resized_size = max(1, round(width * scale)), max(1, round(height * scale))
            side = cv2.resize(image, resized_size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            side_x = (self.left_width - resized_size[0]) // 2
            side_y = index * self.slot_height + (self.slot_height - resized_size[1]) // 2
            packed[side_y:side_y + resized_size[1], side_x:side_x + resized_size[0]] = side
            label = (names or {}).get(source_id, source_id)
            cv2.rectangle(packed, (0, index * self.slot_height), (self.left_width - 1, index * self.slot_height + 30), (24, 32, 46), -1)
            cv2.putText(packed, label[:70], (10, index * self.slot_height + 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
        return packed


@dataclass(frozen=True)
class _Sample:
    images: dict
    global_image: np.ndarray
    metadata: dict
    sources: dict
    status: dict
    progress: dict | None
    slot: int
    wall_elapsed_s: float
    sampled_at_ns: int
    unavailable: str | None = None


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Recording metadata contains non-JSON {type(value).__name__}")


class SimulationRecorder:
    """Bounded sampler and synchronized multi-output encoder, independent of inference."""

    def __init__(self, inputs, sources, destination, view, ffmpeg, stop_monitoring, on_event=None):
        self.inputs, self.sources = inputs, list(sources)
        self.destination, self.view, self.ffmpeg = Path(destination), view, ffmpeg
        self.stop_monitoring, self.on_event = stop_monitoring, on_event
        self.stop_event = threading.Event()
        self.error = None
        self.frames = self.gap_frames = self.dropped_samples = self.unavailable_frames = self.omitted_samples = 0
        self.stop_reason = None
        self.dimensions = None
        self.files = []
        self._condition = threading.Condition()
        self._pending = None
        self._sampler_done = False
        self._sampling_error = None
        self._episode = None
        self._process = None
        self._origin = float(inputs.started_at)
        self._names = {source["id"]: source.get("name", source["id"]) for source in self.sources}
        if len(self._names) != len(self.sources) or not 1 <= len(self._names) <= 2:
            raise ValueError("Simulation recording requires one or two unique source IDs")
        self._sampler = threading.Thread(target=self._sample, name="robobrain-recording-sampler", daemon=True)
        self.thread = threading.Thread(target=self._run, name="robobrain-simulation-recorder", daemon=True)
        self._sampler.start()
        self.thread.start()

    def _snapshot(self):
        snapshot = self.inputs.get()
        global_image = getattr(snapshot, "global_image", None)
        if global_image is None:
            raise RuntimeError("Waiting for the exact global recording camera image")
        if _image(global_image, "Global camera").shape[:2] != (1080, 1920):
            raise ValueError("Global recording image must be exactly 1920×1080")
        images = {source_id: _image(snapshot.views[source_id].image, source_id).copy() for source_id in self._names}
        metadata = copy.deepcopy(getattr(snapshot, "metadata", None) or {})
        sources = {source_id: {"source_index": frame.source_index, "observed_at_monotonic_s": frame.observed_at,
                              "metadata": copy.deepcopy(getattr(frame, "metadata", None))}
                   for source_id, frame in snapshot.views.items() if source_id in self._names}
        return images, global_image.copy(), metadata, sources

    def _sample(self):
        last = None
        warmup_deadline = time.monotonic() + 15
        try:
            while not self.stop_event.is_set():
                unavailable = None
                try:
                    current = self._snapshot()
                    simulation = current[2].get("simulation", {})
                    episode = simulation.get("server_instance_id"), simulation.get("episode_id")
                    if self._episode is not None and episode != self._episode:
                        self.stop_reason = "scene_changed"
                        self.stop_monitoring.set()
                        self.stop_event.set()
                        break
                    self._episode = episode
                    last = current
                except RuntimeError as exc:
                    if last is None:
                        if time.monotonic() >= warmup_deadline:
                            raise RuntimeError(f"No complete recording snapshot within 15 seconds: {exc}") from exc
                        if self.stop_event.wait(.05):
                            break
                        continue
                    current, unavailable = last, str(exc)
                now = time.monotonic()
                elapsed = max(0., now - self._origin)
                slot = math.floor(elapsed * FPS)
                with self.view.lock:
                    status = copy.deepcopy(self.view.status)
                    progress = copy.deepcopy(self.view.progress) if self.view.progress else None
                progress = progress_snapshot(progress, elapsed) if progress else None
                sample = _Sample(*current, status, progress, slot, elapsed, time.monotonic_ns(), unavailable)
                with self._condition:
                    if self.stop_event.is_set():
                        break
                    if self._pending is not None:
                        self.dropped_samples += 1
                    self._pending = sample
                    self._condition.notify()
                self.stop_event.wait(max(0., self._origin + (slot + 1) / FPS - time.monotonic()))
        except Exception as exc:
            self._sampling_error = exc
            self.stop_monitoring.set()
            self.stop_event.set()
        finally:
            with self._condition:
                self._sampler_done = True
                self._condition.notify_all()

    def _next(self):
        with self._condition:
            while self._pending is None and not self._sampler_done and not self.stop_event.is_set():
                self._condition.wait(.1)
            sample, self._pending = self._pending, None
            return sample

    def _command(self, layout, temporary):
        width, height = layout.transport_size
        command = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "rawvideo",
                   "-pixel_format", "bgr24", "-video_size", f"{width}x{height}", "-framerate", str(FPS), "-i", "pipe:0"]
        labels = [f"input{index}" for index in range(len(layout.crops))]
        filters = [f"[0:v]split={len(labels)}" + "".join(f"[{label}]" for label in labels)]
        for index, (name, (x, y, width, height)) in enumerate(layout.crops.items()):
            filters.append(f"[{labels[index]}]crop={width}:{height}:{x}:{y}[output{index}]")
        command += ["-filter_complex_threads", "1", "-filter_complex", ";".join(filters)]
        for index, name in enumerate(layout.crops):
            command += ["-map", f"[output{index}]", "-an", "-c:v", "libx264", "-threads", "1", "-preset", "veryfast",
                        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary[name])]
        return command

    def _run(self):
        process = None
        lock_path = None
        lock_owned = False
        temporary = {}
        previous = None
        try:
            sample = self._next()
            if sample is None:
                raise self._sampling_error or RuntimeError("Recording stopped before a complete simulation frame arrived")
            layout = SimulationLayout(sample.images)
            self.dimensions = layout.combined_size
            directory = self.destination.parent
            directory.mkdir(parents=True, exist_ok=True)
            lock_path = directory / ".recording.lock"
            # A session folder belongs to one recording, including parallel starters.
            with lock_path.open("x", encoding="utf-8") as lock:
                lock.write("SimulationRecorder\n")
            lock_owned = True
            output = {"combined": self.destination, "global": directory / "global.mp4",
                      **{"view/" + source_id: directory / f"view-{source_id}.mp4" for source_id in self._names}}
            if len(set(output.values())) != len(output):
                raise ValueError("Recording output filenames must be distinct")
            metadata_path = directory / "frames.jsonl"
            if any(path.exists() for path in [*output.values(), metadata_path]):
                raise FileExistsError("Recording session already contains output files; use a new session directory")
            temporary = {name: path.with_name("." + path.stem + ".recording.mp4") for name, path in output.items()}
            if any(path.exists() for path in temporary.values()):
                raise FileExistsError("Recording session contains unfinished videos; use a new session directory")
            stderr_path = directory / ".ffmpeg.stderr.log"
            with metadata_path.open("x", encoding="utf-8") as metadata_file, stderr_path.open("x+b") as errors:
                process = self._process = subprocess.Popen(self._command(layout, temporary), stdin=subprocess.PIPE,
                                                           stdout=subprocess.DEVNULL, stderr=errors)
                while sample is not None:
                    skipped = max(0, sample.slot - previous.slot - 1) if previous else 0
                    gap = ("Simulation image unavailable - last received frame" if sample.unavailable else
                           f"Recording gap: {skipped} wall-time samples omitted" if skipped else None)
                    packed = layout.pack(sample.images, sample.global_image, sample.status, progress=sample.progress,
                                         elapsed_s=sample.wall_elapsed_s, gap=gap, names=self._names)
                    process.stdin.write(packed.tobytes())
                    simulation = sample.metadata.get("simulation", {})
                    def identity(value):
                        sim = value.metadata.get("simulation", {})
                        return (sim.get("server_instance_id"), sim.get("episode_id"), sim.get("state_seq"),
                                tuple((key, (info["metadata"] or {}).get("identity", info["source_index"]))
                                      for key, info in value.sources.items()))
                    repeated = previous is not None and identity(sample) == identity(previous)
                    record = {"frame": self.frames, "video_time_s": self.frames / FPS,
                              "wall_elapsed_s": sample.wall_elapsed_s, "sample_slot": sample.slot,
                              "sampled_at_monotonic_ns": sample.sampled_at_ns, "written_at_monotonic_ns": time.monotonic_ns(),
                              "skipped_samples": skipped, "repeated_snapshot": repeated, "unavailable": sample.unavailable,
                              "simulation": simulation, "sources": sample.sources, "status": sample.status, "progress": sample.progress}
                    metadata_file.write(json.dumps(record, default=_json_default, ensure_ascii=False, allow_nan=False) + "\n")
                    metadata_file.flush()
                    if self.on_event:
                        self.on_event({"type": "simulation_frame", "recorded_frame": self.frames,
                                       "time_s": self.frames / FPS, "wall_elapsed_s": sample.wall_elapsed_s,
                                       "skipped_samples": skipped, "unavailable": sample.unavailable, "simulation": simulation})
                    self.frames += 1
                    self.gap_frames += bool(gap)
                    self.unavailable_frames += sample.unavailable is not None
                    self.omitted_samples += skipped
                    previous = sample
                    sample = self._next()
                if self._sampling_error:
                    raise self._sampling_error
                process.stdin.close()
                code = process.wait(timeout=60)
                if code:
                    errors.seek(0)
                    raise RuntimeError(errors.read().decode(errors="replace") or f"FFmpeg exited with code {code}")
                for name, path in output.items():
                    temporary[name].replace(path)
                self.files = [path.name for path in output.values()] + [metadata_path.name]
            stderr_path.unlink(missing_ok=True)
            if self.stop_reason and self.on_event:
                self.on_event({"type": "recording_stopped", "reason": self.stop_reason, "frames": self.frames})
        except Exception as exc:
            self.error = f"Recording failed: {exc}"
            self.stop_monitoring.set()
            self.stop_event.set()
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            if process is not None and process.stdin and not process.stdin.closed:
                process.stdin.close()
            if lock_owned:
                lock_path.unlink(missing_ok=True)
            with self._condition:
                self._condition.notify_all()

    def finish(self, interrupted):
        self.stop_event.set()
        with self._condition:
            self._condition.notify_all()
        self.thread.join(65)
        if self.thread.is_alive():
            if self._process is not None:
                self._process.kill()
            raise RuntimeError("Simulation recording finalization timed out")
        self._sampler.join(2)
        if self._sampler.is_alive():
            raise RuntimeError("Simulation recording sampler did not stop")
        if self.error:
            raise RuntimeError(self.error)
        return {"file": self.destination.name, "files": self.files, "metadata_file": "frames.jsonl", "fps": FPS,
                "width": self.dimensions[0], "height": self.dimensions[1], "global_width": GLOBAL_SIZE[0],
                "global_height": GLOBAL_SIZE[1], "codec": "H.264", "original_audio": "no audio input",
                "interrupted": bool(interrupted), "frames": self.frames, "unavailable_frames": self.unavailable_frames,
                "gap_frames": self.gap_frames, "omitted_samples": self.omitted_samples,
                "dropped_samples": self.dropped_samples, "stop_reason": self.stop_reason,
                "timing": "30 fps presentation; frames.jsonl records wall time and omitted samples"}
