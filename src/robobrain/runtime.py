"""UI lifecycle and input adapters around the unchanged monitoring workflow."""

import asyncio
import base64
from collections import deque
import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time

import cv2
import httpx

from .monitor import Backend, ReplayController, SimulatedCamera, Specialist, image_content, load_config, monitor
from .presentation import EpisodeReport, LiveView, annotate_frame, render_recording
from .sources import CompositeVideo, Inputs, LiveCamera, composite
from .workspace import new_id


def prepare(task, store, frames):
    """Translate saved UI blocks to the existing Specialist contract."""
    definitions, thresholds, crops, specialists = {}, {}, {}, {}
    enabled = [s for s in task["specialists"] if s["enabled"]]
    for spec in enabled:
        blocks = [b for b in spec["blocks"] if b["enabled"]]
        cameras = [b for b in blocks if b["type"] == "camera"]
        if not cameras or not spec["definition"]:
            raise ValueError(f"{spec['name']}: add a camera block and the existing detector configuration JSON")
        definitions[spec["id"]] = spec["definition"]
        thresholds[spec["id"]] = spec["threshold"]
        for block in cameras:
            if block["source"] not in frames:
                raise ValueError(f"{spec['name']}: connect and enable its selected source")
            image = frames[block["source"]].image
            crop = block["crop"] or [0, 0, image.shape[1], image.shape[0]]
            x1, y1, x2, y2 = crop
            if not (0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]):
                raise ValueError(f"{spec['name']}: crop {crop} does not fit {image.shape[1]}×{image.shape[0]}. Adjust the crop or capture resolution.")
        crops[spec["id"]] = cameras[0]["crop"] or [0, 0, frames[cameras[0]["source"]].image.shape[1], frames[cameras[0]["source"]].image.shape[0]]
    config = load_config({**task["config"], "specialists": definitions, "evidence_threshold": thresholds, "crops": crops})
    for spec in enabled:
        blocks = [b for b in spec["blocks"] if b["enabled"]]
        prompt = "\n\n".join(b["text"] for b in blocks if b["type"] == "text")
        if not prompt.strip():
            raise ValueError(f"{spec['name']}: write a system prompt before starting")
        inputs = []
        for block in blocks:
            if block["type"] == "camera":
                inputs.append(block)
            elif block["type"] == "image":
                if not block["asset"]:
                    raise ValueError(f"{spec['name']}: upload the reference image or disable its empty block")
                path = store.asset_path(block["asset"])
                if block["crop"]:
                    image = cv2.imread(str(path))
                    content = image_content(image, block["crop"])
                else:
                    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
                    content = {"type": "image_url", "image_url": {"url": f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")}}
                inputs.append({"content": [{"type": "text", "text": block["alias"]}, content]})
        settings = {**config, **spec["definition"], "name": spec["id"], "crop": crops[spec["id"]],
                    "threshold": spec["threshold"], "reference_images": [], "input_blocks": inputs}
        specialists[spec["id"]] = Specialist(prompt, settings)
    return config, specialists


class Recorder:
    """One disk writer, independent of inference; a 30 fps presentation timeline."""
    def __init__(self, inputs, sources, destination, view, ffmpeg):
        self.inputs, self.sources, self.destination, self.view, self.ffmpeg = inputs, sources, destination, view, ffmpeg
        self.stop_event = threading.Event()
        self.error = None
        self.frames = 0
        self.thread = threading.Thread(target=self._run, name="robobrain-recorder", daemon=True)
        self.thread.start()

    def _run(self):
        process = None
        self.temporary = self.destination.with_name("." + self.destination.stem + ".recording.MOV")
        try:
            sample = composite(self.inputs.get().views, {s["id"]: s["name"] for s in self.sources})
            height, width = sample.shape[:2]
            self.dimensions = width, height
            command = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "rawvideo",
                       "-pixel_format", "bgr24", "-video_size", f"{width}x{height}", "-framerate", "30", "-i", "pipe:0",
                       "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.temporary)]
            with tempfile.TemporaryFile() as errors:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
                self.process = process
                while not self.stop_event.is_set() or self.frames == 0:
                    try:
                        snapshot = self.inputs.get()
                        image = composite(snapshot.views, {s["id"]: s["name"] for s in self.sources})
                        if image.shape[:2] != (height, width):
                            raise RuntimeError("Camera dimensions changed during recording")
                    except Exception as exc:
                        self.error = f"Recording interrupted: {exc}"
                        break
                    with self.view.lock:
                        status = dict(self.view.status)
                    image = annotate_frame(image, snapshot.source_index, status, show_frame_numbers=False)
                    process.stdin.write(image.tobytes())
                    self.frames += 1
                    deadline = self.inputs.started_at + self.frames / 30
                    self.stop_event.wait(max(0, deadline - time.monotonic()))
                process.stdin.close()
                code = process.wait(timeout=60)
                if code:
                    errors.seek(0)
                    raise RuntimeError(errors.read().decode(errors="replace"))
                self.temporary.replace(self.destination)
        except Exception as exc:
            self.error = f"Recording failed: {exc}"
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            if process is not None and process.stdin and not process.stdin.closed:
                process.stdin.close()

    def finish(self, interrupted):
        self.stop_event.set()
        self.thread.join(65)
        if self.thread.is_alive():
            if hasattr(self, "process"):
                self.process.kill()
            raise RuntimeError("Recording finalization timed out")
        if self.error:
            raise RuntimeError(self.error)
        return {"file": self.destination.name, "fps": 30, "width": self.dimensions[0], "height": self.dimensions[1],
                "codec": "H.264", "original_audio": "no audio input",
                "interrupted": interrupted, "frames": self.frames}


class Runtime:
    def __init__(self, store, *, view=None, stop_event=None):
        self.store = store
        self.phase = "idle"
        self.task_id = None
        self.preview_task_id = None
        self.readers = {}
        self.source_settings = {}
        self.action_lock = asyncio.Lock()
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self.job = None
        self.result = {"status": "ON_GOING", "failure_type": None}
        self.error = None
        self.latest = {}
        self.recent = deque(maxlen=80)
        self.verification = None
        self.output = []
        self.recording = False
        self.started_at = None
        self.requests = self.responses = 0
        self.view = view if view is not None else LiveView()

    @property
    def busy(self):
        return self.phase in {"starting", "running", "stopping", "finalizing"}

    async def disconnect(self, source_id=None):
        keys = [source_id] if source_id else list(self.readers)
        for key in keys:
            reader = self.readers.pop(key, None)
            self.source_settings.pop(key, None)
            if reader:
                await asyncio.to_thread(reader.stop)

    async def connect(self, task, source_id):
        if self.preview_task_id != task["id"]:
            await self.disconnect()
        self.preview_task_id = task["id"]
        source = next((s for s in task["sources"] if s["id"] == source_id), None)
        if source is None:
            raise ValueError("Source not found")
        if not source["enabled"]:
            raise ValueError("Enable this source before connecting it")
        await self.disconnect(source_id)
        if task["input_mode"] == "camera":
            if any(str(s["device"]) == str(source["device"]) for s in self.source_settings.values()):
                raise ValueError("That camera is already connected in another source")
            reader = LiveCamera(source)
        else:
            if not source["video"]:
                raise ValueError("Upload a video for this source")
            reader = await asyncio.to_thread(SimulatedCamera, self.store.asset_path(source["video"]))
        try:
            await asyncio.to_thread(reader.start)
        except BaseException:
            await asyncio.to_thread(reader.stop)
            raise
        self.readers[source_id] = reader
        self.source_settings[source_id] = source

    def log(self, event):
        kind = event["type"]
        if kind == "status":
            self.result = dict(event)
        elif kind == "request_started":
            self.requests += 1
        elif kind == "observation":
            self.responses += 1
            self.latest[event["specialist"]] = {key: event.get(key) for key in
                ("verdict", "reason", "error", "latency_s", "received_time_s", "captured_time_s", "source_frames")}
        elif kind == "verification_started":
            self.verification = {"active": True, "round_id": event["round_id"], "slots": [], "responses": 0}
        elif kind == "verification_finished" and self.verification:
            self.verification.update(active=False, counts=event["counts"], status=event["status"])
        if self.verification and event.get("round_id") == self.verification["round_id"]:
            if kind == "request_started" and event.get("slot") not in self.verification["slots"]:
                self.verification["slots"].append(event["slot"])
            if kind == "observation":
                self.verification["responses"] += 1
        if kind in {"verification_started", "verification_finished", "error", "recording_error", "control_action"}:
            self.recent.append(event)

    def state(self):
        sources = {}
        for key, reader in self.readers.items():
            try:
                frame = reader.get()
                sources[key] = {"connected": True, "width": frame.image.shape[1], "height": frame.image.shape[0],
                                "age_s": max(0, time.monotonic() - frame.observed_at)}
            except Exception as exc:
                sources[key] = {"connected": False, "error": str(exc)}
        return {"phase": self.phase, "busy": self.busy, "task_id": self.task_id, "preview_task_id": self.preview_task_id,
                "result": self.result, "error": self.error, "specialists": self.latest, "verification": self.verification,
                "sources": sources, "recording": self.recording, "output": self.output,
                "requests": self.requests, "responses": self.responses, "events": list(self.recent),
                "elapsed_s": time.monotonic() - self.started_at if self.started_at and self.busy else self.result.get("decision_time_s", 0)}

    def start(self, task, record):
        if self.busy:
            raise ValueError("A run is already active")
        self.phase, self.task_id, self.recording = "starting", task["id"], record
        self.error, self.output, self.latest, self.verification = None, [], {}, None
        self.started_at = None
        self.result = {"status": "ON_GOING", "failure_type": None}
        self.requests = self.responses = 0
        self.recent.clear()
        self.stop_event.clear()
        self.job = asyncio.create_task(self._run(copy.deepcopy(task)))

    async def _run(self, task):
        report = recorder = None
        directory = None
        result = {"status": "ON_GOING", "failure_type": None, "source_frame": -1, "decision_time_s": 0.}
        sources = [s for s in task["sources"] if s["enabled"]]
        try:
            if not sources:
                raise ValueError("Enable at least one input source")
            ffmpeg = shutil.which("ffmpeg") if self.recording else None
            if self.recording and not ffmpeg:
                raise ValueError("Recording requires ffmpeg on PATH")
            if self.preview_task_id != task["id"] or task["input_mode"] == "video":
                await self.disconnect()
            for key in list(self.readers):
                if key not in {s["id"] for s in sources}:
                    await self.disconnect(key)
            for source in sources:
                if self.stop_event.is_set():
                    result["stop_reason"] = "user_stop"
                    return
                if source["id"] not in self.readers or self.source_settings[source["id"]] != source:
                    await self.connect(task, source["id"])
            inputs = Inputs(dict(self.readers), task["input_mode"])
            config, specialists = await asyncio.to_thread(prepare, task, self.store, inputs.get().views)
            limits = httpx.Limits(max_connections=config["max_connections"], max_keepalive_connections=config["max_connections"])
            async with httpx.AsyncClient(limits=limits, trust_env=False) as client:
                response = await client.get(config["base_url"].rstrip("/") + "/models", timeout=config["request_timeout"])
                response.raise_for_status()
                if config["model"] not in {m["id"] for m in response.json()["data"]}:
                    raise ValueError("Configured model is not available at the endpoint")
                if task["input_mode"] == "video":
                    # Rewind previewed videos; a run always starts a new episode.
                    await self.disconnect()
                    for source in sources:
                        await self.connect(task, source["id"])
                    inputs = Inputs(dict(self.readers), "video")
                else:
                    inputs.started_at = time.monotonic()
                self.started_at = inputs.started_at
                run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
                directory = self.store.root / "runs" / task["id"] / run_id if self.recording else None
                self.view.show(inputs, task["name"])
                report = EpisodeReport(Path(run_id), task["name"], config, directory, self.view,
                                       input_info={"mode": task["input_mode"], "sources": sources}, task_template=task)
                def log(event):
                    report.log(event)
                    self.log(event)
                if self.recording and task["input_mode"] == "camera":
                    recorder = Recorder(inputs, sources, directory / f"output-{run_id}.MOV", self.view, ffmpeg)
                self.phase = "running" if not self.stop_event.is_set() else "stopping"
                controller = ReplayController()
                controller.start(run_id, log)
                backend = Backend(config, client, specialists=specialists)
                result = await monitor(inputs, backend, config, log, self.stop_event)
                controller.finish(run_id, result, log)
                self.phase = "finalizing"
                if self.recording and not self.stop_event.is_set():
                    while (task["input_mode"] == "video" and not all(r.ended.is_set() for r in self.readers.values())) or time.monotonic() < inputs.started_at + result["decision_time_s"] + 1:
                        if self.stop_event.is_set() or (recorder and recorder.error):
                            break
                        await asyncio.sleep(.05)
        except Exception as exc:
            self.error = str(exc) or type(exc).__name__
            if isinstance(exc, httpx.HTTPError):
                self.error = f"Model endpoint unavailable: {self.error}. Check the vLLM server and SSH forward."
            result.update(stop_reason="runtime_error", error=self.error)
            if self.started_at:
                result["decision_time_s"] = time.monotonic() - self.started_at
            if report:
                report.log({"type": "error", "message": self.error})
        finally:
            self.phase = "finalizing"
            recording = None
            try:
                if recorder:
                    recording = await asyncio.to_thread(recorder.finish, self.stop_event.is_set())
                elif report and self.recording and task["input_mode"] == "video":
                    paths = {s["id"]: self.store.asset_path(s["video"]) for s in sources}
                    recording = await asyncio.to_thread(render_recording, next(iter(paths.values())),
                        directory / f"output-{run_id}.MOV", report.events, result, ffmpeg, self.stop_event,
                        capture=CompositeVideo(paths, {s["id"]: s["name"] for s in sources}),
                        audio_sources=list(paths.values()), show_frame_numbers=False)
            except Exception as exc:
                self.error = str(exc)
                if report:
                    report.log({"type": "recording_error", "message": self.error})
            finally:
                try:
                    if report:
                        await asyncio.to_thread(report.finish, result, recording)
                except Exception as exc:
                    self.error = str(exc)
                try:
                    await self.disconnect()
                except Exception as exc:
                    self.error = str(exc)
                self.result = result
                if self.verification and self.verification["active"]:
                    self.verification.update(active=False, cancelled=True)
                self.phase = "error" if self.error else "finished"
                if directory:
                    self.output = [str(p.relative_to(self.store.root)) for p in sorted(directory.glob("output-*"))]

    async def close(self):
        self.stop_event.set()
        if self.job and not self.job.done():
            await self.job
        await self.disconnect()
