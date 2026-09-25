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

from .endpoints import endpoint_error
from .monitor import Backend, ReplayController, SimulatedCamera, Specialist, image_content, load_config, monitor
from .presentation import EpisodeReport, LiveView, annotate_frame, render_recording
from .progress import ProgressRunner
from .sources import CompositeVideo, Inputs, LiveCamera, composite
from .workspace import new_id
from .simulation import SimulationConnection, SimulationReader
from .audit import ModelAudit
from .recording import SimulationRecorder
import json


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
    def __init__(self, inputs, sources, destination, view, ffmpeg, stop_monitoring, on_event=None):
        self.inputs, self.sources, self.destination, self.view, self.ffmpeg = inputs, sources, destination, view, ffmpeg
        self.stop_monitoring = stop_monitoring
        self.on_event = on_event
        self.gap_frames = 0
        self.stop_event = threading.Event()
        self.error = None
        self.frames = 0
        self.thread = threading.Thread(target=self._run, name="robobrain-recorder", daemon=True)
        self.thread.start()

    def _run(self):
        process = None
        self.temporary = self.destination.with_name("." + self.destination.stem + ".recording.MOV")
        try:
            deadline = time.monotonic() + 15
            while True:
                try:
                    snapshot = self.inputs.get()
                    break
                except RuntimeError:
                    if self.inputs.mode != "mujoco" or time.monotonic() >= deadline or self.stop_event.wait(.05):
                        raise
            sample = composite(snapshot.views, {s["id"]: s["name"] for s in self.sources})
            last_image = sample.copy()
            height, width = sample.shape[:2]
            self.dimensions = width, height
            command = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "rawvideo",
                       "-pixel_format", "bgr24", "-video_size", f"{width}x{height}", "-framerate", "30", "-i", "pipe:0",
                       "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.temporary)]
            with tempfile.TemporaryFile() as errors:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
                self.process = process
                while not self.stop_event.is_set() or self.frames == 0:
                    unavailable = None
                    try:
                        current = self.inputs.get()
                        image = composite(current.views, {s["id"]: s["name"] for s in self.sources})
                        if image.shape[:2] != (height, width):
                            raise ValueError("Camera dimensions changed during recording")
                        snapshot = current
                        last_image = image.copy()
                    except RuntimeError as exc:
                        if self.inputs.mode != "mujoco":
                            raise
                        unavailable = str(exc)
                        self.gap_frames += 1
                        image = last_image.copy()
                    if self.on_event and getattr(snapshot, "metadata", None):
                        self.on_event({"type": "simulation_frame", "recorded_frame": self.frames,
                                       "time_s": self.frames / 30, "unavailable": unavailable,
                                       "simulation": snapshot.metadata["simulation"]})
                    with self.view.lock:
                        status = dict(self.view.status)
                        progress = dict(self.view.progress) if self.view.progress else None
                    image = annotate_frame(image, snapshot.source_index, status, show_frame_numbers=False,
                                           progress=progress, elapsed_s=self.frames / 30)
                    if unavailable:
                        cv2.rectangle(image, (0, 0), (width, 42), (20, 20, 140), -1)
                        cv2.putText(image, "Simulation image unavailable - last received frame", (10, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
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
            self.stop_monitoring.set()
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
                "interrupted": interrupted, "frames": self.frames, "unavailable_frames": self.gap_frames}


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
        self.progress = None
        self.simulation = None
        self.simulation_policy = "off"
        self.simulation_lock = asyncio.Lock()
        self._simulation_tasks = set()
        self._auto_stops = set()
        self._event_log = None

    async def simulation_status(self, task=None):
        if task is not None:
            settings = task.get("simulation") or {}
            base_url = settings.get("base_url", "http://127.0.0.1:8766").rstrip("/")
            if self.simulation is None or self.simulation.base_url != base_url:
                if self.simulation:
                    await asyncio.to_thread(self.simulation.close)
                self.simulation = SimulationConnection(settings)
            else:
                self.simulation.settings.update(settings)
        if self.simulation is None:
            return {"connected": False, "error": None}
        try:
            await asyncio.to_thread(self.simulation.refresh_status)
        except (ValueError, RuntimeError) as exc:
            self.simulation.error = str(exc)
        return self.simulation.state()

    async def robot_command(self, command, request_id=None, expected_context=None):
        if self.simulation is None:
            raise ValueError("Connect to the simulation first")
        async with self.simulation_lock:
            try:
                result = await asyncio.to_thread(self.simulation.command, command, request_id, expected_context)
            except RuntimeError as exc:
                raise ValueError(str(exc)) from exc
            event = {"type": "control_action", "action": command, "mode": "MUJOCO",
                     "result": result, "time_s": time.monotonic() - (self.started_at or time.monotonic())}
            if self._event_log:
                self._event_log(event)
            else:
                self.log(event)
            return result

    async def update_simulation_view(self, source_id, camera):
        if self.simulation is None:
            raise ValueError("Connect the simulation source first")
        try:
            result = await asyncio.to_thread(self.simulation.update_view, source_id, camera)
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
        event = {"type": "camera_changed", "source_id": source_id, "view": result,
                 "time_s": time.monotonic() - (self.started_at or time.monotonic())}
        if self._event_log:
            self._event_log(event)
        return result

    async def simulation_reference(self, source_id):
        if self.simulation is None:
            raise ValueError("Connect the simulation source first")
        try:
            return await asyncio.to_thread(self.simulation.reference, source_id)
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc

    async def _stop_from_outcome(self, event):
        try:
            context = event["evidence"]["context"]
            await self.robot_command("stop", "outcome-" + new_id("stop"), context)
        except (ValueError, RuntimeError) as exc:
            self.recent.append({"type": "control_error", "message": str(exc)})

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
        if task["input_mode"] == "mujoco":
            await self.simulation_status(task)
            reader = SimulationReader(self.simulation, source)
        elif task["input_mode"] == "camera":
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
            evidence = event.get("evidence")
            eligible = event.get("status") == "FAIL" or (self.simulation_policy == "terminal" and event.get("status") == "SUCCESS")
            if (self.simulation_policy != "off" and self.simulation and eligible and evidence
                    and self.simulation.status.get("execution_state") == "RUNNING"):
                context = evidence["context"]
                if context not in self._auto_stops and self.simulation.status.get("attempt_id"):
                    self._auto_stops.add(context)
                    job = asyncio.create_task(self._stop_from_outcome(dict(event)))
                    self._simulation_tasks.add(job)
                    job.add_done_callback(self._simulation_tasks.discard)
        elif kind == "request_started":
            self.requests += 1
        elif kind in {"observation", "source_unavailable"}:
            if not event.get("source_unavailable"):
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
                "progress": self.progress.state() if self.progress else None,
                "simulation": self.simulation.state() if self.simulation else {"connected": False, "error": None},
                "elapsed_s": time.monotonic() - self.started_at if self.started_at and self.busy else self.result.get("decision_time_s", 0)}

    def start(self, task, record):
        if self.busy:
            raise ValueError("A run is already active")
        self.phase, self.task_id, self.recording = "starting", task["id"], record
        self.error, self.output, self.latest, self.verification = None, [], {}, None
        self.started_at = None
        self.result = {"status": "ON_GOING", "failure_type": None}
        self.requests = self.responses = 0
        self.progress = None
        self.recent.clear()
        self.stop_event.clear()
        self.simulation_policy = (task.get("simulation") or {}).get("auto_stop", "off") if task["input_mode"] == "mujoco" else "off"
        self._auto_stops.clear()
        self.job = asyncio.create_task(self._run(copy.deepcopy(task)))

    async def _ready_inputs(self, inputs):
        deadline = time.monotonic() + 15
        while True:
            try:
                return inputs.get()
            except RuntimeError:
                if inputs.mode != "mujoco" or time.monotonic() >= deadline or self.stop_event.is_set():
                    raise
                await asyncio.sleep(.05)

    async def _run(self, task):
        report = recorder = None
        directory = None
        run_directory = None
        simulation_episode = None
        audit_events = None
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
            if task['input_mode'] == 'mujoco':
                recording_run = await asyncio.to_thread(self.simulation.recording_run, self.recording)
                if recording_run:
                    run_directory = Path(recording_run['directory']).resolve()
                    if not run_directory.is_relative_to((self.store.root / 'runs').resolve()):
                        raise ValueError('Configure simulator --run-root to the RoboBrain data directory /runs so both processes share one run folder')
                    simulation_episode = recording_run.get('episode_id')
            first_frame = await self._ready_inputs(inputs)
            if run_directory is not None:
                deadline = time.monotonic() + 15
                while True:
                    identity = first_frame.metadata['simulation']
                    if (identity.get('episode_id') == recording_run.get('episode_id')
                            and identity.get('server_instance_id') == recording_run.get('server_instance_id')
                            and identity['execution_generation'] >= recording_run['minimum_generation']):
                        break
                    if time.monotonic() > deadline:
                        raise RuntimeError('Scene changed while binding recording; start a new monitoring session')
                    await asyncio.sleep(.05)
                    recording_run = await asyncio.to_thread(self.simulation.recording_run, self.recording)
                    run_directory = Path(recording_run['directory']).resolve()
                    if not run_directory.is_relative_to((self.store.root / 'runs').resolve()):
                        raise ValueError('Simulator recording root differs from RoboBrain data/runs')
                    simulation_episode = recording_run.get('episode_id')
                    first_frame = await self._ready_inputs(inputs)
            if simulation_episode:
                inputs.expected_simulation_episode = simulation_episode
                inputs.scene_stop_event = self.stop_event
            config, specialists = await asyncio.to_thread(prepare, task, self.store, first_frame.views)
            limits = httpx.Limits(max_connections=config["max_connections"], max_keepalive_connections=config["max_connections"])
            async with httpx.AsyncClient(limits=limits, trust_env=False) as client:
                backend = Backend(config, client, specialists=specialists)
                await backend.check_model()
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
                if run_directory is not None:
                    directory = run_directory / 'robobrain-logs' / run_id
                    directory.mkdir(parents=True, exist_ok=True)
                    backend.audit = ModelAudit(run_directory / 'robobrain-vlm' / run_id)
                    audit_events = (directory / 'events.jsonl').open('a', encoding='utf-8', buffering=1)
                    with (run_directory / 'robobrain-sessions.jsonl').open('a') as sessions:
                        sessions.write(json.dumps({'session_id': run_id, 'task_id': task['id'],
                            'episode_id': simulation_episode, 'record_video': self.recording,
                            'logs': str(directory.relative_to(run_directory))}) + '\n')
                self.view.show(inputs, task["name"])
                report = EpisodeReport(Path(run_id), task["name"], config, directory, self.view,
                                       input_info={"mode": task["input_mode"], "monitoring": "continuous", "sources": sources}, task_template=task)
                def log(event):
                    if backend.audit:
                        event = backend.audit.clean(event)
                    if audit_events:
                        audit_events.write(json.dumps(event, ensure_ascii=False) + '\n')
                    if event.get('type') == 'simulation_context_changed' and simulation_episode:
                        context = event.get('context')
                        if context and context[1] != simulation_episode:
                            self.stop_event.set()
                    report.log(event)
                    self.log(event)
                self._event_log = log
                if (task.get("progress") or {}).get("enabled"):
                    self.progress = ProgressRunner(task["progress"], self.store, inputs, backend, log)
                    self.progress.start()
                if self.recording and task["input_mode"] in {"camera", "mujoco"}:
                    event_loop = asyncio.get_running_loop()
                    recorder_class = Recorder
                    destination = directory / f"output-{run_id}.MOV"
                    if task['input_mode'] == 'mujoco':
                        if run_directory is None:
                            raise ValueError('Simulator does not support combined run recording')
                        video_directory = run_directory / 'robobrain-video'
                        if video_directory.exists():
                            video_directory = video_directory / 'sessions' / run_id
                        video_directory.mkdir(parents=True, exist_ok=True)
                        recorder_class = SimulationRecorder
                        destination = video_directory / 'combined.mp4'
                    recorder = recorder_class(inputs, sources, destination, self.view, ffmpeg, self.stop_event,
                                              on_event=lambda event: event_loop.call_soon_threadsafe(log, event))
                self.phase = "running" if not self.stop_event.is_set() else "stopping"
                controller = ReplayController() if task["input_mode"] != "mujoco" else None
                if controller:
                    controller.start(run_id, log)
                result = await monitor(inputs, backend, config, log, self.stop_event, continuous=True)
                if controller:
                    controller.finish(run_id, result, log)
                self.phase = "finalizing"
                if self.recording and not self.stop_event.is_set():
                    while (task["input_mode"] == "video" and not all(r.ended.is_set() for r in self.readers.values())) or time.monotonic() < inputs.started_at + result["decision_time_s"] + 1:
                        if self.stop_event.is_set() or (recorder and recorder.error):
                            break
                        await asyncio.sleep(.05)
        except Exception as exc:
            result.update(self.result)
            result.pop("type", None)
            self.error = str(exc) or type(exc).__name__
            if isinstance(exc, httpx.HTTPError):
                self.error = endpoint_error(task["config"]["base_url"], exc)
            result.update(stop_reason="runtime_error", error=self.error)
            if self.started_at:
                result["decision_time_s"] = time.monotonic() - self.started_at
            if report:
                report.log(backend.audit.clean({"type": "error", "message": self.error}) if backend.audit else {"type": "error", "message": self.error})
        finally:
            self.phase = "finalizing"
            if self.progress:
                await self.progress.close()
            recording = None
            try:
                if recorder:
                    recording = await asyncio.to_thread(recorder.finish, self.stop_event.is_set())
                    if run_directory is not None:
                        recording['directory'] = str(recorder.destination.parent.relative_to(run_directory))
                        recording['paths_relative_to'] = str(run_directory)
                        with (run_directory / 'robobrain-sessions.jsonl').open('a') as sessions:
                            sessions.write(json.dumps({'session_id': run_id, 'event': 'monitoring_finished',
                                                       'recording': recording}) + '\n')
                elif report and self.recording and task["input_mode"] == "video":
                    paths = {s["id"]: self.store.asset_path(s["video"]) for s in sources}
                    recording = await asyncio.to_thread(render_recording, next(iter(paths.values())),
                        directory / f"output-{run_id}.MOV", report.events, result, ffmpeg, self.stop_event,
                        capture=CompositeVideo(paths, {s["id"]: s["name"] for s in sources}),
                        audio_sources=list(paths.values()), show_frame_numbers=False)
            except Exception as exc:
                recording_error = str(exc)
                self.error = self.error or recording_error
                if result.get("stop_reason") != "runtime_error":
                    result["stop_reason"] = "recording_error"
                result["error"] = self.error
                if report:
                    report.log(backend.audit.clean({"type": "recording_error", "message": recording_error}) if backend.audit else {"type": "recording_error", "message": recording_error})
            finally:
                self.simulation_policy = "off"
                if self._simulation_tasks:
                    await asyncio.gather(*list(self._simulation_tasks), return_exceptions=True)
                self._event_log = None
                if self.simulation and task['input_mode'] == 'mujoco':
                    self.simulation.stop_recording_feed()
                try:
                    if report:
                        await asyncio.to_thread(report.finish, backend.audit.clean(result) if backend.audit else result, recording)
                except Exception as exc:
                    self.error = str(exc)
                try:
                    await self.disconnect()
                except Exception as exc:
                    self.error = str(exc)
                if audit_events:
                    audit_events.close()
                self.result = result
                if self.verification and self.verification["active"]:
                    self.verification.update(active=False, cancelled=True)
                self.phase = "error" if self.error else "finished"
                if directory:
                    paths = list(directory.glob('output-*'))
                    if run_directory:
                        paths += list(run_directory.rglob('*.mp4')) + list(run_directory.glob('*.json')) + list(run_directory.glob('*.jsonl'))
                        for pattern in ('show-harness-video/frames.jsonl', 'show-harness-video/annotations.jsonl',
                                        'robobrain-video/**/frames.jsonl', 'show-harness-vlm/*.jsonl',
                                        'robobrain-vlm/*/calls.jsonl'):
                            paths += list(run_directory.glob(pattern))
                    self.output = [str(p.relative_to(self.store.root)) for p in sorted(set(paths)) if p.is_file()]

    async def close(self):
        self.stop_event.set()
        if self.job and not self.job.done():
            await self.job
        await self.disconnect()
        if self.simulation:
            await asyncio.to_thread(self.simulation.close)
            self.simulation = None
