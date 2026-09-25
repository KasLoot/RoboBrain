"""Loopback simulation service; network work never owns live MuJoCo data.

Runtime dependencies are injected so protocol and ownership behavior can be checked
without starting a renderer or making a model-provider request.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import json
import math
import queue
import struct
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from typing import Any

PROTOCOL_VERSION = "1"
MAX_VIEWS = 2
MAX_EVENTS = 512
MAX_COMMANDS = 4096


def now_ns() -> int:
    return time.monotonic_ns()


class ProtocolError(Exception):
    def __init__(self, code: int, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


def encode_bundle(header: dict[str, Any], images: list[bytes]) -> bytes:
    """Offsets are relative to the first encoded image, not the JSON header."""
    header = copy.deepcopy(header)
    offset = 0
    for view, data in zip(header["views"], images):
        view.update(offset=offset, length=len(data))
        offset += len(data)
    encoded = json.dumps(header, allow_nan=False, separators=(",", ":")).encode()
    return struct.pack(">I", len(encoded)) + encoded + b"".join(images)


class SimulationService:
    def __init__(self, runtime: Any):
        self.runtime = runtime
        self.server_instance_id = uuid.uuid4().hex
        self.episode_id = uuid.uuid4().hex
        if hasattr(runtime, "set_identity"):
            runtime.set_identity(self.server_instance_id, self.episode_id, generation=0)
        self.attempt_id: str | None = None
        self.execution_generation = 0
        self.service_state = "READY"
        self.execution_state = "IDLE"
        self.last_applied_command: dict | None = None
        self.error: str | None = None
        self.renderer_error: str | None = None
        self._lock = threading.RLock()
        self._closed = threading.Event()
        self._started = False
        self._views: dict[str, dict] = {}
        self._lease: dict | None = None
        self._commands: dict[str, dict] = {}
        self._command_bodies: dict[str, str] = {}
        self._pending: str | None = None
        self._command_queue: queue.Queue = queue.Queue(maxsize=4)
        self._snapshot_queue: queue.Queue = queue.Queue(maxsize=4)
        self._events: deque = deque(maxlen=MAX_EVENTS)
        self._event_seq = 0
        self._bundle_seq = 0
        self._global_subscribers = 0
        self._latest: tuple[dict, list[bytes]] | None = None
        self._deferred_finish: tuple[int, str, str | None] | None = None

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._command_thread = threading.Thread(target=self._command_worker, name="simulation-commands", daemon=True)
            self._render_thread = threading.Thread(target=self._render_worker, name="monitor-renderer", daemon=True)
            self._command_thread.start()
            self._render_thread.start()
            self._event("service_ready")

    def close(self) -> None:
        self._closed.set()
        if self._started:
            self._command_thread.join(timeout=5)
            self._render_thread.join(timeout=5)
        self.runtime.close()
        with self._lock:
            self.service_state = "CLOSED"
            self._event("service_closed")

    def _identity(self) -> dict:
        return {"server_instance_id": self.server_instance_id, "episode_id": self.episode_id,
                "attempt_id": self.attempt_id, "execution_generation": self.execution_generation}

    def _event(self, kind: str, **fields: Any) -> None:
        self._event_seq += 1
        self._events.append({"event_seq": self._event_seq, "event": kind,
                             "server_monotonic_ns": now_ns(), **self._identity(), **fields})

    def status(self) -> dict:
        with self._lock:
            runtime = self.runtime.status()
            if not runtime.get("physics_healthy", True) and self.service_state not in ("ERROR", "CLOSED"):
                self.service_state = "ERROR"
                self.execution_state = "ERROR"
                self.execution_generation += 1
                self.error = runtime.get("physics_error") or "Physics failed; restart the service"
                self._latest = None
                self._event("service_failed", error=self.error)
            lease = self._active_lease()
            return {"protocol_version": PROTOCOL_VERSION, **self._identity(),
                    "service_state": self.service_state, "execution_state": self.execution_state,
                    "executor_state": self.execution_state,
                    "last_applied_command": copy.deepcopy(self.last_applied_command),
                    "pending_command_id": self._pending, "event_seq": self._event_seq,
                    "error": self.error, "renderer_error": self.renderer_error,
                    "control_owner": lease["client_id"] if lease else None,
                    "control_lease_expires_at_server_monotonic_ns": lease["lease_expires_at_server_monotonic_ns"] if lease else None,
                    **runtime}

    def health(self) -> dict:
        received = now_ns()
        with self._lock:
            return {"protocol_version": PROTOCOL_VERSION, "server_instance_id": self.server_instance_id,
                    "ready": self.service_state == "READY", "service_state": self.service_state,
                    "received_at_server_monotonic_ns": received,
                    "sent_at_server_monotonic_ns": now_ns(),
                    "server_received_monotonic_ns": received,
                    "server_sent_monotonic_ns": now_ns(), "server_monotonic_ns": now_ns()}

    def _check_instance(self, body: dict) -> None:
        if body.get("server_instance_id") != self.server_instance_id:
            raise ProtocolError(409, "Service instance changed; read /v1/status before retrying")

    def _active_lease(self) -> dict | None:
        if self._lease and self._lease["lease_expires_at_server_monotonic_ns"] <= now_ns():
            self._lease = None
            self._event("control_lease_expired")
        return self._lease

    def _lease_duration(self, body: dict) -> float:
        try:
            value = float(body.get("lease_seconds", 30))
        except (ValueError, TypeError):
            raise ProtocolError(422, "lease_seconds must be numeric")
        if not math.isfinite(value) or not 1 <= value <= 300:
            raise ProtocolError(422, "lease_seconds must be between 1 and 300")
        return value

    def claim(self, body: dict) -> dict:
        with self._lock:
            self._check_instance(body)
            if self._active_lease():
                raise ProtocolError(409, "Execution control is already leased")
            duration = self._lease_duration(body)
            self._lease = {"server_instance_id": self.server_instance_id,
                           "control_token": uuid.uuid4().hex, "client_id": str(body.get("client_id", "anonymous"))[:128],
                           "lease_expires_at_server_monotonic_ns": now_ns() + int(duration * 1e9)}
            self._event("control_claimed", client_id=self._lease["client_id"])
            return dict(self._lease)

    def _check_control(self, body: dict) -> None:
        self._check_instance(body)
        lease = self._active_lease()
        if not lease or body.get("control_token") != lease["control_token"]:
            raise ProtocolError(403, "Valid execution control lease required")

    def renew(self, body: dict) -> dict:
        with self._lock:
            self._check_control(body)
            self._lease["lease_expires_at_server_monotonic_ns"] = now_ns() + int(self._lease_duration(body) * 1e9)
            return dict(self._lease)

    def release(self, body: dict) -> dict:
        with self._lock:
            self._check_control(body)
            self._lease = None
            self._event("control_released")
            return {"released": True}

    def command(self, body: dict) -> dict:
        with self._lock:
            self._check_control(body)
            request_id = body.get("request_id")
            if not isinstance(request_id, str) or not 1 <= len(request_id) <= 160:
                raise ProtocolError(422, "request_id must be a nonempty string of at most 160 characters")
            try:
                canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
            except (TypeError, ValueError):
                raise ProtocolError(422, "Command body must contain finite JSON values")
            if request_id in self._commands:
                if canonical != self._command_bodies[request_id]:
                    raise ProtocolError(409, "request_id was already used with a different body")
                return copy.deepcopy(self._commands[request_id])
            if len(self._commands) >= MAX_COMMANDS:
                raise ProtocolError(503, "Command history is full; restart service before issuing new commands")
            for name in ("episode_id", "attempt_id", "execution_generation"):
                if name not in body or body[name] != self._identity()[name]:
                    raise ProtocolError(409, f"Obsolete or missing {name}; reconcile current status")
            kind = body.get("command")
            if kind not in ("start", "stop", "reset"):
                raise ProtocolError(422, "command must be start, stop, or reset")
            if self.service_state != "READY":
                raise ProtocolError(409, "Simulation service is not ready")
            if self._pending:
                raise ProtocolError(409, "Wait for the pending command's applied acknowledgement")
            state = self.execution_state
            if kind == "start" and state not in ("IDLE", "STOPPED", "RUNNING"):
                raise ProtocolError(409, "This attempt must be reset before starting")
            if kind == "reset":
                runtime = self.runtime.status()
                if state not in ("IDLE", "STOPPED", "FINISHED", "ERROR"):
                    raise ProtocolError(409, "Stop and await the applied hold before resetting")
                if not runtime.get("physics_healthy", True) or (state == "ERROR" and not runtime.get("hold_applied", False)):
                    raise ProtocolError(409, "Reset requires healthy physics and an applied hold")
            no_op = (kind == "start" and state == "RUNNING") or (kind == "stop" and state in ("IDLE", "STOPPED", "FINISHED"))
            result = {"command_id": request_id, "request_id": request_id, "command": kind,
                      "status": "applied" if no_op else "accepted", "accepted_at_server_monotonic_ns": now_ns(),
                      **self._identity(), "execution_state": state}
            if no_op:
                result["applied_at_server_monotonic_ns"] = now_ns()
                result["no_op"] = True
            self._commands[request_id] = result
            self._command_bodies[request_id] = canonical
            if no_op:
                self._event("command_applied", command=copy.deepcopy(result))
            else:
                self._pending = request_id
                if kind == "stop":
                    self.execution_state = "STOPPING" if state != "ERROR" else "ERROR"
                if kind == "reset":
                    self.service_state = "RESETTING"
                self._event("command_accepted", command=copy.deepcopy(result))
                self._command_queue.put_nowait((request_id, state))
            return copy.deepcopy(result)

    def get_command(self, request_id: str) -> dict:
        with self._lock:
            if request_id not in self._commands:
                raise ProtocolError(404, "Unknown command_id")
            return copy.deepcopy(self._commands[request_id])

    def _execution_finished(self, generation: int, attempt_id: str, error: str | None) -> None:
        with self._lock:
            if generation != self.execution_generation or attempt_id != self.attempt_id:
                if self._pending and self._commands[self._pending]["command"] == "start" and generation == self.execution_generation + 1:
                    self._deferred_finish = (generation, attempt_id, error)
                return
            if self.execution_state != "RUNNING":
                return
            # The runtime installs hold and invalidates targets before reporting completion.
            self.execution_generation += 1
            self.execution_state = "ERROR" if error else "FINISHED"
            self.error = error
            self._latest = None
            self._event("execution_failed" if error else "execution_finished", error=error)

    def _command_worker(self) -> None:
        while not self._closed.is_set():
            try:
                request_id, previous_state = self._command_queue.get(timeout=.05)
            except queue.Empty:
                continue
            with self._lock:
                result = self._commands[request_id]
                kind = result["command"]
                generation = self.execution_generation + 1
                attempt = self.attempt_id or uuid.uuid4().hex
                next_episode = uuid.uuid4().hex if kind == "reset" else self.episode_id
            try:
                if kind == "start":
                    self.runtime.start_execution(generation, attempt,
                                                 lambda error=None, g=generation, a=attempt: self._execution_finished(g, a, error))
                elif kind == "stop":
                    self.runtime.stop_execution(generation)
                else:
                    if hasattr(self.runtime, "set_identity"):
                        self.runtime.set_identity(self.server_instance_id, next_episode, generation=generation)
                    self.runtime.reset(generation)
                with self._lock:
                    if self.service_state == "ERROR":
                        raise RuntimeError(self.error or "Simulation became unavailable while applying command")
                    self.execution_generation = generation
                    if kind == "start":
                        self.attempt_id = attempt
                        self.execution_state = "RUNNING"
                    elif kind == "stop":
                        self.execution_state = "ERROR" if previous_state == "ERROR" else "STOPPED"
                    else:
                        self.episode_id = next_episode
                        self.attempt_id = None
                        self.execution_state = "IDLE"
                        self.service_state = "READY"
                        self.error = None
                    self._latest = None
                    result.update(status="applied", applied_at_server_monotonic_ns=now_ns(),
                                  **self._identity(), execution_state=self.execution_state)
                    self.last_applied_command = copy.deepcopy(result)
                    self._pending = None
                    self._event("command_applied", command=copy.deepcopy(result))
                    if self._deferred_finish:
                        deferred, self._deferred_finish = self._deferred_finish, None
                        self._execution_finished(*deferred)
            except Exception as exc:
                # A failed controller start must still install a hold before exposing ERROR.
                hold_error = None
                try:
                    self.runtime.stop_execution(generation)
                except Exception as stop_exc:
                    hold_error = str(stop_exc)
                with self._lock:
                    self.execution_generation = max(self.execution_generation, generation)
                    self.error = str(exc)
                    self.execution_state = "ERROR"
                    self.service_state = "ERROR" if hold_error else "READY"
                    self._pending = None
                    self._latest = None
                    result.update(status="failed", error=str(exc), hold_error=hold_error, **self._identity())
                    self._event("command_failed", command=copy.deepcopy(result))

    @staticmethod
    def _camera(value: Any) -> dict:
        if not isinstance(value, dict) or value.get("type", "free") != "free":
            raise ProtocolError(422, "Only free cameras are supported")
        try:
            lookat = [float(v) for v in value["lookat_m"]]
            distance = float(value["distance_m"])
            azimuth = float(value["azimuth_deg"])
            elevation = float(value["elevation_deg"])
        except (KeyError, TypeError, ValueError):
            raise ProtocolError(422, "Camera requires lookat_m, distance_m, azimuth_deg and elevation_deg")
        if len(lookat) != 3 or not all(math.isfinite(v) and abs(v) <= 100 for v in lookat):
            raise ProtocolError(422, "lookat_m must contain three finite coordinates in [-100,100]")
        if not all(math.isfinite(v) for v in (distance, azimuth, elevation)) or not .02 <= distance <= 100 or not -90 <= elevation <= 90:
            raise ProtocolError(422, "Invalid camera distance or angles")
        return {"type": "free", "lookat_m": lookat, "distance_m": distance,
                "azimuth_deg": azimuth % 360, "elevation_deg": elevation}

    def views(self) -> list[dict]:
        with self._lock:
            return [{k: copy.deepcopy(v) for k, v in view.items() if k != "view_token"} for view in self._views.values()]

    def create_view(self, body: dict) -> dict:
        with self._lock:
            self._check_instance(body)
            if len(self._views) >= MAX_VIEWS:
                raise ProtocolError(409, "At most two monitor views may be active")
            view_id = body.get("view_id") or uuid.uuid4().hex
            if not isinstance(view_id, str) or not view_id or len(view_id) > 128:
                raise ProtocolError(422, "Invalid view_id")
            if view_id == "__global__":
                raise ProtocolError(422, "Reserved recording view identifier")
            if view_id in self._views:
                raise ProtocolError(409, "view_id already exists")
            try:
                width, height = int(body.get("width", 640)), int(body.get("height", 480))
                fps = float(body.get("requested_fps", 15))
            except (ValueError, TypeError):
                raise ProtocolError(422, "Invalid view dimensions or frame rate")
            if not 64 <= width <= 1280 or not 64 <= height <= 960 or not math.isfinite(fps) or not 1 <= fps <= 30:
                raise ProtocolError(422, "Views support 64..1280 x 64..960 pixels and 1..30 fps")
            view = {"server_instance_id": self.server_instance_id, "view_id": view_id,
                    "view_token": uuid.uuid4().hex, "camera_revision": 1,
                    "camera": self._camera(body.get("camera")), "width": width, "height": height,
                    "requested_fps": fps}
            self._views[view_id] = view
            self._latest = None
            self._event("view_created", view_id=view_id, camera_revision=1)
            return copy.deepcopy(view)

    def _check_view(self, view_id: str, body: dict) -> dict:
        self._check_instance(body)
        if view_id not in self._views:
            raise ProtocolError(404, "Unknown monitor view")
        view = self._views[view_id]
        if body.get("view_token") != view["view_token"]:
            raise ProtocolError(403, "View ownership token required")
        revision = body.get("camera_revision", body.get("expected_camera_revision"))
        if revision != view["camera_revision"]:
            raise ProtocolError(409, "Camera revision changed; reconcile the view before updating")
        return view

    def update_view(self, view_id: str, body: dict) -> dict:
        with self._lock:
            view = self._check_view(view_id, body)
            if any(key in body and body[key] != view[key] for key in ("width", "height")):
                raise ProtocolError(409, "View dimensions are immutable; recreate the view after monitoring stops")
            view["camera"] = self._camera(body.get("camera"))
            view["camera_revision"] += 1
            self._latest = None
            self._event("view_updated", view_id=view_id, camera_revision=view["camera_revision"])
            return copy.deepcopy(view)

    def delete_view(self, view_id: str, body: dict) -> dict:
        with self._lock:
            self._check_view(view_id, body)
            del self._views[view_id]
            self._latest = None
            self._event("view_deleted", view_id=view_id)
            return {"deleted": True}

    def validate_subscription(self, body: dict) -> tuple[list[str], str]:
        with self._lock:
            self._check_instance(body)
            ids = body.get("view_ids")
            if not isinstance(ids, list) or not 1 <= len(ids) <= MAX_VIEWS or any(not isinstance(v, str) for v in ids) or len(set(ids)) != len(ids):
                raise ProtocolError(422, "view_ids must contain one or two distinct view identifiers")
            if any(v not in self._views for v in ids):
                raise ProtocolError(404, "Unknown monitor view")
            if type(body.get('include_global', False)) is not bool:
                raise ProtocolError(422, 'include_global must be boolean')
            encoding = body.get("encoding", "jpeg")
            if encoding not in ("jpeg", "png"):
                raise ProtocolError(422, "encoding must be jpeg or png")
            return ids, encoding

    def snapshot(self, body: dict) -> concurrent.futures.Future:
        ids, encoding = self.validate_subscription(body)
        future: concurrent.futures.Future = concurrent.futures.Future()
        try:
            self._snapshot_queue.put_nowait((ids, encoding, future, body.get('include_global', False)))
        except queue.Full:
            raise ProtocolError(503, "Snapshot renderer is busy; retry later")
        return future

    def _render(self, ids: list[str], encoding: str, include_global=False) -> tuple[dict, list[bytes]]:
        with self._lock:
            if self.service_state != "READY":
                raise ProtocolError(503, "Scene is not ready")
            identity = self._identity()
            views = [copy.deepcopy(self._views[i]) for i in ids]
        rendered = (self.runtime.render_bundle(views, encoding, include_global=True) if include_global
                    else self.runtime.render_bundle(views, encoding))
        if include_global:
            views.append({'view_id': '__global__', 'camera_revision': 1, 'camera': {'fixed': 'lab_overview'},
                          'width': 1920, 'height': 1080})
        with self._lock:
            if identity != self._identity() or self.service_state != "READY":
                raise ProtocolError(409, "Scene or execution changed during capture; retry")
            if rendered.get("execution_generation", identity["execution_generation"]) != identity["execution_generation"]:
                raise ProtocolError(409, "Capture belongs to an obsolete execution generation")
            image_map = {item["view_id"]: item for item in rendered["images"]}
            if set(image_map) != {v["view_id"] for v in views}:
                raise ProtocolError(503, "Renderer returned an incomplete view bundle")
            self._bundle_seq += 1
            header = {"protocol_version": PROTOCOL_VERSION, **identity, "bundle_id": self._bundle_seq,
                      "state_seq": rendered["state_seq"], "sim_time_s": rendered["sim_time_s"],
                      "captured_at_server_monotonic_ns": rendered["captured_at_server_monotonic_ns"],
                      "sent_at_server_monotonic_ns": now_ns(), "views": []}
            data = []
            for view in views:
                item = image_map[view["view_id"]]
                if item.get("width", view["width"]) != view["width"] or item.get("height", view["height"]) != view["height"]:
                    raise ProtocolError(503, "Renderer returned unexpected dimensions")
                header["views"].append({k: copy.deepcopy(view[k]) for k in ("view_id", "camera_revision", "camera", "width", "height")})
                header["views"][-1].update(encoding=encoding, projection=item.get("projection", {"type": "perspective", "roll_deg": 0}))
                data.append(item["data"])
            self.renderer_error = None
            return header, data

    def _render_worker(self) -> None:
        next_frame = 0.0
        try:
            while not self._closed.is_set():
                request = None
                try:
                    request = self._snapshot_queue.get(timeout=.005)
                except queue.Empty:
                    pass
                if request:
                    ids, encoding, future, include_global = request
                    if future.set_running_or_notify_cancel():
                        try:
                            header, data = self._render(ids, encoding, include_global)
                            future.set_result(encode_bundle(header, data))
                        except Exception as exc:
                            future.set_exception(exc)
                if time.monotonic() < next_frame:
                    continue
                with self._lock:
                    ids = list(self._views)
                    fps = max((v["requested_fps"] for v in self._views.values()), default=15)
                next_frame = time.monotonic() + 1 / fps
                if not ids:
                    continue
                try:
                    latest = self._render(ids, "jpeg", self._global_subscribers > 0)
                    with self._lock:
                        self._latest = latest
                except Exception as exc:
                    with self._lock:
                        self.renderer_error = str(exc)
        finally:
            # Render contexts must be destroyed on the same thread that created them.
            if hasattr(self.runtime, "close_renderer"):
                self.runtime.close_renderer()
            while not self._snapshot_queue.empty():
                _, _, future, _ = self._snapshot_queue.get_nowait()
                if not future.done():
                    future.set_exception(ProtocolError(503, "Service closed"))

    def latest_bundle(self, ids: list[str], after_bundle: int, include_global=False) -> tuple[int, bytes] | None:
        with self._lock:
            if any(i not in self._views for i in ids):
                raise ProtocolError(404, "Subscribed monitor view was removed")
            if not self._latest or self._latest[0]["bundle_id"] == after_bundle:
                return None
            header, data = self._latest
            # Revisions may have changed after rendering began; never publish that as current.
            if any(self._views.get(v["view_id"], {}).get("camera_revision") != v["camera_revision"] for v in header["views"] if v["view_id"] != "__global__"):
                return None
            if header["execution_generation"] != self.execution_generation or header["episode_id"] != self.episode_id:
                return None
            selected = [(v, data[i]) for i, v in enumerate(header["views"]) if v["view_id"] in ids]
            if len(selected) != len(ids):
                return None
            if include_global:
                global_items = [(v, data[i]) for i, v in enumerate(header['views']) if v['view_id'] == '__global__']
                if not global_items:
                    return None
                selected.extend(global_items)
            header = copy.deepcopy(header)
            header["views"] = [v for v, _ in selected]
            header["sent_at_server_monotonic_ns"] = now_ns()
            return header["bundle_id"], encode_bundle(header, [d for _, d in selected])

    def events(self, after_seq: int) -> tuple[bool, list[dict]]:
        with self._lock:
            gap = bool(self._events and after_seq < self._events[0]["event_seq"] - 1)
            return gap, [copy.deepcopy(e) for e in self._events if e["event_seq"] > after_seq]


def create_app(runtime: Any):
    from fastapi import Body, FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse, Response, StreamingResponse
    # Postponed annotations in local route definitions must be visible to FastAPI.
    globals().update(Request=Request, WebSocket=WebSocket)
    service = SimulationService(runtime)

    @asynccontextmanager
    async def lifespan(app):
        service.start()
        try:
            yield
        finally:
            await asyncio.to_thread(service.close)

    app = FastAPI(title="Show-Harness MuJoCo", version=PROTOCOL_VERSION, lifespan=lifespan)
    app.state.simulation = service

    @app.exception_handler(ProtocolError)
    async def protocol_error(request: Request, exc: ProtocolError):
        return JSONResponse(status_code=exc.code, content={"detail": exc.detail})

    @app.get("/v1/health")
    def health():
        return service.health()

    @app.get("/v1/status")
    def status():
        return service.status()

    @app.post("/v1/control-session")
    def claim(body: dict = Body(...)):
        return service.claim(body)

    @app.post("/v1/control-session/renew")
    def renew(body: dict = Body(...)):
        return service.renew(body)

    @app.delete("/v1/control-session")
    def release(body: dict = Body(...)):
        return service.release(body)

    @app.post('/v1/recordings')
    def recording(body: dict = Body(...)):
        service._check_instance(body)
        if not hasattr(service.runtime, 'recording_run'):
            raise ProtocolError(501, 'Recording is unavailable in this runtime')
        return service.runtime.recording_run(enable=True)

    @app.post("/v1/commands", status_code=202)
    def command(body: dict = Body(...)):
        return service.command(body)

    @app.get("/v1/commands/{command_id}")
    def get_command(command_id: str):
        return service.get_command(command_id)

    @app.get("/v1/views")
    def views():
        return {"server_instance_id": service.server_instance_id, "views": service.views()}

    @app.post("/v1/views", status_code=201)
    def create_view(body: dict = Body(...)):
        return service.create_view(body)

    @app.patch("/v1/views/{view_id}")
    def update_view(view_id: str, body: dict = Body(...)):
        return service.update_view(view_id, body)

    @app.delete("/v1/views/{view_id}")
    def delete_view(view_id: str, body: dict = Body(...)):
        return service.delete_view(view_id, body)

    @app.post("/v1/snapshots")
    async def snapshot(body: dict = Body(...)):
        future = service.snapshot(body)
        try:
            result = await asyncio.wait_for(asyncio.wrap_future(future), timeout=10)
        except asyncio.TimeoutError:
            raise ProtocolError(504, "Snapshot renderer timed out")
        return Response(content=result, media_type="application/vnd.show-harness.frame-bundle")

    @app.websocket("/v1/frames")
    async def frames(socket: WebSocket):
        await socket.accept()
        include_global = False
        try:
            body = await asyncio.wait_for(socket.receive_json(), timeout=10)
            ids, encoding = service.validate_subscription(body)
            if encoding != "jpeg":
                raise ProtocolError(422, "Live streaming uses jpeg; snapshots also support png")
            include_global = body.get('include_global', False)
            if include_global:
                with service._lock:
                    service._global_subscribers += 1
            after = 0
            while not service._closed.is_set():
                latest = service.latest_bundle(ids, after, include_global)
                if latest:
                    after, data = latest
                    # A blocked client gets no backlog: the next read uses the current latest bundle.
                    await asyncio.wait_for(socket.send_bytes(data), timeout=5)
                await asyncio.sleep(.01)
        except WebSocketDisconnect:
            pass
        except (ProtocolError, asyncio.TimeoutError) as exc:
            try:
                await socket.send_json({"error": str(exc)})
                await socket.close(code=1008)
            except (RuntimeError, WebSocketDisconnect):
                pass
        except RuntimeError:
            # ASGI server reports peer disconnection as RuntimeError on some versions.
            pass

        finally:
            if include_global:
                with service._lock:
                    service._global_subscribers -= 1

    @app.get("/v1/events")
    async def events(request: Request, after_seq: int = 0):
        try:
            after_seq = max(after_seq, int(request.headers.get("last-event-id", "0")))
        except ValueError:
            raise ProtocolError(422, "Invalid Last-Event-ID")
        async def stream():
            cursor = after_seq
            heartbeat = time.monotonic()
            while not service._closed.is_set() and not await request.is_disconnected():
                gap, batch = service.events(cursor)
                if gap:
                    yield "event: history_gap\ndata: " + json.dumps({"reconcile": "/v1/status"}) + "\n\n"
                for event in batch:
                    cursor = event["event_seq"]
                    yield f"id: {cursor}\nevent: {event['event']}\ndata: {json.dumps(event)}\n\n"
                if time.monotonic() - heartbeat >= 15:
                    yield ": heartbeat\n\n"
                    heartbeat = time.monotonic()
                await asyncio.sleep(.05)
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    return app
