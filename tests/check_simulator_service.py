"""Service contract checks: no renderer installation or provider credentials required.

Run: PYTHONDONTWRITEBYTECODE=1 .venv-integration/bin/python -B tests/check_simulator_service.py
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import struct
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulator"))
from fastapi.testclient import TestClient
from PIL import Image
from show_harness_mujoco.service import create_app


class FakeRuntime:
    def __init__(self):
        self.began = time.monotonic()
        self.generation = 0
        self.starts = []
        self.stops = []
        self.callbacks = []
        self.hold_applied = True
        self.closed = False
        self.physics_healthy = True
        self.render_thread_names = set()
        self.render_entered = threading.Event()
        self.render_release = threading.Event()
        self.render_release.set()

    def status(self):
        elapsed = time.monotonic() - self.began
        return {"physics_healthy": self.physics_healthy, "hold_applied": self.hold_applied,
                "state_seq": int(elapsed * 1000), "sim_time_s": elapsed, "real_time_factor": 1.0}

    def start_execution(self, generation, attempt_id, on_finished):
        self.generation = generation
        self.starts.append((generation, attempt_id))
        self.callbacks.append(on_finished)
        self.hold_applied = False

    def stop_execution(self, generation):
        time.sleep(.025)
        self.stops.append(generation)
        self.generation = generation
        self.hold_applied = True

    def reset(self, generation):
        self.generation = generation
        self.hold_applied = True

    def render_bundle(self, views, encoding):
        self.render_thread_names.add(threading.current_thread().name)
        self.render_entered.set()
        self.render_release.wait(3)
        snapshot = self.status()
        images = []
        for view in views:
            color = int(view["camera"]["azimuth_deg"]) % 255
            image = Image.new("RGB", (view["width"], view["height"]), (color, 20, 80))
            out = io.BytesIO()
            image.save(out, "JPEG" if encoding == "jpeg" else "PNG")
            images.append({"view_id": view["view_id"], "data": out.getvalue(), "width": view["width"], "height": view["height"]})
        return {"state_seq": snapshot["state_seq"], "sim_time_s": snapshot["sim_time_s"],
                "captured_at_server_monotonic_ns": time.monotonic_ns(), "execution_generation": self.generation,
                "images": images}

    def close_renderer(self):
        assert threading.current_thread().name == "monitor-renderer"

    def close(self):
        self.closed = True


def unpack(raw):
    size = struct.unpack(">I", raw[:4])[0]
    header = json.loads(raw[4:4 + size])
    pixels = raw[4 + size:]
    total = 0
    for view in header["views"]:
        assert view["offset"] == total
        chunk = pixels[view["offset"]:view["offset"] + view["length"]]
        image = Image.open(io.BytesIO(chunk))
        assert image.size == (view["width"], view["height"])
        total += len(chunk)
    assert total == len(pixels)
    return header


def run():
    runtime = FakeRuntime()
    app = create_app(runtime)
    with TestClient(app) as client:
        state = client.get("/v1/status").json()
        identity = state["server_instance_id"]
        assert state["execution_state"] == "IDLE"
        health = client.get("/v1/health").json()
        assert health["server_sent_monotonic_ns"] >= health["server_received_monotonic_ns"]
        claim = {"server_instance_id": identity, "client_id": "contract-test", "lease_seconds": 30}
        lease = client.post("/v1/control-session", json=claim).json()
        assert client.post("/v1/control-session", json=claim).status_code == 409
        token = lease["control_token"]
        assert client.post("/v1/control-session/renew", json={**claim, "control_token": "wrong"}).status_code == 403
        assert client.post("/v1/control-session/renew", json={**claim, "control_token": token}).status_code == 200

        def body(kind, request_id):
            current = client.get("/v1/status").json()
            return {k: current[k] for k in ("server_instance_id", "episode_id", "attempt_id", "execution_generation")} | {
                "command": kind, "request_id": request_id, "control_token": token}

        def applied(request_id):
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                result = client.get(f"/v1/commands/{request_id}").json()
                if result["status"] != "accepted":
                    assert result["status"] == "applied", result
                    return result
                time.sleep(.005)
            raise AssertionError("command did not apply")

        idle_stop = client.post("/v1/commands", json=body("stop", "idle-stop")).json()
        assert idle_stop["no_op"] and idle_stop["attempt_id"] is None
        assert idle_stop["execution_generation"] == 0
        start = body("start", "start-1")
        assert client.post("/v1/commands", json={**start, "control_token": "invalid"}).status_code == 403
        assert client.post("/v1/commands", json=start).json()["status"] == "accepted"
        started = applied("start-1")
        assert len(runtime.starts) == 1
        assert client.post("/v1/commands", json=start).json() == started
        assert client.post("/v1/commands", json={**start, "command": "stop"}).status_code == 409
        duplicate_start = client.post("/v1/commands", json=body("start", "start-running")).json()
        assert duplicate_start["no_op"] and len(runtime.starts) == 1
        assert client.post("/v1/commands", json=body("reset", "reset-running")).status_code == 409

        camera = {"type": "free", "lookat_m": [.45, 0, .15], "distance_m": .8, "azimuth_deg": 45, "elevation_deg": -25}
        views = []
        for name in ("one", "two"):
            result = client.post("/v1/views", json={"server_instance_id": identity, "view_id": name, "camera": camera,
                                                  "width": 64, "height": 64, "requested_fps": 30})
            assert result.status_code == 201, result.text
            views.append(result.json())
        assert client.post("/v1/views", json={"server_instance_id": identity, "view_id": "three", "camera": camera}).status_code == 409
        assert all("view_token" not in v for v in client.get("/v1/views").json()["views"])
        snapshot_body = {"server_instance_id": identity, "view_ids": ["one", "two"], "encoding": "png"}
        response = client.post("/v1/snapshots", json=snapshot_body)
        assert response.status_code == 200, response.text[:300]
        frame = unpack(response.content)
        assert len(frame["views"]) == 2 and frame["execution_generation"] == started["execution_generation"]
        assert {v["encoding"] for v in frame["views"]} == {"png"}
        assert client.post("/v1/snapshots", json={**snapshot_body, "view_ids": ["missing"]}).status_code == 404

        mutation = {"server_instance_id": identity, "view_token": views[0]["view_token"], "camera_revision": 1,
                    "camera": {**camera, "azimuth_deg": 125}}
        assert client.patch("/v1/views/one", json={**mutation, "view_token": "wrong"}).status_code == 403
        updated = client.patch("/v1/views/one", json=mutation)
        assert updated.status_code == 200 and updated.json()["camera_revision"] == 2
        assert client.patch("/v1/views/one", json=mutation).status_code == 409
        assert client.patch("/v1/views/one", json={**mutation, "camera_revision": 2, "width": 128}).status_code == 409
        revised_frame = unpack(client.post("/v1/snapshots", json=snapshot_body).content)
        assert revised_frame["views"][0]["camera_revision"] == 2
        assert revised_frame["views"][1]["camera_revision"] == 1
        with client.websocket_connect("/v1/frames") as socket:
            socket.send_json({**snapshot_body, "encoding": "jpeg"})
            streamed = unpack(socket.receive_bytes())
            assert len(streamed["views"]) == 2
            assert streamed["views"][0]["camera_revision"] == 2

        # A blocked renderer must not delay accepting/applying a robot hold.
        runtime.render_entered.clear()
        runtime.render_release.clear()
        assert runtime.render_entered.wait(1)
        before = client.get("/v1/status").json()["sim_time_s"]
        stop = body("stop", "stop-1")
        accepted = client.post("/v1/commands", json=stop).json()
        assert accepted["status"] == "accepted"
        assert client.get("/v1/status").json()["execution_state"] == "STOPPING"
        assert client.post("/v1/commands", json=body("start", "start-too-soon")).status_code == 409
        held = applied("stop-1")
        assert runtime.hold_applied and held["execution_state"] == "STOPPED"
        assert client.get("/v1/status").json()["sim_time_s"] > before
        runtime.render_release.set()
        runtime.callbacks[0](None)  # Obsolete provider completion cannot finish this stopped attempt.
        assert client.get("/v1/status").json()["execution_state"] == "STOPPED"
        assert client.post("/v1/commands", json=start).json() == started  # Lost-ack retry is stable.
        assert client.post("/v1/commands", json={**stop, "request_id": "stale-new"}).status_code == 409
        assert client.post("/v1/commands", json=body("start", "resume")).status_code == 202
        resumed = applied("resume")
        assert resumed["attempt_id"] == started["attempt_id"] and len(runtime.starts) == 2
        assert client.post("/v1/commands", json=body("stop", "stop-2")).status_code == 202
        applied("stop-2")
        assert client.post("/v1/commands", json=body("reset", "reset-1")).status_code == 202
        reset = applied("reset-1")
        assert reset["episode_id"] != started["episode_id"] and reset["attempt_id"] is None
        assert reset["execution_state"] == "IDLE"
        runtime.callbacks[1]("late provider error")
        assert client.get("/v1/status").json()["execution_state"] == "IDLE"
        assert client.post("/v1/snapshots", json={**snapshot_body, "server_instance_id": "prior-service"}).status_code == 409

        assert client.request("DELETE", "/v1/control-session", json={"server_instance_id": identity, "control_token": token}).status_code == 200
        assert client.get("/v1/status").json()["execution_state"] == "IDLE"
        assert client.post("/v1/commands", json=body("start", "old-owner")).status_code == 403
        new_lease = client.post("/v1/control-session", json=claim).json()
        assert new_lease["control_token"] != token
        service = app.state.simulation
        with service._lock:
            service._lease["lease_expires_at_server_monotonic_ns"] = 0
        active_lease = client.post("/v1/control-session", json=claim)
        assert active_lease.status_code == 200
        token = active_lease.json()["control_token"]
        assert client.get("/v1/status").json()["execution_state"] == "IDLE"
        assert client.post("/v1/commands", json=body("start", "finish-attempt")).status_code == 202
        finish_start = applied("finish-attempt")
        runtime.generation += 1
        runtime.hold_applied = True
        runtime.callbacks[-1](None)
        finished = client.get("/v1/status").json()
        assert finished["execution_state"] == "FINISHED"
        assert finished["execution_generation"] == finish_start["execution_generation"] + 1
        assert client.post("/v1/commands", json=body("start", "illegal-finished-resume")).status_code == 409
        assert client.post("/v1/commands", json=body("stop", "finished-stop")).json()["no_op"]
        assert client.post("/v1/commands", json=body("reset", "finish-reset")).status_code == 202
        applied("finish-reset")
        assert client.post("/v1/commands", json=body("start", "error-attempt")).status_code == 202
        applied("error-attempt")
        runtime.generation += 1
        runtime.hold_applied = True
        runtime.callbacks[-1]("provider failed")
        assert client.get("/v1/status").json()["execution_state"] == "ERROR"
        assert client.get("/v1/status").json()["service_state"] == "READY"
        assert client.post("/v1/commands", json=body("reset", "error-reset")).status_code == 202
        applied("error-reset")
        runtime.physics_healthy = False
        fatal = client.get("/v1/status").json()
        assert fatal["execution_state"] == fatal["service_state"] == "ERROR"
        assert client.get("/v1/status").json()["execution_generation"] == fatal["execution_generation"]
        assert client.post("/v1/commands", json=body("reset", "fatal-reset")).status_code == 409
        with service._lock:
            for i in range(520):
                service._event("test_event", i=i)
        gap, events = service.events(0)
        assert gap and len(events) == 512
        assert [e["event_seq"] for e in events] == sorted(e["event_seq"] for e in events)
        assert runtime.render_thread_names == {"monitor-renderer"}
    assert runtime.closed
    print("PASS: service ownership, idempotency, cancellation, continuing physics, reset, camera revisions, atomic PNG/WS bundles, and event gaps")


if __name__ == "__main__":
    run()
