"""Real HTTP/WebSocket + MuJoCo integration smoke, without provider calls.

MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 .venv-integration/bin/python -B tests/check_simulation_real.py --record
Recording artifacts are kept under this workspace's .tmp/simulation-real-* directory.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["TMPDIR"] = str(ROOT / ".tmp")
os.environ["XDG_CACHE_HOME"] = str(ROOT / ".cache")
os.environ["MESA_SHADER_CACHE_DIR"] = str(ROOT / ".cache" / "mesa")
os.environ["__GL_SHADER_DISK_CACHE_PATH"] = str(ROOT / ".cache" / "nvidia")
sys.path[:0] = [str(ROOT / "simulator"), str(ROOT / "src")]

import uvicorn
from core.config import load_yaml
from core.sim.service_runtime import ServiceRuntime
from show_harness_mujoco.service import create_app
from robobrain.simulation import SimulationConnection, decode_bundle


def run(record=False):
    config = load_yaml(ROOT / "simulator/configs/robot_mujoco_plug.yaml")
    artifacts = ROOT / ".tmp" / ("simulation-real-" + uuid4().hex[:10])
    artifacts.mkdir(parents=True)
    config["log_dir"] = str(artifacts)
    config["run_root"] = str(artifacts / "runs")
    config.setdefault("recording", {})["fps"] = 10
    runtime = ServiceRuntime(config=config, no_controller=True, record=record, record_global=record)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    app = create_app(runtime)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, name="real-simulation-http", daemon=True)
    thread.start()
    connection = SimulationConnection({"base_url": f"http://127.0.0.1:{port}", "stale_after_s": 5})
    measurements = {"record": record, "record_global": record}
    brain_recorder = None
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(.01)
        assert server.started, "HTTP service failed to start"
        connection.refresh_status()
        camera = {"type": "free", "lookat_m": [.45, 0, .15], "distance_m": .8, "azimuth_deg": 45, "elevation_deg": -25}
        sources = [dict(id=name, camera={**camera, "azimuth_deg": azimuth}, width=640, height=480, fps=15)
                   for name, azimuth in (("camera1", 45), ("camera2", 135))]
        for source in sources:
            connection.add_source(source)

        def bundle():
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                try:
                    result = connection.bundle(["camera1", "camera2"])
                    assert result["camera1"].metadata["simulation"]["state_seq"] == result["camera2"].metadata["simulation"]["state_seq"]
                    return result
                except RuntimeError:
                    time.sleep(.02)
            raise AssertionError(connection.error or "No current coherent two-view bundle")

        first = bundle()
        original_run = connection.refresh_status()['recording_run']['run_id']
        if record:
            import shutil
            from robobrain.sources import Inputs
            from robobrain.simulation import SimulationReader
            from robobrain.presentation import LiveView
            from robobrain.recording import SimulationRecorder
            run_info = connection.recording_run()
            inputs = Inputs({s['id']: SimulationReader(connection, s) for s in sources}, 'mujoco')
            brain_recorder = SimulationRecorder(inputs, sources,
                Path(run_info['directory']) / 'robobrain-video' / 'combined.mp4',
                LiveView(), shutil.which('ffmpeg'), threading.Event())
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline and brain_recorder.frames < 3 and not brain_recorder.error:
                time.sleep(.1)
            assert not brain_recorder.error, brain_recorder.error
            assert brain_recorder.frames >= 3
            packet = connection.request('POST', '/v1/snapshots', {
                'server_instance_id': connection.status['server_instance_id'],
                'view_ids': [connection.sources['camera1']['remote']['view_id']],
                'encoding': 'png', 'include_global': True}).content
            global_meta, global_images = decode_bundle(packet)
            assert global_images['__global__'].shape == (1080, 1920, 3)
            assert global_meta['views'][-1]['camera'] == {'fixed': 'lab_overview'}
        first_context = connection.context(["camera1", "camera2"])
        assert first["camera1"].image.shape == (480, 640, 3)
        assert not (first["camera1"].image == first["camera2"].image).all()
        provenance = first["camera1"].metadata["simulation"]
        measurements["first_capture_to_receive_ms"] = (time.monotonic() - first["camera1"].observed_at) * 1000
        connection.update_view("camera1", {**camera, "azimuth_deg": 80})
        revised = bundle()
        assert revised["camera1"].metadata["camera_revision"] == 2
        assert revised["camera2"].metadata["camera_revision"] == 1
        assert connection.context(["camera1", "camera2"]) != first_context
        png, reference = connection.reference("camera1")
        assert png.startswith(b"\x89PNG") and reference["views"][0]["camera_revision"] == 2
        (artifacts / "inspection-reference.png").write_bytes(png)

        def command(kind):
            result = connection.command(kind)
            assert result["status"] == "applied", result
            measurements.setdefault("command_latency_ms", []).append({"command": kind,
                "acceptance_to_applied": (result["applied_at_server_monotonic_ns"]-result["accepted_at_server_monotonic_ns"]) / 1e6})
            return result

        started = command("start")
        assert started["execution_state"] == "RUNNING"
        bundle()
        stopped = command("stop")
        assert stopped["execution_state"] == "STOPPED"
        stopped_time = connection.refresh_status()["sim_time_s"]
        time.sleep(.15)
        assert connection.refresh_status()["sim_time_s"] > stopped_time
        bundle()
        resumed = command("start")
        assert resumed["attempt_id"] == started["attempt_id"]
        command("stop")
        assert connection.refresh_status()["recording_run"]["run_id"] == original_run
        reset = command("reset")
        assert connection.refresh_status()["recording_run"]["run_id"] != original_run
        assert reset["episode_id"] != provenance["episode_id"] and reset["attempt_id"] is None
        after_reset = bundle()
        assert after_reset["camera1"].metadata["simulation"]["episode_id"] == reset["episode_id"]
        if record:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and not brain_recorder.stop_reason and not brain_recorder.error:
                time.sleep(.1)
            assert brain_recorder.stop_reason == 'scene_changed', brain_recorder.error
            measurements['robobrain_recording'] = brain_recorder.finish(True)
            brain_recorder = None
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                state = connection.refresh_status()
                if state.get("recording_error"):
                    raise AssertionError(state["recording_error"])
                if state.get("recorded_frames", 0) >= 3:
                    break
                time.sleep(.1)
            assert state["recorded_frames"] >= 3
        measurements["status"] = connection.refresh_status()
        assert measurements["status"]["renderer_error"] is None
        assert measurements["status"]["physics_healthy"]
        # Closing monitoring sources never changes the independent robot execution state.
        command("start")
        for source in sources:
            connection.remove_source(source["id"])
        assert connection.refresh_status()["execution_state"] == "RUNNING"
        command("stop")
    finally:
        if brain_recorder:
            brain_recorder.finish(True)
        connection.close()
        server.should_exit = True
        thread.join(15)
        assert not thread.is_alive(), "HTTP server did not close"
    if record:
        directories = sorted((artifacts / "runs").glob("*/show-harness-video"))
        assert len(directories) == 2, directories
        videos = sorted(p.name for p in directories[0].glob("*.mp4"))
        assert {"front.mp4", "side.mp4", "wrist.mp4", "wrist_insert.mp4", "global.mp4", "combined.mp4"}.issubset(videos), videos
        rows = [json.loads(line) for line in (directories[0] / "frames.jsonl").read_text().splitlines()]
        assert len(rows) >= 3
        assert all("state_seq" in row and "execution_generation" in row and row.get("episode_id") and row.get("server_instance_id") for row in rows)
        import cv2
        capture = cv2.VideoCapture(str(directories[0] / "global.mp4"))
        try:
            assert capture.isOpened()
            assert (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (1920, 1080)
            assert capture.read()[0], "Global recording contains no decodable image"
        finally:
            capture.release()
        rb_dir = artifacts / 'runs' / original_run / 'robobrain-video'
        rb_rows = [json.loads(line) for line in (rb_dir / 'frames.jsonl').read_text().splitlines()]
        assert all(row['simulation']['episode_id'] == provenance['episode_id'] for row in rb_rows)
        counts = []
        for path in rb_dir.glob('*.mp4'):
            cap = cv2.VideoCapture(str(path))
            try:
                assert cap.read()[0], path
                counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            finally:
                cap.release()
        assert len(counts) == 4 and len(set(counts)) == 1 and counts[0] == len(rb_rows), counts
        measurements['robobrain_video_counts'] = counts
        measurements["recorded_videos"] = videos
        measurements["recorded_rows"] = len(rows)
    (artifacts / "measurements.json").write_text(json.dumps(measurements, indent=2))
    print("PASS: real MuJoCo with RoboBrain HTTP/WebSocket transport, coherent dual cameras, PNG reference, camera revision, start/stop/resume/reset, independent monitoring, and recording" if record else "PASS: real MuJoCo and RoboBrain HTTP/WebSocket transport")
    print("Artifacts:", artifacts)
    print(json.dumps({k: v for k, v in measurements.items() if k != "status"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true", help="Enable original camera and Full HD global recordings")
    run(parser.parse_args().record)
