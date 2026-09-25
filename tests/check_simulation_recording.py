"""Offline recording layout and synchronized FFmpeg smoke checks; no VLM calls."""

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace

import numpy as np

from robobrain.presentation import LiveView
from robobrain.recording import SimulationLayout, SimulationRecorder


def crop(packed, rectangle):
    x, y, width, height = rectangle
    return packed[y:y + height, x:x + width]


def wait_until(predicate, seconds=20):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.02)
    raise AssertionError("Timed out waiting for recording output")


def layout_checks():
    cameras = {"camera1": np.full((241, 321, 3), (40, 70, 100), np.uint8),
               "camera2": np.full((480, 640, 3), (160, 30, 70), np.uint8)}
    global_image = np.full((1080, 1920, 3), (210, 190, 160), np.uint8)
    layout = SimulationLayout(cameras)
    packed = layout.pack(cameras, global_image, {"status": "FAIL", "failure_type": "DROP"})
    assert all(dimension % 2 == 0 for rectangle in layout.crops.values() for dimension in rectangle[2:])
    assert crop(packed, layout.crops["combined"]).shape == (1080, layout.left_width + 1920, 3)
    global_output = crop(packed, layout.crops["global"])
    assert global_output.shape == global_image.shape
    assert np.array_equal(global_output[:700], global_image[:700]), "Status belongs in the bottom-right corner"
    assert np.any(global_output[800:, 1200:] != global_image[800:, 1200:]), "Global output must show monitoring status"
    assert np.array_equal(global_image[800:, 1200:], np.full((280, 720, 3), (210, 190, 160), np.uint8)), "Never annotate an evidence buffer"
    assert np.array_equal(crop(packed, layout.crops["view/camera1"])[:241, :321], cameras["camera1"])
    assert np.array_equal(crop(packed, layout.crops["view/camera2"]), cameras["camera2"])
    # The two individual views are stacked on the LEFT; the global view is on the RIGHT.
    assert np.array_equal(packed[270, layout.left_width // 2], cameras["camera1"][100, 100])
    assert np.array_equal(packed[810, layout.left_width // 2], cameras["camera2"][100, 100])
    assert np.array_equal(packed[270, layout.left_width + 100], global_image[270, 100])
    # Valid task source IDs may coincide with output role names.
    collision = SimulationLayout({"global": cameras["camera1"]})
    assert "view/global" in collision.crops and "global" in collision.crops
    try:
        layout.pack(cameras, global_image[:720], {"status":"ON_GOING"})
    except ValueError as exc:
        assert "1920" in str(exc)
    else:
        raise AssertionError("A resized global camera was accepted")


class Inputs:
    def __init__(self):
        self.started_at = time.monotonic()
        self.episode = "episode-1"
        self.unavailable = False
        self.counter = 0
        self.global_image = np.full((1080, 1920, 3), (100, 125, 160), np.uint8)
        self.images = {"camera1": np.full((240, 320, 3), (35, 130, 20), np.uint8),
                       "camera2": np.full((240, 320, 3), (130, 30, 190), np.uint8)}

    def get(self):
        if self.unavailable:
            raise RuntimeError("Test stream is stale")
        self.counter += 1
        metadata = {"simulation":{"server_instance_id":"service-1", "episode_id":self.episode,
                                  "state_seq":self.counter, "sim_time_s":self.counter / 100}}
        views = {source_id: SimpleNamespace(image=image, source_index=self.counter, observed_at=time.monotonic(),
                                           metadata={"identity":[self.episode,self.counter], "camera_revision":1})
                 for source_id, image in self.images.items()}
        return SimpleNamespace(views=views, global_image=self.global_image, metadata=metadata)


def encoder_checks(directory):
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        print("FFmpeg smoke check skipped: ffmpeg/ffprobe are not installed")
        return
    inputs, view, stopped = Inputs(), LiveView(), threading.Event()
    sources = [{"id":"camera1","name":"First view"},{"id":"camera2","name":"Second view"}]
    events = []
    destination = directory / "session" / "combined.mp4"
    recorder = SimulationRecorder(inputs, sources, destination, view, ffmpeg, stopped, events.append)
    try:
        wait_until(lambda: recorder.frames >= 2 or recorder.error)
        assert recorder.error is None, recorder.error
        with view.lock:
            view.status = {"status":"FAIL","failure_type":"DROP"}
        inputs.unavailable = True
        wait_until(lambda: recorder.unavailable_frames >= 1 or recorder.error)
        assert recorder.error is None, recorder.error
        inputs.unavailable = False
        previous = recorder.frames
        wait_until(lambda: recorder.frames >= previous + 1 or recorder.error)
        inputs.episode = "episode-2"
        wait_until(stopped.is_set)
    finally:
        info = recorder.finish(True)
    assert recorder.error is None, recorder.error
    assert info["stop_reason"] == "scene_changed"
    assert info["global_width"] == 1920 and info["global_height"] == 1080
    assert info["unavailable_frames"] >= 1
    assert set(info["files"]) == {"combined.mp4", "global.mp4", "view-camera1.mp4", "view-camera2.mp4", "frames.jsonl"}
    rows = [json.loads(line) for line in (destination.parent / "frames.jsonl").read_text().splitlines()]
    assert len(rows) == info["frames"]
    assert all(row["simulation"]["episode_id"] == "episode-1" for row in rows), "Do not mix reset scenes in one recording"
    assert all(set(row["sources"]) == {"camera1","camera2"} for row in rows)
    assert [row["frame"] for row in rows] == list(range(info["frames"]))
    assert all(row["video_time_s"] == row["frame"] / 30 for row in rows)
    assert any(row["unavailable"] and row["status"]["status"] == "FAIL" for row in rows)
    assert all(b["sample_slot"] > a["sample_slot"] for a, b in zip(rows, rows[1:]))
    assert sum(row["skipped_samples"] for row in rows) == info["omitted_samples"]
    counts = []
    for name in info["files"]:
        if not name.endswith(".mp4"):
            continue
        result = json.loads(subprocess.check_output([ffprobe,"-v","error","-show_streams","-of","json",str(destination.parent/name)]))
        stream = result["streams"][0]
        assert stream["r_frame_rate"] == "30/1"
        counts.append(int(stream["nb_frames"]))
        expected = (1920,1080) if name == "global.mp4" else (info["width"],1080) if name == "combined.mp4" else (320,240)
        assert (stream["width"],stream["height"]) == expected
    assert counts == [info["frames"]] * 4, "All videos must have exactly the same timeline"
    assert any(event["type"] == "recording_stopped" for event in events)
    before = destination.read_bytes()
    collision = SimulationRecorder(Inputs(), sources, destination, LiveView(), ffmpeg, threading.Event())
    wait_until(lambda: collision.error)
    try:
        collision.finish(False)
    except RuntimeError as exc:
        assert "already contains" in str(exc)
    else:
        raise AssertionError("Recorder overwrote an existing session")
    assert destination.read_bytes() == before
    locked = directory / "locked"
    locked.mkdir()
    lock = locked / ".recording.lock"
    lock.write_text("another recorder owns this")
    collision = SimulationRecorder(Inputs(), sources, locked / "combined.mp4", LiveView(), ffmpeg, threading.Event())
    wait_until(lambda: collision.error)
    try:
        collision.finish(False)
    except RuntimeError:
        pass
    assert lock.read_text() == "another recorder owns this", "A failed contender must not remove the owner's lock"
    print(f"Encoder smoke passed: four synchronized 30 fps videos, full-HD global, metadata, gaps and reset boundary ({info['frames']} frames).")


if __name__ == "__main__":
    layout_checks()
    scratch = Path(__file__).resolve().parents[1] / ".tmp"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as root:
        encoder_checks(Path(root))
    print("Simulation recording checks passed.")
