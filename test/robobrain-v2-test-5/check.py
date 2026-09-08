# /// script
# requires-python = ">=3.13"
# dependencies = ["httpx>=0.28,<1", "opencv-python>=4.11,<6"]
# ///
"""Small executable checks: uv run --script check.py. No endpoint needed."""
import asyncio
import base64
import json
from pathlib import Path
import tempfile
import time

import cv2
import httpx
import numpy as np

from run import Backend, Frame, SimulatedCamera, SPECIALISTS, decide, image_content, load_config, monitor, parse_observation


def evidence(success, drop):
    return [{"slot": i, "specialist": s, "verdict": "POSITIVE" if i < {"success": success, "drop": drop}[s] else "UNCERTAIN"}
            for i in range(10) for s in SPECIALISTS]


async def checks():
    config = load_config()
    for location, expected in [("STAND", "NEGATIVE"), ("GRIPPER", "NEGATIVE"), ("SOCKET", "NEGATIVE"), ("TABLE", "POSITIVE"), ("FALLING", "POSITIVE"), ("UNKNOWN", "UNCERTAIN")]:
        assert parse_observation("drop", json.dumps({"location": location, "reason": "visible relation"}))["verdict"] == expected
    assert decide(evidence(5, 0), config)[0] == "ON_GOING"
    assert decide(evidence(6, 0), config)[0] == "SUCCESS"
    assert decide(evidence(0, 6), config)[0] == "FAIL"
    assert decide(evidence(6, 6), config)[0] == "ON_GOING"
    for invalid in [evidence(6, 0)[:-1], evidence(6, 0) + [evidence(6, 0)[0]]]:
        try:
            decide(invalid, config)
        except ValueError:
            pass
        else:
            raise AssertionError("Missing/duplicate observations accepted")

    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for crop in config["crops"].values():
        content = image_content(image, crop)
        data = base64.b64decode(content["image_url"]["url"].split(",")[1])
        decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape[:2] == (crop[3] - crop[1], crop[2] - crop[0])

    async def handler(request):
        payload = json.loads(request.content)
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert payload["temperature"] == 0
        return httpx.Response(200, json={"choices": [{"message": {"content": "not JSON"}}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        backend = Backend(config, client)
        observations = await backend.pair(Frame(image, 0, 0, time.monotonic()), time.monotonic(), lambda _: None, normal=True)
        assert all(o["verdict"] == "UNCERTAIN" and "error" in o for o in observations)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "camera.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
        assert writer.isOpened()
        writer.write(np.full((48, 64, 3), 30, np.uint8))
        writer.write(np.full((48, 64, 3), 180, np.uint8))
        writer.release()
        camera = SimulatedCamera(path)
        camera.start()
        try:
            await asyncio.sleep(.14)
            before = camera.get()
            await asyncio.sleep(.2)
            after = camera.get()
            assert before.source_index == after.source_index == 1
            assert 5 <= after.sequence - before.sequence <= 7
            assert np.array_equal(before.image, after.image)
        finally:
            camera.stop()

    class FakeCamera:
        def __init__(self):
            self.started_at = time.monotonic()
        def get(self):
            now = time.monotonic()
            index = int((now - self.started_at) * 30)
            return Frame(None, index, index, now)

    class FakeBackend:
        def __init__(self, conflict_first=False):
            self.active = 0
            self.conflict_first = conflict_first
        async def pair(self, frame, started_at, log, **tags):
            self.active += 1
            try:
                # Out-of-order completion; the final verification pair is slowest.
                await asyncio.sleep(.18 if tags.get("slot") == 9 else .04)
                rows = []
                for specialist in SPECIALISTS:
                    row = {"type": "observation", "specialist": specialist, "source_time_s": frame.source_index / 30,
                           "captured_time_s": frame.observed_at - started_at, **tags,
                           "verdict": "POSITIVE" if specialist == "success" or (self.conflict_first and tags.get("round_id") == 1) else "NEGATIVE"}
                    log(row)
                    rows.append(row)
                return rows
            finally:
                self.active -= 1

    events = []
    backend = FakeBackend()
    config["horizon_duration"] = .4
    result = await monitor(FakeCamera(), backend, config, events.append)
    assert result["status"] == "SUCCESS" and result["evidence_counts"] == {"success": 10, "drop": 0}
    assert sum(e["type"] == "verification_started" for e in events) == 1
    verified = [e for e in events if e["type"] == "observation" and e.get("round_id") == 1]
    assert len(verified) == 20
    assert any(e["type"] == "observation" and e.get("normal") and e.get("round_id") is None for e in events[3:])
    assert backend.active == 0
    assert result["decision_time_s"] >= .04 + .36 + .18
    events = []
    backend = FakeBackend(conflict_first=True)
    result = await monitor(FakeCamera(), backend, config, events.append)
    starts = [e for e in events if e["type"] == "verification_started"]
    finishes = [e for e in events if e["type"] == "verification_finished"]
    assert len(starts) == len(finishes) == 2 and result["status"] == "SUCCESS"
    assert finishes[0]["status"] == "ON_GOING"
    assert starts[1]["time_s"] > finishes[0]["time_s"] + .039
    assert backend.active == 0
    backend = FakeBackend()
    config["episode_timeout"] = .12
    result = await monitor(FakeCamera(), backend, config, lambda _: None)
    assert result["status"] == "ON_GOING" and result["stop_reason"] == "episode_timeout"
    assert backend.active == 0
    print("Checks passed: thresholds, conflicts, unique evidence, native crops, no thinking, malformed replies, 30 fps EOF continuation, one verification round, normal monitoring, complete response wait, cleanup.")


if __name__ == "__main__":
    asyncio.run(checks())
