# /// script
# requires-python = ">=3.13"
# dependencies = ["httpx>=0.28,<1", "opencv-python>=4.11,<6"]
# ///
"""Standalone 30 fps episode monitor. Run with uv run --script run.py --help."""

import argparse
import asyncio
import base64
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import threading
import time

import cv2
import httpx

HERE = Path(__file__).resolve().parent
SPECIALISTS = ("success", "drop")
VERDICTS = ("POSITIVE", "NEGATIVE", "UNCERTAIN")
SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}


def load_config(path=HERE / "config.json", base_url=None):
    config = json.loads(Path(path).read_text())
    if base_url:
        config["base_url"] = base_url.rstrip("/")
    n = config["evidence_horizon"]
    if type(n) is not int or n < 1:
        raise ValueError("evidence_horizon must be a positive integer")
    for name in ("horizon_duration", "request_timeout", "episode_timeout"):
        if type(config[name]) not in (int, float) or not 0 < config[name] < float("inf"):
            raise ValueError(f"{name} must be finite and positive")
    if config["fps"] != 30:
        raise ValueError("This experiment simulates a 30 fps camera")
    if n > config["fps"] * config["horizon_duration"]:
        raise ValueError("Evidence sampling cannot exceed the camera's 30 fps")
    for specialist in SPECIALISTS:
        threshold = config["evidence_threshold"][specialist]
        if type(threshold) is not int or not 0 <= threshold < n:
            raise ValueError("Each evidence_threshold must be an integer in [0, N)")
        x1, y1, x2, y2 = config["crops"][specialist]
        if not all(type(x) is int for x in (x1, y1, x2, y2)) or not (0 <= x1 < x2 and 0 <= y1 < y2):
            raise ValueError(f"Invalid {specialist} crop")
    return config


@dataclass(frozen=True)
class Frame:
    image: object
    sequence: int
    source_index: int
    observed_at: float


class SimulatedCamera:
    """Publish a fresh buffer entry every 1/30 s, including after video EOF."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        self.buffer = deque(maxlen=31)
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.thread = None
        self.started_at = 0.0

    def start(self):
        self.started_at = time.monotonic()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        if not self.ready.wait(10) or self.error:
            self.stop()
            raise RuntimeError(self.error or "Camera did not publish its first frame")

    def _run(self):
        sequence, source_index = 0, -1
        image, ended = None, False
        try:
            while not self.stopped.is_set():
                deadline = self.started_at + sequence / 30
                if self.stopped.wait(max(0, deadline - time.monotonic())):
                    break
                if not ended:
                    ok, decoded = self.cap.read()
                    if ok:
                        image = decoded
                        source_index += 1
                    else:
                        ended = True
                if image is None:
                    raise ValueError("Video contains no readable frames")
                frame = Frame(image, sequence, source_index, time.monotonic())
                with self.lock:
                    self.buffer.append(frame)
                self.ready.set()
                sequence += 1
        except Exception as exc:
            self.error = str(exc)
            self.ready.set()

    def get(self):
        if self.error:
            raise RuntimeError(self.error)
        with self.lock:
            return self.buffer[-1]

    def stop(self):
        self.stopped.set()
        if self.thread:
            self.thread.join()
        self.cap.release()


def image_content(image, crop):
    x1, y1, x2, y2 = crop
    height, width = image.shape[:2]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Crop {crop} exceeds image dimensions {width}x{height}")
    # Preserve the native crop dimensions; the application never resizes VLM input.
    ok, encoded = cv2.imencode(".jpg", image[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("JPEG encoding failed")
    url = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
    return {"type": "image_url", "image_url": {"url": url}}


class Backend:
    def __init__(self, config, client):
        self.config, self.client = config, client
        self.prompts = {s: (HERE / "prompts" / f"{s}.md").read_text() for s in SPECIALISTS}

    async def pair(self, frame, started_at, log, *, normal, round_id=None, slot=None):
        return await asyncio.gather(*(
            self.ask(s, frame, started_at, log, normal=normal, round_id=round_id, slot=slot)
            for s in SPECIALISTS
        ))

    async def ask(self, specialist, frame, started_at, log, **tags):
        observation = {
            "type": "observation", "specialist": specialist,
            "sequence": frame.sequence, "source_frame": frame.source_index,
            "source_time_s": frame.source_index / 30,
            "captured_time_s": frame.observed_at - started_at,
            **tags,
        }
        sent = time.monotonic()
        try:
            response = await self.client.post(
                self.config["base_url"].rstrip("/") + "/chat/completions",
                json={
                    "model": self.config["model"], "temperature": 0,
                    "max_tokens": self.config["max_tokens"],
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [
                        {"role": "system", "content": self.prompts[specialist]},
                        {"role": "user", "content": [
                            {"type": "text", "text": "Current camera observation:"},
                            image_content(frame.image, self.config["crops"][specialist]),
                        ]},
                    ],
                    "response_format": {"type": "json_schema", "json_schema": {
                        "name": "visual_observation", "strict": True, "schema": SCHEMA,
                    }},
                },
                timeout=self.config["request_timeout"],
            )
            response.raise_for_status()
            body = response.json()
            message = body["choices"][0]["message"]
            observation["raw"] = message.get("content")
            if message.get("reasoning") or message.get("reasoning_content"):
                raise ValueError("Endpoint emitted reasoning despite enable_thinking=False")
            parsed = json.loads(message["content"])
            if parsed.get("verdict") not in VERDICTS or not isinstance(parsed.get("reason"), str):
                raise ValueError("Invalid specialist verdict/reason")
            observation.update(verdict=parsed["verdict"], reason=parsed["reason"])
            observation["usage"] = body.get("usage", {})
        except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError) as exc:
            observation.update(verdict="UNCERTAIN", reason="Request unavailable", error=str(exc)[:500])
        observation.update(
            sent_time_s=sent - started_at,
            received_time_s=time.monotonic() - started_at,
            latency_s=time.monotonic() - sent,
        )
        log(observation)
        return observation


def decide(observations, config):
    expected = {(i, s) for i in range(config["evidence_horizon"]) for s in SPECIALISTS}
    keys = [(o["slot"], o["specialist"]) for o in observations]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Verification must contain exactly one response per slot and specialist")
    counts = {s: sum(o["specialist"] == s and o["verdict"] == "POSITIVE" for o in observations) for s in SPECIALISTS}
    success = counts["success"] > config["evidence_threshold"]["success"]
    drop = counts["drop"] > config["evidence_threshold"]["drop"]
    status = "SUCCESS" if success and not drop else "FAIL" if drop and not success else "ON_GOING"
    return status, counts


async def verify(camera, backend, config, first_pair, started_at, triggered_at, round_id, log):
    tasks = [first_pair]
    try:
        for slot in range(1, config["evidence_horizon"]):
            target = triggered_at + slot * config["horizon_duration"] / config["evidence_horizon"]
            await asyncio.sleep(max(0, target - time.monotonic()))
            tasks.append(asyncio.create_task(backend.pair(
                camera.get(), started_at, log, normal=False, round_id=round_id, slot=slot,
            )))
        return [o for pair in await asyncio.gather(*tasks) for o in pair]
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def monitor(camera, backend, config, log):
    started_at = camera.started_at
    normal = asyncio.create_task(backend.pair(camera.get(), started_at, log, normal=True))
    verification = None
    round_id, rearm_after = 0, -1.0
    consecutive_errors = 0
    result = {"status": "ON_GOING", "failure_type": None, "failure_description": None}
    log({"type": "status", **result, "decision_time_s": 0.0})
    try:
        async with asyncio.timeout(config["episode_timeout"]):
            while True:
                pending = [normal] + ([verification] if verification else [])
                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                if verification in done:
                    observations = verification.result()
                    status, counts = decide(observations, config)
                    log({"type": "verification_finished", "round_id": round_id,
                         "counts": counts, "status": status,
                         "time_s": time.monotonic() - started_at})
                    verification = None
                    rearm_after = time.monotonic() - started_at
                    if status != "ON_GOING":
                        result.update(status=status, decision_time_s=rearm_after,
                                      evidence_counts=counts, verification_round=round_id)
                        if status == "FAIL":
                            result.update(failure_type="DROP_FAILURE",
                                          failure_description="The plug is visibly dropped outside the gripper, stand, and socket.")
                        result["evidence_source_time_s"] = max(o["source_time_s"] for o in observations)
                        log({"type": "status", **result})
                        return result
                if normal in done:
                    pair = normal.result()
                    consecutive_errors = consecutive_errors + 1 if all("error" in o for o in pair) else 0
                    if consecutive_errors >= 5:
                        raise RuntimeError("Five consecutive normal batches failed; check the model endpoint")
                    trigger = (
                        verification is None
                        and pair[0]["captured_time_s"] >= rearm_after
                        and any(o["verdict"] == "POSITIVE" for o in pair)
                    )
                    frame = camera.get()
                    if trigger:
                        round_id += 1
                    triggered_at = time.monotonic()
                    # This next normal pair is also verification slot 0 when triggered.
                    normal = asyncio.create_task(backend.pair(
                        frame, started_at, log, normal=True,
                        round_id=round_id if trigger else None, slot=0 if trigger else None,
                    ))
                    if trigger:
                        log({"type": "verification_started", "round_id": round_id,
                             "time_s": triggered_at - started_at,
                             "trigger_source_time_s": pair[0]["source_time_s"]})
                        verification = asyncio.create_task(verify(
                            camera, backend, config, normal, started_at, triggered_at, round_id, log,
                        ))
    except TimeoutError:
        result.update(decision_time_s=time.monotonic() - started_at, stop_reason="episode_timeout")
        log({"type": "status", **result})
        return result
    finally:
        tasks = [normal] + ([verification] if verification else [])
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def summary(rows):
    return {
        "episodes": len(rows),
        "correct": sum(r["status"] == r["expected_status"] for r in rows),
        "success_correct": sum(r["status"] == r["expected_status"] == "SUCCESS" for r in rows),
        "drop_detected": sum(r["status"] == r["expected_status"] == "FAIL" for r in rows),
        "false_failures": sum(r["status"] == "FAIL" and r["expected_status"] == "SUCCESS" for r in rows),
        "false_successes": sum(r["status"] == "SUCCESS" and r["expected_status"] == "FAIL" for r in rows),
        "unresolved": sum(r["status"] == "ON_GOING" for r in rows),
        "request_errors": sum(r["request_errors"] for r in rows),
        "requests": sum(r["requests"] for r in rows),
        "results": rows,
    }


async def run(args):
    config = load_config(args.config, args.base_url)
    root = args.videos.resolve()
    annotations = json.loads((root / "annotation.json").read_text())
    if len({a["video"] for a in annotations}) != len(annotations):
        raise ValueError("Duplicate episode annotations")
    if args.episodes:
        selected = {f"traj-{n}.MOV" for n in args.episodes}
        annotations = [a for a in annotations if a["video"] in selected]
        if selected != {a["video"] for a in annotations}:
            raise ValueError("An episode is missing from annotation.json")
    for annotation in annotations:
        path = (root / annotation["video"]).resolve()
        if not path.is_relative_to(root) or not path.is_file() or annotation["status"] not in ("SUCCESS", "FAIL"):
            raise ValueError(f"Invalid episode annotation: {annotation}")
    output = args.output or HERE / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    config["source_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(output / "run_config.json", config)
    (output / "run.py").write_bytes(Path(__file__).read_bytes())
    (output / "prompts").mkdir()
    for path in (HERE / "prompts").glob("*.md"):
        (output / "prompts" / path.name).write_bytes(path.read_bytes())
    rows = []
    limits = httpx.Limits(max_connections=24, max_keepalive_connections=24)
    async with httpx.AsyncClient(limits=limits, trust_env=False) as client:
        response = await client.get(config["base_url"].rstrip("/") + "/models", timeout=10)
        response.raise_for_status()
        if config["model"] not in {m["id"] for m in response.json()["data"]}:
            raise ValueError("Configured model not available at endpoint")
        backend = Backend(config, client)
        for annotation in annotations:
            observations = []
            with (output / (Path(annotation["video"]).stem + ".jsonl")).open("w", buffering=1) as stream:
                def log(event):
                    stream.write(json.dumps(event) + "\n")
                    if event["type"] == "observation":
                        observations.append(event)
                camera = SimulatedCamera(root / annotation["video"])
                try:
                    await asyncio.to_thread(camera.start)
                    result = await monitor(camera, backend, config, log)
                finally:
                    camera.stop()
            latencies = [o["latency_s"] for o in observations]
            normal_times = sorted({o["sent_time_s"] for o in observations if o["normal"] and o["specialist"] == "success"})
            result.update(video=annotation["video"], expected_status=annotation["status"],
                          requests=len(observations), request_errors=sum("error" in o for o in observations),
                          request_latency_median_s=statistics.median(latencies) if latencies else None,
                          request_latency_p95_s=sorted(latencies)[min(len(latencies)-1, int(.95*len(latencies)))] if latencies else None,
                          normal_hz=(len(normal_times)-1)/(normal_times[-1]-normal_times[0]) if len(normal_times)>1 else None)
            rows.append(result)
            write_json(output / "summary.json", summary(rows))
            print(json.dumps(result), flush=True)
    print(f"Results: {output / 'summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True, help="Episode directory containing annotation.json")
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--base-url", help="Override the OpenAI-compatible endpoint")
    parser.add_argument("--episodes", type=int, nargs="+", help="Episode numbers; omit to run all")
    parser.add_argument("--output", type=Path, help="New output directory; existing directories are never overwritten")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
