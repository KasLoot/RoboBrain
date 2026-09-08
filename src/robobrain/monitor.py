"""Experiment 6 monitor: native crops, paired specialists and bounded verification."""

import asyncio
import base64
from collections import deque
from dataclasses import dataclass
import json
import itertools
from pathlib import Path
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
DROP_LOCATIONS = ("STAND", "GRIPPER", "SOCKET", "TABLE", "FALLING", "UNKNOWN")
DROP_SCHEMA = {
    "type": "object",
    "properties": {
        "location": {"type": "string", "enum": list(DROP_LOCATIONS)},
        "reason": {"type": "string"},
    },
    "required": ["location", "reason"],
    "additionalProperties": False,
}


def parse_observation(specialist, content):
    parsed = json.loads(content)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("reason"), str):
        raise ValueError("Invalid specialist observation")
    expected = {"location", "reason"} if specialist == "drop" else {"verdict", "reason"}
    if set(parsed) != expected:
        raise ValueError("Unexpected fields in specialist observation")
    if specialist == "drop":
        location = parsed.get("location")
        if location not in DROP_LOCATIONS:
            raise ValueError("Invalid plug location")
        verdict = "POSITIVE" if location in ("TABLE", "FALLING") else "UNCERTAIN" if location == "UNKNOWN" else "NEGATIVE"
        return {**parsed, "verdict": verdict}
    if parsed.get("verdict") not in VERDICTS:
        raise ValueError("Invalid specialist verdict")
    return parsed


def load_config(path=HERE / "config.json", base_url=None):
    config = json.loads(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    if base_url:
        config["base_url"] = base_url.rstrip("/")
    if not isinstance(config["model"], str) or not config["model"].strip():
        raise ValueError("model must be a nonempty string")
    if type(config["max_tokens"]) is not int or config["max_tokens"] < 1:
        raise ValueError("max_tokens must be a positive integer")
    try:
        endpoint = httpx.URL(config["base_url"])
    except (TypeError, httpx.InvalidURL) as exc:
        raise ValueError("base_url must be a valid HTTP(S) URL") from exc
    if endpoint.scheme not in ("http", "https") or not endpoint.host:
        raise ValueError("base_url must be a valid HTTP(S) URL")
    n = config["evidence_horizon"]
    if type(n) is not int or n < 1:
        raise ValueError("evidence_horizon must be a positive integer")
    for name in ("horizon_duration", "request_timeout", "episode_timeout"):
        if type(config[name]) not in (int, float) or not 0 < config[name] < float("inf"):
            raise ValueError(f"{name} must be finite and positive")
    if config["fps"] != 30:
        raise ValueError("The camera simulator runs at 30 fps")
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
        self.ended = threading.Event()
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
                        self.ended.set()
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
        self.request_ids = itertools.count(1)
        self.prompts = {s: (HERE / "prompts" / f"{s}.md").read_text() for s in SPECIALISTS}
        self.references = {s: [] for s in SPECIALISTS}
        for specialist, references in config.get("references", {}).items():
            for reference in references:
                path = (HERE / "references" / reference["file"]).resolve()
                if not path.is_relative_to(HERE / "references"):
                    raise ValueError("Reference image must be inside the package references directory")
                self.references[specialist].extend([
                    {"type": "text", "text": reference["label"]},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")}},
                ])

    async def pair(self, frame, started_at, log, *, normal, round_id=None, slot=None):
        return await asyncio.gather(*(
            self.ask(s, frame, started_at, log, normal=normal, round_id=round_id, slot=slot)
            for s in SPECIALISTS
        ))

    async def ask(self, specialist, frame, started_at, log, **tags):
        observation = {
            "type": "observation", "specialist": specialist,
            "request_id": next(self.request_ids),
            "sequence": frame.sequence, "source_frame": frame.source_index,
            "source_time_s": frame.source_index / 30,
            "captured_time_s": frame.observed_at - started_at,
            **tags,
        }
        sent = time.monotonic()
        log({**observation, "type": "request_started", "sent_time_s": sent - started_at})
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
                            *self.references[specialist],
                            {"type": "text", "text": "Current camera observation:"},
                            image_content(frame.image, self.config["crops"][specialist]),
                        ]},
                    ],
                    "response_format": {"type": "json_schema", "json_schema": {
                        "name": "visual_observation", "strict": True,
                        "schema": DROP_SCHEMA if specialist == "drop" else SCHEMA,
                    }},
                },
                timeout=self.config["request_timeout"],
            )
            observation["http_status"] = response.status_code
            observation["response_text"] = response.text
            response.raise_for_status()
            body = response.json()
            observation["response"] = body
            message = body["choices"][0]["message"]
            if not isinstance(message, dict):
                raise ValueError("Endpoint returned an invalid assistant message")
            observation["raw"] = message.get("content")
            if message.get("reasoning") or message.get("reasoning_content"):
                raise ValueError("Endpoint emitted reasoning despite enable_thinking=False")
            observation.update(parse_observation(specialist, message["content"]))
            observation["usage"] = body.get("usage", {})
        except asyncio.CancelledError:
            observation.update(verdict="UNCERTAIN", reason="Request cancelled", cancelled=True)
            raise
        except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError) as exc:
            detail = str(exc)
            if isinstance(exc, httpx.HTTPStatusError):
                detail += ": " + exc.response.text[:800]
            observation.update(verdict="UNCERTAIN", reason="Request unavailable", error=detail[:1200])
        finally:
            received = time.monotonic()
            observation.update(sent_time_s=sent - started_at,
                               received_time_s=received - started_at, latency_s=received - sent)
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


async def monitor(camera, backend, config, log, stop_event=None):
    started_at = camera.started_at
    normal = asyncio.create_task(backend.pair(camera.get(), started_at, log, normal=True))
    verification = None
    round_id, rearm_after = 0, -1.0
    consecutive_errors = 0
    result = {"status": "ON_GOING", "failure_type": None, "failure_description": None,
              "source_frame": -1}
    try:
        log({"type": "status", **result, "decision_time_s": 0.0})
        async with asyncio.timeout(config["episode_timeout"]):
            while True:
                if stop_event is not None and stop_event.is_set():
                    result.update(decision_time_s=time.monotonic() - started_at, stop_reason="user_stop")
                    log({"type": "status", **result})
                    return result
                pending = [normal] + ([verification] if verification else [])
                done, _ = await asyncio.wait(pending, timeout=.05 if stop_event is not None else None,
                                            return_when=asyncio.FIRST_COMPLETED)
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
                        result["source_frame"] = round(result["evidence_source_time_s"] * 30)
                        log({"type": "status", **result})
                        return result
                if normal in done:
                    pair = normal.result()
                    consecutive_errors = consecutive_errors + 1 if all("error" in o for o in pair) else 0
                    if consecutive_errors >= 5:
                        raise RuntimeError("Five consecutive normal batches failed; check the model endpoint")
                    result["source_frame"] = round(pair[0]["source_time_s"] * 30)
                    log({"type": "status", **result, "decision_time_s": time.monotonic() - started_at})
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
