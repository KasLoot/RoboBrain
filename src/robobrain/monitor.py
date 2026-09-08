"""Experiment 12 specialists with bounded verification and production event logging."""

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
VERDICTS = ("POSITIVE", "NEGATIVE", "UNCERTAIN")


class Specialist:
    """One standard observation and evidence workflow, configured by visual task."""
    def __init__(self, system_prompt, config):
        self.prompt, self.config = system_prompt, config
        self.name, self.crop = config["name"], config["crop"]
        self.field, self.values = config["field"], config["values"]
        self.failure_type = config["failure_type"]
        self.threshold = config["threshold"]
        if not isinstance(self.field, str) or not self.field or self.field == "reason":
            raise ValueError("Invalid specialist observation field")
        if (not isinstance(self.values, dict) or not self.values
                or not all(isinstance(k, str) and isinstance(v, str) and v in VERDICTS
                           for k, v in self.values.items())):
            raise ValueError("Invalid observation-to-vote mapping")
        if self.failure_type is not None and (not isinstance(self.failure_type, str) or not self.failure_type):
            raise ValueError("failure_type must be null for success or a nonempty failure name")
        self.schema = {"type": "object", "properties": {
            self.field: {"type": "string", "enum": list(self.values)},
            "reason": {"type": "string"}}, "required": [self.field, "reason"],
            "additionalProperties": False}
        self.references = []
        for reference in config.get("reference_images", []):
            path = (HERE / "references" / reference["file"]).resolve()
            if not path.is_relative_to(HERE / "references"):
                raise ValueError("Reference must be inside the package references directory")
            self.references.extend([
                {"type": "text", "text": reference["label"]},
                {"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")}},
            ])

    def parse(self, content):
        parsed = json.loads(content)
        if not isinstance(parsed, dict) or set(parsed) != {self.field, "reason"}:
            raise ValueError("Invalid specialist response fields")
        if (not isinstance(parsed["reason"], str) or not isinstance(parsed[self.field], str)
                or parsed[self.field] not in self.values):
            raise ValueError("Invalid specialist observation")
        return {**parsed, "verdict": self.values[parsed[self.field]]}

    def evidence_count(self, observations):
        return sum(o["specialist"] == self.name and o["verdict"] == "POSITIVE" for o in observations)

    def confirmed(self, observations):
        return self.evidence_count(observations) > self.threshold

    async def observe(self, frame, backend, started_at, log, **tags):
        specialist = self.name
        observation = {
            "type": "observation", "specialist": specialist,
            "request_id": next(backend.request_ids),
            "sequence": frame.sequence, "source_frame": frame.source_index,
            "source_time_s": frame.source_index / 30,
            "captured_time_s": frame.observed_at - started_at,
            **tags,
        }
        sent = time.monotonic()
        log({**observation, "type": "request_started", "sent_time_s": sent - started_at})
        try:
            response = await backend.client.post(
                self.config["base_url"].rstrip("/") + "/chat/completions",
                json={
                    "model": self.config["model"], "temperature": 0,
                    "max_tokens": self.config["max_tokens"],
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [
                        {"role": "system", "content": self.prompt},
                        {"role": "user", "content": [
                            *self.references,
                            {"type": "text", "text": "Current camera observation:"},
                            image_content(frame.image, self.crop),
                        ]},
                    ],
                    "response_format": {"type": "json_schema", "json_schema": {
                        "name": "visual_observation", "strict": True,
                        "schema": self.schema,
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
                raise ValueError("Invalid assistant message")
            observation["raw"] = message.get("content")
            if message.get("reasoning") or message.get("reasoning_content"):
                raise ValueError("Endpoint emitted reasoning despite enable_thinking=False")
            observation.update(self.parse(message["content"]))
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


def load_config(path=HERE / "config.json", base_url=None):
    config = json.loads(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    if base_url:
        config["base_url"] = base_url.rstrip("/")
    if not isinstance(config["model"], str) or not config["model"].strip():
        raise ValueError("model must be a nonempty string")
    for name in ("max_tokens", "max_connections"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
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
    definitions = config.get("specialists")
    if not isinstance(definitions, dict) or not definitions:
        raise ValueError("Configuration must define its specialists; use the current config.json as a template")
    if any(not isinstance(name, str) or not name.isidentifier() or not isinstance(spec, dict)
           for name, spec in definitions.items()):
        raise ValueError("Each specialist must have an identifier name and a settings object")
    config["enabled_specialists"] = list(definitions)
    for specialist in config["enabled_specialists"]:
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
    """Shared transport; specialists never own independent verification schedulers."""
    def __init__(self, config, client):
        self.config, self.client = config, client
        self.request_ids = itertools.count(1)
        self.specialists = {}
        for name in config["enabled_specialists"]:
            settings = {**config, **config["specialists"][name], "name": name,
                        "crop": config["crops"][name], "threshold": config["evidence_threshold"][name],
                        "reference_images": config.get("references", {}).get(name, [])}
            self.specialists[name] = Specialist((HERE / "prompts" / f"{name}.md").read_text(), settings)

    async def batch(self, frame, started_at, log, *, normal, round_id=None, slot=None):
        return await asyncio.gather(*(
            specialist.observe(frame, self, started_at, log, normal=normal, round_id=round_id, slot=slot)
            for specialist in self.specialists.values()
        ))


def decide(observations, config, specialists):
    expected = {(i, name) for i in range(config["evidence_horizon"]) for name in specialists}
    keys = [(o["slot"], o["specialist"]) for o in observations]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Verification requires one response per slot and specialist")
    counts = {name: specialist.evidence_count(observations) for name, specialist in specialists.items()}
    confirmed = [s for s in specialists.values() if s.confirmed(observations)]
    successes = [s for s in confirmed if s.failure_type is None]
    failures = [s for s in confirmed if s.failure_type is not None]
    status = "SUCCESS" if successes and not failures else "FAIL" if failures and not successes else "ON_GOING"
    return status, counts, failures


class ReplayController:
    """Log proposed VA hand-offs once per attempt. No robot APIs are called."""
    def __init__(self):
        self.finished = set()

    def start(self, attempt, log):
        for action in ("SCENE_READY", "START_INSERTION_VA"):
            log({"type": "control_action", "attempt_id": attempt, "action": action, "mode": "SIMULATED"})

    def finish(self, attempt, result, log):
        if result["status"] == "ON_GOING" or attempt in self.finished:
            return
        self.finished.add(attempt)
        actions = ["STOP_INSERTION_VA"]
        if result["status"] == "FAIL":
            actions += ["RELEASE_GRIPPER", "START_RECOVERY_VA"]
        actions += ["RESET_SCENE"]
        for action in actions:
            log({"type": "control_action", "attempt_id": attempt, "action": action,
                 "idempotency_key": f"{attempt}:{action}", "mode": "SIMULATED",
                 "status": result["status"], "failure_type": result.get("failure_type")})


async def verify(camera, backend, config, first_batch, started_at, triggered_at, round_id, log):
    tasks = [first_batch]
    try:
        for slot in range(1, config["evidence_horizon"]):
            target = triggered_at + slot * config["horizon_duration"] / config["evidence_horizon"]
            await asyncio.sleep(max(0, target - time.monotonic()))
            tasks.append(asyncio.create_task(backend.batch(
                camera.get(), started_at, log, normal=False, round_id=round_id, slot=slot,
            )))
        return [o for batch in await asyncio.gather(*tasks) for o in batch]
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def monitor(camera, backend, config, log, stop_event=None):
    started_at = camera.started_at
    normal = asyncio.create_task(backend.batch(camera.get(), started_at, log, normal=True))
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
                    status, counts, failures = decide(observations, config, backend.specialists)
                    log({"type": "verification_finished", "round_id": round_id,
                         "counts": counts, "status": status,
                         "time_s": time.monotonic() - started_at})
                    verification = None
                    rearm_after = time.monotonic() - started_at
                    if status != "ON_GOING":
                        result.update(status=status, decision_time_s=rearm_after,
                                      evidence_counts=counts, verification_round=round_id)
                        if status == "FAIL":
                            primary = failures[0]
                            positives = [o for o in observations if o["specialist"] == primary.name and o["verdict"] == "POSITIVE"]
                            result.update(failure_type=primary.failure_type,
                                          confirmed_failure_types=[s.failure_type for s in failures],
                                          failure_description=max(positives, key=lambda o: o["source_time_s"])["reason"])
                        result["evidence_source_time_s"] = max(o["source_time_s"] for o in observations)
                        result["source_frame"] = round(result["evidence_source_time_s"] * 30)
                        log({"type": "status", **result})
                        return result
                if normal in done:
                    batch = normal.result()
                    consecutive_errors = consecutive_errors + 1 if all("error" in o for o in batch) else 0
                    if consecutive_errors >= 5:
                        raise RuntimeError("Five consecutive normal batches failed; check the model endpoint")
                    result["source_frame"] = round(batch[0]["source_time_s"] * 30)
                    log({"type": "status", **result, "decision_time_s": time.monotonic() - started_at})
                    trigger = (
                        verification is None
                        and batch[0]["captured_time_s"] >= rearm_after
                        and any(o["verdict"] == "POSITIVE" for o in batch)
                    )
                    frame = camera.get()
                    if trigger:
                        round_id += 1
                    triggered_at = time.monotonic()
                    # This next normal batch is also verification slot 0 when triggered.
                    normal = asyncio.create_task(backend.batch(
                        frame, started_at, log, normal=True,
                        round_id=round_id if trigger else None, slot=0 if trigger else None,
                    ))
                    if trigger:
                        log({"type": "verification_started", "round_id": round_id,
                             "time_s": triggered_at - started_at,
                             "trigger_source_time_s": batch[0]["source_time_s"]})
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
