"""Experiment 12 specialists with bounded verification and production event logging."""

import asyncio
import base64
from collections import deque
from dataclasses import dataclass, field
import json
import itertools
from pathlib import Path
import threading
import time

import cv2
import httpx

from .endpoints import cloud_model, model_info, model_headers, native_ollama, official_openai

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

    def current_content(self, frame):
        """UI input assembly only; the legacy request remains identical."""
        if "input_blocks" not in self.config:
            return [{"type": "text", "text": "Current camera observation:"}, image_content(frame.image, self.crop)]
        content = []
        for block in self.config["input_blocks"]:
            if "content" in block:
                content.extend(block["content"])
            else:
                image = frame.views[block["source"]].image
                crop = block["crop"] or [0, 0, image.shape[1], image.shape[0]]
                content.extend([{"type": "text", "text": block["alias"]}, image_content(image, crop)])
        return content

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
        if hasattr(frame, "views"):
            observation["source_frames"] = {
                name: {"source_frame": view.source_index, "captured_time_s": view.observed_at - started_at}
                for name, view in frame.views.items()
            }
        if getattr(frame, "metadata", None):
            observation["evidence"] = {"context": frame.metadata["context"],
                                       "identity": frame.metadata["identity"],
                                       "observed_at": frame.observed_at,
                                       "simulation": frame.metadata["simulation"]}
        sent = time.monotonic()
        log({**observation, "type": "request_started", "sent_time_s": sent - started_at})
        try:
            response = await backend.complete(self, frame)
            observation["http_status"] = response.status_code
            observation["queue_wait_s"] = response.extensions.get("model_queue_wait_s", 0.)
            observation["response_text"] = response.text
            response.raise_for_status()
            body = response.json()
            observation["response"] = body
            message = body["message"] if backend.native else body["choices"][0]["message"]
            if not isinstance(message, dict):
                raise ValueError("Invalid assistant message")
            observation["raw"] = message.get("content")
            if message.get("reasoning") or message.get("reasoning_content") or message.get("thinking"):
                raise ValueError("Endpoint emitted reasoning despite thinking being disabled")
            if backend.native and (body.get("done") is not True or body.get("done_reason") == "length"):
                raise ValueError("Ollama response is incomplete or reached the output token limit")
            observation.update(self.parse(message["content"]))
            observation["usage"] = ({"prompt_tokens": body.get("prompt_eval_count", 0),
                                     "completion_tokens": body.get("eval_count", 0)}
                                    if backend.native else body.get("usage", {}))
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
    config = json.loads(json.dumps(path)) if isinstance(path, dict) else json.loads(Path(path).read_text())
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
    if type(n) is not int or n < 0:
        raise ValueError("evidence_horizon must be a nonnegative integer (0 disables verification)")
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
        if type(threshold) is not int or threshold < 0:
            raise ValueError("Each evidence_threshold must be a nonnegative integer")
        if n and threshold >= n:
            raise ValueError("Each evidence_threshold must be less than evidence_horizon when verification is enabled")
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
    metadata: dict | None = field(default=None, kw_only=True)


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
    def __init__(self, config, client, *, specialists=None, request_gate=None, audit=None):
        self.config, self.client = config, client
        self.audit = audit
        self.native = native_ollama(config["base_url"])
        default_limit = min(3, config.get("max_connections", 36)) if self.native else config.get("max_connections", 36)
        limit = config.get("max_in_flight", default_limit)
        if type(limit) is not int or limit < 1:
            raise ValueError("max_in_flight must be a positive integer")
        self.request_gate = request_gate if request_gate is not None else asyncio.Semaphore(limit)
        self.model, self.info = config["model"], {}
        self.request_ids = itertools.count(1)
        if specialists is not None:
            self.specialists = specialists
            return
        self.specialists = {}
        for name in config["enabled_specialists"]:
            settings = {**config, **config["specialists"][name], "name": name,
                        "crop": config["crops"][name], "threshold": config["evidence_threshold"][name],
                        "reference_images": config.get("references", {}).get(name, [])}
            self.specialists[name] = Specialist((HERE / "prompts" / f"{name}.md").read_text(), settings)

    async def check_model(self):
        self.info = await model_info(self.client, self.config["base_url"], self.config["model"],
                                     self.config["request_timeout"], require_vision=True)
        self.model = self.info["model"]
        return self.info

    async def complete(self, specialist, frame):
        queued_at = time.monotonic()
        async with self.request_gate:
            waited = time.monotonic() - queued_at
            response = await self._complete(specialist, frame)
            response.extensions["model_queue_wait_s"] = waited
            return response

    async def _complete(self, specialist, frame):
        content = [*specialist.references, *specialist.current_content(frame)]
        base_url = self.config["base_url"].rstrip("/")
        if self.native:
            text, images = [], []
            for part in content:
                if part["type"] == "text":
                    text.append(part["text"])
                else:
                    header, separator, encoded = part["image_url"]["url"].partition(",")
                    if not separator or not header.startswith("data:image/") or not header.endswith(";base64"):
                        raise ValueError("Ollama requires embedded base64 images")
                    images.append(encoded)
                    text.append(f"[Image {len(images)}]")
            payload = {"model": self.model, "stream": False,
                       "messages": [{"role": "system", "content": specialist.prompt},
                                    {"role": "user", "content": "\n\n".join(text), "images": images}],
                       "options": {"temperature": 0, "num_predict": self.config["max_tokens"]}}
            if self.info.get("thinking", True):
                payload["think"] = False
            if cloud_model(base_url, self.model):
                # Cloud cannot enforce a schema; preserve the same output contract in the request and validate locally.
                payload["messages"][-1]["content"] += "\n\nReturn only JSON matching this response schema, without Markdown:\n" + json.dumps(specialist.schema)
            else:
                payload["format"] = specialist.schema
            path = "/chat"
        else:
            payload = {"model": self.model, "temperature": 0, "max_tokens": self.config["max_tokens"],
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "system", "content": specialist.prompt}, {"role": "user", "content": content}],
                       "response_format": {"type": "json_schema", "json_schema": {
                           "name": "visual_observation", "strict": True, "schema": specialist.schema}}}
            if official_openai(base_url):
                payload.pop("chat_template_kwargs", None)
                payload["max_completion_tokens"] = payload.pop("max_tokens")
                effort = self.config.get("reasoning_effort")
                if effort is not None:
                    payload["reasoning_effort"] = effort
                    if effort != "none":
                        payload.pop("temperature", None)
            path = "/chat/completions"
        call_id = None
        started = time.monotonic()
        try:
            if self.audit:
                from uuid import uuid4
                call_id = uuid4().hex
                # Progress supplies a history; retain every evidence identity.
                frames = [item[1] for item in frame] if isinstance(frame, (list, tuple)) else [frame]
                evidence = [getattr(item, 'metadata', None) for item in frames]
                begin = asyncio.create_task(asyncio.to_thread(self.audit.begin, payload,
                    specialist=getattr(specialist, 'name', 'progress'), evidence=evidence, call_id=call_id))
                try:
                    await asyncio.shield(begin)
                except asyncio.CancelledError:
                    await begin
                    raise
            response = await self.client.post(base_url + path, json=payload, headers=model_headers(base_url),
                                              timeout=self.config["request_timeout"])
            if self.audit:
                await asyncio.to_thread(self.audit.event, {'event': 'response_received', 'call_id': call_id,
                    'http_status': response.status_code, 'duration_s': time.monotonic() - started,
                    'response': response.text, 'provider_request_id': response.headers.get('x-request-id')})
            return response
        except BaseException as exc:
            if self.audit:
                await asyncio.shield(asyncio.to_thread(self.audit.event, {
                    'event': 'request_cancelled' if isinstance(exc, asyncio.CancelledError) else 'request_failed',
                    'call_id': call_id, 'duration_s': time.monotonic() - started, 'error': str(exc)}))
            raise

    async def batch(self, frame, started_at, log, *, normal, round_id=None, slot=None):
        return await asyncio.gather(*(
            specialist.observe(frame, self, started_at, log, normal=normal, round_id=round_id, slot=slot)
            for specialist in self.specialists.values()
        ))


def decide(observations, config, specialists):
    n = config["evidence_horizon"]
    expected = {(i, name) for i in (range(n) if n else (None,)) for name in specialists}
    keys = [(o.get("slot"), o["specialist"]) for o in observations]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Decision requires one response per expected slot and specialist")
    counts = {name: specialist.evidence_count(observations) for name, specialist in specialists.items()}
    confirmed = [s for s in specialists.values() if (s.confirmed(observations) if n else counts[s.name] > 0)]
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


async def input_batch(camera, backend, started_at, log, **tags):
    try:
        frame = camera.get()
        if getattr(camera, "mode", None) == "mujoco" and tags.get("normal"):
            while frame.metadata["identity"] == getattr(camera, "_last_normal_identity", None):
                await asyncio.sleep(.01)
                frame = camera.get()
            camera._last_normal_identity = frame.metadata["identity"]
    except RuntimeError as exc:
        if getattr(camera, "mode", None) != "mujoco":
            raise
        # Preserve complete verification slots while the remote stream recovers.
        await asyncio.sleep(.05)
        now = time.monotonic() - started_at
        observations = [{"type": "observation", "specialist": name, "verdict": "UNCERTAIN",
                         "reason": "Simulation image unavailable", "source_unavailable": str(exc),
                         "source_frame": -1, "source_time_s": now, "captured_time_s": now,
                         "received_time_s": now, "latency_s": 0., **tags}
                        for name in backend.specialists]
        for observation in observations:
            log({**observation, "type": "source_unavailable"})
        return observations
    return await backend.batch(frame, started_at, log, **tags)


def current_observations(camera, observations):
    """Reject stale/repeated remote evidence without changing local video semantics."""
    if getattr(camera, "mode", None) != "mujoco":
        return observations
    seen, valid = set(), []
    for observation in observations:
        evidence = observation.get("evidence")
        identity = (observation["specialist"], evidence["identity"]) if evidence else None
        accepted = evidence and camera.accepts_evidence(evidence) and identity not in seen
        if identity:
            seen.add(identity)
        valid.append(observation if accepted else {
            **observation, "verdict": "UNCERTAIN", "evidence_rejected": True,
            "reason": "Stale, repeated, or obsolete simulation observation"})
    return valid


def attach_evidence(result, observations):
    evidence = [o["evidence"] for o in observations if o.get("evidence") and not o.get("evidence_rejected")]
    if evidence:
        result["evidence"] = max(evidence, key=lambda e: e["observed_at"])
    else:
        result.pop("evidence", None)


async def verify(camera, backend, config, first_batch, started_at, triggered_at, round_id, log):
    tasks = [first_batch]
    try:
        for slot in range(1, config["evidence_horizon"]):
            target = triggered_at + slot * config["horizon_duration"] / config["evidence_horizon"]
            await asyncio.sleep(max(0, target - time.monotonic()))
            tasks.append(asyncio.create_task(input_batch(
                camera, backend, started_at, log, normal=False, round_id=round_id, slot=slot,
            )))
        return [o for batch in await asyncio.gather(*tasks) for o in batch]
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def monitor(camera, backend, config, log, stop_event=None, *, continuous=False):
    """Continuous sessions re-arm visual events; batch episodes stop on a decision."""
    started_at = camera.started_at
    normal = asyncio.create_task(input_batch(camera, backend, started_at, log, normal=True))
    verification = None
    evidence_context = camera.evidence_context() if hasattr(camera, "evidence_context") else None
    round_id, rearm_after = 0, -1.0
    confirmed = set()
    consecutive_errors = 0
    result = {"status": "ON_GOING", "failure_type": None, "failure_description": None,
              "source_frame": -1}
    try:
        log({"type": "status", **result, "decision_time_s": 0.0})
        async with asyncio.timeout(None if continuous else config["episode_timeout"]):
            while True:
                if stop_event is not None and stop_event.is_set():
                    result.update(decision_time_s=time.monotonic() - started_at, stop_reason="user_stop")
                    log({"type": "status", **result})
                    return result
                context = camera.evidence_context() if hasattr(camera, "evidence_context") else None
                if context != evidence_context:
                    old = [normal] + ([verification] if verification else [])
                    for job in old:
                        job.cancel()
                    await asyncio.gather(*old, return_exceptions=True)
                    if verification:
                        log({"type": "verification_finished", "round_id": round_id, "cancelled": True,
                             "counts": {}, "status": "ON_GOING", "time_s": time.monotonic() - started_at})
                    evidence_context = context
                    verification = None
                    confirmed.clear()
                    rearm_after = -1.
                    result = {"status": "ON_GOING", "failure_type": None, "source_frame": -1,
                              "decision_time_s": time.monotonic() - started_at}
                    log({"type": "simulation_context_changed", "context": context, "time_s": result["decision_time_s"]})
                    # A run owner can synchronously end this recording on episode reset.
                    # Do not dispatch a new-scene request into the previous run's audit.
                    if stop_event is not None and stop_event.is_set():
                        result["stop_reason"] = "simulation_episode_changed"
                        log({"type": "status", **result})
                        return result
                    log({"type": "status", **result})
                    normal = asyncio.create_task(input_batch(camera, backend, started_at, log, normal=True))
                if result.get("evidence") and hasattr(camera, "accepts_evidence") and not camera.accepts_evidence(result["evidence"]):
                    confirmed.clear()
                    result = {"status": "ON_GOING", "failure_type": None, "source_frame": -1,
                              "evidence_unavailable": True, "decision_time_s": time.monotonic() - started_at}
                    log({"type": "status", **result})
                pending = [normal] + ([verification] if verification else [])
                done, _ = await asyncio.wait(pending, timeout=.05 if stop_event is not None else None,
                                            return_when=asyncio.FIRST_COMPLETED)
                if verification in done:
                    observations = current_observations(camera, verification.result())
                    status, counts, failures = decide(observations, config, backend.specialists)
                    log({"type": "verification_finished", "round_id": round_id,
                         "counts": counts, "status": status,
                         "time_s": time.monotonic() - started_at})
                    verification = None
                    rearm_after = time.monotonic() - started_at
                    if status != "ON_GOING" or continuous:
                        result.update(status=status, decision_time_s=rearm_after,
                                      evidence_counts=counts, verification_round=round_id,
                                      failure_type=None, failure_description=None)
                        result.pop("confirmed_failure_types", None)
                        confirmed = ({name for name, count in counts.items()
                                      if count > backend.specialists[name].threshold}
                                     if status != "ON_GOING" else set())
                        if status == "FAIL":
                            primary = failures[0]
                            positives = [o for o in observations if o["specialist"] == primary.name and o["verdict"] == "POSITIVE"]
                            result.update(failure_type=primary.failure_type,
                                          confirmed_failure_types=[s.failure_type for s in failures],
                                          failure_description=max(positives, key=lambda o: o["source_time_s"])["reason"])
                        result["evidence_source_time_s"] = max(o["source_time_s"] for o in observations)
                        result["source_frame"] = round(result["evidence_source_time_s"] * 30)
                        attach_evidence(result, observations)
                        log({"type": "status", **result})
                        if not continuous:
                            return result
                if normal in done:
                    batch = current_observations(camera, normal.result())
                    consecutive_errors = consecutive_errors + 1 if all("error" in o for o in batch) else 0
                    if batch and all(o.get("http_status") == 429 for o in batch):
                        # Do not resubmit failed observations. Back off before the next
                        # fresh normal sample, allowing a provider's pending work to drain.
                        delay = min(8., 2. ** max(0, consecutive_errors - 1))
                        if stop_event is not None:
                            await asyncio.to_thread(stop_event.wait, delay)
                            if stop_event.is_set():
                                continue
                        else:
                            await asyncio.sleep(delay)
                    if consecutive_errors >= 5:
                        raise RuntimeError("Five consecutive normal batches failed; check the model endpoint")
                    if config["evidence_horizon"] == 0:
                        status, counts, failures = decide(batch, config, backend.specialists)
                        evidence_time = max(o["source_time_s"] for o in batch)
                        result = {"status": status, "failure_type": None, "failure_description": None,
                                  "decision_time_s": time.monotonic() - started_at, "evidence_counts": counts,
                                  "evidence_source_time_s": evidence_time, "source_frame": round(evidence_time * 30)}
                        if status == "FAIL":
                            primary = failures[0]
                            result.update(failure_type=primary.failure_type,
                                          confirmed_failure_types=[s.failure_type for s in failures],
                                          failure_description=next(o["reason"] for o in batch if o["specialist"] == primary.name))
                        attach_evidence(result, batch)
                        log({"type": "status", **result})
                        if not continuous and status != "ON_GOING":
                            return result
                        normal = asyncio.create_task(input_batch(camera, backend, started_at, log, normal=True))
                        continue
                    fresh = batch[0]["captured_time_s"] >= rearm_after
                    positive_specialists = {o["specialist"] for o in batch if o["verdict"] == "POSITIVE"}
                    # A fresh normal observation can clear a displayed event;
                    # only a complete verification round can confirm a new one.
                    if continuous and confirmed and fresh and not confirmed.issubset(positive_specialists):
                        confirmed.clear()
                        result = {"status": "ON_GOING", "failure_type": None,
                                  "failure_description": None, "source_frame": -1}
                    if result["status"] == "ON_GOING":
                        result["source_frame"] = round(batch[0]["source_time_s"] * 30)
                        log({"type": "status", **result, "decision_time_s": time.monotonic() - started_at})
                    trigger = (
                        verification is None
                        and fresh
                        and bool(positive_specialists - confirmed if continuous else positive_specialists)
                    )
                    if trigger:
                        round_id += 1
                    triggered_at = time.monotonic()
                    # This next normal batch is also verification slot 0 when triggered.
                    normal = asyncio.create_task(input_batch(
                        camera, backend, started_at, log, normal=True,
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
