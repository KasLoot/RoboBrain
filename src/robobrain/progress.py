"""Independent predefined-node progress observations. Never supplies outcome votes."""

import asyncio
import base64
from collections import deque
import json
import time

import cv2
import httpx

from .monitor import Backend, Specialist, image_content
from .presentation import progress_snapshot
from .workspace import ProgressDefinition


class ProgressObserver:
    def __init__(self, definition, store, views):
        self.definition = definition
        self.references = []
        cameras, prompts = [], []
        for block in definition.blocks:
            if not block.enabled:
                continue
            if block.type == "text":
                if block.text.strip():
                    prompts.append(block.text.strip())
            elif block.type == "image":
                if not block.asset:
                    raise ValueError("Progress reference image is missing")
                path = store.asset_path(block.asset)
                if block.crop:
                    content = image_content(cv2.imread(str(path)), block.crop)
                else:
                    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
                    content = {"type": "image_url", "image_url": {"url": f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")}}
                self.references.extend([{"type": "text", "text": block.alias}, content])
            elif block.type == "camera":
                if block.source not in views:
                    raise ValueError("Connect and enable the progress source: " + block.source)
                height, width = views[block.source].image.shape[:2]
                crop = block.crop or [0, 0, width, height]
                x1, y1, x2, y2 = crop
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError(f"Progress crop {crop} does not fit {width}x{height}")
                cameras.append(block.model_dump())
        self.sources = list(dict.fromkeys(b["source"] for b in cameras))
        self.config = {"input_blocks": cameras}
        self.schema = {"type": "object", "properties": {
            "reason": {"type": "string"},
            "node_id": {"type": "string", "enum": [n.id for n in definition.nodes] + ["UNKNOWN"]}},
            "required": ["reason", "node_id"], "additionalProperties": False}
        self.prompt = "\n\n".join([
            "You track task progress by selecting one predefined node. You do not decide SUCCESS or FAIL.",
            *prompts,
            "Progress nodes and their observable criteria:\n" + json.dumps(
                [{"id": n.id, "label": n.label, "criteria": n.criteria} for n in definition.nodes], ensure_ascii=False),
            "Select a node only when ALL its criteria are supported by the current visual evidence. "
            "Compare every node: an unfinished task can match a partially completed node; "
            "it does not automatically match the initial node. "
            "Use UNKNOWN when visibility is insufficient, no node fits, or multiple incompatible nodes fit. "
            "Do not force forward progression: nodes can be skipped, repeated, or revisited. "
            "Reference images are examples, never current evidence. Timestamped observations are oldest to newest; "
            "classify the newest view and use older views only to interpret motion. "
            "Return exactly reason and node_id as JSON. First give one short sentence of visible evidence in reason, "
            "then the node_id whose criteria match that evidence. No reasoning steps."
        ])

    def current_content(self, history):
        content = []
        for captured_time, frame in history:
            content.append({"type": "text", "text": f"Observation at {captured_time:.3f} seconds (newest observation determines current progress):"})
            content.extend(Specialist.current_content(self, frame))
        return content

    def parse(self, raw):
        value = json.loads(raw)
        if (not isinstance(value, dict) or set(value) != {"node_id", "reason"}
                or not isinstance(value["node_id"], str) or not isinstance(value["reason"], str)
                or value["node_id"] not in self.schema["properties"]["node_id"]["enum"]):
            raise ValueError("Progress response must contain a configured node_id (or UNKNOWN) and reason")
        return value


class ProgressTracker:
    def __init__(self, definition):
        self.definition = definition
        self.nodes = {n.id: n for n in definition.nodes}
        self.candidate, self.matches, self.last_key = None, 0, None
        self.state = {"enabled": True, "node_id": "UNKNOWN", "label": "Not observed", "quality": "unknown",
                      "completed": None, "total": None, "unit_label": definition.unit_label,
                      "reason": "Waiting for progress evidence", "captured_time_s": None,
                      "confirmed_captured_time_s": None, "stale_after_s": definition.stale_after_s,
                      "candidate_id": None, "matches": 0, "confirmations": definition.confirmations}

    def observe(self, observation, key, now):
        captured = observation["captured_time_s"]
        if key == self.last_key:
            return None
        self.last_key = key
        node_id = observation["node_id"]
        if node_id == "UNKNOWN" or now - captured > self.definition.stale_after_s:
            self.candidate, self.matches = None, 0
            quality = "stale" if now - captured > self.definition.stale_after_s else "unknown"
        else:
            self.matches = self.matches + 1 if self.candidate == node_id else 1
            self.candidate = node_id
            quality = "confirmed" if self.matches >= self.definition.confirmations else "pending"
            if quality == "confirmed":
                node = self.nodes[node_id]
                self.state.update(node_id=node.id, label=node.label, completed=node.completed, total=node.total,
                                  confirmed_captured_time_s=captured, confirmed_source_frames=observation.get("source_frames", {}))
        self.state.update(quality=quality, reason=observation["reason"], captured_time_s=captured,
                          candidate_id=self.candidate, matches=min(self.matches, self.definition.confirmations),
                          time_s=now, source_frames=observation.get("source_frames", {}), error=observation.get("error"))
        return {"type": "progress_state", **self.state}


class ProgressRunner:
    def __init__(self, definition, store, inputs, outcome_backend, log):
        self.definition = ProgressDefinition.model_validate(definition)
        self.store, self.inputs, self.log = store, inputs, log
        self.config = {**outcome_backend.config, "max_tokens": self.definition.max_tokens, "max_connections": 1}
        self.model, self.info = outcome_backend.model, dict(outcome_backend.info)
        self.audit = getattr(outcome_backend, "audit", None)
        self.request_gate = getattr(outcome_backend, "request_gate", None)
        self.started_at = inputs.started_at
        self.tracker = ProgressTracker(self.definition)
        self.history = deque(maxlen=50)
        self.requests = self.responses = 0
        self.error = None
        self.job = None
        self.closed_at = None

    def start(self):
        self.log({"type": "progress_state", **self.tracker.state, "time_s": 0.})
        self.job = asyncio.create_task(self.run())

    def state(self):
        now = self.closed_at if self.closed_at is not None else time.monotonic() - self.started_at
        return {**progress_snapshot(self.tracker.state, now),
                "requests": self.requests, "responses": self.responses, "error": self.error,
                "history": list(self.history)}

    async def run(self):
        last_key, request_id = None, 0
        context = None
        history = deque(maxlen=self.definition.history_frames)
        try:
            while True:
                try:
                    initial = self.inputs.get()
                    break
                except RuntimeError:
                    if getattr(self.inputs, "mode", None) != "mujoco":
                        raise
                    await asyncio.sleep(.05)
            # One audit/recording session belongs to one simulator episode. Generation
            # and camera changes can continue; a reset must not send new-scene calls to it.
            initial_context = self.inputs.evidence_context() if hasattr(self.inputs, "evidence_context") else None
            initial_simulation = (getattr(initial, "metadata", None) or {}).get("simulation", {})
            episode_identity = None
            if getattr(self.inputs, "mode", None) == "mujoco":
                episode_identity = (initial_simulation.get("server_instance_id"), initial_simulation.get("episode_id"))
                if not all(episode_identity):
                    episode_identity = tuple(initial_context[:2]) if initial_context else None
            observer = await asyncio.to_thread(ProgressObserver, self.definition, self.store, initial.views)
            async with httpx.AsyncClient(limits=httpx.Limits(max_connections=1, max_keepalive_connections=1), trust_env=False) as client:
                backend = Backend(self.config, client, specialists={}, request_gate=self.request_gate, audit=self.audit)
                backend.model, backend.info = self.model, self.info
                while True:
                    next_sample = time.monotonic() + self.definition.interval_s
                    try:
                        frame = self.inputs.get()
                    except RuntimeError as exc:
                        if getattr(self.inputs, "mode", None) != "mujoco":
                            raise
                        self.error = str(exc)
                        self.tracker.candidate, self.tracker.matches = None, 0
                        self.tracker.state.update(quality="unavailable", reason=str(exc), candidate_id=None, matches=0)
                        history.clear()
                        self.log({"type": "progress_state", **self.tracker.state,
                                  "time_s": time.monotonic() - self.started_at})
                        await asyncio.sleep(self.definition.interval_s)
                        continue
                    current_context = self.inputs.evidence_context() if hasattr(self.inputs, "evidence_context") else None
                    if episode_identity and current_context and tuple(current_context[:2]) != episode_identity:
                        self.tracker.candidate, self.tracker.matches = None, 0
                        self.tracker.state.update(quality="unavailable", reason="Simulation episode changed; monitoring session ended",
                                                  candidate_id=None, matches=0)
                        history.clear()
                        self.log({"type": "progress_episode_ended", "previous_episode": episode_identity,
                                  "context": current_context, "time_s": time.monotonic() - self.started_at})
                        self.log({"type": "progress_state", **self.tracker.state,
                                  "time_s": time.monotonic() - self.started_at})
                        return
                    if current_context != context:
                        context = current_context
                        self.tracker = ProgressTracker(self.definition)
                        history.clear()
                        last_key = None
                        self.log({"type": "progress_state", **self.tracker.state, "time_s": time.monotonic() - self.started_at})
                    key = (frame.metadata["identity"] if getattr(frame, "metadata", None) else
                           tuple((name, frame.views[name].source_index) for name in observer.sources))
                    if key != last_key:
                        last_key = key
                        captured = min(frame.views[name].observed_at for name in observer.sources) - self.started_at
                        history.append((captured, frame))
                        request_id += 1
                        observation = {"type": "progress_observation", "request_id": f"progress-{request_id}",
                                       "node_id": "UNKNOWN", "reason": "Progress request unavailable",
                                       "captured_time_s": captured, "source_frame": frame.source_index,
                                       "source_frames": {name: {"source_frame": frame.views[name].source_index,
                                           "captured_time_s": frame.views[name].observed_at - self.started_at} for name in observer.sources}}
                        if getattr(frame, "metadata", None):
                            observation["evidence"] = {"context": frame.metadata["context"],
                                                       "identity": frame.metadata["identity"],
                                                       "observed_at": frame.observed_at,
                                                       "simulation": frame.metadata["simulation"]}
                        sent = time.monotonic()
                        self.requests += 1
                        self.log({**observation, "type": "progress_request_started", "sent_time_s": sent - self.started_at})
                        try:
                            response = await backend.complete(observer, list(history))
                            observation.update(http_status=response.status_code, response_text=response.text,
                                               queue_wait_s=response.extensions.get("model_queue_wait_s", 0.))
                            response.raise_for_status()
                            body = response.json()
                            observation["response"] = body
                            message = body["message"] if backend.native else body["choices"][0]["message"]
                            if not isinstance(message, dict):
                                raise ValueError("Invalid progress response message")
                            observation["raw"] = message.get("content")
                            if any(message.get(k) for k in ("thinking", "reasoning", "reasoning_content")):
                                raise ValueError("Progress endpoint emitted reasoning despite thinking being disabled")
                            if backend.native and (body.get("done") is not True or body.get("done_reason") == "length"):
                                raise ValueError("Progress response was incomplete")
                            observation.update(observer.parse(message["content"]))
                            self.error = None
                        except asyncio.CancelledError:
                            observation.update(cancelled=True, reason="Progress request cancelled")
                            raise
                        except (httpx.HTTPError, ValueError, TypeError, KeyError, IndexError) as exc:
                            self.error = str(exc) or type(exc).__name__
                            observation.update(error=self.error[:1200], reason="Progress request unavailable")
                        finally:
                            received = time.monotonic()
                            observation.update(sent_time_s=sent - self.started_at, received_time_s=received - self.started_at, latency_s=received - sent)
                            self.responses += 1
                            if observation.get("evidence") and not self.inputs.accepts_evidence(observation["evidence"]):
                                observation.update(discarded=True, discarded_node_id=observation["node_id"], node_id="UNKNOWN",
                                                   reason="Obsolete or stale simulation observation")
                            # Serialize the discard decision along with the raw response;
                            # audit consumers must not mistake it for accepted progress.
                            self.log(observation)
                            if observation.get("discarded"):
                                # Keep this in finally: reset may cancel the worker while
                                # awaiting the provider, before it can reach the next loop.
                                history.clear()
                                self.tracker.candidate, self.tracker.matches = None, 0
                                self.tracker.state.update(quality="stale", reason=observation["reason"], candidate_id=None, matches=0)
                                self.log({"type": "progress_state", **self.tracker.state,
                                          "time_s": time.monotonic() - self.started_at})
                        if observation.get("discarded"):
                            continue
                        previous = self.tracker.state["node_id"]
                        event = self.tracker.observe(observation, key, received - self.started_at)
                        if event:
                            if event["node_id"] != previous and event["quality"] == "confirmed":
                                self.history.append({k: event[k] for k in ("node_id", "label", "time_s", "completed", "total")})
                            self.log(event)
                    await asyncio.sleep(max(0, next_sample - time.monotonic()))
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.error = str(exc) or type(exc).__name__
            self.tracker.state.update(quality="unavailable", reason=self.error, error=self.error)
            self.log({"type": "progress_state", **self.tracker.state, "time_s": time.monotonic() - self.started_at})

    async def close(self):
        if self.job and not self.job.done():
            self.job.cancel()
            await asyncio.gather(self.job, return_exceptions=True)
        self.closed_at = time.monotonic() - self.started_at
