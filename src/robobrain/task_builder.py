"""Ollama task authoring, separate from the monitoring model and scheduler."""

import asyncio
import base64
import copy
import json
import os
import time
from typing import Literal

import cv2
import httpx
from pydantic import ConfigDict, Field, field_validator

from .monitor import HERE, Specialist, VERDICTS, load_config
from .endpoints import model_info, ollama_headers
from .presentation import write_json
from .workspace import Model, ProgressDefinition, ProgressNode, SimulationSettings, Source, Task, new_id


class Settings(Model):
    base_url: str = "https://ollama.com/api"
    model: str = Field(default="kimi-k3:cloud", min_length=1, max_length=150)
    think: bool = True
    max_tokens: int = Field(default=8192, ge=1024, le=32768)
    timeout: float = Field(default=180, ge=10, le=600)

    @field_validator("base_url")
    @classmethod
    def endpoint(cls, value):
        try:
            url = httpx.URL(value.strip().rstrip("/"))
        except httpx.InvalidURL as exc:
            raise ValueError("Enter a valid Ollama endpoint URL") from exc
        if (url.username or url.password or url.query or url.fragment or
                url.path != "/api" or not (url == httpx.URL("https://ollama.com/api") or
                (url.scheme == "http" and url.host in {"localhost", "127.0.0.1", "::1"}))):
            raise ValueError("Use https://ollama.com/api or a local Ollama /api endpoint")
        return str(url)


class Attachment(Model):
    asset: str = Field(min_length=1, max_length=180)
    label: str = Field(min_length=1, max_length=500)
    reference: bool = False


class Inputs(Model):
    base_task_id: str | None = None
    input_mode: Literal["camera", "video", "mujoco"] = "camera"
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)
    sources: list[Source] = Field(min_length=1, max_length=2)


class Turn(Model):
    action: Literal["chat", "generate"]
    text: str = Field(default="", max_length=12000)
    attachments: list[Attachment] = Field(default_factory=list, max_length=8)
    inputs: Inputs


class DraftSpecialist(Model):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=120)
    prompt: str = Field(min_length=40, max_length=20000)
    failure_type: str | None = Field(max_length=100)
    camera_sources: list[str] = Field(min_length=1, max_length=2)
    reference_assets: list[str] = Field(default_factory=list, max_length=24)

    @field_validator("failure_type")
    @classmethod
    def failure_name(cls, value):
        if value is not None and (not value.isascii() or not value.isidentifier() or not value.isupper()):
            raise ValueError("Failure type must be null or an uppercase identifier")
        return value


class DraftProgress(Model):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    prompt: str = Field(min_length=1, max_length=20000)
    nodes: list[ProgressNode] = Field(min_length=1, max_length=64)
    camera_sources: list[str] = Field(min_length=1, max_length=2)
    reference_assets: list[str] = Field(default_factory=list, max_length=24)
    unit_label: str = Field(default="items", min_length=1, max_length=40)
    history_frames: int = Field(default=1, ge=1, le=3)


class Draft(Model):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=120)
    summary: str = Field(min_length=1, max_length=3000)
    specialists: list[DraftSpecialist] = Field(max_length=32)
    progress: DraftProgress | None = None


class CreateDraft(Model):
    revision: int
    draft: Draft


class TaskBuilder:
    def __init__(self, store):
        self.store = store
        self.root = store.root / "task-builder"
        self.root.mkdir(exist_ok=True)
        settings = self.root / "settings.json"
        self.settings = Settings.model_validate_json(settings.read_text()) if settings.exists() else Settings()
        self.job = None
        self.phase, self.error, self.partial = "idle", None, ""
        self.started_at = None
        self.connection = None
        # ponytail: one shared conversation per local workspace; add session IDs for concurrent editors.
        self.path = self.root / "conversation.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else self.empty()

    def empty(self):
        return {"task_id": new_id("task"), "revision": 0, "messages": [], "attachments": [],
                "draft": None, "config": None, "usage": [], "created": False,
                "inputs": {"base_task_id": None, "input_mode": "camera", "simulation": SimulationSettings().model_dump(), "sources": [
                    Source(id="camera1", name="Camera 1", device="0").model_dump(),
                    Source(id="camera2", name="Camera 2", device="1", enabled=False).model_dump()]}}

    @property
    def busy(self):
        return self.job is not None and not self.job.done()

    def save(self):
        write_json(self.path, self.data)

    def state(self):
        visible = [{"role": m["role"], "content": m["content"], "attachments": m.get("attachments", [])}
                   for m in self.data["messages"] if not m.get("internal")]
        return {"busy": self.busy, "phase": self.phase, "error": self.error, "partial": self.partial,
                "elapsed_s": time.monotonic() - self.started_at if self.busy and self.started_at else 0,
                "settings": self.settings.model_dump(), "key_configured": bool(os.environ.get("OLLAMA_API_KEY", "").strip()),
                "connection": self.connection, "messages": visible,
                **{k: self.data[k] for k in ("task_id", "revision", "draft", "inputs", "attachments", "usage", "created")}}

    def idle(self):
        if self.busy:
            raise ValueError("Wait for the builder or cancel its current response")

    def reset(self):
        self.idle()
        self.data = self.empty()
        self.phase, self.error, self.partial = "idle", None, ""
        self.save()
        return self.state()

    def configure(self, settings):
        self.idle()
        changed_model = (settings.base_url, settings.model) != (self.settings.base_url, self.settings.model)
        write_json(self.root / "settings.json", settings.model_dump())
        self.settings, self.connection = settings, None
        if changed_model:
            self.reset()  # Kimi requires consistent model-specific conversation history.
        return self.state()

    def client(self):
        return httpx.AsyncClient(base_url=self.settings.base_url + "/", headers=ollama_headers(self.settings.base_url),
                                 timeout=httpx.Timeout(self.settings.timeout, connect=15), trust_env=False)

    async def resolve(self, client):
        self.connection = await model_info(client, self.settings.base_url, self.settings.model,
                                           httpx.Timeout(self.settings.timeout, connect=15))
        return self.connection["model"]

    async def check(self):
        self.idle()
        try:
            async with self.client() as client:
                await self.resolve(client)
        except httpx.HTTPError as exc:
            raise ValueError(self.http_error(exc)) from exc
        return self.state()

    @staticmethod
    def http_error(exc):
        if isinstance(exc, httpx.HTTPStatusError):
            return f"Ollama returned HTTP {exc.response.status_code}. Check the builder API key, model access and usage limits."
        return "Ollama connection failed or timed out. Check the builder endpoint and try again."

    def start(self, turn):
        self.idle()
        if self.data["created"]:
            raise ValueError("Start a new conversation to create another task")
        if not turn.text.strip() and not turn.attachments and (turn.action == "chat" or not self.data["messages"]):
            raise ValueError("Describe the task or attach an image first")
        if len(self.data["messages"]) >= 100:
            raise ValueError("This conversation is full. Create the task or start a new conversation")
        inputs = turn.inputs.model_dump()
        enabled = [s for s in inputs["sources"] if s["enabled"]]
        if not enabled or len({s["id"] for s in inputs["sources"]}) != len(inputs["sources"]):
            raise ValueError("Choose at least one source with a unique ID")
        for source in enabled:
            if inputs["input_mode"] == "video":
                if not source["video"] or self.store.asset_path(source["video"]).suffix.lower() not in {".mov", ".mp4", ".avi", ".mkv"}:
                    raise ValueError("Choose an uploaded video for each enabled source")
            elif inputs["input_mode"] == "camera" and not source["device"].strip():
                raise ValueError("Enter a camera device for each enabled source")
        attachments = {a["asset"]: a for a in self.data["attachments"]}
        for attachment in turn.attachments:
            if self.store.asset_path(attachment.asset).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                raise ValueError("Chat attachments must be JPEG or PNG images")
            attachments[attachment.asset] = attachment.model_dump()
        if len(attachments) > 24:
            raise ValueError("Use at most 24 images in one conversation")
        config = (self.store.get(inputs["base_task_id"])["config"] if inputs["base_task_id"] else load_config())
        self.data.update(inputs=inputs, config=copy.deepcopy(config), attachments=list(attachments.values()), draft=None)
        self.data["revision"] += 1
        if turn.text.strip() or turn.attachments:
            self.data["messages"].append({"role": "user", "content": turn.text.strip() or "Please inspect the attached scene.",
                                           "attachments": [a.model_dump() for a in turn.attachments]})
        self.phase, self.error, self.partial = "connecting", None, ""
        self.save()
        self.started_at = time.monotonic()
        self.job = asyncio.create_task(self.run(turn.action))
        return self.state()

    def messages(self, action):
        inventory = {"input_mode": self.data["inputs"]["input_mode"], "sources": [s for s in self.data["inputs"]["sources"] if s["enabled"]],
                     "images": self.data["attachments"]}
        system = (HERE / "prompts" / "task_builder.md").read_text() + "\n\nCurrent allowed inputs:\n" + json.dumps(inventory)
        system += ("\n\nCurrent mode: DISCUSSION. Reply in concise conversational prose, never task JSON. The user must use Generate draft to generate the template."
                   if action == "chat" else "\n\nCurrent mode: GENERATION. Return task JSON matching the schema in the final instruction.")
        messages = [{"role": "system", "content": system}]
        for stored in self.data["messages"]:
            message = {k: stored[k] for k in ("role", "content", "thinking") if k in stored}
            if stored.get("attachments"):
                message["images"] = []
                for attachment in stored["attachments"]:
                    image = cv2.imread(str(self.store.asset_path(attachment["asset"])))
                    if image is None:
                        raise ValueError("An attached image is no longer readable")
                    if max(image.shape[:2]) > 2048:
                        scale = 2048 / max(image.shape[:2])
                        image = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
                    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if not ok:
                        raise ValueError("Could not encode a chat image")
                    message["images"].append(base64.b64encode(encoded).decode("ascii"))
                    message["content"] += "\nAttached image: " + attachment["label"] + " (asset: " + attachment["asset"] + ")"
            messages.append(message)
        return messages

    async def completion(self, client, model, messages, *, visible):
        message = {"role": "assistant", "content": "", "thinking": ""}
        payload = {"model": model, "messages": messages, "stream": True,
                   "options": {"num_predict": self.settings.max_tokens}}
        if self.connection["thinking"]:
            payload["think"] = self.settings.think
        size, finished = 0, False
        async with client.stream("POST", "chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                size += len(line)
                if size > 2_000_000:
                    raise ValueError("The builder response exceeded its size limit")
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise ValueError("Ollama could not generate this response. Check model access and retry.")
                for key in ("content", "thinking"):
                    message[key] += chunk.get("message", {}).get(key, "")
                if visible:
                    self.partial = message["content"]
                if chunk.get("done"):
                    self.data["usage"].append({k: chunk.get(k, 0) for k in
                                               ("prompt_eval_count", "prompt_eval_cached_count", "eval_count", "total_duration")})
                    if chunk.get("done_reason") == "length":
                        raise ValueError("The model reached its output limit. Increase the builder's output tokens and retry")
                    finished = True
                    break
        if not finished or not message["content"].strip():
            raise ValueError("The builder response was incomplete. Retry the request")
        return message

    def compile(self, draft):
        if not draft.specialists:
            raise ValueError(draft.summary)
        if sum(s.failure_type is None for s in draft.specialists) > 1:
            raise ValueError("Combine all completion conditions into one success specialist")
        if len({s.name.casefold() for s in draft.specialists}) != len(draft.specialists):
            raise ValueError("Specialists must have distinct names")
        inputs = self.data["inputs"]
        sources = {s["id"]: s for s in inputs["sources"] if s["enabled"]}
        references = {a["asset"]: a for a in self.data["attachments"] if a["reference"]}
        config = copy.deepcopy(self.data["config"])
        config.update(specialists={}, evidence_threshold={}, crops={}, references={})
        n = config["evidence_horizon"]
        threshold = min(5, n - 1) if n else 5
        specialists = []

        def blocks_for(spec):
            if (not set(spec.camera_sources) <= sources.keys() or len(set(spec.camera_sources)) != len(spec.camera_sources)
                    or not set(spec.reference_assets) <= references.keys() or len(set(spec.reference_assets)) != len(spec.reference_assets)):
                raise ValueError("Use unique enabled source IDs and only images approved as specialist references")
            blocks = [{"id": "prompt", "type": "text", "alias": "System prompt", "text": spec.prompt}]
            for i, asset in enumerate(spec.reference_assets):
                self.store.asset_path(asset)
                blocks.append({"id": f"reference_{i}", "type": "image", "alias": references[asset]["label"], "asset": asset})
            for source in spec.camera_sources:
                blocks.append({"id": "view_" + source, "type": "camera", "source": source,
                               "alias": "Current view: " + sources[source]["name"], "crop": None})
            return blocks

        for index, spec in enumerate(draft.specialists, 1):
            name = f"specialist_{index}"
            definition = {"field": "verdict", "values": {v: v for v in VERDICTS}, "failure_type": spec.failure_type}
            blocks = blocks_for(spec)
            config["specialists"][name] = definition
            config["evidence_threshold"][name] = threshold
            first = sources[spec.camera_sources[0]]
            config["crops"][name] = [0, 0, first["width"], first["height"]]
            Specialist(spec.prompt, {**definition, "name": name, "crop": config["crops"][name], "threshold": threshold})
            specialists.append({"id": name, "name": spec.name, "definition": definition, "threshold": threshold, "blocks": blocks})
        config = load_config(config)
        progress = (ProgressDefinition(enabled=True, nodes=draft.progress.nodes, blocks=blocks_for(draft.progress),
                                       unit_label=draft.progress.unit_label, history_frames=draft.progress.history_frames).model_dump()
                    if draft.progress else None)
        return Task.model_validate({"id": self.data["task_id"], "name": draft.name, "config": config,
                                    "input_mode": inputs["input_mode"], "sources": inputs["sources"],
                                    "simulation": inputs.get("simulation", {}),
                                    "specialists": specialists, "progress": progress}).model_dump()

    async def run(self, action):
        try:
            async with asyncio.timeout(self.settings.timeout):
                async with self.client() as client:
                    model = await self.resolve(client)
                    if self.data["attachments"] and not self.connection["vision"]:
                        raise ValueError("The selected builder model does not support image input. Choose a vision model or start a text-only conversation")
                    messages = await asyncio.to_thread(self.messages, action)
                    if action == "chat":
                        self.phase = "chatting"
                        message = await self.completion(client, model, messages, visible=True)
                        self.data["messages"].append(message)
                    else:
                        self.phase = "generating"
                        instruction = {"role": "user", "content": "Generate the task draft now. Return only JSON matching this schema:\n" + json.dumps(Draft.model_json_schema()), "internal": True}
                        self.data["messages"].append(instruction)
                        messages.append({k: instruction[k] for k in ("role", "content")})
                        for attempt in range(2):
                            message = await self.completion(client, model, messages, visible=False)
                            self.data["messages"].append({**message, "internal": True})
                            messages.append(message)
                            try:
                                draft = Draft.model_validate_json(message["content"])
                                if not draft.specialists:
                                    self.data["messages"].append({"role": "assistant", "content": draft.summary})
                                    break
                                self.compile(draft)
                                self.data["draft"] = draft.model_dump()
                                break
                            except ValueError as exc:
                                if attempt:
                                    raise ValueError("The generated draft is invalid: " + str(exc)[:1500]) from exc
                                self.phase = "repairing"
                                repair = {"role": "user", "content": "Correct the JSON draft using this validation error. Return the full corrected JSON only:\n" + str(exc)[:4000]}
                                messages.append(repair)
                                self.data["messages"].append({**repair, "internal": True})
            self.phase = "ready" if self.data["draft"] else "idle"
        except asyncio.CancelledError:
            self.phase, self.error = "cancelled", "Response cancelled. No task was created."
        except (httpx.HTTPError, TimeoutError) as exc:
            self.phase, self.error = "error", self.http_error(exc)
        except Exception as exc:
            self.phase, self.error = "error", str(exc)[:2000]
        finally:
            self.partial = ""
            self.save()

    async def cancel(self):
        if self.busy:
            self.job.cancel()
            await asyncio.gather(self.job, return_exceptions=True)
            self.phase, self.error, self.partial = "cancelled", "Response cancelled. No task was created.", ""
            self.save()
        return self.state()

    def create(self, body):
        self.idle()
        if body.revision != self.data["revision"]:
            raise ValueError("The draft changed. Reload the builder before creating the task")
        existing = next((t for t in self.store.tasks if t["id"] == self.data["task_id"]), None)
        if existing:
            return copy.deepcopy(existing)
        if self.data["draft"] is None:
            raise ValueError("Generate a valid draft before creating a task")
        task = self.store.put(self.compile(body.draft))
        self.data.update(draft=body.draft.model_dump(), created=True)
        self.save()
        return task
