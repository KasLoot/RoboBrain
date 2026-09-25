"""Saved task templates and their local assets. Detector settings retain their existing format."""

import copy
import json
from pathlib import Path
import re
import shutil
from typing import Literal
from urllib.parse import urlsplit
from uuid import uuid4

import cv2
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .monitor import HERE
from .presentation import write_json


def new_id(prefix):
    return prefix + "_" + uuid4().hex[:12]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("name", check_fields=False)
    @classmethod
    def nonempty_name(cls, value):
        if not value.strip():
            raise ValueError("Name cannot be empty")
        return value.strip()


class Block(Model):
    id: str
    type: Literal["text", "image", "camera"]
    alias: str = ""
    enabled: bool = True
    text: str = ""
    asset: str | None = None
    source: str = "camera1"
    crop: list[int] | None = None

    @field_validator("crop")
    @classmethod
    def valid_crop(cls, crop):
        if crop is not None and (len(crop) != 4 or not (0 <= crop[0] < crop[2] and 0 <= crop[1] < crop[3])):
            raise ValueError("Crop must be [left, top, right, bottom] with positive dimensions")
        return crop


class Detector(Model):
    id: str
    name: str = Field(min_length=1, max_length=120)
    enabled: bool = True
    definition: dict | None = None
    threshold: int = 5
    blocks: list[Block] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def valid_id(cls, value):
        if not value.isidentifier():
            raise ValueError("Specialist ID must be an identifier")
        return value


class ProgressNode(Model):
    id: str
    label: str = Field(min_length=1, max_length=120)
    criteria: str = Field(min_length=1, max_length=6000)
    completed: int | None = Field(default=None, ge=0, strict=True)
    total: int | None = Field(default=None, ge=1, strict=True)

    @field_validator("id")
    @classmethod
    def valid_id(cls, value):
        if not value.isidentifier() or value.upper() == "UNKNOWN":
            raise ValueError("Node ID must be an identifier; UNKNOWN is reserved")
        return value

    @model_validator(mode="after")
    def valid_count(self):
        if (self.completed is None) != (self.total is None) or (self.total is not None and self.completed > self.total):
            raise ValueError("Supply both completed and total, with 0 <= completed <= total")
        if not self.label.strip() or not self.criteria.strip():
            raise ValueError("Node label and observable criteria cannot be blank")
        return self


class ProgressDefinition(Model):
    id: Literal["progress"] = "progress"
    name: str = "Task progress"
    enabled: bool = False
    nodes: list[ProgressNode] = Field(default_factory=list, max_length=64)
    blocks: list[Block] = Field(default_factory=list)
    unit_label: str = Field(default="items", min_length=1, max_length=40)
    interval_s: float = Field(default=1., ge=.05, le=60, allow_inf_nan=False)
    confirmations: int = Field(default=2, ge=1, le=10, strict=True)
    stale_after_s: float = Field(default=5., gt=0, le=600, allow_inf_nan=False)
    history_frames: int = Field(default=1, ge=1, le=3, strict=True)
    max_tokens: int = Field(default=192, ge=32, le=2048, strict=True)

    @model_validator(mode="after")
    def valid_nodes(self):
        if len({n.id for n in self.nodes}) != len(self.nodes):
            raise ValueError("Progress node IDs must be unique")
        totals = {n.total for n in self.nodes if n.total is not None}
        if len(totals) > 1 or (totals and any(n.total is None for n in self.nodes)):
            raise ValueError("Count nodes must all use the same total; otherwise leave counts empty for every node")
        if self.enabled and (not self.nodes or not any(b.enabled and b.type == "camera" for b in self.blocks)):
            raise ValueError("Enabled progress needs predefined nodes and a camera/video block")
        return self


class FreeCamera(Model):
    type: Literal["free"] = "free"
    lookat_m: list[float] = Field(default_factory=lambda: [.45, 0., .15], min_length=3, max_length=3)
    distance_m: float = Field(default=.8, ge=.05, le=20, allow_inf_nan=False)
    azimuth_deg: float = Field(default=45., ge=-360, le=360, allow_inf_nan=False)
    elevation_deg: float = Field(default=-25., ge=-89.9, le=89.9, allow_inf_nan=False)

    @field_validator("lookat_m")
    @classmethod
    def finite_target(cls, values):
        if any(not -20 <= value <= 20 for value in values):
            raise ValueError("Camera target must contain three finite metre coordinates between -20 and 20")
        return values


class SimulationSettings(Model):
    base_url: str = "http://127.0.0.1:8766"
    auto_stop: Literal["off", "fail", "terminal"] = "off"
    stale_after_s: float = Field(default=5., gt=0, le=600, allow_inf_nan=False)

    @field_validator("base_url")
    @classmethod
    def local_service(cls, value):
        value = value.strip().rstrip("/")
        try:
            url = urlsplit(value)
            port = url.port
        except ValueError as exc:
            raise ValueError("Enter a valid local simulation URL") from exc
        if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}
                or url.username or url.password or url.query or url.fragment or url.path
                or (port is not None and port < 1)):
            raise ValueError("Use a loopback HTTP simulation URL, such as http://127.0.0.1:8766")
        return value


class Source(Model):
    id: str
    name: str
    enabled: bool = True
    device: str = "0"
    video: str | None = None
    width: int = Field(default=1920, ge=1, le=8192)
    height: int = Field(default=1080, ge=1, le=8192)
    fps: int = Field(default=30, ge=1, le=120)
    camera: FreeCamera = Field(default_factory=FreeCamera)

    @field_validator("id")
    @classmethod
    def valid_id(cls, value):
        if not value.isidentifier():
            raise ValueError("Source ID must be an identifier")
        return value


class Task(Model):
    id: str
    name: str = Field(min_length=1, max_length=120)
    config: dict
    input_mode: Literal["camera", "video", "mujoco"] = "camera"
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)
    sources: list[Source] = Field(default_factory=list, max_length=2)
    specialists: list[Detector] = Field(default_factory=list, max_length=32)
    progress: ProgressDefinition | None = None

    @field_validator("id")
    @classmethod
    def valid_id(cls, value):
        if not value.isidentifier():
            raise ValueError("Task ID must be an identifier")
        return value


    @model_validator(mode="after")
    def simulation_dimensions(self):
        if self.input_mode == "mujoco" and any(
                not (16 <= source.width <= 1920 and 16 <= source.height <= 1080 and source.fps <= 30)
                for source in self.sources):
            raise ValueError("MuJoCo views require width 16–1920, height 16–1080 and FPS 1–30")
        return self


class Workspace:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.assets = self.root / "assets"
        self.assets.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "tasks.json"
        if not self.path.exists():
            self.save([self.seed()])
        self.tasks = [Task.model_validate(t).model_dump() for t in json.loads(self.path.read_text())["tasks"]]

    def seed(self, config=None):
        config = copy.deepcopy(config) if config is not None else json.loads((HERE / "config.json").read_text())
        config.pop("enabled_specialists", None)
        specialists = []
        for name, definition in config["specialists"].items():
            blocks = [{"id": new_id("block"), "type": "text", "alias": "System prompt",
                       "text": (HERE / "prompts" / f"{name}.md").read_text()}]
            for ref in config["references"].get(name, []):
                reference_path = (HERE / "references" / ref["file"]).resolve()
                if not reference_path.is_relative_to(HERE / "references"):
                    raise ValueError("Reference must be inside the package references directory")
                asset = "insert-plug-" + ref["file"]
                shutil.copy2(reference_path, self.assets / asset)
                blocks.append({"id": new_id("block"), "type": "image", "alias": ref["label"], "asset": asset})
            blocks.append({"id": new_id("block"), "type": "camera", "alias": "Current camera observation:",
                           "source": "camera1", "crop": config["crops"][name]})
            specialists.append({"id": name, "name": name.capitalize(), "definition": definition,
                                "threshold": config["evidence_threshold"][name], "blocks": blocks})
        return Task.model_validate({"id": "insert_plug", "name": "Insert plug", "config": config,
                                   "sources": [{"id": "camera1", "name": "Camera 1", "device": "0"},
                                               {"id": "camera2", "name": "Camera 2", "device": "1", "enabled": False}],
                                   "specialists": specialists}).model_dump()

    def save(self, tasks):
        write_json(self.path, {"version": 1, "tasks": tasks})
        self.tasks = tasks

    def get(self, task_id):
        for task in self.tasks:
            if task["id"] == task_id:
                return copy.deepcopy(task)
        raise ValueError("Task not found")

    def put(self, task):
        task = Task.model_validate(task).model_dump()
        for entries in (task["sources"], task["specialists"]):
            if len({e["id"] for e in entries}) != len(entries):
                raise ValueError("IDs must be unique")
        sources = {s["id"] for s in task["sources"]}
        for specialist in [*task["specialists"], *([task["progress"]] if task["progress"] else [])]:
            if len({b["id"] for b in specialist["blocks"]}) != len(specialist["blocks"]):
                raise ValueError("Block IDs must be unique")
            for block in specialist["blocks"]:
                if block["type"] == "camera" and block["source"] not in sources:
                    raise ValueError("Camera block refers to an unknown source")
                if block["asset"]:
                    self.asset_path(block["asset"])
        for source in task["sources"]:
            if source["video"]:
                self.asset_path(source["video"])
        tasks = copy.deepcopy(self.tasks)
        for i, current in enumerate(tasks):
            if current["id"] == task["id"]:
                tasks[i] = task
                break
        else:
            tasks.append(task)
        self.save(tasks)
        return task

    def create(self, name, copy_from=None):
        if not isinstance(name, str):
            raise ValueError("Task name must be text")
        if copy_from:
            task = self.get(copy_from)
        else:
            config = json.loads((HERE / "config.json").read_text())
            task = {"config": config, "sources": [{"id": "camera1", "name": "Camera 1", "device": "0"},
                                                   {"id": "camera2", "name": "Camera 2", "device": "1", "enabled": False}]}
        task.update(id=new_id("task"), name=name.strip())
        return self.put(task)

    def delete(self, task_id):
        self.get(task_id)
        # Assets remain available to other templates and recorded runs.
        self.save([t for t in self.tasks if t["id"] != task_id])

    def asset_path(self, name):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name.startswith("."):
            raise ValueError("Invalid asset name")
        path = self.assets / name
        if not path.is_file() or path.is_symlink():
            raise ValueError("Asset does not exist")
        return path

    def upload(self, file, filename, kind):
        suffix = Path(filename or "").suffix.lower()
        allowed = {".jpg", ".jpeg", ".png"} if kind == "image" else {".mov", ".mp4", ".avi", ".mkv"}
        if suffix not in allowed:
            raise ValueError("Supported formats: " + ", ".join(sorted(allowed)))
        path = self.assets / (new_id("asset") + suffix)
        limit = 50 * 1024**2 if kind == "image" else 2 * 1024**3
        try:
            size = 0
            with path.open("xb") as output:
                while chunk := file.read(1024**2):
                    size += len(chunk)
                    if size > limit:
                        raise ValueError("File exceeds the upload limit (50 MB images / 2 GB videos)")
                    output.write(chunk)
            if kind == "image":
                image = cv2.imread(str(path))
                if image is None or image.shape[0] * image.shape[1] > 40_000_000:
                    raise ValueError("Image is unreadable or exceeds 40 megapixels")
                height, width = image.shape[:2]
            else:
                cap = cv2.VideoCapture(str(path))
                try:
                    ok, image = cap.read()
                    if not ok:
                        raise ValueError("Video contains no readable frames")
                    height, width = image.shape[:2]
                finally:
                    cap.release()
            return {"asset": path.name, "width": width, "height": height, "filename": filename}
        except BaseException:
            path.unlink(missing_ok=True)
            raise
