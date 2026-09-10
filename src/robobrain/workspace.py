"""Saved task templates and their local assets. Detector settings retain their existing format."""

import copy
import json
from pathlib import Path
import re
import shutil
from typing import Literal
from uuid import uuid4

import cv2
from pydantic import BaseModel, ConfigDict, Field, field_validator

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


class Source(Model):
    id: str
    name: str
    enabled: bool = True
    device: str = "0"
    video: str | None = None
    width: int = Field(default=1920, ge=1, le=8192)
    height: int = Field(default=1080, ge=1, le=8192)
    fps: int = Field(default=30, ge=1, le=120)

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
    input_mode: Literal["camera", "video"] = "camera"
    sources: list[Source] = Field(default_factory=list, max_length=2)
    specialists: list[Detector] = Field(default_factory=list, max_length=32)

    @field_validator("id")
    @classmethod
    def valid_id(cls, value):
        if not value.isidentifier():
            raise ValueError("Task ID must be an identifier")
        return value


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
        for specialist in task["specialists"]:
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
