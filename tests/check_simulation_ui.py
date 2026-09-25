"""MuJoCo schema, authoring and operational route checks; no simulator or model calls."""

import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import AsyncMock, patch

import cv2
import numpy as np
from fastapi.testclient import TestClient
from pydantic import ValidationError

from robobrain.task_builder import Draft, Inputs as BuilderInputs
from robobrain.web import create_app
from robobrain.workspace import FreeCamera, SimulationSettings, Task


def rejected(model, payload):
    try:
        model.model_validate(payload)
    except ValidationError:
        return
    raise AssertionError(f"Accepted invalid {model.__name__}: {payload}")


def checks(root):
    app = create_app(root)
    store, runtime, builder = app.state.store, app.state.runtime, app.state.builder
    original = store.get("insert_plug")
    assert original["input_mode"] == "camera" and original["sources"][0]["width"] == 1920
    assert original["simulation"]["auto_stop"] == "off"
    task = copy.deepcopy(original)
    task.update(id="simulation_check", name="Simulation check", input_mode="mujoco",
                simulation={"base_url":"http://127.0.0.1:8766", "auto_stop":"fail", "stale_after_s":3.5})
    for index, source in enumerate(task["sources"]):
        source.update(width=640, height=480, fps=15, enabled=True,
                      camera={**FreeCamera().model_dump(), "azimuth_deg":45.+90*index})
    validated = Task.model_validate(task).model_dump()
    assert validated["sources"][1]["camera"]["azimuth_deg"] == 135
    assert len(validated["sources"]) == 2
    for url in ["https://remote.example", "http://127.0.0.1:8766/v1", "http://user@localhost:8766", "file:///etc/passwd"]:
        rejected(SimulationSettings, {"base_url":url})
    for camera in [{"lookat_m":[0,0]}, {"lookat_m":[0,float("nan"),0]},
                   {"distance_m":0}, {"elevation_deg":90}, {"type":"fixed"}, {"width":800}]:
        rejected(FreeCamera, camera)
    rejected(Task, {**task,"sources":[*task["sources"], task["sources"][0]]})
    bad=copy.deepcopy(task)
    bad["sources"][0]["fps"]=60
    rejected(Task,bad)
    inputs = BuilderInputs(input_mode="mujoco", sources=task["sources"], simulation=task["simulation"])
    builder.data.update(inputs=inputs.model_dump(),config=task["config"])
    draft=Draft.model_validate({"name":"Orange plug task", "summary":"Observe insertion and retreat.",
        "specialists":[{"name":"Complete", "prompt":"Return POSITIVE only when the plug is seated, released and the gripper has visibly retreated.",
            "failure_type":None,"camera_sources":["camera1","camera2"],"reference_assets":[]}],
        "progress":{"prompt":"Observe progress independently of outcomes.",
            "nodes":[{"id":"SEATED_HELD","label":"Seated and held","criteria":"The pin is seated while fingers still grasp the handle."}],
            "camera_sources":["camera1"],"reference_assets":[]}})
    compiled=builder.compile(draft)
    assert compiled["input_mode"]=="mujoco" and compiled["simulation"]==task["simulation"]
    assert compiled["sources"][1]["camera"]==task["sources"][1]["camera"]
    assert len(compiled["specialists"])==1 and compiled["progress"]["enabled"]
    assert compiled["progress"]["nodes"][0]["completed"] is None
    assert "mujoco" in builder.messages("chat")[0]["content"]
    ok, encoded=cv2.imencode(".png",np.zeros((48,64,3),np.uint8))
    assert ok
    provenance={"episode_id":"scene-1","state_seq":42,"view_id":"view-1","camera_revision":3}
    status={"connected":True,"service_state":"READY","executor_state":"RUNNING"}
    with patch.object(runtime,"robot_command",AsyncMock(return_value={"status":"accepted"})) as command, \
         patch.object(runtime,"update_simulation_view",AsyncMock(return_value={"camera_revision":4})) as view, \
         patch.object(runtime,"simulation_status",AsyncMock(return_value=status)) as check, \
         patch.object(runtime,"simulation_reference",AsyncMock(return_value=(encoded.tobytes(),provenance))) as reference, \
         patch.object(runtime,"connect",AsyncMock()) as connect:
        with TestClient(app,base_url="http://127.0.0.1",headers={"X-RoboBrain":"1"}) as client:
            saved=client.put("/api/tasks/simulation_check",json=task)
            assert saved.status_code==200,saved.text
            assert saved.json()["simulation"]==task["simulation"]
            assert client.get("/api/simulation/status?task_id=simulation_check").json()==status
            assert check.call_args.args[0]["id"]=="simulation_check"
            preview=client.post("/api/builder/simulation/connect",json=inputs.model_dump())
            assert preview.status_code==200,preview.text
            assert connect.await_count==2
            assert connect.call_args.args[0]["id"]==builder.data["task_id"]
            assert connect.call_args.args[0]["simulation"]==task["simulation"]
            captured=client.post("/api/simulation/references/camera1")
            assert captured.status_code==200,captured.text
            asset=captured.json()["asset"]
            assert store.asset_path(asset).read_bytes()==encoded.tobytes()
            assert json.loads((store.assets/(asset+".json")).read_text())["provenance"]==provenance
            runtime.phase="running"
            runtime.task_id="simulation_check"
            try:
                assert client.get("/api/simulation/status?task_id=insert_plug").status_code==422
                other={**task,"id":"other_simulation","name":"Other simulation"}
                store.put(other)
                assert client.get("/api/simulation/status?task_id=other_simulation").status_code==409
                assert client.get("/api/simulation/status?task_id=simulation_check").status_code==200
                assert check.call_args.args==(),"Status polling must not replace active connection settings"
                for path,payload in [("/api/simulation/references/camera1",None),
                                     ("/api/builder/simulation/connect",inputs.model_dump()),
                                     ("/api/tasks",{"name":"locked"})]:
                    assert client.post(path,json=payload).status_code==409,path
                assert client.put("/api/tasks/simulation_check",json=task).status_code==409
                assert client.post("/api/simulation/commands",json={"command":"stop","request_id":"retry-1"}).status_code==200
                assert command.call_args.args==("stop","retry-1")
                assert client.post("/api/simulation/commands",json={"command":"start"}).status_code==200
                assert client.post("/api/simulation/views/camera1",json=FreeCamera(azimuth_deg=120).model_dump()).status_code==200
                assert view.call_args.args[1]["azimuth_deg"]==120
                assert client.post("/api/simulation/views/camera1",json={**FreeCamera().model_dump(),"width":800}).status_code==422
                assert client.post("/api/simulation/commands",json={"command":"release"}).status_code==422
                assert client.post("/api/simulation/commands",json={"command":"stop"},headers={"X-RoboBrain":""}).status_code==403
                assert client.post("/api/simulation/commands",json={"command":"stop"},headers={"Origin":"https://untrusted.example"}).status_code==403
                before=command.await_count
                assert client.post("/api/run/stop").status_code==200
                assert command.await_count==before,"Stopping monitoring must not command the robot"
                assert reference.await_count==1,"A locked capture must not reach the simulator"
            finally:
                runtime.phase="idle"
    persisted=Task.model_validate(store.get("simulation_check")).model_dump()
    assert persisted["sources"]==validated["sources"]
    print("Simulation UI checks passed: schema bounds, loopback URL, builder roundtrip, references/provenance, live operation locks and separate robot/monitor commands.")


if __name__ == "__main__":
    scratch=Path(__file__).resolve().parents[1]/".tmp"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        checks(Path(directory))
