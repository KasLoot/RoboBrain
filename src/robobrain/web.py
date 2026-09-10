"""Local browser interface for RoboBrain."""

import argparse
import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from .runtime import Runtime, prepare
from .sources import Inputs
from .workspace import Workspace


def create_app(data_dir=Path(".robobrain")):
    store = Workspace(data_dir)
    runtime = Runtime(store)

    @asynccontextmanager
    async def lifespan(app):
        yield
        await runtime.close()

    app = FastAPI(title="RoboBrain", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.store = store
    app.state.runtime = runtime

    @app.middleware("http")
    async def local_access(request, call_next):
        if request.url.hostname not in {"127.0.0.1", "localhost"}:
            return JSONResponse({"detail": "Use the local RoboBrain address"}, status_code=403)
        mutation = request.method not in {"GET", "HEAD", "OPTIONS"}
        if mutation:
            origin = request.headers.get("origin")
            if request.headers.get("x-robobrain") != "1" or (origin and urlsplit(origin).netloc != request.url.netloc):
                return JSONResponse({"detail": "Only the local RoboBrain interface may change this workspace"}, status_code=403)
        if mutation and request.url.path != "/api/run/stop":
            # One active run is an explicit product constraint. Serialize Start
            # with saves/connections so a stale browser cannot race the run lock.
            async with runtime.action_lock:
                if runtime.busy:
                    return JSONResponse({"detail": "Configuration is locked until the run and recording finish"}, status_code=409)
                response = await call_next(request)
        else:
            response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; frame-ancestors 'none'"
        return response

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(cv2.error)
    async def invalid_image(request, exc):
        return JSONResponse({"detail": "Could not decode this image or video"}, status_code=422)

    @app.exception_handler(StopIteration)
    async def missing(request, exc):
        return JSONResponse({"detail": "Item not found"}, status_code=404)

    @app.get("/api/tasks")
    async def tasks():
        return store.tasks

    @app.post("/api/tasks")
    async def create(body: dict):
        return store.create(body.get("name", "New task"), body.get("copy_from"))

    @app.put("/api/tasks/{task_id}")
    async def save(task_id: str, task: dict):
        if task.get("id") != task_id:
            raise ValueError("Task ID cannot be changed")
        saved = store.put(task)
        await runtime.disconnect()
        return saved

    @app.delete("/api/tasks/{task_id}")
    async def delete(task_id: str):
        store.delete(task_id)
        if runtime.preview_task_id == task_id:
            await runtime.disconnect()
        return {"ok": True}

    @app.get("/api/state")
    async def state():
        return runtime.state()

    @app.get("/api/events")
    async def events(request: Request):
        async def stream():
            while not await request.is_disconnected():
                yield "data: " + json.dumps(runtime.state()) + "\n\n"
                await asyncio.sleep(.25)
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})

    @app.post("/api/tasks/{task_id}/sources/{source_id}/connect")
    async def connect(task_id: str, source_id: str):
        try:
            await runtime.connect(store.get(task_id), source_id)
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
        return runtime.state()

    @app.post("/api/sources/{source_id}/disconnect")
    async def disconnect(source_id: str):
        await runtime.disconnect(source_id)
        return runtime.state()

    @app.get("/api/preview/{source_id}")
    def preview(source_id: str):
        if source_id not in runtime.readers:
            raise HTTPException(404, "Source is not connected")
        try:
            frame = runtime.readers[source_id].get()
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        image = frame.image
        if image.shape[1] > 960:
            image = cv2.resize(image, (960, max(1, round(image.shape[0] * 960 / image.shape[1]))))
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            raise ValueError("Could not encode preview")
        return Response(encoded.tobytes(), media_type="image/jpeg", headers={"X-Image-Width": str(frame.image.shape[1]), "X-Image-Height": str(frame.image.shape[0])})

    @app.post("/api/assets/{kind}")
    async def upload(kind: str, file: UploadFile = File(...)):
        if kind not in {"image", "video"}:
            raise ValueError("Invalid upload type")
        return await asyncio.to_thread(store.upload, file.file, file.filename, kind)

    @app.get("/api/assets/{name}")
    async def asset(name: str):
        return FileResponse(store.asset_path(name))

    @app.post("/api/model/check")
    async def check_model(body: dict):
        task = store.get(body.get("task_id"))
        config = task["config"]
        try:
            async with httpx.AsyncClient(trust_env=False, timeout=8) as client:
                response = await client.get(config["base_url"].rstrip("/") + "/models")
                response.raise_for_status()
                available = {m["id"] for m in response.json()["data"]}
                if config["model"] not in available:
                    raise ValueError("Configured model is not available")
        except httpx.HTTPError as exc:
            raise ValueError(f"Model connection failed: {exc}. Check the server and SSH tunnel.") from exc
        return {"ok": True, "model": config["model"]}

    @app.get("/api/tasks/{task_id}/request/{specialist_id}")
    async def request_preview(task_id: str, specialist_id: str):
        task = store.get(task_id)
        if runtime.preview_task_id != task_id or not runtime.readers:
            raise ValueError("Connect the selected sources to preview the current request")
        frame = Inputs(dict(runtime.readers), task["input_mode"]).get()
        _, specialists = await asyncio.to_thread(prepare, task, store, frame.views)
        if specialist_id not in specialists:
            raise ValueError("Enable this specialist to preview its request")
        specialist = specialists[specialist_id]
        return {"system": specialist.prompt, "content": await asyncio.to_thread(specialist.current_content, frame), "schema": specialist.schema}

    @app.post("/api/run")
    async def run(body: dict):
        if type(body.get("record", False)) is not bool:
            raise ValueError("record must be a boolean")
        runtime.start(store.get(body.get("task_id")), body.get("record", False))
        return runtime.state()

    @app.post("/api/run/stop")
    async def stop():
        runtime.stop_event.set()
        if runtime.phase in {"starting", "running"}:
            runtime.phase = "stopping"
        return runtime.state()

    @app.get("/api/output/{path:path}")
    async def output(path: str):
        destination = (store.root / path).resolve()
        if not destination.is_relative_to(store.root / "runs") or not destination.is_file():
            raise HTTPException(404)
        return FileResponse(destination, filename=destination.name)

    @app.get("/")
    async def index():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/static/{name}")
    async def static(name: str):
        if name not in {"app.js", "style.css"}:
            raise HTTPException(404)
        return FileResponse(Path(__file__).parent / "static" / name)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(".robobrain"), help="Saved tasks, assets and recordings")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    import uvicorn
    print(f"RoboBrain UI: http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(create_app(args.data), host="127.0.0.1", port=args.port, log_level="warning", timeout_graceful_shutdown=3)


if __name__ == "__main__":
    main()
