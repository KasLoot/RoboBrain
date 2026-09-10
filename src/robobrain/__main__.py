"""Monitor robot episodes with independent success, drop and rotation specialists."""

import argparse
import asyncio
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
import tempfile

import cv2
import httpx

from .monitor import Backend, HERE, ReplayController, SimulatedCamera, load_config, monitor
from .presentation import EpisodeReport, LiveView, annotate_frame, render_recording

DEFAULT_VIDEOS = Path("video/insert_plug-full_model-aug-28-105trail-1x")
WINDOW = "RoboBrain"


def select_videos(source, episodes):
    source = source.expanduser().resolve()
    if source.is_file():
        videos, dataset = [source], source.parent.name
    elif source.is_dir():
        videos = sorted((p for p in source.iterdir() if p.is_file() and p.suffix.lower() in {".mov", ".mp4", ".avi", ".mkv"}),
                        key=lambda p: [int(s) if s.isdigit() else s.casefold() for s in re.split(r"(\d+)", p.name)])
        dataset = source.name
    else:
        raise ValueError(f"Video input does not exist: {source}. Use --videos to select a dataset.")
    if episodes:
        wanted = {f"traj-{number}" for number in episodes}
        videos = [p for p in videos if p.stem in wanted]
        missing = wanted - {p.stem for p in videos}
        if missing:
            raise ValueError("Missing episodes: " + ", ".join(sorted(missing)))
    if not videos:
        raise ValueError(f"No episode videos found in {source}")
    if len({p.stem for p in videos}) != len(videos):
        raise ValueError("Episode filenames must have unique stems for their output folders")
    return dataset, videos


async def run_episodes(args, config, dataset, videos, view, stop_event):
    limits = httpx.Limits(max_connections=config["max_connections"], max_keepalive_connections=config["max_connections"])
    async with httpx.AsyncClient(limits=limits, trust_env=False) as client:
        response = await client.get(config["base_url"].rstrip("/") + "/models", timeout=config["request_timeout"])
        response.raise_for_status()
        if config["model"] not in {m["id"] for m in response.json()["data"]}:
            raise ValueError("Configured model is not available at the endpoint")
        backend = Backend(config, client)
        controller = ReplayController()
        for index, video in enumerate(videos, 1):
            if stop_event.is_set():
                break
            print(f"[{index}/{len(videos)}] {dataset}/{video.name}", flush=True)
            camera = SimulatedCamera(video)
            directory = args.output / dataset / video.stem if args.record else None
            report = None
            result = {"status": "ON_GOING", "failure_type": None, "failure_description": None,
                      "source_frame": -1, "decision_time_s": 0.}
            error = None
            recording = None
            try:
                await asyncio.to_thread(camera.start)
                frame = camera.get()
                height, width = frame.image.shape[:2]
                if any(crop[2] > width or crop[3] > height for crop in config["crops"].values()):
                    raise ValueError(f"Configured crops do not fit {video.name}: {width}x{height}")
                view.show(camera, f"{dataset}/{video.name}")
                report = EpisodeReport(video, dataset, config, directory, view)
                attempt = f"{dataset}/{video.stem}"
                controller.start(attempt, report.log)
                result = await monitor(camera, backend, config, report.log, stop_event)
                controller.finish(attempt, result, report.log)
                print(f"  {result['status']} | Failure type: {result.get('failure_type') or 'NONE'}"
                      f" | {result['decision_time_s']:.2f}s", flush=True)
                if args.canvas:
                    # Finish the source clip and make a late terminal decision readable.
                    while not stop_event.is_set() and (not camera.ended.is_set() or
                            time.monotonic() - camera.started_at < result["decision_time_s"] + 1.):
                        camera.get()
                        await asyncio.sleep(.02)
            except Exception as exc:
                error = exc
                if report:
                    with view.lock:
                        result.update(source_frame=view.status.get("source_frame", -1))
                if camera.started_at:
                    result["decision_time_s"] = time.monotonic() - camera.started_at
                result.update(stop_reason="runtime_error", error=str(exc))
                if report:
                    report.log({"type": "error", "message": str(exc), "time_s": result["decision_time_s"]})
                    report.log({"type": "status", **result})
            finally:
                camera.stop()
                if report:
                    try:
                        if args.record:
                            destination = directory / f"output-{video.stem}.MOV"
                            print(f"  Saving {destination}", flush=True)
                            recording = render_recording(video, destination, report.events, result, args.ffmpeg, stop_event)
                    except Exception as exc:
                        report.log({"type": "recording_error", "message": str(exc)})
                        raise
                    finally:
                        report.finish(result, recording)
            if error is not None:
                raise error


async def run_cameras(args, config, dataset, devices, view, stop_event):
    from .runtime import Runtime
    from .workspace import Workspace
    with tempfile.TemporaryDirectory(prefix="robobrain-camera-") as temporary:
        store = Workspace(args.output / "camera" if args.record else temporary)
        task = store.seed(config)
        task["sources"] = [{"id": f"camera{i+1}", "name": f"Camera {i+1}", "enabled": True,
                            "device": device, "video": None, "width": 1920, "height": 1080, "fps": 30}
                           for i, device in enumerate(devices)]
        runtime = Runtime(store, view=view, stop_event=stop_event)
        runtime.start(task, args.record)
        await runtime.job
        if runtime.error:
            raise RuntimeError(runtime.error)
        print(f"{runtime.result['status']} | Failure type: {runtime.result.get('failure_type') or 'NONE'}", flush=True)
        for path in runtime.output:
            print(store.root / path, flush=True)


def run_application(args, config, dataset, videos, runner=run_episodes):
    """HighGUI stays on the main thread, including on macOS/Cocoa."""
    view, stop_event, errors = LiveView(), threading.Event(), []

    def worker():
        try:
            asyncio.run(runner(args, config, dataset, videos, view, stop_event))
        except Exception as exc:
            errors.append(exc)

    window_open = False
    if args.canvas:
        if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise ValueError("--canvas requires a desktop display; omit it for a headless run")
        if hasattr(cv2, "currentUIFramework") and not cv2.currentUIFramework():
            raise ValueError("--canvas requires a GUI-enabled OpenCV installation")
    thread = threading.Thread(target=worker, name="robobrain-inference")
    thread.start()
    previous = None
    try:
        while thread.is_alive():
            try:
                if args.canvas and not stop_event.is_set():
                    snapshot = view.snapshot()
                    if snapshot:
                        if not window_open:
                            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
                            cv2.resizeWindow(WINDOW, 1280, 720)
                            window_open = True
                        frame, name, status = snapshot
                        key = (name, frame.sequence, status.get("decision_time_s"))
                        if key != previous:
                            cv2.setWindowTitle(WINDOW, f"RoboBrain | {name}")
                            cv2.imshow(WINDOW, annotate_frame(frame.image, frame.source_index, status))
                            previous = key
                    if window_open:
                        keypress = cv2.waitKey(1) & 0xFF
                        if keypress in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                            stop_event.set()
                            cv2.destroyWindow(WINDOW)
                            window_open = False
                thread.join(.01 if window_open else .05)
            except KeyboardInterrupt:
                stop_event.set()
            except cv2.error as exc:
                errors.append(exc)
                stop_event.set()
                window_open = False
    finally:
        stop_event.set()
        thread.join()
        if window_open:
            cv2.destroyWindow(WINDOW)
    if errors:
        raise errors[0]


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "ui":
        from .web import main as ui_main
        del sys.argv[1]
        return ui_main()
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--videos", type=Path, help="Dataset directory or single clip")
    inputs.add_argument("--camera", action="append", help="Camera index or Linux device path; repeat for a second camera")
    parser.add_argument("--episodes", type=int, nargs="+", help="Selected trajectory numbers, e.g. --episodes 9 94")
    parser.add_argument("--record", action="store_true", help="Write an annotated MOV, raw log and readable JSON for each episode")
    parser.add_argument("--canvas", action="store_true", help="Show the live annotated camera; Q/Esc or closing the window stops the run")
    parser.add_argument("--output", type=Path, default=Path("output"), help="Recording root (default: output)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing recordings for the selected episodes")
    parser.add_argument("--config", type=Path, default=HERE / "config.json", help="Monitor configuration JSON")
    parser.add_argument("--base-url", help="Override the vLLM endpoint (default: http://127.0.0.1:9000/v1)")
    parser.add_argument("--episode-timeout", type=float, help="Video-episode watchdog in seconds (default: 90; camera runs continue until stopped)")
    args = parser.parse_args()
    try:
        config = load_config(args.config, args.base_url)
        if args.episode_timeout is not None:
            if not 0 < args.episode_timeout < float("inf"):
                raise ValueError("--episode-timeout must be finite and positive")
            config["episode_timeout"] = args.episode_timeout
        args.output = args.output.expanduser().resolve()
        if args.camera:
            if args.episodes or len(args.camera) > 2:
                parser.error("--camera supports one or two devices and cannot be combined with --episodes")
            if args.episode_timeout is not None:
                parser.error("--episode-timeout applies to --videos; camera monitoring continues until stopped")
            return run_application(args, config, "camera", args.camera, runner=run_cameras)
        dataset, videos = select_videos(args.videos, args.episodes)
        args.ffmpeg = shutil.which("ffmpeg") if args.record else None
        if args.record and not args.ffmpeg:
            raise ValueError("--record requires ffmpeg on PATH to preserve the original audio")
        if args.record:
            for video in videos:
                directory = args.output / dataset / video.stem
                if directory.exists() and not args.overwrite:
                    raise ValueError(f"Output already exists: {directory}. Use --overwrite or select another --output directory.")
                if directory.exists() and not directory.is_dir():
                    raise ValueError(f"Output is not a directory: {directory}")
        run_application(args, config, dataset, videos)
    except (ValueError, RuntimeError, OSError, KeyError, httpx.HTTPError, cv2.error) as exc:
        detail = str(exc) or type(exc).__name__
        if isinstance(exc, httpx.HTTPError):
            detail = f"Model endpoint unavailable ({config['base_url']}): {detail}. Check the server and SSH tunnel."
        parser.exit(1, f"robobrain: {detail}\n")


if __name__ == "__main__":
    main()
