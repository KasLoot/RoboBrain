"""Shared camera readers and frame snapshots for the existing monitor."""

from dataclasses import dataclass
import sys
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import threading
import time

import cv2
import numpy as np

from .monitor import Frame


def _capture(source, device, connection, stopped, ready, lock, counter, timestamp):
    """Keep native capture state on a fresh process's main thread."""
    cap = memory = None
    try:
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        cap = cv2.VideoCapture(device, backend)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {device}. Check permission, device selection, and other camera applications.")
        for prop, key in [(cv2.CAP_PROP_FRAME_WIDTH, "width"), (cv2.CAP_PROP_FRAME_HEIGHT, "height"), (cv2.CAP_PROP_FPS, "fps")]:
            cap.set(prop, source[key])
        first_deadline = time.monotonic() + 10
        index = 0
        target = None
        while not stopped.is_set():
            ok, image = cap.read()
            if not ok or image is None:
                if index == 0 and time.monotonic() < first_deadline:
                    stopped.wait(.05)
                    continue
                raise RuntimeError(f"Camera {device} stopped delivering images")
            observed_at = time.monotonic()
            if target is None:
                connection.send({"shape": image.shape})
                if not connection.poll(10):
                    raise RuntimeError("Camera buffer setup timed out")
                memory = SharedMemory(name=connection.recv(), track=False)
                target = np.ndarray(image.shape, dtype=np.uint8, buffer=memory.buf)
            if image.shape != target.shape:
                raise RuntimeError("Camera resolution changed; reconnect the source")
            with lock:
                target[:] = image
                counter.value, timestamp.value = index, observed_at
            ready.set()
            index += 1
    except Exception as exc:
        try:
            connection.send({"error": str(exc)})
        except (BrokenPipeError, EOFError, OSError):
            pass
        ready.set()
    finally:
        if memory is not None:
            memory.close()
        if cap is not None:
            cap.release()
        connection.close()


class LiveCamera:
    """One shared frame per camera; bounded shutdown even if a native read hangs."""
    def __init__(self, source):
        self.source = source
        device = str(source["device"])
        if device.isdecimal():
            self.device = int(device)
        elif sys.platform.startswith("linux") and device.startswith(("/dev/video", "/dev/v4l/")):
            self.device = device
        else:
            raise ValueError("Use a camera index (0, 1, …), or a /dev/video… device on Linux")
        context = mp.get_context("spawn")
        self.connection, self.child_connection = context.Pipe()
        self.stopped, self.ready = context.Event(), context.Event()
        self.lock = context.Lock()
        self.counter, self.timestamp = context.Value('q', -1, lock=False), context.Value('d', 0, lock=False)
        self.process = context.Process(target=_capture, args=(source, self.device, self.child_connection,
            self.stopped, self.ready, self.lock, self.counter, self.timestamp), daemon=True, name=f"camera-{device}")
        self.read_lock = threading.Lock()
        self.memory = self.array = self.latest = None
        self.error = None
        self.closed = False
        self.ended = threading.Event()
        self.started_at = 0.

    def start(self):
        self.started_at = time.monotonic()
        self.process.start()
        self.child_connection.close()
        try:
            if not self.connection.poll(30):
                raise RuntimeError("Camera did not produce a frame. Check macOS camera permission and device selection.")
            message = self.connection.recv()
            if "error" in message:
                raise RuntimeError(message["error"])
            shape = tuple(message["shape"])
            self.memory = SharedMemory(create=True, size=int(np.prod(shape)))
            self.array = np.ndarray(shape, dtype=np.uint8, buffer=self.memory.buf)
            self.connection.send(self.memory.name)
            if not self.ready.wait(10):
                raise RuntimeError("Camera did not publish its first frame")
            self.get()
        except BaseException:
            self.stop()
            raise

    def get(self):
        with self.read_lock:
            if self.closed:
                if self.latest is not None:
                    return self.latest
                raise RuntimeError(self.error or "Camera is disconnected")
            if self.connection.poll():
                try:
                    self.error = self.connection.recv().get("error", "Camera stopped")
                except EOFError:
                    self.error = "Camera capture process stopped"
            if self.error:
                raise RuntimeError(self.error)
            if self.array is None or self.counter.value < 0:
                raise RuntimeError("Camera is still connecting")
            if not self.lock.acquire(timeout=.5):
                raise RuntimeError("Camera frame buffer is unavailable")
            try:
                self.latest = Frame(self.array.copy(), self.counter.value, self.counter.value, self.timestamp.value)
            finally:
                self.lock.release()
            return self.latest

    def stop(self):
        if self.closed:
            return
        self.stopped.set()
        if self.process.pid is not None:
            self.process.join(3)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(2)
        with self.read_lock:
            self.closed = True
            self.connection.close()
            self.child_connection.close()
            self.array = None
            if self.memory is not None:
                self.memory.close()
                self.memory.unlink()
                self.memory = None


@dataclass(frozen=True)
class Snapshot(Frame):
    views: dict
    global_image: object = None


class Inputs:
    """Freeze one set of source frames for all specialists in a batch."""
    def __init__(self, readers, mode):
        self.readers, self.mode = readers, mode
        self.started_at = time.monotonic()
        self.sequence = 0

    def get(self):
        global_image = None
        if self.mode == "mujoco":
            connection = next(iter(self.readers.values())).connection
            if getattr(connection, 'include_global', False):
                views, global_image = connection.bundle(list(self.readers), with_global=True)
            else:
                views = connection.bundle(list(self.readers))
        else:
            views = {name: reader.get() for name, reader in self.readers.items()}
        first = next(iter(views.values()))
        expected = getattr(self, 'expected_simulation_episode', None)
        if expected and first.metadata['simulation']['episode_id'] != expected:
            self.scene_stop_event.set()
            raise RuntimeError('Scene reset: start a new monitoring session for the new run')
        captured = max(v.observed_at for v in views.values())
        # The unchanged monitor/report contract uses a 30 fps timeline. Physical
        # camera frame IDs and read timestamps are retained separately in views.
        index = first.source_index if self.mode == "video" else max(0, round((captured - self.started_at) * 30))
        self.sequence += 1
        return Snapshot(first.image, self.sequence, index, captured, views, global_image=global_image, metadata=first.metadata)

    def evidence_context(self):
        if self.mode != "mujoco":
            return None
        return next(iter(self.readers.values())).connection.context(list(self.readers))

    def accepts_evidence(self, evidence):
        if self.mode != "mujoco":
            return True
        connection = next(iter(self.readers.values())).connection
        if not evidence or evidence.get("context") != self.evidence_context():
            return False
        if time.monotonic() - evidence.get("observed_at", 0) > connection.settings.get("stale_after_s", 5):
            return False
        try:
            connection.bundle(list(self.readers))
        except RuntimeError:
            return False
        return True


def composite(views, names):
    """Preserve native pixels; pad differing camera sizes on a shared canvas."""
    height = max(f.image.shape[0] for f in views.values())
    height += height % 2
    images = []
    for name, frame in views.items():
        image = cv2.copyMakeBorder(frame.image, 0, height - frame.image.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        cv2.rectangle(image, (12, 12), (min(image.shape[1] - 1, 320), 50), (24, 32, 46), -1)
        cv2.putText(image, names[name], (22, 38), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1, cv2.LINE_AA)
        images.append(image)
    result = cv2.hconcat(images)
    if result.shape[1] % 2:
        result = cv2.copyMakeBorder(result, 0, 0, 0, 1, cv2.BORDER_CONSTANT)
    return result


class CompositeVideo:
    """Read every source frame for post-run export; hold shorter clips at EOF."""
    def __init__(self, paths, names):
        self.caps = {key: cv2.VideoCapture(str(path)) for key, path in paths.items()}
        self.frames, self.names = {}, names

    def read(self):
        advanced = False
        for key, cap in self.caps.items():
            ok, image = cap.read()
            if ok:
                index = self.frames[key].source_index + 1 if key in self.frames else 0
                self.frames[key] = Frame(image, index, index, 0.)
                advanced = True
            elif key not in self.frames:
                raise ValueError(f"Cannot read recording source {self.names[key]}")
        return (True, composite(self.frames, self.names)) if advanced else (False, None)

    def release(self):
        for cap in self.caps.values():
            cap.release()
