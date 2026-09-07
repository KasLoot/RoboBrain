import cv2
import threading
import time


class SimulatedCamera:
    def __init__(self, video_path, loop=True):
        self.cap = cv2.VideoCapture(video_path)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_interval = 1.0 / self.fps

        self.loop = loop

        self.latest_frame = None
        self.lock = threading.Lock()

        self.running = False
        self.thread = None

    def start(self):
        self.running = True

        self.thread = threading.Thread(
            target=self._run,
            daemon=True
        )

        self.thread.start()

    def _run(self):
        next_frame_time = time.monotonic()

        while self.running:
            ok, frame = self.cap.read()

            if not ok:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            # Replace the current camera image.
            with self.lock:
                self.latest_frame = frame

            next_frame_time += self.frame_interval

            delay = next_frame_time - time.monotonic()

            if delay > 0:
                time.sleep(delay)

    def get(self):
        with self.lock:
            if self.latest_frame is None:
                return None

            return self.latest_frame.copy()

    def stop(self):
        self.running = False

        if self.thread:
            self.thread.join()

        self.cap.release()


def main():
    camera = SimulatedCamera("video/insert_plug-aug-28-105trail.MOV")
    camera.start()

    while True:
        frame = camera.get()

        if frame is not None:
            # Process exactly as if it came from a camera
            cv2.imshow("camera", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    camera.stop()