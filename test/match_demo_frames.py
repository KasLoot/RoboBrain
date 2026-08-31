"""Find source frames that most closely match the unannotated demos."""

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VIDEO_PATH = ROOT / "video" / "failure_part.mp4"
PROMPT_DIRECTORY = ROOT / "src" / "robobrain" / "prompt"
DEMO_NAMES = ("FAIL_DEMO_1.jpg", "FAIL_DEMO_2.jpg", "FAIL_DEMO_3.jpg")
CROP = (350, 50, 650, 300)


def main():
    demos = {
        name: cv2.imread(str(PROMPT_DIRECTORY / name)) for name in DEMO_NAMES
    }
    capture = cv2.VideoCapture(str(VIDEO_PATH))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {VIDEO_PATH}")

    best = {name: [] for name in DEMO_NAMES}
    frame_index = 0
    x1, y1, x2, y2 = CROP
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            crop = frame[y1:y2, x1:x2]
            for name, demo in demos.items():
                difference = float(
                    np.mean(
                        np.abs(
                            crop.astype(np.float32)
                            - demo.astype(np.float32)
                        )
                    )
                )
                candidates = best[name]
                candidates.append((difference, frame_index))
                candidates.sort()
                del candidates[10:]
            frame_index += 1
    finally:
        capture.release()

    for name, candidates in best.items():
        print(name)
        print(" ".join(f"{index}:{difference:.3f}" for difference, index in candidates))


if __name__ == "__main__":
    main()
