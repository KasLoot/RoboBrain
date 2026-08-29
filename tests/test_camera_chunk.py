import unittest
from unittest.mock import patch

import numpy as np

from robobrain.__main__ import SimulatedCamera, build_message


class FakeVideoCapture:
    def __init__(self, frames):
        self.frames = iter(frames)

    def get(self, _property):
        return 1_000_000

    def read(self):
        try:
            return True, next(self.frames)
        except StopIteration:
            return False, None

    def release(self):
        pass


class SimulatedCameraTest(unittest.TestCase):
    def test_get_returns_inclusive_frame_chunk(self):
        frames = [
            np.full((2, 2, 3), frame_index, dtype=np.uint8)
            for frame_index in range(5)
        ]
        capture = FakeVideoCapture(frames)

        with patch("robobrain.__main__.cv2.VideoCapture", return_value=capture):
            camera = SimulatedCamera("unused.mp4", loop=False, chunk_num=2)

        camera.running = True
        camera._run()

        frame_chunk, current_frame_index = camera.get()

        self.assertEqual(current_frame_index, 4)
        self.assertEqual(
            [int(frame[0, 0, 0]) for frame in frame_chunk],
            [2, 3, 4],
        )

        frame_chunk[0][:] = 99
        fresh_chunk, _ = camera.get()
        self.assertEqual(int(fresh_chunk[0][0, 0, 0]), 2)

    def test_get_returns_available_frames_during_warmup(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        capture = FakeVideoCapture([frame])

        with patch("robobrain.__main__.cv2.VideoCapture", return_value=capture):
            camera = SimulatedCamera("unused.mp4", loop=False, chunk_num=3)

        camera.running = True
        camera._run()

        frame_chunk, current_frame_index = camera.get()

        self.assertEqual(current_frame_index, 0)
        self.assertEqual(len(frame_chunk), 1)

    def test_chunk_num_must_be_non_negative_integer(self):
        with patch("robobrain.__main__.cv2.VideoCapture"):
            with self.assertRaises(ValueError):
                SimulatedCamera("unused.mp4", chunk_num=-1)
            with self.assertRaises(TypeError):
                SimulatedCamera("unused.mp4", chunk_num=1.5)


class BuildMessageTest(unittest.TestCase):
    def test_images_are_ordered_between_system_and_user_text(self):
        frames = [
            np.full((2, 2, 3), value, dtype=np.uint8)
            for value in (10, 20, 30)
        ]

        messages = build_message("system", "user", frames)["messages"]

        self.assertEqual(messages[0].type, "system")
        self.assertEqual(messages[0].content, "system")
        self.assertEqual(messages[1].type, "human")

        content = messages[1].content
        self.assertEqual(
            [part["type"] for part in content],
            ["image_url", "image_url", "image_url", "text"],
        )
        self.assertEqual(content[-1], {"type": "text", "text": "user"})
        for image_part in content[:-1]:
            self.assertTrue(
                image_part["image_url"]["url"].startswith(
                    "data:image/jpeg;base64,"
                )
            )


if __name__ == "__main__":
    unittest.main()
