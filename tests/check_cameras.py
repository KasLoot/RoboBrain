"""Explicit hardware smoke check: uv run python tests/check_cameras.py --devices 0 1."""
import argparse
import time
from robobrain.sources import LiveCamera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--devices', nargs='+', default=['0', '1'])
    args = parser.parse_args()
    for attempt in range(2):
        cameras = []
        try:
            for index, device in enumerate(args.devices):
                camera = LiveCamera({'id': f'camera{index+1}', 'device': device, 'width': 1920, 'height': 1080, 'fps': 30})
                cameras.append(camera)
                camera.start()
            before = [camera.get() for camera in cameras]
            time.sleep(.3)
            for camera, previous in zip(cameras, before):
                frame = camera.get()
                assert frame.sequence > previous.sequence
                assert frame.observed_at > previous.observed_at
                print(f'Cycle {attempt+1}, device {camera.device}: {frame.image.shape[1]}x{frame.image.shape[0]}, frames advance', flush=True)
        finally:
            for camera in cameras:
                camera.stop()
                assert not camera.process.is_alive()
    print('Camera capture, release, and reopen checks passed.')


if __name__ == '__main__':
    main()
