# /// script
# requires-python = ">=3.13"
# dependencies = ["httpx>=0.28,<1", "opencv-python>=4.11,<6"]
# ///
"""Probe known support errors with 30 concurrent-verification requests per frozen image."""
import argparse
import asyncio
import json
from pathlib import Path
import time

import cv2
import httpx

from run import Backend, Frame, HERE, load_config, write_json

CASES = [
    ("no-force-4", "insert_plug-no_force-aug-28-1x/traj-4.MOV", 773, True),
    ("stand-36", "insert_plug-full_model-aug-27-1x/traj-36.MOV", 118, False),
]


async def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    config = load_config()
    results = []
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=36), trust_env=False) as client:
        backend = Backend(config, client)
        for name, video, index, expected_drop in CASES:
            cap = cv2.VideoCapture(str(args.videos_root / video))
            try:
                for _ in range(index + 1):
                    ok, image = cap.read()
                    if not ok:
                        raise ValueError(f"Cannot decode {video} frame {index}")
            finally:
                cap.release()
            started, tasks = time.monotonic(), []
            for slot in range(config["evidence_horizon"]):
                await asyncio.sleep(max(0, started + slot * config["horizon_duration"] / config["evidence_horizon"] - time.monotonic()))
                tasks.append(asyncio.create_task(backend.batch(
                    Frame(image, index, index, time.monotonic()), started, lambda _: None,
                    normal=False, round_id=1, slot=slot)))
            observations = [o for batch in await asyncio.gather(*tasks) for o in batch]
            counts = {s.name: s.evidence_count(observations) for s in backend.specialists.values()}
            confirmed = [s.name for s in backend.specialists.values() if s.confirmed(observations)]
            correct = set(confirmed) == ({"drop"} if expected_drop else set())
            result = {"case": name, "video": video, "source_frame": index,
                      "mode": "frozen-image classification probe, not episode playback",
                      "expected_drop": expected_drop, "evidence_counts": counts,
                      "correct": correct, "request_errors": sum("error" in o for o in observations)}
            write_json(args.output / f"{name}.json", {**result, "observations": observations})
            results.append(result)
            print(json.dumps(result), flush=True)
    write_json(args.output / "summary.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-root", type=Path, default=Path("video"))
    parser.add_argument("--output", type=Path, default=HERE / "regression-probe-results")
    asyncio.run(main(parser.parse_args()))
