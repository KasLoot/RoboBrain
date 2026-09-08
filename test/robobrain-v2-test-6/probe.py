# /// script
# requires-python = ">=3.13"
# dependencies = ["httpx>=0.28,<1", "opencv-python>=4.11,<6"]
# ///
"""Compare the two endpoints on manually reviewed visual cases, without streaming."""
import argparse
import asyncio
import json
from pathlib import Path
import time

import cv2
import httpx

from run import Backend, Frame, HERE, SPECIALISTS, load_config, write_json

# Positive labels refer to the visible state at this timestamp, not the episode label.
CASES = [
    (1, 0, None), (1, 7, None), (1, 13, None), (1, 15, None), (1, 16.6, "success"),
    (9, 0, None), (9, 20, None), (9, 32.4, None), (9, 33.2, "drop"),
    (82, 0, None), (82, 10, None), (82, 21, None), (82, 23, "drop"),
    (105, 14, None), (105, 20, "success"),
    (9, 1.2, None), (9, 2.4, None), (9, 8.2, None), (9, 9.6, None),
    (1, 9.9, None), (1, 11.5, None),
    (58, 14.3, None), (58, 15.2, "success"),
    (2, 14.7, "success"), (11, 17.1, "success"), (15, 15.7, "success"),
    (44, 443 / 30, "success"),
]


async def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    frames = []
    for episode, seconds, positive in CASES:
        capture = cv2.VideoCapture(str(args.videos / f"traj-{episode}.MOV"))
        index = round(seconds * 30)
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = capture.read()
        if not ok:
            capture.release()
            capture = cv2.VideoCapture(str(args.videos / f"traj-{episode}.MOV"))
            for _ in range(index + 1):
                ok, image = capture.read()
                if not ok:
                    break
        capture.release()
        if not ok:
            raise ValueError(f"Unreadable probe frame: {episode}, {seconds}")
        frames.append((episode, seconds, positive, image, index))
    async def endpoint(url):
        config = load_config(base_url=url)
        rows = []
        async with httpx.AsyncClient(limits=httpx.Limits(max_connections=24), trust_env=False) as client:
            backend = Backend(config, client)
            for episode, seconds, positive, image, index in frames:
                started = time.monotonic()
                observations = await backend.pair(Frame(image, index, index, started), started, lambda _: None, normal=False)
                for observation in observations:
                    expected = observation["specialist"] == positive
                    observation.update(episode=episode, time_s=seconds, expected_positive=expected,
                                       correct=(observation["verdict"] == "POSITIVE") == expected)
                    rows.append(observation)
                print(json.dumps({"endpoint": url, "episode": episode, "time_s": seconds,
                                  "observations": [{k: o.get(k) for k in ("specialist", "verdict", "reason", "correct", "latency_s", "error")} for o in observations]}), flush=True)
        write_json(args.output / ("port-" + url.split(":")[-1].split("/")[0] + ".json"), rows)
        return {"base_url": url, "correct": sum(r["correct"] for r in rows), "total": len(rows), "errors": sum("error" in r for r in rows)}
    results = await asyncio.gather(*(endpoint(url) for url in args.base_urls))
    write_json(args.output / "summary.json", results)
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--base-urls", nargs="+", default=["http://127.0.0.1:8000/v1", "http://127.0.0.1:9000/v1"])
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
