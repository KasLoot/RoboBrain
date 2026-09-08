# /// script
# requires-python = ">=3.13"
# dependencies = ["httpx>=0.28,<1", "opencv-python>=4.11,<6"]
# ///
"""Replay the three datasets sequentially with one frozen experimental candidate."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from run import HERE, run

DATASETS = [
    ("all-105", "insert_plug-full_model-aug-28-105trail-1x"),
    ("aug27-45", "insert_plug-full_model-aug-27-1x"),
    ("no-force-48", "insert_plug-no_force-aug-28-1x"),
]


async def main(args):
    manifest = json.loads((HERE / "candidate-sha256.json").read_text())
    for name, dataset in DATASETS:
        for path, expected in manifest.items():
            if hashlib.sha256((HERE / path).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Candidate changed during evaluation: {path}")
        annotations = json.loads((args.videos_root / dataset / "annotation.json").read_text())
        names = [row["video"] for row in annotations]
        if not names or len(names) != len(set(names)):
            raise ValueError(f"Empty or duplicate annotations in {dataset}")
        print(json.dumps({"dataset": dataset, "clips": len(names),
                          "dry_run": args.dry_run}), flush=True)
        if args.dry_run:
            continue
        await run(argparse.Namespace(
            videos=args.videos_root / dataset, config=HERE / "config.json",
            base_url=args.base_url, episodes=None, specialists=None,
            output=args.output_root / name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-root", type=Path, default=Path("video"))
    parser.add_argument("--output-root", type=Path, default=HERE / "runs")
    parser.add_argument("--base-url")
    parser.add_argument("--dry-run", action="store_true", help="Validate selection without model requests")
    asyncio.run(main(parser.parse_args()))
