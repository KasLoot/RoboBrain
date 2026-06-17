"""
Import-safe message construction for Gemma 4 multimodal benchmarking.

This module does not call the model. It only builds OpenAI-compatible messages.
It supports two layouts:

1) user_refs   (recommended): system role is text-only; user role contains static
   reference image/stage pairs first, then the changing live image/query.
2) system_refs (legacy): system role contains the static reference images/stages;
   user role contains only the live image/query.

For llama.cpp prompt-cache reuse, the important property is that the static prefix
appears before the changing live image in the rendered prompt. It does not need to
be in the literal OpenAI "system" role.
"""

from __future__ import annotations

from pathlib import Path
import base64
import mimetypes
from typing import Iterable, List


DEFAULT_IMAGE_DIR = Path("/home/yuxin/Documents/pap_stage_demo")
DEFAULT_LIVE_DIR = DEFAULT_IMAGE_DIR / "test_images"

SYSTEM_TEXT = (
    "The robot repeatedly performs a pick-and-place manipulation cycle on a tabletop. In each cycle, the robot:\n"
    "1) Starts at a home position,\n"
    "2) Approaches a randomly placed object\n"
    "3) Grasps the object,\n"
    "4) Transports the object to the mat,\n"
    "5) Places the object on the mat,\n"
    "6) Returns to the home position.\n"
    "The VLM model (you) acts as the robot’s perceptual supervisor, interpreting visual input and determining:\n"
    "1) Which stage of the manipulation cycle the robot is currently executing\n"
    "2) The confidence percentage for the current stage\n"
    "3) Unexpected conditions when the scene does not match any known stage (e.g., object missing, occlusion, robot off-trajectory).\n"
)

STAGE_FILES = [
    "s0.png",
    "s1.png",
    "s3.png",
    "s4.png",
    "s6.png",
    "s7.png",
]

STAGE_LABELS = [
    "Stage 1 — Home Position / Idle: Robot is stationary at its initial home pose, with no object interaction underway.",
    "Stage 2 — Approach Object: Robot moves from home position toward a detected object on the tabletop.",
    "Stage 3 — Grasping Interaction: Robot aligns with the object, makes contact, and performs the grasping action.",
    "Stage 4 — Transport to the Mat: Robot lifts the object and moves it through free space toward the mat.",
    "Stage 5 — Placement on the Mat: Robot lowers the object on the mat and releases it.",
    "Stage 6 — Return to Home: Robot retracts from the mat area and navigates back to the home position, completing the cycle.",
]

LIVE_QUERY_TEXT = (
    "The image above is the current live frame. Robot task: Pick up a object on the table and place it on the mat. Judge which canonical stage it matches.\n\n"
    "Output strictly and only these three lines:\n"
    "[Stage: N]\n"
    "[Confidence: XX]\n"
    "[Anomaly: none]\n\n"
    "Use Stage N as an integer from 1 to 6. Use Confidence as an integer from 0 to 100. "
    "If the scene does not match any known stage, keep the nearest stage and describe the issue in Anomaly."
)


def image_data_url(path: str | Path) -> str:
    path = Path(path)
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        mime_type = "image/png"

    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{data}"


def image_block_from_data_url(data_url: str) -> dict:
    return {
        "type": "image_url",
        "image_url": {
            "url": data_url,
        },
    }


def image_block(path: str | Path, *, detail: str | None = None) -> dict:
    block = image_block_from_data_url(image_data_url(path))
    if detail:
        block["image_url"]["detail"] = detail
    return block


def image_file_block(relative_path: str, *, detail: str | None = None) -> dict:
    block = {
        "type": "image_url",
        "image_url": {
            "url": f"file://{relative_path}",
        },
    }
    if detail:
        block["image_url"]["detail"] = detail
    return block


def text_block(text: str) -> dict:
    return {
        "type": "text",
        "text": text,
    }


def default_stage_paths(image_dir: str | Path = DEFAULT_IMAGE_DIR) -> List[Path]:
    image_dir = Path(image_dir)
    return [image_dir / name for name in STAGE_FILES]


def image_path_block(
    path: str | Path,
    *,
    image_dir: str | Path,
    image_transport: str,
    detail: str | None = None,
) -> dict:
    if image_transport == "data-url":
        return image_block(path, detail=detail)
    if image_transport == "file-url":
        return image_file_block(media_relative(path, image_dir), detail=detail)
    raise ValueError("image_transport must be 'file-url' or 'data-url'")


def reference_blocks(
    stage_paths: Iterable[str | Path],
    *,
    image_dir: str | Path,
    image_transport: str,
    image_detail: str | None = None,
) -> List[dict]:
    blocks: List[dict] = []
    for path, label in zip(stage_paths, STAGE_LABELS):
        blocks.append(text_block(label))
        blocks.append(
            image_path_block(
                path,
                image_dir=image_dir,
                image_transport=image_transport,
                detail=image_detail,
            )
        )
        
    return blocks


def discover_live_images(
    live_dir: str | Path = DEFAULT_LIVE_DIR,
    glob_pattern: str = "*.png",
) -> List[Path]:
    live_dir = Path(live_dir)
    paths = sorted(p for p in live_dir.glob(glob_pattern) if p.is_file())
    return paths

def media_relative(path: str | Path, media_root: str | Path) -> str:
    return Path(path).resolve().relative_to(Path(media_root).resolve()).as_posix()

def build_messages(
    live_path: str | Path,
    *,
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    layout: str = "user_refs",
    image_transport: str = "file-url",
    image_detail: str | None = None,
) -> list[dict]:
    """
    Build messages for a specific live image.

    layout="user_refs" is recommended based on your benchmark: text-only system
    message, then the static reference images/stages at the start of the user
    message, then the changing live image/query.

    layout="system_refs" matches your earlier design: system content includes
    the reference image/stage pairs, and the user content contains live image/query.
    """
    image_dir = Path(image_dir)
    stage_paths = default_stage_paths(image_dir)
    refs = reference_blocks(
        stage_paths,
        image_dir=image_dir,
        image_transport=image_transport,
        image_detail=image_detail,
    )
    live = [
        image_path_block(
            live_path,
            image_dir=image_dir,
            image_transport=image_transport,
            detail=image_detail,
        ),
        text_block(LIVE_QUERY_TEXT),
    ]

    if layout == "user_refs":
        return [
            {"role": "system", "content": SYSTEM_TEXT},
            {"role": "user", "content": refs + live},
        ]

    if layout == "system_refs":
        return [
            {"role": "system", "content": [text_block(SYSTEM_TEXT)] + refs},
            {"role": "user", "content": live},
        ]

    raise ValueError(f"Unknown layout: {layout!r}. Use 'user_refs' or 'system_refs'.")
