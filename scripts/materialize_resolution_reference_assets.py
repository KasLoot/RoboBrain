"""Materialize native, resolution-specific VLM reference crops.

This script deliberately never resizes an image. Standard references are
cropped directly from the canonical video for each supported resolution.
Rotation references are cropped either from that video or from an unannotated
native still whose crop already has the required tight-panel dimensions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIRECTORY = REPOSITORY_ROOT / "src" / "robobrain" / "prompt"
REFERENCE_ROOT = PROMPT_DIRECTORY / "reference_frames"

STANDARD_FRAME_INDICES = (
    29,
    1074,
    4073,
    4600,
    6040,
    7301,
    7569,
    9447,
    9612,
    10176,
    15850,
    15950,
    17074,
    17125,
)
TIGHT_FRAME_INDICES = (4600, 7301, 7569, 9447, 17074)


@dataclass(frozen=True)
class MaterializationProfile:
    name: str
    source_size: tuple[int, int]
    video_path: Path
    standard_crop: tuple[int, int, int, int]
    tight_crop: tuple[int, int, int, int]
    rotated_stills: tuple[
        tuple[str, Path, tuple[int, int, int, int]],
        ...,
    ]

    @property
    def standard_size(self):
        x1, y1, x2, y2 = self.standard_crop
        return x2 - x1, y2 - y1

    @property
    def tight_size(self):
        x1, y1, x2, y2 = self.tight_crop
        return x2 - x1, y2 - y1


PROFILES = (
    MaterializationProfile(
        name="848x480",
        source_size=(848, 480),
        video_path=(
            REPOSITORY_ROOT
            / "video"
            / "insert_plug-full_model-aug-28-105trail-1x-w848h480.MOV"
        ),
        standard_crop=(350, 50, 650, 300),
        tight_crop=(420, 90, 600, 260),
        rotated_stills=(
            (
                "rotated_a_tight.png",
                PROMPT_DIRECTORY / "FAIL_DEMO_2.jpg",
                (70, 40, 250, 210),
            ),
            (
                "rotated_b_tight.png",
                PROMPT_DIRECTORY / "FAIL_DEMO_3.jpg",
                (70, 40, 250, 210),
            ),
        ),
    ),
    MaterializationProfile(
        name="1920x1080",
        source_size=(1920, 1080),
        video_path=(
            REPOSITORY_ROOT
            / "video"
            / "insert_plug-full_model-aug-28-105trail-1x.MOV"
        ),
        standard_crop=(792, 112, 1472, 675),
        tight_crop=(951, 202, 1358, 585),
        # These unannotated stills retain enough native pixels for a direct
        # 407x383 crop. No arrow overlays or interpolation are introduced.
        rotated_stills=(
            (
                "rotated_a_tight.png",
                PROMPT_DIRECTORY / "fail-1.png",
                (170, 80, 577, 463),
            ),
            (
                "rotated_b_tight.png",
                PROMPT_DIRECTORY / "fail-4.png",
                (170, 80, 577, 463),
            ),
        ),
    ),
)


def crop_exact(image, crop_box, expected_size, source_name):
    if image is None:
        raise RuntimeError(f"Could not read {source_name}")
    if image.ndim != 3:
        raise ValueError(f"Expected a color image for {source_name}")

    x1, y1, x2, y2 = crop_box
    height, width = image.shape[:2]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(
            f"Crop {crop_box} is outside {width}x{height} for {source_name}"
        )
    crop = image[y1:y2, x1:x2].copy()
    actual_size = crop.shape[1], crop.shape[0]
    if actual_size != expected_size:
        raise ValueError(
            f"Crop for {source_name} is {actual_size}, expected {expected_size}"
        )
    return crop


def write_png(output_path, image):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not write {output_path}")


def materialize_profile(profile):
    output_directory = REFERENCE_ROOT / profile.name
    capture = cv2.VideoCapture(str(profile.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {profile.video_path}")

    detected_size = (
        round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    if detected_size != profile.source_size:
        capture.release()
        raise ValueError(
            f"{profile.video_path} is {detected_size}, expected "
            f"{profile.source_size}"
        )

    try:
        for frame_index in STANDARD_FRAME_INDICES:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Could not read frame {frame_index} from {profile.video_path}"
                )
            crop = crop_exact(
                frame,
                profile.standard_crop,
                profile.standard_size,
                f"{profile.video_path}:frame {frame_index}",
            )
            write_png(
                output_directory / f"full_f{frame_index:06d}_crop.png",
                crop,
            )

        for frame_index in TIGHT_FRAME_INDICES:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Could not read frame {frame_index} from {profile.video_path}"
                )
            crop = crop_exact(
                frame,
                profile.tight_crop,
                profile.tight_size,
                f"{profile.video_path}:frame {frame_index}",
            )
            write_png(
                output_directory / f"full_f{frame_index:06d}_tight.png",
                crop,
            )
    finally:
        capture.release()

    for output_name, source_path, crop_box in profile.rotated_stills:
        source_image = cv2.imread(str(source_path))
        crop = crop_exact(
            source_image,
            crop_box,
            profile.tight_size,
            str(source_path),
        )
        write_png(output_directory / output_name, crop)

    manifest = {
        "profile": profile.name,
        "source_resolution": list(profile.source_size),
        "standard_crop": list(profile.standard_crop),
        "standard_size": list(profile.standard_size),
        "tight_crop": list(profile.tight_crop),
        "tight_size": list(profile.tight_size),
        "resized": False,
        "video_source": str(profile.video_path.relative_to(REPOSITORY_ROOT)),
        "standard_frame_indices": list(STANDARD_FRAME_INDICES),
        "tight_frame_indices": list(TIGHT_FRAME_INDICES),
        "rotated_stills": [
            {
                "output": output_name,
                "source": str(source_path.relative_to(REPOSITORY_ROOT)),
                "crop": list(crop_box),
            }
            for output_name, source_path, crop_box in profile.rotated_stills
        ],
    }
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Materialized {profile.name}: standard={profile.standard_size}, "
        f"tight={profile.tight_size} -> {output_directory}"
    )


def main():
    for profile in PROFILES:
        materialize_profile(profile)


if __name__ == "__main__":
    main()
