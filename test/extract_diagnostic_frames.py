"""Extract clean diagnostic frames without touching production outputs."""

from pathlib import Path

import cv2


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VIDEO_PATH = REPOSITORY_ROOT / "video" / "failure_part.mp4"
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "artifacts" / "frames"
FRAME_INDICES = (
    435,
    465,
    493,
    551,
    609,
    667,
    725,
    783,
    840,
    897,
    955,
    970,
    1012,
    1258,
    1316,
    1377,
)
CROP_TOP_LEFT = (350, 50)
CROP_BOTTOM_RIGHT = (650, 300)


def label_frame(image, label):
    labeled = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.6, labeled.shape[1] / 1200)
    thickness = max(1, round(scale * 2))
    (width, height), baseline = cv2.getTextSize(
        label, font, scale, thickness
    )
    cv2.rectangle(
        labeled,
        (0, 0),
        (width + 20, height + baseline + 20),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        labeled,
        label,
        (10, height + 10),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return labeled


def fit_to_cell(image, cell_width, cell_height):
    scale = min(cell_width / image.shape[1], cell_height / image.shape[0])
    resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )
    canvas = 255 * (
        cv2.UMat(cell_height, cell_width, cv2.CV_8UC3).get()
    )
    top = (cell_height - resized.shape[0]) // 2
    left = (cell_width - resized.shape[1]) // 2
    canvas[
        top : top + resized.shape[0],
        left : left + resized.shape[1],
    ] = resized
    return canvas


def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(VIDEO_PATH))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {VIDEO_PATH}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        f"video={VIDEO_PATH} fps={fps:.6f} frames={frame_count} "
        f"size={width}x{height}"
    )

    x1, y1 = CROP_TOP_LEFT
    x2, y2 = CROP_BOTTOM_RIGHT
    full_cells = []
    crop_cells = []

    try:
        for frame_index in FRAME_INDICES:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {frame_index}")

            timestamp = frame_index / fps
            label = f"frame {frame_index} ({timestamp:.2f}s)"
            crop = frame[y1:y2, x1:x2]
            cv2.imwrite(
                str(OUTPUT_DIRECTORY / f"frame_{frame_index:04d}_full.jpg"),
                frame,
            )
            cv2.imwrite(
                str(OUTPUT_DIRECTORY / f"frame_{frame_index:04d}_crop.jpg"),
                crop,
            )
            full_cells.append(
                fit_to_cell(label_frame(frame, label), 480, 300)
            )
            crop_cells.append(
                fit_to_cell(label_frame(crop, label), 480, 400)
            )
    finally:
        capture.release()

    columns = 4
    for name, cells in (
        ("contact_sheet_full.jpg", full_cells),
        ("contact_sheet_crop.jpg", crop_cells),
    ):
        rows = []
        for start in range(0, len(cells), columns):
            row = cells[start : start + columns]
            if len(row) < columns:
                row.extend([255 * row[0] for _ in range(columns - len(row))])
            rows.append(cv2.hconcat(row))
        cv2.imwrite(str(OUTPUT_DIRECTORY / name), cv2.vconcat(rows))


if __name__ == "__main__":
    main()
