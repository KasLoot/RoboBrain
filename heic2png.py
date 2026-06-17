from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_INPUT = Path("assets/dataset/pap_v6/HEIC_images")
DEFAULT_OUTPUT = Path("assets/dataset/pap_v6")
DEFAULT_SIZE = (640, 480)
HEIC_EXTENSIONS = {".heic", ".heif"}


def positive_int(value: str) -> int:
	try:
		number = int(value)
	except ValueError as error:
		raise argparse.ArgumentTypeError(f"invalid positive integer: {value}") from error

	if number <= 0:
		raise argparse.ArgumentTypeError(f"expected a positive integer: {value}")

	return number


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Convert HEIC/HEIF images to PNG files.",
	)
	parser.add_argument(
		"input",
		nargs="?",
		type=Path,
		default=DEFAULT_INPUT,
		help=f"HEIC file or directory to convert (default: {DEFAULT_INPUT})",
	)
	parser.add_argument(
		"output",
		nargs="?",
		type=Path,
		default=DEFAULT_OUTPUT,
		help=f"Output PNG file or directory (default: {DEFAULT_OUTPUT})",
	)
	parser.add_argument(
		"--recursive",
		action="store_true",
		help="Search input directories recursively.",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite existing PNG files instead of skipping them.",
	)
	parser.add_argument(
		"--size",
		nargs=2,
		type=positive_int,
		default=DEFAULT_SIZE,
		metavar=("WIDTH", "HEIGHT"),
		help=f"Resize output PNGs to WIDTH HEIGHT pixels (default: {DEFAULT_SIZE[0]} {DEFAULT_SIZE[1]}).",
	)
	parser.add_argument(
		"--no-resize",
		action="store_true",
		help="Keep original dimensions instead of resizing.",
	)
	parser.add_argument(
		"--compression",
		type=int,
		default=9,
		choices=range(10),
		metavar="0-9",
		help="PNG compression level, where 0 is fastest and 9 is smallest (default: 9).",
	)
	return parser.parse_args()


def register_heic_support() -> None:
	try:
		from pillow_heif import register_heif_opener
	except ImportError as error:
		raise RuntimeError(
			"Missing dependency: install HEIC support with `pip install pillow-heif Pillow`."
		) from error

	register_heif_opener()


def find_images(input_path: Path, recursive: bool) -> list[Path]:
	if input_path.is_file():
		if input_path.suffix.lower() not in HEIC_EXTENSIONS:
			raise ValueError(f"Input file is not HEIC/HEIF: {input_path}")
		return [input_path]

	if not input_path.is_dir():
		raise FileNotFoundError(f"Input path does not exist: {input_path}")

	pattern = "**/*" if recursive else "*"
	return sorted(
		path
		for path in input_path.glob(pattern)
		if path.is_file() and path.suffix.lower() in HEIC_EXTENSIONS
	)


def build_output_path(
	image_path: Path,
	input_path: Path,
	output_path: Path,
	frame_index: int | None = None,
) -> Path:
	if input_path.is_file() and output_path.suffix.lower() == ".png":
		base_path = output_path
	else:
		relative_path = image_path.relative_to(input_path) if input_path.is_dir() else image_path.name
		base_path = output_path / relative_path
		base_path = base_path.with_suffix(".png")

	if frame_index is None:
		return base_path

	return base_path.with_name(f"{base_path.stem}_{frame_index + 1:03d}{base_path.suffix}")


def convert_image(
	image_path: Path,
	input_path: Path,
	output_path: Path,
	overwrite: bool,
	compression: int,
	size: tuple[int, int] | None,
) -> tuple[int, int]:
	from PIL import Image, ImageSequence

	converted = 0
	skipped = 0
	resample_filter = getattr(Image, "Resampling", Image).LANCZOS

	with Image.open(image_path) as image:
		frame_count = getattr(image, "n_frames", 1)
		for frame_index, frame in enumerate(ImageSequence.Iterator(image)):
			destination = build_output_path(
				image_path,
				input_path,
				output_path,
				frame_index if frame_count > 1 else None,
			)
			if destination.exists() and not overwrite:
				skipped += 1
				continue

			destination.parent.mkdir(parents=True, exist_ok=True)
			png_frame = frame.convert("RGBA") if "A" in frame.getbands() else frame.convert("RGB")
			if size is not None and png_frame.size != size:
				png_frame = png_frame.resize(size, resample_filter)
			png_frame.save(destination, format="PNG", compress_level=compression)
			converted += 1

	return converted, skipped


def main() -> int:
	args = parse_args()

	try:
		register_heic_support()
		images = find_images(args.input, args.recursive)
	except (FileNotFoundError, RuntimeError, ValueError) as error:
		print(f"Error: {error}", file=sys.stderr)
		return 1

	if not images:
		print(f"No HEIC/HEIF images found in {args.input}")
		return 0

	size = None if args.no_resize else tuple(args.size)
	total_converted = 0
	total_skipped = 0
	for image_path in images:
		try:
			converted, skipped = convert_image(
				image_path=image_path,
				input_path=args.input,
				output_path=args.output,
				overwrite=args.overwrite,
				compression=args.compression,
				size=size,
			)
		except Exception as error:  # PIL can raise several decoder-specific exceptions.
			print(f"Failed: {image_path} ({error})", file=sys.stderr)
			continue

		total_converted += converted
		total_skipped += skipped
		print(f"Done: {image_path} ({converted} converted, {skipped} skipped)")

	print(f"Finished: {total_converted} converted, {total_skipped} skipped")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
