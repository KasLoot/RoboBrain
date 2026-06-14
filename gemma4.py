import argparse
from datetime import datetime
from pathlib import Path

from PIL import Image
from google import genai
from google.genai import types

import re

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL_NAME = "gemma-4-26b-a4b-it"
DEFAULT_THINKING_LEVEL = "minimal"
DEFAULT_PROMPT_VERSION = "8.0"
DEFAULT_DATASET_DIR = BASE_DIR / "assets/dataset/pap_v4"
DEFAULT_DATASET_TAG = "v4d"
SETUP_IMAGE_PATH = BASE_DIR / "assets/dataset/bimanual_robot_setup_framework_v2.png"
MODEL_ALIASES = {
    "26b": "gemma-4-26b-a4b-it",
    "26ba4b": "gemma-4-26b-a4b-it",
    "gma4_26ba4b": "gemma-4-26b-a4b-it",
    "gemma-4-26b-a4b-it": "gemma-4-26b-a4b-it",
    "31b": "gemma-4-31b-it",
    "gma4_31b": "gemma-4-31b-it",
    "gemma-4-31b-it": "gemma-4-31b-it",
}
MODEL_FILE_LABELS = {
    "gemma-4-26b-a4b-it": "gma4_26ba4b",
    "gemma-4-31b-it": "gma4_31b",
}
USER_QUERY = "- Current Task: Move the yellow block to the white mat with the right arm."
RESULTS_DIR = BASE_DIR / "results"


def parse_prompt_version(prompt_version):
    match = re.search(r"\d+(?:\.\d+)?", prompt_version.strip())
    if not match:
        raise argparse.ArgumentTypeError("prompt version must look like 8, 8.0, v8, or v8p")

    version = match.group(0)
    if "." not in version:
        version = f"{version}.0"
    return version


def prompt_file_label(prompt_version):
    label_version = prompt_version[:-2] if prompt_version.endswith(".0") else prompt_version.replace(".", "_")
    return f"v{label_version}p"


def prompt_file_path(prompt_version):
    return BASE_DIR / f"assets/prompt/[V{prompt_version}] Prompt.md"


def resolve_path(path):
    return path if path.is_absolute() else BASE_DIR / path


def parse_thinking_level(thinking_level):
    normalized_level = thinking_level.strip().lower()
    if normalized_level not in {"minimal", "high"}:
        raise argparse.ArgumentTypeError("thinking level must be minimal or high")
    return normalized_level


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Gemma 4 image-sequence experiment and save a markdown report.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=(
            "Gemma model to run. You can use the full model name or aliases "
            "26b, 26ba4b, gma4_26ba4b, 31b, or gma4_31b. "
            f"(default: {DEFAULT_MODEL_NAME})"
        ),
    )
    parser.add_argument(
        "--thinking-level",
        default=DEFAULT_THINKING_LEVEL,
        type=parse_thinking_level,
        choices=["minimal", "high"],
        help="Thinking level to use and include in the report filename. Use minimal or high. (default: minimal)",
    )
    parser.add_argument(
        "--prompt-version",
        default=DEFAULT_PROMPT_VERSION,
        type=parse_prompt_version,
        help=(
            "Prompt version to load from assets/prompt, such as 7, 7.0, v7, or v7p. "
            f"(default: {DEFAULT_PROMPT_VERSION})"
        ),
    )
    parser.add_argument(
        "--dataset-tag",
        default=DEFAULT_DATASET_TAG,
        help=f"Dataset tag to include in the report filename. (default: {DEFAULT_DATASET_TAG})",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Directory containing PNG frames. (default: {DEFAULT_DATASET_DIR.relative_to(BASE_DIR)})",
    )
    return parser.parse_args()


def resolve_model_name(model_name):
    return MODEL_ALIASES.get(model_name.strip().lower(), model_name)


def model_file_label(model_name):
    if model_name in MODEL_FILE_LABELS:
        return MODEL_FILE_LABELS[model_name]

    label = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()
    return label or "model"


def natural_frame_key(path):
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", path.stem)
    )

def build_md_file_name(model_name, prompt_name, dataset_name, thinking_level):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{prompt_name}_{dataset_name}_{thinking_level}_{timestamp}.md"


def markdown_image_path(image_path):
    try:
        return (Path("..") / image_path.relative_to(BASE_DIR)).as_posix()
    except ValueError:
        return image_path.as_posix()


def display_path(path):
    try:
        return path.relative_to(BASE_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def main():
    args = parse_args()
    model_name = resolve_model_name(args.model)
    thinking_level = args.thinking_level
    prompt_version = args.prompt_version
    prompt_label = prompt_file_label(prompt_version)
    system_prompt_path = prompt_file_path(prompt_version)
    image_dir = resolve_path(args.dataset_dir)
    dataset_tag = args.dataset_tag

    client = genai.Client()
    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    setup_image = Image.open(SETUP_IMAGE_PATH)

    report_items = []
    past_frames = []
    sliding_window_size = 4

    all_frames = sorted(image_dir.glob("*.png"), key=natural_frame_key)
    num_frames = len(all_frames)
    for i in range(num_frames):
        current_frame = Image.open(all_frames[i])
        if len(past_frames) >= sliding_window_size:
            contents = [setup_image, *past_frames[-sliding_window_size:], current_frame, USER_QUERY]
        else:
            contents = [setup_image, *past_frames, current_frame, USER_QUERY]
        response = client.models.generate_content(
            model=model_name,
            config=types.GenerateContentConfig(
                temperature=0,
                system_instruction=system_prompt,
                thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
            ),
            contents=contents,
        )

        response_text = response.text or ""
        print("-" * 50)
        print(f"Image: {all_frames[i].name}")
        print(response_text)

        past_frames.append(current_frame)
        report_items.append(
            {
                "image_path": all_frames[i],
                "response": response_text,
            }
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = RESULTS_DIR / build_md_file_name(
        model_file_label(model_name),
        prompt_label,
        dataset_tag,
        thinking_level,
    )

    lines = [
        "# Gemma 4 Experiment Report",
        "",
        "## Basic Information",
        "",
        f"1. Model name: `{model_name}`",
        f"2. System prompt file name: `{system_prompt_path.name}`",
        f"3. Prompt tag: `{prompt_label}`",
        f"4. Dataset directory: `{display_path(image_dir)}`",
        f"5. Dataset tag: `{dataset_tag}`",
        f"6. Setup image: `{display_path(SETUP_IMAGE_PATH)}`",
        f"7. User query: `{USER_QUERY}`",
        f"8. Thinking level: `{thinking_level}`",
        "",
        "## Images with Response",
        "",
    ]

    for index, item in enumerate(report_items, start=1):
        image_relative_path = markdown_image_path(item["image_path"])
        response_text = item["response"].replace("```", "````")

        lines.extend(
            [
                f"### Image {index}",
                "",
                f"![Image {index}]({image_relative_path})",
                "",
                "Response:",
                "",
                "```text",
                response_text,
                "```",
                "",
            ]
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
