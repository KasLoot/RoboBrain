from datetime import datetime
from pathlib import Path

from PIL import Image
from google import genai
from google.genai import types

import re

BASE_DIR = Path(__file__).resolve().parent

MODEL_NAME = "gemma-4-26b-a4b-it"
SYSTEM_PROMPT_PATH = BASE_DIR / "assets/prompt/[V4.0] Prompt.md"
USER_QUERY = "Current Task: Move the yellow block from the orange mat to the white mat with the right arm."
IMAGE_DIR = BASE_DIR / "assets/dataset/pap_v3.1"
RESULTS_DIR = BASE_DIR / "results"

client = genai.Client()

system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

report_items = []
past_frames = []
sliding_window_size = 4

def natural_frame_key(path):
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", path.stem)
    )


all_frames = sorted(IMAGE_DIR.glob("*.png"), key=natural_frame_key)
num_frames = len(all_frames)
for i in range(num_frames):

    current_frame = Image.open(all_frames[i])
    if len(past_frames) >= sliding_window_size:
        contents = [*past_frames[-sliding_window_size:], current_frame, USER_QUERY]
    else:
        contents = [*past_frames, current_frame, USER_QUERY]
    response = client.models.generate_content(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(
            temperature=0,
            system_instruction=system_prompt,
        ),
        contents=contents,
    )

    response_text = response.text or ""
    print("-"*50)
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

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_path = RESULTS_DIR / f"gma4_26ba4b_v4p_v3d_{timestamp}.md"

lines = [
    "# Gemma 4 Experiment Report",
    "",
    "## Basic Information",
    "",
    f"1. Model name: `{MODEL_NAME}`",
    f"2. System prompt file name: `{SYSTEM_PROMPT_PATH.name}`",
    f"3. User query: `{USER_QUERY}`",
    "",
    "## Images with Response",
    "",
]

for index, item in enumerate(report_items, start=1):
    image_relative_path = Path("..") / item["image_path"].relative_to(BASE_DIR)
    image_relative_path = image_relative_path.as_posix()
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
