from datetime import datetime
from pathlib import Path

from PIL import Image
from google import genai
from google.genai import types

BASE_DIR = Path(__file__).resolve().parent

MODEL_NAME = "gemma-4-26b-a4b-it"
SYSTEM_PROMPT_PATH = BASE_DIR / "assets/prompt/[V0.3] Prompt.md"
USER_QUERY = "Current Task: Move the yellow block from the orange mat to the white mat with the right arm."
IMAGE_DIR = BASE_DIR / "assets/dataset/pap_v3"
RESULTS_DIR = BASE_DIR / "results"

client = genai.Client()

system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

report_items = []

for i in range(1, 15):
    image_path = IMAGE_DIR / f"00-{i*2}.png"

    current_frame = Image.open(image_path)
    response = client.models.generate_content(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(
            temperature=0,
            system_instruction=system_prompt,
        ),
        contents=[current_frame, USER_QUERY],
    )

    response_text = response.text or ""
    print("-"*50)
    print(f"Image: {image_path.name}")
    print(response_text)

    report_items.append(
        {
            "image_path": image_path,
            "response": response_text,
        }
    )

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_path = RESULTS_DIR / f"gma4_26ba4b_v3p_{timestamp}.md"

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