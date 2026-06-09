from pathlib import Path
import base64
import mimetypes
import re
from openai import OpenAI


MODEL = "gemma4-12b-q4"
BASE_URL = "http://127.0.0.1:8080/v1"




def image_data_url(path: str | Path) -> str:
    path = Path(path)
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        mime_type = "image/png"

    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{data}"


def image_block(path: str | Path) -> dict:
    return {
        "type": "image_url",
        "image_url": {
            "url": image_data_url(path),
        },
    }


def text_block(text: str) -> dict:
    return {
        "type": "text",
        "text": text,
    }


def parse_bracket_response(text: str) -> dict:
    """
    Parses lines like:
    [Task progress: 60%]
    [Task stage: Stage 3 — Transport to the Mat]
    """
    out = {}
    for line in text.splitlines():
        line = line.strip()
        m = re.fullmatch(r"\[(.+?):\s*(.*?)\]", line)
        if m:
            key, value = m.group(1), m.group(2)
            out[key] = value
    return out


image_dir = Path("/home/yuxin/Documents/pap_stage_demo")

stage_paths = [
    image_dir / "s0.png",
    image_dir / "s1.png",
    image_dir / "s3.png",
    image_dir / "s4.png",
    image_dir / "s6.png",
    image_dir / "s7.png",
]

live_path = image_dir / "test_images/t1.png"

# For Gemma 4, Google recommends placing image content before text for multimodal prompts.
# So each reference image is sent first, then its label.
system_prompt = [
    text_block(
        "You are a visual-judgement module for a robot performing a pick-and-place task.\n"
        "You must compare the live frame against the ordered canonical reference stages and output only the required schema."
    ),

    image_block(stage_paths[0]),
    text_block("Image 1: Stage 0 — Home Position / Idle, 0%."),

    image_block(stage_paths[1]),
    text_block("Image 2: Stage 1 — Approach Object, ~20%."),

    image_block(stage_paths[2]),
    text_block("Image 3: Stage 2 — Grasping Interaction, ~40%."),

    image_block(stage_paths[3]),
    text_block("Image 4: Stage 3 — Transport to the Mat, ~60%."),

    image_block(stage_paths[4]),
    text_block("Image 5: Stage 4 — Placement on the Mat, ~80%."),

    image_block(stage_paths[5]),
    text_block("Image 6: Stage 5 — Return to Home, 100%."),
]

user_content = [
    image_block(live_path),
    text_block(
        "Image 7 above is the current live frame. Judge which canonical stage it matches.\n\n"
        "Output strictly and only in this schema:\n"
        "[Task progress: XX%]\n"
        "[Task stage: Stage N — <short description>]\n"
        "[Success or Failure]\n"
        "[Probability: XX%]\n"
        "[Something else: <empty or anomaly description>]"
    ),
]

messages = [
    {
        "role": "system",
        "content": system_prompt,
    },
    {
        "role": "user",
        "content": user_content,
    },
]

if __name__ == "__main__":
    client = OpenAI(
    base_url=BASE_URL,
    api_key="sk-no-key-required",
    )
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=128,
        temperature=0.0,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False,
            }
        },
    )

    raw_text = response.choices[0].message.content
    print(raw_text)

    # Optionally, parse the structured response into a dictionary.
    # parsed = parse_bracket_response(raw_text)
    # print(parsed)
