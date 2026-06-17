from openai import OpenAI
from pathlib import Path
import base64
from tqdm import tqdm
import os
import re

def resolve_model(host, model_name):
    match host:
        case "ollama":
            base_url = "https://ollama.com/v1/"
            api_key = os.environ.get("OLLAMA_API_KEY")
            match model_name:
                case "gemma-4-26B-A4B-it":
                    return base_url, api_key, "gemma4:26b-a4b-it-bf16"
                case "gemma-4-31b-it":
                    return base_url, api_key, "gemma4:31b-it-bf16"
                case "gemma-4-31b-it-bf16-cloud":
                    return base_url, api_key, "gemma4:31b-cloud"
                case _:
                    raise ValueError(f"Unknown model name: {model_name}")

base_url, api_key, model = resolve_model("ollama", "gemma-4-31b-it-bf16-cloud")

client = OpenAI(
    base_url=base_url,
    api_key=api_key,  # required but ignored
)


def get_sorted_files_with_extension(folder, extension):
    extension = extension if extension.startswith(".") else f".{extension}"
    normalized_extension = extension.lower()

    paths = (
        path
        for path in Path(folder).iterdir()
        if path.is_file() and path.suffix.lower() == normalized_extension
    )
    return [str(path) for path in sorted(paths, key=natural_file_key)]

def natural_file_key(path):
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", Path(path).stem)
    )

def build_openai_prompt(window_size, message, response, system_prompt, sketch_image_path, current_frame_path, user_prompt):
    with open(sketch_image_path, 'rb') as f:
        sketch_image_data = f.read()

    with open(current_frame_path, 'rb') as f:
        current_frame_data = f.read()

    sketch_image = {"type": "input_image", "image_url": f"data:image/png;base64,{base64.b64encode(sketch_image_data).decode('utf-8')}", "detail": "auto"}
    current_frame = {"type": "input_image", "image_url": f"data:image/png;base64,{base64.b64encode(current_frame_data).decode('utf-8')}", "detail": "auto"}
    user_query = {"type": "input_text", "text": user_prompt}

    if not message:
        # Build first prompt
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                sketch_image,
                current_frame,
                user_query
            ]}
        ]
    else:
        user_content = message[-1]["content"]
        message_frame_list = [part for part in user_content if part.get("type") == "input_image"][1:]  # Exclude the first image (sketch_image)
        if len(message_frame_list) > window_size:
            message_frame_list = message_frame_list[-window_size:]
        current_frame = {"type": "input_image", "image_url": f"data:image/png;base64,{base64.b64encode(current_frame_data).decode('utf-8')}", "detail": "auto"}
        message[-1]["content"] = [sketch_image, *message_frame_list, current_frame, user_query]

    return message



with open("assets/prompt/[V8.0] Prompt.md", "r") as f:
    system_prompt = f.read()
sketch_image_path = "assets/bimanual_robot_setup_framework_v2.png"
user_prompt = "Current Task: Move the yellow block to the white mat with the right arm."

message = []
response = None
frames = get_sorted_files_with_extension("assets/dataset/pap_v4", "png")
print(f"Frames: {frames}")
frames_num = len(frames)
print(f"Total frames to process: {frames_num}")
for i in tqdm(range(frames_num)):
    current_frame_path = frames[i]
    print(f"Processing frame {i + 1}/{frames_num}: {current_frame_path}")
    message = build_openai_prompt(window_size=4, message=message, response=response, system_prompt=system_prompt, sketch_image_path=sketch_image_path, current_frame_path=current_frame_path, user_prompt=user_prompt)
    response = client.responses.create(
        model=model,
        reasoning={"effort": "none"},
        input=message,
        max_output_tokens=64,
    )
    print(response.output_text)
