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
                case "gemma-4-26b-a4b-it":
                    return base_url, api_key, "gemma4:26b-a4b-it-bf16"
                case "gemma-4-31b-it":
                    return base_url, api_key, "gemma4:31b-it-bf16"
                case "gemma-4-31b-it-bf16-cloud":
                    return base_url, api_key, "gemma4:31b-cloud"
                case _:
                    raise ValueError(f"Unknown model name: {model_name}")
        case "google":
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            api_key = os.environ.get("GEMINI_API_KEY")
            match model_name:
                case "gemma-4-26B-A4B-it":
                    return base_url, api_key, "gemma-4-26b-a4b-it"
                case "gemma-4-31b-it":
                    return base_url, api_key, "gemma-4-31b-it"
                case _:
                    raise ValueError(f"Unknown model name: {model_name}")




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

def build_openai_prompt(window_size, message, response, system_prompt, sketch_image, current_frame_path, user_prompt):

    with open(current_frame_path, 'rb') as f:
        current_frame_data = f.read()

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
        message[-1]["content"] = [sketch_image, *message_frame_list, current_frame, user_query]

    return message

def build_openai_prompt_v2(window_size, message: list, response, system_prompt, sketch_image, current_frame_path, user_prompt):

    with open(current_frame_path, 'rb') as f:
        current_frame_data = f.read()

    
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
        message.extend([
            {"role": "assistant", "content": response.output_text},
            {"role": "user", "content": [
                current_frame,
                user_query
            ]}
            ])
        max_frame_turns = max(1, window_size)
        frame_turn_count = sum(1 for item in message[1:] if item.get("role") == "user")

        while frame_turn_count > max_frame_turns and len(message) >= 4:
            if message[1].get("role") == "user" and message[2].get("role") == "assistant":
                del message[1:3]
                first_user_content = message[1]["content"]
                if not isinstance(first_user_content, list):
                    first_user_content = [first_user_content]
                if not first_user_content or first_user_content[0] != sketch_image:
                    message[1]["content"] = [sketch_image, *first_user_content]
                frame_turn_count -= 1
            else:
                break

    return message


def main():
    base_url, api_key, model = resolve_model("ollama", "gemma-4-31b-it-bf16-cloud")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,  # required but ignored
    )

    with open("assets/prompt/[V8.0] Prompt.md", "r") as f:
        system_prompt = f.read()
    sketch_image_path = "assets/bimanual_robot_setup_framework_v2.png"
    user_prompt = "Current Task: Move the yellow block to the white mat with the right arm."

    message = []
    response = None
    frames = get_sorted_files_with_extension("assets/dataset/pap_v4", "png")
    print(f"All Frames: {frames}")
    frames_num = len(frames)
    with open(sketch_image_path, 'rb') as f:
        sketch_image_data = f.read()
    sketch_image = {"type": "input_image", "image_url": f"data:image/png;base64,{base64.b64encode(sketch_image_data).decode('utf-8')}", "detail": "auto"}
    print(f"Total frames to process: {frames_num}")

    for i in range(frames_num):
        current_frame_path = frames[i]
        print("-"*50)
        print(f"Processing frame {i + 1}/{frames_num}: {current_frame_path}")
        message = build_openai_prompt_v2(window_size=4, message=message, response=response, system_prompt=system_prompt, sketch_image=sketch_image, current_frame_path=current_frame_path, user_prompt=user_prompt)
        response = client.responses.create(
            model=model,
            reasoning={"effort": "none"},
            input=message,
            max_output_tokens=64,
            temperature=1.0,
            top_p=0.95,
        )
        print(response.output_text)


if __name__ == "__main__":
    main()