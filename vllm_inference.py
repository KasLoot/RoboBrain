from openai import OpenAI
import base64
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import sys
import time
import requests

MODEL = "google/gemma-4-26B-A4B-it"
SERVER_BASE_URL = "http://localhost:8000"
CLEAR_SERVER_CACHES = True
WINDOW_SIZE = 4
IMAGE_DIR = Path("./pap_v4")
SETUP_IMAGE_PATH = Path("bimanual_robot_setup_framework_v2.png")
PROMPT_PATH = Path("[V8.0] Prompt.md")
USER_QUERY = "Current Task: Move the yellow block to the white mat with the right arm."

resp = requests.post(f"{SERVER_BASE_URL}/reset_prefix_cache")
resp.raise_for_status()

client = OpenAI(
    base_url=f"{SERVER_BASE_URL}/v1",
    api_key="EMPTY"
)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def numeric_stem(path):
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.stem)


def load_task_images(image_dir):
    image_paths = sorted(Path(image_dir).glob("*.png"), key=numeric_stem)
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in {image_dir}.")
    print("Using task images:", ", ".join(str(path) for path in image_paths))
    return [(path, encode_image(path)) for path in image_paths]


def image_part(image_b64):
    return {
        "type": "input_image",
        "image_url": f"data:image/png;base64,{image_b64}",
        "detail": "auto",
    }


def text_part(text):
    return {"type": "input_text", "text": text}


def post_server_endpoint(endpoint):
    url = f"{SERVER_BASE_URL}{endpoint}"
    request = Request(url, method="POST")
    with urlopen(request, timeout=30) as response:
        return response.status


def clear_server_caches():
    endpoints = [
        "/reset_prefix_cache",
        "/reset_mm_cache",
        "/reset_encoder_cache",
    ]

    print("Clearing vLLM server caches...", flush=True)
    for endpoint in endpoints:
        try:
            status = post_server_endpoint(endpoint)
        except HTTPError as exc:
            if exc.code == 404:
                raise RuntimeError(
                    f"{endpoint} is not available on the running vLLM server. "
                    "Restart the server with VLLM_SERVER_DEV_MODE=1 to expose "
                    "the cache reset endpoints."
                ) from exc
            raise RuntimeError(f"Failed to call {endpoint}: HTTP {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError(
                f"Could not reach vLLM server at {SERVER_BASE_URL}: {exc.reason}"
            ) from exc

        if status != 200:
            raise RuntimeError(f"Failed to call {endpoint}: HTTP {status}")
        print(f"Cleared {endpoint}")


def build_messages(system_prompt, sketch_setup, frame_images, responses, current_index):
    content = [
        text_part("Here is a sketch image of the robot setup."),
        image_part(sketch_setup),
        text_part(USER_QUERY),
        text_part("Previous frames and predictions:"),
    ]

    for frame_index, (image_path, image_b64) in enumerate(frame_images[:current_index]):
        content.extend([
            text_part(f"Frame {frame_index + 1}: {image_path.name}"),
            image_part(image_b64),
            text_part(
                f"Model prediction for frame {frame_index + 1}:\n"
                f"{responses[frame_index]}"
            ),
        ])

    current_image_path, current_image_b64 = frame_images[current_index]
    content.extend([
        text_part(f"Current frame {current_index + 1}: {current_image_path.name}"),
        image_part(current_image_b64),
    ])

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def build_messages_v2(system_prompt, sketch_setup, frame_images, responses, current_index):
    window_start = max(0, current_index - WINDOW_SIZE + 1)
    window_frame_images = frame_images[window_start:current_index + 1]

    content = []
    content.extend([image_part(sketch_setup)])

    
    for frame_index, (image_path, image_b64) in enumerate(
        window_frame_images,
        start=window_start,
    ):
        content.extend([
            image_part(image_b64),
        ])

    # content.append(text_part("Previous frames predictions:"))

    # for frame_index, response in enumerate(responses[window_start:current_index], start=window_start):
    #     content.append(text_part(f"{response[frame_index]}"))

    content.append(text_part(USER_QUERY))

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def run_request(messages):
    request_start = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=64,
        temperature=0.0,
        extra_body={
        "chat_template_kwargs": {"enable_thinking": False},
    },
    )
    request_elapsed = time.perf_counter() - request_start
    return response.choices[0].message.content, request_elapsed


if CLEAR_SERVER_CACHES:
    cache_clear_start = time.perf_counter()
    try:
        clear_server_caches()
    except RuntimeError as exc:
        print(f"Cache reset failed: {exc}", file=sys.stderr)
        sys.exit(1)
    cache_clear_elapsed = time.perf_counter() - cache_clear_start
else:
    cache_clear_elapsed = 0.0

prep_start = time.perf_counter()
SYSTEM_PROMPT = PROMPT_PATH.read_text()
sketch_setup = encode_image(SETUP_IMAGE_PATH)
frame_images = load_task_images(IMAGE_DIR)
prep_elapsed = time.perf_counter() - prep_start

responses = []
timings = []

for current_index, (image_path, _) in enumerate(frame_images):
    print(f"\n=== Request {current_index + 1}/{len(frame_images)}: {image_path} ===")
    messages = build_messages_v2(
        SYSTEM_PROMPT,
        sketch_setup,
        frame_images,
        responses,
        current_index,
    )
    response_text, request_elapsed = run_request(messages)
    responses.append(response_text)
    timings.append(request_elapsed)

    print(response_text)
    print(f"Request/response time: {request_elapsed:.3f} s")

print("\n=== Timing summary ===")
print(f"Server cache clearing time: {cache_clear_elapsed:.3f} s")
print(f"Asset preparation time: {prep_elapsed:.3f} s")
for index, ((image_path, _), elapsed) in enumerate(zip(frame_images, timings), start=1):
    print(f"Request {index} ({image_path.name}): {elapsed:.3f} s")
print(f"Total request/response time: {sum(timings):.3f} s")
print(f"Average request/response time: {sum(timings) / len(timings):.3f} s")
