from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI


DECOMPOSITION_SYSTEM_PROMPT = """You decompose tabletop manipulation goals into ordered short-horizon pick-and-place subtasks.

Use the user query, setup sketch, current scene image, and visible scene state.
The VLM must infer visible movable objects when the query refers to all objects or all blocks.
Respect explicit user ordering, object, source, destination, and arm constraints from the user query.
Do not invent user preferences.
Do not output explanations.
Return exactly one JSON object."""


VALIDATION_SYSTEM_PROMPT = """You are the final task validator for a tabletop manipulation episode.

Independently verify the original long-horizon task after the monitor has reported every subtask complete.
Use the initial scene image, ordered completion-evidence images, the final scene image, the ordered plan,
and the monitor completion records.
Treat monitor SUCCESS reports as claims to audit, not as proof.
Check every planned target, destination, explicit order or arm constraint that the supplied evidence can verify,
and whether the final scene satisfies the overall user intent.
Do not fail a task solely because execution-history constraints cannot be established from still images;
list those constraints under unverifiable_constraints instead.
Return verdict SUCCESS only when every required final-state effect is visibly satisfied and no supplied evidence
contradicts a required execution constraint. Otherwise return FAIL.
Do not output explanations outside the JSON object.
Return exactly one JSON object."""


LONG_HORIZON_MEMORY_ADDENDUM = """

Long-horizon memory addendum:
The current user prompt may include an episode memory summary and a Current Task selected from a longer task.
Evaluate only the Current Task.
Objects already completed and resting on the destination mat are allowed.
Do not treat completed objects on the destination as anomalies unless they physically obstruct the current target placement.
The stage/progress fields remain per-current-subtask, not global long-horizon progress.
"""


UNSPECIFIED_VALUES = {"", "unspecified", "unknown", "none", "null", "n/a", "not specified"}


def resolve_model(host, model_name):
    host = host.strip().lower()
    model_name = model_name.strip().lower()

    match host:
        case "ollama":
            base_url = "https://ollama.com/v1/"
            api_key = os.environ.get("OLLAMA_API_KEY")
            endpoint = "responses"
            match model_name:
                case "gemma-4-31b-it-bf16-cloud":
                    return base_url, api_key, "gemma4:31b-cloud", endpoint
                case _:
                    raise ValueError(f"Unknown model name: {model_name}")
        case "google":
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            api_key = os.environ.get("GEMINI_API_KEY")
            endpoint = "chat/completions"
            match model_name:
                case "gemma-4-26b-a4b-it":
                    return base_url, api_key, "gemma-4-26b-a4b-it", endpoint
                case "gemma-4-31b-it":
                    return base_url, api_key, "gemma-4-31b-it", endpoint
                case _:
                    raise ValueError(f"Unknown model name: {model_name}")
        case "nvidia":
            base_url = "https://integrate.api.nvidia.com/v1"
            api_key = os.environ.get("NVIDIA_API_KEY")
            endpoint = "chat/completions"
            match model_name:
                case "diffusiongemma-26b-a4b-it":
                    return base_url, api_key, "google/diffusiongemma-26b-a4b-it", endpoint
                case _:
                    raise ValueError(f"Unknown model name: {model_name}")
        case "vllm":
            base_url = "http://localhost:8000/v1"
            api_key = "EMPTY"
            endpoint = "chat/completions"
            match model_name:
                case "diffusiongemma-26b-a4b-it":
                    return base_url, api_key, "google/diffusiongemma-26B-B4B-it", endpoint
                case "gemma-4-26b-a4b-it":
                    return base_url, api_key, "google/gemma-4-26B-A4B-it", endpoint
                case "gemma-4-31b-it":
                    return base_url, api_key, "google/gemma-4-31B-it", endpoint
                case _:
                    raise ValueError(f"Unknown model name: {model_name}")
        case _:
            raise ValueError(f"Unknown host: {host}")


def natural_file_key(path):
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", Path(path).stem)
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


def get_response_text(end_point, response):
    if end_point == "responses":
        return response.output_text
    if end_point == "chat/completions":
        return response.choices[0].message.content
    raise ValueError(f"Unknown endpoint: {end_point}")


def get_completion_tokens(end_point, response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    if end_point == "responses":
        return getattr(usage, "output_tokens", 0) or 0
    if end_point == "chat/completions":
        return getattr(usage, "completion_tokens", 0) or 0
    raise ValueError(f"Unknown endpoint: {end_point}")


def create_response(
    client: OpenAI,
    end_point,
    model,
    message,
    max_tokens=128,
    temperature=1.0,
    top_p=0.95,
):
    if end_point == "responses":
        return client.responses.create(
            model=model,
            reasoning={"effort": "none"},
            input=message,
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    if end_point == "chat/completions":
        request_kwargs = {
            "model": model,
            "messages": message,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "reasoning_effort": "minimal",
        }
        return client.chat.completions.create(**request_kwargs)
    raise ValueError(f"Unknown endpoint: {end_point}")


@dataclass
class SubTask:
    id: str
    target: str
    source: str
    destination: str
    arm: str
    instruction: str
    status: str = "PENDING"


@dataclass
class EpisodeMemory:
    long_task: str
    user_constraints: dict[str, Any]
    subtasks: list[SubTask]
    current_index: int = 0
    completed: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, Any] | None = None

    def __post_init__(self):
        self._activate_current()

    def _activate_current(self):
        current = self.current_subtask()
        if current is not None and current.status == "PENDING":
            current.status = "RUNNING"

    def current_subtask(self):
        if self.current_index >= len(self.subtasks):
            return None
        return self.subtasks[self.current_index]

    def is_done(self):
        return self.current_index >= len(self.subtasks)

    def mark_success(self, frame_idx: int, supervisor_output: dict):
        current = self.current_subtask()
        if current is None:
            raise ValueError("Cannot mark success because all subtasks are complete.")

        current.status = "SUCCESS"
        self.completed.append(
            {
                "id": current.id,
                "target": current.target,
                "instruction": current.instruction,
                "frame_idx": frame_idx,
                "supervisor_output": supervisor_output,
            }
        )
        self.current_index += 1
        self._activate_current()

    def to_dict(self):
        return {
            "long_task": self.long_task,
            "user_constraints": self.user_constraints,
            "subtasks": [asdict(subtask) for subtask in self.subtasks],
            "current_index": self.current_index,
            "completed": self.completed,
            "task_complete": self.is_done(),
            "validation": self.validation,
            "task_validated": self.validation is not None,
            "task_success": (
                self.validation is not None and self.validation.get("verdict") == "SUCCESS"
            ),
        }

    def save(self, path: str):
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")

    def compact_summary(self):
        completed_targets = [item.get("target") for item in self.completed]
        current = self.current_subtask()
        current_target = current.target if current is not None else "none"
        remaining_start = self.current_index + 1 if current is not None else self.current_index
        remaining_targets = [subtask.target for subtask in self.subtasks[remaining_start:]]

        return "\n".join(
            [
                "Episode memory:",
                f"- completed targets: {json.dumps(completed_targets)}",
                f"- current target: {current_target}",
                f"- remaining targets: {json.dumps(remaining_targets)}",
                f"- user constraints: {json.dumps(self.user_constraints, sort_keys=True)}",
            ]
        )

    def build_current_task_prompt(self):
        current = self.current_subtask()
        current_instruction = current.instruction if current is not None else "none"
        return "\n\n".join(
            [
                f"Long-horizon task: {self.long_task}",
                f"Current Task: {current_instruction}",
                self.compact_summary(),
                (
                    "Judge only the Current Task. Previously completed objects on the "
                    "destination are allowed unless they obstruct the current target placement."
                ),
            ]
        )


def _mime_type_for_path(path: str):
    suffix = Path(path).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def _image_data_url(path: str):
    with open(path, "rb") as f:
        image_data = f.read()
    image_b64 = base64.b64encode(image_data).decode("utf-8")
    return f"data:{_mime_type_for_path(path)};base64,{image_b64}"


def make_image_part_from_path(path: str, endpoint: str):
    image_url = _image_data_url(path)
    if endpoint == "responses":
        return {"type": "input_image", "image_url": image_url, "detail": "auto"}
    if endpoint == "chat/completions":
        return {
            "type": "image_url",
            "image_url": {"url": image_url, "detail": "auto"},
        }
    raise ValueError(f"Unknown endpoint: {endpoint}")


def normalize_image_part_for_endpoint(image_part: dict, endpoint: str):
    if endpoint == "responses":
        if image_part.get("type") == "input_image":
            return image_part
        if image_part.get("type") == "image_url":
            image_url = image_part.get("image_url", {})
            if isinstance(image_url, dict):
                return {
                    "type": "input_image",
                    "image_url": image_url["url"],
                    "detail": image_url.get("detail", "auto"),
                }
        raise ValueError("Unsupported image part for responses endpoint.")

    if endpoint == "chat/completions":
        if image_part.get("type") == "image_url":
            return image_part
        if image_part.get("type") == "input_image":
            return {
                "type": "image_url",
                "image_url": {
                    "url": image_part["image_url"],
                    "detail": image_part.get("detail", "auto"),
                },
            }
        raise ValueError("Unsupported image part for chat/completions endpoint.")

    raise ValueError(f"Unknown endpoint: {endpoint}")


def make_text_part(text: str, endpoint: str):
    if endpoint == "responses":
        return {"type": "input_text", "text": text}
    if endpoint == "chat/completions":
        return {"type": "text", "text": text}
    raise ValueError(f"Unknown endpoint: {endpoint}")


def strip_markdown_fence(text: str):
    stripped = text.strip()
    fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    return stripped


def extract_json_object(text: str):
    stripped = strip_markdown_fence(text)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for start_index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(stripped[start_index:])
            return obj
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not parse a JSON object from model response: {text[:500]}")


def _is_unspecified(value):
    if value is None:
        return True
    return str(value).strip().lower() in UNSPECIFIED_VALUES


def _require_mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"Expected {name} to be a JSON object.")
    return value


def _require_fields(mapping: dict, fields: list[str], context: str):
    missing = [field_name for field_name in fields if field_name not in mapping]
    if missing:
        raise ValueError(f"Missing required field(s) in {context}: {', '.join(missing)}")


def _require_nonempty_string(mapping: dict, field_name: str, context: str):
    value = mapping.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected {context}.{field_name} to be a non-empty string.")
    return value.strip()


def _append_arm_to_instruction_if_needed(instruction: str, arm: str):
    if arm.lower() in instruction.lower():
        return instruction
    return f"{instruction.rstrip()} Use the {arm}."


def parse_decomposition_response(raw_text: str, long_horizon_query: str, default_arm_fallback: str):
    plan = _require_mapping(extract_json_object(raw_text), "decomposition plan")
    _require_fields(plan, ["long_task", "user_constraints", "subtasks", "current_subtask_id"], "plan")
    _require_nonempty_string(plan, "long_task", "plan")

    user_constraints = _require_mapping(plan["user_constraints"], "plan.user_constraints")
    _require_fields(user_constraints, ["explicit_order", "arm", "source", "destination"], "plan.user_constraints")
    if not isinstance(user_constraints["explicit_order"], list):
        raise ValueError("Expected plan.user_constraints.explicit_order to be a list.")

    raw_subtasks = plan["subtasks"]
    if not isinstance(raw_subtasks, list) or not raw_subtasks:
        raise ValueError("Expected plan.subtasks to be a non-empty list.")

    subtasks: list[SubTask] = []
    seen_ids: set[str] = set()
    required_subtask_fields = ["id", "target", "source", "destination", "arm", "instruction"]

    for index, raw_subtask in enumerate(raw_subtasks, start=1):
        subtask_data = _require_mapping(raw_subtask, f"plan.subtasks[{index}]")
        _require_fields(subtask_data, required_subtask_fields, f"plan.subtasks[{index}]")

        subtask_id = _require_nonempty_string(subtask_data, "id", f"plan.subtasks[{index}]")
        if subtask_id in seen_ids:
            raise ValueError(f"Duplicate subtask id in decomposition plan: {subtask_id}")
        seen_ids.add(subtask_id)

        target = _require_nonempty_string(subtask_data, "target", f"plan.subtasks[{index}]")
        source = _require_nonempty_string(subtask_data, "source", f"plan.subtasks[{index}]")
        destination = _require_nonempty_string(subtask_data, "destination", f"plan.subtasks[{index}]")
        instruction = _require_nonempty_string(subtask_data, "instruction", f"plan.subtasks[{index}]")

        raw_arm = subtask_data.get("arm")
        if _is_unspecified(raw_arm):
            arm = default_arm_fallback
            print(
                (
                    f"Decomposition left arm unspecified for {subtask_id}; "
                    f"using fallback arm: {default_arm_fallback}"
                ),
                file=sys.stderr,
            )
            instruction = _append_arm_to_instruction_if_needed(instruction, arm)
        else:
            arm = str(raw_arm).strip()

        subtasks.append(
            SubTask(
                id=subtask_id,
                target=target,
                source=source,
                destination=destination,
                arm=arm,
                instruction=instruction,
            )
        )

    current_subtask_id = _require_nonempty_string(plan, "current_subtask_id", "plan")
    if current_subtask_id != subtasks[0].id:
        raise ValueError(
            "Expected plan.current_subtask_id to match the first ordered subtask id. "
            f"Got {current_subtask_id!r}; first subtask is {subtasks[0].id!r}."
        )

    return EpisodeMemory(
        long_task=long_horizon_query,
        user_constraints=user_constraints,
        subtasks=subtasks,
    )


def build_decomposition_user_prompt(long_horizon_query: str):
    schema = {
        "long_task": "string",
        "user_constraints": {
            "explicit_order": ["string"],
            "arm": "string or unspecified",
            "source": "string",
            "destination": "string",
        },
        "subtasks": [
            {
                "id": "subtask_001",
                "target": "string",
                "source": "string",
                "destination": "string",
                "arm": "string or unspecified",
                "instruction": "string",
            }
        ],
        "current_subtask_id": "subtask_001",
    }
    return "\n\n".join(
        [
            "Long-horizon user query:",
            long_horizon_query,
            "Return exactly one JSON object matching this schema:",
            json.dumps(schema, indent=2),
            (
                "The subtasks array must be ordered in execution order. Each subtask must be "
                "one short-horizon pick-and-place instruction. If a field is truly unspecified "
                'by the query and scene, use the string "unspecified".'
            ),
        ]
    )


def decompose_long_horizon_task(
    client,
    endpoint: str,
    model: str,
    sketch_image: dict,
    first_frame_path: str,
    long_horizon_query: str,
    default_arm_fallback: str = "right arm",
    max_tokens: int = 1024,
) -> EpisodeMemory:
    prompt = build_decomposition_user_prompt(long_horizon_query)
    message = [
        {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                normalize_image_part_for_endpoint(sketch_image, endpoint),
                make_image_part_from_path(first_frame_path, endpoint),
                make_text_part(prompt, endpoint),
            ],
        },
    ]
    response = create_response(
        client=client,
        end_point=endpoint,
        model=model,
        message=message,
        max_tokens=max_tokens,
    )
    return parse_decomposition_response(
        raw_text=get_response_text(endpoint, response),
        long_horizon_query=long_horizon_query,
        default_arm_fallback=default_arm_fallback,
    )


def _ensure_sketch_on_first_user_turn(message: list, sketch_image: dict):
    if len(message) < 2 or message[1].get("role") != "user":
        return
    first_user_content = message[1]["content"]
    if not isinstance(first_user_content, list):
        first_user_content = [first_user_content]
    if not first_user_content or first_user_content[0] != sketch_image:
        message[1]["content"] = [sketch_image, *first_user_content]


def build_monitor_prompt(
    endpoint,
    window_size,
    message,
    response,
    system_prompt,
    sketch_image,
    current_frame_path,
    current_task_prompt,
):
    sketch_for_endpoint = normalize_image_part_for_endpoint(sketch_image, endpoint)
    current_frame = make_image_part_from_path(current_frame_path, endpoint)
    user_query = make_text_part(current_task_prompt, endpoint)

    if not message:
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [sketch_for_endpoint, current_frame, user_query],
            },
        ]

    if response is None:
        raise ValueError("Previous response is required when extending monitor conversation history.")

    message.extend(
        [
            {"role": "assistant", "content": get_response_text(endpoint, response)},
            {
                "role": "user",
                "content": [current_frame, user_query],
            },
        ]
    )

    max_frame_turns = max(1, window_size)
    frame_turn_count = sum(1 for item in message[1:] if item.get("role") == "user")

    while frame_turn_count > max_frame_turns and len(message) >= 4:
        if message[1].get("role") == "user" and message[2].get("role") == "assistant":
            del message[1:3]
            _ensure_sketch_on_first_user_turn(message, sketch_for_endpoint)
            frame_turn_count -= 1
        else:
            break

    _ensure_sketch_on_first_user_turn(message, sketch_for_endpoint)
    return message


def load_supervisor_system_prompt(prompt_path: str):
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read() + LONG_HORIZON_MEMORY_ADDENDUM


def parse_supervisor_output(raw_text: str):
    try:
        parsed = extract_json_object(raw_text)
    except ValueError as exc:
        return {
            "status": "PARSE_ERROR",
            "parse_error": str(exc),
            "raw_text": raw_text,
        }
    if not isinstance(parsed, dict):
        return {
            "status": "PARSE_ERROR",
            "parse_error": "Supervisor response JSON was not an object.",
            "raw_text": raw_text,
        }
    parsed.setdefault("raw_text", raw_text)
    return parsed


def build_validation_prompt(memory: EpisodeMemory):
    schema = {
        "verdict": "SUCCESS or FAIL",
        "confidence": "number from 0.00 to 1.00",
        "summary": "one short sentence",
        "verified_subtasks": ["subtask id"],
        "failed_subtasks": [
            {
                "id": "subtask id",
                "reason": "one short sentence",
            }
        ],
        "unverifiable_constraints": ["one short description"],
    }
    validation_context = {
        "long_task": memory.long_task,
        "user_constraints": memory.user_constraints,
        "ordered_subtasks": [asdict(subtask) for subtask in memory.subtasks],
        "monitor_completion_records": memory.completed,
    }
    return "\n\n".join(
        [
            "Validate this completed long-horizon episode:",
            json.dumps(validation_context, indent=2),
            "Return exactly one JSON object matching this schema:",
            json.dumps(schema, indent=2),
        ]
    )


def parse_validation_output(raw_text: str):
    try:
        parsed = extract_json_object(raw_text)
        parsed = _require_mapping(parsed, "validation result")
        _require_fields(
            parsed,
            [
                "verdict",
                "confidence",
                "summary",
                "verified_subtasks",
                "failed_subtasks",
                "unverifiable_constraints",
            ],
            "validation result",
        )

        verdict = _require_nonempty_string(parsed, "verdict", "validation result").upper()
        if verdict not in {"SUCCESS", "FAIL"}:
            raise ValueError("Expected validation result.verdict to be SUCCESS or FAIL.")
        parsed["verdict"] = verdict

        confidence = parsed["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("Expected validation result.confidence to be a number.")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Expected validation result.confidence to be between 0 and 1.")

        _require_nonempty_string(parsed, "summary", "validation result")
        for field_name in ("verified_subtasks", "failed_subtasks", "unverifiable_constraints"):
            if not isinstance(parsed[field_name], list):
                raise ValueError(f"Expected validation result.{field_name} to be a list.")

        parsed["raw_text"] = raw_text
        return parsed
    except ValueError as exc:
        return {
            "verdict": "PARSE_ERROR",
            "parse_error": str(exc),
            "raw_text": raw_text,
        }


def validate_completed_task(
    client,
    endpoint: str,
    model: str,
    sketch_image: dict,
    initial_frame_path: str,
    final_frame_path: str,
    memory: EpisodeMemory,
    completion_frame_paths: list[tuple[str, str]] | None = None,
    max_tokens: int = 512,
):
    content = [
        normalize_image_part_for_endpoint(sketch_image, endpoint),
        make_text_part("INITIAL SCENE IMAGE:", endpoint),
        make_image_part_from_path(initial_frame_path, endpoint),
    ]
    for subtask_id, frame_path in completion_frame_paths or []:
        if frame_path == final_frame_path:
            continue
        content.extend(
            [
                make_text_part(
                    f"MONITOR COMPLETION EVIDENCE FOR {subtask_id}:",
                    endpoint,
                ),
                make_image_part_from_path(frame_path, endpoint),
            ]
        )
    content.extend(
        [
            make_text_part("FINAL SCENE IMAGE (LAST SUBTASK COMPLETION):", endpoint),
            make_image_part_from_path(final_frame_path, endpoint),
            make_text_part(build_validation_prompt(memory), endpoint),
        ]
    )
    message = [
        {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": content,
        },
    ]
    response = create_response(
        client=client,
        end_point=endpoint,
        model=model,
        message=message,
        max_tokens=max_tokens,
    )
    raw_text = get_response_text(endpoint, response)
    return parse_validation_output(raw_text), response


def supervisor_says_success(supervisor_output: dict):
    return str(supervisor_output.get("status", "")).strip().upper() == "SUCCESS"


def append_jsonl(path: str, record: dict):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")


def publish_subtask(subtask: SubTask):
    print(subtask.instruction, flush=True)


def load_long_horizon_query(args):
    if args.query and args.query_file:
        raise ValueError("Use only one of --query or --query-file.")
    if args.query_file:
        query = Path(args.query_file).read_text(encoding="utf-8").strip()
    else:
        query = (args.query or "").strip()
    if not query:
        raise ValueError("Provide a long-horizon query with --query or --query-file.")
    return query


def parse_args():
    parser = argparse.ArgumentParser(
        description="Long-horizon memory/decomposition inference for VLM robot supervision."
    )
    parser.add_argument("--host", default="ollama")
    parser.add_argument("--model-name", default="gemma-4-31b-it-bf16-cloud")
    parser.add_argument("--dataset-path", default="assets/dataset/pap_v6")
    parser.add_argument("--frame-extension", default="png")
    parser.add_argument("--sketch-image-path", default="assets/bimanual_robot_setup_framework_v2.png")
    parser.add_argument("--prompt-path", default="assets/prompt/[V11.0] Prompt.md")
    parser.add_argument("--query")
    parser.add_argument("--query-file")
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--monitor-max-tokens", type=int, default=192)
    parser.add_argument("--decomposition-max-tokens", type=int, default=1024)
    parser.add_argument("--validation-max-tokens", type=int, default=512)
    parser.add_argument("--default-arm", default="right arm")
    parser.add_argument("--memory-path", default="mem_episode_memory.json")
    parser.add_argument("--output-jsonl-path", default="mem_agent_output.txt")
    parser.add_argument("--validation-output-path", default="mem_validation_output.json")
    return parser.parse_args()


def main():
    args = parse_args()
    long_horizon_query = load_long_horizon_query(args)

    base_url, api_key, model, endpoint = resolve_model(args.host, args.model_name)
    if not api_key:
        raise ValueError(
            "Missing API key for selected provider. Set NVIDIA_API_KEY, GEMINI_API_KEY, or OLLAMA_API_KEY."
        )

    frames = get_sorted_files_with_extension(args.dataset_path, args.frame_extension)
    if not frames:
        raise ValueError(
            f"No .{args.frame_extension.lstrip('.')} frames found in dataset path: {args.dataset_path}"
        )

    print("Run Configs:", file=sys.stderr)
    print(f"  Host: {base_url}", file=sys.stderr)
    print(f"  Endpoint: {endpoint}", file=sys.stderr)
    print(f"  Model: {model}", file=sys.stderr)
    print(f"  Dataset: {args.dataset_path}", file=sys.stderr)
    print(f"  Frames: {len(frames)}", file=sys.stderr)

    client = OpenAI(base_url=base_url, api_key=api_key)
    system_prompt = load_supervisor_system_prompt(args.prompt_path)
    sketch_image = make_image_part_from_path(args.sketch_image_path, "responses")

    memory = decompose_long_horizon_task(
        client=client,
        endpoint=endpoint,
        model=model,
        sketch_image=sketch_image,
        first_frame_path=frames[0],
        long_horizon_query=long_horizon_query,
        default_arm_fallback=args.default_arm,
        max_tokens=args.decomposition_max_tokens,
    )
    memory.save(args.memory_path)

    current = memory.current_subtask()
    if current is not None:
        publish_subtask(current)

    message = []
    response = None
    generation_times: list[float] = []
    generated_tokens: list[int] = []
    last_processed_frame_path = None

    for frame_idx, current_frame_path in enumerate(frames, start=1):
        if memory.is_done():
            break

        current = memory.current_subtask()
        current_task_prompt = memory.build_current_task_prompt()
        message = build_monitor_prompt(
            endpoint=endpoint,
            window_size=args.window_size,
            message=message,
            response=response,
            system_prompt=system_prompt,
            sketch_image=sketch_image,
            current_frame_path=current_frame_path,
            current_task_prompt=current_task_prompt,
        )

        request_start_time = time.time()
        response = create_response(
            client=client,
            end_point=endpoint,
            model=model,
            message=message,
            max_tokens=args.monitor_max_tokens,
        )
        request_end_time = time.time()

        response_text = get_response_text(endpoint, response)
        last_processed_frame_path = current_frame_path
        elapsed = request_end_time - request_start_time
        tokens = get_completion_tokens(endpoint, response)
        generation_times.append(elapsed)
        generated_tokens.append(tokens)

        print(response_text, flush=True)
        supervisor_output = parse_supervisor_output(response_text)

        append_jsonl(
            args.output_jsonl_path,
            {
                "frame_index": frame_idx,
                "frame_path": current_frame_path,
                "current_subtask": asdict(current) if current is not None else None,
                "supervisor": supervisor_output,
                "response_time_s": elapsed,
                "generated_tokens": tokens,
            },
        )

        if supervisor_says_success(supervisor_output):
            memory.mark_success(frame_idx=frame_idx, supervisor_output=supervisor_output)
            memory.save(args.memory_path)
            message = []
            response = None

            next_subtask = memory.current_subtask()
            if next_subtask is not None:
                publish_subtask(next_subtask)

    if memory.is_done():
        print("All subtasks completed. Running final task validation.", file=sys.stderr)
        completion_frame_paths = [
            (item["id"], frames[item["frame_idx"] - 1])
            for item in memory.completed
            if 1 <= item.get("frame_idx", 0) <= len(frames)
        ]
        validation_start_time = time.time()
        validation_output, validation_response = validate_completed_task(
            client=client,
            endpoint=endpoint,
            model=model,
            sketch_image=sketch_image,
            initial_frame_path=frames[0],
            final_frame_path=last_processed_frame_path or frames[-1],
            memory=memory,
            completion_frame_paths=completion_frame_paths,
            max_tokens=args.validation_max_tokens,
        )
        validation_elapsed = time.time() - validation_start_time
        validation_record = {
            "validation": validation_output,
            "initial_frame_path": frames[0],
            "completion_frame_paths": [
                {"subtask_id": subtask_id, "frame_path": frame_path}
                for subtask_id, frame_path in completion_frame_paths
            ],
            "final_frame_path": last_processed_frame_path or frames[-1],
            "response_time_s": validation_elapsed,
            "generated_tokens": get_completion_tokens(endpoint, validation_response),
        }
        memory.validation = validation_output
        memory.save(args.memory_path)
        validation_path = Path(args.validation_output_path)
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        with validation_path.open("w", encoding="utf-8") as f:
            json.dump(validation_record, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(json.dumps(validation_output, ensure_ascii=False), flush=True)
        print(
            f"Final validation verdict: {validation_output.get('verdict', 'PARSE_ERROR')}",
            file=sys.stderr,
        )
    else:
        print("Frame stream ended before all subtasks completed.", file=sys.stderr)

    if generation_times:
        total_tokens = sum(generated_tokens)
        print(f"Total generated monitor tokens: {total_tokens}", file=sys.stderr)
        print(
            f"Average monitor response time: {sum(generation_times) / len(generation_times):.2f} s",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
