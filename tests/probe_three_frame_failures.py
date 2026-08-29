"""Evaluate RoboBrain failures with at most three images per inference.

This is a standalone diagnostic; it does not import or modify production code.
It compares the current single-image prompt with two temporal candidates:

* ``temporal-full``: three labelled, full-resolution observations.
* ``temporal-crop``: the same observations cropped around the interaction.
* ``forensic-crop``: a tighter crop plus an explicit failure-first checklist.
* ``checklist-crop``: the tight crop with observable per-flag decisions.
* ``sequenced-checklist-crop``: one labelled chat message per observation,
  avoiding loss of text/image interleaving in the Ollama message adapter.
* ``grounded-single``: a tight newest-frame crop with the same checklist,
  isolating temporal fusion from single-image visual grounding.
* ``state-tracking-crop``: requires an explicit state record for each
  observation before applying the failure checks.
* ``compact-tracking-crop``: records only three short trajectory states before
  returning the production status fields.

Examples:

    uv run python tests/probe_three_frame_failures.py --dry-run
    uv run python tests/probe_three_frame_failures.py --live \
        --modes baseline-single temporal-full temporal-crop
    uv run python tests/probe_three_frame_failures.py --live \
        --cases rotation-late drop-transition --modes temporal-crop
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import cv2
from langchain.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama


MAX_IMAGES = 3
DEFAULT_VIDEO = Path("video/failure_part.mp4")
DEFAULT_SYSTEM_PROMPT = Path("src/robobrain/prompt/brain_prompt.md")
DEFAULT_TASK_PROMPT = Path("src/robobrain/prompt/TASK.md")
DEFAULT_MODEL = "gemma4:31b-cloud"
DEFAULT_BASE_URL = "https://ollama.com"


TEMPORAL_RULES = """

# Temporal observation rules

- You receive one to three labelled observations of the same task attempt.
  They are ordered from OLDEST to NEWEST. Judge the status at NEWEST while
  using earlier observations as visual evidence of motion and transitions.
- A failure shown by any transition in the supplied observations must produce
  FAIL, even if the robot can recover or act again afterward.
- A plug rotates in the gripper when the plug changes orientation relative to
  the gripper jaws. Do not confuse this with rigid motion of the arm, gripper,
  and plug together.
- A plugging action misses the socket pinhole when the plug tip passes beside
  the intended opening, presses the socket housing, or remains uninserted
  after an insertion attempt.
- A plug drop is the transition from held by the gripper, to unsupported or
  falling, to resting on the table while the gripper is empty or retreats.
- Proximity to or occlusion by the socket is not evidence of COMPLETE. Return
  COMPLETE only when insertion is visually clear.
"""


FORENSIC_RULES = """

# Scene-specific visual grounding and decision order

- The movable plug is the small white object immediately below the black
  gripper/wrist. The larger white object with three round outlets is the
  stationary target socket strip; do not confuse it with the plug.
- First, silently track the small white plug in every observation relative to
  (a) the black gripper jaws, (b) the target outlet opening, and (c) the table.
- Then apply this strict priority: check every TASK_FAILURE_FLAG first; check
  COMPLETE second; use ON_GOING only after ruling out all visible failures.
- Motion blur of the small white plug between a held observation and an
  on-table observation is positive visual evidence that it fell. A retreating
  or empty gripper does not turn that event back into ON_GOING.
- This task never intentionally places the plug on the table. Therefore, a
  loose plug resting on the table while the gripper is empty or retreating is
  sufficient evidence for a drop failure even when the exact falling instant
  was not sampled.
- Do not call a drop merely because the plug contacts the socket housing while
  the gripper remains over or around it. A drop requires clear separation from
  the gripper plus an unsupported/falling plug or an empty/retreating gripper
  with the plug loose on the table. Otherwise evaluate rotation and socket
  miss first.
- During an insertion attempt, a changing plug angle relative to the gripper,
  or contact beside rather than inside the outlet opening, is a failure even
  while the robot remains near the socket.
"""


CHECKLIST_RULES = """

# Required failure checklist

The schema below supersedes the shorter response schema above for this probe.
Fill every check from visible evidence before choosing status. ``status`` MUST
be ``FAIL`` when any check is true; otherwise it may be ``COMPLETE`` or
``ON_GOING`` according to the task definitions.

```json
{
  "failure_checks": {
    "plug_dropped_or_falling": {
      "detected": "true | false",
      "evidence": "brief visible evidence or none"
    },
    "plug_rotated_relative_to_gripper": {
      "detected": "true | false",
      "evidence": "brief visible evidence or none"
    },
    "plug_about_to_drop": {
      "detected": "true | false",
      "evidence": "brief visible evidence or none"
    },
    "plug_missed_socket_opening": {
      "detected": "true | false",
      "evidence": "brief visible evidence or none"
    }
  },
  "status": "ON_GOING | COMPLETE | FAIL",
  "failure_description": "none | matching TASK_FAILURE_FLAG"
}
```
"""


STATE_TRACKING_RULES = """

# Required observation-by-observation tracking

The schema below supersedes every earlier response schema for this probe.
Inspect and record each observation separately before filling the failure
checks. Never copy a state from one observation into another. Use ``UNKNOWN``
when occlusion prevents a conclusion.

For ``plug_support``, choose exactly one of ``HELD_BY_GRIPPER``, ``FALLING``,
``LOOSE_ON_TABLE``, ``INSERTED_IN_SOCKET``, or ``UNKNOWN``. Describe
``plug_orientation`` relative to the black gripper jaws, not merely relative
to the camera.

```json
{
  "observations": {
    "OLDEST": {
      "plug_support": "state",
      "plug_orientation": "brief description",
      "socket_relation": "brief description"
    },
    "MIDDLE": {
      "plug_support": "state",
      "plug_orientation": "brief description",
      "socket_relation": "brief description"
    },
    "NEWEST": {
      "plug_support": "state",
      "plug_orientation": "brief description",
      "socket_relation": "brief description"
    }
  },
  "failure_checks": {
    "plug_dropped_or_falling": true,
    "plug_rotated_relative_to_gripper": false,
    "plug_about_to_drop": false,
    "plug_missed_socket_opening": false
  },
  "status": "FAIL",
  "failure_description": "matching TASK_FAILURE_FLAG"
}
```

Decision rules:

1. ``LOOSE_ON_TABLE`` with an empty or retreating gripper is a drop failure.
2. A change in plug orientation relative to the jaws is a rotation failure.
3. Contact beside the opening while the robot is still advancing or holding
   position is ``ON_GOING``. It becomes a miss only when earlier contact or
   approach is followed by clear offset motion or withdrawal without
   insertion.
4. If any failure check is true, ``status`` must be ``FAIL``.
"""


COMPACT_TRACKING_RULES = """

# Compact temporal state decision

This schema supersedes every earlier response schema for this probe. Inspect
each labelled observation independently and emit exactly one trajectory state
for each, in OLDEST-to-NEWEST order. Allowed states are:

``NOT_VISIBLE``, ``HELD_APPROACHING``, ``HELD_CONTACTING_OPENING``,
``HELD_OFFSET_FROM_OPENING``, ``HELD_WITHDRAWING_UNINSERTED``,
``ROTATED_IN_GRIPPER``, ``FALLING``, ``LOOSE_ON_TABLE``, and ``INSERTED``.

Use ``LOOSE_ON_TABLE`` only when the plug is physically loose on the flat
worktable in front of/below the socket strip and the gripper is clearly empty
or retreating. A plug contacting or lying against the socket housing while the
gripper remains over it is ``HELD_OFFSET_FROM_OPENING``, not
``LOOSE_ON_TABLE``.

Decision rules:

- ``HELD_APPROACHING`` or ``HELD_CONTACTING_OPENING`` at NEWEST is still
  ``ON_GOING``; incomplete insertion alone is not a failure.
- ``ROTATED_IN_GRIPPER``, ``FALLING``, or ``LOOSE_ON_TABLE`` in any supplied
  observation requires ``FAIL`` with the matching TASK_FAILURE_FLAG.
- Offset contact followed by withdrawal without insertion requires ``FAIL``
  for "plugging action misses the target socket pinhole".
- Three time-spaced observations that all remain
  ``HELD_CONTACTING_OPENING`` without insertion progress also require ``FAIL``
  for "plugging action misses the target socket pinhole". In contrast,
  approach followed by first contact at NEWEST remains ``ON_GOING``.
- ``COMPLETE`` requires ``INSERTED`` at NEWEST.

```json
{
  "trajectory": ["state", "state", "state"],
  "status": "ON_GOING | COMPLETE | FAIL",
  "failure_description": "none | matching TASK_FAILURE_FLAG"
}
```
"""


@dataclass(frozen=True)
class ProbeCase:
    name: str
    frame_indices: tuple[int, int, int]
    expected_status: str
    failure_terms: tuple[str, ...] = ()


CASES = {
    case.name: case
    for case in (
        ProbeCase(
            name="ongoing-control",
            frame_indices=(100, 280, 440),
            expected_status="ON_GOING",
        ),
        ProbeCase(
            name="rotation-early",
            frame_indices=(460, 590, 720),
            expected_status="FAIL",
            failure_terms=(
                "rotat",
                "miss",
                "pinhole",
                "misalign",
                "not insert",
                "outside",
            ),
        ),
        ProbeCase(
            name="rotation-late",
            frame_indices=(720, 850, 975),
            expected_status="FAIL",
            failure_terms=(
                "rotat",
                "miss",
                "pinhole",
                "misalign",
                "not insert",
                "outside",
            ),
        ),
        ProbeCase(
            name="rotation-span",
            frame_indices=(460, 720, 975),
            expected_status="FAIL",
            failure_terms=(
                "rotat",
                "miss",
                "pinhole",
                "misalign",
                "not insert",
                "outside",
            ),
        ),
        ProbeCase(
            name="drop-onset",
            frame_indices=(975, 980, 984),
            expected_status="FAIL",
            failure_terms=("drop", "fall", "table", "empty gripper"),
        ),
        ProbeCase(
            name="drop-transition",
            frame_indices=(975, 984, 996),
            expected_status="FAIL",
            failure_terms=("drop", "fall", "table", "empty gripper"),
        ),
        ProbeCase(
            name="drop-aftermath",
            frame_indices=(984, 996, 1023),
            expected_status="FAIL",
            failure_terms=("drop", "fall", "table", "empty gripper"),
        ),
    )
}

# A single deployable schedule that exposes both slow insertion/rotation
# failures and sub-second drops while keeping the payload at three images.
UNIFIED_LONG_LAG_FRAMES = 130
UNIFIED_SHORT_LAG_FRAMES = 12
UNIFIED_ROTATION_TERMS = (
    "rotat",
    "miss",
    "pinhole",
    "misalign",
    "not insert",
    "outside",
)
UNIFIED_DROP_TERMS = ("drop", "fall", "table", "empty gripper")


def unified_indices(current_frame: int) -> tuple[int, int, int]:
    return (
        current_frame - UNIFIED_LONG_LAG_FRAMES,
        current_frame - UNIFIED_SHORT_LAG_FRAMES,
        current_frame,
    )


for unified_case in (
    ProbeCase(
        name="uniform-control-440",
        frame_indices=unified_indices(440),
        expected_status="ON_GOING",
    ),
    ProbeCase(
        name="uniform-rotation-590",
        frame_indices=unified_indices(590),
        expected_status="FAIL",
        failure_terms=UNIFIED_ROTATION_TERMS,
    ),
    ProbeCase(
        name="uniform-rotation-720",
        frame_indices=unified_indices(720),
        expected_status="FAIL",
        failure_terms=UNIFIED_ROTATION_TERMS,
    ),
    ProbeCase(
        name="uniform-rotation-850",
        frame_indices=unified_indices(850),
        expected_status="FAIL",
        failure_terms=UNIFIED_ROTATION_TERMS,
    ),
    ProbeCase(
        name="uniform-rotation-975",
        frame_indices=unified_indices(975),
        expected_status="FAIL",
        failure_terms=UNIFIED_ROTATION_TERMS,
    ),
    ProbeCase(
        name="uniform-drop-984",
        frame_indices=unified_indices(984),
        expected_status="FAIL",
        failure_terms=UNIFIED_DROP_TERMS,
    ),
    ProbeCase(
        name="uniform-drop-996",
        frame_indices=unified_indices(996),
        expected_status="FAIL",
        failure_terms=UNIFIED_DROP_TERMS,
    ),
    ProbeCase(
        name="uniform-drop-1023",
        frame_indices=unified_indices(1023),
        expected_status="FAIL",
        failure_terms=UNIFIED_DROP_TERMS,
    ),
):
    CASES[unified_case.name] = unified_case

MODES = (
    "baseline-single",
    "grounded-single",
    "temporal-full",
    "temporal-crop",
    "forensic-crop",
    "checklist-crop",
    "sequenced-checklist-crop",
    "state-tracking-crop",
    "compact-tracking-crop",
)

SINGLE_IMAGE_MODES = {"baseline-single", "grounded-single"}


def parse_json_object(content: Any) -> dict[str, Any] | None:
    """Extract a JSON object from plain text or a Markdown code fence."""
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        text_parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        content = "\n".join(text_parts)
    if not isinstance(content, str):
        return None

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def read_frames(
    video_path: Path,
    wanted_indices: set[int],
) -> tuple[dict[int, Any], float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frames: dict[int, Any] = {}
    final_index = max(wanted_indices)
    frame_index = 0
    try:
        while frame_index <= final_index:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in wanted_indices:
                frames[frame_index] = frame.copy()
            frame_index += 1
    finally:
        capture.release()

    missing = sorted(wanted_indices.difference(frames))
    if missing:
        raise RuntimeError(f"Could not read requested frames: {missing}")
    if fps <= 0:
        raise RuntimeError(f"Video reported invalid FPS: {fps}")
    return frames, fps


def interaction_crop(frame: Any) -> Any:
    """Crop the gripper, plug, socket, and nearby support surface."""
    height, width = frame.shape[:2]
    left = round(width * 0.35)
    right = round(width * 0.86)
    bottom = round(height * 0.70)
    return frame[:bottom, left:right].copy()


def tight_interaction_crop(frame: Any) -> Any:
    """Tighter crop used to test whether object scale limits perception."""
    height, width = frame.shape[:2]
    left = round(width * 0.415)
    right = round(width * 0.79)
    top = round(height * 0.02)
    bottom = round(height * 0.62)
    return frame[top:bottom, left:right].copy()


def encode_jpeg(frame: Any) -> str:
    # The production build_message function also uses OpenCV's default JPEG
    # settings, keeping this comparison representative.
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Could not JPEG-encode an observation")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def image_part(frame: Any) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64," + encode_jpeg(frame),
        },
    }


def temporal_system_prompt(current_prompt: str) -> str:
    # Remove the direct contradiction before adding the temporal rules. This
    # only changes the in-memory probe prompt, never the production file.
    prompt = current_prompt.replace(
        "You will be given a single frame containing the robot and the workspace.",
        "You will be given one to three frames containing the robot and the workspace.",
    )
    return prompt + TEMPORAL_RULES


def build_messages(
    mode: str,
    case: ProbeCase,
    frames: dict[int, Any],
    fps: float,
    current_system_prompt: str,
    task_prompt: str,
) -> tuple[list[Any], list[tuple[int, Any]]]:
    observations = [(index, frames[index]) for index in case.frame_indices]
    if mode in SINGLE_IMAGE_MODES:
        observations = observations[-1:]
        if mode == "grounded-single":
            system_prompt = current_system_prompt + FORENSIC_RULES + CHECKLIST_RULES
            frame = tight_interaction_crop(observations[0][1])
        else:
            system_prompt = current_system_prompt
            frame = observations[0][1]
        content: list[dict[str, Any]] = [image_part(frame)]
    else:
        system_prompt = temporal_system_prompt(current_system_prompt)
        forensic_modes = {
            "forensic-crop",
            "checklist-crop",
            "sequenced-checklist-crop",
            "state-tracking-crop",
            "compact-tracking-crop",
        }
        checklist_modes = {
            "checklist-crop",
            "sequenced-checklist-crop",
        }
        if mode in forensic_modes:
            system_prompt += FORENSIC_RULES
        if mode in checklist_modes:
            system_prompt += CHECKLIST_RULES
        if mode == "state-tracking-crop":
            system_prompt += STATE_TRACKING_RULES
        if mode == "compact-tracking-crop":
            system_prompt += COMPACT_TRACKING_RULES
        newest_index = observations[-1][0]

        if mode in (
            "sequenced-checklist-crop",
            "state-tracking-crop",
            "compact-tracking-crop",
        ):
            messages: list[Any] = [SystemMessage(content=system_prompt)]
            for position, (frame_index, frame) in enumerate(
                observations,
                start=1,
            ):
                frame = tight_interaction_crop(frame)
                seconds_before_newest = (newest_index - frame_index) / fps
                age = (
                    "OLDEST"
                    if position == 1
                    else "NEWEST"
                    if position == 3
                    else "MIDDLE"
                )
                observation_content = [
                    {
                        "type": "text",
                        "text": (
                            f"Observation {position}/3: {age}; "
                            f"{seconds_before_newest:.2f} seconds before "
                            "NEWEST."
                        ),
                    },
                    image_part(frame),
                ]
                if position == 3:
                    observation_content.append(
                        {"type": "text", "text": task_prompt}
                    )
                messages.append(HumanMessage(content=observation_content))

            image_count = sum(
                part.get("type") == "image_url"
                for message in messages
                if isinstance(message.content, list)
                for part in message.content
            )
            if image_count > MAX_IMAGES:
                raise AssertionError(
                    f"Probe payload has {image_count} images; maximum is "
                    f"{MAX_IMAGES}"
                )
            return messages, observations

        content = [
            {
                "type": "text",
                "text": (
                    "Compare these observations of one continuous task "
                    "attempt. Their labels define chronological order."
                ),
            }
        ]
        for position, (frame_index, frame) in enumerate(observations, start=1):
            if mode == "temporal-crop":
                frame = interaction_crop(frame)
            elif mode in ("forensic-crop", "checklist-crop"):
                frame = tight_interaction_crop(frame)
            seconds_before_newest = (newest_index - frame_index) / fps
            age = "OLDEST" if position == 1 else "NEWEST" if position == 3 else "MIDDLE"
            content.extend(
                (
                    {
                        "type": "text",
                        "text": (
                            f"Observation {position}/3: {age}; "
                            f"{seconds_before_newest:.2f} seconds before NEWEST."
                        ),
                    },
                    image_part(frame),
                )
            )

    image_count = sum(part.get("type") == "image_url" for part in content)
    if image_count > MAX_IMAGES:
        raise AssertionError(
            f"Probe payload has {image_count} images; maximum is {MAX_IMAGES}"
        )
    content.append({"type": "text", "text": task_prompt})
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=content),
    ]
    return messages, observations


def score_response(
    case: ProbeCase,
    parsed: dict[str, Any] | None,
) -> dict[str, Any]:
    status = str(parsed.get("status", "")).upper() if parsed else ""
    description = str(parsed.get("failure_description", "")) if parsed else ""
    status_pass = status == case.expected_status
    description_pass = True
    if case.failure_terms:
        lowered_description = description.lower()
        description_pass = any(
            term in lowered_description for term in case.failure_terms
        )
    return {
        "status": status or None,
        "failure_description": description or None,
        "status_pass": status_pass,
        "description_pass": description_pass,
        "passed": status_pass and description_pass,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument(
        "--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT
    )
    parser.add_argument("--task-prompt", type=Path, default=DEFAULT_TASK_PROMPT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(CASES),
        default=list(CASES),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=list(MODES),
    )
    run_group = parser.add_mutually_exclusive_group(required=True)
    run_group.add_argument(
        "--live",
        action="store_true",
        help="Invoke the same Ollama cloud model used by production.",
    )
    run_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate extraction and payloads without model calls.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")

    selected_cases = [CASES[name] for name in args.cases]
    wanted_indices = {
        frame_index
        for case in selected_cases
        for frame_index in case.frame_indices
    }
    frames, fps = read_frames(args.video, wanted_indices)
    system_prompt = args.system_prompt.read_text(encoding="utf-8")
    task_prompt = args.task_prompt.read_text(encoding="utf-8")

    prepared: list[tuple[str, ProbeCase, list[Any], int]] = []
    for case in selected_cases:
        for mode in args.modes:
            messages, observations = build_messages(
                mode=mode,
                case=case,
                frames=frames,
                fps=fps,
                current_system_prompt=system_prompt,
                task_prompt=task_prompt,
            )
            image_count = 1 if mode in SINGLE_IMAGE_MODES else len(observations)
            prepared.append((mode, case, messages, image_count))
            print(
                json.dumps(
                    {
                        "type": "prepared",
                        "case": case.name,
                        "mode": mode,
                        "frame_indices": (
                            [case.frame_indices[-1]]
                            if mode in SINGLE_IMAGE_MODES
                            else list(case.frame_indices)
                        ),
                        "image_count": image_count,
                        "expected_status": case.expected_status,
                    }
                ),
                flush=True,
            )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "type": "summary",
                    "live": False,
                    "payloads": len(prepared),
                    "max_images_observed": max(item[3] for item in prepared),
                    "image_limit": MAX_IMAGES,
                }
            )
        )
        return 0

    model = ChatOllama(
        model=args.model,
        base_url=args.base_url,
        temperature=0,
        sync_client_kwargs={"timeout": args.timeout},
    )
    results: list[dict[str, Any]] = []
    for mode, case, messages, image_count in prepared:
        for repeat in range(1, args.repeats + 1):
            start = time.perf_counter()
            response = model.invoke(messages)
            latency_seconds = time.perf_counter() - start
            parsed = parse_json_object(response.content)
            score = score_response(case, parsed)
            result = {
                "type": "result",
                "case": case.name,
                "mode": mode,
                "repeat": repeat,
                "frame_indices": (
                    [case.frame_indices[-1]]
                    if mode in SINGLE_IMAGE_MODES
                    else list(case.frame_indices)
                ),
                "image_count": image_count,
                "expected_status": case.expected_status,
                "latency_seconds": round(latency_seconds, 3),
                **score,
                "raw_response": response.content,
            }
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

    summary: dict[str, Any] = {}
    for mode in args.modes:
        mode_results = [result for result in results if result["mode"] == mode]
        summary[mode] = {
            "passed": sum(bool(result["passed"]) for result in mode_results),
            "attempts": len(mode_results),
            "pass_rate": (
                sum(bool(result["passed"]) for result in mode_results)
                / len(mode_results)
            ),
            "mean_latency_seconds": round(
                sum(float(result["latency_seconds"]) for result in mode_results)
                / len(mode_results),
                3,
            ),
        }
    print(
        json.dumps(
            {
                "type": "summary",
                "live": True,
                "model": args.model,
                "image_limit": MAX_IMAGES,
                "by_mode": summary,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
