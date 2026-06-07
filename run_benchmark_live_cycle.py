"""
Streaming benchmark for OpenAI-compatible chat-completions vision models.

This version is designed for your real deployment case:
- static reference image/stage prefix is reused;
- live image changes on each request;
- llama.cpp server timings are captured when the backend is local llama-server.

Local llama.cpp example:
    python run_benchmark_live_cycle.py \
      --runs 11 \
      --warmup 1 \
      --base-url http://127.0.0.1:8080/v1 \
      --model gemma4-12b-q4 \
      --backend llama \
      --image-transport file-url \
      --live-dir /home/yuxin/Documents/pap_stage_demo/test_images_256px_jpg \
      --live-glob '*.jpg' \
      --layout user_refs \
      --label-csv /home/yuxin/Documents/pap_stage_demo/pap_test_labels.csv \
      --out-dir benchmark_logs_gemma4-12bq4it

OpenAI cloud example:
    python run_benchmark_live_cycle.py \
      --runs 11 \
      --warmup 0 \
      --model gpt-5.4-mini \
      --backend openai \
      --image-transport data-url \
      --live-dir /home/yuxin/Documents/pap_stage_demo/test_images_512px_jpg \
      --live-glob '*.jpg' \
      --label-csv /home/yuxin/Documents/pap_stage_demo/pap_test_labels.csv \
      --out-dir benchmark_logs_openai_gpt54mini

Recommended server flags:
    ./build/bin/llama-server ... -np 1 --cache-prompt --metrics --jinja
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
from openai import OpenAI

from gemma4_messages_loop import build_messages, discover_live_images


DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:8080/v1"


# -----------------------------
# Generic utilities
# -----------------------------

def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def summarize(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return {
            "n": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "stdev": None,
        }

    return {
        "n": len(xs),
        "min": min(xs),
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "p90": percentile(xs, 90),
        "p95": percentile(xs, 95),
        "p99": percentile(xs, 99),
        "max": max(xs),
        "stdev": statistics.stdev(xs) if len(xs) >= 2 else 0.0,
    }


def safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or den == 0:
        return None
    return num / den


def nested_get(d: Optional[Dict[str, Any]], *keys: str) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def first_present(d: Optional[Dict[str, Any]], candidates: List[Tuple[str, ...]]) -> Any:
    for path in candidates:
        val = nested_get(d, *path)
        if val is not None:
            return val
    return None


def as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def as_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def optional_float(value: str) -> Optional[float]:
    if value.lower() in {"none", "null", "omit"}:
        return None
    return float(value)


def optional_temperature(value: str) -> Optional[float] | str:
    if value.lower() == "auto":
        return "auto"
    return optional_float(value)


def strip_trailing_slash(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return value.rstrip("/")


def infer_backend(base_url: Optional[str], backend: str) -> str:
    if backend != "auto":
        return backend
    if base_url is None:
        return "openai"

    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    if host in {"127.0.0.1", "localhost", "::1"}:
        return "llama"
    return "openai"


def infer_image_transport(image_transport: str, backend: str) -> str:
    if image_transport != "auto":
        return image_transport
    return "file-url" if backend == "llama" else "data-url"


def infer_max_tokens_param(max_tokens_param: str, backend: str) -> str:
    if max_tokens_param != "auto":
        return max_tokens_param
    return "max_tokens" if backend == "llama" else "max_completion_tokens"


def infer_temperature(temperature: Optional[float] | str, backend: str) -> Optional[float]:
    if temperature != "auto":
        return temperature
    return 0.0 if backend == "llama" else None


def metrics_base_from_openai_base_url(base_url: Optional[str]) -> Optional[str]:
    if base_url is None:
        return None
    base_url = strip_trailing_slash(base_url)
    if base_url is None:
        return None
    return base_url[:-3] if base_url.endswith("/v1") else base_url


def resolve_api_key(api_key: Optional[str], backend: str) -> str:
    if api_key:
        return api_key
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key
    if backend == "llama":
        return "sk-no-key-required"
    raise RuntimeError("OPENAI_API_KEY is not set. Pass --api-key or export OPENAI_API_KEY.")


def create_chat_client(api_key: str, base_url: Optional[str]) -> OpenAI:
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def model_dump_or_dict(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def get_extra_field(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    direct = getattr(value, key, None)
    if direct is not None:
        return direct
    model_extra = getattr(value, "model_extra", None)
    if isinstance(model_extra, dict):
        return model_extra.get(key)
    return None


# -----------------------------
# Output parsing / task metrics
# -----------------------------

def parse_number(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", str(text))
    if not m:
        return None
    return float(m.group(1))


def parse_stage_num(text: Optional[str]) -> Optional[int]:
    if text is None:
        return None
    m = re.search(r"Stage\s+(\d+)", str(text), flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    n = parse_number(str(text))
    return None if n is None else int(n)


def parse_robot_bracket_response(text: str) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {
        "task_progress": None,
        "task_progress_pct": None,
        "task_stage": None,
        "task_stage_num": None,
        "status": None,
        "probability": None,
        "probability_pct": None,
        "confidence": None,
        "confidence_pct": None,
        "anomaly": None,
        "schema_valid": False,
    }

    key_values: Dict[str, str] = {}
    bare_status: Optional[str] = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m_kv = re.fullmatch(r"\[(.+?):\s*(.*?)\]", line)
        if m_kv:
            key_values[m_kv.group(1).strip().lower()] = m_kv.group(2).strip()
            continue

        m_bare = re.fullmatch(r"\[(Success|Failure)\]", line, flags=re.IGNORECASE)
        if m_bare:
            bare_status = m_bare.group(1).capitalize()

    progress = key_values.get("progress") or key_values.get("task progress")
    stage = key_values.get("stage") or key_values.get("task stage")
    status = key_values.get("status") or bare_status
    probability = key_values.get("probability")
    confidence = key_values.get("confidence") or probability
    anomaly = key_values.get("anomaly")
    if anomaly is None:
        anomaly = key_values.get("something else")

    parsed["task_progress"] = progress
    parsed["task_stage"] = stage
    parsed["status"] = status
    parsed["probability"] = probability
    parsed["confidence"] = confidence
    parsed["anomaly"] = anomaly

    parsed["task_progress_pct"] = parse_number(progress)
    parsed["task_stage_num"] = parse_stage_num(stage)
    parsed["probability_pct"] = parse_number(probability)
    parsed["confidence_pct"] = parse_number(confidence)

    old_schema_valid = all(
        parsed[k] is not None
        for k in ["task_progress", "task_stage", "status", "probability", "anomaly"]
    )
    compact_schema_valid = (
        parsed["task_stage_num"] is not None
        and parsed["confidence_pct"] is not None
        and parsed["anomaly"] is not None
    )
    parsed["schema_valid"] = old_schema_valid or compact_schema_valid

    return parsed


def load_stage_label_csv(csv_path: str | Path) -> Dict[str, int]:
    path = Path(csv_path)
    labels: Dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return labels

        normalized_header = [h.strip().lower() for h in header]
        try:
            frame_idx = normalized_header.index("frame_name")
            stage_idx = normalized_header.index("stage")
        except ValueError as e:
            raise ValueError(
                f"{path} must contain 'frame_name' and 'stage' columns"
            ) from e

        for row_num, row in enumerate(reader, start=2):
            if not row or len(row) <= max(frame_idx, stage_idx):
                continue
            frame_name = row[frame_idx].strip()
            stage_text = row[stage_idx].strip()
            if not frame_name and not stage_text:
                continue
            if not frame_name or not stage_text:
                raise ValueError(f"Incomplete label row {row_num} in {path}: {row}")
            try:
                labels[frame_name.lower()] = int(stage_text)
            except ValueError as e:
                raise ValueError(f"Invalid stage on row {row_num} in {path}: {stage_text!r}") from e

    return labels


def lookup_stage_label(labels: Dict[str, int], image_path: Path) -> Optional[int]:
    if not labels:
        return None

    exact = image_path.name.lower()
    if exact in labels:
        return labels[exact]

    stem = image_path.stem.lower()
    for label_name, stage in labels.items():
        if Path(label_name).stem.lower() == stem:
            return stage

    return None


def add_correctness_fields(
    result: Dict[str, Any],
    expected_stage_num: Optional[int],
    expected_status: Optional[str],
    expected_progress_pct: Optional[float],
) -> None:
    parsed = result.get("parsed_output", {}) or {}

    pred_stage = parsed.get("task_stage_num")
    pred_status = parsed.get("status")
    pred_progress = parsed.get("task_progress_pct")

    result["expected_stage_num"] = expected_stage_num
    result["expected_status"] = expected_status
    result["expected_progress_pct"] = expected_progress_pct

    result["predicted_stage_num"] = pred_stage
    result["predicted_status"] = pred_status
    result["predicted_progress_pct"] = pred_progress
    result["predicted_probability_pct"] = parsed.get("probability_pct")
    result["predicted_confidence_pct"] = parsed.get("confidence_pct")
    result["schema_valid"] = parsed.get("schema_valid")

    result["stage_correct"] = (
        None if expected_stage_num is None or pred_stage is None else pred_stage == expected_stage_num
    )
    result["status_correct"] = (
        None
        if expected_status is None or pred_status is None
        else pred_status.lower() == expected_status.lower()
    )
    result["progress_abs_error_pct"] = (
        None
        if expected_progress_pct is None or pred_progress is None
        else abs(float(pred_progress) - float(expected_progress_pct))
    )


# -----------------------------
# Server helper endpoints
# -----------------------------

def count_input_tokens(base: str, model: str, messages: List[Dict[str, Any]], max_tokens: int) -> Optional[int]:
    url = f"{base}/v1/chat/completions/input_tokens"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        with httpx.Client(timeout=None) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            return data.get("input_tokens")
    except Exception as e:
        print(f"[warn] input token count failed: {e}")
        return None


def count_output_tokens_text_only(base: str, text: str) -> Optional[int]:
    try:
        with httpx.Client(timeout=None) as client:
            r = client.post(
                f"{base}/tokenize",
                json={
                    "content": text,
                    "add_special": False,
                    "parse_special": True,
                },
            )
            r.raise_for_status()
            return len(r.json()["tokens"])
    except Exception as e:
        print(f"[warn] output tokenization failed: {e}")
        return None


def get_prometheus_metrics(base: str) -> Optional[str]:
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(f"{base}/metrics")
            r.raise_for_status()
            return r.text
    except Exception:
        return None


def parse_metric_value(metrics_text: Optional[str], metric_name: str) -> Optional[float]:
    if not metrics_text:
        return None
    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(metric_name):
            try:
                return float(line.split()[-1])
            except Exception:
                return None
    return None


# -----------------------------
# GPU telemetry
# -----------------------------

def sample_nvidia_smi(stop_event: threading.Event, samples: List[Dict[str, Any]], interval_s: float = 0.05) -> None:
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]

    while not stop_event.is_set():
        t = time.perf_counter_ns()
        try:
            out = subprocess.check_output(cmd, text=True).strip()
            line = out.splitlines()[0]
            row = next(csv.reader([line], skipinitialspace=True))
            samples.append({
                "t_ns": t,
                "timestamp": row[0],
                "gpu_index": int(row[1]),
                "gpu_name": row[2],
                "gpu_util_pct": float(row[3]),
                "mem_util_pct": float(row[4]),
                "mem_used_mb": float(row[5]),
                "mem_total_mb": float(row[6]),
                "power_w": float(row[7]),
                "temp_c": float(row[8]),
            })
        except Exception:
            pass
        time.sleep(interval_s)


def add_gpu_summary(result: Dict[str, Any], samples: List[Dict[str, Any]]) -> None:
    result["gpu_samples_n"] = len(samples)
    if not samples:
        result["gpu_summary"] = None
        result["gpu_util_mean_pct"] = None
        result["gpu_util_max_pct"] = None
        result["mem_used_max_mb"] = None
        result["power_mean_w"] = None
        result["power_max_w"] = None
        result["temp_max_c"] = None
        return

    summary = {
        "gpu_util_mean_pct": statistics.mean(s["gpu_util_pct"] for s in samples),
        "gpu_util_max_pct": max(s["gpu_util_pct"] for s in samples),
        "mem_used_max_mb": max(s["mem_used_mb"] for s in samples),
        "power_mean_w": statistics.mean(s["power_w"] for s in samples),
        "power_max_w": max(s["power_w"] for s in samples),
        "temp_max_c": max(s["temp_c"] for s in samples),
    }
    result["gpu_summary"] = summary
    result.update(summary)


# -----------------------------
# Benchmark request
# -----------------------------

def extract_server_timing_fields(result: Dict[str, Any]) -> None:
    timings = result.get("server_timings")

    cache_n = first_present(timings, [("cache_n",), ("cache_tokens",)])
    prompt_n = first_present(timings, [("prompt_n",), ("n_prompt_tokens",)])
    predicted_n = first_present(timings, [("predicted_n",), ("n_predicted",), ("completion_n",)])
    prompt_ms = first_present(timings, [("prompt_ms",), ("prompt_time_ms",)])
    predicted_ms = first_present(timings, [("predicted_ms",), ("predicted_time_ms",)])

    result["cache_n"] = as_int(cache_n)
    result["prompt_n"] = as_int(prompt_n)
    result["predicted_n"] = as_int(predicted_n)
    result["prompt_ms"] = as_float(prompt_ms)
    result["predicted_ms"] = as_float(predicted_ms)

    result["cache_hit_ratio"] = safe_div(
        result["cache_n"],
        (result["cache_n"] or 0) + (result["prompt_n"] or 0),
    )

    prompt_s = None if result["prompt_ms"] is None else result["prompt_ms"] / 1000.0
    predicted_s = None if result["predicted_ms"] is None else result["predicted_ms"] / 1000.0

    result["server_prompt_s"] = prompt_s
    result["server_predicted_s"] = predicted_s
    result["server_model_time_s"] = (
        None if prompt_s is None or predicted_s is None else prompt_s + predicted_s
    )
    result["client_unaccounted_s"] = (
        None if result["server_model_time_s"] is None else result["total_latency_s"] - result["server_model_time_s"]
    )
    result["ttft_minus_prompt_s"] = (
        None if prompt_s is None or result["ttft_s"] is None else result["ttft_s"] - prompt_s
    )

    result["server_prompt_tps"] = safe_div(result["prompt_n"], prompt_s)
    result["server_predicted_tps"] = safe_div(result["predicted_n"], predicted_s)


def benchmark_streaming_request(
    *,
    client: OpenAI,
    backend: str,
    base_url: Optional[str],
    metrics_base_url: Optional[str],
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    max_tokens_param: str,
    temperature: Optional[float],
    extra_body: Optional[Dict[str, Any]],
    request_id: str,
    live_image_path: Path,
    live_image_index: int,
    image_transport: str,
    input_tokens: Optional[int] = None,
    use_gpu_telemetry: bool = False,
    expected_stage_num: Optional[int] = None,
    expected_status: Optional[str] = None,
    expected_progress_pct: Optional[float] = None,
) -> Dict[str, Any]:
    request_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request_kwargs[max_tokens_param] = max_tokens
    if temperature is not None:
        request_kwargs["temperature"] = temperature
    if extra_body:
        request_kwargs["extra_body"] = extra_body

    generated_parts: List[str] = []
    chunk_times_ns: List[int] = []
    content_chunk_times_ns: List[int] = []

    first_chunk_ns = None
    first_content_ns = None
    last_content_ns = None
    finish_ns = None

    finish_reason = None
    usage = None
    timings = None
    response_id = None

    gpu_samples: List[Dict[str, Any]] = []
    stop_event = threading.Event()
    gpu_thread: Optional[threading.Thread] = None

    if use_gpu_telemetry:
        gpu_thread = threading.Thread(
            target=sample_nvidia_smi,
            args=(stop_event, gpu_samples),
            daemon=True,
        )
        gpu_thread.start()

    metrics_before = get_prometheus_metrics(metrics_base_url) if metrics_base_url else None
    t0 = time.perf_counter_ns()

    try:
        stream = client.chat.completions.create(**request_kwargs)
        for chunk in stream:
            now_ns = time.perf_counter_ns()
            chunk_times_ns.append(now_ns)

            if first_chunk_ns is None:
                first_chunk_ns = now_ns
            if response_id is None:
                response_id = get_extra_field(chunk, "id")

            chunk_usage = get_extra_field(chunk, "usage")
            if chunk_usage is not None:
                usage = model_dump_or_dict(chunk_usage)

            chunk_timings = get_extra_field(chunk, "timings")
            if chunk_timings is not None:
                timings = model_dump_or_dict(chunk_timings)

            choices = get_extra_field(chunk, "choices") or []
            if not choices:
                continue

            choice = choices[0]
            chunk_finish_reason = get_extra_field(choice, "finish_reason")
            if chunk_finish_reason is not None:
                finish_reason = chunk_finish_reason

            delta = get_extra_field(choice, "delta") or {}
            content = get_extra_field(delta, "content")
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict)
                )
            if content:
                if first_content_ns is None:
                    first_content_ns = now_ns
                last_content_ns = now_ns
                content_chunk_times_ns.append(now_ns)
                generated_parts.append(str(content))
    finally:
        if finish_ns is None:
            finish_ns = time.perf_counter_ns()
        if use_gpu_telemetry:
            stop_event.set()
            if gpu_thread is not None:
                gpu_thread.join(timeout=1.0)

    metrics_after = get_prometheus_metrics(metrics_base_url) if metrics_base_url else None
    text = "".join(generated_parts)

    total_s = (finish_ns - t0) / 1e9
    time_to_first_chunk_s = None if first_chunk_ns is None else (first_chunk_ns - t0) / 1e9
    ttft_s = None if first_content_ns is None else (first_content_ns - t0) / 1e9
    decode_s = None if first_content_ns is None else (finish_ns - first_content_ns) / 1e9
    last_token_latency_s = None if last_content_ns is None else (finish_ns - last_content_ns) / 1e9

    inter_content_gaps_s = [
        (b - a) / 1e9
        for a, b in zip(content_chunk_times_ns, content_chunk_times_ns[1:])
    ]

    generated_chars = len(text)
    generated_lines = len([ln for ln in text.splitlines() if ln.strip()])

    completion_tokens_from_usage = None
    prompt_tokens_from_usage = None
    total_tokens_from_usage = None
    if usage:
        completion_tokens_from_usage = usage.get("completion_tokens")
        prompt_tokens_from_usage = usage.get("prompt_tokens")
        total_tokens_from_usage = usage.get("total_tokens")

    output_tokens_est = completion_tokens_from_usage
    if output_tokens_est is None and metrics_base_url:
        output_tokens_est = count_output_tokens_text_only(metrics_base_url, text)

    pred_total_before = parse_metric_value(metrics_before, "llamacpp:tokens_predicted_total")
    pred_total_after = parse_metric_value(metrics_after, "llamacpp:tokens_predicted_total")
    prompt_total_before = parse_metric_value(metrics_before, "llamacpp:prompt_tokens_total")
    prompt_total_after = parse_metric_value(metrics_after, "llamacpp:prompt_tokens_total")

    result: Dict[str, Any] = {
        "request_id": request_id,
        "response_id": response_id,
        "live_image_index": live_image_index,
        "live_image_path": str(live_image_path),
        "live_image_name": live_image_path.name,
        "backend": backend,
        "base_url": base_url,
        "model": model,
        "image_transport": image_transport,
        "max_tokens_param": max_tokens_param,
        "temperature": temperature,
        "finish_reason": finish_reason,
        "input_tokens": input_tokens,

        "total_latency_s": total_s,
        "time_to_first_sse_chunk_s": time_to_first_chunk_s,
        "ttft_s": ttft_s,
        "decode_time_s": decode_s,
        "last_token_to_done_s": last_token_latency_s,
        "ttft_to_decode_ratio": safe_div(ttft_s, decode_s),
        "prefill_share_of_total": safe_div(ttft_s, total_s),
        "decode_share_of_total": safe_div(decode_s, total_s),

        "num_sse_chunks": len(chunk_times_ns),
        "num_content_chunks": len(content_chunk_times_ns),
        "inter_content_gap_mean_s": statistics.mean(inter_content_gaps_s) if inter_content_gaps_s else None,
        "inter_content_gap_median_s": statistics.median(inter_content_gaps_s) if inter_content_gaps_s else None,
        "inter_content_gap_p90_s": percentile(inter_content_gaps_s, 90),
        "inter_content_gap_p95_s": percentile(inter_content_gaps_s, 95),
        "inter_content_gap_p99_s": percentile(inter_content_gaps_s, 99),
        "inter_content_gap_max_s": max(inter_content_gaps_s) if inter_content_gaps_s else None,

        "generated_chars": generated_chars,
        "generated_lines": generated_lines,
        "output_tokens": output_tokens_est,
        "prompt_tokens_from_usage": prompt_tokens_from_usage,
        "completion_tokens_from_usage": completion_tokens_from_usage,
        "total_tokens_from_usage": total_tokens_from_usage,
        "tokens_per_second_e2e": safe_div(output_tokens_est, total_s),
        "tokens_per_second_decode": safe_div(output_tokens_est, decode_s),
        "chars_per_second_e2e": safe_div(generated_chars, total_s),

        "server_timings": timings,
        "server_usage": usage,
        "prometheus_prompt_tokens_delta": (
            None if prompt_total_before is None or prompt_total_after is None else prompt_total_after - prompt_total_before
        ),
        "prometheus_predicted_tokens_delta": (
            None if pred_total_before is None or pred_total_after is None else pred_total_after - pred_total_before
        ),
        "text": text,
    }

    extract_server_timing_fields(result)
    result["parsed_output"] = parse_robot_bracket_response(text)
    add_correctness_fields(result, expected_stage_num, expected_status, expected_progress_pct)
    add_gpu_summary(result, gpu_samples)

    return result


# -----------------------------
# Export / summary
# -----------------------------

def flatten_for_csv(row: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in row.items():
        if key in {"server_timings", "server_usage", "parsed_output", "gpu_summary"}:
            flat[key] = json.dumps(value, ensure_ascii=False)
        elif key == "text":
            flat[key] = value.replace("\n", "\\n") if isinstance(value, str) else value
        else:
            flat[key] = value
    return flat


def save_results(out_dir: Path, runs: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "runs.jsonl"
    csv_path = out_dir / "runs.csv"
    summary_path = out_dir / "summary.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in runs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if runs:
        flat_rows = [flatten_for_csv(r) for r in runs]
        all_keys = sorted({k for row in flat_rows for k in row.keys()})
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(flat_rows)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def make_summary(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "total_latency_s",
        "ttft_s",
        "decode_time_s",
        "ttft_to_decode_ratio",
        "prefill_share_of_total",
        "tokens_per_second_decode",
        "tokens_per_second_e2e",
        "inter_content_gap_p95_s",
        "cache_hit_ratio",
        "cache_n",
        "prompt_n",
        "predicted_n",
        "prompt_ms",
        "predicted_ms",
        "server_model_time_s",
        "client_unaccounted_s",
        "ttft_minus_prompt_s",
        "prometheus_prompt_tokens_delta",
        "prometheus_predicted_tokens_delta",
        "mem_used_max_mb",
        "gpu_util_mean_pct",
        "gpu_util_max_pct",
        "power_mean_w",
        "temp_max_c",
        "input_tokens",
        "output_tokens",
    ]

    summary: Dict[str, Any] = {key: summarize([r.get(key) for r in runs]) for key in keys}

    # Per-image latency summary helps identify image-specific outliers.
    by_image: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        by_image.setdefault(r["live_image_name"], []).append(r)

    summary["by_live_image"] = {
        name: {
            "n": len(rows),
            "total_latency_s": summarize([x.get("total_latency_s") for x in rows]),
            "ttft_s": summarize([x.get("ttft_s") for x in rows]),
            "prompt_ms": summarize([x.get("prompt_ms") for x in rows]),
            "prompt_n": summarize([x.get("prompt_n") for x in rows]),
            "stage_accuracy": (
                sum(1 for x in rows if x.get("stage_correct") is True) /
                max(1, sum(1 for x in rows if x.get("stage_correct") is not None))
            ),
        }
        for name, rows in by_image.items()
    }

    schema_vals = [r.get("schema_valid") for r in runs]
    stage_vals = [r.get("stage_correct") for r in runs if r.get("stage_correct") is not None]
    status_vals = [r.get("status_correct") for r in runs if r.get("status_correct") is not None]
    progress_errs = [r.get("progress_abs_error_pct") for r in runs if r.get("progress_abs_error_pct") is not None]

    summary["accuracy"] = {
        "schema_valid_rate": sum(1 for v in schema_vals if v is True) / len(schema_vals) if schema_vals else None,
        "stage_accuracy_n": len(stage_vals),
        "stage_accuracy": sum(1 for v in stage_vals if v is True) / len(stage_vals) if stage_vals else None,
        "status_accuracy": sum(1 for v in status_vals if v is True) / len(status_vals) if status_vals else None,
        "progress_abs_error_pct": summarize(progress_errs),
    }

    stage_confusion: Dict[str, int] = {}
    for r in runs:
        expected = r.get("expected_stage_num")
        predicted = r.get("predicted_stage_num")
        if expected is None or predicted is None:
            continue
        key = f"{expected}->{predicted}"
        stage_confusion[key] = stage_confusion.get(key, 0) + 1
    summary["stage_confusion"] = stage_confusion

    return summary


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base-url",
        "--base",
        dest="base_url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help=(
            "OpenAI-compatible API base URL. Omit for OpenAI cloud. "
            f"Use {DEFAULT_LOCAL_BASE_URL} for local llama-server."
        ),
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="API key. Defaults to OPENAI_API_KEY, or sk-no-key-required for --backend llama.",
    )
    p.add_argument(
        "--backend",
        choices=["auto", "openai", "llama"],
        default="auto",
        help="Backend behavior. auto treats localhost base URLs as llama.cpp and no base URL as OpenAI cloud.",
    )
    p.add_argument("--model", required=True, help="Model name to use in the API requests. [gpt-5.4-mini, gpt-5,4-nano, gemma4-12b-it]")
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument(
        "--max-tokens-param",
        choices=["auto", "max_tokens", "max_completion_tokens"],
        default="auto",
        help="Chat Completions token-limit field to send. auto uses max_completion_tokens for OpenAI cloud.",
    )
    p.add_argument(
        "--temperature",
        type=optional_temperature,
        default="auto",
        help="Sampling temperature. auto uses 0.0 for llama.cpp and omits temperature for OpenAI cloud.",
    )
    p.add_argument("--image-dir", default="/home/yuxin/Documents/pap_stage_demo")
    p.add_argument("--live-dir", default="/home/yuxin/Documents/pap_stage_demo/test_images")
    p.add_argument("--live-glob", default="*.png")
    p.add_argument("--layout", choices=["user_refs", "system_refs"], default="user_refs")
    p.add_argument(
        "--image-transport",
        choices=["auto", "file-url", "data-url"],
        default="auto",
        help="auto uses file:// for local llama.cpp and base64 data URLs for OpenAI cloud.",
    )
    p.add_argument(
        "--image-detail",
        choices=["auto", "low", "high"],
        default=None,
        help="Optional OpenAI image detail hint.",
    )
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--count-input-tokens", action="store_true")
    p.add_argument(
        "--label-csv",
        default=None,
        help=(
            "CSV with frame_name and stage columns. If omitted, "
            "<image-dir>/pap_test_labels.csv is used when present."
        ),
    )
    p.add_argument("--expected-stage", type=int, default=None)
    p.add_argument("--expected-status", default=None)
    p.add_argument("--expected-progress", type=float, default=None)
    p.add_argument("--out-dir", default="benchmark_logs_live_cycle")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_url = strip_trailing_slash(args.base_url)
    backend = infer_backend(base_url, args.backend)
    if backend == "llama" and base_url is None:
        base_url = DEFAULT_LOCAL_BASE_URL

    image_transport = infer_image_transport(args.image_transport, backend)
    if backend == "openai" and image_transport == "file-url":
        raise ValueError("OpenAI cloud cannot read local file:// image URLs. Use --image-transport data-url.")
    max_tokens_param = infer_max_tokens_param(args.max_tokens_param, backend)
    temperature = infer_temperature(args.temperature, backend)

    api_key = resolve_api_key(args.api_key, backend)
    client = create_chat_client(api_key, base_url)
    metrics_base_url = metrics_base_from_openai_base_url(base_url) if backend == "llama" else None
    llama_extra_body = (
        {
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if backend == "llama"
        else None
    )

    live_paths = discover_live_images(args.live_dir, args.live_glob)
    if not live_paths:
        raise FileNotFoundError(f"No live images found in {args.live_dir!r} with glob {args.live_glob!r}")

    label_csv = Path(args.label_csv) if args.label_csv else Path(args.image_dir) / "pap_test_labels.csv"
    stage_labels: Dict[str, int] = {}
    if label_csv.exists():
        stage_labels = load_stage_label_csv(label_csv)
    elif args.label_csv:
        raise FileNotFoundError(f"Label CSV not found: {label_csv}")

    print(f"Found {len(live_paths)} live images:")
    for i, p in enumerate(live_paths):
        print(f"  {i:02d}: {p.name}")
    print()
    print(f"backend: {backend}")
    print(f"base_url: {base_url or 'OpenAI default'}")
    print(f"model: {args.model}")
    print(f"image_transport: {image_transport}")
    print(f"max_tokens_param: {max_tokens_param}")
    print(f"temperature: {temperature}")
    print(f"layout: {args.layout}")
    print(f"runs: {args.runs}, warmup: {args.warmup}")
    if stage_labels:
        print(f"label_csv: {label_csv} ({len(stage_labels)} labels)")
        missing_labels = [p.name for p in live_paths if lookup_stage_label(stage_labels, p) is None]
        if missing_labels:
            print(f"[warn] {len(missing_labels)} live images have no stage label: {missing_labels}")
    elif args.expected_stage is not None:
        print("[warn] using one fixed --expected-stage for every image")
    else:
        print("[warn] no stage labels loaded; stage accuracy will be null")
    print()

    # Warmup requests populate static-prefix cache and trigger CUDA/kernel init.
    for wi in range(args.warmup):
        idx = wi % len(live_paths)
        live_path = live_paths[idx]
        messages = build_messages(
            live_path,
            image_dir=args.image_dir,
            layout=args.layout,
            image_transport=image_transport,
            image_detail=args.image_detail,
        )
        expected_stage = lookup_stage_label(stage_labels, live_path)
        if expected_stage is None:
            expected_stage = args.expected_stage
        _ = benchmark_streaming_request(
            client=client,
            backend=backend,
            base_url=base_url,
            metrics_base_url=metrics_base_url,
            model=args.model,
            messages=messages,
            max_tokens=args.max_tokens,
            max_tokens_param=max_tokens_param,
            temperature=temperature,
            extra_body=llama_extra_body,
            request_id=f"warmup-{wi:03d}",
            live_image_path=live_path,
            live_image_index=idx,
            image_transport=image_transport,
            use_gpu_telemetry=False,
            expected_stage_num=expected_stage,
            expected_status=args.expected_status,
            expected_progress_pct=args.expected_progress,
        )

    runs: List[Dict[str, Any]] = []

    for i in range(args.runs):
        # Offset by warmup so the first measured run is not usually identical to the final warmup request.
        idx = (args.warmup + i) % len(live_paths)
        live_path = live_paths[idx]

        # Build messages before timing the HTTP/model request.
        # This keeps client-side image reading/base64 encoding out of total_latency_s,
        # but we still record it as client_prepare_s.
        prep_t0 = time.perf_counter_ns()
        messages = build_messages(
            live_path,
            image_dir=args.image_dir,
            layout=args.layout,
            image_transport=image_transport,
            image_detail=args.image_detail,
        )
        prep_t1 = time.perf_counter_ns()
        client_prepare_s = (prep_t1 - prep_t0) / 1e9

        input_tokens = None
        if args.count_input_tokens and metrics_base_url:
            input_tokens = count_input_tokens(metrics_base_url, args.model, messages, args.max_tokens)
        elif args.count_input_tokens:
            print("[warn] --count-input-tokens is only implemented for llama-server local endpoints")

        expected_stage = lookup_stage_label(stage_labels, live_path)
        if expected_stage is None:
            expected_stage = args.expected_stage

        r = benchmark_streaming_request(
            client=client,
            backend=backend,
            base_url=base_url,
            metrics_base_url=metrics_base_url,
            model=args.model,
            messages=messages,
            max_tokens=args.max_tokens,
            max_tokens_param=max_tokens_param,
            temperature=temperature,
            extra_body=llama_extra_body,
            request_id=f"bench-{i:03d}",
            live_image_path=live_path,
            live_image_index=idx,
            image_transport=image_transport,
            input_tokens=input_tokens,
            use_gpu_telemetry=args.gpu,
            expected_stage_num=expected_stage,
            expected_status=args.expected_status,
            expected_progress_pct=args.expected_progress,
        )
        r["client_prepare_s"] = client_prepare_s
        runs.append(r)

        print(
            f"{i:03d} "
            f"img={idx:02d}:{live_path.name} "
            f"total={r['total_latency_s']:.4f}s "
            f"ttft={r['ttft_s']:.4f}s "
            f"decode={r['decode_time_s']:.4f}s "
            f"tok/s={r['tokens_per_second_decode']} "
            f"cache_n={r.get('cache_n')} "
            f"prompt_n={r.get('prompt_n')} "
            f"prompt_ms={r.get('prompt_ms')} "
            f"expected_stage={r.get('expected_stage_num')} "
            f"stage={r.get('predicted_stage_num')} "
            f"status={r.get('predicted_status')}"
        )

    summary = make_summary(runs)
    save_results(Path(args.out_dir), runs, summary)

    print("\n=== SUMMARY ===")
    for key in [
        "total_latency_s",
        "ttft_s",
        "decode_time_s",
        "ttft_to_decode_ratio",
        "prefill_share_of_total",
        "tokens_per_second_decode",
        "cache_hit_ratio",
        "cache_n",
        "prompt_n",
        "prompt_ms",
        "predicted_ms",
        "server_model_time_s",
        "client_unaccounted_s",
        "ttft_minus_prompt_s",
        "mem_used_max_mb",
        "gpu_util_mean_pct",
        "input_tokens",
        "output_tokens",
    ]:
        print(key)
        print(json.dumps(summary[key], indent=2))

    print("accuracy")
    print(json.dumps(summary["accuracy"], indent=2))
    print("stage_confusion")
    print(json.dumps(summary["stage_confusion"], indent=2))

    print("\nSaved:")
    print(f"  {Path(args.out_dir) / 'runs.jsonl'}")
    print(f"  {Path(args.out_dir) / 'runs.csv'}")
    print(f"  {Path(args.out_dir) / 'summary.json'}")


if __name__ == "__main__":
    main()
