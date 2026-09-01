"""Latency instrumentation for vLLM's OpenAI-compatible API."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any

from langchain_openai import ChatOpenAI


VLLM_METRICS_METADATA_KEY = "vllm_metrics"


class VLLMTimingChatOpenAI(ChatOpenAI):
    """Preserve vLLM's per-request metrics on the returned AI message.

    ``ChatOpenAI`` otherwise drops the non-standard top-level ``metrics`` field
    returned by vLLM.
    """

    def _create_chat_result(self, response, generation_info=None):
        response_metrics = None
        if isinstance(response, dict):
            response_metrics = response.get("metrics")
        elif hasattr(response, "model_dump"):
            response_metrics = response.model_dump(warnings=False).get("metrics")

        if response_metrics is not None:
            if hasattr(response_metrics, "model_dump"):
                response_metrics = response_metrics.model_dump(warnings=False)
            generation_info = dict(generation_info or {})
            generation_info[VLLM_METRICS_METADATA_KEY] = response_metrics

        return super()._create_chat_result(response, generation_info)


def _non_negative_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0:
        return None
    return value


def latency_components(response, round_trip_ms: float) -> dict[str, float | None]:
    """Split one completed call into remote generation and network overhead.

    vLLM reports three consecutive remote phases: scheduler queue, time from
    scheduling to the first token (which includes prefill), and decode from the
    first token to the last token. Their sum is used as generation latency.

    The client-observed round trip minus that sum is the network estimate. It
    also contains HTTP/SDK serialization and any server work outside vLLM's
    measured engine phases, so it is deliberately named an estimate in logs.
    """

    round_trip_ms = _non_negative_number(round_trip_ms)
    response_metadata = getattr(response, "response_metadata", {}) or {}
    metrics = response_metadata.get(VLLM_METRICS_METADATA_KEY) or {}
    if not isinstance(metrics, Mapping):
        metrics = {}

    queue_ms = _non_negative_number(metrics.get("queue_time_ms"))
    ttft_ms = _non_negative_number(metrics.get("time_to_first_token_ms"))
    decode_ms = _non_negative_number(metrics.get("generation_time_ms"))
    mean_itl_ms = _non_negative_number(metrics.get("mean_itl_ms"))
    tokens_per_second = _non_negative_number(metrics.get("tokens_per_second"))

    generation_ms = None
    network_ms = None
    if queue_ms is not None and ttft_ms is not None and decode_ms is not None:
        generation_ms = queue_ms + ttft_ms + decode_ms
        if round_trip_ms is not None:
            # Timer resolution and instrumentation boundaries can produce a
            # tiny negative residual. Network latency itself cannot be negative.
            network_ms = max(0.0, round_trip_ms - generation_ms)

    def rounded(value):
        return round(value, 3) if value is not None else None

    return {
        "round_trip_ms": rounded(round_trip_ms),
        "network_latency_estimate_ms": rounded(network_ms),
        "generation_latency_ms": rounded(generation_ms),
        "queue_ms": rounded(queue_ms),
        "time_to_first_token_ms": rounded(ttft_ms),
        "decode_ms": rounded(decode_ms),
        "mean_inter_token_ms": rounded(mean_itl_ms),
        "tokens_per_second": rounded(tokens_per_second),
    }


def invoke_with_latency(vlm, messages):
    """Invoke a chat model and return its response plus latency components."""

    started_at = time.perf_counter()
    response = vlm.invoke(messages)
    round_trip_ms = (time.perf_counter() - started_at) * 1000
    return response, latency_components(response, round_trip_ms)
