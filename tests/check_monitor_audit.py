"""Offline provider audit integration checks; no real network or model requests."""

import asyncio
import base64
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import httpx
import numpy as np

from robobrain.audit import ModelAudit
from robobrain.monitor import Backend, Frame, Specialist, decide, load_config
from robobrain.progress import ProgressObserver, ProgressRunner
from robobrain.sources import Snapshot
from robobrain.workspace import ProgressDefinition


SECRET = "audit-test-provider-key-123456789"
CONTROL_SECRET = "audit-test-control-token-987654321"
MODEL = "audit-test-model"
CONTEXT = ("server-1", "episode-1", "attempt-1", 7, ("camera1", 3))


def frame(index):
    # The two frames differ visibly; auditing must preserve crop pixels and history.
    pixels = np.indices((48, 64)).sum(axis=0).astype(np.uint8)
    image = np.dstack((pixels, np.full_like(pixels, index * 20), 255 - pixels))
    metadata = {"simulation":{"server_instance_id":"server-1", "episode_id":"episode-1",
                              "attempt_id":"attempt-1", "execution_generation":7, "state_seq":index},
                "context":CONTEXT, "identity":(*CONTEXT, index), "view_id":"view-1", "camera_revision":3,
                "control_token":CONTROL_SECRET}
    source = Frame(image, index, index, 10 + index / 10, metadata=metadata)
    return Snapshot(image, index, index, source.observed_at, {"camera1":source}, metadata=metadata)


def fixture(root):
    root.mkdir(parents=True)
    reference = np.full((16, 24, 3), (70, 130, 190), np.uint8)
    ok, png = cv2.imencode(".png", reference)
    assert ok
    (root / "reference.png").write_bytes(png.tobytes())
    url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    specialist = Specialist("Observe whether the object is released. " + SECRET, {
        "name":"success", "crop":[4, 3, 60, 45], "field":"verdict",
        "values":{value:value for value in ("POSITIVE", "NEGATIVE", "UNCERTAIN")},
        "failure_type":None, "threshold":0})
    specialist.references = [{"type":"text", "text":"Released reference example"},
                              {"type":"image_url", "image_url":{"url":url}}]
    definition = ProgressDefinition.model_validate({"enabled":True, "history_frames":2,
        "interval_s":.05, "confirmations":1, "stale_after_s":30,
        "nodes":[{"id":"HELD", "label":"Held", "criteria":"The object is visibly held."}],
        "blocks":[{"id":"context", "type":"text", "text":"Track held state independently of completion."},
                  {"id":"reference", "type":"image", "asset":"reference.png", "alias":"Reference only"},
                  {"id":"view", "type":"camera", "source":"camera1", "alias":"Current view"}]})
    store = SimpleNamespace(asset_path=lambda name: root / name)
    observer = ProgressObserver(definition, store, frame(1).views)
    return specialist, observer, definition, store


def configuration(native=False):
    config = load_config()
    config.update(base_url="http://127.0.0.1:11434/api" if native else "https://api.openai.com/v1",
                  model=MODEL, reasoning_effort="none", evidence_horizon=0, max_in_flight=2)
    return config


def reply(payload, *, native=False, status=200, extra=None):
    result = ({"verdict":"POSITIVE", "reason":"The object is visibly released."}
              if "Observe whether" in payload["messages"][0]["content"] else
              {"node_id":"HELD", "reason":"The fingers surround the object."})
    message = {"role":"assistant", "content":json.dumps(result)}
    body = ({"model":MODEL, "message":message, "done":True, "done_reason":"stop"} if native else
            {"id":"provider-call-1", "model":MODEL, "choices":[{"message":message}], "usage":{"prompt_tokens":12,"completion_tokens":8}})
    if extra:
        body.update(extra)
    return httpx.Response(status, json=body, headers={"x-request-id":"provider-request-1"})


def records(directory):
    return [json.loads(line) for line in (directory / "calls.jsonl").read_text().splitlines()]


def pairs(directory, expected):
    grouped = defaultdict(list)
    for event in records(directory):
        assert event["utc"] and event["monotonic_ns"] > 0
        grouped[event["call_id"]].append(event)
    assert len(grouped) == len(expected), grouped
    endings = []
    for call_id, events in grouped.items():
        assert call_id and events[0]["event"] == "request_started", events
        assert len(events) == 2, "Each tested provider call must have one start and one terminal event"
        assert events[0]["system"] == "robobrain"
        assert events[0]["model"] == MODEL
        assert events[1]["duration_s"] >= 0
        endings.append(events[1]["event"])
    assert sorted(endings) == sorted(expected), endings
    return list(grouped.values())


def asset_refs(value):
    if isinstance(value, dict):
        if set(value) == {"asset", "sha256", "bytes"}:
            yield value
        for child in value.values():
            yield from asset_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from asset_refs(child)


def assert_private_assets(directory, provider_payloads):
    text = (directory / "calls.jsonl").read_text()
    assert SECRET not in text and CONTROL_SECRET not in text
    assert "data:image/" not in text and "[redacted]" in text
    encoded_images = []
    for payload in provider_payloads:
        user = payload["messages"][1]
        if "images" in user:
            encoded_images.extend(user["images"])
        else:
            encoded_images.extend(part["image_url"]["url"].split(",", 1)[1]
                                  for part in user["content"] if part["type"] == "image_url")
    assert encoded_images and all(encoded not in text for encoded in encoded_images)
    refs = [ref for event in records(directory) for ref in asset_refs(event)]
    assert refs
    for ref in refs:
        path = directory / ref["asset"]
        assert path.resolve().is_relative_to(directory.resolve())
        content = path.read_bytes()
        assert len(content) == ref["bytes"] and hashlib.sha256(content).hexdigest() == ref["sha256"]
        assert cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR) is not None
    expected = {hashlib.sha256(base64.b64decode(encoded)).hexdigest() for encoded in encoded_images}
    assert {ref["sha256"] for ref in refs} == expected, "Audit assets must be the exact provider image bytes"
    assert len(list((directory / "assets").iterdir())) == len(expected), "Repeated reference/history images must be deduplicated"


async def payload_and_semantic_checks(root, native):
    specialist, observer, _, _ = fixture(root / "fixtures")
    config = configuration(native)
    audit = ModelAudit(root / "audit")
    payloads = []

    async def handle(request):
        if not native:
            assert request.headers["Authorization"] == "Bearer " + SECRET
        payload = json.loads(request.content)
        payloads.append(payload)
        return reply(payload, native=native, extra={"diagnostic":SECRET})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        plain = Backend(config, client, specialists={"success":specialist})
        audited = Backend(config, client, specialists={"success":specialist}, audit=audit)
        observed = []
        for backend in (plain, audited):
            log = []
            result = await specialist.observe(frame(2), backend, 10., log.append, normal=True)
            assert result["verdict"] == "POSITIVE" and result["reason"] == "The object is visibly released."
            assert log[0]["type"] == "request_started" and log[-1]["type"] == "observation"
            observed.append(result)
        assert payloads[0] == payloads[1], "Auditing must not modify provider prompts, settings or image bytes"
        assert decide([observed[0]], config, plain.specialists)[0] == "SUCCESS"
        assert decide([observed[1]], config, audited.specialists)[0] == "SUCCESS"
        history = [(0.1, frame(1)), (0.2, frame(2))]
        progress = []
        for backend in (plain, audited):
            response = await backend.complete(observer, history)
            body = response.json()
            message = body["message"] if native else body["choices"][0]["message"]
            progress.append(observer.parse(message["content"]))
        assert payloads[2] == payloads[3], "Auditing must not alter progress history or schema"
        assert progress == [{"node_id":"HELD", "reason":"The fingers surround the object."}] * 2
        assert decide([observed[1]], config, audited.specialists)[0] == "SUCCESS", "Progress must not change the outcome vote"
    entries = pairs(audit.directory, ["response_received", "response_received"])
    assert {start["role"] for start, _ in entries} == {"success", "progress"}
    outcome_start = next(start for start, _ in entries if start["role"] == "success")
    progress_start = next(start for start, _ in entries if start["role"] == "progress")
    assert [item["simulation"]["state_seq"] for item in outcome_start["evidence"]] == [2]
    assert [item["simulation"]["state_seq"] for item in progress_start["evidence"]] == [1, 2]
    assert all(item["control_token"] == "[redacted]" for item in progress_start["evidence"])
    for start, end in entries:
        assert end["http_status"] == 200 and end["provider_request_id"] == "provider-request-1"
        assert json.loads(end["response"])["model"] == start["model"]
        assert start["evidence"][-1]["identity"][-1] == 2
    assert_private_assets(audit.directory, [payloads[1], payloads[3]])


async def terminal_checks(root):
    specialist, observer, _, _ = fixture(root / "fixtures")
    audit = ModelAudit(root / "audit")
    mode, entered = "http_error", asyncio.Event()

    async def handle(request):
        payload = json.loads(request.content)
        if mode == "http_error":
            return httpx.Response(429, json={"error":{"message":"Rate limited: " + SECRET},"model":MODEL},
                                  headers={"x-request-id":"provider-rate-limit"})
        if mode == "transport_error":
            raise httpx.ReadTimeout("Test timeout " + SECRET, request=request)
        if mode == "unexpected_error":
            raise RuntimeError("Test transport failure " + SECRET)
        if mode == "cancel":
            entered.set()
            await asyncio.Event().wait()
        return reply(payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        backend = Backend(configuration(), client, specialists={"success":specialist}, audit=audit)
        result = await specialist.observe(frame(2), backend, 10., lambda event:None, normal=True)
        assert result["verdict"] == "UNCERTAIN" and result["http_status"] == 429
        mode = "transport_error"
        result = await specialist.observe(frame(2), backend, 10., lambda event:None, normal=True)
        assert result["verdict"] == "UNCERTAIN" and "Test timeout" in result["error"]
        mode = "unexpected_error"
        try:
            await backend.complete(observer, [(0.2, frame(2))])
        except RuntimeError as exc:
            assert "Test transport failure" in str(exc)
        else:
            raise AssertionError("Unexpected provider error was swallowed")
        mode = "cancel"
        call = asyncio.create_task(backend.complete(observer, [(0.1, frame(1)), (0.2, frame(2))]))
        await asyncio.wait_for(entered.wait(), 2)
        call.cancel()
        try:
            await call
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("Provider cancellation was swallowed")
    grouped = pairs(audit.directory, ["response_received", "request_failed", "request_failed", "request_cancelled"])
    limited = next(end for _, end in grouped if end["event"] == "response_received")
    assert limited["http_status"] == 429 and limited["provider_request_id"] == "provider-rate-limit"
    assert SECRET not in (audit.directory / "calls.jsonl").read_text()

    # Cancellation while the begin event is being written must not produce an orphan terminal event.
    entered_writer, release_writer = threading.Event(), threading.Event()
    begin_audit = ModelAudit(root / "cancel_during_begin")
    original_begin = begin_audit.begin
    def slow_begin(*args, **kwargs):
        entered_writer.set()
        assert release_writer.wait(5), "Timed out releasing audit writer"
        return original_begin(*args, **kwargs)
    contacted = []
    async def must_not_contact(request):
        contacted.append(request)
        raise AssertionError("Cancelled-before-dispatch request reached the provider")
    with patch.object(begin_audit, "begin", slow_begin):
        async with httpx.AsyncClient(transport=httpx.MockTransport(must_not_contact)) as client:
            backend = Backend(configuration(), client, specialists={}, audit=begin_audit)
            call = asyncio.create_task(backend.complete(specialist, frame(2)))
            try:
                assert await asyncio.to_thread(entered_writer.wait, 2)
                call.cancel()
            finally:
                release_writer.set()
            try:
                await call
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("Cancellation during audit begin was swallowed")
    assert not contacted
    pairs(begin_audit.directory, ["request_cancelled"])


async def progress_runner_check(root):
    specialist, _, definition, store = fixture(root / "fixtures")
    audit = ModelAudit(root / "audit")
    events, done = [], asyncio.Event()
    class Inputs:
        started_at = time.monotonic()
        counter = 0
        def evidence_context(self):
            return CONTEXT
        def accepts_evidence(self, evidence):
            return evidence["context"] == CONTEXT
        def get(self):
            self.counter += 1
            snapshot = frame(self.counter)
            observed = time.monotonic()
            source = Frame(snapshot.image, snapshot.sequence, snapshot.source_index, observed, metadata=snapshot.metadata)
            return Snapshot(source.image, source.sequence, source.source_index, observed, {"camera1":source}, metadata=source.metadata)
    inputs = Inputs()
    async def handle(request):
        return reply(json.loads(request.content))
    def log(event):
        events.append(event)
        if sum(item["type"] == "progress_observation" for item in events) >= 2:
            done.set()
    original_client = httpx.AsyncClient
    def mock_client(**kwargs):
        return original_client(transport=httpx.MockTransport(handle), **kwargs)
    async with original_client(transport=httpx.MockTransport(handle)) as client:
        outcome = Backend(configuration(), client, specialists={"success":specialist}, audit=audit)
        with patch("robobrain.progress.httpx.AsyncClient", mock_client):
            runner = ProgressRunner(definition.model_dump(), store, inputs, outcome, log)
            runner.start()
            try:
                await asyncio.wait_for(done.wait(), 3)
            finally:
                await runner.close()
    observations = [event for event in events if event["type"] == "progress_observation"]
    assert len(observations) >= 2 and all(item["node_id"] == "HELD" for item in observations)
    starts = [event for event in records(audit.directory) if event["event"] == "request_started"]
    assert len(starts) == len(observations) and all(event["role"] == "progress" for event in starts)
    assert len(starts[1]["evidence"]) == 2, "Runner must audit every historical evidence frame"
    pairs(audit.directory, ["response_received"] * len(observations))
    assert set(outcome.specialists) == {"success"}, "Progress must not become an outcome specialist"
    assert not any(event["type"] == "status" for event in events), "Progress audit must not emit task outcomes"


async def run(root):
    with patch.dict(os.environ, {"OPENAI_API_KEY":SECRET, "TEST_AUDIT_SECRET":CONTROL_SECRET}):
        await payload_and_semantic_checks(root / "openai", native=False)
        await payload_and_semantic_checks(root / "ollama", native=True)
        await terminal_checks(root / "terminal")
        await progress_runner_check(root / "runner")
    print("Monitor audit checks passed: unchanged outcome/progress payloads and decisions, image assets/deduplication, secret redaction, evidence/model correlation, HTTP errors, failures, paired cancellation and ProgressRunner integration.")


if __name__ == "__main__":
    scratch = Path(__file__).resolve().parents[1] / ".tmp"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        asyncio.run(run(Path(directory)))
