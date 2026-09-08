"""Production checks: uv run python tests/check_robobrain.py."""
import asyncio
import base64
import json
from pathlib import Path
import tempfile
import time
import math
import shutil
import subprocess
import threading
from unittest.mock import patch

import cv2
import httpx
import numpy as np

from robobrain.monitor import Backend, Frame, ReplayController, SimulatedCamera, decide, image_content, load_config, monitor
from robobrain.presentation import EpisodeReport, LiveView, annotate_frame, render_recording
from robobrain.__main__ import select_videos


SPECIALISTS = ("success", "drop", "rotation")


def evidence(success, drop, rotation=0):
    return [{"slot": i, "specialist": s, "verdict": "POSITIVE" if i < {"success": success, "drop": drop, "rotation": rotation}[s] else "UNCERTAIN"}
            for i in range(10) for s in SPECIALISTS]


async def checks():
    config = load_config()
    specs = Backend(config, None).specialists
    assert tuple(specs) == SPECIALISTS and config['max_connections'] == 36
    for location, expected in [("STAND", "NEGATIVE"), ("GRIPPER", "NEGATIVE"), ("SOCKET", "NEGATIVE"), ("TABLE", "POSITIVE"), ("FALLING", "UNCERTAIN"), ("UNKNOWN", "UNCERTAIN")]:
        assert specs['drop'].parse(json.dumps({"location": location, "reason": "visible relation"}))["verdict"] == expected
    for pose, expected in [('HELD_NORMAL', 'NEGATIVE'), ('HELD_ROTATED', 'POSITIVE'), ('NOT_HELD', 'NEGATIVE'), ('NOT_INSERTING', 'NEGATIVE'), ('UNKNOWN', 'UNCERTAIN')]:
        assert specs['rotation'].parse(json.dumps({'grasp_state': pose, 'reason': 'visible grasp'}))['verdict'] == expected
    assert decide(evidence(5, 0), config, specs)[0] == "ON_GOING"
    assert decide(evidence(6, 0), config, specs)[0] == "SUCCESS"
    assert decide(evidence(0, 6), config, specs)[0] == "FAIL"
    assert decide(evidence(6, 6), config, specs)[0] == "ON_GOING"
    assert decide(evidence(0, 0, 5), config, specs)[0] == "ON_GOING"
    assert decide(evidence(0, 0, 6), config, specs)[0] == "FAIL"
    assert decide(evidence(6, 0, 6), config, specs)[0] == "ON_GOING"
    assert [s.failure_type for s in decide(evidence(0, 6, 6), config, specs)[2]] == ['DROP_FAILURE', 'ROTATION_FAILURE']
    try:
        specs['success'].parse('{"verdict":"POSITIVE","reason":"visible","type":"status","status":"SUCCESS"}')
    except ValueError:
        pass
    else:
        raise AssertionError('Model output must not overwrite trusted event metadata')
    for invalid in [evidence(6, 0)[:-1], evidence(6, 0) + [evidence(6, 0)[0]]]:
        try:
            decide(invalid, config, specs)
        except ValueError:
            pass
        else:
            raise AssertionError("Missing/duplicate observations accepted")

    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for crop in config["crops"].values():
        content = image_content(image, crop)
        data = base64.b64decode(content["image_url"]["url"].split(",")[1])
        decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape[:2] == (crop[3] - crop[1], crop[2] - crop[0])

    malformed_messages = iter([None, {"content": "not JSON"}, {"content": '{"grasp_state":"HELD_ROTATED","reason":"visible"}', "reasoning": "Unexpected thinking"}])
    async def handler(request):
        payload = json.loads(request.content)
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert payload["temperature"] == 0
        return httpx.Response(200, json={"choices": [{"message": next(malformed_messages)}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        backend = Backend(config, client)
        observations = await backend.batch(Frame(image, 0, 0, time.monotonic()), time.monotonic(), lambda _: None, normal=True)
        assert len(observations) == 3
        assert all(o["verdict"] == "UNCERTAIN" and "error" in o for o in observations)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "camera.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
        assert writer.isOpened()
        writer.write(np.full((48, 64, 3), 30, np.uint8))
        writer.write(np.full((48, 64, 3), 180, np.uint8))
        writer.release()
        camera = SimulatedCamera(path)
        camera.start()
        try:
            await asyncio.sleep(.14)
            before = camera.get()
            await asyncio.sleep(.2)
            after = camera.get()
            assert before.source_index == after.source_index == 1
            assert 5 <= after.sequence - before.sequence <= 7
            assert np.array_equal(before.image, after.image)
        finally:
            camera.stop()

    class FakeCamera:
        def __init__(self):
            self.started_at = time.monotonic()
        def get(self):
            now = time.monotonic()
            index = int((now - self.started_at) * 30)
            return Frame(None, index, index, now)

    class FakeBackend:
        def __init__(self, conflict_first=False, positive='success'):
            self.specialists = specs
            self.active = 0
            self.conflict_first = conflict_first
            self.positive = positive
        async def batch(self, frame, started_at, log, **tags):
            self.active += 1
            try:
                # Out-of-order completion; the final verification batch is slowest.
                await asyncio.sleep(.18 if tags.get("slot") == 9 else .04)
                rows = []
                for specialist in SPECIALISTS:
                    row = {"type": "observation", "specialist": specialist, "source_time_s": frame.source_index / 30,
                           "reason": "visible evidence",
                           "captured_time_s": frame.observed_at - started_at, **tags,
                           "verdict": "POSITIVE" if specialist == self.positive or (self.conflict_first and tags.get("round_id") == 1) else "NEGATIVE"}
                    log(row)
                    rows.append(row)
                return rows
            finally:
                self.active -= 1

    events = []
    backend = FakeBackend()
    config["horizon_duration"] = .4
    result = await monitor(FakeCamera(), backend, config, events.append)
    assert result["status"] == "SUCCESS" and result["evidence_counts"] == {"success": 10, "drop": 0, "rotation": 0}
    assert sum(e["type"] == "verification_started" for e in events) == 1
    verified = [e for e in events if e["type"] == "observation" and e.get("round_id") == 1]
    assert len(verified) == 30
    assert any(e["type"] == "observation" and e.get("normal") and e.get("round_id") is None for e in events[3:])
    assert backend.active == 0
    assert result["decision_time_s"] >= .04 + .36 + .18
    events = []
    backend = FakeBackend(conflict_first=True)
    result = await monitor(FakeCamera(), backend, config, events.append)
    starts = [e for e in events if e["type"] == "verification_started"]
    finishes = [e for e in events if e["type"] == "verification_finished"]
    assert len(starts) == len(finishes) == 2 and result["status"] == "SUCCESS"
    assert finishes[0]["status"] == "ON_GOING"
    assert starts[1]["time_s"] > finishes[0]["time_s"] + .039
    assert backend.active == 0
    backend = FakeBackend(positive='rotation')
    result = await monitor(FakeCamera(), backend, config, lambda _: None)
    assert result['status'] == 'FAIL' and result['failure_type'] == 'ROTATION_FAILURE'
    assert result['confirmed_failure_types'] == ['ROTATION_FAILURE']
    assert result['failure_description'] == 'visible evidence' and result['source_frame'] >= 0
    assert backend.active == 0
    actions = []
    controller = ReplayController()
    controller.start('attempt', actions.append)
    controller.finish('attempt', result, actions.append)
    controller.finish('attempt', result, actions.append)
    assert [e['action'] for e in actions] == ['SCENE_READY', 'START_INSERTION_VA', 'STOP_INSERTION_VA', 'RELEASE_GRIPPER', 'START_RECOVERY_VA', 'RESET_SCENE']
    assert all(e['mode'] == 'SIMULATED' for e in actions)
    controller.finish('unresolved', {'status': 'ON_GOING'}, actions.append)
    assert len(actions) == 6
    backend = FakeBackend()
    config["episode_timeout"] = .12
    result = await monitor(FakeCamera(), backend, config, lambda _: None)
    assert result["status"] == "ON_GOING" and result["stop_reason"] == "episode_timeout"
    assert backend.active == 0
    print("Checks passed: thresholds, conflicts, unique evidence, native crops, no thinking, malformed replies, 30 fps EOF continuation, one verification round, normal monitoring, complete response wait, cleanup.")

    async def blocked(request):
        await asyncio.sleep(10)
        raise AssertionError("Cancelled request unexpectedly completed")
    async with httpx.AsyncClient(transport=httpx.MockTransport(blocked)) as client:
        backend = Backend(load_config(), client)
        events = []
        task = asyncio.create_task(backend.batch(Frame(image, 0, 0, time.monotonic()), time.monotonic(), events.append, normal=True))
        await asyncio.sleep(.05)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        started = [e for e in events if e['type'] == 'request_started']
        finished = [e for e in events if e['type'] == 'observation']
        assert len(started) == len(finished) == 3
        assert {e['request_id'] for e in started} == {e['request_id'] for e in finished}
        assert all(e['cancelled'] and e['verdict'] == 'UNCERTAIN' for e in finished)

    stop = threading.Event()
    async def stop_soon():
        await asyncio.sleep(.08)
        stop.set()
    stopper = asyncio.create_task(stop_soon())
    backend = FakeBackend()
    result = await monitor(FakeCamera(), backend, load_config(), lambda _: None, stop)
    await stopper
    assert result['status'] == 'ON_GOING' and result['stop_reason'] == 'user_stop' and backend.active == 0


def recording_checks():
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    assert ffmpeg and ffprobe, 'Recording checks require ffmpeg and ffprobe'
    image = np.full((1080, 1920, 3), 170, np.uint8)
    state = {'status': 'FAIL', 'failure_type': 'ROTATION_FAILURE', 'source_frame': 3}
    real_put_text = cv2.putText
    with patch('robobrain.presentation.cv2.putText', wraps=real_put_text) as draw:
        annotated = annotate_frame(image, 9, state)
    calls = {call.args[1]: call.args for call in draw.call_args_list}
    assert 'Frame 10' in calls and 'Operating Status' in calls and ' (frame 4):' in calls
    assert calls['Operating Status'][5] == calls[' (frame 4):'][5] == (255, 145, 50)
    assert calls['Operating Status'][6] > calls[' (frame 4):'][6]
    assert 'Status: FAIL' in calls and 'Failure type: ROTATION_FAILURE' in calls
    assert np.all(image == 170) and not np.array_equal(image, annotated)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        video = root/'traj-1.MOV'
        subprocess.run([ffmpeg, '-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc2=size=320x240:rate=30',
                        '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000', '-t', '1.2',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(video)], check=True)
        assert select_videos(root, [1])[1] == [video.resolve()]
        initial = {'type': 'status', 'status': 'ON_GOING', 'failure_type': None, 'source_frame': -1, 'decision_time_s': 0.}
        late = {'type': 'status', 'status': 'FAIL', 'failure_type': 'ROTATION_FAILURE', 'source_frame': 35, 'decision_time_s': 1.6}
        displayed = []
        def inspect_frame(image, source_frame, state):
            displayed.append((source_frame, state['status']))
            return image
        with patch('robobrain.presentation.annotate_frame', side_effect=inspect_frame):
            render_recording(video, root/'late.MOV', [initial, late], late, ffmpeg, threading.Event())
        assert len(displayed) == math.ceil((1.6 + 1) * 30)
        assert [r[0] for r in displayed[:36]] == list(range(36))
        assert all(r[0] == 35 for r in displayed[36:])
        assert all(r[1] == 'ON_GOING' for r in displayed[:48])
        assert all(r[1] == 'FAIL' for r in displayed[48:])
        probe = json.loads(subprocess.check_output([ffprobe, '-v', 'error', '-show_streams', '-of', 'json', str(root/'late.MOV')]))
        assert any(s['codec_type'] == 'audio' and s['codec_name'] == 'aac' for s in probe['streams'])
        stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        assert int(stream['nb_frames']) == 78 and stream['r_frame_rate'] == '30/1'
        assert (stream['width'], stream['height']) == (320, 240)
        early = {**late, 'decision_time_s': .1}
        displayed.clear()
        with patch('robobrain.presentation.annotate_frame', side_effect=inspect_frame):
            render_recording(video, root/'early.MOV', [initial, early], early, ffmpeg, threading.Event())
        assert len(displayed) == 36, 'Early decisions must not truncate the original clip'

        output = root/'output'/'dataset'/'traj-1'
        report = EpisodeReport(video, 'dataset', load_config(), output, LiveView())
        report.log(initial)
        body = {'choices': [{'message': {'content': '{"grasp_state":"HELD_ROTATED","reason":"Plug rotated in fingers"}'}}]}
        observation = {'type': 'observation', 'request_id': 1, 'specialist': 'rotation', 'source_frame': 35,
                       'captured_time_s': 1.2, 'sent_time_s': 1.3, 'received_time_s': 1.5, 'latency_s': .2,
                       'normal': False, 'round_id': 1, 'slot': 0, 'verdict': 'POSITIVE', 'reason': 'Plug rotated in fingers',
                       'grasp_state': 'HELD_ROTATED',
                       'raw': body['choices'][0]['message']['content'], 'response': body, 'response_text': json.dumps(body)}
        report.log(observation)
        report.log(late)
        report.finish(late)
        readable = json.loads((output/'output-traj-1.json').read_text())
        assert readable['requests'][0]['source_frame'] == 36
        assert readable['result']['failure_type'] == 'ROTATION_FAILURE'
        assert readable['requests'][0]['grasp_state'] == 'HELD_ROTATED'
        assert readable['requests'][0]['model_output']['grasp_state'] == 'HELD_ROTATED'
        assert readable['requests'][0]['response'] == body
        raw = [json.loads(line) for line in (output/'output-traj-1.log').read_text().splitlines()]
        assert observation in raw
        assert not list(output.glob('*.tmp'))
    print('Recording checks passed: overlay formatting, unmodified inference frames, complete source, late status timing, original audio, frame rate and readable/raw outputs.')


if __name__ == '__main__':
    asyncio.run(checks())
    recording_checks()
