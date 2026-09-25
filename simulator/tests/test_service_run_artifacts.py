"""Per-scene rotation and interrupted-command poses use the applied physics boundary."""
import json
import tempfile
from pathlib import Path
import time

import numpy as np
from core.config import load_yaml
from core.sim.service_runtime import ROOT, ServiceRuntime, _SnapshotSession


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_run_rotation_and_interrupted_pose_uses_old_execution():
    with tempfile.TemporaryDirectory() as tmp:
        config = load_yaml(ROOT / 'configs/robot_mujoco_plug.yaml')
        config['camera_resolution'] = 64
        runtime = ServiceRuntime(config=config, no_controller=True, run_root=tmp)
        runtime.set_identity('server', 'old-scene')
        old_dir = Path(runtime.recording_run()['directory'])
        try:
            runtime.start_execution(1, 'attempt', lambda error: None)
            session = _SnapshotSession(runtime, 1, 'attempt')
            control = session.record_control_start(token='GRASP', step_idx=0, stage='GRASP', source='test')
            session.control_gripper(True)
            runtime.stop_execution(2)
            boundary = runtime._boundary_poses[1]
            runtime.start_execution(3, 'attempt', lambda error: None)
            assert runtime.recording_run()['directory'] == str(old_dir)
            runtime.stop_execution(4)
            runtime.set_identity('server', 'new-scene', generation=5)
            runtime.reset(5)
            new_dir = Path(runtime.recording_run()['directory'])
            assert new_dir != old_dir
            session.record_control_end(control, token='GRASP', status='interrupted', error='stopped')
            result = rows(old_dir / 'robot-trajectory.jsonl')[-1]
            assert result['pose']['episode_id'] == 'old-scene'
            assert result['pose']['execution_generation'] == 1
            assert result['pose']['captured_at_execution_boundary']
            np.testing.assert_allclose(result['pose']['tcp_pose'], boundary['tcp_pose'])
            assert result['pose']['gripper_position_m'][0] < .01
            initial = rows(new_dir / 'robot-trajectory.jsonl')
            assert len(initial) == 1 and initial[0]['event'] == 'initial_pose'
            assert initial[0]['pose']['episode_id'] == 'new-scene'
            assert initial[0]['pose']['gripper_position_m'][0] > .07
            assert rows(old_dir / 'robot-trajectory.jsonl')[0]['pose']['episode_id'] == 'old-scene'
            session.close()
        finally:
            runtime.close()
        assert all(item['episode_id'] == 'old-scene' for item in rows(old_dir / 'physics-events.jsonl'))
        assert rows(new_dir / 'physics-events.jsonl')[0]['kind'] == 'reset'
