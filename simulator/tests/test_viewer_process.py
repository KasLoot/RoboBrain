"""Native viewer transport/environment guards; no window or model API required."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from core.sim.viewer_process import _packet, _viewer_environment


class ViewerProcessTests(unittest.TestCase):
    def test_child_gets_glfw_without_credentials_or_external_cache_paths(self):
        with TemporaryDirectory() as temporary:
            cache = Path(temporary).resolve()
            source = {'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl',
                      'OPENAI_API_KEY': 'private-key', 'PROVIDER_TOKEN': 'private-token',
                      'CLIENT_SECRET': 'private-secret', 'SERVICE_PASSWORD': 'private-password',
                      'DISPLAY': ':77', 'XAUTHORITY': '/read-only/display-cookie',
                      'PYTHONPATH': '/existing/code', 'HOME': '/existing/home'}
            with patch.dict(os.environ, source, clear=True):
                result = _viewer_environment(cache)
                self.assertEqual(dict(os.environ), source)
            for key in ('OPENAI_API_KEY', 'PROVIDER_TOKEN', 'CLIENT_SECRET', 'SERVICE_PASSWORD', 'PYOPENGL_PLATFORM'):
                self.assertNotIn(key, result)
            self.assertEqual(result['MUJOCO_GL'], 'glfw')
            self.assertEqual(result['PYTHONDONTWRITEBYTECODE'], '1')
            self.assertEqual(result['DISPLAY'], ':77')
            self.assertEqual(result['XAUTHORITY'], '/read-only/display-cookie')
            self.assertEqual(result['HOME'], '/existing/home')
            self.assertTrue(result['PYTHONPATH'].endswith(os.pathsep + '/existing/code'))
            for key in ('TMPDIR', 'XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'XDG_STATE_HOME',
                        'MESA_SHADER_CACHE_DIR', '__GL_SHADER_DISK_CACHE_PATH'):
                self.assertTrue(Path(result[key]).is_relative_to(cache))
                self.assertTrue(Path(result[key]).is_dir())

    def test_state_packets_are_delimited_finite_and_bounded(self):
        value = {'type': 'state', 'state': [1.0, 2.0], 'state_seq': 9}
        encoded = _packet(value)
        self.assertTrue(encoded.endswith(b'\n'))
        self.assertEqual(json.loads(encoded), value)
        with self.assertRaises(ValueError):
            _packet({'state': [float('nan')]})
        with patch('core.sim.viewer_process.MAX_PACKET_BYTES', 16):
            with self.assertRaisesRegex(ValueError, 'bounded transport'):
                _packet(value)
