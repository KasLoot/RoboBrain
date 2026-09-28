"""RGB + registered image-depth surface measurement; no simulator state inputs."""
from copy import deepcopy
import math
import numpy as np
import pytest

from core.sim.visual_insertion import estimate_insertion


def planes(insertion_mm):
    rgb = np.empty((128, 128, 3), np.uint8)
    rgb[:] = [20, 60, 200]
    rgb[40:88, 40:88] = [220, 85, 20]
    depth = np.full((128, 128), .1 + .082 - insertion_mm / 1000.)
    depth[40:88, 40:88] = .1
    return {'wrist': rgb, 'wrist_depth_m': depth, 'wrist_camera_axes': {'sightline': [0, 0, -1]},
            'wrist_depth_calibration': {'near_m': 0., 'far_m': .3}}


@pytest.mark.parametrize('depth_mm,qualifies', [(12, False), (25, False), (27, False), (29, True), (30, True)])
def test_plane_gap_with_conservative_margin(depth_mm, qualifies):
    result = estimate_insertion(planes(depth_mm))
    assert result['quality'] == 'measured' and result['qualified'] is qualifies
    assert result['estimate_m'] == pytest.approx(depth_mm / 1000.)
    assert result['lower_bound_m'] == pytest.approx(depth_mm / 1000. - .003)
    assert result['orange']['pixels'] >= 64 and result['blue']['pixels'] >= 64
    assert result['orange']['spread_m'] == pytest.approx(0.)
    assert result['blue']['spread_m'] == pytest.approx(0.)


def test_surface_occlusion_or_missing_dense_depth_does_not_qualify():
    for kind in ('orange_hidden', 'blue_hidden', 'depth_missing', 'orange_invalid'):
        obs = planes(30)
        if kind == 'orange_hidden': obs['wrist'][40:88, 40:88] = [100, 100, 100]
        if kind == 'blue_hidden': obs['wrist'][:40] = [100, 100, 100]; obs['wrist'][88:] = [100,100,100]; obs['wrist'][:, :40] = [100,100,100]; obs['wrist'][:,88:] = [100,100,100]
        if kind == 'depth_missing': del obs['wrist_depth_m']
        if kind == 'orange_invalid': obs['wrist_depth_m'][40:88,40:88] = np.nan
        result = estimate_insertion(obs)
        assert result['quality'] == 'uncertain' and not result['qualified'], kind


def test_tilt_and_depth_shape_are_validated():
    obs = planes(30)
    theta = math.radians(2)
    obs['wrist_camera_axes']['sightline'] = [math.sin(theta), 0, -math.cos(theta)]
    result = estimate_insertion(obs)
    assert not result['qualified'] and result['camera_tilt_deg'] == pytest.approx(2.)
    obs = planes(30)
    obs['wrist_depth_m'] = np.zeros((1, 1))
    assert estimate_insertion(obs)['quality'] == 'uncertain'
    obs = planes(30)
    del obs['wrist_camera_axes']
    assert estimate_insertion(obs)['quality'] == 'uncertain'
    for invalid in (None, {}, {'wrist': 'invalid', 'wrist_depth_m': 'not an array'}):
        assert estimate_insertion(invalid)['quality'] == 'uncertain'


def test_nonplanar_or_thin_surface_does_not_masquerade_as_top():
    obs = planes(30)
    obs['wrist_depth_m'][40:88, 40:88] = np.linspace(.09, .12, 48)[None, :]
    assert estimate_insertion(obs)['quality'] == 'uncertain'
    obs = planes(30)
    obs['wrist'][40:88, 40:88] = [100, 100, 100]
    obs['wrist'][50:53, 40:88] = [220, 85, 20]
    assert estimate_insertion(obs)['quality'] == 'uncertain'


def test_object_state_fields_cannot_influence_measurement():
    obs = planes(12)
    before = estimate_insertion(obs)
    obs.update(object_pose=[0, 0, 0], true_insertion_m=.05, success=True, contact_force=100.)
    assert estimate_insertion(obs) == before
    assert not estimate_insertion(obs, uncertainty_m=0.)['qualified']


def test_thin_perspective_side_plane_cannot_be_mistaken_for_top():
    # A close, peripheral vertical side produces a narrow depth range and would
    # satisfy scalar-spread/area checks. Its 3D normal still faces sideways.
    size = 512
    rgb = np.empty((size, size, 3), np.uint8)
    rgb[:] = [20, 60, 200]
    rgb[100:200, 450:465] = [220, 85, 20]
    depth = np.full((size, size), .072)  # Would imply ~30 mm if side depth were used.
    fovy = 80.
    focal = size / (2 * math.tan(math.radians(fovy) / 2))
    x = (np.arange(450, 465) + .5 - size / 2) / focal
    plane_x = .020 * ((457.5 - size / 2) / focal)
    side_depth = plane_x / x
    assert np.ptp(side_depth) < .0015
    depth[100:200, 450:465] = side_depth[None, :]
    obs = {'wrist': rgb, 'wrist_depth_m': depth,
           'wrist_camera_axes': {'sightline': [0, 0, -1]},
           'wrist_depth_calibration': {'near_m': 0., 'far_m': .3, 'vertical_fov_deg': fovy}}
    result = estimate_insertion(obs)
    assert result['quality'] == 'uncertain' and not result['qualified']
    assert 'side or tilted plane' in result['reason'], result


def test_genuine_top_planes_use_calibrated_intrinsics_and_report_fit():
    obs = planes(30)
    obs['wrist_depth_calibration']['vertical_fov_deg'] = 70.
    result = estimate_insertion(obs)
    assert result['qualified'] and result['vertical_fov_deg'] == 70.
    for color in ('orange', 'blue'):
        assert result[color]['normal_tilt_deg'] == pytest.approx(0., abs=1e-5)
        assert result[color]['plane_rms_m'] == pytest.approx(0., abs=1e-8)
    # Recorded older observations of the fixed Wrist camera remain supported.
    del obs['wrist_depth_calibration']['vertical_fov_deg']
    assert estimate_insertion(obs)['vertical_fov_deg'] == 80.
    obs['wrist_depth_calibration']['vertical_fov_deg'] = float('nan')
    assert estimate_insertion(obs)['quality'] == 'uncertain'
