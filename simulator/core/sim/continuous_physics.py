"""Explicit solver settings for continuous-physics service mode.

Elliptic cones and increased friction impedance reduce soft-constraint creep
without changing contact geometry, friction coefficients or actuator forces.
NoSlip remains an explicit experimental option; the plug service profile keeps
it off because dense seated contacts make it too slow for real-time stepping.
See https://mujoco.readthedocs.io/en/latest/modeling.html#slow-slippage .
The original decision-synchronous session never calls this helper.
"""
from collections.abc import Mapping
import math


CONES = {'pyramidal': 0, 'elliptic': 1}


def configure_continuous_physics(model, cfg=None):
    """Validate and apply numerical solver options; return effective settings.

    Absent settings retain the model's cone/impratio and disable NoSlip. The task
    profile must opt into elliptic cones and an increased ``impratio``. All values
    are validated before changing the model. No timestep, forces, gains, gravity,
    geometry, friction coefficients, or joints are changed.
    """
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, Mapping):
        raise ValueError('Continuous physics configuration must be a mapping')
    section = cfg.get('continuous_physics', {})
    if not isinstance(section, Mapping):
        raise ValueError('continuous_physics must be a mapping')
    iterations = section.get('noslip_iterations', 0)
    if isinstance(iterations, bool) or not isinstance(iterations, int) or not 0 <= iterations <= 20:
        raise ValueError('continuous_physics.noslip_iterations must be an integer in [0,20]')
    default_cone = next(name for name, value in CONES.items() if value == int(model.opt.cone))
    cone = section.get('cone', default_cone)
    if not isinstance(cone, str) or cone not in CONES:
        raise ValueError('continuous_physics.cone must be pyramidal or elliptic')
    ratio = section.get('impratio', float(model.opt.impratio))
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        raise ValueError('continuous_physics.impratio must be finite and positive')
    try:
        ratio = float(ratio)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('continuous_physics.impratio must be finite and positive') from exc
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError('continuous_physics.impratio must be finite and positive')
    model.opt.noslip_iterations = iterations
    model.opt.cone = CONES[cone]
    model.opt.impratio = float(ratio)
    return {'cone': cone, 'impratio': float(model.opt.impratio),
            'noslip_iterations': int(model.opt.noslip_iterations),
            'noslip_tolerance': float(model.opt.noslip_tolerance),
            'timestep_s': float(model.opt.timestep)}
