# Modified in RoboBrain: require half-pin insertion before release in task prompts.
"""Shared completion clearance and prompt guidance, not an action-enforcement gate.

The prompt helper describes the requested milestones without enforcing them or
exposing measured object state. Constants describe the nominal plug geometry.
"""
from collections.abc import Mapping
import math


PLUG_PIN_LENGTH_M = 0.05
PLUG_MIN_INSERTION_M = 0.025
PLUG_RETREAT_CLEARANCE_M = 0.05
PLUG_RETREAT_MARGIN_M = 0.005


def plug_retreat_clearance_m(cfg: Mapping | None = None) -> float:
    """Read the configured strict TCP-to-handle-top clearance in metres."""
    section = (cfg or {}).get("plug_success", {})
    if not isinstance(section, Mapping):
        raise ValueError("plug_success must be a mapping")
    raw = section.get("retreat_clearance_m", PLUG_RETREAT_CLEARANCE_M)
    try:
        clearance = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("plug_success.retreat_clearance_m must be finite and positive") from exc
    if isinstance(raw, bool) or not math.isfinite(clearance) or clearance <= 0:
        raise ValueError("plug_success.retreat_clearance_m must be finite and positive")
    return clearance


def plug_success_prompt(cfg: Mapping | None = None) -> str:
    """Describe controlled release and the final state, never measured object state."""
    clearance = plug_retreat_clearance_m(cfg)
    pin_mm = PLUG_PIN_LENGTH_M * 1000
    minimum_mm = PLUG_MIN_INSERTION_M * 1000
    exposed_mm = pin_mm - minimum_mm
    return (
        f"PLUG RELEASE MILESTONE: insert at least {minimum_mm:g} mm of the {pin_mm:g} mm-long silver pin "
        "into the actual bore while still held, then intentionally release. No more than "
        f"{exposed_mm:g} mm of silver shaft should remain above the rim; verify real bore entry from "
        "CURRENT external views, not a shaft projected behind the socket. The remaining "
        "insertion may settle under gravity after controlled release; full seating while "
        "held is not required. Verify the final plug is fully seated. "
        "PLUG COMPLETION CLEARANCE: after controlled release and full seating, keep the fingers "
        "open and retreat until the TCP (the grasp-point midpoint) is strictly more than "
        f"{clearance * 1000:g} mm vertically above the top of the seated orange handle. "
        f"Aim for at least {(clearance + PLUG_RETREAT_MARGIN_M) * 1000:g} mm to leave a margin. "
        "This is the world-Z gap from the handle top to the TCP, not merely the distance "
        "moved since RELEASE or the presence of a small visible gap. Confirm the seated "
        "plug remains stationary and complete this retreat before the final DONE."
    )
