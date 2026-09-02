"""Candidate safety-fusion policies used only by experiment replays."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConsecutiveSafetyFusion:
    """Fuse same-frame drop consensus and consecutive rotation consensus."""

    rotation_required: int = 3

    def __post_init__(self):
        if self.rotation_required < 1:
            raise ValueError("rotation_required must be positive")
        self.rotation_streak = 0
        self.failure_type = None

    def update(self, rotation_a, rotation_b, drop_a, drop_b):
        if self.failure_type is not None:
            return self.failure_type

        if drop_a == drop_b == "D":
            self.failure_type = "DROP_OR_CONTROL_LOSS"
            return self.failure_type

        if rotation_a == rotation_b == "R":
            self.rotation_streak += 1
        else:
            self.rotation_streak = 0

        if self.rotation_streak >= self.rotation_required:
            self.failure_type = "ROTATED_OR_MISALIGNED"
        return self.failure_type
