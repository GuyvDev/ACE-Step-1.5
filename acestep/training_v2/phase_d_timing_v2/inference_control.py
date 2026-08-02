"""
Inference Control Schedule for Phase D2.

Control strength scheduling during diffusion/sampling loops.
"""

from __future__ import annotations

from typing import Literal
import math


def strength_for_step(
    step_frac: float,
    schedule: Literal["constant", "early_decay"],
) -> float:
    """Compute control strength for a given diffusion step.

    Args:
        step_frac: Fraction of diffusion complete, in [0, 1].
            0.0 = start (noise), 1.0 = end (sample).
        schedule: "constant" (always 1.0) or "early_decay"
            (cosine ramp down after step_frac > 0.5).

    Returns:
        Control strength in [0, 1].
    """

    if schedule == "constant":
        return 1.0

    elif schedule == "early_decay":
        # After 50% of steps, cosine ramp down to 0
        if step_frac < 0.5:
            return 1.0
        else:
            # Cosine decay from 50% to 100%
            t = (step_frac - 0.5) / 0.5  # [0, 1] for second half
            return 0.5 * (1.0 + math.cos(math.pi * t))  # [1, 0]

    else:
        raise ValueError(f"Unknown schedule: {schedule}")
