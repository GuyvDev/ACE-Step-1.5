"""Canonical seconds-to-latent-frame conversion for Phase D Timing v2."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_EVEN


LATENT_FRAME_RATE_HZ = 25.0


def seconds_to_frames(seconds: float, frame_rate_hz: float = LATENT_FRAME_RATE_HZ) -> int:
    """Convert non-negative seconds to frames with deterministic half-even rounding."""
    if seconds < 0.0 or frame_rate_hz <= 0.0:
        raise ValueError("seconds must be non-negative and frame_rate_hz must be positive")
    quota = Decimal(str(seconds)) * Decimal(str(frame_rate_hz))
    return int(quota.quantize(Decimal("1"), rounding=ROUND_HALF_EVEN))


def frames_to_seconds(frames: int, frame_rate_hz: float = LATENT_FRAME_RATE_HZ) -> float:
    """Convert a non-negative integer frame count to physical seconds."""
    if frames < 0 or frame_rate_hz <= 0.0:
        raise ValueError("frames must be non-negative and frame_rate_hz must be positive")
    return float(frames) / float(frame_rate_hz)
