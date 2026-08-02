"""Tests for the canonical timing-unit conversion."""

from acestep.training_v2.phase_d_timing_v2.timing_units import seconds_to_frames


def test_seconds_to_frames_uses_half_even_rounding():
    """Exact half-frame ties use round-half-even at 25 Hz."""
    assert seconds_to_frames(0.10) == 2
    assert seconds_to_frames(0.14) == 4
    assert seconds_to_frames(1.00) == 25
