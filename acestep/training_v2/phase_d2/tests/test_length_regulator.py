"""Tests for explicit duration-based Phase D2 frame regulation."""

import pytest
import torch

from acestep.training_v2.phase_d2.length_regulator import length_regulate


def test_length_regulator_exact_target_and_order():
    """Durations expand to the exact target without event reordering."""
    features = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    dense, event_ids, audit = length_regulate(
        features, torch.tensor([0.2, 0.4, 0.2]), frame_rate_hz=25.0, target_frames=20
    )
    assert dense.shape == (20, 2)
    assert event_ids.tolist() == sorted(event_ids.tolist())
    assert sum(audit["event_frame_counts"]) == 20
    assert audit["event_order_preserved"]


def test_length_regulator_duration_stretch_adds_only_target_event_frames():
    """Stretching one event adds frames to that event without collisions."""
    features = torch.eye(3)
    base = length_regulate(features, torch.tensor([0.2, 0.2, 0.2]), 25.0)[2]
    stretched = length_regulate(features, torch.tensor([0.2, 0.4, 0.2]), 25.0)[2]
    base_counts = base["event_frame_counts"]
    stretch_counts = stretched["event_frame_counts"]
    assert stretch_counts[0] == base_counts[0]
    assert stretch_counts[2] == base_counts[2]
    assert stretch_counts[1] > base_counts[1]


def test_length_regulator_rejects_impossible_budget():
    """A target shorter than one frame per event fails closed."""
    with pytest.raises(ValueError, match="at least one frame"):
        length_regulate(torch.eye(3), torch.ones(3), 25.0, target_frames=2)
