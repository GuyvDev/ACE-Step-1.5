"""Tests for physical duration-based Phase D2 frame regulation."""

import pytest
import torch

from acestep.training_v2.phase_d_timing_v2.length_regulator import (
    InfeasibleEdit,
    allocate_duration_frames,
    length_regulate,
)


def test_one_second_uses_real_latent_frame_rate():
    """One physical second maps to the configured latent frame rate."""
    assert allocate_duration_frames(torch.tensor([1.0]), 25.0).tolist() == [25]


def test_fractional_durations_allocate_remainder_deterministically():
    """Fractional physical frames use stable largest-remainder allocation."""
    first = allocate_duration_frames(torch.tensor([0.1, 0.1, 0.1]), 25.0)
    second = allocate_duration_frames(torch.tensor([0.1, 0.1, 0.1]), 25.0)
    assert first.tolist() == [3, 3, 2]
    assert torch.equal(first, second)


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
    assert audit["physical_total_frames_rounded"] == 20
    assert audit["target_frames_role"] == "final_global_resampling_target"
    assert audit["global_resampling_factor"] == pytest.approx(1.0)
    assert audit["collisions"] == 0


def test_length_regulator_duration_stretch_adds_only_target_event_frames():
    """Stretching one event adds frames to that event without collisions."""
    features = torch.eye(3)
    base = length_regulate(features, torch.tensor([0.2, 0.2, 0.2]), 25.0)[2]
    stretched = length_regulate(features, torch.tensor([0.2, 0.4, 0.2]), 25.0)[2]
    assert stretched["event_frame_counts"][0] == base["event_frame_counts"][0]
    assert stretched["event_frame_counts"][2] == base["event_frame_counts"][2]
    assert stretched["event_frame_counts"][1] > base["event_frame_counts"][1]


def test_fixed_budget_is_audited_and_expanded_budget_fails_closed():
    """Only a physical factor-one frame budget is accepted."""
    features = torch.eye(2)
    fixed = length_regulate(features, torch.tensor([0.4, 0.6]), 25.0, 25)[2]
    assert fixed["physical_frame_quotas"] == pytest.approx([10.0, 15.0])
    assert fixed["event_frame_counts"] == [10, 15]
    with pytest.raises(InfeasibleEdit, match="forbidden global resampling"):
        length_regulate(features, torch.tensor([0.4, 0.6]), 25.0, 30)


def test_length_regulator_rejects_impossible_budget():
    """A target shorter than one frame per event fails closed."""
    with pytest.raises(ValueError, match="at least one frame"):
        length_regulate(torch.eye(3), torch.ones(3), 25.0, target_frames=2)
