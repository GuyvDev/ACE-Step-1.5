"""Focused tests for the approved Phase D Timing v2 contract."""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from acestep.training_v2.phase_d_timing_v2.config import PhaseD2Config
from acestep.training_v2.phase_d_timing_v2.length_regulator import InfeasibleEdit, length_regulate
from acestep.training_v2.phase_d_timing_v2.losses import masked_mean, timing_v2_objective
from acestep.training_v2.phase_d_timing_v2.model_integration import PhaseD2Injector
from acestep.training_v2.phase_d_timing_v2.residual_adapter import ZeroInitTemporalAdapter
from acestep.training_v2.phase_d_timing_v2.timing_condition import build_timing_condition


def test_production_architecture_matches_approved_contract():
    """Defaults use the four approved layers and exactly 3,028,608 parameters."""
    config = PhaseD2Config()
    injector = PhaseD2Injector(nn.ModuleList([nn.Identity() for _ in range(24)]), config)
    assert config.hidden_size == 2048
    assert config.condition_hidden_size == 256
    assert config.adapter_bottleneck_size == 128
    assert injector.injection_indices == [0, 4, 8, 12]
    assert sum(parameter.numel() for parameter in injector.get_adapter_parameters()) == 3_028_608


def test_global_resampling_is_rejected():
    """A nonphysical target frame budget fails closed."""
    with pytest.raises(InfeasibleEdit, match="forbidden global resampling"):
        length_regulate(torch.eye(2), torch.tensor([0.4, 0.6]), 25.0, 30)


def test_control_active_mask_is_frame_aligned():
    """The active mask follows only explicitly targeted primary events."""
    events = [
        {"duration_sec": 0.4, "silence": True},
        {"duration_sec": 0.2, "word_id": 1, "target_phrase_mask": True},
        {"duration_sec": 0.4, "silence": True},
    ]
    _, event_ids, audit = build_timing_condition(events, 25, PhaseD2Config())
    active = torch.tensor(audit["control_active_mask"])
    assert torch.equal(active, event_ids == 1)
    assert audit["phoneme_role"] == "auxiliary_identity_only"


def test_condition_dropout_drops_whole_examples_during_training():
    """Ten-percent training dropout uses the explicit all-zero route."""
    adapter = ZeroInitTemporalAdapter(8, 16, 4, condition_dropout=0.10)
    adapter.train()
    with torch.no_grad():
        adapter.output_projection.weight.fill_(0.1)
        adapter.output_projection.bias.fill_(0.1)
    hidden = torch.randn(2, 5, 16)
    condition = torch.randn(2, 5, 8)
    with patch("torch.rand", return_value=torch.zeros(2, 1, 1)):
        assert torch.equal(adapter(hidden, condition), torch.zeros_like(hidden))
    with patch("torch.rand", return_value=torch.ones(2, 1, 1)):
        assert not torch.equal(adapter(hidden, condition), torch.zeros_like(hidden))


def test_masked_losses_match_hand_computation():
    """Masked terms divide by active-mask mass, not total canvas frames."""
    error = torch.tensor([[[2.0, 4.0], [100.0, 100.0], [8.0, 10.0]]])
    mask = torch.tensor([[True, False, False]])
    assert masked_mean(error, mask).item() == pytest.approx(6.0, abs=1e-6)

    phase_d = torch.tensor([[[2.0], [3.0], [4.0]]])
    target = torch.zeros_like(phase_d)
    c25 = torch.ones_like(phase_d)
    target_mask = torch.tensor([[True, False, False]])
    boundary_mask = torch.tensor([[False, True, False]])
    total, terms = timing_v2_objective(
        phase_d, target, c25, target_mask, boundary_mask
    )
    assert terms["inside"].item() == pytest.approx(2.0)
    assert terms["outside"].item() == pytest.approx(3.0)
    assert terms["boundary"].item() == pytest.approx(2.0)
    assert total.item() == pytest.approx(11.5)
