"""Tests for real-runtime wrapper resolution and parameter ownership."""

import torch
import torch.nn as nn

from acestep.training_v2.phase_d_timing_v2.config import PhaseD2Config
from acestep.training_v2.phase_d_timing_v2.runtime_integration import attach_frozen_c25_d2, unwrap_real_dit


class TinyDiT(nn.Module):
    """Small real-DiT-shaped test double."""

    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 32})()
        self.patch_size = 2
        self.layers = nn.ModuleList([nn.Linear(32, 32) for _ in range(4)])


class PeftShaped(nn.Module):
    """Minimal wrapper exposing PEFT's get_base_model contract."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def get_base_model(self):
        return self.base


def test_attach_freezes_base_and_keeps_parameter_ownership_disjoint():
    """Only D2 parameters remain trainable and injector owns no C25 tensors."""
    model = nn.Module()
    model.decoder = PeftShaped(TinyDiT())
    config = PhaseD2Config(
        hidden_size=32,
        condition_hidden_size=16,
        adapter_bottleneck_size=8,
        num_adapter_layers=2,
        injection_layer_indices=[0, 1],
    )
    injector, audit = attach_frozen_c25_d2(model, config)
    assert audit["base_trainable_count"] == 0
    assert audit["parameter_id_overlap"] == 0
    assert audit["d2_parameter_count"] > 0
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert {id(p) for p in injector.parameters()} == {
        id(p) for p in injector.get_adapter_parameters()
    }
    injector.detach()


def test_unwrap_real_dit_resolves_peft_shape():
    """PEFT-shaped decoder wrappers resolve to the actual layer stack."""
    dit = TinyDiT()
    assert unwrap_real_dit(PeftShaped(dit)) is dit
