"""Tests for null-route guarantee: no condition -> bit-identical outputs."""

import pytest
import torch
import torch.nn as nn

from acestep.training_v2.phase_d_timing_v2.config import PhaseD2Config
from acestep.training_v2.phase_d_timing_v2.model_integration import PhaseD2Injector


class TupleLayer(nn.Module):
    """Small stand-in for AceStepDiTLayer's `(hidden_states, ...)` return."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, value):
        return (self.proj(value), "attention-audit")


@pytest.fixture
def dummy_transformer_stack():
    """Fixture: small dummy transformer stack."""
    layers = nn.ModuleList([
        nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=1024,
            batch_first=True,
            norm_first=True,
        )
        for _ in range(4)
    ])
    return layers


@pytest.fixture
def config():
    """Fixture: default config."""
    return PhaseD2Config(hidden_size=256, injection_layer_indices=[0, 1, 2, 3])


def test_null_route_identical_output(dummy_transformer_stack, config):
    """With no condition set, injector outputs are identical to base."""
    injector = PhaseD2Injector(dummy_transformer_stack, config)
    for layer in dummy_transformer_stack:
        layer.eval()
    x = torch.randn(2, 50, 256)  # [B, T, H]

    # Capture the genuinely unhooked parent route first.
    with torch.no_grad():
        output_base = x.clone()
        for layer in dummy_transformer_stack:
            output_base = layer(output_base)

    injector.attach()
    with torch.no_grad():
        output_injected = x.clone()
        for layer in dummy_transformer_stack:
            output_injected = layer(output_injected)

    assert torch.equal(output_base, output_injected)

    injector.detach()


def test_null_route_with_no_condition(dummy_transformer_stack, config):
    """Injector with attached hooks but no condition set."""
    injector = PhaseD2Injector(dummy_transformer_stack, config)
    injector.attach()

    # Eval mode
    for layer in dummy_transformer_stack:
        layer.eval()

    x = torch.randn(2, 50, 256)

    # Forward with no condition set
    with torch.no_grad():
        y = x.clone()
        for layer in dummy_transformer_stack:
            y = layer(y)

    # Should match base forward (deterministic in eval mode)
    with torch.no_grad():
        y2 = x.clone()
        for layer in dummy_transformer_stack:
            y2 = layer(y2)

    assert torch.allclose(y, y2, atol=1e-5)

    injector.detach()


def test_condition_changes_output(dummy_transformer_stack, config):
    """With condition set, outputs should change."""
    injector = PhaseD2Injector(dummy_transformer_stack, config)
    for layer in dummy_transformer_stack:
        layer.eval()
    # Make one trained-like residual nonzero; a fresh zero-init adapter is
    # expected to be identical even with a condition.
    with torch.no_grad():
        injector.adapters[0].output_projection.weight.normal_(std=0.01)
    injector.attach()

    x = torch.randn(2, 50, 256)

    # Forward with no condition
    with torch.no_grad():
        y_no_cond = x.clone()
        for layer in dummy_transformer_stack:
            y_no_cond = layer(y_no_cond)

    # Forward with condition
    cond = torch.randn(2, 50, config.condition_dim)
    injector.set_condition(cond, control_strength=1.0)

    with torch.no_grad():
        y_with_cond = x.clone()
        for layer in dummy_transformer_stack:
            y_with_cond = layer(y_with_cond)

    assert not torch.allclose(y_no_cond, y_with_cond)

    injector.detach()


def test_real_dit_tuple_and_patch_length_contract():
    """Tuple outputs and 2x shorter DiT sequences receive a valid residual."""
    hidden_size = 32
    config = PhaseD2Config(hidden_size=hidden_size, num_adapter_layers=1, injection_layer_indices=[0])
    layers = nn.ModuleList([TupleLayer(hidden_size)])
    injector = PhaseD2Injector(layers, config)
    with torch.no_grad():
        injector.adapters[0].output_projection.weight.normal_(std=0.01)
    injector.attach()
    injector.set_condition(torch.randn(1, 20, config.condition_dim))
    output = layers[0](torch.randn(2, 10, hidden_size))
    assert isinstance(output, tuple)
    assert output[0].shape == (2, 10, hidden_size)
    assert output[1] == "attention-audit"
    injector.detach()


def test_tuple_null_route_is_bit_identical():
    """An attached tuple-returning ACE-like layer is exact under null routing."""
    hidden_size = 32
    config = PhaseD2Config(hidden_size=hidden_size, num_adapter_layers=1, injection_layer_indices=[0])
    layers = nn.ModuleList([TupleLayer(hidden_size)])
    layers.eval()
    value = torch.randn(1, 10, hidden_size)
    with torch.no_grad():
        unhooked = layers[0](value)
    injector = PhaseD2Injector(layers, config)
    injector.attach()
    with torch.no_grad():
        hooked = layers[0](value)
    assert torch.equal(unhooked[0], hooked[0])
    assert unhooked[1] == hooked[1]
    injector.detach()


def test_explicit_zero_condition_is_bit_identical_after_training_like_change(
    dummy_transformer_stack, config
):
    """An attached explicit-zero control follows the exact base trajectory."""
    for layer in dummy_transformer_stack:
        layer.eval()
    value = torch.randn(2, 50, 256)
    with torch.no_grad():
        baseline = value.clone()
        for layer in dummy_transformer_stack:
            baseline = layer(baseline)
    injector = PhaseD2Injector(dummy_transformer_stack, config)
    with torch.no_grad():
        injector.adapters[0].output_projection.weight.normal_(std=0.01)
    injector.attach()
    injector.set_condition(torch.zeros(2, 50, config.condition_dim))
    with torch.no_grad():
        controlled = value.clone()
        for layer in dummy_transformer_stack:
            controlled = layer(controlled)
    assert torch.equal(baseline, controlled)
    injector.detach()
