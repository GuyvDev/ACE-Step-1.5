"""Tests for zero-initialization guarantee."""

import pytest
import torch

from acestep.training_v2.phase_d2.temporal_adapter import ZeroInitTemporalAdapter


@pytest.fixture
def adapter():
    """Fixture: fresh zero-init adapter."""
    return ZeroInitTemporalAdapter(condition_dim=100, hidden_size=256)


def test_adapter_up_weights_zero_init(adapter):
    """Up-projection weights initialized to zero."""
    assert torch.allclose(adapter.up.weight, torch.zeros_like(adapter.up.weight), atol=1e-6)
    assert torch.allclose(adapter.up.bias, torch.zeros_like(adapter.up.bias), atol=1e-6)


def test_adapter_gate_zero_init(adapter):
    """Gate parameter initialized to zero."""
    assert torch.allclose(adapter.gate, torch.zeros(1), atol=1e-6)


def test_adapter_zero_output_on_zero_condition(adapter):
    """Fresh adapter outputs zero for any input."""
    hidden_states = torch.randn(2, 50, 256)
    condition = torch.randn(2, 50, 100)
    
    with torch.no_grad():
        output = adapter(hidden_states, condition)
    
    # Should be exactly zero (or very close due to zero-init)
    assert torch.allclose(output, torch.zeros_like(output), atol=1e-5)


def test_adapter_zero_output_with_control_strength_zero(adapter):
    """Adapter outputs zero when control_strength=0."""
    hidden_states = torch.randn(2, 50, 256)
    condition = torch.randn(2, 50, 100)
    
    with torch.no_grad():
        output = adapter(hidden_states, condition, control_strength=0.0)
    
    assert torch.allclose(output, torch.zeros_like(output), atol=1e-5)


def test_adapter_output_shape(adapter):
    """Adapter output has correct shape."""
    hidden_states = torch.randn(2, 50, 256)
    condition = torch.randn(2, 50, 100)
    
    output = adapter(hidden_states, condition)
    
    assert output.shape == hidden_states.shape
