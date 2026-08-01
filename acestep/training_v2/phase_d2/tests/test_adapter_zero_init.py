"""Tests for exact zero initialization and content dependence."""

import pytest
import torch

from acestep.training_v2.phase_d2.temporal_adapter import ZeroInitTemporalAdapter


@pytest.fixture
def adapter():
    """Return a fresh small zero-initialized adapter."""
    return ZeroInitTemporalAdapter(condition_dim=100, hidden_size=256)


def test_adapter_output_projection_is_exact_zero(adapter):
    """The residual-producing projection initializes exactly to zero."""
    assert torch.equal(
        adapter.output_projection.weight,
        torch.zeros_like(adapter.output_projection.weight),
    )
    assert torch.equal(
        adapter.output_projection.bias,
        torch.zeros_like(adapter.output_projection.bias),
    )


def test_fresh_adapter_is_exact_zero_for_nonzero_control(adapter):
    """A fresh adapter returns exact zero for every non-null condition."""
    hidden_states = torch.randn(2, 50, 256)
    condition = torch.randn(2, 50, 100)
    assert torch.equal(adapter(hidden_states, condition), torch.zeros_like(hidden_states))


def test_trained_like_adapter_is_exact_zero_for_zero_control(adapter):
    """Explicit zero control remains exact after output parameters change."""
    with torch.no_grad():
        adapter.output_projection.weight.normal_(std=0.01)
        adapter.output_projection.bias.normal_(std=0.01)
    hidden_states = torch.randn(2, 50, 256)
    condition = torch.zeros(2, 50, 100)
    assert torch.equal(adapter(hidden_states, condition), torch.zeros_like(hidden_states))


def test_control_strength_zero_is_exact(adapter):
    """A disabled control strength returns exact zero."""
    with torch.no_grad():
        adapter.output_projection.weight.normal_(std=0.01)
    hidden_states = torch.randn(2, 50, 256)
    condition = torch.randn(2, 50, 100)
    assert torch.equal(
        adapter(hidden_states, condition, control_strength=0.0),
        torch.zeros_like(hidden_states),
    )


def test_adapter_depends_on_hidden_and_condition(adapter):
    """A trained-like adapter changes with hidden content and control content."""
    with torch.no_grad():
        adapter.output_projection.weight.normal_(std=0.01)
    hidden_a = torch.randn(2, 12, 256)
    hidden_b = hidden_a + torch.randn_like(hidden_a) * 0.2
    condition_a = torch.randn(2, 12, 100)
    condition_b = condition_a + torch.randn_like(condition_a) * 0.2
    base = adapter(hidden_a, condition_a)
    assert not torch.equal(base, adapter(hidden_b, condition_a))
    assert not torch.equal(base, adapter(hidden_a, condition_b))


def test_gradients_reach_hidden_and_condition_projections(adapter):
    """Trained-like output weights pass gradients into both projections."""
    with torch.no_grad():
        adapter.output_projection.weight.normal_(std=0.01)
    loss = adapter(torch.randn(2, 8, 256), torch.randn(2, 8, 100)).square().mean()
    loss.backward()
    assert adapter.hidden_projection.weight.grad.abs().sum() > 0
    assert adapter.condition_projection.weight.grad.abs().sum() > 0


def test_adapter_output_shape(adapter):
    """Adapter output matches the hidden-state shape."""
    hidden_states = torch.randn(2, 50, 256)
    output = adapter(hidden_states, torch.randn(2, 50, 100))
    assert output.shape == hidden_states.shape
