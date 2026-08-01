"""Tests for parameter freezing: base model frozen, adapters trainable."""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from acestep.training_v2.phase_d2.config import PhaseD2Config
from acestep.training_v2.phase_d2.model_integration import PhaseD2Injector


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
    """Fixture: config."""
    return PhaseD2Config()


def test_base_model_frozen(dummy_transformer_stack, config):
    """Base model parameters have requires_grad=False."""
    injector = PhaseD2Injector(dummy_transformer_stack, config)
    
    # Freeze base model parameters
    for param in dummy_transformer_stack.parameters():
        param.requires_grad = False
    
    # Adapters should have requires_grad=True
    for param in injector.adapters.parameters():
        assert param.requires_grad
    
    for param in injector.encoder.parameters():
        assert param.requires_grad


def test_adapter_gradient_flow(dummy_transformer_stack, config):
    """Gradients flow only to adapter parameters."""
    # Setup
    injector = PhaseD2Injector(dummy_transformer_stack, config)
    injector.attach()
    
    # Freeze base
    for param in dummy_transformer_stack.parameters():
        param.requires_grad = False
    
    # Optimizer for adapters only
    adapter_params = injector.get_adapter_parameters()
    optimizer = optim.SGD(adapter_params, lr=0.01)
    
    # Forward pass
    x = torch.randn(2, 50, 256, requires_grad=False)
    cond = torch.randn(2, 50, config.condition_dim)
    injector.set_condition(cond)
    
    # Forward through stack
    y = x.clone()
    for layer in dummy_transformer_stack:
        y = layer(y)
    
    # Loss and backward
    loss = y.mean()
    loss.backward()
    
    # Check gradients
    for param in injector.adapters.parameters():
        if param.requires_grad:
            # Some adapter params should have gradients
            assert param.grad is not None
    
    for param in dummy_transformer_stack.parameters():
        # Base params should have no gradients
        assert param.grad is None
    
    injector.detach()


def test_optimizer_step_updates_only_adapters(dummy_transformer_stack, config):
    """Optimizer step updates only adapter parameters."""
    injector = PhaseD2Injector(dummy_transformer_stack, config)
    injector.attach()
    
    # Save base parameters before
    base_params_before = [p.clone() for p in dummy_transformer_stack.parameters()]
    
    # Freeze base
    for param in dummy_transformer_stack.parameters():
        param.requires_grad = False
    
    # Get adapter params and preserve their initial values.
    adapter_params = injector.get_adapter_parameters()
    adapter_params_before = [p.detach().clone() for p in adapter_params]
    optimizer = optim.SGD(adapter_params, lr=1.0)

    x = torch.randn(2, 20, 256)
    cond = torch.randn(2, 20, config.condition_dim)
    injector.set_condition(cond)
    y = x
    for layer in dummy_transformer_stack:
        y = layer(y)
    y.square().mean().backward()
    optimizer.step()
    
    # Base params should be unchanged
    for param_before, param_after in zip(base_params_before, dummy_transformer_stack.parameters()):
        assert torch.allclose(param_before, param_after)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(adapter_params_before, adapter_params)
    )
    injector.detach()
