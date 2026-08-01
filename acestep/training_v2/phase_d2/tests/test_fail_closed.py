"""Tests for fail-closed design: validation and graceful rejection."""

import pytest
import torch
import torch.nn as nn

from acestep.training_v2.phase_d2.config import PhaseD2Config
from acestep.training_v2.phase_d2.model_integration import PhaseD2Injector
from acestep.training_v2.phase_d2.sidecar_schema import validate_sidecar


@pytest.fixture
def dummy_transformer_stack():
    """Fixture: small transformer stack."""
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


def test_sidecar_validator_rejects_missing_field():
    """Validator raises on missing required field."""
    invalid_sidecar = {"absolute_time": torch.zeros(100)}  # Missing other fields
    
    with pytest.raises(KeyError):
        validate_sidecar(invalid_sidecar)


def test_sidecar_validator_rejects_wrong_version():
    """Validator raises on schema version mismatch."""
    sidecar = {"_schema_version": "phase_d2_sidecar_v999"}
    
    with pytest.raises(ValueError, match="Schema version"):
        validate_sidecar(sidecar)


def test_injector_rejects_wrong_condition_shape(dummy_transformer_stack, config):
    """Injector.set_condition raises on invalid shape."""
    injector = PhaseD2Injector(dummy_transformer_stack, config)
    
    # Wrong shape: 2D instead of 3D
    bad_cond = torch.randn(2, 50)
    
    with pytest.raises(ValueError, match="3D"):
        injector.set_condition(bad_cond)


def test_injector_rejects_wrong_condition_dim(dummy_transformer_stack, config):
    """Injector.set_condition raises on wrong condition_dim."""
    injector = PhaseD2Injector(dummy_transformer_stack, config)
    
    # Wrong last dimension
    bad_cond = torch.randn(2, 50, 999)
    
    with pytest.raises(ValueError, match="condition_dim"):
        injector.set_condition(bad_cond)


def test_injector_invalid_injection_layer_indices(dummy_transformer_stack, config):
    """Injector raises on out-of-bounds layer indices."""
    config.injection_layer_indices = [0, 1, 100]  # 100 is out of bounds
    
    with pytest.raises(ValueError, match="exceeds"):
        PhaseD2Injector(dummy_transformer_stack, config)


def test_bad_config_invalid_schedule():
    """Config raises on invalid control_schedule."""
    with pytest.raises(ValueError, match="control_schedule"):
        PhaseD2Config(control_schedule="invalid_schedule")


def test_bad_config_invalid_schema_version():
    """Config raises on invalid schema_version."""
    with pytest.raises(ValueError, match="schema_version"):
        PhaseD2Config(schema_version="phase_d2_sidecar_v999")


def test_bad_config_negative_frame_rate():
    """Config raises on negative frame rate."""
    with pytest.raises(ValueError, match="latent_frame_rate_hz"):
        PhaseD2Config(latent_frame_rate_hz=-1.0)
