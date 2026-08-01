"""Tests for checkpoint save/load."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path

from acestep.training_v2.phase_d2.config import PhaseD2Config
from acestep.training_v2.phase_d2.model_integration import PhaseD2Injector
from acestep.training_v2.phase_d2.checkpointing import save_adapters, load_adapters


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


def test_save_load_roundtrip(dummy_transformer_stack, config, tmp_path):
    """Save and load produces identical adapter parameters."""
    # Create injector and set seed for reproducibility
    torch.manual_seed(42)
    injector1 = PhaseD2Injector(dummy_transformer_stack, config)
    
    # Perturb adapter weights slightly to verify they're saved
    for adapter in injector1.adapters:
        with torch.no_grad():
            adapter.condition_projection.weight.add_(0.01)
            adapter.output_projection.weight.add_(0.001)
    
    # Save
    checkpoint_path = tmp_path / "adapters.pt"
    save_adapters(injector1, checkpoint_path, include_config=True)
    
    # Load checkpoint directly to verify structure
    loaded_checkpoint = torch.load(checkpoint_path)
    assert "state_dict" in loaded_checkpoint
    assert "tensor_hashes" in loaded_checkpoint
    
    # Verify both namespaces were saved.
    assert any("adapters" in k for k in loaded_checkpoint["state_dict"].keys())
    assert any(k.startswith("encoder.") for k in loaded_checkpoint["state_dict"])

    # Construct a differently initialized injector, perform the public load,
    # and compare every trainable tensor rather than only the file structure.
    torch.manual_seed(7)
    layers2 = nn.ModuleList([
        nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=1024,
            batch_first=True,
            norm_first=True,
        )
        for _ in range(4)
    ])
    injector2 = PhaseD2Injector(layers2, config)
    load_adapters(injector2, checkpoint_path)

    for (name1, value1), (name2, value2) in zip(
        injector1.encoder.state_dict().items(),
        injector2.encoder.state_dict().items(),
    ):
        assert name1 == name2
        assert torch.equal(value1, value2)
    for adapter1, adapter2 in zip(injector1.adapters, injector2.adapters):
        for (name1, value1), (name2, value2) in zip(
            adapter1.state_dict().items(), adapter2.state_dict().items()
        ):
            assert name1 == name2
            assert torch.equal(value1, value2)


def test_load_rejects_legacy_missing_encoder_namespace(
    dummy_transformer_stack, config, tmp_path
):
    """The previously silent encoder omission is now a hard failure."""
    injector = PhaseD2Injector(dummy_transformer_stack, config)
    checkpoint_path = tmp_path / "legacy_broken.pt"
    legacy_state = dict(injector.encoder.state_dict())
    torch.save({"state_dict": legacy_state, "tensor_hashes": {}}, checkpoint_path)

    with pytest.raises(ValueError, match="Incomplete encoder state"):
        load_adapters(injector, checkpoint_path, verify_hashes=False)


def test_checkpoint_contains_config(dummy_transformer_stack, config, tmp_path):
    """Saved checkpoint includes config."""
    injector = PhaseD2Injector(dummy_transformer_stack, config)
    
    checkpoint_path = tmp_path / "adapters.pt"
    save_adapters(injector, checkpoint_path, include_config=True)
    
    loaded = torch.load(checkpoint_path)
    assert "config" in loaded
    assert loaded["config"]["schema_version"] == "phase_d2_sidecar_v1"


def test_load_missing_checkpoint_raises(dummy_transformer_stack, config):
    """Loading from non-existent path raises FileNotFoundError."""
    injector = PhaseD2Injector(dummy_transformer_stack, config)
    
    with pytest.raises(FileNotFoundError):
        load_adapters(injector, "/nonexistent/path.pt")
