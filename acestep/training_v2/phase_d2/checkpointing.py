"""
Checkpointing for Phase D2.

Save and load adapter+encoder parameters with integrity checking.
"""

from __future__ import annotations

from typing import Dict, Any
from pathlib import Path
import hashlib
import torch

from acestep.training_v2.phase_d2.model_integration import PhaseD2Injector


def save_adapters(
    injector: PhaseD2Injector,
    checkpoint_path: str | Path,
    include_config: bool = True,
) -> None:
    """Save adapter and encoder parameters to checkpoint.
    
    Saves:
    - adapter+encoder state_dict
    - config (if include_config=True)
    - SHA256 of each tensor for integrity checking
    
    Args:
        injector: PhaseD2Injector instance.
        checkpoint_path: Path to save checkpoint.
        include_config: Whether to include config in checkpoint.
    
    Raises:
        IOError: If save fails.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect adapter+encoder state
    state_dict = {
        f"encoder.{key}": value
        for key, value in injector.encoder.state_dict().items()
    }
    for i, adapter in enumerate(injector.adapters):
        adapter_state = adapter.state_dict()
        for key, value in adapter_state.items():
            state_dict[f"adapters.{i}.{key}"] = value
    
    # Compute tensor hashes
    tensor_hashes = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            tensor_bytes = value.cpu().numpy().tobytes()
            tensor_hash = hashlib.sha256(tensor_bytes).hexdigest()
            tensor_hashes[key] = tensor_hash
    
    # Package checkpoint
    checkpoint = {
        "state_dict": state_dict,
        "tensor_hashes": tensor_hashes,
    }
    
    if include_config:
        checkpoint["config"] = injector.config.to_dict()
    
    # Save
    torch.save(checkpoint, checkpoint_path)


def load_adapters(
    injector: PhaseD2Injector,
    checkpoint_path: str | Path,
    verify_hashes: bool = True,
) -> Dict[str, Any]:
    """Load adapter and encoder parameters from checkpoint.
    
    Optionally verifies tensor SHA256 hashes.
    
    Args:
        injector: PhaseD2Injector instance to load into.
        checkpoint_path: Path to checkpoint.
        verify_hashes: If True, verify tensor integrity (slow for large models).
    
    Returns:
        Metadata dict from checkpoint.
    
    Raises:
        ValueError: If verification fails.
        FileNotFoundError: If checkpoint not found.
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    state_dict = checkpoint["state_dict"]
    tensor_hashes = checkpoint.get("tensor_hashes", {})
    
    # Verify hashes if requested
    if verify_hashes and tensor_hashes:
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                tensor_bytes = value.cpu().numpy().tobytes()
                computed_hash = hashlib.sha256(tensor_bytes).hexdigest()
                expected_hash = tensor_hashes.get(key)
                
                if expected_hash and computed_hash != expected_hash:
                    raise ValueError(
                        f"Hash mismatch for {key}: "
                        f"expected {expected_hash}, got {computed_hash}. "
                        f"Checkpoint may be corrupted."
                    )
    
    # Load state fail-closed.  The original Phase D2 implementation saved
    # encoder keys without the ``encoder.`` namespace and consequently loaded
    # zero encoder tensors while reporting success.  Requiring the complete
    # namespaced key set prevents a partial or legacy-broken checkpoint from
    # being mistaken for a valid round trip.
    encoder_state = {
        k[len("encoder."):]: v
        for k, v in state_dict.items()
        if k.startswith("encoder.")
    }
    expected_encoder_keys = set(injector.encoder.state_dict())
    if set(encoder_state) != expected_encoder_keys:
        missing = sorted(expected_encoder_keys - set(encoder_state))
        unexpected = sorted(set(encoder_state) - expected_encoder_keys)
        raise ValueError(
            "Incomplete encoder state in Phase D2 checkpoint; "
            f"missing={missing}, unexpected={unexpected}. "
            "Legacy checkpoints written before the encoder namespace fix "
            "must not be treated as verified."
        )
    injector.encoder.load_state_dict(encoder_state, strict=True)
    
    for i, adapter in enumerate(injector.adapters):
        adapter_prefix = f"adapters.{i}."
        adapter_state = {
            k[len(adapter_prefix):]: v
            for k, v in state_dict.items()
            if k.startswith(adapter_prefix)
        }
        expected_adapter_keys = set(adapter.state_dict())
        if set(adapter_state) != expected_adapter_keys:
            missing = sorted(expected_adapter_keys - set(adapter_state))
            unexpected = sorted(set(adapter_state) - expected_adapter_keys)
            raise ValueError(
                f"Incomplete adapter state for adapter {i}; "
                f"missing={missing}, unexpected={unexpected}."
            )
        adapter.load_state_dict(adapter_state, strict=True)
    
    return {"config": checkpoint.get("config")}
