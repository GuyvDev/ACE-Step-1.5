"""Integrity-checked checkpoints for Control Surface 4."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from acestep.training_v2.surface4.performance_regulator import PerformanceRegulator

SCHEMA = "surface4_performance_regulator_v1"


def _tensor_hash(tensor: torch.Tensor) -> str:
    """Hash one tensor's exact contiguous CPU bytes."""
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def save_regulator(
    regulator: PerformanceRegulator,
    checkpoint_path: str | Path,
    train_state: dict[str, Any],
) -> None:
    """Save the regulator state, tensor hashes, and explicit training state."""
    state = {name: tensor.detach().cpu() for name, tensor in regulator.state_dict().items()}
    payload = {
        "schema": SCHEMA,
        "model": state,
        "tensor_hashes": {name: _tensor_hash(tensor) for name, tensor in state.items()},
        "train_state": train_state,
    }
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_regulator(
    regulator: PerformanceRegulator,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Load a complete regulator checkpoint after verifying every tensor hash."""
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA:
        raise RuntimeError(f"unexpected Surface 4 checkpoint schema: {payload.get('schema')}")
    state = payload.get("model", {})
    if set(state) != set(regulator.state_dict()):
        raise RuntimeError(
            "Surface 4 checkpoint keys differ: "
            f"missing={sorted(set(regulator.state_dict()) - set(state))}, "
            f"unexpected={sorted(set(state) - set(regulator.state_dict()))}"
        )
    hashes = payload.get("tensor_hashes", {})
    corrupt = [
        name for name, tensor in state.items()
        if hashes.get(name) != _tensor_hash(tensor)
    ]
    if corrupt:
        raise RuntimeError(f"Surface 4 checkpoint hash failure: {corrupt}")
    regulator.load_state_dict(state, strict=True)
    return dict(payload.get("train_state", {}))


__all__ = ["SCHEMA", "load_regulator", "save_regulator"]
