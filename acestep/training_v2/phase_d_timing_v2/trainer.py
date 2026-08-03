"""B3 - training core for Phase D Timing v2.

Model-agnostic so the sampler, loss normalisation, checkpointing and exact
resume can be proven on a tiny CPU fixture without loading the 2.7 B parent.

Resume restores: model, optimizer, scheduler, scaler, global step, epoch, data
position, sampler state, and the Python / NumPy / Torch-CPU / all Torch-CUDA
RNG states.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict, field
from typing import Any, Iterator

import numpy as np
import torch

EPS = 1e-6


# ---------------------------------------------------------------- sampler
@dataclass
class PairedSamplerState:
    epoch: int = 0
    position: int = 0
    seed: int = 20260803


class DeterministicPairedSampler:
    """Deterministic 50/50 original / counterfactual interleave.

    Both pools are shuffled per epoch from a seed derived from
    ``(seed, epoch)``, then emitted strictly alternately. The shorter pool is
    deterministically oversampled by wrapping, which is recorded rather than
    hidden. State is (epoch, position) so resume is exact.
    """

    def __init__(self, originals: list[int], counterfactuals: list[int], seed: int = 20260803):
        if not originals or not counterfactuals:
            raise ValueError("both pools must be non-empty")
        self.originals = list(originals)
        self.counterfactuals = list(counterfactuals)
        self.state = PairedSamplerState(seed=seed)
        self.epoch_length = 2 * max(len(self.originals), len(self.counterfactuals))

    def _order(self, epoch: int) -> tuple[list[int], list[int]]:
        rng = random.Random((self.state.seed, epoch).__hash__() & 0xFFFFFFFF)
        a, b = list(self.originals), list(self.counterfactuals)
        rng.shuffle(a)
        rng.shuffle(b)
        return a, b

    def index_at(self, epoch: int, position: int) -> tuple[int, str]:
        a, b = self._order(epoch)
        pair, side = divmod(position, 2)
        if side == 0:
            return a[pair % len(a)], "original"
        return b[pair % len(b)], "counterfactual"

    def __iter__(self) -> Iterator[tuple[int, str]]:
        while True:
            if self.state.position >= self.epoch_length:
                self.state.epoch += 1
                self.state.position = 0
            item = self.index_at(self.state.epoch, self.state.position)
            # Advance BEFORE yielding. A generator suspends at the yield, so
            # incrementing afterwards would leave state_dict() pointing at an
            # item that was already consumed - resume would replay it.
            self.state.position += 1
            yield item

    def oversampling(self) -> dict[str, Any]:
        longer = max(len(self.originals), len(self.counterfactuals))
        return {
            "originals": len(self.originals),
            "counterfactuals": len(self.counterfactuals),
            "epoch_length": self.epoch_length,
            "original_repeat_factor": round(longer / len(self.originals), 4),
            "counterfactual_repeat_factor": round(longer / len(self.counterfactuals), 4),
            "ratio": "50/50 by construction (strict alternation)",
        }

    def state_dict(self) -> dict[str, Any]:
        return asdict(self.state)

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.state = PairedSamplerState(**payload)


# ---------------------------------------------------------------- losses
def masked_mean(error: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """sum(mask * error) / sum(mask) - never a mean over all frames.

    The mask is broadcast to the error's shape before summing. Using the raw
    mask sum would divide a per-channel numerator by a per-frame denominator,
    silently scaling every loss term by the channel count.
    """
    weight = mask.expand_as(error).sum()
    return (mask * error).sum() / torch.clamp(weight, min=EPS)


def timing_losses(
    flow_d: torch.Tensor,
    flow_target: torch.Tensor,
    flow_c25: torch.Tensor,
    target_mask: torch.Tensor,
    boundary_mask: torch.Tensor,
    weights: tuple[float, float, float] = (1.0, 3.0, 0.25),
) -> dict[str, torch.Tensor]:
    """Mask-normalised inside / outside / boundary terms."""
    inside_err = (flow_d - flow_target).abs()
    outside_err = (flow_d - flow_c25).abs()
    outside_mask = torch.clamp(1.0 - target_mask - boundary_mask, min=0.0)

    l_inside = masked_mean(inside_err, target_mask)
    l_outside = masked_mean(outside_err, outside_mask)
    l_boundary = masked_mean(outside_err, boundary_mask)
    w_in, w_out, w_bnd = weights
    total = w_in * l_inside + w_out * l_outside + w_bnd * l_boundary
    return {"loss": total, "l_inside": l_inside, "l_outside": l_outside, "l_boundary": l_boundary}


# ---------------------------------------------------------------- assertions
def assert_frozen_parent(model: torch.nn.Module, trainable: list[torch.nn.Parameter]) -> dict[str, Any]:
    """Parent frozen, optimizer owns only adapter tensors, no id overlap."""
    base_ids = {id(p) for p in model.parameters()}
    trainable_ids = {id(p) for p in trainable}
    overlap = base_ids & trainable_ids
    base_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    if overlap:
        raise RuntimeError(f"parameter-ID overlap between parent and adapters: {len(overlap)}")
    if base_trainable:
        raise RuntimeError(f"parent has {base_trainable} trainable parameters, expected 0")
    return {
        "base_parameter_count": sum(p.numel() for p in model.parameters()),
        "base_trainable_count": 0,
        "d2_trainable_count": sum(p.numel() for p in trainable),
        "parameter_id_overlap": 0,
    }


def assert_zero_init(adapter: torch.nn.Module, condition_dim: int, hidden: torch.Tensor) -> dict[str, Any]:
    """Explicit zero condition must give an exactly zero residual."""
    zero = torch.zeros(hidden.shape[0], hidden.shape[1], condition_dim,
                       dtype=hidden.dtype, device=hidden.device)
    with torch.no_grad():
        residual = adapter(hidden_states=hidden, condition=zero)
    value = float(residual.abs().max())
    if value != 0.0:
        raise RuntimeError(f"zero condition produced non-zero residual: {value}")
    return {"zero_condition_max_abs_residual": value, "zero_init_exact": True}


# ---------------------------------------------------------------- rng state
def capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu() if torch.is_tensor(state["torch_cpu"]) else state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all([s.cpu() if torch.is_tensor(s) else s for s in state["torch_cuda"]])


# ---------------------------------------------------------------- checkpoint
@dataclass
class TrainState:
    global_step: int = 0
    epoch: int = 0
    data_position: int = 0
    best_metric: float = float("inf")
    history: list[dict[str, Any]] = field(default_factory=list)


def save_checkpoint(
    path,
    adapters: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    sampler: DeterministicPairedSampler,
    state: TrainState,
) -> None:
    torch.save({
        "schema": "phase_d_timing_v2_checkpoint_v1",
        "model": adapters.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "sampler": sampler.state_dict(),
        "train_state": asdict(state),
        "rng": capture_rng(),
    }, path)


def load_checkpoint(
    path,
    adapters: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    sampler: DeterministicPairedSampler,
) -> TrainState:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "phase_d_timing_v2_checkpoint_v1":
        raise RuntimeError(f"unexpected checkpoint schema: {payload.get('schema')}")
    adapters.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    sampler.load_state_dict(payload["sampler"])
    restore_rng(payload["rng"])
    return TrainState(**payload["train_state"])


__all__ = [
    "DeterministicPairedSampler", "PairedSamplerState", "TrainState",
    "masked_mean", "timing_losses", "assert_frozen_parent", "assert_zero_init",
    "capture_rng", "restore_rng", "save_checkpoint", "load_checkpoint",
]
