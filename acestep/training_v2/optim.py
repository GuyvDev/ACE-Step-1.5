"""
Side-Step Optimizer & Scheduler Factories

Provides ``build_optimizer()`` and ``build_scheduler()`` so that
``trainer_fixed.py`` doesn't need to hard-code AdamW / CosineAnnealing.

Supported optimizers:
    adamw       -- torch.optim.AdamW (default, fused on CUDA)
    adamw8bit   -- bitsandbytes.optim.AdamW8bit (optional dep)
    adafactor   -- transformers.optimization.Adafactor
    prodigy     -- prodigyopt.Prodigy (optional dep, auto-tunes LR)

Supported schedulers:
    cosine              -- warmup + CosineAnnealingLR (single smooth decay)
    cosine_restarts     -- warmup + CosineAnnealingWarmRestarts (cyclical)
    linear              -- warmup + LinearLR decay to near-zero
    constant            -- warmup then flat LR
    constant_with_warmup -- alias for constant
"""

from __future__ import annotations

import logging
import math
from typing import Iterable

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ConstantLR,
    LRScheduler,
    LinearLR,
    SequentialLR,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def build_optimizer(
    params: Iterable,
    optimizer_type: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    device_type: str = "cuda",
) -> torch.optim.Optimizer:
    """Create exactly the requested optimizer or raise.

    Missing optional dependencies and unknown optimizer names are fatal.
    """
    optimizer_type = optimizer_type.lower().strip()

    if optimizer_type == "adamw8bit":
        try:
            from bitsandbytes.optim import AdamW8bit
            logger.info("[Side-Step] Using AdamW8bit optimizer (lower VRAM)")
            return AdamW8bit(params, lr=lr, weight_decay=weight_decay)
        except ImportError as exc:
            raise RuntimeError(
                "optimizer_type='adamw8bit' requires bitsandbytes>=0.45.0"
            ) from exc

    if optimizer_type == "adafactor":
        try:
            from transformers.optimization import Adafactor
            logger.info("[Side-Step] Using Adafactor optimizer (minimal state memory)")
            return Adafactor(
                params,
                lr=lr,
                weight_decay=weight_decay,
                scale_parameter=False,
                relative_step=False,
            )
        except ImportError as exc:
            raise RuntimeError(
                "optimizer_type='adafactor' requires transformers"
            ) from exc

    if optimizer_type == "prodigy":
        try:
            from prodigyopt import Prodigy
            logger.info(
                "[Side-Step] Using Prodigy optimizer (adaptive LR -- set LR=1.0 for best results)"
            )
            return Prodigy(
                params,
                lr=lr,
                weight_decay=weight_decay,
            )
        except ImportError as exc:
            raise RuntimeError(
                "optimizer_type='prodigy' requires prodigyopt>=1.1.2"
            ) from exc

    if optimizer_type != "adamw":
        raise ValueError(f"unsupported optimizer_type: {optimizer_type!r}")
    kwargs = {"lr": lr, "weight_decay": weight_decay}
    if device_type == "cuda":
        kwargs["fused"] = True
    logger.info("[Side-Step] Using AdamW optimizer")
    return AdamW(params, **kwargs)


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------


class GroupRatioCosineAnnealingLR(LRScheduler):
    """Cosine decay by one scalar factor, preserving every group LR ratio."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        min_factor: float = 0.01,
        last_epoch: int = -1,
    ) -> None:
        if int(T_max) <= 0:
            raise ValueError(f"T_max must be positive, got {T_max}")
        if not 0.0 <= float(min_factor) <= 1.0:
            raise ValueError(f"min_factor must be in [0,1], got {min_factor}")
        self.T_max = int(T_max)
        self.min_factor = float(min_factor)
        super().__init__(optimizer, last_epoch=last_epoch)

    def _factor(self, step: int) -> float:
        progress = min(max(int(step), 0), self.T_max) / self.T_max
        return self.min_factor + (1.0 - self.min_factor) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    def get_lr(self) -> list[float]:
        factor = self._factor(self.last_epoch)
        return [base_lr * factor for base_lr in self.base_lrs]

    def _get_closed_form_lr(self) -> list[float]:
        return self.get_lr()

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    total_steps: int = 1000,
    warmup_steps: int = 500,
    lr: float = 1e-4,
    optimizer_type: str = "adamw",
    n_restarts: int = 4,
):
    """Create a learning rate scheduler from a string key.

    Args:
        n_restarts: Number of cosine restart cycles for the
            ``cosine_restarts`` scheduler.  Ignored by other types.

    When the optimizer is Prodigy, defaults to constant schedule
    (Prodigy manages LR internally).
    """
    scheduler_type = scheduler_type.lower().strip()

    if optimizer_type == "prodigy" and scheduler_type not in ("constant", "constant_with_warmup"):
        raise ValueError("Prodigy requires an explicitly selected constant scheduler")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    warmup_steps = int(warmup_steps)
    if warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError(
            f"warmup_steps must be in [0, total_steps), got {warmup_steps}/{total_steps}"
        )
    warmup_sched = None
    if warmup_steps > 0:
        warmup_sched = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

    remaining = max(1, total_steps - warmup_steps)

    if scheduler_type in ("constant", "constant_with_warmup"):
        main_sched = ConstantLR(optimizer, factor=1.0, total_iters=total_steps)
    elif scheduler_type == "linear":
        main_sched = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=remaining,
        )
    elif scheduler_type == "cosine_restarts":
        # Cyclical cosine: LR resets to peak multiple times during training.
        # T_0 = cycle length = remaining / n_restarts.
        main_sched = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, remaining // max(1, n_restarts)),
            T_mult=1,
            eta_min=lr * 0.01,
        )
    elif scheduler_type == "cosine":
        # Preserve per-group LR ratios for timing training.  A scalar eta_min
        # would collapse every group toward the base LR floor.
        if len(optimizer.param_groups) > 1:
            main_sched = GroupRatioCosineAnnealingLR(
                optimizer,
                T_max=remaining,
                min_factor=0.01,
            )
        else:
            main_sched = CosineAnnealingLR(
                optimizer,
                T_max=remaining,
                eta_min=lr * 0.01,
            )
    else:
        raise ValueError(f"unsupported scheduler_type: {scheduler_type!r}")

    if warmup_sched is None:
        return main_sched
    return SequentialLR(optimizer, [warmup_sched, main_sched], milestones=[warmup_steps])
