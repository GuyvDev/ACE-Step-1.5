from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


EXPRESSIVITY_TARGET_NAMES = (
    "f0_target",
    "energy_target",
    "terminal_decay",
    "cv_ratio_target",
)

NUM_RELEASE_CLASSES = 3


def build_release_class_targets(terminal_decay: torch.Tensor) -> torch.Tensor:
    if terminal_decay.ndim == 3 and terminal_decay.shape[-1] == 1:
        terminal_decay = terminal_decay[..., 0]
    elif terminal_decay.ndim == 2 and terminal_decay.shape[-1] == 1:
        terminal_decay = terminal_decay[:, 0]
    clipped = terminal_decay < 0.55
    carried = terminal_decay > 1.15
    targets = torch.ones_like(terminal_decay, dtype=torch.long)
    targets = torch.where(clipped, torch.zeros_like(targets), targets)
    targets = torch.where(carried, torch.full_like(targets, 2), targets)
    return targets


def build_confidence_weights(
    alignment_confidence: Optional[torch.Tensor],
    beat_confidence: Optional[torch.Tensor],
    timing_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    reference = alignment_confidence if alignment_confidence is not None else beat_confidence
    if reference is None:
        if timing_mask is None:
            raise ValueError("Need at least one of alignment_confidence, beat_confidence, or timing_mask")
        weight = timing_mask.float()
    else:
        weight = torch.ones_like(reference, dtype=torch.float32)
    if alignment_confidence is not None:
        weight = weight * alignment_confidence.float()
    if beat_confidence is not None:
        weight = weight * beat_confidence.float()
    if timing_mask is not None:
        weight = weight * timing_mask.float()
    return weight.clamp(min=0.0)


def compute_expressivity_loss(
    predicted_targets: torch.Tensor,
    predicted_release_logits: torch.Tensor,
    *,
    f0_targets: Optional[torch.Tensor],
    energy_targets: Optional[torch.Tensor],
    terminal_decay: Optional[torch.Tensor],
    cv_ratio_targets: Optional[torch.Tensor],
    release_targets: Optional[torch.Tensor],
    alignment_confidence: Optional[torch.Tensor],
    beat_confidence: Optional[torch.Tensor],
    timing_mask: Optional[torch.Tensor],
    f0_weight: float = 0.4,
    energy_weight: float = 0.4,
    terminal_weight: float = 0.5,
    cv_ratio_weight: float = 0.3,
    release_weight: float = 0.2,
) -> torch.Tensor:
    weights = build_confidence_weights(alignment_confidence, beat_confidence, timing_mask).unsqueeze(-1)
    denom = weights.sum().clamp(min=1.0)
    total = predicted_targets.new_tensor(0.0)

    if f0_targets is not None:
        total = total + f0_weight * ((F.huber_loss(predicted_targets[..., 0:1], f0_targets, reduction="none")) * weights).sum() / denom
    if energy_targets is not None:
        total = total + energy_weight * ((F.huber_loss(predicted_targets[..., 1:2], energy_targets, reduction="none")) * weights).sum() / denom
    if terminal_decay is not None:
        total = total + terminal_weight * ((F.huber_loss(predicted_targets[..., 2:3], terminal_decay, reduction="none")) * weights).sum() / denom
    if cv_ratio_targets is not None:
        total = total + cv_ratio_weight * ((F.huber_loss(predicted_targets[..., 3:4], cv_ratio_targets, reduction="none")) * weights).sum() / denom
    if release_targets is not None:
        ce = F.cross_entropy(
            predicted_release_logits.reshape(-1, predicted_release_logits.shape[-1]),
            release_targets.reshape(-1),
            reduction="none",
        ).view_as(release_targets).unsqueeze(-1)
        total = total + release_weight * (ce * weights).sum() / denom
    return total


class DecoderExpressivitySupervisor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )
        self.expressivity_head = nn.Linear(hidden_size, len(EXPRESSIVITY_TARGET_NAMES))
        self.release_head = nn.Linear(hidden_size, NUM_RELEASE_CLASSES)
        nn.init.zeros_(self.expressivity_head.weight)
        nn.init.zeros_(self.expressivity_head.bias)
        nn.init.zeros_(self.release_head.weight)
        nn.init.zeros_(self.release_head.bias)

    def forward(
        self,
        *,
        decoder_event_states: torch.Tensor,
        f0_targets: Optional[torch.Tensor],
        energy_targets: Optional[torch.Tensor],
        terminal_decay: Optional[torch.Tensor],
        cv_ratio_targets: Optional[torch.Tensor],
        release_targets: Optional[torch.Tensor],
        alignment_confidence: Optional[torch.Tensor],
        beat_confidence: Optional[torch.Tensor],
        timing_mask: Optional[torch.Tensor],
        f0_weight: float = 0.4,
        energy_weight: float = 0.4,
        terminal_weight: float = 0.5,
        cv_ratio_weight: float = 0.3,
        release_weight: float = 0.2,
    ) -> torch.Tensor:
        hidden = self.proj(decoder_event_states)
        return compute_expressivity_loss(
            self.expressivity_head(hidden),
            self.release_head(hidden),
            f0_targets=f0_targets,
            energy_targets=energy_targets,
            terminal_decay=terminal_decay,
            cv_ratio_targets=cv_ratio_targets,
            release_targets=release_targets,
            alignment_confidence=alignment_confidence,
            beat_confidence=beat_confidence,
            timing_mask=timing_mask,
            f0_weight=f0_weight,
            energy_weight=energy_weight,
            terminal_weight=terminal_weight,
            cv_ratio_weight=cv_ratio_weight,
            release_weight=release_weight,
        )
