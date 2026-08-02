"""
Configurable Loss Registry for Phase D2.

Provides individual loss functions and a registry for combining them
based on config flags.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(error: torch.Tensor, mask: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Return an active-mask-normalized sum, never a mean over inactive frames."""
    active = mask.to(device=error.device, dtype=error.dtype)
    while active.ndim < error.ndim:
        active = active.unsqueeze(-1)
    return (active * error).sum() / active.sum().clamp_min(epsilon)


def timing_v2_objective(
    phase_d_flow: torch.Tensor,
    target_flow: torch.Tensor,
    c25_flow: torch.Tensor,
    target_mask: torch.Tensor,
    boundary_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the normative inside, outside, and four-frame-boundary objective."""
    inside = masked_mean((phase_d_flow - target_flow).abs(), target_mask)
    outside_mask = ~(target_mask.bool() | boundary_mask.bool())
    outside = masked_mean((phase_d_flow - c25_flow).abs(), outside_mask)
    boundary = masked_mean((phase_d_flow - c25_flow).abs(), boundary_mask)
    terms = {"inside": inside, "outside": outside, "boundary": boundary}
    return inside + 3.0 * outside + 0.25 * boundary, terms


def l_null_preservation(
    d2_output: torch.Tensor,
    c25_output: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """L1 loss ensuring D2 preserves C25 output when condition is null.

    Args:
        d2_output: [B, T, H] output from D2-modified model.
        c25_output: [B, T, H] reference output from frozen C25 model.
        reduction: "mean" or "sum".

    Returns:
        Scalar loss tensor.
    """
    return F.l1_loss(d2_output, c25_output, reduction=reduction)


def l_outside_preservation(
    d2_output: torch.Tensor,
    c25_output: torch.Tensor,
    window_mask: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """L1 loss ensuring D2 does not modify frames outside intervention window.

    Args:
        d2_output: [B, T, H] output from D2.
        c25_output: [B, T, H] reference C25 output.
        window_mask: [B, T] bool (True = inside window, False = outside).
        reduction: "mean" or "sum".

    Returns:
        Scalar loss on outside frames only.
    """
    outside_mask = ~window_mask  # [B, T]
    outside_mask = outside_mask.unsqueeze(-1).float()  # [B, T, 1]

    diff = F.l1_loss(d2_output, c25_output, reduction="none")  # [B, T, H]
    masked_diff = masked_mean(diff, outside_mask)

    return masked_diff


def l_word_duration(
    predicted_log_durations: torch.Tensor,
    target_log_durations: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """MSE on log-space word durations.

    Args:
        predicted_log_durations: [B, n_words] or [B, T] log durations in seconds.
        target_log_durations: [B, n_words] or [B, T] target log durations.
        mask: Optional [B, n_words] bool (True = valid).
        reduction: "mean" or "sum".

    Returns:
        Scalar loss.
    """
    loss = F.mse_loss(predicted_log_durations, target_log_durations, reduction="none")

    if mask is not None:
        mask = mask.float()
        loss = masked_mean(loss, mask)
    else:
        loss = loss.mean() if reduction == "mean" else loss.sum()

    return loss


def l_phrase_duration(
    predicted_log_durations: torch.Tensor,
    target_log_durations: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """MSE on log-space phrase durations.

    Args:
        predicted_log_durations: [B, n_phrases] log durations.
        target_log_durations: [B, n_phrases] target log durations.
        mask: Optional [B, n_phrases] bool.
        reduction: "mean" or "sum".

    Returns:
        Scalar loss.
    """
    return l_word_duration(
        predicted_log_durations,
        target_log_durations,
        mask=mask,
        reduction=reduction,
    )


def l_f0(
    predicted_f0: torch.Tensor,
    target_f0: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Masked L1 loss on F0 values.

    Args:
        predicted_f0: [B, T] Hz or normalized F0.
        target_f0: [B, T] target F0.
        mask: Optional [B, T] bool (True = voiced/valid).
        reduction: "mean" or "sum".

    Returns:
        Scalar loss.
    """
    loss = F.l1_loss(predicted_f0, target_f0, reduction="none")

    if mask is not None:
        mask = mask.float()
        loss = masked_mean(loss, mask)
    else:
        loss = loss.mean() if reduction == "mean" else loss.sum()

    return loss


def l_counterfactual_direction(
    realized_shift: torch.Tensor,
    requested_shift: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Hinge-like loss: penalize shifts in wrong direction.

    If requested_shift > 0, we want realized_shift > 0.
    If requested_shift < 0, we want realized_shift < 0.

    Loss = ReLU(-realized_shift * sign(requested_shift))

    Args:
        realized_shift: [B, T] or scalar actual measured shift.
        requested_shift: [B, T] or scalar requested shift direction.
        reduction: "mean" or "sum".

    Returns:
        Scalar loss.
    """
    # Get sign of requested shift; treat 0 as "don't care" (loss=0)
    sign = torch.sign(requested_shift)

    # Loss when realized*sign < 0 (opposite direction)
    loss = F.relu(-realized_shift * sign)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def l_flow(
    predicted_flow: torch.Tensor,
    target_flow: torch.Tensor,
) -> torch.Tensor:
    """Placeholder flow loss (stub for future use).

    Args:
        predicted_flow: Some flow representation.
        target_flow: Target flow.

    Returns:
        Zero loss (passthrough).
    """
    return torch.tensor(0.0, device=predicted_flow.device, dtype=predicted_flow.dtype)


class LossRegistry:
    """Registry of Phase D2 losses with configurable enabling."""

    LOSS_FUNCTIONS = {
        "flow": l_flow,
        "null_preservation": l_null_preservation,
        "outside_preservation": l_outside_preservation,
        "word_duration": l_word_duration,
        "phrase_duration": l_phrase_duration,
        "f0": l_f0,
        "counterfactual_direction": l_counterfactual_direction,
    }

    def __init__(
        self,
        enabled_losses: Dict[str, bool] | None = None,
        loss_weights: Dict[str, float] | None = None,
    ):
        """Initialize registry.

        Args:
            enabled_losses: {loss_name: bool} to enable/disable losses.
                Defaults: all False except null_preservation and outside_preservation.
            loss_weights: {loss_name: weight} for weighted combination.
                Defaults: all 1.0.
        """
        if enabled_losses is None:
            enabled_losses = {
                "null_preservation": True,
                "outside_preservation": True,
            }

        self.enabled_losses = {name: False for name in self.LOSS_FUNCTIONS}
        self.enabled_losses.update(enabled_losses)

        if loss_weights is None:
            loss_weights = {name: 1.0 for name in self.LOSS_FUNCTIONS}

        self.loss_weights = {name: 1.0 for name in self.LOSS_FUNCTIONS}
        self.loss_weights.update(loss_weights)

    def compute_total_loss(
        self,
        loss_dict: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute weighted sum of enabled losses.

        Args:
            loss_dict: {loss_name: loss_tensor} from individual loss functions.

        Returns:
            Tuple of:
            - total_loss: Scalar tensor.
            - loss_breakdown: {loss_name: float value} for logging.
        """
        total_loss = torch.tensor(0.0, dtype=torch.float32)
        breakdown = {}

        for name, loss_tensor in loss_dict.items():
            if name not in self.LOSS_FUNCTIONS:
                continue

            if self.enabled_losses.get(name, False):
                weight = self.loss_weights.get(name, 1.0)
                total_loss = total_loss + weight * loss_tensor
                breakdown[name] = loss_tensor.detach().item()
            else:
                breakdown[name] = 0.0

        return total_loss, breakdown
