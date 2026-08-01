"""ACE-compatible temporal residual hook for the Phase D2 injector."""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F


def _hidden_output(output: torch.Tensor | tuple) -> tuple[torch.Tensor, Callable]:
    """Extract hidden states and a structure-preserving output rebuilder."""
    if isinstance(output, torch.Tensor):
        return output, lambda value: value
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0], lambda value: (value, *output[1:])
    raise TypeError(
        "Phase D2 injection requires a Tensor or tuple whose first element "
        f"is a Tensor; got {type(output)!r}."
    )


def _match_batch(value: torch.Tensor, batch_size: int, label: str) -> torch.Tensor:
    """Expand a singleton batch or fail when a control batch is incompatible."""
    if value.shape[0] == 1 and batch_size > 1:
        value = value.expand(batch_size, *value.shape[1:])
    if value.shape[0] != batch_size:
        raise ValueError(f"{label} batch {value.shape[0]} does not match hidden batch {batch_size}.")
    return value


def build_temporal_residual_hook(injector: Any, adapter: Any) -> Callable:
    """Build one hook using injector state and a layer-specific residual adapter."""
    def hook(module: Any, inputs: tuple, output: torch.Tensor | tuple) -> torch.Tensor | tuple:
        """Add a masked, temporally aligned D2 residual to an ACE layer output."""
        del module, inputs
        if injector._current_condition is None:
            return output

        hidden_states, rebuild = _hidden_output(output)
        if hidden_states.dim() != 3:
            raise ValueError(f"ACE/D2 hidden states must be [B,T,H], got {tuple(hidden_states.shape)}.")
        if hidden_states.shape[-1] != injector.config.hidden_size:
            raise ValueError(
                f"Hidden size {hidden_states.shape[-1]} does not match "
                f"D2 config.hidden_size {injector.config.hidden_size}."
            )
        condition = injector._current_condition.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        condition = _match_batch(condition, hidden_states.shape[0], "Condition")
        condition = injector.encoder(condition)
        if condition.shape[1] != hidden_states.shape[1]:
            condition = F.interpolate(
                condition.transpose(1, 2),
                size=hidden_states.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        residual = adapter(
            hidden_states=hidden_states,
            condition=condition,
            control_strength=injector._current_control_strength,
        )
        if injector._current_vocal_mask is not None:
            mask = injector._current_vocal_mask.to(hidden_states.device)
            mask = _match_batch(mask, hidden_states.shape[0], "Vocal mask")
            if mask.shape[1] != hidden_states.shape[1]:
                mask = F.interpolate(
                    mask.float().unsqueeze(1),
                    size=hidden_states.shape[1],
                    mode="nearest",
                ).squeeze(1)
            residual = residual * mask.unsqueeze(-1).float()
        return rebuild(hidden_states + residual)

    return hook
