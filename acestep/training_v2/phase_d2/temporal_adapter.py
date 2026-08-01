"""Content-dependent, zero-initialized temporal residual adapter for Phase D2."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroInitTemporalAdapter(nn.Module):
    """Produce a hidden-and-condition-dependent residual with an exact zero route.

    The final projection is initialized exactly to zero. An explicit all-zero
    condition is also masked to exact zero after training, while nonzero controls
    flow through independent hidden-state and condition projections.
    """

    def __init__(
        self,
        condition_dim: int,
        hidden_size: int,
        adapter_dim: int | None = None,
    ) -> None:
        """Initialize projections and the zero-initialized residual output."""
        super().__init__()
        self.condition_dim = condition_dim
        self.hidden_size = hidden_size
        self.adapter_dim = adapter_dim or (hidden_size // 2)

        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.condition_norm = nn.LayerNorm(condition_dim)
        self.hidden_projection = nn.Linear(hidden_size, self.adapter_dim, bias=False)
        self.condition_projection = nn.Linear(condition_dim, self.adapter_dim, bias=False)
        self.temporal_projection = nn.Linear(self.adapter_dim, self.adapter_dim)
        self.output_projection = nn.Linear(self.adapter_dim, hidden_size)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        condition: torch.Tensor,
        control_strength: float = 1.0,
    ) -> torch.Tensor:
        """Return a residual for [batch, frames, hidden] inputs."""
        if hidden_states.shape[:-1] != condition.shape[:-1]:
            raise ValueError("hidden_states and condition batch/frame dimensions must match")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states width does not match adapter hidden_size")
        if condition.shape[-1] != self.condition_dim:
            raise ValueError("condition width does not match adapter condition_dim")

        active = condition.ne(0).any(dim=-1, keepdim=True)
        if not bool(active.any()) or control_strength == 0.0:
            return torch.zeros_like(hidden_states)
        hidden = self.hidden_projection(self.hidden_norm(hidden_states))
        control = self.condition_projection(self.condition_norm(condition))
        fused = F.silu(self.temporal_projection(F.silu(hidden + control)))
        residual = self.output_projection(fused)
        return residual * active.to(residual.dtype) * control_strength
