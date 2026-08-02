"""
Temporal Condition Encoder for Phase D2.

Encodes per-frame condition tensors via coarse (phrase-level) and fine (frame-level)
pathways, returning [B, T, hidden_size] latent.
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConditionEncoder(nn.Module):
    """Encode dense per-frame conditions into latent features.

    Architecture:
    - Input: [B, T, condition_dim]
    - Coarse pathway: average-pool over T, project, broadcast back to [B, T, hidden]
    - Fine pathway: depthwise-like conv + residual MLPs, output [B, T, hidden]
    - Concatenate coarse + fine, project to [B, T, hidden_size]

    Attributes:
        condition_dim: Input condition dimensionality.
        hidden_size: Output latent dimension.
    """

    def __init__(
        self,
        condition_dim: int,
        hidden_size: int = 256,
        n_fine_layers: int = 2,
    ):
        """Initialize encoder.

        Args:
            condition_dim: Input condition vector dimension.
            hidden_size: Output hidden dimension.
            n_fine_layers: Number of residual MLP blocks in fine pathway.
        """
        super().__init__()

        self.condition_dim = condition_dim
        self.hidden_size = hidden_size
        self.n_fine_layers = n_fine_layers

        # Coarse pathway: global averaging + projection
        self.coarse_proj_down = nn.Linear(condition_dim, hidden_size // 2)
        self.coarse_proj_up = nn.Linear(hidden_size // 2, hidden_size)

        # Fine pathway
        self.fine_proj = nn.Linear(condition_dim, hidden_size)

        # Lightweight depthwise-like processing
        self.fine_conv1d = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=3,
            padding=1,
            groups=min(hidden_size, 32),  # depthwise or grouped
            bias=False,
        )

        # Residual MLP stack
        self.fine_mlps = nn.ModuleList([
            _ResidualMLP(hidden_size, hidden_size * 2)
            for _ in range(n_fine_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(2 * hidden_size, hidden_size)

    def forward(
        self,
        condition: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode condition tensor.

        Args:
            condition: [B, T, condition_dim] float tensor.
            mask: Optional [B, T] bool mask (True = valid, False = padding/ignore).

        Returns:
            [B, T, hidden_size] float tensor.
        """
        B, T, C = condition.shape
        assert C == self.condition_dim

        # Coarse pathway: global averaging (ignoring mask for simplicity)
        coarse_pool = condition.mean(dim=1, keepdim=True)  # [B, 1, C]
        coarse = F.relu(self.coarse_proj_down(coarse_pool))  # [B, 1, H/2]
        coarse = self.coarse_proj_up(coarse)  # [B, 1, H]
        coarse = coarse.expand(B, T, -1)  # [B, T, H]

        # Fine pathway: per-frame processing
        fine = F.relu(self.fine_proj(condition))  # [B, T, H]

        # Conv1d: transpose to [B, H, T]
        fine = fine.transpose(1, 2)
        fine = F.relu(self.fine_conv1d(fine))
        fine = fine.transpose(1, 2)  # Back to [B, T, H]

        # Residual MLP blocks
        for mlp in self.fine_mlps:
            fine = mlp(fine)  # [B, T, H]

        # Concatenate pathways
        fused = torch.cat([coarse, fine], dim=-1)  # [B, T, 2H]

        # Output projection
        output = self.output_proj(fused)  # [B, T, H]

        return output


class _ResidualMLP(nn.Module):
    """Simple residual MLP: x + MLP(LN(x))."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        """Initialize.

        Args:
            hidden_size: Input/output dimension.
            intermediate_size: Hidden layer dimension.
        """
        super().__init__()

        self.ln = nn.LayerNorm(hidden_size)
        self.up = nn.Linear(hidden_size, intermediate_size)
        self.down = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual MLP.

        Args:
            x: [B, T, hidden_size] or [B, hidden_size].

        Returns:
            Same shape as input.
        """
        residual = x
        x = self.ln(x)
        x = F.silu(self.up(x))
        x = self.down(x)
        return residual + x
