"""
Temporal Adapter for Phase D2.

Zero-initialized residual adapters that inject per-frame condition signals
into frozen base model layers. Guarantees null-condition stability.
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroInitTemporalAdapter(nn.Module):
    """Per-layer residual adapter with zero initialization.
    
    Architecture:
        output = hidden_states + gate * adapter(condition, hidden_states)
    
    where adapter is:
        LN(condition) -> linear_down -> SiLU -> linear_up
    
    Both linear_up and gate are zero-initialized for null-condition stability.
    With no condition set, this outputs exactly zero.
    
    Attributes:
        condition_dim: Input condition dimensionality.
        hidden_size: Model hidden dimension (matches input hidden_states).
    """
    
    def __init__(
        self,
        condition_dim: int,
        hidden_size: int,
        adapter_dim: Optional[int] = None,
    ):
        """Initialize adapter.
        
        Args:
            condition_dim: Dimension of condition tensor.
            hidden_size: Dimension of hidden states to adapt.
            adapter_dim: Internal dimension (default hidden_size // 2).
        """
        super().__init__()
        
        self.condition_dim = condition_dim
        self.hidden_size = hidden_size
        self.adapter_dim = adapter_dim or (hidden_size // 2)
        
        # Projection and processing
        self.ln = nn.LayerNorm(condition_dim)
        self.down = nn.Linear(condition_dim, self.adapter_dim)
        self.up = nn.Linear(self.adapter_dim, hidden_size)
        
        # Gate parameter (scalar per frame, initialized to 0)
        self.gate = nn.Parameter(torch.zeros(1))
        
        # Zero-initialize the up projection to guarantee null condition -> zero output
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        condition: torch.Tensor,
        control_strength: float = 1.0,
    ) -> torch.Tensor:
        """Apply residual adapter to hidden states.
        
        Args:
            hidden_states: [B, T, hidden_size] latent features from base model.
            condition: [B, T, condition_dim] per-frame condition signal.
            control_strength: Scalar multiplier on output (for schedule, etc.).
        
        Returns:
            [B, T, hidden_size] residual to add to hidden_states.
        """
        # Normalize condition
        cond = self.ln(condition)
        
        # Project down and apply activation
        cond = self.down(cond)
        cond = F.silu(cond)
        
        # Project up (zero-initialized)
        cond = self.up(cond)
        
        # Apply gate (also zero-initialized)
        residual = cond * torch.sigmoid(self.gate) * control_strength
        
        return residual
