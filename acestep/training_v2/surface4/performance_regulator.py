"""Symbolic frame-aligned residual regulator for Control Surface 4."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class PerformanceRegulator(nn.Module):
    """Map phone identities and eight frame features to a 64-D residual."""

    def __init__(
        self,
        phone_vocab_size: int = 128,
        hidden_size: int = 128,
        feature_dim: int = 8,
        context_dim: int = 64,
    ) -> None:
        """Build the fixed Surface 4 architecture with an exact-zero output."""
        super().__init__()
        self.phone_embedding = nn.Embedding(phone_vocab_size, hidden_size)
        self.feature_projection = nn.Linear(feature_dim, hidden_size)
        self.normalization = nn.LayerNorm(hidden_size)
        self.temporal = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=3,
            padding=1,
            groups=hidden_size,
        )
        self.mixer = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, context_dim, bias=False)
        nn.init.zeros_(self.output_projection.weight)
        self._active_plan: dict[str, torch.Tensor] | None = None
        self._inference_residual_scale = 1.0

    @property
    def trainable_parameter_count(self) -> int:
        """Return the number of parameters in the regulator."""
        return sum(parameter.numel() for parameter in self.parameters())

    def set_plan(self, plan: dict[str, Any]) -> None:
        """Validate and activate one production-shaped dense 25 Hz plan."""
        required = {"phone_ids", "phone_weights", "features", "valid_mask"}
        missing = sorted(required - set(plan))
        if missing:
            raise ValueError(f"performance plan missing keys: {missing}")
        phone_ids = torch.as_tensor(plan["phone_ids"], dtype=torch.long)
        phone_weights = torch.as_tensor(plan["phone_weights"], dtype=torch.float32)
        features = torch.as_tensor(plan["features"], dtype=torch.float32)
        valid_mask = torch.as_tensor(plan["valid_mask"], dtype=torch.float32)
        if phone_ids.ndim == 2:
            phone_ids = phone_ids.unsqueeze(0)
            phone_weights = phone_weights.unsqueeze(0)
            features = features.unsqueeze(0)
            valid_mask = valid_mask.unsqueeze(0)
        if phone_ids.ndim != 3 or phone_weights.shape != phone_ids.shape:
            raise ValueError("phone ids/weights must have matching [B, T, K] shapes")
        if features.shape != (*phone_ids.shape[:2], self.feature_projection.in_features):
            raise ValueError("features must have shape [B, T, 8]")
        if valid_mask.shape != phone_ids.shape[:2]:
            raise ValueError("valid_mask must have shape [B, T]")
        if not bool(torch.isfinite(phone_weights).all()) or bool((phone_weights < 0).any()):
            raise ValueError("phone weights must be finite and non-negative")
        weight_sums = phone_weights.sum(dim=-1)
        if not torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6):
            raise ValueError("phone weights must sum to one on every frame")
        self._active_plan = {
            "phone_ids": phone_ids,
            "phone_weights": phone_weights,
            "features": features,
            "valid_mask": valid_mask,
        }

    def clear_plan(self) -> None:
        """Disable the regulator without changing the context tensor."""
        self._active_plan = None


    
    @property
    def inference_residual_scale(self) -> float:
        """Return the non-trainable inference-only residual multiplier."""
        return self._inference_residual_scale

    def set_inference_residual_scale(self, scale: float) -> None:
        """Set a finite inference-only multiplier without adding model state."""
        value = float(scale)
        if not torch.isfinite(torch.tensor(value)) or not 0.0 <= value <= 1.0:
            raise ValueError("Surface 4 residual scale must be finite within [0, 1]")
        self._inference_residual_scale = value

    def forward(
        self,
        phone_ids: torch.Tensor,
        phone_weights: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Return a frame-aligned residual from a dense performance plan."""
        device = self.output_projection.weight.device
        phone_ids = phone_ids.to(device=device)
        weights = phone_weights.to(device=device, dtype=self.phone_embedding.weight.dtype)
        features = features.to(device=device, dtype=self.feature_projection.weight.dtype)
        phone_hidden = (self.phone_embedding(phone_ids) * weights.unsqueeze(-1)).sum(dim=-2)
        hidden = self.normalization(phone_hidden + self.feature_projection(features))
        hidden = F.silu(self.temporal(hidden.transpose(1, 2)).transpose(1, 2))
        hidden = F.silu(self.mixer(hidden))
        return self.output_projection(hidden)

    def apply_to_context(self, context_latents: torch.Tensor) -> torch.Tensor:
        """Add the active residual to only the existing first 64 context channels."""
        if self._active_plan is None or self._inference_residual_scale == 0.0:
            return context_latents
        plan = self._active_plan
        if plan["phone_ids"].shape[1] != context_latents.shape[1]:
            raise RuntimeError(
                "performance plan/context frame mismatch: "
                f"{plan['phone_ids'].shape[1]} != {context_latents.shape[1]}"
            )
        batch = context_latents.shape[0]
        plan_batch = plan["phone_ids"].shape[0]
        if plan_batch not in (1, batch):
            raise RuntimeError(f"performance plan batch {plan_batch} cannot serve batch {batch}")
        tensors = {
            key: value.expand(batch, *value.shape[1:]) if plan_batch == 1 else value
            for key, value in plan.items()
        }
        residual = self(
            tensors["phone_ids"],
            tensors["phone_weights"],
            tensors["features"],
        ).to(device=context_latents.device, dtype=context_latents.dtype)
        residual = residual * tensors["valid_mask"].to(
            device=context_latents.device,
            dtype=context_latents.dtype,
        ).unsqueeze(-1)
        if self._inference_residual_scale != 1.0:
            residual = residual * self._inference_residual_scale
        if residual.shape[-1] != 64 or context_latents.shape[-1] < 64:
            raise RuntimeError("Surface 4 requires exactly 64 acoustic-context channels")
        regulated = context_latents[..., :64] + residual
        return torch.cat([regulated, context_latents[..., 64:]], dim=-1)
