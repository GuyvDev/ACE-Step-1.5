"""
Model Integration for Phase D2.

PhaseD2Injector attaches adapters to frozen base model layers via forward hooks,
ensuring null-route stability and adapter-only gradient flow.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional
import torch
import torch.nn as nn

from acestep.training_v2.phase_d_timing_v2.config import PhaseD2Config
from acestep.training_v2.phase_d_timing_v2.residual_hook import build_temporal_residual_hook
from acestep.training_v2.phase_d_timing_v2.temporal_encoder import TemporalConditionEncoder
from acestep.training_v2.phase_d_timing_v2.residual_adapter import ZeroInitTemporalAdapter


class PhaseD2Injector(nn.Module):
    """Inject Phase D2 adapters into frozen base model layers.

    Design:
    - Takes a list of transformer layers (duck-typed nn.Module).
    - Attaches adapters to the first N layers via register_forward_hook.
    - Wraps layer outputs additively: output -> output + adapter(condition).
    - Null condition routes exactly to zero (fail-closed).
    - Base model parameters are never modified.

    Attributes:
        config: PhaseD2Config.
        layers: Reference list of transformer layers.
        encoder: TemporalConditionEncoder for condition processing.
        adapters: List of ZeroInitTemporalAdapter, one per injected layer.
    """

    def __init__(
        self,
        transformer_layers: List[nn.Module],
        config: PhaseD2Config,
    ):
        """Initialize injector.

        Args:
            transformer_layers: List of nn.Module transformer layers from base model.
            config: PhaseD2Config specifying injection points and architecture.

        Raises:
            ValueError: If num_adapter_layers exceeds number of layers.
        """
        super().__init__()

        self.config = config
        # Keep frozen C25 layers as non-owned references. Registering them as a
        # child module makes injector.parameters() include the entire DiT and
        # breaks optimizer/checkpoint ownership audits.
        object.__setattr__(self, "_base_layers", list(transformer_layers))

        # Determine injection layer indices
        if config.injection_layer_indices is not None:
            injection_indices = config.injection_layer_indices
        else:
            n_inject = min(config.num_adapter_layers, len(transformer_layers))
            injection_indices = list(range(n_inject))

        if not injection_indices:
            raise ValueError("At least one injection layer is required.")
        if min(injection_indices) < 0 or max(injection_indices) >= len(transformer_layers):
            raise ValueError(
                f"Injection layer index {max(injection_indices)} exceeds "
                f"number of layers {len(transformer_layers)}."
            )

        self.injection_indices = injection_indices

        # Create condition encoder
        self.encoder = TemporalConditionEncoder(
            condition_dim=config.condition_dim,
            hidden_size=config.condition_hidden_size,
        )

        # Create adapters for each injection point
        self.adapters = nn.ModuleList([
            ZeroInitTemporalAdapter(
                condition_dim=config.condition_hidden_size,
                hidden_size=config.hidden_size,
                adapter_dim=config.adapter_bottleneck_size,
            )
            for _ in injection_indices
        ])

        # State: current condition and control strength
        self._current_condition: Optional[torch.Tensor] = None
        self._current_control_strength: float = 1.0
        self._current_vocal_mask: Optional[torch.Tensor] = None

        # Registered hooks (stored for cleanup)
        self._hooks: Dict[int, Callable] = {}

    def set_condition(
        self,
        condition_tensor: torch.Tensor,
        vocal_mask: Optional[torch.Tensor] = None,
        control_strength: float = 1.0,
    ) -> None:
        """Set active condition for injection.

        Args:
            condition_tensor: [B, T, condition_dim] condition tensor (validated).
            vocal_mask: Optional [B, T] bool mask (True = vocal, False = silence).
            control_strength: Scalar multiplier on adapter outputs (0.0 to 1.0).

        Raises:
            ValueError: If condition_tensor shape is invalid.
        """
        if condition_tensor.dim() != 3:
            raise ValueError(
                f"condition_tensor must be 3D [B, T, C], "
                f"got shape {condition_tensor.shape}."
            )

        if condition_tensor.shape[-1] != self.config.condition_dim:
            raise ValueError(
                f"condition_tensor last dim {condition_tensor.shape[-1]} "
                f"does not match config.condition_dim {self.config.condition_dim}."
            )

        self._current_condition = condition_tensor
        self._current_control_strength = float(control_strength)
        self._current_vocal_mask = vocal_mask

    def clear_condition(self) -> None:
        """Clear active condition (null route)."""
        self._current_condition = None
        self._current_control_strength = 1.0
        self._current_vocal_mask = None

    def attach(self) -> None:
        """Register forward hooks on injection layers.

        Each hook wraps the layer output by adding adapter residuals.
        Must call detach() before re-attaching.
        """
        if self._hooks:
            raise RuntimeError("PhaseD2Injector is already attached; detach before re-attaching.")

        for layer_idx, adapter_idx in enumerate(self.injection_indices):
            layer = self._base_layers[adapter_idx]
            adapter = self.adapters[layer_idx]

            hook_fn = build_temporal_residual_hook(self, adapter)
            hook_handle = layer.register_forward_hook(hook_fn)
            self._hooks[adapter_idx] = hook_handle

    def detach(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hooks.values():
            handle.remove()
        self._hooks.clear()

    def get_adapter_parameters(self) -> List[torch.nn.Parameter]:
        """Return list of only adapter parameters (for optimization)."""
        params = []
        params.extend(self.encoder.parameters())
        for adapter in self.adapters:
            params.extend(adapter.parameters())
        return params
