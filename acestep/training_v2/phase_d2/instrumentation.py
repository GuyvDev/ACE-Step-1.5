"""
Instrumentation for Phase D2.

Recording and analysis of adapter activations for debugging and ablation studies.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Callable, Any
from contextlib import contextmanager
import torch
import torch.nn as nn


class ActivationRecorder:
    """Context manager to record adapter activations during forward pass.
    
    Captures:
    - Adapter residual norms per layer
    - Gate values per layer
    - Overall activation statistics
    """
    
    def __init__(self):
        """Initialize recorder."""
        self.records: Dict[str, Any] = {
            "residual_norms": {},     # {layer_idx: [B, T] norms}
            "gate_values": {},        # {layer_idx: [B, T] sigmoid(gate) values}
            "output_norms": {},       # {layer_idx: [B, T] norms}
            "max_residual": None,
            "mean_residual": None,
        }
        self._hooks: List[Callable] = []
    
    @contextmanager
    def recording(self, adapters: nn.ModuleList):
        """Context manager to record adapter activations.
        
        Args:
            adapters: ModuleList of ZeroInitTemporalAdapter instances.
        
        Yields:
            self (for use as `with recorder.recording(adapters) as rec:`)
        """
        hooks = []
        
        try:
            # Register hooks on adapter modules to capture their outputs
            for layer_idx, adapter in enumerate(adapters):
                def make_hook(idx: int) -> Callable:
                    def hook(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
                        # output is [B, T, H] residual
                        residual_norm = torch.norm(output, p=2, dim=-1)  # [B, T]
                        self.records["residual_norms"][idx] = residual_norm.detach()
                        
                        # Gate value
                        if hasattr(module, "gate"):
                            gate_val = torch.sigmoid(module.gate).item()
                            self.records["gate_values"][idx] = gate_val
                    
                    return hook
                
                hook_fn = make_hook(layer_idx)
                handle = adapter.register_forward_hook(hook_fn)
                hooks.append(handle)
            
            yield self
        
        finally:
            # Cleanup hooks
            for handle in hooks:
                handle.remove()
    
    def summarize(self) -> Dict[str, float]:
        """Summarize recorded activations.
        
        Returns:
            Dictionary with scalar statistics.
        """
        summary = {}
        
        # Residual norms statistics
        if self.records["residual_norms"]:
            all_norms = torch.cat([
                norm.flatten() for norm in self.records["residual_norms"].values()
            ])
            summary["mean_residual_norm"] = all_norms.mean().item()
            summary["max_residual_norm"] = all_norms.max().item()
            summary["std_residual_norm"] = all_norms.std().item()
        
        # Gate statistics
        if self.records["gate_values"]:
            gate_vals = list(self.records["gate_values"].values())
            summary["mean_gate"] = sum(gate_vals) / len(gate_vals)
            summary["max_gate"] = max(gate_vals)
        
        return summary
