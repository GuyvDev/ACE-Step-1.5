"""
Phase D2: Dense Per-Frame Prosody Controller for ACE-Step

A new timing control architecture that replaces Phase D1's sparse event tokens
with a dense per-frame conditioning pipeline. Given a sidecar of per-frame
linguistic, timing, and prosody fields, this module:

1. Validates and builds dense condition tensors (condition_builder.py)
2. Encodes conditions via coarse (phrase-level) and fine (frame-level) pathways
3. Injects learned residual adapters into frozen base model layers
4. Provides fail-closed guarantee: null condition -> zero residual
5. Enables ablation and intervention studies via counterfactuals

Schema: phase_d2_sidecar_v1 (see sidecar_schema.py)
"""

from __future__ import annotations

from acestep.training_v2.phase_d_timing_v2.config import PhaseD2Config
from acestep.training_v2.phase_d_timing_v2.timing_condition import build_timing_condition
from acestep.training_v2.phase_d_timing_v2.sidecar_schema import validate_sidecar

__all__ = [
    "PhaseD2Config",
    "build_timing_condition",
    "validate_sidecar",
]
