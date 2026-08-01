"""
Sidecar Schema for Phase D2 (phase_d2_sidecar_v1).

Defines the expected structure and validation for per-frame dense condition dictionaries.
Each field is per-frame (shape [n_frames] or [n_frames, n_values]).
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Set
import math

import torch
import numpy as np


# Required field names per sidecar schema v1
REQUIRED_FIELDS: Set[str] = {
    "absolute_time",            # seconds, [n_frames]
    "normalized_song_time",     # fraction of total song [0, 1], [n_frames]
    "beat_relative_time",       # seconds offset from beat, [n_frames]
    "bar_position",             # beat number within bar [0, beats_per_bar), [n_frames]
    "section_id",               # int, [n_frames]
    "phrase_id",                # int, [n_frames]
    "word_id",                  # int, [n_frames]
    "phoneme_id",               # int, [n_frames]
    "pos_in_phrase",            # 0..n-1, [n_frames]
    "pos_in_word",              # 0..n-1, [n_frames]
    "pos_in_phoneme",           # 0..n-1, [n_frames]
    "word_start_pulse",         # beat position of word start, [n_frames]
    "word_end_pulse",           # beat position of word end, [n_frames]
    "phrase_start_pulse",       # beat position of phrase start, [n_frames]
    "phrase_end_pulse",         # beat position of phrase end, [n_frames]
    "vowel_nucleus",            # bool or 0/1, [n_frames]
    "target_phrase_duration",   # seconds, [n_frames]
    "target_word_duration",     # seconds, [n_frames]
    "target_phoneme_duration",  # seconds, [n_frames]
    "target_f0",                # Hz or normalized, [n_frames]
    "target_note",              # MIDI number or normalized, [n_frames]
    "note_onset",               # bool or 0/1, [n_frames]
    "note_end",                 # bool or 0/1, [n_frames]
    "voiced",                   # bool or 0/1, [n_frames]
    "energy",                   # normalized amplitude, [n_frames]
    "breath",                   # bool or 0/1, [n_frames]
    "pause",                    # bool or 0/1, [n_frames]
    "vocal_active",             # bool or 0/1, [n_frames]
    "local_tempo",              # BPM or ratio, [n_frames]
    "rubato_offset",            # seconds, [n_frames]
}

OPTIONAL_FIELDS: Set[str] = {
    "metadata",                 # arbitrary dict for debugging
    "frame_rate",               # Hz (informational)
}


def validate_sidecar(
    sidecar_dict: Dict[str, Any],
    n_frames: Optional[int] = None,
    schema_version: str = "phase_d2_sidecar_v1",
) -> None:
    """Validate a sidecar dictionary against the schema.
    
    Checks:
    - All required fields present.
    - Schema version matches.
    - Each field has shape [n_frames] or [n_frames, ...].
    - No NaN or inf values.
    - No type mismatches.
    
    Args:
        sidecar_dict: Dictionary with per-frame fields.
        n_frames: Expected frame count. If None, inferred from first field.
        schema_version: Expected version string (default "phase_d2_sidecar_v1").
    
    Raises:
        ValueError: If any validation check fails (fail-closed design).
        KeyError: If required fields are missing.
    """
    
    # Check schema version
    declared_version = sidecar_dict.get("_schema_version", schema_version)
    if declared_version != schema_version:
        raise ValueError(
            f"Schema version mismatch: expected '{schema_version}', "
            f"got '{declared_version}'."
        )
    
    # Infer n_frames from first required field
    if n_frames is None:
        for field_name in REQUIRED_FIELDS:
            if field_name in sidecar_dict:
                val = sidecar_dict[field_name]
                if isinstance(val, torch.Tensor):
                    n_frames = val.shape[0]
                elif isinstance(val, (list, tuple)):
                    n_frames = len(val)
                elif isinstance(val, np.ndarray):
                    n_frames = val.shape[0]
                else:
                    raise ValueError(f"Field '{field_name}' has unsupported type: {type(val)}")
                break
    
    if n_frames is None or n_frames <= 0:
        raise ValueError(f"Could not infer n_frames; must be > 0. Got {n_frames}.")
    
    # Check all required fields present
    missing = REQUIRED_FIELDS - set(sidecar_dict.keys())
    if missing:
        raise KeyError(
            f"Missing required fields: {sorted(missing)}"
        )
    
    # Validate each required field
    for field_name in REQUIRED_FIELDS:
        _validate_field(sidecar_dict[field_name], field_name, n_frames)
    
    # Validate optional fields if present
    for field_name in OPTIONAL_FIELDS:
        if field_name in sidecar_dict:
            _validate_field(sidecar_dict[field_name], field_name, n_frames, optional=True)


def _validate_field(
    field_value: Any,
    field_name: str,
    n_frames: int,
    optional: bool = False,
) -> None:
    """Validate a single field.
    
    Args:
        field_value: The field value (Tensor, list, array, etc.).
        field_name: Name for error messages.
        n_frames: Expected first dimension.
        optional: If True, skip non-fatal checks.
    
    Raises:
        ValueError: If field is malformed.
    """
    
    # Skip metadata (dict)
    if field_name == "metadata":
        return
    
    # Convert to tensor if needed
    if isinstance(field_value, torch.Tensor):
        tensor = field_value
    elif isinstance(field_value, (list, tuple)):
        try:
            tensor = torch.tensor(field_value, dtype=torch.float32)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Field '{field_name}' cannot be converted to tensor: {e}"
            ) from e
    elif isinstance(field_value, np.ndarray):
        # Convert numpy array to tensor
        try:
            tensor = torch.from_numpy(field_value.astype(np.float32))
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Field '{field_name}' cannot convert numpy array to tensor: {e}"
            ) from e
    else:
        raise ValueError(
            f"Field '{field_name}' must be Tensor, list, tuple, or numpy array, "
            f"got {type(field_value)}"
        )
    
    # Check shape
    if tensor.shape[0] != n_frames:
        raise ValueError(
            f"Field '{field_name}' has shape {tensor.shape}, "
            f"but expected first dim {n_frames}."
        )
    
    # Check for NaN and inf
    if torch.isnan(tensor).any():
        raise ValueError(f"Field '{field_name}' contains NaN values.")
    if torch.isinf(tensor).any():
        raise ValueError(f"Field '{field_name}' contains inf values.")
