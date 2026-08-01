"""
Phase D2 Configuration.

Defines PhaseD2Config dataclass for dense per-frame prosody control,
with condition group toggles, architecture hyperparameters, and schema versioning.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any
import json


@dataclass
class PhaseD2Config:
    """Configuration for Phase D2 prosody controller.
    
    Specifies condition groups (linguistic, timing, prosody), adapter architecture,
    and control schedule behavior.
    
    Attributes:
        latent_frame_rate_hz: Frame rate of base model latent sequence (default 25.0 Hz).
        hidden_size: Frozen DiT hidden dimension.
        condition_hidden_size: Width of the lightweight temporal encoder.
        adapter_bottleneck_size: Per-layer residual adapter bottleneck width.
        num_adapter_layers: Number of transformer layers to inject adapters (default 6).
        injection_layer_indices: Explicit layer indices for adapter injection, if provided.
            If None, defaults to first num_adapter_layers layers.
        condition_dim: Computed total dimension of enabled condition groups.
            Updated automatically by compute_condition_dim().
        zero_init: If True, adapter weights initialized to zero for null-condition stability.
        control_schedule: "constant" or "early_decay" (cosine decay after 50% of training).
        schema_version: Sidecar format version; must be "phase_d2_sidecar_v1".
        
        Condition groups (all booleans, default all True):
            linguistic: Word/phoneme IDs and linguistic features.
            absolute_timing: Absolute time in seconds.
            beat_relative_timing: Beat-phase and beat-relative features.
            melody_prosody: Target F0, note onset/offset, vowel nucleus.
            breath_energy: Breath and energy markers.
            vocal_activity: Voice activity and pause indicators.
            structure: Section/phrase/word boundary IDs.
            explicit_duration: Phrase, word, phoneme, and phoneme/note duration ratio.
    """
    
    # Frame rate and dimensions
    latent_frame_rate_hz: float = 25.0
    hidden_size: int = 256
    condition_hidden_size: int = 256
    adapter_bottleneck_size: int = 128
    num_adapter_layers: int = 6
    injection_layer_indices: List[int] | None = None
    
    # Condition composition
    condition_dim: int = 0  # Computed by compute_condition_dim()
    zero_init: bool = True
    control_schedule: str = "constant"
    schema_version: str = "phase_d2_sidecar_v1"
    
    # Condition groups (toggles)
    linguistic: bool = True
    absolute_timing: bool = True
    beat_relative_timing: bool = True
    melody_prosody: bool = True
    breath_energy: bool = True
    vocal_activity: bool = True
    structure: bool = True
    explicit_duration: bool = True
    
    def __post_init__(self) -> None:
        """Validate and compute derived fields."""
        if self.schema_version != "phase_d2_sidecar_v1":
            raise ValueError(
                f"Unsupported schema_version: {self.schema_version}. "
                "Must be 'phase_d2_sidecar_v1'."
            )
        if self.control_schedule not in ("constant", "early_decay"):
            raise ValueError(
                f"control_schedule must be 'constant' or 'early_decay', "
                f"got {self.control_schedule}."
            )
        if self.latent_frame_rate_hz <= 0:
            raise ValueError(f"latent_frame_rate_hz must be > 0, got {self.latent_frame_rate_hz}.")
        if min(self.hidden_size, self.condition_hidden_size, self.adapter_bottleneck_size) <= 0:
            raise ValueError("D2 model, condition, and adapter widths must all be positive.")
        
        self.compute_condition_dim()
    
    def compute_condition_dim(self) -> None:
        """Compute total condition dimension from enabled groups.
        
        Group dimensions (hardcoded per schema v1):
            linguistic: 4 (word_id, phoneme_id, pos_in_word, pos_in_phrase)
            absolute_timing: 2 (absolute_time, normalized_song_time)
            beat_relative_timing: 3 (beat_relative_time, bar_position, local_tempo)
            melody_prosody: 5 (target_f0, target_note, note_onset, note_end, vowel_nucleus)
            breath_energy: 2 (energy, breath)
            vocal_activity: 2 (voiced, vocal_active)
            structure: 4 (section_id, phrase_id, word_id, pause)
            explicit_duration: 4 (phrase, word, phoneme, phoneme/note ratio)
        """
        dim = 0
        if self.linguistic:
            dim += 4
        if self.absolute_timing:
            dim += 2
        if self.beat_relative_timing:
            dim += 3
        if self.melody_prosody:
            dim += 5
        if self.breath_energy:
            dim += 2
        if self.vocal_activity:
            dim += 2
        if self.structure:
            dim += 4
        if self.explicit_duration:
            dim += 4
        
        self.condition_dim = dim
    
    def get_condition_group_slices(self) -> Dict[str, tuple[int, int]]:
        """Return {group_name: (start_idx, end_idx)} for condition tensor assembly.
        
        Raises ValueError if no groups are enabled.
        """
        if self.condition_dim == 0:
            raise ValueError("No condition groups enabled; condition_dim is 0.")
        
        slices: Dict[str, tuple[int, int]] = {}
        idx = 0
        
        if self.linguistic:
            slices["linguistic"] = (idx, idx + 4)
            idx += 4
        if self.absolute_timing:
            slices["absolute_timing"] = (idx, idx + 2)
            idx += 2
        if self.beat_relative_timing:
            slices["beat_relative_timing"] = (idx, idx + 3)
            idx += 3
        if self.melody_prosody:
            slices["melody_prosody"] = (idx, idx + 5)
            idx += 5
        if self.breath_energy:
            slices["breath_energy"] = (idx, idx + 2)
            idx += 2
        if self.vocal_activity:
            slices["vocal_activity"] = (idx, idx + 2)
            idx += 2
        if self.structure:
            slices["structure"] = (idx, idx + 4)
            idx += 4
        if self.explicit_duration:
            slices["explicit_duration"] = (idx, idx + 4)
            idx += 4
        
        return slices
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PhaseD2Config:
        """Create from dictionary (e.g., loaded from JSON).
        
        Args:
            data: Dictionary with config fields.
        
        Returns:
            PhaseD2Config instance.
        
        Raises:
            TypeError: If required fields are missing.
        """
        try:
            return cls(**data)
        except TypeError as e:
            raise TypeError(f"Invalid config dict: {e}") from e
