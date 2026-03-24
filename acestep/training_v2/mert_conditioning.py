"""
Utilities for optional precomputed MERT voice conditioning.

This module supports two stages:

1. Offline extraction of frozen MERT hidden states from a short reference clip.
2. Lightweight trainable aggregation/projection of those precomputed features
   into ACE-Step decoder conditioning space.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
from transformers import AutoModel, Wav2Vec2FeatureExtractor


def load_audio_clip_mono_16k(
    audio_path: str | Path,
    start_sec: float = 0.0,
    duration_sec: float = 3.0,
    target_sample_rate: int = 16000,
) -> torch.Tensor:
    """Load a mono clip from disk and resample to the requested rate."""
    try:
        wav, sr = torchaudio.load(str(audio_path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
    except Exception:
        wav_np, sr = sf.read(str(audio_path), always_2d=False)
        wav = torch.as_tensor(wav_np, dtype=torch.float32)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        elif wav.ndim == 2:
            wav = wav.transpose(0, 1)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
        else:
            raise ValueError(f"Unsupported audio shape from soundfile for {audio_path}: {wav_np.shape}")

    start_frame = max(0, int(round(float(start_sec) * sr)))
    end_frame = wav.shape[-1]
    if duration_sec > 0:
        end_frame = min(end_frame, start_frame + int(round(float(duration_sec) * sr)))
    wav = wav[:, start_frame:end_frame]

    if wav.numel() == 0:
        wav = torch.zeros(1, max(1, int(round(float(duration_sec) * 16000.0))), dtype=torch.float32)

    if sr != target_sample_rate:
        wav = torchaudio.functional.resample(wav, sr, target_sample_rate)

    return wav.squeeze(0).float()


@dataclass
class FrozenMERTExtractorConfig:
    model_name_or_path: str = "m-a-p/MERT-v1-330M"
    local_files_only: bool = True
    trust_remote_code: bool = True
    device: str = "cpu"
    max_duration_sec: float = 3.0


class FrozenMERTExtractor:
    """Frozen feature extractor for offline MERT hidden-state export."""

    def __init__(self, config: FrozenMERTExtractorConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            config.model_name_or_path,
            local_files_only=config.local_files_only,
            trust_remote_code=config.trust_remote_code,
        )
        self.model = AutoModel.from_pretrained(
            config.model_name_or_path,
            local_files_only=config.local_files_only,
            trust_remote_code=config.trust_remote_code,
        )
        self.model.eval()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extract(
        self,
        audio_path: str | Path,
        start_sec: float = 0.0,
        duration_sec: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return stacked hidden states and a time mask.

        Returns:
            hidden_states: [layers, time, hidden]
            attention_mask: [time]
        """
        clip = load_audio_clip_mono_16k(
            audio_path=audio_path,
            start_sec=start_sec,
            duration_sec=duration_sec or self.config.max_duration_sec,
            target_sample_rate=int(getattr(self.feature_extractor, "sampling_rate", 16000)),
        )
        features = self.feature_extractor(
            clip.cpu().numpy(),
            sampling_rate=int(getattr(self.feature_extractor, "sampling_rate", 16000)),
            return_tensors="pt",
        )
        features = {k: v.to(self.device) for k, v in features.items()}
        outputs = self.model(**features, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if not hidden_states:
            raise RuntimeError("MERT returned no hidden states")

        stacked = torch.stack([hs.squeeze(0) for hs in hidden_states], dim=0)
        stacked = stacked.detach().cpu().to(torch.float16)
        time_mask = torch.ones(stacked.shape[1], dtype=torch.float16)
        return stacked, time_mask


class FrozenMERTFeatureExtractor:
    """Backward-compatible wrapper used by preprocessing."""

    def __init__(
        self,
        model_name_or_path: str = "m-a-p/MERT-v1-330M",
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
        local_files_only: bool = True,
        trust_remote_code: bool = True,
        max_duration_sec: float = 3.0,
    ) -> None:
        del dtype  # kept for API compatibility
        self._extractor = FrozenMERTExtractor(
            FrozenMERTExtractorConfig(
                model_name_or_path=model_name_or_path,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
                device=device,
                max_duration_sec=max_duration_sec,
            )
        )

    def extract_from_file(
        self,
        audio_path: str | Path,
        *,
        start_sec: float = 0.0,
        duration_sec: float = 3.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._extractor.extract(
            audio_path=audio_path,
            start_sec=start_sec,
            duration_sec=duration_sec,
        )


class PrecomputedMERTConditioner(nn.Module):
    """Trainable aggregation/projection head for precomputed MERT features."""

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 2048,
        num_layers: int = 25,
        hidden_size: Optional[int] = None,
        dropout: float = 0.1,
        scale: float = 1.0,
        use_layer_aggregation: bool = True,
    ) -> None:
        super().__init__()
        resolved_hidden = int(hidden_size or input_dim)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_layers = int(num_layers)
        self.hidden_size = resolved_hidden
        self.scale = float(scale)
        self.use_layer_aggregation = bool(use_layer_aggregation)
        self.dropout = nn.Dropout(float(dropout))
        if self.use_layer_aggregation:
            self.layer_logits = nn.Parameter(torch.zeros(self.num_layers, dtype=torch.float32))
        else:
            self.register_parameter("layer_logits", None)
        self.proj = nn.Linear(self.hidden_size, self.output_dim, bias=True)

    def forward(
        self,
        ref_voice_features: torch.Tensor,
        ref_voice_attention_mask: torch.Tensor,
        scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project precomputed MERT features into ACE conditioning space.

        Args:
            ref_voice_features: [B, layers, T, H] or [B, T, H]
            ref_voice_attention_mask: [B, T]
        """
        if ref_voice_features.ndim not in (3, 4):
            raise ValueError(
                "Expected ref_voice_features with shape [B, L, T, H] or [B, T, H], "
                f"got {tuple(ref_voice_features.shape)}"
            )
        if ref_voice_features.ndim == 4:
            _bsz, layers, _time, hidden = ref_voice_features.shape
            if layers != self.num_layers:
                raise ValueError(f"Expected {self.num_layers} MERT layers, got {layers}")
            if hidden != self.hidden_size:
                raise ValueError(f"Expected hidden size {self.hidden_size}, got {hidden}")
            if self.use_layer_aggregation and self.layer_logits is not None:
                weights = torch.softmax(self.layer_logits.to(ref_voice_features.device), dim=0)
                weights = weights.view(1, self.num_layers, 1, 1).to(ref_voice_features.dtype)
                merged = (ref_voice_features * weights).sum(dim=1)
            else:
                merged = ref_voice_features.mean(dim=1)
        else:
            _bsz, _time, hidden = ref_voice_features.shape
            if hidden != self.hidden_size:
                raise ValueError(f"Expected hidden size {self.hidden_size}, got {hidden}")
            merged = ref_voice_features

        conditioned = self.proj(merged)
        conditioned = self.dropout(conditioned)
        applied_scale = self.scale if scale is None else float(scale)
        conditioned = conditioned * applied_scale
        return conditioned, ref_voice_attention_mask

    def export_config(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "dropout": float(self.dropout.p),
            "scale": self.scale,
            "use_layer_aggregation": self.use_layer_aggregation,
        }


def load_saved_bridge(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(str(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"Invalid bridge checkpoint payload: {path}")
    return payload
