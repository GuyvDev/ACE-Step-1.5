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





def _frame_rms(x: torch.Tensor, sample_rate: int, win_sec: float = 0.20, hop_sec: float = 0.10) -> tuple[torch.Tensor, torch.Tensor]:
    win = max(1, int(round(sample_rate * win_sec)))
    hop = max(1, int(round(sample_rate * hop_sec)))
    if x.numel() < win:
        x = torch.nn.functional.pad(x, (0, win - x.numel()))
    frames = x.unfold(0, win, hop)
    rms = torch.sqrt(torch.mean(frames.float() ** 2, dim=-1) + 1e-12)
    times = torch.arange(frames.shape[0], dtype=torch.float32) * hop_sec
    return rms, times


def _estimate_frame_f0(x: torch.Tensor, sample_rate: int, win_sec: float = 0.08, hop_sec: float = 0.04) -> tuple[torch.Tensor, torch.Tensor]:
    win = max(1, int(round(sample_rate * win_sec)))
    hop = max(1, int(round(sample_rate * hop_sec)))
    if x.numel() < win:
        x = torch.nn.functional.pad(x, (0, win - x.numel()))
    frames = x.unfold(0, win, hop).float()
    min_lag = max(1, int(sample_rate / 450.0))
    max_lag = max(min_lag + 1, int(sample_rate / 70.0))
    f0 = torch.full((frames.shape[0],), float("nan"), dtype=torch.float32)
    for idx, fr in enumerate(frames):
        if torch.sqrt(torch.mean(fr * fr) + 1e-12) < 0.008:
            continue
        fr = fr - fr.mean()
        ac = torch.nn.functional.conv1d(
            fr.view(1, 1, -1),
            fr.flip(0).view(1, 1, -1),
            padding=fr.numel() - 1,
        ).view(-1)[fr.numel() - 1 :]
        if ac.numel() <= max_lag or float(ac[0]) <= 1e-9:
            continue
        ac = ac / (ac[0] + 1e-12)
        seg = ac[min_lag:max_lag]
        if seg.numel() == 0:
            continue
        lag = int(torch.argmax(seg).item()) + min_lag
        if float(ac[lag]) >= 0.25:
            f0[idx] = float(sample_rate / lag)
    times = torch.arange(frames.shape[0], dtype=torch.float32) * hop_sec
    return f0, times


def select_reference_crop_starts(
    audio_path: str | Path,
    *,
    crop_duration_sec: float = 3.0,
    top_k: int = 3,
    base_start_sec: float = 0.0,
) -> list[dict[str, float | str]]:
    """Select deterministic V3 reference crops from existing audio only."""
    top_k = max(1, int(top_k))
    crop_duration_sec = max(0.25, float(crop_duration_sec))
    wav = load_audio_clip_mono_16k(audio_path, start_sec=0.0, duration_sec=0.0)
    sr = 16000
    total_sec = max(crop_duration_sec, float(wav.numel()) / sr)
    latest = max(0.0, total_sec - crop_duration_sec)
    base = min(max(0.0, float(base_start_sec)), latest)

    def clamp_start(v: float) -> float:
        return float(min(max(0.0, v), latest))

    rms, rms_times = _frame_rms(wav, sr)
    f0, f0_times = _estimate_frame_f0(wav, sr)
    voiced = torch.isfinite(f0)
    starts: list[dict[str, float | str]] = [
        {"label": "phrase_beginning", "start_sec": clamp_start(base)},
        {"label": "phrase_ending", "start_sec": clamp_start(latest)},
    ]
    if rms.numel() > 0:
        sustained_idx = int(torch.argmax(rms).item())
        starts.append({
            "label": "sustained_vowel",
            "start_sec": clamp_start(float(rms_times[sustained_idx]) - crop_duration_sec / 2.0),
        })
    if bool(voiced.any()):
        finite_f0 = f0[voiced]
        finite_times = f0_times[voiced]
        starts.append({
            "label": "low_register",
            "start_sec": clamp_start(float(finite_times[int(torch.argmin(finite_f0).item())]) - crop_duration_sec / 2.0),
        })
        starts.append({
            "label": "high_register",
            "start_sec": clamp_start(float(finite_times[int(torch.argmax(finite_f0).item())]) - crop_duration_sec / 2.0),
        })

    deduped: list[dict[str, float | str]] = []
    seen: set[int] = set()
    for item in starts:
        bucket = int(round(float(item["start_sec"]) * 10.0))
        if bucket in seen:
            continue
        seen.add(bucket)
        deduped.append(item)
        if len(deduped) >= top_k:
            break
    while len(deduped) < top_k:
        frac = len(deduped) / max(top_k - 1, 1)
        deduped.append({"label": f"fallback_{len(deduped)}", "start_sec": clamp_start(frac * latest)})
    return deduped[:top_k]


def pad_and_stack_mert_crops(crops: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    if not crops:
        raise ValueError("No MERT crops to stack")
    max_t = max(int(features.shape[-2]) for features, _mask in crops)
    padded_features = []
    padded_masks = []
    for features, mask in crops:
        if features.shape[-2] < max_t:
            pad_t = max_t - features.shape[-2]
            features = torch.nn.functional.pad(features, (0, 0, 0, pad_t))
            mask = torch.nn.functional.pad(mask, (0, pad_t))
        padded_features.append(features)
        padded_masks.append(mask)
    return torch.stack(padded_features, dim=0), torch.stack(padded_masks, dim=0)


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
        max_reference_crops: int = 5,
        use_crop_attention: bool = True,
    ) -> None:
        super().__init__()
        resolved_hidden = int(hidden_size or input_dim)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_layers = int(num_layers)
        self.hidden_size = resolved_hidden
        self.scale = float(scale)
        self.use_layer_aggregation = bool(use_layer_aggregation)
        self.max_reference_crops = max(1, int(max_reference_crops))
        self.use_crop_attention = bool(use_crop_attention)
        self.dropout = nn.Dropout(float(dropout))
        if self.use_layer_aggregation:
            self.layer_logits = nn.Parameter(torch.zeros(self.num_layers, dtype=torch.float32))
        else:
            self.register_parameter("layer_logits", None)
        if self.use_crop_attention:
            self.crop_logits = nn.Parameter(torch.zeros(self.max_reference_crops, dtype=torch.float32))
        else:
            self.register_parameter("crop_logits", None)
        self.proj = nn.Linear(self.hidden_size, self.output_dim, bias=True)

    def forward(
        self,
        ref_voice_features: torch.Tensor,
        ref_voice_attention_mask: torch.Tensor,
        scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project precomputed MERT features into ACE conditioning space.

        Args:
            ref_voice_features: [B, crops, layers, T, H], [B, layers, T, H] or [B, T, H]
            ref_voice_attention_mask: [B, crops, T] or [B, T]
        """
        if ref_voice_features.ndim not in (3, 4, 5):
            raise ValueError(
                "Expected ref_voice_features with shape [B, C, L, T, H], [B, L, T, H] or [B, T, H], "
                f"got {tuple(ref_voice_features.shape)}"
            )
        if ref_voice_features.ndim == 5:
            _bsz, crops, layers, _time, hidden = ref_voice_features.shape
            if crops > self.max_reference_crops:
                raise ValueError(f"Expected at most {self.max_reference_crops} crops, got {crops}")
            if layers != self.num_layers:
                raise ValueError(f"Expected {self.num_layers} MERT layers, got {layers}")
            if hidden != self.hidden_size:
                raise ValueError(f"Expected hidden size {self.hidden_size}, got {hidden}")
            if self.use_layer_aggregation and self.layer_logits is not None:
                layer_weights = torch.softmax(self.layer_logits.to(ref_voice_features.device), dim=0)
                layer_weights = layer_weights.view(1, 1, self.num_layers, 1, 1).to(ref_voice_features.dtype)
                merged = (ref_voice_features * layer_weights).sum(dim=2)
            else:
                merged = ref_voice_features.mean(dim=2)
            if ref_voice_attention_mask.ndim != 3:
                raise ValueError(
                    "Expected crop-aware ref_voice_attention_mask with shape [B, C, T], "
                    f"got {tuple(ref_voice_attention_mask.shape)}"
                )
            conditioned_crops = self.proj(merged)
            if self.use_crop_attention:
                # V4: expose all crop frames as a reference-token bank. The DiT
                # cross-attention can then select useful timbre fragments per
                # generation state instead of averaging unaligned crop timelines.
                conditioned = conditioned_crops.reshape(
                    conditioned_crops.shape[0],
                    conditioned_crops.shape[1] * conditioned_crops.shape[2],
                    conditioned_crops.shape[3],
                )
                ref_voice_attention_mask = ref_voice_attention_mask.reshape(
                    ref_voice_attention_mask.shape[0],
                    ref_voice_attention_mask.shape[1] * ref_voice_attention_mask.shape[2],
                )
            elif self.crop_logits is not None:
                crop_weights = torch.softmax(self.crop_logits[:crops].to(ref_voice_features.device), dim=0)
                crop_weights = crop_weights.view(1, crops, 1, 1).to(conditioned_crops.dtype)
                conditioned = (conditioned_crops * crop_weights).sum(dim=1)
                ref_voice_attention_mask = ref_voice_attention_mask.amax(dim=1)
            else:
                conditioned = conditioned_crops.mean(dim=1)
                ref_voice_attention_mask = ref_voice_attention_mask.amax(dim=1)
        elif ref_voice_features.ndim == 4:
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

        if ref_voice_features.ndim != 5:
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
            "max_reference_crops": self.max_reference_crops,
            "use_crop_attention": self.use_crop_attention,
        }


def load_saved_bridge(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(str(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"Invalid bridge checkpoint payload: {path}")
    return payload
