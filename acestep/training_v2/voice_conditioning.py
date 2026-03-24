"""
MERT-based reference voice conditioning for ACE-Step LoRA training/inference.

This module keeps the MERT backbone frozen and only trains a lightweight
layer-aggregation + projection head that converts reference voice clips into
the 64-dim acoustic sequence expected by ACE-Step's existing timbre path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torchaudio
from safetensors.torch import load_file, save_file
from transformers import AutoModel, Wav2Vec2FeatureExtractor


def _clamp_clip_window(
    total_sec: float,
    start_sec: float,
    duration_sec: float,
) -> Tuple[float, float]:
    start = max(0.0, float(start_sec))
    duration = max(0.1, float(duration_sec))
    if total_sec <= 0:
        return 0.0, duration
    if start >= total_sec:
        start = max(0.0, total_sec - duration)
    end = min(total_sec, start + duration)
    start = max(0.0, end - duration)
    return start, max(0.1, end - start)


def _resample_if_needed(wav: torch.Tensor, src_sr: int, target_sr: int) -> torch.Tensor:
    if src_sr == target_sr:
        return wav
    return torchaudio.functional.resample(wav, src_sr, target_sr)


def _ensure_1d_mono(wav: torch.Tensor) -> torch.Tensor:
    if wav.dim() == 3 and wav.shape[0] == 1:
        wav = wav.squeeze(0)
    if wav.dim() == 2:
        wav = wav.mean(dim=0)
    if wav.dim() != 1:
        raise ValueError(f"Expected mono waveform, got shape={tuple(wav.shape)}")
    return wav.contiguous()


def _pool_time(hidden: torch.Tensor, max_frames: int) -> torch.Tensor:
    if hidden.shape[1] <= max_frames:
        return hidden
    pooled = torch.nn.functional.adaptive_avg_pool1d(
        hidden.transpose(1, 2),
        max_frames,
    )
    return pooled.transpose(1, 2).contiguous()


class MertReferenceConditioner(nn.Module):
    """Frozen MERT backbone + trainable layer aggregation + projection head."""

    def __init__(
        self,
        model_name_or_path: str,
        output_dim: int = 64,
        *,
        max_frames: int = 750,
        local_files_only: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.model_name_or_path = str(model_name_or_path)
        self.output_dim = int(output_dim)
        self.max_frames = int(max_frames)
        self.local_files_only = bool(local_files_only)
        self.dropout = nn.Dropout(float(dropout))

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_name_or_path,
            local_files_only=self.local_files_only,
        )
        self.backbone = AutoModel.from_pretrained(
            self.model_name_or_path,
            local_files_only=self.local_files_only,
            output_hidden_states=True,
            trust_remote_code=True,
        )
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        num_hidden_layers = int(getattr(self.backbone.config, "num_hidden_layers", 24)) + 1
        hidden_size = int(getattr(self.backbone.config, "hidden_size", 1024))
        self.layer_weights = nn.Parameter(torch.zeros(num_hidden_layers))
        self.output_proj = nn.Linear(hidden_size, self.output_dim, bias=True)

        self._feature_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def sample_rate(self) -> int:
        sr = getattr(self.feature_extractor, "sampling_rate", None)
        return int(sr or 24000)

    def _cache_key(self, path: str, start_sec: float, duration_sec: float) -> str:
        return f"{Path(path).resolve()}::{start_sec:.3f}::{duration_sec:.3f}"

    def _encode_waveform(self, wav: torch.Tensor, sample_rate: int) -> Tuple[torch.Tensor, torch.Tensor]:
        wav = _ensure_1d_mono(wav)
        wav = _resample_if_needed(wav, sample_rate, self.sample_rate)
        processed = self.feature_extractor(
            wav.cpu().numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        input_values = processed["input_values"]
        attention_mask = processed.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_values, dtype=torch.long)

        with torch.no_grad():
            outputs = self.backbone(
                input_values=input_values.to(self.backbone.device),
                attention_mask=attention_mask.to(self.backbone.device),
                output_hidden_states=True,
            )
        hidden_states = torch.stack(
            [layer.detach().cpu().to(torch.float16) for layer in outputs.hidden_states],
            dim=0,
        )
        seq_mask = torch.ones(hidden_states.shape[1], hidden_states.shape[2], dtype=torch.long)
        return hidden_states.squeeze(1), seq_mask.squeeze(0)

    def _load_clip_hidden_states(
        self,
        path: str,
        start_sec: float,
        duration_sec: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = self._cache_key(path, start_sec, duration_sec)
        cached = self._feature_cache.get(key)
        if cached is not None:
            return cached

        wav, sr = torchaudio.load(path)
        wav = _ensure_1d_mono(wav)
        total_sec = float(wav.shape[-1]) / float(sr)
        start_sec, duration_sec = _clamp_clip_window(total_sec, start_sec, duration_sec)
        start_idx = int(round(start_sec * sr))
        end_idx = int(round((start_sec + duration_sec) * sr))
        clip = wav[start_idx:end_idx]
        hidden_states, seq_mask = self._encode_waveform(clip, sr)
        self._feature_cache[key] = (hidden_states, seq_mask)
        return hidden_states, seq_mask

    def _load_tensor_hidden_states(
        self,
        wav: torch.Tensor,
        sample_rate: int,
        cache_key: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if cache_key and cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        hidden_states, seq_mask = self._encode_waveform(wav, sample_rate)
        if cache_key:
            self._feature_cache[cache_key] = (hidden_states, seq_mask)
        return hidden_states, seq_mask

    def _aggregate_hidden_states(
        self,
        hidden_states: torch.Tensor,
        out_device: torch.device,
        out_dtype: torch.dtype,
        scale: float,
    ) -> torch.Tensor:
        hidden_states = hidden_states.to(device=out_device, dtype=torch.float32)
        weights = torch.softmax(self.layer_weights.float(), dim=0).view(-1, 1, 1)
        merged = torch.sum(weights * hidden_states, dim=0)
        merged = _pool_time(merged.unsqueeze(0), self.max_frames).squeeze(0)
        merged = self.output_proj(self.dropout(merged))
        merged = merged * float(scale)
        return merged.to(dtype=out_dtype)

    def _pack_sequences(
        self,
        sequences: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        out_device: torch.device,
        out_dtype: torch.dtype,
        scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        packed_states: List[torch.Tensor] = []
        order_mask: List[int] = []
        for batch_idx, (hidden_states, seq_mask) in enumerate(sequences):
            projected = self._aggregate_hidden_states(hidden_states, out_device, out_dtype, scale)
            valid_len = min(projected.shape[0], seq_mask.shape[0])
            projected = projected[:valid_len]
            packed_states.append(projected)
            order_mask.append(batch_idx)

        if not packed_states:
            empty = torch.zeros(1, 1, self.output_dim, device=out_device, dtype=out_dtype)
            mask = torch.zeros(1, device=out_device, dtype=torch.long)
            return empty, mask

        max_len = max(state.shape[0] for state in packed_states)
        padded_states: List[torch.Tensor] = []
        for state in packed_states:
            if state.shape[0] < max_len:
                pad = torch.zeros(
                    max_len - state.shape[0],
                    state.shape[1],
                    device=out_device,
                    dtype=out_dtype,
                )
                state = torch.cat([state, pad], dim=0)
            padded_states.append(state)

        return torch.stack(padded_states, dim=0), torch.tensor(
            order_mask,
            device=out_device,
            dtype=torch.long,
        )

    def build_condition_from_metadata(
        self,
        metadata_list: Sequence[Dict[str, object]],
        out_device: torch.device,
        out_dtype: torch.dtype,
        *,
        scale: float = 1.0,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        sequences: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for meta in metadata_list:
            ref_path = str(meta.get("ref_voice_audio_path") or "").strip()
            if not ref_path:
                return None, None
            start_sec = float(meta.get("ref_voice_start_sec") or 0.0)
            duration_sec = float(meta.get("ref_voice_duration_sec") or 3.0)
            hidden_states, seq_mask = self._load_clip_hidden_states(ref_path, start_sec, duration_sec)
            sequences.append((hidden_states, seq_mask))
        return self._pack_sequences(sequences, out_device, out_dtype, scale)

    def build_condition_from_audio_tensors(
        self,
        refer_audioss: Sequence[Sequence[torch.Tensor]],
        out_device: torch.device,
        out_dtype: torch.dtype,
        *,
        scale: float = 1.0,
        sample_rate: int = 48000,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        sequences: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for batch_idx, refer_audios in enumerate(refer_audioss):
            if not refer_audios:
                return None, None
            wav = refer_audios[0]
            cache_key = f"tensor::{batch_idx}::{wav.data_ptr()}"
            hidden_states, seq_mask = self._load_tensor_hidden_states(
                wav,
                sample_rate=sample_rate,
                cache_key=cache_key,
            )
            sequences.append((hidden_states, seq_mask))
        return self._pack_sequences(sequences, out_device, out_dtype, scale)


def save_voice_conditioner(module: MertReferenceConditioner, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(module.state_dict(), str(out / "voice_conditioner.safetensors"))
    (out / "voice_conditioner_config.json").write_text(
        json.dumps(
            {
                "model_name_or_path": module.model_name_or_path,
                "output_dim": module.output_dim,
                "max_frames": module.max_frames,
                "local_files_only": module.local_files_only,
                "dropout": float(module.dropout.p),
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )


def load_voice_conditioner(
    output_dir: str,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Optional[MertReferenceConditioner]:
    out = Path(output_dir)
    cfg_path = out / "voice_conditioner_config.json"
    weights_path = out / "voice_conditioner.safetensors"
    if not cfg_path.exists() or not weights_path.exists():
        return None

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    module = MertReferenceConditioner(
        model_name_or_path=cfg["model_name_or_path"],
        output_dim=int(cfg.get("output_dim", 64)),
        max_frames=int(cfg.get("max_frames", 750)),
        local_files_only=bool(cfg.get("local_files_only", False)),
        dropout=float(cfg.get("dropout", 0.0)),
    )
    state = load_file(str(weights_path))
    module.load_state_dict(state, strict=True)
    if device is not None:
        module = module.to(device)
    if dtype is not None:
        module.output_proj = module.output_proj.to(dtype=dtype)
    return module
