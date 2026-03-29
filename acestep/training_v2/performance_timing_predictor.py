"""Performance Timing Predictor for Phase E2-E4."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from acestep.training_v2.performance_inputs import (
    PREDICTOR_FEATURE_NAMES,
    PREDICTOR_FEATURE_VOCAB_SIZES,
)
from acestep.training_v2.release_targets import (
    EXPRESSIVITY_TARGET_NAMES,
    NUM_RELEASE_CLASSES,
    build_release_class_targets,
    compute_expressivity_loss,
)
from acestep.training_v2.timing_conditioning import (
    GLOBAL_FEATURE_NAMES,
    PHRASE_FEATURE_NAMES,
    TIMING_TARGET_NAMES,
)


@dataclass
class PerformanceTimingPredictorConfig:
    feature_vocab_sizes: list[int] = field(default_factory=lambda: list(PREDICTOR_FEATURE_VOCAB_SIZES))
    feature_names: list[str] = field(default_factory=lambda: list(PREDICTOR_FEATURE_NAMES))
    target_names: list[str] = field(default_factory=lambda: list(TIMING_TARGET_NAMES))
    expressivity_target_names: list[str] = field(default_factory=lambda: list(EXPRESSIVITY_TARGET_NAMES))
    phrase_feature_names: list[str] = field(default_factory=lambda: list(PHRASE_FEATURE_NAMES))
    global_feature_names: list[str] = field(default_factory=lambda: list(GLOBAL_FEATURE_NAMES))
    hidden_size: int = 256
    num_heads: int = 4
    num_layers: int = 2
    ff_ratio: int = 4
    dropout: float = 0.1
    output_dim: int = 2048
    max_seq_len: int = 2048
    max_phrase_ids: int = 256
    condition_scale: float = 1.0
    enable_global_condition: bool = True
    enable_expressivity_heads: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PerformanceTimingPredictorConfig":
        normalized = dict(d)
        if "feature_vocab_sizes" not in normalized:
            normalized["feature_vocab_sizes"] = list(PREDICTOR_FEATURE_VOCAB_SIZES)
        if "feature_names" not in normalized:
            normalized["feature_names"] = list(PREDICTOR_FEATURE_NAMES)
        if "target_names" not in normalized:
            normalized["target_names"] = list(TIMING_TARGET_NAMES)
        if "expressivity_target_names" not in normalized:
            normalized["expressivity_target_names"] = list(EXPRESSIVITY_TARGET_NAMES)
        if "phrase_feature_names" not in normalized:
            normalized["phrase_feature_names"] = list(PHRASE_FEATURE_NAMES)
        if "global_feature_names" not in normalized:
            normalized["global_feature_names"] = list(GLOBAL_FEATURE_NAMES)
        if "condition_scale" not in normalized:
            normalized["condition_scale"] = 1.0
        if "enable_global_condition" not in normalized:
            normalized["enable_global_condition"] = True
        if "enable_expressivity_heads" not in normalized:
            normalized["enable_expressivity_heads"] = True
        return cls(**{k: v for k, v in normalized.items() if k in cls.__dataclass_fields__})


class PerformanceTimingPredictor(nn.Module):
    """Predict decoder-facing timing states and timing targets."""

    def __init__(self, config: PerformanceTimingPredictorConfig) -> None:
        super().__init__()
        self.config = config
        hidden_size = int(config.hidden_size)
        output_dim = int(config.output_dim)

        self.feature_embeddings = nn.ModuleList(
            [nn.Embedding(int(vocab), hidden_size) for vocab in config.feature_vocab_sizes]
        )
        self.feature_type_embeddings = nn.Embedding(len(config.feature_vocab_sizes), hidden_size)
        self.pos_emb = nn.Embedding(config.max_seq_len, hidden_size)
        self.phrase_id_emb = nn.Embedding(config.max_phrase_ids, hidden_size)
        self.phrase_feature_proj = nn.Sequential(
            nn.Linear(len(config.phrase_feature_names), hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
        )
        self.global_feature_proj = nn.Sequential(
            nn.Linear(len(config.global_feature_names), output_dim),
            nn.SiLU(),
            nn.LayerNorm(output_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=config.num_heads,
            dim_feedforward=hidden_size * config.ff_ratio,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.projected_head = nn.Linear(hidden_size, output_dim)
        self.target_head = nn.Linear(hidden_size, len(config.target_names))
        self.global_summary_proj = (
            nn.Sequential(
                nn.Linear(hidden_size, output_dim),
                nn.SiLU(),
                nn.LayerNorm(output_dim),
            )
            if config.enable_global_condition
            else None
        )
        self.expressivity_head = (
            nn.Linear(hidden_size, len(config.expressivity_target_names))
            if config.enable_expressivity_heads
            else None
        )
        self.release_head = (
            nn.Linear(hidden_size, NUM_RELEASE_CLASSES)
            if config.enable_expressivity_heads
            else None
        )
        self.output_gate = nn.Parameter(torch.tensor(-8.0))
        self.global_gate = nn.Parameter(torch.tensor(-8.0))
        self._init_weights()

    def _init_weights(self) -> None:
        for emb in list(self.feature_embeddings) + [self.feature_type_embeddings, self.pos_emb, self.phrase_id_emb]:
            nn.init.normal_(emb.weight, std=0.02)
        nn.init.zeros_(self.projected_head.weight)
        nn.init.zeros_(self.projected_head.bias)
        nn.init.zeros_(self.target_head.weight)
        nn.init.zeros_(self.target_head.bias)
        if self.expressivity_head is not None:
            nn.init.zeros_(self.expressivity_head.weight)
            nn.init.zeros_(self.expressivity_head.bias)
        if self.release_head is not None:
            nn.init.zeros_(self.release_head.weight)
            nn.init.zeros_(self.release_head.bias)
        nn.init.zeros_(self.phrase_feature_proj[0].weight)
        nn.init.zeros_(self.phrase_feature_proj[0].bias)
        nn.init.xavier_uniform_(self.global_feature_proj[0].weight)
        nn.init.zeros_(self.global_feature_proj[0].bias)
        if self.global_summary_proj is not None:
            nn.init.xavier_uniform_(self.global_summary_proj[0].weight)
            nn.init.zeros_(self.global_summary_proj[0].bias)

    @staticmethod
    def _masked_mean(hidden: torch.Tensor, timing_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if timing_mask is None:
            return hidden.mean(dim=1)
        weights = timing_mask.float().unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

    def encode(
        self,
        predictor_inputs: Optional[torch.Tensor] = None,
        timing_tokens: Optional[torch.Tensor] = None,
        timing_mask: Optional[torch.Tensor] = None,
        phrase_features: Optional[torch.Tensor] = None,
        global_features: Optional[torch.Tensor] = None,
        phrase_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        source_tokens = predictor_inputs if predictor_inputs is not None else timing_tokens
        if source_tokens is None:
            raise ValueError("Need predictor_inputs or timing_tokens")
        if source_tokens.ndim != 3:
            raise ValueError(f"predictor inputs must be [B, N, F], got {tuple(source_tokens.shape)}")
        batch_size, num_events, num_features = source_tokens.shape
        if num_features != len(self.config.feature_vocab_sizes):
            raise ValueError(
                f"predictor feature width mismatch: expected {len(self.config.feature_vocab_sizes)}, "
                f"got {num_features}"
            )

        pos_ids = torch.arange(num_events, device=source_tokens.device).unsqueeze(0).expand(batch_size, -1)
        pos_ids = pos_ids.clamp(0, self.config.max_seq_len - 1)
        feature_type_ids = torch.arange(num_features, device=source_tokens.device)
        type_emb = self.feature_type_embeddings(feature_type_ids).view(1, 1, num_features, -1)

        x = self.pos_emb(pos_ids)
        stacked = []
        for idx in range(num_features):
            vocab_size = int(self.config.feature_vocab_sizes[idx])
            tok = source_tokens[..., idx].clamp(0, vocab_size - 1)
            stacked.append(self.feature_embeddings[idx](tok))
        features = torch.stack(stacked, dim=2) + type_emb
        x = x + features.sum(dim=2)

        if phrase_ids is not None:
            phrase_ids = phrase_ids.clamp(min=0, max=self.config.max_phrase_ids - 1)
            x = x + self.phrase_id_emb(phrase_ids)

        if phrase_features is not None:
            x = x + self.phrase_feature_proj(phrase_features.to(x.dtype))

        pad_mask = None if timing_mask is None else ~timing_mask
        hidden = self.transformer(x, src_key_padding_mask=pad_mask)
        hidden = self.norm(hidden)
        projected = self.projected_head(hidden)
        projected = projected * (torch.sigmoid(self.output_gate) * self.config.condition_scale)

        global_condition = None
        if self.global_summary_proj is not None:
            global_condition = self.global_summary_proj(self._masked_mean(hidden, timing_mask))
            if global_features is not None:
                global_condition = global_condition + self.global_feature_proj(global_features.to(global_condition.dtype))
            global_condition = global_condition * (torch.sigmoid(self.global_gate) * self.config.condition_scale)

        predicted_targets = self.target_head(hidden)
        predicted_expressivity = (
            self.expressivity_head(hidden) if self.expressivity_head is not None else None
        )
        predicted_release_logits = (
            self.release_head(hidden) if self.release_head is not None else None
        )
        attn_mask = timing_mask.float() if timing_mask is not None else torch.ones(
            batch_size, num_events, device=source_tokens.device
        )
        return {
            "hidden": hidden,
            "projected": projected,
            "attn_mask": attn_mask,
            "global_condition": global_condition,
            "predicted_targets": predicted_targets,
            "predicted_expressivity": predicted_expressivity,
            "predicted_release_logits": predicted_release_logits,
        }

    def compute_loss(
        self,
        predicted_targets: torch.Tensor,
        timing_targets: torch.Tensor,
        timing_mask: Optional[torch.Tensor] = None,
        dur_weight: float = 1.0,
        onset_weight: float = 0.5,
        pause_weight: float = 0.5,
        phrase_weight: float = 0.3,
        tempo_weight: float = 0.2,
        terminal_weight: float = 0.2,
    ) -> torch.Tensor:
        if predicted_targets.shape[-1] != len(self.config.target_names):
            raise ValueError(
                f"predicted_targets width mismatch: expected {len(self.config.target_names)}, "
                f"got {predicted_targets.shape[-1]}"
            )
        if timing_targets.shape[-1] != len(self.config.target_names):
            raise ValueError(
                f"timing_targets width mismatch: expected {len(self.config.target_names)}, "
                f"got {timing_targets.shape[-1]}"
            )

        valid = (
            timing_mask.float().unsqueeze(-1)
            if timing_mask is not None
            else torch.ones(predicted_targets.shape[:2], device=predicted_targets.device).unsqueeze(-1)
        )
        n_valid = valid.sum().clamp(min=1.0)

        pred_dur = predicted_targets[..., 0:1]
        tgt_dur = timing_targets[..., 0:1]
        l_dur = (F.huber_loss(pred_dur, tgt_dur, reduction="none") * valid).sum() / n_valid

        pred_onset_beat = predicted_targets[..., 1:2]
        tgt_onset_beat = timing_targets[..., 1:2]
        l_onset_beat = (F.huber_loss(pred_onset_beat, tgt_onset_beat, reduction="none") * valid).sum() / n_valid

        pred_pause_exist = predicted_targets[..., 2:3]
        tgt_pause_exist = timing_targets[..., 2:3]
        l_pause_exist = (
            F.binary_cross_entropy_with_logits(pred_pause_exist, tgt_pause_exist, reduction="none") * valid
        ).sum() / n_valid

        pred_pause_dur = predicted_targets[..., 3:4]
        tgt_pause_dur = timing_targets[..., 3:4]
        pause_present = (tgt_pause_exist > 0.5).float()
        l_pause_dur = (
            F.huber_loss(pred_pause_dur, tgt_pause_dur, reduction="none") * valid * pause_present
        ).sum() / (valid * pause_present).sum().clamp(min=1.0)
        l_pause = 0.5 * l_pause_exist + 0.5 * l_pause_dur

        pred_onset_downbeat = predicted_targets[..., 4:5]
        tgt_onset_downbeat = timing_targets[..., 4:5]
        l_onset_downbeat = (
            F.huber_loss(pred_onset_downbeat, tgt_onset_downbeat, reduction="none") * valid
        ).sum() / n_valid
        l_onset = 0.5 * l_onset_beat + 0.5 * l_onset_downbeat

        pred_phrase = predicted_targets[..., 5:6]
        tgt_phrase = timing_targets[..., 5:6].clamp(0.0, 2.0) / 2.0
        l_phrase = (F.huber_loss(pred_phrase, tgt_phrase, reduction="none") * valid).sum() / n_valid

        pred_tempo = predicted_targets[..., 6:7]
        tgt_tempo = timing_targets[..., 6:7]
        l_tempo = (F.huber_loss(pred_tempo, tgt_tempo, reduction="none") * valid).sum() / n_valid

        pred_terminal = predicted_targets[..., 7:8]
        tgt_terminal = timing_targets[..., 7:8]
        l_terminal = (F.huber_loss(pred_terminal, tgt_terminal, reduction="none") * valid).sum() / n_valid

        return (
            dur_weight * l_dur
            + onset_weight * l_onset
            + pause_weight * l_pause
            + phrase_weight * l_phrase
            + tempo_weight * l_tempo
            + terminal_weight * l_terminal
        )

    def forward(
        self,
        predictor_inputs: Optional[torch.Tensor] = None,
        timing_tokens: Optional[torch.Tensor] = None,
        timing_mask: Optional[torch.Tensor] = None,
        timing_targets: Optional[torch.Tensor] = None,
        phrase_features: Optional[torch.Tensor] = None,
        global_features: Optional[torch.Tensor] = None,
        phrase_ids: Optional[torch.Tensor] = None,
        f0_targets: Optional[torch.Tensor] = None,
        energy_targets: Optional[torch.Tensor] = None,
        terminal_decay: Optional[torch.Tensor] = None,
        cv_ratio_targets: Optional[torch.Tensor] = None,
        release_targets: Optional[torch.Tensor] = None,
        alignment_confidence: Optional[torch.Tensor] = None,
        beat_confidence: Optional[torch.Tensor] = None,
        dur_weight: float = 1.0,
        onset_weight: float = 0.5,
        pause_weight: float = 0.5,
        phrase_weight: float = 0.3,
        tempo_weight: float = 0.2,
        terminal_weight: float = 0.2,
        expressivity_f0_weight: float = 0.4,
        expressivity_energy_weight: float = 0.4,
        expressivity_terminal_decay_weight: float = 0.5,
        expressivity_cv_ratio_weight: float = 0.3,
        expressivity_release_weight: float = 0.2,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.encode(
            predictor_inputs=predictor_inputs,
            timing_tokens=timing_tokens,
            timing_mask=timing_mask,
            phrase_features=phrase_features,
            global_features=global_features,
            phrase_ids=phrase_ids,
        )
        source_tokens = predictor_inputs if predictor_inputs is not None else timing_tokens
        if source_tokens is None:
            raise ValueError("Need predictor_inputs or timing_tokens")
        predictor_loss = torch.tensor(0.0, device=source_tokens.device)
        if timing_targets is not None:
            predictor_loss = self.compute_loss(
                predicted_targets=outputs["predicted_targets"],
                timing_targets=timing_targets,
                timing_mask=timing_mask,
                dur_weight=dur_weight,
                onset_weight=onset_weight,
                pause_weight=pause_weight,
                phrase_weight=phrase_weight,
                tempo_weight=tempo_weight,
                terminal_weight=terminal_weight,
            )
        if release_targets is None and terminal_decay is not None:
            release_targets = build_release_class_targets(terminal_decay)
        expressivity_loss = torch.tensor(0.0, device=source_tokens.device)
        if outputs.get("predicted_expressivity") is not None and outputs.get("predicted_release_logits") is not None:
            expressivity_loss = compute_expressivity_loss(
                outputs["predicted_expressivity"],
                outputs["predicted_release_logits"],
                f0_targets=f0_targets,
                energy_targets=energy_targets,
                terminal_decay=terminal_decay,
                cv_ratio_targets=cv_ratio_targets,
                release_targets=release_targets,
                alignment_confidence=alignment_confidence,
                beat_confidence=beat_confidence,
                timing_mask=timing_mask,
                f0_weight=expressivity_f0_weight,
                energy_weight=expressivity_energy_weight,
                terminal_weight=expressivity_terminal_decay_weight,
                cv_ratio_weight=expressivity_cv_ratio_weight,
                release_weight=expressivity_release_weight,
            )
        outputs["predictor_loss"] = predictor_loss
        outputs["expressivity_loss"] = expressivity_loss
        return outputs

    def export_config(self) -> Dict[str, Any]:
        return self.config.to_dict()

    def apply_inference_controls(
        self,
        outputs: Dict[str, torch.Tensor],
        *,
        phrase_features: Optional[torch.Tensor],
        relaxed_vs_tight: float = 0.0,
        phrase_final_slowing: float = 0.0,
        pause_emphasis: float = 0.0,
        speech_like_vs_legato: float = 0.0,
        release_intensity: float = 0.0,
        timing_guidance_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        projected = outputs["projected"] * float(timing_guidance_scale)
        global_condition = outputs.get("global_condition")
        targets = outputs.get("predicted_targets")
        expressivity = outputs.get("predicted_expressivity")

        phrase_final_mask = None
        if phrase_features is not None and phrase_features.shape[-1] > 0:
            phrase_final_mask = phrase_features[..., -1:].to(projected.dtype)

        local_scale = 1.0 + 0.08 * float(relaxed_vs_tight)
        projected = projected * local_scale
        if global_condition is not None:
            global_condition = global_condition * (1.0 + 0.12 * float(relaxed_vs_tight))

        if phrase_final_mask is not None and abs(float(phrase_final_slowing)) > 1e-6:
            projected = projected * (1.0 + 0.18 * float(phrase_final_slowing) * phrase_final_mask)
            if global_condition is not None:
                pooled = phrase_final_mask.mean(dim=1)
                global_condition = global_condition * (1.0 + 0.15 * float(phrase_final_slowing) * pooled)

        if targets is not None:
            targets = targets.clone()
            targets[..., 2:4] = targets[..., 2:4] * (1.0 + 0.20 * float(pause_emphasis))
            targets[..., 0:1] = targets[..., 0:1] * (1.0 + 0.10 * float(relaxed_vs_tight))
            targets[..., 5:6] = targets[..., 5:6] + 0.20 * float(speech_like_vs_legato)
            targets[..., 7:8] = targets[..., 7:8] + 0.25 * float(phrase_final_slowing)

        if expressivity is not None:
            expressivity = expressivity.clone()
            expressivity[..., 1:2] = expressivity[..., 1:2] * (1.0 - 0.10 * float(speech_like_vs_legato))
            expressivity[..., 2:3] = expressivity[..., 2:3] * (1.0 + 0.20 * float(release_intensity))

        adjusted = dict(outputs)
        adjusted["projected"] = projected
        adjusted["global_condition"] = global_condition
        adjusted["predicted_targets"] = targets
        adjusted["predicted_expressivity"] = expressivity
        return adjusted
