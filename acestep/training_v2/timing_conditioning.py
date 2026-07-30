"""
Enhanced timing / prosody conditioning branch for ACE-Step Phase D.

This version keeps the original "small conditioning branch" footprint, but
upgrades the representation from a minimal word-level timing sketch to a richer
event-level timing schema that can carry phoneme-level alignments when the
preprocessing stack provides them.

Core upgrades over the original V1:
- supports 8 timing feature channels instead of 4
- supports richer timing targets (phrase / tempo / terminal shaping)
- adds a typed timing stream and learned gate before concatenation
- remains backward compatible with legacy 4-feature timing sidecars
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional, Sequence, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Quantisation constants
# ---------------------------------------------------------------------------

TIMING_FEATURE_NAMES: Tuple[str, ...] = (
    "dur_bin",
    "onset_beat_bin",
    "pause_bin",
    "beat_phase_bin",
    "onset_downbeat_bin",
    "phrase_boundary_bin",
    "tempo_bin",
    "terminal_bin",
)

LEGACY_FEATURE_NAMES: Tuple[str, ...] = (
    "dur_bin",
    "onset_beat_bin",
    "pause_bin",
    "beat_phase_bin",
)

TIMING_TARGET_NAMES: Tuple[str, ...] = (
    "log_dur",
    "onset_beat_norm",
    "pause_exists",
    "pause_dur_log",
    "onset_downbeat_norm",
    "phrase_boundary",
    "tempo_ratio_log",
    "terminal_ratio_log",
)

PHRASE_FEATURE_NAMES: Tuple[str, ...] = (
    "phrase_index_norm",
    "phrase_progress",
    "phrase_remaining",
    "phrase_span_ratio",
    "phrase_duration_log",
    "pause_after_log",
    "tempo_ratio_log",
    "terminal_ratio_log",
    "boundary_strength",
    "is_phrase_final",
)

GLOBAL_FEATURE_NAMES: Tuple[str, ...] = (
    "n_phrases_norm",
    "mean_phrase_duration_log",
    "mean_pause_log",
    "pause_rate",
    "mean_tempo_ratio_log",
    "terminal_rate",
    "audio_duration_log",
    "mean_events_per_phrase_log",
)

# log(duration seconds)
DUR_LOG_MIN: float = -3.0
DUR_LOG_MAX: float = 0.7
N_DUR_BINS: int = 16

# onset deviation normalized by local beat period
ONSET_DEV_MIN: float = -0.5
ONSET_DEV_MAX: float = 0.5
N_ONSET_DEV_BINS: int = 16

# pause duration seconds
PAUSE_EDGES: Tuple[float, ...] = (0.05, 0.15, 0.3, 0.6, 1.0, 2.0)
N_PAUSE_BINS: int = 8

# beat phase [0, 1)
N_BEAT_PHASE_BINS: int = 8

# phrase boundary type: none, minor, major
N_PHRASE_BINS: int = 3

# local tempo ratio log-binned around 1.0x
TEMPO_RATIO_LOG_MIN: float = math.log(0.5)
TEMPO_RATIO_LOG_MAX: float = math.log(1.8)
N_TEMPO_BINS: int = 12

# terminal shaping ratio log-binned around 1.0x
TERMINAL_RATIO_LOG_MIN: float = math.log(0.5)
TERMINAL_RATIO_LOG_MAX: float = math.log(2.5)
N_TERMINAL_BINS: int = 8

TIMING_FEATURE_VOCAB_SIZES: Tuple[int, ...] = (
    N_DUR_BINS,
    N_ONSET_DEV_BINS,
    N_PAUSE_BINS,
    N_BEAT_PHASE_BINS,
    N_ONSET_DEV_BINS,
    N_PHRASE_BINS,
    N_TEMPO_BINS,
    N_TERMINAL_BINS,
)


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def _quantise_symmetric(value: float, vmin: float, vmax: float, n_bins: int) -> int:
    clamped = max(vmin, min(vmax, float(value)))
    frac = (clamped - vmin) / max(vmax - vmin, 1e-6)
    return int(frac * (n_bins - 1) + 0.5)


def quantise_duration(log_dur: float) -> int:
    return _quantise_symmetric(log_dur, DUR_LOG_MIN, DUR_LOG_MAX, N_DUR_BINS)


def quantise_onset_dev(normalised_dev: float) -> int:
    return _quantise_symmetric(normalised_dev, ONSET_DEV_MIN, ONSET_DEV_MAX, N_ONSET_DEV_BINS)


def quantise_pause(pause_dur: float) -> int:
    if pause_dur <= 0.0:
        return 0
    for i, edge in enumerate(PAUSE_EDGES):
        if pause_dur < edge:
            return i + 1
    return N_PAUSE_BINS - 1


def quantise_beat_phase(phase: float) -> int:
    clamped = max(0.0, min(0.9999, float(phase)))
    return int(clamped * N_BEAT_PHASE_BINS)


def quantise_tempo_ratio_log(value: float) -> int:
    return _quantise_symmetric(value, TEMPO_RATIO_LOG_MIN, TEMPO_RATIO_LOG_MAX, N_TEMPO_BINS)


def quantise_terminal_ratio_log(value: float) -> int:
    return _quantise_symmetric(value, TERMINAL_RATIO_LOG_MIN, TERMINAL_RATIO_LOG_MAX, N_TERMINAL_BINS)


def _nearest_time_and_period(time_s: float, anchors: Sequence[float]) -> Tuple[float, float]:
    if not anchors:
        return 0.0, 0.5
    if len(anchors) == 1:
        return float(anchors[0]), 0.5
    best_idx = min(range(len(anchors)), key=lambda i: abs(float(anchors[i]) - time_s))
    nearest = float(anchors[best_idx])
    if best_idx == 0:
        period = max(float(anchors[1]) - nearest, 0.1)
    elif best_idx == len(anchors) - 1:
        period = max(nearest - float(anchors[best_idx - 1]), 0.1)
    else:
        left = nearest - float(anchors[best_idx - 1])
        right = float(anchors[best_idx + 1]) - nearest
        period = max(0.5 * (left + right), 0.1)
    return nearest, period


def upgrade_legacy_timing_tokens(
    tokens: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Upgrade old 4-feature timing tensors to the new 8-feature schema."""
    if tokens.ndim < 2:
        raise ValueError(f"timing_tokens must have at least 2 dims, got {tuple(tokens.shape)}")
    if tokens.shape[-1] == len(TIMING_FEATURE_NAMES):
        return tokens, targets
    if tokens.shape[-1] != len(LEGACY_FEATURE_NAMES):
        raise ValueError(f"Unsupported timing token width: {tokens.shape[-1]}")

    leading_shape = tokens.shape[:-1]
    flat_tokens = tokens.reshape(-1, tokens.shape[-1])
    upgraded = torch.zeros(flat_tokens.shape[0], len(TIMING_FEATURE_NAMES), dtype=tokens.dtype, device=tokens.device)
    upgraded[:, : len(LEGACY_FEATURE_NAMES)] = flat_tokens
    upgraded[:, 5] = 0
    upgraded[:, 6] = N_TEMPO_BINS // 2
    upgraded[:, 7] = 0
    upgraded = upgraded.view(*leading_shape, len(TIMING_FEATURE_NAMES))

    upgraded_targets = targets
    if targets is not None:
        if targets.shape[-1] == len(TIMING_TARGET_NAMES):
            upgraded_targets = targets
        elif targets.shape[-1] == 4:
            flat_targets = targets.reshape(-1, targets.shape[-1])
            tgt = torch.zeros(flat_targets.shape[0], len(TIMING_TARGET_NAMES), dtype=targets.dtype, device=targets.device)
            tgt[:, :4] = flat_targets
            upgraded_targets = tgt.view(*targets.shape[:-1], len(TIMING_TARGET_NAMES))
        else:
            raise ValueError(f"Unsupported timing target width: {targets.shape[-1]}")
    return upgraded, upgraded_targets


def _baseline_duration_for_terminal(events: Sequence[Dict[str, float]], idx: int) -> float:
    if not events:
        return 0.2
    label = str(events[idx].get("label", "")).strip().lower()
    is_vowel = bool(events[idx].get("is_vowel", False)) or any(ch in "aeiou" for ch in label)
    candidates = []
    for ev in events:
        ev_label = str(ev.get("label", "")).strip().lower()
        ev_is_vowel = bool(ev.get("is_vowel", False)) or any(ch in "aeiou" for ch in ev_label)
        if ev_is_vowel == is_vowel:
            candidates.append(max(float(ev["end"]) - float(ev["start"]), 1e-4))
    if not candidates:
        candidates = [max(float(ev["end"]) - float(ev["start"]), 1e-4) for ev in events]
    candidates = sorted(candidates)
    return float(candidates[len(candidates) // 2])


def build_timing_tokens_from_events(
    events: Sequence[Dict[str, Any]],
    beat_times: Sequence[float],
    downbeat_times: Optional[Sequence[float]] = None,
    song_bpm_hint: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build timing tensors from aligned timing events.

    Each event must have at least:
    - start
    - end

    Optional fields:
    - label
    - pause_after
    - phrase_boundary (0 none, 1 minor, 2 major)
    - phrase_break_after
    - is_phrase_final
    - beat_period
    - local_tempo_ratio
    - is_vowel
    """
    if not events:
        raise ValueError("Cannot build timing tokens from an empty event list")

    beats = [float(x) for x in beat_times] if beat_times else [0.0]
    downbeats = [float(x) for x in downbeat_times] if downbeat_times else beats
    if song_bpm_hint and song_bpm_hint > 1e-3:
        base_period = 60.0 / float(song_bpm_hint)
    elif len(beats) >= 2:
        beat_diffs = [max(beats[i + 1] - beats[i], 0.1) for i in range(len(beats) - 1)]
        beat_diffs = sorted(beat_diffs)
        base_period = float(beat_diffs[len(beat_diffs) // 2])
    else:
        base_period = 0.5

    tokens_list = []
    targets_list = []
    for i, ev in enumerate(events):
        t0 = float(ev["start"])
        t1 = float(ev["end"])
        dur = max(t1 - t0, 1e-4)
        log_dur = math.log(dur)

        if i + 1 < len(events):
            default_gap = max(0.0, float(events[i + 1]["start"]) - t1)
        else:
            default_gap = 0.0
        gap = max(0.0, float(ev.get("pause_after", default_gap)))

        nearest_beat, beat_period = _nearest_time_and_period(t0, beats)
        beat_period = max(float(ev.get("beat_period", beat_period)), 0.1)
        onset_beat_norm = (t0 - nearest_beat) / beat_period
        beat_phase = ((t0 - nearest_beat) % beat_period) / beat_period

        nearest_downbeat, downbeat_period = _nearest_time_and_period(t0, downbeats)
        downbeat_period = max(downbeat_period, 0.1)
        onset_downbeat_norm = (t0 - nearest_downbeat) / downbeat_period

        phrase_boundary = int(ev.get("phrase_boundary", 2 if i == len(events) - 1 else 0))
        phrase_break_after = max(float(ev.get("phrase_break_after", gap if phrase_boundary else 0.0)), 0.0)

        local_tempo_ratio = float(ev.get("local_tempo_ratio", beat_period / max(base_period, 1e-6)))
        tempo_ratio_log = math.log(max(local_tempo_ratio, 1e-4))

        baseline = _baseline_duration_for_terminal(events, i)
        terminal_ratio = dur / max(baseline, 1e-4)
        if not bool(ev.get("is_phrase_final", phrase_boundary > 0)):
            terminal_ratio = 1.0
        terminal_ratio_log = math.log(max(terminal_ratio, 1e-4))

        tokens_list.append(
            [
                quantise_duration(log_dur),
                quantise_onset_dev(onset_beat_norm),
                quantise_pause(gap),
                quantise_beat_phase(beat_phase),
                quantise_onset_dev(onset_downbeat_norm),
                max(0, min(N_PHRASE_BINS - 1, phrase_boundary)),
                quantise_tempo_ratio_log(tempo_ratio_log),
                quantise_terminal_ratio_log(terminal_ratio_log),
            ]
        )
        targets_list.append(
            [
                log_dur,
                onset_beat_norm,
                float(gap > 0.05),
                math.log(max(gap, 1e-4)),
                onset_downbeat_norm,
                float(phrase_boundary),
                tempo_ratio_log,
                terminal_ratio_log,
            ]
        )

    tokens = torch.tensor(tokens_list, dtype=torch.long)
    targets = torch.tensor(targets_list, dtype=torch.float32)
    mask = torch.ones(tokens.shape[0], dtype=torch.bool)
    return tokens, targets, mask


def build_timing_tokens(
    word_starts: Sequence[float],
    word_ends: Sequence[float],
    beat_times: Sequence[float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward-compatible helper from the original Phase D implementation."""
    events = []
    for i, (start, end) in enumerate(zip(word_starts, word_ends)):
        phrase_boundary = 2 if i == len(word_starts) - 1 else 0
        events.append(
            {
                "start": float(start),
                "end": float(end),
                "label": f"w{i}",
                "phrase_boundary": phrase_boundary,
                "is_phrase_final": bool(phrase_boundary),
            }
        )
    return build_timing_tokens_from_events(events, beat_times=beat_times, downbeat_times=beat_times)


# ---------------------------------------------------------------------------
# Generator-coupled local timing helpers
# ---------------------------------------------------------------------------


def build_local_timing_attention_mask(
    *,
    event_starts_sec: torch.Tensor,
    event_ends_sec: torch.Tensor,
    audio_durations_sec: torch.Tensor,
    timing_mask: torch.Tensor,
    query_len: int,
    dtype: torch.dtype,
    sigma_sec: float,
    window_sec: float,
    shift_sec: float = 0.0,
) -> torch.Tensor:
    """Build a fail-closed monotonic query-to-event additive attention mask.

    Every latent query is anchored to its absolute position in the song. Events
    inside ``window_sec`` receive a Gaussian distance bias, while all other
    events are masked. If a genuine gap contains no event in the window, the
    nearest valid event remains visible so SDPA never receives an all-masked row.
    """
    if query_len <= 0:
        raise ValueError(f"query_len must be positive, got {query_len}")
    if sigma_sec <= 0.0 or window_sec <= 0.0:
        raise ValueError(
            f"local timing sigma/window must be positive, got {sigma_sec}/{window_sec}"
        )
    if event_starts_sec.ndim != 2 or event_ends_sec.shape != event_starts_sec.shape:
        raise ValueError("event starts/ends must have identical [B, N] shapes")
    if timing_mask.shape != event_starts_sec.shape:
        raise ValueError("timing mask must match event starts/ends")
    batch, events = event_starts_sec.shape
    durations = audio_durations_sec.reshape(batch, 1, 1).float()
    if not bool(torch.isfinite(durations).all()) or bool((durations <= 0).any()):
        raise ValueError("audio durations must be finite and positive")

    starts = event_starts_sec.float().unsqueeze(1) + float(shift_sec)
    ends = event_ends_sec.float().unsqueeze(1) + float(shift_sec)
    valid = timing_mask.bool().unsqueeze(1)
    valid = valid & torch.isfinite(starts) & torch.isfinite(ends) & (ends >= starts)
    if not bool(valid.reshape(batch, -1).any(dim=1).all()):
        raise ValueError("every sample must contain at least one valid timing event")

    query_fraction = (
        (torch.arange(query_len, device=starts.device, dtype=torch.float32) + 0.5)
        / float(query_len)
    )
    query_time = query_fraction.view(1, query_len, 1) * durations
    distance = torch.maximum(starts - query_time, query_time - ends).clamp_min(0.0)
    gaussian = -0.5 * torch.square(distance / float(sigma_sec))
    allowed = valid & (distance <= float(window_sec))

    # Keep the nearest valid event visible in real pauses and at song edges.
    nearest_distance = distance.masked_fill(~valid, float("inf"))
    nearest_index = nearest_distance.argmin(dim=-1, keepdim=True)
    has_local = allowed.any(dim=-1, keepdim=True)
    nearest = torch.zeros_like(allowed).scatter_(-1, nearest_index, True)
    allowed = allowed | (nearest & ~has_local)

    minimum = torch.finfo(dtype).min
    additive = gaussian.to(dtype=dtype).masked_fill(~allowed, minimum)
    if additive.shape != (batch, query_len, events):
        raise RuntimeError(f"local timing mask shape mismatch: {tuple(additive.shape)}")
    if bool(torch.all(additive == minimum, dim=-1).any()):
        raise RuntimeError("local timing mask contains an all-masked query")
    return additive.unsqueeze(1)


def build_event_frame_weights(
    *,
    event_starts_sec: torch.Tensor,
    event_ends_sec: torch.Tensor,
    audio_durations_sec: torch.Tensor,
    timing_mask: torch.Tensor,
    query_len: int,
    extra_weight: float,
    margin_sec: float,
) -> torch.Tensor:
    """Return normalized [B, T] weights emphasizing aligned lyric frames."""
    if extra_weight < 0.0 or margin_sec < 0.0:
        raise ValueError("event flow weight and margin must be non-negative")
    batch = event_starts_sec.shape[0]
    durations = audio_durations_sec.reshape(batch, 1, 1).float()
    query_time = (
        (torch.arange(query_len, device=event_starts_sec.device, dtype=torch.float32) + 0.5)
        / float(query_len)
    ).view(1, query_len, 1) * durations
    starts = event_starts_sec.float().unsqueeze(1) - float(margin_sec)
    ends = event_ends_sec.float().unsqueeze(1) + float(margin_sec)
    valid = timing_mask.bool().unsqueeze(1)
    covered = (valid & (query_time >= starts) & (query_time <= ends)).any(dim=-1)
    weights = 1.0 + float(extra_weight) * covered.float()
    return weights / weights.mean(dim=1, keepdim=True).clamp_min(1e-6)


# ---------------------------------------------------------------------------
# TimingEncoder
# ---------------------------------------------------------------------------


@dataclass
class TimingEncoderConfig:
    feature_vocab_sizes: list[int] = field(default_factory=lambda: list(TIMING_FEATURE_VOCAB_SIZES))
    feature_names: list[str] = field(default_factory=lambda: list(TIMING_FEATURE_NAMES))
    target_names: list[str] = field(default_factory=lambda: list(TIMING_TARGET_NAMES))
    hidden_size: int = 256
    num_heads: int = 4
    num_layers: int = 2
    ff_ratio: int = 4
    dropout: float = 0.1
    output_dim: int = 2048
    max_seq_len: int = 2048
    use_stream_type_embedding: bool = True
    condition_scale: float = 1.0
    phrase_feature_names: list[str] = field(default_factory=lambda: list(PHRASE_FEATURE_NAMES))
    global_feature_names: list[str] = field(default_factory=lambda: list(GLOBAL_FEATURE_NAMES))
    phrase_feature_dim: int = len(PHRASE_FEATURE_NAMES)
    global_feature_dim: int = len(GLOBAL_FEATURE_NAMES)
    enable_phrase_features: bool = True
    enable_phrase_modulation: bool = False
    enable_absolute_time_conditioning: bool = False
    absolute_time_num_frequencies: int = 4
    global_condition_scale: float = 1.0
    init_profile: str = "safe"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TimingEncoderConfig":
        normalized = dict(d)
        if "feature_vocab_sizes" not in normalized:
            normalized["feature_vocab_sizes"] = list(TIMING_FEATURE_VOCAB_SIZES)
        if "feature_names" not in normalized:
            normalized["feature_names"] = list(TIMING_FEATURE_NAMES)
        if "target_names" not in normalized:
            normalized["target_names"] = list(TIMING_TARGET_NAMES)
        if "use_stream_type_embedding" not in normalized:
            normalized["use_stream_type_embedding"] = True
        if "condition_scale" not in normalized:
            normalized["condition_scale"] = 1.0
        if "phrase_feature_names" not in normalized:
            normalized["phrase_feature_names"] = list(PHRASE_FEATURE_NAMES)
        if "global_feature_names" not in normalized:
            normalized["global_feature_names"] = list(GLOBAL_FEATURE_NAMES)
        if "phrase_feature_dim" not in normalized:
            normalized["phrase_feature_dim"] = len(PHRASE_FEATURE_NAMES)
        if "global_feature_dim" not in normalized:
            normalized["global_feature_dim"] = len(GLOBAL_FEATURE_NAMES)
        if "enable_phrase_features" not in normalized:
            normalized["enable_phrase_features"] = any(
                key in d for key in ("phrase_feature_names", "phrase_feature_dim")
            )
        normalized.setdefault("enable_absolute_time_conditioning", False)
        normalized.setdefault("absolute_time_num_frequencies", 4)
        if "enable_phrase_modulation" not in normalized:
            normalized["enable_phrase_modulation"] = False
        if "global_condition_scale" not in normalized:
            normalized["global_condition_scale"] = 1.0
        if "init_profile" not in normalized:
            normalized["init_profile"] = "safe"
        return cls(**{k: v for k, v in normalized.items() if k in cls.__dataclass_fields__})


class TimingEncoder(nn.Module):
    """Event-level timing encoder with typed output stream and richer losses."""

    def __init__(self, config: TimingEncoderConfig) -> None:
        super().__init__()
        self.config = config
        if config.init_profile not in {"safe", "historical_v4", "suppressed_new"}:
            raise ValueError(f"Unsupported timing init profile: {config.init_profile}")
        H = config.hidden_size

        self.feature_embeddings = nn.ModuleList(
            [nn.Embedding(int(vocab), H) for vocab in config.feature_vocab_sizes]
        )
        self.feature_type_embeddings = nn.Embedding(len(config.feature_vocab_sizes), H)
        self.pos_emb = nn.Embedding(config.max_seq_len, H)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=H,
            nhead=config.num_heads,
            dim_feedforward=H * config.ff_ratio,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(H)
        self.proj = nn.Linear(H, config.output_dim, bias=True)
        self.phrase_feature_proj = (
            nn.Sequential(
                nn.Linear(config.phrase_feature_dim, H),
                nn.SiLU(),
                nn.LayerNorm(H),
            )
            if config.enable_phrase_features and config.phrase_feature_dim > 0
            else None
        )
        self.phrase_gate = (
            nn.Parameter(torch.tensor(-4.0))
            if self.phrase_feature_proj is not None
            else None
        )
        absolute_feature_dim = 4 + 4 * max(0, config.absolute_time_num_frequencies)
        self.absolute_time_proj = (
            nn.Sequential(
                nn.Linear(absolute_feature_dim, H),
                nn.SiLU(),
                nn.LayerNorm(H),
            )
            if config.enable_absolute_time_conditioning
            else None
        )
        self.absolute_time_gate = (
            nn.Parameter(torch.tensor(0.0))
            if self.absolute_time_proj is not None
            else None
        )
        self.global_summary_proj = (
            nn.Sequential(
                nn.Linear(H, config.output_dim),
                nn.SiLU(),
                nn.LayerNorm(config.output_dim),
            )
            if config.enable_phrase_modulation
            else None
        )
        self.global_feature_proj = (
            nn.Sequential(
                nn.Linear(config.global_feature_dim, config.output_dim),
                nn.SiLU(),
                nn.LayerNorm(config.output_dim),
            )
            if config.enable_phrase_modulation and config.global_feature_dim > 0
            else None
        )
        self.stream_type_embedding = (
            nn.Parameter(torch.zeros(config.output_dim))
            if config.use_stream_type_embedding
            else None
        )
        gate_init = 0.0 if config.init_profile == "historical_v4" else -8.0
        self.output_gate = nn.Parameter(torch.tensor(gate_init))
        self.global_gate = (
            nn.Parameter(torch.tensor(gate_init))
            if config.enable_phrase_modulation or config.init_profile in {"safe", "suppressed_new"}
            else None
        )

        self.prediction_head = TimingPredictionHead(H)

        self._init_weights()

    def _init_weights(self) -> None:
        for emb in list(self.feature_embeddings) + [self.feature_type_embeddings, self.pos_emb]:
            nn.init.normal_(emb.weight, std=0.02)
        if self.config.init_profile == "historical_v4":
            nn.init.xavier_uniform_(self.proj.weight)
        else:
            nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        if self.phrase_feature_proj is not None:
            if self.config.init_profile == "historical_v4":
                nn.init.xavier_uniform_(self.phrase_feature_proj[0].weight)
            else:
                nn.init.zeros_(self.phrase_feature_proj[0].weight)
            nn.init.zeros_(self.phrase_feature_proj[0].bias)
        if self.absolute_time_proj is not None:
            nn.init.xavier_uniform_(self.absolute_time_proj[0].weight)
            nn.init.zeros_(self.absolute_time_proj[0].bias)
        if self.global_summary_proj is not None:
            nn.init.xavier_uniform_(self.global_summary_proj[0].weight)
            nn.init.zeros_(self.global_summary_proj[0].bias)
        if self.global_feature_proj is not None:
            nn.init.xavier_uniform_(self.global_feature_proj[0].weight)
            nn.init.zeros_(self.global_feature_proj[0].bias)
        if self.stream_type_embedding is not None:
            if self.config.init_profile == "historical_v4":
                nn.init.normal_(self.stream_type_embedding, std=0.02)
            else:
                nn.init.zeros_(self.stream_type_embedding)
        self.prediction_head.reset_parameters()

    @staticmethod
    def _masked_mean(hidden: torch.Tensor, timing_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if timing_mask is None:
            return hidden.mean(dim=1)
        weights = timing_mask.float().unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

    def encode(
        self,
        timing_tokens: torch.Tensor,
        timing_mask: Optional[torch.Tensor] = None,
        phrase_features: Optional[torch.Tensor] = None,
        global_features: Optional[torch.Tensor] = None,
        event_starts_sec: Optional[torch.Tensor] = None,
        event_ends_sec: Optional[torch.Tensor] = None,
        audio_durations_sec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if timing_tokens.ndim != 3:
            raise ValueError(f"timing_tokens must be [B, N, F], got {tuple(timing_tokens.shape)}")
        B, N, F = timing_tokens.shape
        if F not in (len(LEGACY_FEATURE_NAMES), len(self.config.feature_vocab_sizes)):
            raise ValueError(f"Unsupported timing feature width: {F}")
        if F == len(LEGACY_FEATURE_NAMES):
            timing_tokens, _ = upgrade_legacy_timing_tokens(timing_tokens)
            F = timing_tokens.shape[-1]

        pos_ids = torch.arange(N, device=timing_tokens.device).unsqueeze(0).expand(B, -1)
        pos_ids = pos_ids.clamp(0, self.config.max_seq_len - 1)

        x = self.pos_emb(pos_ids)
        feature_type_ids = torch.arange(F, device=timing_tokens.device)
        type_emb = self.feature_type_embeddings(feature_type_ids).view(1, 1, F, -1)
        stacked = []
        for idx in range(F):
            vocab_size = int(self.config.feature_vocab_sizes[idx])
            tok = timing_tokens[..., idx].clamp(0, vocab_size - 1)
            stacked.append(self.feature_embeddings[idx](tok))
        features = torch.stack(stacked, dim=2) + type_emb
        x = x + features.sum(dim=2)
        if self.absolute_time_proj is not None:
            if event_starts_sec is None or event_ends_sec is None or audio_durations_sec is None:
                raise ValueError(
                    "absolute-time conditioning requires event starts, event ends, and audio durations"
                )
            durations = audio_durations_sec.to(x.dtype).reshape(B, 1).clamp(min=1e-4)
            starts = torch.nan_to_num(event_starts_sec.to(x.dtype), nan=0.0).clamp(min=0.0)
            ends = torch.nan_to_num(event_ends_sec.to(x.dtype), nan=0.0)
            ends = torch.maximum(ends, starts)
            start_norm = (starts / durations).clamp(0.0, 1.0)
            end_norm = (ends / durations).clamp(0.0, 1.0)
            center_norm = 0.5 * (start_norm + end_norm)
            span_norm = (end_norm - start_norm).clamp(min=0.0)
            absolute_features = [start_norm, end_norm, center_norm, span_norm]
            for exponent in range(max(0, self.config.absolute_time_num_frequencies)):
                angle = math.pi * (2.0 ** exponent)
                absolute_features.extend(
                    [
                        torch.sin(angle * start_norm),
                        torch.cos(angle * start_norm),
                        torch.sin(angle * end_norm),
                        torch.cos(angle * end_norm),
                    ]
                )
            absolute_inputs = torch.stack(absolute_features, dim=-1)
            absolute_hidden = self.absolute_time_proj(absolute_inputs)
            absolute_hidden = absolute_hidden * torch.sigmoid(self.absolute_time_gate)
            if timing_mask is not None:
                absolute_hidden = absolute_hidden * timing_mask.unsqueeze(-1).to(absolute_hidden.dtype)
            x = x + absolute_hidden
        if phrase_features is not None:
            if self.phrase_feature_proj is None:
                raise ValueError("phrase_features were provided but phrase feature support is disabled")
            if phrase_features.ndim != 3:
                raise ValueError(f"phrase_features must be [B, N, P], got {tuple(phrase_features.shape)}")
            phrase_inputs = torch.nan_to_num(
                phrase_features.to(x.dtype),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp(min=-5.0, max=5.0)
            phrase_hidden = self.phrase_feature_proj(phrase_inputs)
            if self.phrase_gate is not None:
                phrase_hidden = phrase_hidden * torch.sigmoid(self.phrase_gate)
            x = x + phrase_hidden

        pad_mask = None if timing_mask is None else ~timing_mask
        hidden = self.transformer(x, src_key_padding_mask=pad_mask)
        hidden = self.norm(hidden)
        self._last_pre_projection_rms = float(hidden.detach().float().square().mean().sqrt().item())
        projected = self.proj(hidden)

        if self.stream_type_embedding is not None:
            projected = projected + self.stream_type_embedding.view(1, 1, -1)
        projected = projected * (torch.sigmoid(self.output_gate) * self.config.condition_scale)
        self._last_output_rms = float(projected.detach().float().square().mean().sqrt().item())

        attn_mask = timing_mask.float() if timing_mask is not None else torch.ones(B, N, device=timing_tokens.device)
        global_condition = None
        if self.global_summary_proj is not None:
            pooled_hidden = self._masked_mean(hidden, timing_mask)
            global_condition = self.global_summary_proj(pooled_hidden)
            if global_features is not None and self.global_feature_proj is not None:
                if global_features.ndim != 2:
                    raise ValueError(f"global_features must be [B, G], got {tuple(global_features.shape)}")
                global_condition = global_condition + self.global_feature_proj(global_features.to(global_condition.dtype))
            if self.global_gate is None:
                raise RuntimeError("global timing modulation is enabled without a global gate")
            global_condition = global_condition * (
                torch.sigmoid(self.global_gate) * self.config.global_condition_scale
            )
        return hidden, projected, attn_mask, global_condition

    def compute_timing_loss(
        self,
        hidden: torch.Tensor,
        timing_targets: torch.Tensor,
        timing_mask: Optional[torch.Tensor] = None,
        dur_weight: float = 1.0,
        onset_weight: float = 0.5,
        pause_weight: float = 0.5,
        phrase_weight: float = 0.3,
        tempo_weight: float = 0.2,
        terminal_weight: float = 0.2,
    ) -> torch.Tensor:
        return self.prediction_head.compute_loss(
            hidden=hidden,
            timing_targets=timing_targets,
            timing_mask=timing_mask,
            dur_weight=dur_weight,
            onset_weight=onset_weight,
            pause_weight=pause_weight,
            phrase_weight=phrase_weight,
            tempo_weight=tempo_weight,
            terminal_weight=terminal_weight,
        )

    def forward(
        self,
        timing_tokens: torch.Tensor,
        timing_mask: Optional[torch.Tensor] = None,
        timing_targets: Optional[torch.Tensor] = None,
        phrase_features: Optional[torch.Tensor] = None,
        global_features: Optional[torch.Tensor] = None,
        event_starts_sec: Optional[torch.Tensor] = None,
        event_ends_sec: Optional[torch.Tensor] = None,
        audio_durations_sec: Optional[torch.Tensor] = None,
        dur_weight: float = 1.0,
        onset_weight: float = 0.5,
        pause_weight: float = 0.5,
        phrase_weight: float = 0.3,
        tempo_weight: float = 0.2,
        terminal_weight: float = 0.2,
    ) -> Dict[str, torch.Tensor]:
        hidden, projected, attn_mask, global_condition = self.encode(
            timing_tokens,
            timing_mask,
            phrase_features=phrase_features,
            global_features=global_features,
            event_starts_sec=event_starts_sec,
            event_ends_sec=event_ends_sec,
            audio_durations_sec=audio_durations_sec,
        )
        if timing_targets is not None:
            t_loss = self.compute_timing_loss(
                hidden=hidden,
                timing_targets=timing_targets,
                timing_mask=timing_mask,
                dur_weight=dur_weight,
                onset_weight=onset_weight,
                pause_weight=pause_weight,
                phrase_weight=phrase_weight,
                tempo_weight=tempo_weight,
                terminal_weight=terminal_weight,
            )
        else:
            t_loss = torch.tensor(0.0, device=timing_tokens.device)
        return {
            "hidden": hidden,
            "projected": projected,
            "attn_mask": attn_mask,
            "global_condition": global_condition,
            "timing_loss": t_loss,
        }

    def export_config(self) -> Dict[str, Any]:
        return self.config.to_dict()


class TimingPredictionHead(nn.Module):
    """Shared timing-target predictor used by both encoder and decoder supervision."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.dur_head = nn.Linear(hidden_size, 1)
        self.onset_beat_head = nn.Linear(hidden_size, 1)
        self.pause_exist_head = nn.Linear(hidden_size, 1)
        self.pause_dur_head = nn.Linear(hidden_size, 1)
        self.onset_downbeat_head = nn.Linear(hidden_size, 1)
        self.phrase_head = nn.Linear(hidden_size, 1)
        self.tempo_head = nn.Linear(hidden_size, 1)
        self.terminal_head = nn.Linear(hidden_size, 1)

    def reset_parameters(self) -> None:
        for head in (
            self.dur_head,
            self.onset_beat_head,
            self.pause_exist_head,
            self.pause_dur_head,
            self.onset_downbeat_head,
            self.phrase_head,
            self.tempo_head,
            self.terminal_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def compute_loss(
        self,
        hidden: torch.Tensor,
        timing_targets: torch.Tensor,
        timing_mask: Optional[torch.Tensor] = None,
        dur_weight: float = 1.0,
        onset_weight: float = 0.5,
        pause_weight: float = 0.5,
        phrase_weight: float = 0.3,
        tempo_weight: float = 0.2,
        terminal_weight: float = 0.2,
    ) -> torch.Tensor:
        if timing_targets.shape[-1] == 4:
            dummy = torch.zeros(*timing_targets.shape[:-1], 4, dtype=torch.long, device=timing_targets.device)
            _, timing_targets = upgrade_legacy_timing_tokens(dummy, timing_targets)
            assert timing_targets is not None
        if timing_targets.shape[-1] != len(TIMING_TARGET_NAMES):
            raise ValueError(f"Unsupported timing target width: {timing_targets.shape[-1]}")

        if timing_mask is not None:
            valid = timing_mask.float().unsqueeze(-1)
        else:
            valid = torch.ones(hidden.shape[:2], device=hidden.device).unsqueeze(-1)
        n_valid = valid.sum().clamp(min=1.0)

        pred_dur = self.dur_head(hidden)
        tgt_dur = timing_targets[..., 0:1]
        l_dur = (F.huber_loss(pred_dur, tgt_dur, reduction="none") * valid).sum() / n_valid

        pred_onset_beat = self.onset_beat_head(hidden)
        tgt_onset_beat = timing_targets[..., 1:2]
        l_onset_beat = (F.huber_loss(pred_onset_beat, tgt_onset_beat, reduction="none") * valid).sum() / n_valid

        pred_pause_exist = self.pause_exist_head(hidden)
        tgt_pause_exist = timing_targets[..., 2:3]
        l_pause_exist = (
            F.binary_cross_entropy_with_logits(pred_pause_exist, tgt_pause_exist, reduction="none") * valid
        ).sum() / n_valid

        pred_pause_dur = self.pause_dur_head(hidden)
        tgt_pause_dur = timing_targets[..., 3:4]
        pause_present = (tgt_pause_exist > 0.5).float()
        l_pause_dur = (
            F.huber_loss(pred_pause_dur, tgt_pause_dur, reduction="none") * valid * pause_present
        ).sum() / (valid * pause_present).sum().clamp(min=1.0)
        l_pause = 0.5 * l_pause_exist + 0.5 * l_pause_dur

        pred_onset_downbeat = self.onset_downbeat_head(hidden)
        tgt_onset_downbeat = timing_targets[..., 4:5]
        l_onset_downbeat = (
            F.huber_loss(pred_onset_downbeat, tgt_onset_downbeat, reduction="none") * valid
        ).sum() / n_valid
        l_onset = 0.5 * l_onset_beat + 0.5 * l_onset_downbeat

        pred_phrase = self.phrase_head(hidden)
        tgt_phrase = timing_targets[..., 5:6].clamp(0.0, 2.0) / 2.0
        l_phrase = (F.huber_loss(pred_phrase, tgt_phrase, reduction="none") * valid).sum() / n_valid

        pred_tempo = self.tempo_head(hidden)
        tgt_tempo = timing_targets[..., 6:7]
        l_tempo = (F.huber_loss(pred_tempo, tgt_tempo, reduction="none") * valid).sum() / n_valid

        pred_terminal = self.terminal_head(hidden)
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

class DecoderTimingSupervisor(nn.Module):
    """Decoder-side timing supervisor over pooled output trajectories."""

    def __init__(self, input_dim: int, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        )
        self.prediction_head = TimingPredictionHead(hidden_size)
        nn.init.xavier_uniform_(self.in_proj[0].weight)
        nn.init.zeros_(self.in_proj[0].bias)
        self.prediction_head.reset_parameters()

    def forward(
        self,
        decoder_event_states: torch.Tensor,
        timing_targets: torch.Tensor,
        timing_mask: Optional[torch.Tensor] = None,
        dur_weight: float = 1.0,
        onset_weight: float = 0.5,
        pause_weight: float = 0.5,
        phrase_weight: float = 0.3,
        tempo_weight: float = 0.2,
        terminal_weight: float = 0.2,
    ) -> torch.Tensor:
        hidden = self.in_proj(decoder_event_states)
        return self.prediction_head.compute_loss(
            hidden=hidden,
            timing_targets=timing_targets,
            timing_mask=timing_mask,
            dur_weight=dur_weight,
            onset_weight=onset_weight,
            pause_weight=pause_weight,
            phrase_weight=phrase_weight,
            tempo_weight=tempo_weight,
            terminal_weight=terminal_weight,
        )


def load_saved_timing_module(path: str, map_location: str | torch.device = "cpu") -> dict:
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(f"Timing module checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"Invalid timing module checkpoint at: {path}")
    return payload
