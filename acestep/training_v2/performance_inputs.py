from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Optional, Sequence

import torch

from acestep.training_v2.timing_conditioning import quantise_beat_phase


PREDICTOR_FEATURE_NAMES = (
    "phoneme_id",
    "note_pitch_bin",
    "note_dur_bin",
    "beat_phase_bin",
    "downbeat_flag_bin",
    "word_boundary_bin",
    "clause_boundary_bin",
    "phrase_pos_bin",
    "section_tag_bin",
    "style_tag_bin",
)

PREDICTOR_FEATURE_VOCAB_SIZES = (
    512,
    130,
    32,
    8,
    2,
    2,
    3,
    16,
    8,
    8,
)

SECTION_TAG_TO_ID = {
    "unknown": 0,
    "intro": 1,
    "verse": 2,
    "pre": 3,
    "chorus": 4,
    "bridge": 5,
    "outro": 6,
    "tag": 7,
}

STYLE_TAG_TO_ID = {
    "unknown": 0,
    "slow": 1,
    "moderate": 2,
    "fast": 3,
    "laid_back": 4,
    "tight": 5,
    "speech_like": 6,
    "legato": 7,
}

NOTE_DUR_LOG_MIN = math.log(0.03)
NOTE_DUR_LOG_MAX = math.log(4.0)
PHRASE_POS_BINS = 16
SECTION_RE = re.compile(r"^\s*[\[(<]?\s*(intro|verse|pre[- ]?chorus|chorus|bridge|outro|tag)\s*[\])>]?\s*$", re.I)


def _hash_token(text: str, vocab_size: int) -> int:
    if vocab_size <= 1:
        return 0
    token = (text or "").strip().lower()
    if not token:
        return 0
    digest = hashlib.md5(token.encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], "big")
    return 1 + (value % (vocab_size - 1))


def quantise_note_duration(duration_s: float) -> int:
    duration_s = max(float(duration_s), 1e-4)
    log_value = math.log(duration_s)
    frac = (min(max(log_value, NOTE_DUR_LOG_MIN), NOTE_DUR_LOG_MAX) - NOTE_DUR_LOG_MIN) / max(
        NOTE_DUR_LOG_MAX - NOTE_DUR_LOG_MIN,
        1e-6,
    )
    return int(frac * (PREDICTOR_FEATURE_VOCAB_SIZES[2] - 1) + 0.5)


def quantise_phrase_position(progress: float) -> int:
    progress = max(0.0, min(0.9999, float(progress)))
    return int(progress * (PHRASE_POS_BINS - 1) + 0.5)


def style_tag_to_id(value: Any) -> int:
    if value is None:
        return 0
    normalized = str(value).strip().lower().replace(" ", "_")
    if not normalized:
        return 0
    normalized = normalized.replace("pre_chorus", "pre")
    return STYLE_TAG_TO_ID.get(normalized, _hash_token(normalized, PREDICTOR_FEATURE_VOCAB_SIZES[-1]))


def section_tag_to_id(value: Any) -> int:
    if value is None:
        return 0
    normalized = str(value).strip().lower().replace(" ", "")
    if normalized in {"prechorus", "pre-chorus"}:
        normalized = "pre"
    return SECTION_TAG_TO_ID.get(normalized, 0)


def infer_global_section_tag(lyrics_text: str) -> int:
    if not lyrics_text:
        return 0
    for line in str(lyrics_text).splitlines():
        match = SECTION_RE.match(line.strip())
        if match:
            token = match.group(1).lower().replace(" ", "").replace("-", "")
            if token == "prechorus":
                token = "pre"
            return SECTION_TAG_TO_ID.get(token, 0)
    return 0


def _nearest_anchor(time_s: float, anchors: Sequence[float]) -> tuple[float, float]:
    if not anchors:
        return 0.0, 0.5
    if len(anchors) == 1:
        return float(anchors[0]), 0.5
    best_idx = min(range(len(anchors)), key=lambda i: abs(float(anchors[i]) - time_s))
    anchor = float(anchors[best_idx])
    if best_idx == 0:
        period = max(float(anchors[1]) - anchor, 0.1)
    elif best_idx == len(anchors) - 1:
        period = max(anchor - float(anchors[best_idx - 1]), 0.1)
    else:
        left = anchor - float(anchors[best_idx - 1])
        right = float(anchors[best_idx + 1]) - anchor
        period = max(0.5 * (left + right), 0.1)
    return anchor, period


def _event_group_last_indices(events: Sequence[dict[str, Any]]) -> set[int]:
    if not events:
        return set()
    last_indices: set[int] = set()
    current_key = None
    for idx, ev in enumerate(events):
        word_idx = int(ev.get("word_index", -1))
        key = ("word_index", word_idx) if word_idx >= 0 else ("word", str(ev.get("word", ev.get("label", ""))).strip().lower())
        if current_key is None:
            current_key = key
            continue
        if key != current_key:
            last_indices.add(idx - 1)
            current_key = key
    last_indices.add(len(events) - 1)
    return last_indices


def build_predictor_inputs_from_events(
    events: Sequence[dict[str, Any]],
    *,
    beat_times: Optional[Sequence[float]] = None,
    downbeat_times: Optional[Sequence[float]] = None,
    lyrics_text: str = "",
    global_style: Optional[dict[str, Any]] = None,
) -> torch.Tensor:
    if not events:
        return torch.zeros(0, len(PREDICTOR_FEATURE_NAMES), dtype=torch.long)

    beats = [float(x) for x in (beat_times or [])]
    downbeats = [float(x) for x in (downbeat_times or [])] or beats
    section_tag = infer_global_section_tag(lyrics_text)
    if not section_tag:
        for ev in events:
            section_tag = section_tag_to_id(ev.get("section_tag"))
            if section_tag:
                break
    style_tag = style_tag_to_id((global_style or {}).get("pace"))
    if not style_tag:
        for ev in events:
            style_tag = style_tag_to_id(ev.get("style_tag"))
            if style_tag:
                break

    word_end_indices = _event_group_last_indices(events)
    phrase_groups: list[list[int]] = []
    current_group: list[int] = []
    for idx, ev in enumerate(events):
        current_group.append(idx)
        if int(ev.get("phrase_boundary", 0)) > 0:
            phrase_groups.append(current_group)
            current_group = []
    if current_group:
        phrase_groups.append(current_group)

    phrase_progress = [0.0] * len(events)
    for phrase in phrase_groups:
        denom = max(len(phrase) - 1, 1)
        for local_idx, event_idx in enumerate(phrase):
            phrase_progress[event_idx] = local_idx / denom

    rows = []
    for idx, ev in enumerate(events):
        start = float(ev.get("start", 0.0))
        end = float(ev.get("end", start))
        midpoint = 0.5 * (start + end)
        beat_anchor, beat_period = _nearest_anchor(start, beats)
        downbeat_anchor, downbeat_period = _nearest_anchor(start, downbeats)
        beat_phase = (start - beat_anchor) / max(beat_period, 1e-4)
        beat_phase = beat_phase - math.floor(beat_phase)
        is_downbeat = 1 if abs(start - downbeat_anchor) <= 0.15 * max(downbeat_period, 1e-4) else 0

        note_pitch = int(ev.get("note_pitch_midi", -1))
        if note_pitch < 0:
            note_pitch_bin = 0
        else:
            note_pitch_bin = max(1, min(129, note_pitch + 1))

        note_duration = float(ev.get("note_duration", max(end - start, 1e-4)))
        clause_boundary = max(0, min(2, int(ev.get("phrase_boundary", 0))))
        rows.append(
            [
                _hash_token(str(ev.get("label", "")), PREDICTOR_FEATURE_VOCAB_SIZES[0]),
                note_pitch_bin,
                quantise_note_duration(note_duration),
                quantise_beat_phase(beat_phase),
                is_downbeat,
                1 if idx in word_end_indices else 0,
                clause_boundary,
                quantise_phrase_position(phrase_progress[idx]),
                section_tag,
                style_tag,
            ]
        )
    return torch.tensor(rows, dtype=torch.long)
