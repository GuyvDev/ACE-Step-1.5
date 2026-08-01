"""Duration-regulated event conditions for the established D2 control path."""

from __future__ import annotations

from typing import Any

import torch

from acestep.training_v2.phase_d2.config import PhaseD2Config
from acestep.training_v2.phase_d2.length_regulator import length_regulate


def _event_feature(event: dict[str, Any], config: PhaseD2Config) -> list[float]:
    """Map one phoneme/note or silence event into the configured schema."""
    silence = bool(event.get("silence", False))
    duration = float(event["duration_sec"])
    note_duration = max(float(event.get("note_duration_sec", duration)), 1e-6)
    phrase_duration = max(float(event.get("phrase_duration_sec", duration)), 1e-6)
    word_duration = max(float(event.get("word_duration_sec", duration)), 1e-6)
    values: list[float] = []
    if config.linguistic:
        values.extend([
            float(event.get("word_id", -1)) / 100.0,
            float(event.get("phoneme_id", -1)) / 100.0,
            float(event.get("pos_in_word", 0.0)),
            float(event.get("pos_in_phrase", 0.0)),
        ])
    if config.absolute_timing:
        values.extend([0.0, 0.0])
    if config.beat_relative_timing:
        values.extend([
            max(-1.0, min(1.0, float(event.get("beat_offset_sec", 0.0)) / 0.5)),
            float(event.get("bar_position", 0.0)) / 4.0,
            max(-1.0, min(1.0, 2.0 * float(event.get("local_tempo", 1.0)) - 1.0)),
        ])
    if config.melody_prosody:
        values.extend([
            max(0.0, min(1.0, float(event.get("f0_hz", 0.0)) / 300.0)),
            max(-1.0, min(1.0, float(event.get("midi_note", 0.0)) / 60.0 - 1.0)),
            0.0, 0.0, float(event.get("vowel", False)),
        ])
    if config.breath_energy:
        values.extend([
            max(-1.0, min(1.0, 2.0 * float(event.get("energy", 0.0)) - 1.0)),
            float(event.get("breath", False)),
        ])
    if config.vocal_activity:
        values.extend([float(not silence and event.get("voiced", True)), float(not silence)])
    if config.structure:
        values.extend([
            0.0,
            0.0 if not silence else -0.005,
            float(event.get("word_id", -1)) / 100.0,
            float(silence),
        ])
    if config.explicit_duration:
        values.extend([
            min(1.0, phrase_duration / 15.0),
            min(1.0, word_duration / 5.0),
            min(1.0, duration / 2.0),
            min(2.0, duration / note_duration) - 1.0,
        ])
    return values


def build_regulated_condition(
    events: list[dict[str, Any]],
    target_frames: int,
    config: PhaseD2Config,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Expand ordered positive-duration events to exactly ``target_frames``."""
    if not events:
        raise ValueError("regulated condition requires at least one event")
    durations = torch.tensor([float(event["duration_sec"]) for event in events])
    features = torch.tensor([_event_feature(event, config) for event in events])
    if features.shape[1] != config.condition_dim:
        raise RuntimeError("regulated feature width differs from D2 config")
    dense, event_ids, audit = length_regulate(
        features,
        durations,
        config.latent_frame_rate_hz,
        target_frames,
    )
    frame_time = torch.arange(target_frames, dtype=dense.dtype) / max(target_frames - 1, 1)
    if config.absolute_timing:
        start, _ = config.get_condition_group_slices()["absolute_timing"]
        dense[:, start] = 2.0 * frame_time - 1.0
        dense[:, start + 1] = frame_time
    if config.melody_prosody:
        start, _ = config.get_condition_group_slices()["melody_prosody"]
        boundaries = torch.cat([torch.tensor([True]), event_ids[1:] != event_ids[:-1]])
        ends = torch.cat([event_ids[:-1] != event_ids[1:], torch.tensor([True])])
        dense[boundaries, start + 2] = 1.0
        dense[ends, start + 3] = 1.0
    audit.update({
        "schema": "phase_d2_regulated_condition_v1",
        "condition_dim": config.condition_dim,
        "event_count": len(events),
        "duration_sum_sec": float(durations.sum()),
        "overlap_frame_count": 0,
    })
    return dense.unsqueeze(0), event_ids, audit
