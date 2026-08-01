"""Duration-regulated, example-specific conditions for the D2 control path."""

from __future__ import annotations

from typing import Any

import torch

from acestep.training_v2.phase_d2.config import PhaseD2Config
from acestep.training_v2.phase_d2.length_regulator import length_regulate


def _event_feature(
    event: dict[str, Any],
    config: PhaseD2Config,
    event_start_sec: float,
    total_duration_sec: float,
) -> list[float]:
    """Map one ordered sidecar event and its real onset into the D2 schema."""
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
        phrase_start = float(event.get("phrase_start_sec", 0.0))
        phrase_relative_start = (event_start_sec - phrase_start) / phrase_duration
        values.extend([
            max(-1.0, min(1.0, 2.0 * event_start_sec / total_duration_sec - 1.0)),
            max(-1.0, min(1.0, phrase_relative_start)),
        ])
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
            0.0,
            0.0,
            float(event.get("vowel", False)),
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
            float(event.get("section_id", 0)) / 50.0,
            float(event.get("phrase_id", 0)) / 200.0 if not silence else -0.005,
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
    """Expand ordered events to a global frame budget with real timing channels."""
    if not events:
        raise ValueError("regulated condition requires at least one event")
    durations = torch.tensor([float(event["duration_sec"]) for event in events])
    if not bool(torch.isfinite(durations).all()) or bool((durations <= 0).any()):
        raise ValueError("event durations must be finite and positive")
    total_duration = float(durations.sum())
    event_starts = torch.cat([torch.zeros(1), durations.cumsum(0)[:-1]])
    features = torch.tensor([
        _event_feature(event, config, float(start), total_duration)
        for event, start in zip(events, event_starts)
    ])
    if features.shape[1] != config.condition_dim:
        raise RuntimeError("regulated feature width differs from D2 config")
    dense, event_ids, audit = length_regulate(
        features,
        durations,
        config.latent_frame_rate_hz,
        target_frames,
    )
    if config.melody_prosody:
        start, _ = config.get_condition_group_slices()["melody_prosody"]
        boundaries = torch.cat([torch.tensor([True]), event_ids[1:] != event_ids[:-1]])
        ends = torch.cat([event_ids[:-1] != event_ids[1:], torch.tensor([True])])
        dense[boundaries, start + 2] = 1.0
        dense[ends, start + 3] = 1.0
    audit.update({
        "schema": "phase_d2_regulated_condition_v2",
        "condition_dim": config.condition_dim,
        "event_count": len(events),
        "duration_sum_sec": total_duration,
        "event_start_seconds": event_starts.tolist(),
        "absolute_timing_source": "sidecar_event_start_seconds_and_phrase_relative_start",
        "overlap_frame_count": 0,
    })
    return dense.unsqueeze(0), event_ids, audit
