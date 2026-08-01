"""Tests for event-to-frame D2 conditions used by the real C25 gate."""

import pytest
import torch

from acestep.training_v2.phase_d2.config import PhaseD2Config
from acestep.training_v2.phase_d2.regulated_condition import build_regulated_condition


def _events() -> list[dict[str, float | int | bool]]:
    return [
        {
            "duration_sec": 0.2,
            "phrase_duration_sec": 0.6,
            "word_duration_sec": 0.2,
            "note_duration_sec": 0.2,
            "word_id": 0,
            "phoneme_id": 11,
            "f0_hz": 120.0,
            "midi_note": 57.0,
            "energy": 0.6,
            "vowel": True,
            "voiced": True,
        },
        {"duration_sec": 0.2, "silence": True},
        {
            "duration_sec": 0.2,
            "phrase_duration_sec": 0.6,
            "word_duration_sec": 0.2,
            "note_duration_sec": 0.4,
            "word_id": 1,
            "phoneme_id": 12,
            "f0_hz": 140.0,
            "midi_note": 61.0,
            "energy": 0.8,
            "voiced": True,
        },
    ]


def test_regulated_condition_has_exact_frame_budget_and_duration_fields():
    """Expansion is exact and exposes the explicit-duration group."""
    config = PhaseD2Config()
    dense, event_ids, audit = build_regulated_condition(_events(), 15, config)
    assert dense.shape == (1, 15, config.condition_dim)
    assert event_ids.shape == (15,)
    assert audit["target_frames"] == 15
    duration_slice = config.get_condition_group_slices()["explicit_duration"]
    duration_values = dense[0, :, slice(*duration_slice)]
    assert torch.isfinite(duration_values).all()
    assert not torch.equal(duration_values[event_ids == 0], duration_values[event_ids == 2])


def test_regulated_condition_marks_silence_and_event_boundaries():
    """Silence and note boundary channels follow the allocated events."""
    config = PhaseD2Config()
    dense, event_ids, _ = build_regulated_condition(_events(), 15, config)
    groups = config.get_condition_group_slices()
    vocal_start, _ = groups["vocal_activity"]
    melody_start, _ = groups["melody_prosody"]
    silence_frames = event_ids == 1
    assert torch.equal(dense[0, silence_frames, vocal_start:vocal_start + 2], torch.zeros((5, 2)))
    assert dense[0, event_ids == 0, melody_start + 2].sum() == pytest.approx(1.0)
    assert dense[0, event_ids == 0, melody_start + 3].sum() == pytest.approx(1.0)


def test_regulated_condition_rejects_empty_or_nonpositive_events():
    """Invalid event lists fail closed before reaching the real model."""
    config = PhaseD2Config()
    with pytest.raises(ValueError, match="at least one event"):
        build_regulated_condition([], 10, config)
    invalid = _events()
    invalid[0]["duration_sec"] = 0.0
    with pytest.raises(ValueError, match="finite and positive"):
        build_regulated_condition(invalid, 10, config)


def _shifted_events(shift_sec: float) -> list[dict[str, float | bool]]:
    """Build one event on a fixed two-second canvas with adjustable onset."""
    start = 0.75 + shift_sec
    return [
        {"duration_sec": start, "silence": True},
        {
            "duration_sec": 0.5,
            "phrase_duration_sec": 0.5,
            "phrase_start_sec": start,
            "word_id": 1,
            "phoneme_id": 1,
        },
        {"duration_sec": 2.0 - start - 0.5, "silence": True},
    ]


def test_absolute_timing_relocates_for_counterfactual_phrase_shifts():
    """Original, -0.5 s, and +0.5 s controls occupy distinct timing frames."""
    config = PhaseD2Config(latent_frame_rate_hz=20.0)
    timing_start, timing_end = config.get_condition_group_slices()["absolute_timing"]
    locations = []
    timing_values = []
    for shift in (0.0, -0.5, 0.5):
        dense, event_ids, audit = build_regulated_condition(_shifted_events(shift), 40, config)
        event_frames = torch.where(event_ids == 1)[0]
        locations.append((int(event_frames[0]), int(event_frames[-1])))
        timing_values.append(dense[0, event_frames[0], timing_start:timing_end])
        assert audit["absolute_timing_source"].startswith("sidecar_event_start_seconds")
    assert locations == [(15, 24), (5, 14), (25, 34)]
    assert not torch.equal(timing_values[0], timing_values[1])
    assert not torch.equal(timing_values[0], timing_values[2])
