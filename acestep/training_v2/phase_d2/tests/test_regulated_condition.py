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
