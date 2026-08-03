"""Contract tests for the two-tier Phase D timing condition."""

from __future__ import annotations

import pytest
import torch

from acestep.training_v2.phase_d_timing_v2.plan_condition import (
    CEILING_V1, PLAN_TIER_V1, PULSE_CHANNELS_V1, SCHEMAS, boundary_mask,
    build_plan_condition, plan_events, schema_dim, target_mask_from_span,
)

WORDS = [
    {"word_id": 0, "start_sec": 1.0, "end_sec": 1.4},
    {"word_id": 1, "start_sec": 1.4, "end_sec": 1.9},
    {"word_id": 2, "start_sec": 2.5, "end_sec": 3.0},
    {"word_id": 3, "start_sec": 3.0, "end_sec": 3.6},
]
PHRASES = [
    {"phrase_id": 0, "start_sec": 1.0, "end_sec": 1.9, "first_word_id": 0, "last_word_id": 1},
    {"phrase_id": 1, "start_sec": 2.5, "end_sec": 3.6, "first_word_id": 2, "last_word_id": 3},
]
RESTS = [{"start_sec": 1.9, "end_sec": 2.5, "duration_sec": 0.6}]
TOTAL, FRAMES = 4.0, 100
SPAN = (2.5, 3.6)


def acoustics(n=100):
    return {"f0_hz": [120.0] * n, "energy_rms": [0.1] * n, "voiced_fraction": [1.0] * n}


def test_schema_widths_are_versioned_and_distinct():
    assert schema_dim("plan_tier_v1") == len(PLAN_TIER_V1) + len(PULSE_CHANNELS_V1)
    assert schema_dim("ceiling_v1") == schema_dim("plan_tier_v1") + 5
    assert SCHEMAS["plan_tier_v1"] != SCHEMAS["ceiling_v1"]


def test_plan_tier_rejects_target_derived_acoustics():
    with pytest.raises(ValueError, match="never receive target-derived"):
        build_plan_condition(WORDS, PHRASES, RESTS, TOTAL, FRAMES,
                             target_span=SPAN, acoustics=acoustics(),
                             schema="plan_tier_v1")


def test_ceiling_requires_acoustics():
    with pytest.raises(ValueError, match="requires target-derived"):
        build_plan_condition(WORDS, PHRASES, RESTS, TOTAL, FRAMES,
                             target_span=SPAN, schema="ceiling_v1")


def test_plan_tier_contains_no_banned_channel():
    banned = {"word_id", "phoneme_id", "midi_note", "beat_offset_sec",
              "bar_position", "local_tempo", "f0_norm", "energy_norm",
              "measured_voiced"}
    assert not banned & set(SCHEMAS["plan_tier_v1"])


def test_ceiling_is_a_strict_superset_of_plan_tier():
    assert set(SCHEMAS["plan_tier_v1"]) < set(SCHEMAS["ceiling_v1"])


def test_shape_and_frame_count():
    dense, _, audit = build_plan_condition(WORDS, PHRASES, RESTS, TOTAL, FRAMES,
                                           target_span=SPAN)
    assert dense.shape == (1, FRAMES, schema_dim("plan_tier_v1"))
    assert audit["condition_dim"] == schema_dim("plan_tier_v1")
    assert audit["word_events"] == 4


def test_rests_are_events_and_explicit_rest_is_flagged():
    events = plan_events(WORDS, PHRASES, RESTS, TOTAL, SPAN)
    rests = [e for e in events if e["kind"] == "rest"]
    assert any(e["explicit"] for e in rests)
    assert sum(e["duration_sec"] for e in events) == pytest.approx(TOTAL, abs=1e-6)


def test_events_cover_canvas_without_gaps_or_overlap():
    events = plan_events(WORDS, PHRASES, RESTS, TOTAL, SPAN)
    for a, b in zip(events, events[1:]):
        assert a["end"] == pytest.approx(b["start"], abs=1e-9)
    assert events[0]["start"] == pytest.approx(0.0)
    assert events[-1]["end"] == pytest.approx(TOTAL)


def test_target_mask_marks_only_the_target_span():
    dense, _, audit = build_plan_condition(WORDS, PHRASES, RESTS, TOTAL, FRAMES,
                                           target_span=SPAN)
    col = list(PLAN_TIER_V1).index("target_mask")
    marked = dense[0, :, col]
    assert audit["target_frames_marked"] > 0
    assert marked[:int(SPAN[0] * 25) - 1].sum() == 0


def test_shifted_plan_changes_the_condition():
    shifted = [{**w, "start_sec": w["start_sec"] - 0.5, "end_sec": w["end_sec"] - 0.5}
               if w["word_id"] >= 2 else w for w in WORDS]
    a, _, _ = build_plan_condition(WORDS, PHRASES, RESTS, TOTAL, FRAMES, target_span=SPAN)
    b, _, _ = build_plan_condition(shifted, PHRASES, RESTS, TOTAL, FRAMES, target_span=SPAN)
    assert not torch.equal(a, b)


def test_no_channel_is_constant_across_a_realistic_plan():
    dense, _, _ = build_plan_condition(WORDS, PHRASES, RESTS, TOTAL, FRAMES,
                                       target_span=SPAN)
    stds = dense[0].std(dim=0)
    constant = [SCHEMAS["plan_tier_v1"][i] for i, s in enumerate(stds) if float(s) == 0.0]
    assert constant == [], f"constant plan-tier channels: {constant}"


def test_ceiling_adds_exactly_the_acoustic_channels():
    plan, _, _ = build_plan_condition(WORDS, PHRASES, RESTS, TOTAL, FRAMES, target_span=SPAN)
    ceiling, _, _ = build_plan_condition(WORDS, PHRASES, RESTS, TOTAL, FRAMES,
                                         target_span=SPAN, acoustics=acoustics(),
                                         schema="ceiling_v1")
    assert torch.equal(plan[0], ceiling[0, :, :plan.shape[2]])
    assert ceiling.shape[2] - plan.shape[2] == 5


def test_acoustics_are_resampled_to_the_frame_count():
    _, _, audit = build_plan_condition(WORDS, PHRASES, RESTS, TOTAL, FRAMES,
                                       target_span=SPAN, acoustics=acoustics(37),
                                       schema="ceiling_v1")
    assert audit["condition_dim"] == schema_dim("ceiling_v1")


def test_boundary_band_is_outside_the_target_and_non_overlapping():
    target = target_mask_from_span(SPAN, FRAMES)
    band = boundary_mask(SPAN, FRAMES, band_frames=4)
    assert float((target * band).sum()) == 0.0
    assert float(band.sum()) == 8.0


def test_boundary_band_clips_at_canvas_edges():
    band = boundary_mask((0.0, 1.0), FRAMES, band_frames=4)
    assert float(band[:1].sum()) == 0.0
    assert float(band.sum()) == 4.0


def test_boundary_band_rejects_empty_span():
    with pytest.raises(ValueError, match="empty target span"):
        boundary_mask((2.0, 2.0), FRAMES)


def test_unknown_schema_is_rejected():
    with pytest.raises(ValueError, match="unknown condition schema"):
        build_plan_condition(WORDS, PHRASES, RESTS, TOTAL, FRAMES, schema="nope")


def test_empty_word_list_is_rejected():
    with pytest.raises(ValueError, match="at least one word"):
        plan_events([], PHRASES, RESTS, TOTAL)
