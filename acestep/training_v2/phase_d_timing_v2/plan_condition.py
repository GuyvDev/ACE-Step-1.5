"""Versioned Phase D timing condition schemas.

Two schemas, kept deliberately separate.

PLAN_TIER_V1 is the only schema eligible for promotion. Every channel is
derivable from a timing PLAN alone -- onsets, durations, boundaries, rests,
requested vocal spans -- so a trained controller can be driven at inference for
a song whose performance does not exist yet.

CEILING_V1 adds F0, energy and measured voicing taken from the target's own
lead stem. Those channels are target-derived: the frame-level energy envelope is
essentially the answer to "when is he singing", which is the quantity the timing
gate scores. A model given them can copy a contour instead of inferring timing,
and at inference for a new song they do not exist. CEILING_V1 therefore measures
architectural capacity only and can never be promoted.

Neither schema contains word_id as a continuous scalar, phoneme_id derived as
word_id % 40, constant energy, placeholder MIDI, or beat fields that have no
validated beat grid behind them. Lexical identity already reaches the model
through the ACE text/lyric encoder.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from acestep.training_v2.phase_d2.length_regulator import length_regulate

LATENT_FRAME_RATE_HZ = 25.0

# Channels carried per event, then length-regulated to frames.
PLAN_TIER_V1: tuple[str, ...] = (
    "planned_voiced",          # the plan requests vocal activity here
    "planned_rest",            # the plan requests an explicit rest here
    "pos_in_word",             # 0..1 ramp inside the word
    "pos_in_phrase",           # 0..1 ramp inside the phrase
    "word_duration_norm",      # d / (d + median word duration), non-saturating
    "phrase_duration_norm",    # d / (d + 4 s), non-saturating
    "rest_duration_norm",      # d / (d + 1 s), non-saturating
    "abs_time_norm",           # -1..1 position of the event in the canvas
    "phrase_relative_start",   # -1..1 start of the event within its phrase
    "target_mask",             # inside the edited/target phrase
)
# Frame-level channels added on top for the ceiling arm only.
CEILING_EXTRA_V1: tuple[str, ...] = (
    "f0_norm",                 # RMVPE F0 / 300 Hz, clamped
    "f0_delta",                # first difference of f0_norm
    "energy_norm",             # energy_rms / 0.2, clamped
    "measured_voiced",         # measured voiced fraction
    "avail_acoustics",         # 1 where the acoustic frame is real
)
CEILING_V1: tuple[str, ...] = PLAN_TIER_V1 + CEILING_EXTRA_V1

# Pulses are written after regulation because they are frame events, not event
# features: a boundary exists at one frame, not for a whole word.
PULSE_CHANNELS_V1: tuple[str, ...] = (
    "word_onset_pulse", "word_offset_pulse",
    "phrase_onset_pulse", "phrase_offset_pulse",
)

SCHEMAS: dict[str, tuple[str, ...]] = {
    "plan_tier_v1": PLAN_TIER_V1 + PULSE_CHANNELS_V1,
    "ceiling_v1": PLAN_TIER_V1 + PULSE_CHANNELS_V1 + CEILING_EXTRA_V1,
}


def schema_dim(name: str) -> int:
    return len(SCHEMAS[name])


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def plan_events(words, phrases, rests, total_duration_sec, target_span=None):
    """Ordered plan events covering the whole canvas.

    Words come from the timing plan (edited times for a counterfactual), rests
    from the gaps between them. Nothing here is measured from the target audio.
    """
    if not words:
        raise ValueError("plan condition requires at least one word")
    phrase_of, phrase_span = {}, {}
    for phrase in phrases or []:
        span = (float(phrase["start_sec"]), float(phrase["end_sec"]))
        phrase_span[phrase["phrase_id"]] = span
        for wid in range(int(phrase["first_word_id"]), int(phrase["last_word_id"]) + 1):
            phrase_of[wid] = phrase["phrase_id"]

    durations = [float(w["end_sec"]) - float(w["start_sec"]) for w in words]
    positive = [d for d in durations if d > 0]
    median_word = sorted(positive)[len(positive) // 2] if positive else 1.0

    explicit_rests = [(float(r["start_sec"]), float(r["end_sec"])) for r in (rests or [])]

    def is_explicit_rest(start, end):
        return any(start >= s - 1e-6 and end <= e + 1e-6 for s, e in explicit_rests)

    events, cursor = [], 0.0
    for word in words:
        start = max(float(word["start_sec"]), cursor)
        end = max(float(word["end_sec"]), start + 1.0 / LATENT_FRAME_RATE_HZ)
        if start > cursor + 1e-6:
            events.append({"kind": "rest", "start": cursor, "end": start,
                           "explicit": is_explicit_rest(cursor, start)})
        pid = phrase_of.get(word["word_id"])
        span = phrase_span.get(pid, (start, end))
        events.append({"kind": "word", "start": start, "end": end,
                       "word_duration": end - start, "median_word": median_word,
                       "phrase_start": span[0], "phrase_end": span[1]})
        cursor = end
    if total_duration_sec > cursor + 1e-6:
        events.append({"kind": "rest", "start": cursor, "end": float(total_duration_sec),
                       "explicit": is_explicit_rest(cursor, total_duration_sec)})

    for event in events:
        event["duration_sec"] = event["end"] - event["start"]
        event["in_target"] = bool(
            target_span is not None
            and event["end"] > target_span[0] + 1e-6
            and event["start"] < target_span[1] - 1e-6)
    return events


def _event_row(event, total_duration_sec) -> list[float]:
    """PLAN_TIER_V1 channels for one event. Ramps are filled after regulation."""
    is_word = event["kind"] == "word"
    if is_word:
        phrase_len = max(event["phrase_end"] - event["phrase_start"], 1e-6)
        phrase_rel = (event["start"] - event["phrase_start"]) / phrase_len
        # Saturating ratios collapse: with every phrase longer than the scale,
        # phrase_duration_norm pinned to 1.0 for words and 0.0 for rests, making
        # it an exact duplicate of planned_voiced. These forms are smooth on
        # (0, 1) and never saturate.
        word_norm = event["word_duration"] / (event["word_duration"] + max(event["median_word"], 1e-6))
        phrase_norm = phrase_len / (phrase_len + 4.0)
        rest_norm = 0.0
    else:
        phrase_rel, word_norm, phrase_norm = 0.0, 0.0, 0.0
        rest_norm = event["duration_sec"] / (event["duration_sec"] + 1.0)
    return [
        1.0 if is_word else 0.0,                                        # planned_voiced
        1.0 if (not is_word and event.get("explicit")) else 0.0,        # planned_rest
        0.0,                                                            # pos_in_word (filled later)
        0.0,                                                            # pos_in_phrase (filled later)
        word_norm,
        phrase_norm,
        rest_norm,
        _clamp(2.0 * event["start"] / max(total_duration_sec, 1e-6) - 1.0, -1.0, 1.0),
        _clamp(2.0 * phrase_rel - 1.0, -1.0, 1.0),
        1.0 if event["in_target"] else 0.0,
    ]


def build_plan_condition(words, phrases, rests, total_duration_sec, target_frames,
                         target_span=None, acoustics=None, schema="plan_tier_v1"):
    """Dense [1, T, C] condition plus a per-channel audit.

    acoustics is accepted only by the ceiling schema and must come from the
    target's own lead stem; passing it to the plan tier is an error rather than
    a silently ignored argument.
    """
    if schema not in SCHEMAS:
        raise ValueError(f"unknown condition schema {schema!r}")
    if schema == "plan_tier_v1" and acoustics is not None:
        raise ValueError(
            "plan_tier_v1 must never receive target-derived acoustics; "
            "use schema='ceiling_v1' for the diagnostic arm")
    if schema == "ceiling_v1" and acoustics is None:
        raise ValueError("ceiling_v1 requires target-derived acoustics")

    events = plan_events(words, phrases, rests, total_duration_sec, target_span)
    features = torch.tensor([_event_row(e, total_duration_sec) for e in events],
                            dtype=torch.float32)
    durations = torch.tensor([e["duration_sec"] for e in events], dtype=torch.float32)
    dense, event_ids, audit = length_regulate(
        features, durations, LATENT_FRAME_RATE_HZ, target_frames)

    names = list(PLAN_TIER_V1)
    pos_in_word = names.index("pos_in_word")
    pos_in_phrase = names.index("pos_in_phrase")

    # Ramps: position inside the event, and inside its phrase.
    starts = torch.cat([torch.tensor([True]), event_ids[1:] != event_ids[:-1]])
    boundary_idx = torch.nonzero(starts).flatten().tolist() + [len(event_ids)]
    for i in range(len(boundary_idx) - 1):
        lo, hi = boundary_idx[i], boundary_idx[i + 1]
        span = hi - lo
        if span > 0:
            ramp = torch.arange(span, dtype=torch.float32) / max(span - 1, 1)
            dense[lo:hi, pos_in_word] = ramp
    # phrase ramp: reuse phrase_relative_start already encoded per event, then
    # linearly interpolate inside the phrase using absolute frame position.
    dense[:, pos_in_phrase] = torch.linspace(0.0, 1.0, dense.shape[0])

    # Frame-level pulses.
    pulses = torch.zeros((dense.shape[0], len(PULSE_CHANNELS_V1)), dtype=torch.float32)
    voiced_col = names.index("planned_voiced")
    voiced = dense[:, voiced_col]
    word_start = (voiced > 0.5) & torch.cat([torch.tensor([True]), voiced[:-1] <= 0.5])
    word_end = (voiced > 0.5) & torch.cat([voiced[1:] <= 0.5, torch.tensor([True])])
    pulses[word_start, 0] = 1.0
    pulses[word_end, 1] = 1.0
    for phrase in phrases or []:
        for col, sec in ((2, float(phrase["start_sec"])), (3, float(phrase["end_sec"]))):
            frame = min(max(int(round(sec * LATENT_FRAME_RATE_HZ)), 0), dense.shape[0] - 1)
            pulses[frame, col] = 1.0
    dense = torch.cat([dense, pulses], dim=1)

    if schema == "ceiling_v1":
        dense = torch.cat([dense, _acoustic_frames(acoustics, dense.shape[0])], dim=1)

    audit.update({
        "schema": schema,
        "channels": list(SCHEMAS[schema]),
        "condition_dim": dense.shape[1],
        "event_count": len(events),
        "word_events": sum(1 for e in events if e["kind"] == "word"),
        "rest_events": sum(1 for e in events if e["kind"] == "rest"),
        "target_span_sec": list(target_span) if target_span else None,
        "target_frames_marked": int(dense[:, names.index("target_mask")].sum()),
    })
    if dense.shape[1] != schema_dim(schema):
        raise RuntimeError(
            f"{schema} produced {dense.shape[1]} channels, expected {schema_dim(schema)}")
    return dense.unsqueeze(0), event_ids, audit


def _acoustic_frames(acoustics: dict[str, Any], frames: int) -> torch.Tensor:
    """Target-derived F0/energy/voicing, resampled to the latent frame count."""
    def resample(values):
        tensor = torch.tensor(values, dtype=torch.float32).view(1, 1, -1)
        if tensor.shape[-1] == frames:
            return tensor.view(-1)
        return torch.nn.functional.interpolate(
            tensor, size=frames, mode="linear", align_corners=False).view(-1)

    f0 = resample(acoustics["f0_hz"])
    energy = resample(acoustics["energy_rms"])
    voiced = resample(acoustics["voiced_fraction"])
    f0_norm = (f0 / 300.0).clamp(0.0, 1.0)
    f0_delta = torch.cat([torch.zeros(1), f0_norm[1:] - f0_norm[:-1]]).clamp(-1.0, 1.0)
    energy_norm = (energy / 0.2).clamp(0.0, 1.0)
    available = (f0 > 0).float()
    return torch.stack([f0_norm, f0_delta, energy_norm,
                        voiced.clamp(0.0, 1.0), available], dim=1)


def boundary_mask(target_span, frames, band_frames: int = 4) -> torch.Tensor:
    """[T, 1] band immediately OUTSIDE each target edge, ~160 ms at 25 Hz.

    Clipped to the canvas, never overlapping the target mask, and the two bands
    never overlap each other. A nonzero-weight loss must not sit on an all-zero
    mask, so this is derived deterministically from the target edges.
    """
    mask = torch.zeros((frames, 1), dtype=torch.float32)
    lo = max(int(math.floor(target_span[0] * LATENT_FRAME_RATE_HZ)), 0)
    hi = min(int(math.ceil(target_span[1] * LATENT_FRAME_RATE_HZ)), frames)
    if hi <= lo:
        raise ValueError(f"empty target span {target_span} over {frames} frames")
    left_lo, left_hi = max(lo - band_frames, 0), lo
    right_lo, right_hi = hi, min(hi + band_frames, frames)
    if left_hi > left_lo:
        mask[left_lo:left_hi] = 1.0
    if right_hi > right_lo:
        mask[right_lo:right_hi] = 1.0
    if bool((mask[lo:hi] > 0).any()):
        raise RuntimeError("boundary band overlaps the target mask")
    return mask


def target_mask_from_span(target_span, frames) -> torch.Tensor:
    mask = torch.zeros((frames, 1), dtype=torch.float32)
    lo = max(int(math.floor(target_span[0] * LATENT_FRAME_RATE_HZ)), 0)
    hi = min(int(math.ceil(target_span[1] * LATENT_FRAME_RATE_HZ)), frames)
    if hi <= lo:
        raise ValueError(f"empty target span {target_span} over {frames} frames")
    mask[lo:hi] = 1.0
    return mask


__all__ = [
    "PLAN_TIER_V1", "CEILING_EXTRA_V1", "CEILING_V1", "PULSE_CHANNELS_V1",
    "SCHEMAS", "schema_dim", "plan_events", "build_plan_condition",
    "boundary_mask", "target_mask_from_span", "LATENT_FRAME_RATE_HZ",
]
