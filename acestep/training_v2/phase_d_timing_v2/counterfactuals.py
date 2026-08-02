"""
Counterfactual Generation for Phase D2.

Create modified event sequences for intervention studies (shifts, stretches, etc.).
"""

from __future__ import annotations

from typing import Dict, Any, Literal, Tuple
import copy
import numpy as np


_EVENT_LOCAL_FIELDS = (
    "section_id", "phrase_id", "word_id", "phoneme_id",
    "pos_in_phrase", "pos_in_word", "pos_in_phoneme",
    "word_start_pulse", "word_end_pulse",
    "phrase_start_pulse", "phrase_end_pulse", "vowel_nucleus",
    "target_phrase_duration", "target_word_duration",
    "target_phoneme_duration", "target_f0", "target_note",
    "note_onset", "note_end", "voiced", "energy", "breath",
    "pause", "vocal_active", "rubato_offset",
)


def _silence_value(key: str) -> float | int:
    if key in {"section_id", "phrase_id", "word_id", "phoneme_id"}:
        return -1
    if key == "pause":
        return 1.0
    return 0.0


def _validate_span(sidecar: Dict[str, Any], phrase_span: tuple[int, int]) -> tuple[int, int, int]:
    n_frames = len(sidecar["absolute_time"])
    start_frame, end_frame = phrase_span
    if not (0 <= start_frame < end_frame <= n_frames):
        raise ValueError(
            f"phrase_span must satisfy 0 <= start < end <= {n_frames}; "
            f"got {phrase_span}."
        )
    for key in _EVENT_LOCAL_FIELDS:
        if key in sidecar and len(sidecar[key]) != n_frames:
            raise ValueError(f"Field {key!r} has inconsistent frame count")
    return start_frame, end_frame, n_frames


def make_counterfactual(
    sidecar_events: Dict[str, Any],
    phrase_span: tuple[int, int],
    kind: Literal["shift", "stretch", "f0", "pause"],
    value: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Create a modified sidecar with counterfactual intervention.

    Args:
        sidecar_events: Original sidecar dict with per-frame fields.
        phrase_span: (start_frame, end_frame) indices for intervention window.
        kind: Type of intervention:
            - "shift": temporal shift of events in seconds
            - "stretch": duration scaling factor (1.0 = no change)
            - "f0": add to target F0 (Hz or semitones)
            - "pause": insert pause of given duration (seconds)
        value: Magnitude of change
            - shift: ±0.25, ±0.5, ±1.0, ±2.0 seconds
            - stretch: 0.8, 0.9, 1.1, 1.25
            - f0: ±100 Hz or ±2 semitones
            - pause: 0.5, 1.0, 2.0 seconds

    Returns:
        Tuple of:
        - modified_sidecar: Copied and modified dict
        - intervention_record: {kind, value, expected_direction, phrase_span}
    """

    modified = copy.deepcopy(sidecar_events)
    start_frame, end_frame, n_frames = _validate_span(sidecar_events, phrase_span)
    frame_rate = float(sidecar_events.get("metadata", {}).get("frame_rate", 0.0))
    if frame_rate <= 0:
        times = np.asarray(sidecar_events["absolute_time"], dtype=np.float64)
        if len(times) < 2 or np.any(np.diff(times) <= 0):
            raise ValueError("Cannot infer a valid frame rate from sidecar")
        frame_rate = 1.0 / float(np.median(np.diff(times)))

    record = {
        "kind": kind,
        "value": value,
        "phrase_span": phrase_span,
        "expected_direction": None,
    }

    if kind == "shift":
        shift_frames = int(round(value * frame_rate))
        if value != 0 and shift_frames == 0:
            raise ValueError(
                f"Shift {value}s is below one frame at {frame_rate} Hz"
            )
        dst_start = start_frame + shift_frames
        dst_end = end_frame + shift_frames
        if dst_start < 0 or dst_end > n_frames:
            raise ValueError(
                f"Shift would truncate the phrase: destination [{dst_start}, {dst_end}) "
                f"outside [0, {n_frames})."
            )

        for key in _EVENT_LOCAL_FIELDS:
            if key not in modified:
                continue
            source = np.asarray(sidecar_events[key])[start_frame:end_frame].copy()
            modified[key][start_frame:end_frame] = _silence_value(key)
            modified[key][dst_start:dst_end] = source

        record["source_span"] = [start_frame, end_frame]
        record["destination_span"] = [dst_start, dst_end]
        record["realized_shift_frames"] = shift_frames
        record["realized_shift_sec"] = shift_frames / frame_rate
        record["expected_direction"] = "earlier" if value < 0 else "later"

    elif kind == "stretch":
        if value <= 0:
            raise ValueError(f"stretch factor must be > 0, got {value}")
        source_len = end_frame - start_frame
        stretched_len = max(1, int(round(source_len * value)))
        dst_end = start_frame + stretched_len
        if dst_end > n_frames:
            raise ValueError(
                f"Stretch would truncate the phrase at frame {dst_end} > {n_frames}"
            )
        sample_idx = np.rint(
            np.linspace(0, source_len - 1, stretched_len)
        ).astype(np.int64)
        clear_end = max(end_frame, dst_end)
        for key in _EVENT_LOCAL_FIELDS:
            if key not in modified:
                continue
            source = np.asarray(sidecar_events[key])[start_frame:end_frame].copy()
            modified[key][start_frame:clear_end] = _silence_value(key)
            modified[key][start_frame:dst_end] = source[sample_idx]

        for key in ("target_phrase_duration", "target_word_duration", "target_phoneme_duration"):
            if key in modified:
                modified[key][start_frame:dst_end] *= value
        for key in ("pos_in_phrase", "pos_in_word"):
            if key in modified:
                modified[key][start_frame:dst_end] = np.linspace(
                    0.0, 1.0, stretched_len, dtype=np.float32
                )
        if "local_tempo" in modified:
            original_tempo = np.asarray(sidecar_events["local_tempo"])[start_frame:end_frame]
            modified["local_tempo"][start_frame:dst_end] = original_tempo[sample_idx] / value

        record["source_span"] = [start_frame, end_frame]
        record["destination_span"] = [start_frame, dst_end]
        record["realized_stretch_factor"] = stretched_len / source_len
        record["expected_direction"] = "faster" if value < 1.0 else "slower"

    elif kind == "f0":
        # Shift target F0
        modified["target_f0"][start_frame:end_frame] += value
        record["expected_direction"] = "higher" if value > 0 else "lower"

    elif kind == "pause":
        # Mark frames as pause
        modified["pause"][start_frame:end_frame] = 1.0
        modified["vocal_active"][start_frame:end_frame] = 0.0
        record["expected_direction"] = "paused"

    else:
        raise ValueError(f"Unknown counterfactual kind: {kind}")

    modified.setdefault("metadata", {})["last_counterfactual"] = record.copy()
    return modified, record
