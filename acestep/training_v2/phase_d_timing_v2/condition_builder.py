"""
Condition Builder for Phase D2.

Assembles per-frame dense condition tensors from linguistic/timing/prosody events,
and provides audit trails for debugging and ablation.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch

from acestep.training_v2.phase_d_timing_v2.config import PhaseD2Config
from acestep.training_v2.phase_d_timing_v2.sidecar_schema import validate_sidecar


def build_dense_condition(
    events: List[Dict[str, Any]],
    beat_times: Optional[List[float]],
    duration_sec: float,
    n_frames: int,
    config: Optional[PhaseD2Config] = None,
) -> Dict[str, Any]:
    """Build dense per-frame condition fields from linguistic and timing events.

    Event intervals are rasterized onto the latent frame grid.  Silence is
    represented explicitly (ID ``-1``, ``vocal_active=0``, ``pause=1``), so an
    omitted or malformed event cannot silently become an all-vocal condition.

    Args:
        events: List of event dicts, each with keys like:
            {section_id, phrase_id, word_id, phoneme_id, phoneme, word,
             start_sec, end_sec, [f0_hz], [energy]}
        beat_times: List of beat onset times in seconds (for beat alignment).
        duration_sec: Total duration of audio in seconds.
        n_frames: Number of latent frames.
        config: PhaseD2Config (used to infer frame_rate if None, defaults to 25 Hz).

    Returns:
        Dictionary with per-frame fields matching sidecar schema v1:
        {
            "absolute_time": [n_frames],
            "normalized_song_time": [n_frames],
            ...
            "metadata": {event_to_frame mapping, interpolation method, etc.}
        }
    """

    if config is None:
        config = PhaseD2Config()

    if duration_sec <= 0:
        raise ValueError(f"duration_sec must be > 0, got {duration_sec}")
    if n_frames <= 0:
        raise ValueError(f"n_frames must be > 0, got {n_frames}")

    frame_rate = config.latent_frame_rate_hz

    # Build time grid
    frame_times = np.arange(n_frames) / frame_rate  # seconds

    # Initialize all fields with defaults
    sidecar = {
        "absolute_time": frame_times.astype(np.float32),
        "normalized_song_time": np.clip(frame_times / duration_sec, 0.0, 1.0).astype(np.float32),
        "beat_relative_time": np.zeros(n_frames, dtype=np.float32),
        "bar_position": np.zeros(n_frames, dtype=np.float32),
        "section_id": np.full(n_frames, -1, dtype=np.int32),
        "phrase_id": np.full(n_frames, -1, dtype=np.int32),
        "word_id": np.full(n_frames, -1, dtype=np.int32),
        "phoneme_id": np.full(n_frames, -1, dtype=np.int32),
        "pos_in_phrase": np.zeros(n_frames, dtype=np.float32),
        "pos_in_word": np.zeros(n_frames, dtype=np.float32),
        "pos_in_phoneme": np.zeros(n_frames, dtype=np.float32),
        "word_start_pulse": np.zeros(n_frames, dtype=np.float32),
        "word_end_pulse": np.zeros(n_frames, dtype=np.float32),
        "phrase_start_pulse": np.zeros(n_frames, dtype=np.float32),
        "phrase_end_pulse": np.zeros(n_frames, dtype=np.float32),
        "vowel_nucleus": np.zeros(n_frames, dtype=np.float32),
        "target_phrase_duration": np.zeros(n_frames, dtype=np.float32),
        "target_word_duration": np.zeros(n_frames, dtype=np.float32),
        "target_phoneme_duration": np.zeros(n_frames, dtype=np.float32),
        "target_f0": np.zeros(n_frames, dtype=np.float32),
        "target_note": np.zeros(n_frames, dtype=np.float32),
        "note_onset": np.zeros(n_frames, dtype=np.float32),
        "note_end": np.zeros(n_frames, dtype=np.float32),
        "voiced": np.zeros(n_frames, dtype=np.float32),
        "energy": np.zeros(n_frames, dtype=np.float32),
        "breath": np.zeros(n_frames, dtype=np.float32),
        "pause": np.ones(n_frames, dtype=np.float32),
        "vocal_active": np.zeros(n_frames, dtype=np.float32),
        "local_tempo": np.ones(n_frames, dtype=np.float32),   # default: 1.0 (100% tempo)
        "rubato_offset": np.zeros(n_frames, dtype=np.float32),
        "_schema_version": "phase_d2_sidecar_v1",
    }

    event_to_frame: List[Dict[str, Any]] = []
    occupancy = np.zeros(n_frames, dtype=np.int32)

    def interval_to_frames(start_sec: float, end_sec: float) -> tuple[int, int]:
        if not np.isfinite(start_sec) or not np.isfinite(end_sec):
            raise ValueError(f"Non-finite event interval: {start_sec}, {end_sec}")
        if start_sec < 0 or end_sec <= start_sec or end_sec > duration_sec + 1e-6:
            raise ValueError(
                "Event interval must satisfy 0 <= start < end <= duration; "
                f"got [{start_sec}, {end_sec}] for duration {duration_sec}."
            )
        start_frame = int(np.floor(start_sec * frame_rate))
        end_frame = int(np.ceil(end_sec * frame_rate))
        return max(0, start_frame), min(n_frames, max(start_frame + 1, end_frame))

    # Populate all event-local fields.
    if events:
        for event_index, event in enumerate(events):
            if "start_sec" not in event or "end_sec" not in event:
                raise KeyError(f"Event {event_index} lacks start_sec/end_sec")
            start_sec = float(event["start_sec"])
            end_sec = float(event["end_sec"])
            start_frame, end_frame = interval_to_frames(start_sec, end_sec)
            span = slice(start_frame, end_frame)
            span_len = end_frame - start_frame
            occupancy[span] += 1

            for key in ("phoneme_id", "word_id", "section_id", "phrase_id"):
                sidecar[key][span] = int(event.get(key, -1))

            position = np.linspace(0.0, 1.0, span_len, dtype=np.float32)
            sidecar["pos_in_phoneme"][span] = position
            sidecar["target_phoneme_duration"][span] = end_sec - start_sec
            sidecar["vocal_active"][span] = float(event.get("vocal_active", 1.0))
            sidecar["pause"][span] = float(event.get("pause", 0.0))
            sidecar["voiced"][span] = float(event.get("voiced", "f0_hz" in event))

            phoneme = str(event.get("phoneme", "")).lower().strip()
            is_vowel = bool(event.get("vowel_nucleus", phoneme[:1] in "aeiouy"))
            sidecar["vowel_nucleus"][span] = float(is_vowel)

            scalar_fields = {
                "target_f0": event.get("f0_hz", event.get("target_f0")),
                "target_note": event.get("midi_note", event.get("target_note")),
                "energy": event.get("energy"),
                "breath": event.get("breath"),
                "rubato_offset": event.get("rubato_offset"),
            }
            for key, value in scalar_fields.items():
                if value is not None:
                    sidecar[key][span] = float(value)

            if scalar_fields["target_note"] is not None:
                sidecar["note_onset"][start_frame] = 1.0
                sidecar["note_end"][end_frame - 1] = 1.0

            event_to_frame.append({
                "event_index": event_index,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "start_frame": start_frame,
                "end_frame_exclusive": end_frame,
            })

        # Aggregate word and phrase extents.  This remains correct when a word
        # is represented by multiple phoneme events.
        for id_keys, pos_key, duration_key, start_pulse, end_pulse in (
            (("section_id", "phrase_id", "word_id"), "pos_in_word", "target_word_duration", "word_start_pulse", "word_end_pulse"),
            (("section_id", "phrase_id"), "pos_in_phrase", "target_phrase_duration", "phrase_start_pulse", "phrase_end_pulse"),
        ):
            groups: Dict[tuple[int, ...], List[Dict[str, Any]]] = {}
            for event in events:
                group_id = tuple(int(event.get(key, -1)) for key in id_keys)
                groups.setdefault(group_id, []).append(event)
            for grouped_events in groups.values():
                group_start = min(float(e["start_sec"]) for e in grouped_events)
                group_end = max(float(e["end_sec"]) for e in grouped_events)
                start_frame, end_frame = interval_to_frames(group_start, group_end)
                span_len = end_frame - start_frame
                sidecar[pos_key][start_frame:end_frame] = np.linspace(
                    0.0, 1.0, span_len, dtype=np.float32
                )
                sidecar[duration_key][start_frame:end_frame] = group_end - group_start
                if start_pulse is not None:
                    sidecar[start_pulse][start_frame] = 1.0
                    sidecar[end_pulse][end_frame - 1] = 1.0

    # Compute nearest-beat offset, beat-in-bar, and a local tempo ratio.
    if beat_times:
        beats = np.asarray(beat_times, dtype=np.float64)
        if beats.ndim != 1 or len(beats) < 2 or not np.all(np.isfinite(beats)):
            raise ValueError("beat_times must contain at least two finite values")
        if np.any(np.diff(beats) <= 0):
            raise ValueError("beat_times must be strictly increasing")
        right = np.searchsorted(beats, frame_times, side="left")
        right = np.clip(right, 0, len(beats) - 1)
        left = np.clip(right - 1, 0, len(beats) - 1)
        nearest = np.where(
            np.abs(frame_times - beats[left]) <= np.abs(frame_times - beats[right]),
            left,
            right,
        )
        sidecar["beat_relative_time"] = (frame_times - beats[nearest]).astype(np.float32)
        sidecar["bar_position"] = (nearest % 4).astype(np.float32)
        beat_intervals = np.diff(beats)
        median_interval = float(np.median(beat_intervals))
        interval_idx = np.clip(nearest, 0, len(beat_intervals) - 1)
        sidecar["local_tempo"] = (
            median_interval / beat_intervals[interval_idx]
        ).astype(np.float32)

    # Add metadata audit
    sidecar["metadata"] = {
        "frame_rate": float(frame_rate),
        "n_frames": n_frames,
        "duration_sec": float(duration_sec),
        "n_events": len(events),
        "event_to_frame_method": "floor-start/ceil-end",
        "interpolation_method": "step-hold",
        "beat_alignment": "beat_times" if beat_times else "none",
        "event_to_frame": event_to_frame,
        "overlap_frame_count": int(np.count_nonzero(occupancy > 1)),
        "silence_frame_count": int(np.count_nonzero(sidecar["vocal_active"] == 0)),
    }

    return sidecar


def build_condition_tensor(
    sidecar: Dict[str, Any],
    config: PhaseD2Config,
) -> Tuple[torch.Tensor, Dict[str, tuple[int, int]], Dict[str, Any]]:
    """Assemble per-frame condition tensor from sidecar dict.

    Validates sidecar, selects enabled groups per config, and constructs
    [1, n_frames, condition_dim] tensor.

    Args:
        sidecar: Dictionary from build_dense_condition().
        config: PhaseD2Config specifying which groups to include.

    Returns:
        Tuple of:
        - condition_tensor: [1, n_frames, condition_dim] float32 tensor
        - group_slices: {group_name: (start_idx, end_idx)} for ablation
        - audit: {frame_rate, method, normalization notes, etc.}

    Raises:
        ValueError: If sidecar fails validation.
    """

    # Validate sidecar
    validate_sidecar(sidecar)

    n_frames = len(sidecar["absolute_time"])

    # Get group slice boundaries
    group_slices = config.get_condition_group_slices()

    # Allocate condition tensor
    cond = torch.zeros(
        (1, n_frames, config.condition_dim),
        dtype=torch.float32,
    )

    # Fill in each enabled group
    idx = 0

    if config.linguistic:
        # Normalize IDs to [0, 1] (stub: use raw values clamped)
        word_id = torch.from_numpy(sidecar["word_id"].astype(np.float32))
        phoneme_id = torch.from_numpy(sidecar["phoneme_id"].astype(np.float32))
        pos_in_word = torch.from_numpy(sidecar["pos_in_word"].astype(np.float32))
        pos_in_phrase = torch.from_numpy(sidecar["pos_in_phrase"].astype(np.float32))

        # Normalize (stub: clamp to [-1, 1])
        word_id = torch.clamp(word_id / 100.0, -1, 1)
        phoneme_id = torch.clamp(phoneme_id / 100.0, -1, 1)
        pos_in_word = torch.clamp(pos_in_word / 20.0, -1, 1)
        pos_in_phrase = torch.clamp(pos_in_phrase / 100.0, -1, 1)

        cond[0, :, idx:idx+4] = torch.stack([
            word_id, phoneme_id, pos_in_word, pos_in_phrase
        ], dim=1)
        idx += 4

    if config.absolute_timing:
        absolute_time = torch.from_numpy(sidecar["absolute_time"].astype(np.float32))
        normalized_song_time = torch.from_numpy(sidecar["normalized_song_time"].astype(np.float32))

        # Normalize absolute_time to [-1, 1] by dividing by max (stub)
        max_time = absolute_time.max().item() + 1e-6
        absolute_time = torch.clamp(2 * absolute_time / max_time - 1, -1, 1)

        cond[0, :, idx:idx+2] = torch.stack([
            absolute_time, normalized_song_time
        ], dim=1)
        idx += 2

    if config.beat_relative_timing:
        beat_relative_time = torch.from_numpy(sidecar["beat_relative_time"].astype(np.float32))
        bar_position = torch.from_numpy(sidecar["bar_position"].astype(np.float32))
        local_tempo = torch.from_numpy(sidecar["local_tempo"].astype(np.float32))

        # Normalize
        beat_relative_time = torch.clamp(beat_relative_time / 0.5, -1, 1)
        bar_position = torch.clamp(bar_position / 4.0, -1, 1)
        local_tempo = torch.clamp(2 * local_tempo - 1, -1, 1)  # tempo ratio around 1.0

        cond[0, :, idx:idx+3] = torch.stack([
            beat_relative_time, bar_position, local_tempo
        ], dim=1)
        idx += 3

    if config.melody_prosody:
        target_f0 = torch.from_numpy(sidecar["target_f0"].astype(np.float32))
        target_note = torch.from_numpy(sidecar["target_note"].astype(np.float32))
        note_onset = torch.from_numpy(sidecar["note_onset"].astype(np.float32))
        note_end = torch.from_numpy(sidecar["note_end"].astype(np.float32))
        vowel_nucleus = torch.from_numpy(sidecar["vowel_nucleus"].astype(np.float32))

        # Normalize (stub)
        target_f0 = torch.clamp(target_f0 / 300.0, -1, 1)
        target_note = torch.clamp(target_note / 60.0 - 1, -1, 1)

        cond[0, :, idx:idx+5] = torch.stack([
            target_f0, target_note, note_onset, note_end, vowel_nucleus
        ], dim=1)
        idx += 5

    if config.breath_energy:
        energy = torch.from_numpy(sidecar["energy"].astype(np.float32))
        breath = torch.from_numpy(sidecar["breath"].astype(np.float32))

        # Normalize (energy already in [0, 1] typically)
        energy = torch.clamp(2 * energy - 1, -1, 1)

        cond[0, :, idx:idx+2] = torch.stack([
            energy, breath
        ], dim=1)
        idx += 2

    if config.vocal_activity:
        voiced = torch.from_numpy(sidecar["voiced"].astype(np.float32))
        vocal_active = torch.from_numpy(sidecar["vocal_active"].astype(np.float32))

        cond[0, :, idx:idx+2] = torch.stack([
            voiced, vocal_active
        ], dim=1)
        idx += 2

    if config.structure:
        section_id = torch.from_numpy(sidecar["section_id"].astype(np.float32))
        phrase_id = torch.from_numpy(sidecar["phrase_id"].astype(np.float32))
        word_id = torch.from_numpy(sidecar["word_id"].astype(np.float32))
        pause = torch.from_numpy(sidecar["pause"].astype(np.float32))

        # Normalize
        section_id = torch.clamp(section_id / 50.0, -1, 1)
        phrase_id = torch.clamp(phrase_id / 200.0, -1, 1)
        word_id = torch.clamp(word_id / 100.0, -1, 1)

        cond[0, :, idx:idx+4] = torch.stack([
            section_id, phrase_id, word_id, pause
        ], dim=1)
        idx += 4

    if config.explicit_duration:
        phrase_duration = torch.from_numpy(sidecar["target_phrase_duration"].astype(np.float32))
        word_duration = torch.from_numpy(sidecar["target_word_duration"].astype(np.float32))
        phoneme_duration = torch.from_numpy(sidecar["target_phoneme_duration"].astype(np.float32))
        phoneme_word_ratio = phoneme_duration / word_duration.clamp_min(1e-6)
        cond[0, :, idx:idx+4] = torch.stack([
            torch.clamp(phrase_duration / 15.0, 0, 1),
            torch.clamp(word_duration / 5.0, 0, 1),
            torch.clamp(phoneme_duration / 2.0, 0, 1),
            torch.clamp(phoneme_word_ratio, 0, 2) - 1.0,
        ], dim=1)
        idx += 4

    audit = {
        "frame_rate": config.latent_frame_rate_hz,
        "n_frames": n_frames,
        "n_groups_enabled": sum([
            config.linguistic,
            config.absolute_timing,
            config.beat_relative_timing,
            config.melody_prosody,
            config.breath_energy,
            config.vocal_activity,
            config.structure,
            config.explicit_duration,
        ]),
        "normalization_method": "clamp to [-1, 1]",
        "sidecar_metadata": sidecar.get("metadata", {}),
    }

    return cond, group_slices, audit
