from __future__ import annotations

from typing import Sequence

import numpy as np


def _slice_samples(audio: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    start = max(0, int(round(float(start_s) * sr)))
    end = max(start + 1, int(round(float(end_s) * sr)))
    return audio[start:end]


def rms_energy(segment: np.ndarray) -> float:
    if segment.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(segment.astype(np.float32))) + 1e-8))


def event_energy_targets(
    audio: np.ndarray,
    sr: int,
    event_starts: Sequence[float],
    event_ends: Sequence[float],
) -> np.ndarray:
    values = []
    for start_s, end_s in zip(event_starts, event_ends):
        segment = _slice_samples(audio, sr, start_s, end_s)
        values.append(rms_energy(segment))
    if not values:
        return np.zeros(0, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    denom = float(np.percentile(values, 95)) if values.size > 1 else float(values.max())
    denom = max(denom, 1e-4)
    return np.clip(values / denom, 0.0, 2.0).astype(np.float32)


def event_terminal_decay_targets(
    audio: np.ndarray,
    sr: int,
    event_starts: Sequence[float],
    event_ends: Sequence[float],
) -> np.ndarray:
    values = []
    for start_s, end_s in zip(event_starts, event_ends):
        segment = _slice_samples(audio, sr, start_s, end_s)
        if segment.size < 3:
            values.append(1.0)
            continue
        n = segment.size
        head = segment[: max(1, n // 3)]
        tail = segment[-max(1, n // 3) :]
        ratio = rms_energy(tail) / max(rms_energy(head), 1e-4)
        values.append(float(np.clip(ratio, 0.0, 2.5)))
    return np.asarray(values, dtype=np.float32)
