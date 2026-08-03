"""B4 - causal evaluator core for Phase D Timing v2.

Pure-math layer: no ASR, no GPU. Callers supply matched word timings; this
module computes the four error terms, plan reliance, direction/magnitude,
locality and the pass/measurable split.

The decisive quantity is ``E_shuf_vs_shuf``. A large ``E_shuf_vs_real`` alone
is ambiguous - the model may have faithfully followed a wrong plan (control) or
simply broken (disruption). Only ``E_shuf_vs_shuf`` separates the two.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

# ------------------------------------------------------------------ thresholds
PLAN_RELIANCE_MIN = 0.30
E_MISSING_MARGIN_SEC = 0.060
E_SHUF_REAL_MARGIN_SEC = 0.060
E_SHUF_SHUF_TOLERANCE_SEC = 0.040
SHIFT_TOLERANCE_SEC = 0.100
DURATION_TOLERANCE = 0.10
NEIGHBOUR_MAX_MOVEMENT_SEC = 0.100
MASK_ONLY_MAX_MOVEMENT_SEC = 0.060
IDENTITY_MAX_DROP = 0.05
MIN_SHARED_WORDS = 6
EPS = 1e-12


@dataclass(frozen=True)
class Usability:
    """C25 floor gate for a single render."""

    vocal_activity: float
    coverage: float
    wer: float

    @property
    def audio_usable(self) -> bool:
        return self.vocal_activity >= 0.15 and self.coverage >= 0.60 and self.wer <= 0.50


def median_abs_error(predicted: dict[int, float], reference: dict[int, float]) -> tuple[float | None, int]:
    """Median |predicted - reference| over shared word ids."""
    shared = sorted(set(predicted) & set(reference))
    if not shared:
        return None, 0
    errors = [abs(predicted[i] - reference[i]) for i in shared]
    return float(np.median(errors)), len(shared)


# --------------------------------------------------------------- wrong plans
def donor_plan(target_ids: list[int], donor_starts: list[float], canvas_sec: float) -> dict[int, float]:
    """Map a donor phrase's timing onto the target word ids.

    The donor must have the same word count. Order is preserved, so the result
    is monotonic, collision-free and inside the canvas.
    """
    if len(target_ids) != len(donor_starts):
        raise ValueError("donor plan requires matching word counts")
    starts = sorted(donor_starts)
    if starts[0] < 0.0 or starts[-1] > canvas_sec:
        raise ValueError("donor plan leaves the canvas")
    return {wid: starts[i] for i, wid in enumerate(target_ids)}


def jitter_plan(
    starts: dict[int, float],
    canvas_sec: float,
    magnitude_sec: float = 0.300,
    seed: int = 20260803,
    min_gap_sec: float = 0.020,
) -> dict[int, float]:
    """Perturb each word start, then repair to stay monotonic and in-canvas.

    Never reorders words: the output preserves the input's word ordering.
    """
    rng = random.Random(seed)
    ids = sorted(starts, key=lambda i: starts[i])
    out: dict[int, float] = {}
    previous = 0.0
    for index, wid in enumerate(ids):
        offset = rng.uniform(-magnitude_sec, magnitude_sec)
        value = starts[wid] + offset
        value = max(value, previous + (min_gap_sec if index else 0.0))
        value = min(value, canvas_sec)
        out[wid] = value
        previous = value
    return out


def is_monotonic(plan: dict[int, float], order: Iterable[int]) -> bool:
    sequence = [plan[i] for i in order if i in plan]
    return all(b >= a for a, b in zip(sequence, sequence[1:]))


def has_collision(plan: dict[int, float], order: Iterable[int], min_gap_sec: float = 0.0) -> bool:
    sequence = [plan[i] for i in order if i in plan]
    return any((b - a) < min_gap_sec for a, b in zip(sequence, sequence[1:]))


# ------------------------------------------------------------------- reliance
def plan_reliance(e_missing: float | None, e_correct: float | None) -> float | None:
    if e_missing is None or e_correct is None:
        return None
    return (e_missing - e_correct) / max(e_missing, EPS)


def evaluate_causality(
    real_starts: dict[int, float],
    correct_render: dict[int, float],
    missing_render: dict[int, float],
    shuffled_render: dict[int, float],
    shuffled_plan: dict[int, float],
    usable_correct: bool,
    usable_missing: bool,
    usable_shuffled: bool,
) -> dict[str, Any]:
    """Compute the four error terms, reliance and the pass/measurable split."""
    e_correct, n_correct = median_abs_error(correct_render, real_starts)
    e_missing, n_missing = median_abs_error(missing_render, real_starts)
    e_shuf_real, n_sr = median_abs_error(shuffled_render, real_starts)
    e_shuf_shuf, n_ss = median_abs_error(shuffled_render, shuffled_plan)

    # Reliance is defined only when BOTH the correct and missing renders are
    # usable. An ASR-empty null is undefined, never "a large error".
    measurable = bool(
        usable_correct and usable_missing
        and e_correct is not None and e_missing is not None
        and n_correct >= MIN_SHARED_WORDS and n_missing >= MIN_SHARED_WORDS
    )
    reliance = plan_reliance(e_missing, e_correct) if measurable else None

    checks: dict[str, bool | None] = {
        "plan_reliance_pass": (reliance is not None and reliance >= PLAN_RELIANCE_MIN) if measurable else None,
        "e_missing_margin_pass": (
            (e_missing - e_correct) >= E_MISSING_MARGIN_SEC if measurable else None
        ),
        "e_shuf_real_margin_pass": (
            (e_shuf_real - e_correct) >= E_SHUF_REAL_MARGIN_SEC
            if (usable_shuffled and usable_correct and e_shuf_real is not None and e_correct is not None)
            else None
        ),
        "e_shuf_shuf_following_pass": (
            e_shuf_shuf <= e_correct + E_SHUF_SHUF_TOLERANCE_SEC
            if (usable_shuffled and usable_correct and e_shuf_shuf is not None and e_correct is not None)
            else None
        ),
    }
    decided = [v for v in checks.values() if v is not None]
    return {
        "E_correct": e_correct,
        "E_missing": e_missing,
        "E_shuf_vs_real": e_shuf_real,
        "E_shuf_vs_shuf": e_shuf_shuf,
        "shared_words": {"correct": n_correct, "missing": n_missing,
                         "shuffled_vs_real": n_sr, "shuffled_vs_shuffled": n_ss},
        "plan_reliance": reliance,
        "plan_reliance_measurable": measurable,
        "checks": checks,
        "plan_reliance_passed": bool(decided) and all(decided) and measurable,
        "thresholds": {
            "plan_reliance_min": PLAN_RELIANCE_MIN,
            "e_missing_margin_sec": E_MISSING_MARGIN_SEC,
            "e_shuf_real_margin_sec": E_SHUF_REAL_MARGIN_SEC,
            "e_shuf_shuf_tolerance_sec": E_SHUF_SHUF_TOLERANCE_SEC,
        },
    }


# ---------------------------------------------------------- direction / scale
def timing_direction_and_magnitude(
    baseline: dict[int, float], variant: dict[int, float], requested_shift_sec: float
) -> dict[str, Any]:
    shared = sorted(set(baseline) & set(variant))
    if len(shared) < MIN_SHARED_WORDS:
        return {"measurable": False, "shared_words": len(shared)}
    deltas = [variant[i] - baseline[i] for i in shared]
    realized = float(np.median(deltas))
    return {
        "measurable": True,
        "shared_words": len(shared),
        "realized_shift_sec": realized,
        "requested_shift_sec": requested_shift_sec,
        "direction_correct": bool(np.sign(realized) == np.sign(requested_shift_sec)) if requested_shift_sec else None,
        "magnitude_error_sec": abs(realized - requested_shift_sec),
        "magnitude_correct": abs(realized - requested_shift_sec) <= SHIFT_TOLERANCE_SEC,
    }


def duration_scale(
    baseline: dict[int, float], variant: dict[int, float], requested_ratio: float
) -> dict[str, Any]:
    """Theil-Sen slope of variant vs baseline starts - robust, multi-anchor."""
    shared = sorted(set(baseline) & set(variant))
    if len(shared) < MIN_SHARED_WORDS:
        return {"measurable": False, "shared_words": len(shared)}
    xs = np.array([baseline[i] for i in shared])
    ys = np.array([variant[i] for i in shared])
    slopes = [
        (ys[j] - ys[i]) / (xs[j] - xs[i])
        for i in range(len(xs)) for j in range(i + 1, len(xs))
        if abs(xs[j] - xs[i]) > 1e-6
    ]
    if not slopes:
        return {"measurable": False, "shared_words": len(shared)}
    realized = float(np.median(slopes))
    return {
        "measurable": True,
        "shared_words": len(shared),
        "realized_scale": realized,
        "requested_scale": requested_ratio,
        "scale_error": abs(realized - requested_ratio),
        "duration_correct": abs(realized - requested_ratio) <= DURATION_TOLERANCE,
    }


def locality(
    baseline: dict[int, float],
    variant: dict[int, float],
    previous_ids: set[int],
    next_ids: set[int],
    target_ids: set[int],
) -> dict[str, Any]:
    def movement(ids: set[int]) -> float | None:
        shared = sorted((set(baseline) & set(variant)) & ids)
        if not shared:
            return None
        return float(np.median([abs(variant[i] - baseline[i]) for i in shared]))

    previous_movement = movement(previous_ids)
    next_movement = movement(next_ids)
    outside_movement = movement((set(baseline) | set(variant)) - target_ids)
    passes = [
        m <= NEIGHBOUR_MAX_MOVEMENT_SEC
        for m in (previous_movement, next_movement, outside_movement) if m is not None
    ]
    return {
        "previous_phrase_movement_sec": previous_movement,
        "next_phrase_movement_sec": next_movement,
        "outside_target_movement_sec": outside_movement,
        "locality_pass": bool(passes) and all(passes),
        "threshold_sec": NEIGHBOUR_MAX_MOVEMENT_SEC,
    }


def mask_only_pass(baseline: dict[int, float], mask_only: dict[int, float]) -> dict[str, Any]:
    """A correct mask with unedited timing must not move anything."""
    error, shared = median_abs_error(mask_only, baseline)
    return {
        "mask_only_movement_sec": error,
        "shared_words": shared,
        "mask_only_pass": error is not None and error <= MASK_ONLY_MAX_MOVEMENT_SEC,
        "threshold_sec": MASK_ONLY_MAX_MOVEMENT_SEC,
    }


def identity_pass(c25_cosine: float, phase_d_cosine: float) -> dict[str, Any]:
    drop = c25_cosine - phase_d_cosine
    return {
        "c25_cosine": c25_cosine,
        "phase_d_cosine": phase_d_cosine,
        "identity_drop": drop,
        "identity_pass": drop <= IDENTITY_MAX_DROP,
        "threshold": IDENTITY_MAX_DROP,
    }


def f0_preservation(reference_hz: np.ndarray, generated_hz: np.ndarray) -> dict[str, Any]:
    """Median absolute cents error on frames voiced in both."""
    ref = np.asarray(reference_hz, dtype=float)
    gen = np.asarray(generated_hz, dtype=float)
    n = min(len(ref), len(gen))
    ref, gen = ref[:n], gen[:n]
    both = (ref > 0) & (gen > 0)
    if not both.any():
        return {"measurable": False, "voiced_overlap": 0.0}
    cents = 1200.0 * np.log2(gen[both] / ref[both])
    return {
        "measurable": True,
        "voiced_overlap": float(both.mean()),
        "median_abs_cents": float(np.median(np.abs(cents))),
    }


__all__ = [
    "Usability", "median_abs_error", "donor_plan", "jitter_plan", "is_monotonic",
    "has_collision", "plan_reliance", "evaluate_causality",
    "timing_direction_and_magnitude", "duration_scale", "locality",
    "mask_only_pass", "identity_pass", "f0_preservation",
    "PLAN_RELIANCE_MIN", "E_MISSING_MARGIN_SEC", "E_SHUF_REAL_MARGIN_SEC",
    "E_SHUF_SHUF_TOLERANCE_SEC", "MIN_SHARED_WORDS",
]
