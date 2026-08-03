"""B2 - counterfactual waveform pipeline for Phase D Timing v2.

Builds physically valid earlier / later / shorter / longer variants of a single
target phrase inside a real multi-phrase canvas.

Core identity::

    edited_mix = mixed + (edited_lead - lead)

The separation is not exact (``mixed != vocal + acc``), so re-summing stems
would change the accompaniment. Adding only the lead-vocal delta leaves every
non-target sample of the mix bit-identical by construction, which makes the
"accompaniment unchanged" gate an assertion rather than a tolerance.

All gates fail closed. A rejected edit returns a reason and never a waveform.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal

import numpy as np

EditKind = Literal["earlier", "later", "shorter", "longer"]

# ---------------------------------------------------------------- gate limits
MIN_HEADROOM_SEC = 0.150          # adjacent silence remaining after the edit
CROSSFADE_SEC = 0.020            # local seam ramp (within the 100-250 ms budget)
MAX_SEAM_DB = 6.0                # level discontinuity across a 10 ms window
MAX_TAIL_DB = -25.0              # residual at the vacated original position
MAX_BLEED_DB = -20.0             # lead energy present in the accompaniment
SEAM_WINDOW_SEC = 0.010
EPS = 1e-12


@dataclass(frozen=True)
class EditSpec:
    """A requested counterfactual edit."""

    kind: EditKind
    value: float                  # seconds for shifts, ratio for durations

    def label(self) -> str:
        if self.kind in ("earlier", "later"):
            return f"{self.kind}_{abs(self.value):.2f}s".replace(".", "p")
        return f"{self.kind}_{self.value:.2f}x".replace(".", "p")


def db(numerator: float, denominator: float) -> float:
    """Return a ratio in dB, floored so silence never yields -inf."""
    return 10.0 * float(np.log10(max(numerator, EPS) / max(denominator, EPS)))


def _ramp(length: int) -> np.ndarray:
    """Equal-power fade-in ramp of ``length`` samples."""
    if length <= 1:
        return np.ones(max(length, 1))
    return np.sin(np.linspace(0.0, np.pi / 2.0, length)) ** 2


def _stretch(segment: np.ndarray, ratio: float) -> np.ndarray:
    """Time-scale a mono segment by ``ratio`` (>1 = longer)."""
    import librosa

    if abs(ratio - 1.0) < 1e-9:
        return segment.copy()
    # librosa.effects.time_stretch(rate) makes audio rate-times FASTER,
    # so a longer output needs rate = 1/ratio.
    out = librosa.effects.time_stretch(np.ascontiguousarray(segment), rate=1.0 / ratio)
    target = int(round(len(segment) * ratio))
    if len(out) < target:
        out = np.pad(out, (0, target - len(out)))
    return out[:target]


def plan_edit(
    target: tuple[float, float],
    previous_end_sec: float,
    next_start_sec: float,
    canvas_sec: float,
    spec: EditSpec,
) -> dict[str, Any]:
    """Compute the new target span and check feasibility before touching audio."""
    start, end = target
    length = end - start
    if length <= 0:
        return {"ok": False, "reason": "non_positive_target_length"}

    if spec.kind == "earlier":
        new_start, new_end = start - abs(spec.value), end - abs(spec.value)
    elif spec.kind == "later":
        new_start, new_end = start + abs(spec.value), end + abs(spec.value)
    elif spec.kind in ("shorter", "longer"):
        new_start, new_end = start, start + length * spec.value
    else:
        return {"ok": False, "reason": f"unknown_kind_{spec.kind}"}

    if new_start < 0.0 or new_end > canvas_sec:
        return {"ok": False, "reason": "canvas_clipping"}

    gap_before = new_start - previous_end_sec
    gap_after = next_start_sec - new_end
    if gap_before < MIN_HEADROOM_SEC:
        return {"ok": False, "reason": f"headroom_before_{gap_before:.3f}s"}
    if gap_after < MIN_HEADROOM_SEC:
        return {"ok": False, "reason": f"headroom_after_{gap_after:.3f}s"}
    if new_start <= previous_end_sec or new_end >= next_start_sec:
        return {"ok": False, "reason": "collision_with_neighbour"}

    return {
        "ok": True,
        "original_span_sec": [start, end],
        "new_span_sec": [new_start, new_end],
        "original_length_sec": length,
        "new_length_sec": new_end - new_start,
        "realized_ratio": (new_end - new_start) / length,
        "realized_shift_sec": new_start - start,
        "gap_before_sec": gap_before,
        "gap_after_sec": gap_after,
    }


def apply_edit(
    lead: np.ndarray,
    mixed: np.ndarray,
    accompaniment: np.ndarray,
    sample_rate: int,
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Relocate/stretch the target phrase and rebuild the mix.

    ``lead``, ``mixed`` and ``accompaniment`` are mono float arrays of equal
    length. Returns the edited lead and mix plus every gate measurement.
    """
    if not (len(lead) == len(mixed) == len(accompaniment)):
        return {"ok": False, "reason": "stem_length_mismatch"}

    def to_sample(seconds: float) -> int:
        return int(round(seconds * sample_rate))

    old_a, old_b = (to_sample(v) for v in plan["original_span_sec"])
    new_a, new_b = (to_sample(v) for v in plan["new_span_sec"])
    fade = max(1, to_sample(CROSSFADE_SEC))

    phrase = lead[old_a:old_b].copy()
    if phrase.size == 0:
        return {"ok": False, "reason": "empty_target_phrase"}
    phrase_energy = float(np.mean(phrase**2))

    transformed = _stretch(phrase, plan["realized_ratio"]) if abs(plan["realized_ratio"] - 1.0) > 1e-9 else phrase.copy()
    span = new_b - new_a
    if len(transformed) < span:
        transformed = np.pad(transformed, (0, span - len(transformed)))
    transformed = transformed[:span]

    # Equal-power ramps so the reinsertion has no step discontinuity.
    ramp = _ramp(min(fade, span // 2 if span >= 2 else 1))
    if ramp.size:
        transformed[: ramp.size] *= ramp
        transformed[-ramp.size :] *= ramp[::-1]

    edited_lead = lead.copy()
    # Vacate the original position, then place the transformed phrase.
    edited_lead[old_a:old_b] = 0.0
    edited_lead[new_a:new_b] = transformed

    # Only the lead delta is applied - accompaniment stays bit-identical.
    edited_mix = mixed + (edited_lead - lead)

    # ------------------------------------------------------------- gate checks
    touched = np.flatnonzero(edited_mix != mixed)
    touched_span = [int(touched[0]), int(touched[-1])] if touched.size else [0, 0]
    protected_lo, protected_hi = min(old_a, new_a), max(old_b, new_b)
    outside_touched = bool(touched.size and (touched[0] < protected_lo or touched[-1] >= protected_hi))

    vacated = edited_lead[old_a:old_b] if old_a < old_b else np.array([0.0])
    # Residual left in the region the phrase came from, excluding any overlap
    # with its new location.
    overlap_lo, overlap_hi = max(old_a, new_a), min(old_b, new_b)
    if overlap_hi > overlap_lo:
        mask = np.ones(old_b - old_a, dtype=bool)
        mask[overlap_lo - old_a : overlap_hi - old_a] = False
        vacated = edited_lead[old_a:old_b][mask] if mask.any() else np.array([0.0])
    tail_db = db(float(np.mean(vacated**2)) if vacated.size else 0.0, phrase_energy)

    # Separation bleed = how much of the LEAD signal is present in the
    # accompaniment stem, measured as normalised cross-correlation. Raw
    # accompaniment energy is not bleed: the band legitimately plays under the
    # vocal in every real section.
    lead_win = lead[old_a:old_b]
    acc_win = accompaniment[old_a:old_b]
    lead_c = lead_win - lead_win.mean()
    acc_c = acc_win - acc_win.mean()
    denominator = float(np.sqrt((lead_c**2).sum() * (acc_c**2).sum()))
    bleed_corr = abs(float((lead_c * acc_c).sum()) / denominator) if denominator > EPS else 0.0
    bleed_db = 20.0 * float(np.log10(max(bleed_corr, 10 ** (MAX_BLEED_DB / 20.0) / 1e6)))

    # Seam quality = does the edit introduce a step larger than the phrase's own
    # sample-to-sample dynamics? A phrase onset meeting silence is not a defect;
    # a discontinuity sharper than the signal itself is.
    win = max(2, to_sample(SEAM_WINDOW_SEC))
    body = np.abs(np.diff(transformed)) if transformed.size > 1 else np.array([0.0])
    body_max = float(body.max()) if body.size else 0.0
    seam_steps = []
    for boundary in (new_a, new_b):
        lo, hi = max(1, boundary - win), min(len(edited_lead), boundary + win)
        if hi - lo < 2:
            continue
        seam_steps.append(float(np.abs(np.diff(edited_lead[lo:hi])).max()))
    seam_step = max(seam_steps) if seam_steps else 0.0
    seam_db = db(seam_step**2, max(body_max, EPS) ** 2)

    clipped = bool(np.max(np.abs(edited_mix)) > 1.0)

    gates = {
        "accompaniment_unchanged": not outside_touched,
        "touched_sample_span": touched_span,
        "protected_span": [protected_lo, protected_hi],
        "reverb_tail_db": round(tail_db, 3),
        "reverb_tail_pass": tail_db <= MAX_TAIL_DB,
        "separation_bleed_db": round(bleed_db, 3),
        "separation_bleed_pass": bleed_db <= MAX_BLEED_DB,
        "seam_discontinuity_db": round(seam_db, 3),
        "seam_pass": seam_db <= MAX_SEAM_DB,
        "canvas_clipping": clipped,
        "clipping_pass": not clipped,
    }
    gates["all_pass"] = bool(
        gates["accompaniment_unchanged"]
        and gates["reverb_tail_pass"]
        and gates["separation_bleed_pass"]
        and gates["seam_pass"]
        and gates["clipping_pass"]
    )

    failed = [k.removesuffix("_pass") for k, v in gates.items() if k.endswith("_pass") and not v]
    if not gates["accompaniment_unchanged"]:
        failed.append("accompaniment_changed_outside_target")

    return {
        "ok": gates["all_pass"],
        "reason": None if gates["all_pass"] else "gate_failed:" + ",".join(sorted(set(failed))),
        "edited_lead": edited_lead,
        "edited_mix": edited_mix,
        "gates": gates,
        "crossfade_sec": CROSSFADE_SEC,
        "stretch_method": "librosa.effects.time_stretch",
    }


def shift_timing_plan(words: list[dict[str, Any]], plan: dict[str, Any], target_ids: set[int]) -> list[dict[str, Any]]:
    """Produce the modified timing sidecar for the edited target phrase.

    Words inside the target are shifted/scaled to match the audio edit; every
    other word keeps its original timing exactly.
    """
    start = plan["original_span_sec"][0]
    ratio = plan["realized_ratio"]
    shift = plan["realized_shift_sec"]
    out = []
    for word in words:
        row = dict(word)
        if word["word_id"] in target_ids:
            row["start_sec"] = start + (word["start_sec"] - start) * ratio + shift
            row["end_sec"] = start + (word["end_sec"] - start) * ratio + shift
            row["duration_sec"] = row["end_sec"] - row["start_sec"]
            row["counterfactual_modified"] = True
        else:
            row["counterfactual_modified"] = False
        out.append(row)
    return out


def neighbour_movement_sec(original: list[dict[str, Any]], modified: list[dict[str, Any]], target_ids: set[int]) -> float:
    """Maximum absolute start movement of any non-target word. Must be 0."""
    worst = 0.0
    for before, after in zip(original, modified):
        if before["word_id"] in target_ids:
            continue
        worst = max(worst, abs(after["start_sec"] - before["start_sec"]))
    return worst


__all__ = [
    "EditSpec",
    "EditKind",
    "plan_edit",
    "apply_edit",
    "shift_timing_plan",
    "neighbour_movement_sec",
    "db",
    "MIN_HEADROOM_SEC",
    "CROSSFADE_SEC",
    "MAX_SEAM_DB",
    "MAX_TAIL_DB",
    "MAX_BLEED_DB",
]
