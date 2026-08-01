"""Duration-based physical frame expansion for the Phase D2 condition stream."""

from __future__ import annotations

from typing import Any

import torch


def _allocation(
    durations_sec: torch.Tensor,
    frame_rate_hz: float,
    target_frames: int | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Convert seconds to physical frames and apply one optional global resampling."""
    values = torch.as_tensor(durations_sec, dtype=torch.float64)
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("durations_sec must be a non-empty rank-1 tensor")
    if frame_rate_hz <= 0 or not bool(torch.isfinite(values).all()) or bool((values <= 0).any()):
        raise ValueError("durations and frame_rate_hz must be finite and positive")

    physical_quotas = values * frame_rate_hz
    physical_total = max(values.numel(), int(round(float(physical_quotas.sum()))))
    desired_total = int(target_frames) if target_frames is not None else physical_total
    if desired_total < values.numel():
        raise ValueError("target_frames must allocate at least one frame per event")

    scaled_quotas = physical_quotas / physical_quotas.sum() * desired_total
    floors = torch.floor(scaled_quotas)
    counts = floors.to(torch.long).clamp_min(1)
    delta = desired_total - int(counts.sum())
    remainders = scaled_quotas - floors
    while delta > 0:
        order = torch.argsort(remainders, descending=True, stable=True)
        for index in order.tolist():
            if delta == 0:
                break
            counts[index] += 1
            delta -= 1
    while delta < 0:
        candidates = torch.where(counts > 1)[0]
        if candidates.numel() == 0:
            raise RuntimeError("minimum-one constraint exceeds requested target")
        order = candidates[
            torch.argsort(remainders[candidates], descending=False, stable=True)
        ]
        for index in order.tolist():
            if delta == 0:
                break
            if counts[index] > 1:
                counts[index] -= 1
                delta += 1
    if int(counts.sum()) != desired_total:
        raise RuntimeError("duration frame allocation failed to preserve target length")

    starts = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)[:-1]])
    ends = counts.cumsum(0)
    audit = {
        "method": "physical_seconds_then_global_resample_largest_remainder_v2",
        "frame_rate_hz": float(frame_rate_hz),
        "duration_seconds": values.tolist(),
        "physical_frame_quotas": physical_quotas.tolist(),
        "physical_total_frames_rounded": physical_total,
        "target_frames_role": "final_global_resampling_target",
        "requested_target_frames": int(target_frames) if target_frames is not None else None,
        "global_resampling_factor": desired_total / physical_total,
        "target_frames": desired_total,
        "event_frame_counts": counts.tolist(),
        "event_start_frames": starts.tolist(),
        "event_end_frames_exclusive": ends.tolist(),
        "rounding_remainder_policy": "largest_fraction_first_stable_event_order",
        "collisions": 0,
    }
    return counts, audit


def allocate_duration_frames(
    durations_sec: torch.Tensor,
    frame_rate_hz: float,
    target_frames: int | None = None,
) -> torch.Tensor:
    """Return deterministic frame counts after physical conversion and resampling."""
    return _allocation(durations_sec, frame_rate_hz, target_frames)[0]


def length_regulate(
    event_features: torch.Tensor,
    durations_sec: torch.Tensor,
    frame_rate_hz: float,
    target_frames: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Repeat event features over allocated latent frames and return an audit."""
    if event_features.ndim != 2:
        raise ValueError("event_features must have shape [events, channels]")
    if event_features.shape[0] != torch.as_tensor(durations_sec).numel():
        raise ValueError("event feature and duration counts differ")
    counts, audit = _allocation(durations_sec, frame_rate_hz, target_frames)
    event_ids = torch.arange(event_features.shape[0], device=event_features.device)
    device_counts = counts.to(event_features.device)
    dense = torch.repeat_interleave(event_features, device_counts, dim=0)
    frame_event_ids = torch.repeat_interleave(event_ids, device_counts)
    audit["event_order_preserved"] = bool(
        torch.equal(torch.unique_consecutive(frame_event_ids), event_ids)
    )
    return dense, frame_event_ids, audit
