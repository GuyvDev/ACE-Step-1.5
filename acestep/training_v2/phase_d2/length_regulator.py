"""Duration-based frame expansion for the Phase D2 condition stream."""

from __future__ import annotations

from typing import Any

import torch


def allocate_duration_frames(
    durations_sec: torch.Tensor,
    frame_rate_hz: float,
    target_frames: int | None = None,
) -> torch.Tensor:
    """Convert positive event durations to integer frame counts.

    Largest-remainder allocation preserves event order, gives every event at
    least one frame, and exactly matches ``target_frames`` when provided.
    """
    values = torch.as_tensor(durations_sec, dtype=torch.float64)
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("durations_sec must be a non-empty rank-1 tensor")
    if frame_rate_hz <= 0 or not bool(torch.isfinite(values).all()) or bool((values <= 0).any()):
        raise ValueError("durations and frame_rate_hz must be finite and positive")
    desired_total = int(target_frames) if target_frames is not None else int(round(float(values.sum() * frame_rate_hz)))
    if desired_total < values.numel():
        raise ValueError("target_frames must allocate at least one frame per event")

    remaining = desired_total - values.numel()
    weights = values / values.sum()
    quotas = weights * remaining
    extra = torch.floor(quotas).to(torch.long)
    unassigned = remaining - int(extra.sum())
    if unassigned:
        remainder = quotas - extra.to(quotas.dtype)
        order = torch.argsort(remainder, descending=True, stable=True)
        extra[order[:unassigned]] += 1
    counts = extra + 1
    if int(counts.sum()) != desired_total:
        raise RuntimeError("duration frame allocation failed to preserve target length")
    return counts


def length_regulate(
    event_features: torch.Tensor,
    durations_sec: torch.Tensor,
    frame_rate_hz: float,
    target_frames: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Repeat each event feature vector over its allocated ACE latent frames.

    Returns the dense frame tensor, a frame-to-event index, and an audit record.
    """
    if event_features.ndim != 2:
        raise ValueError("event_features must have shape [events, channels]")
    if event_features.shape[0] != torch.as_tensor(durations_sec).numel():
        raise ValueError("event feature and duration counts differ")
    counts = allocate_duration_frames(durations_sec, frame_rate_hz, target_frames)
    event_ids = torch.arange(event_features.shape[0], device=event_features.device)
    dense = torch.repeat_interleave(event_features, counts.to(event_features.device), dim=0)
    frame_event_ids = torch.repeat_interleave(event_ids, counts.to(event_features.device))
    audit = {
        "method": "largest_remainder_min_one_v1",
        "frame_rate_hz": float(frame_rate_hz),
        "target_frames": int(dense.shape[0]),
        "event_frame_counts": counts.tolist(),
        "event_order_preserved": bool(torch.equal(torch.unique_consecutive(frame_event_ids), event_ids)),
    }
    return dense, frame_event_ids, audit
