"""
Preservation Analysis for Phase D2.

Helpers to compute intervention windows and measure preservation outside windows.
"""

from __future__ import annotations

from typing import Tuple, Optional
import torch
import torch.nn.functional as F


def make_window_mask(
    n_frames: int,
    start_sec: float,
    end_sec: float,
    frame_rate: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create binary window mask in frame space.

    Args:
        n_frames: Total number of frames.
        start_sec: Start time of window in seconds.
        end_sec: End time of window in seconds.
        frame_rate: Frame rate in Hz.
        device: Device for output tensor (default CPU).

    Returns:
        [n_frames] bool tensor (True = inside window, False = outside).
    """
    if device is None:
        device = torch.device("cpu")

    frame_times = torch.arange(n_frames, dtype=torch.float32, device=device) / frame_rate

    mask = (frame_times >= start_sec) & (frame_times < end_sec)

    return mask


def expand_window_mask(
    mask: torch.Tensor,
    margin_frames: int = 0,
) -> torch.Tensor:
    """Expand a window mask by adding margin frames on both sides.

    Args:
        mask: [n_frames] bool tensor.
        margin_frames: Number of frames to expand on each side.

    Returns:
        Expanded [n_frames] bool tensor.
    """
    if margin_frames <= 0:
        return mask

    # Dilation: any frame within margin_frames of a True position becomes True
    # Simple approach: max-pool with kernel size 2*margin+1
    mask_float = mask.float().unsqueeze(0).unsqueeze(0)  # [1, 1, 1, n_frames]

    # Actually, we want 1D dilation. Use F.max_pool1d.
    expanded = F.max_pool1d(
        mask_float,
        kernel_size=2 * margin_frames + 1,
        stride=1,
        padding=margin_frames,
    )

    return expanded.squeeze(0).squeeze(0).bool()


def measure_preservation(
    d2_output: torch.Tensor,
    c25_output: torch.Tensor,
    window_mask: torch.Tensor,
    metric: str = "l1",
) -> Tuple[float, float, float]:
    """Measure preservation inside and outside window.

    Args:
        d2_output: [B, T, H] or [T, H] D2 output.
        c25_output: [B, T, H] or [T, H] C25 reference.
        window_mask: [B, T] or [T] bool (True = inside, False = outside).
        metric: "l1", "l2", "max", or "cosine".

    Returns:
        Tuple of (inside_dist, outside_dist, ratio).
    """
    # Handle batched or unbatched
    if d2_output.dim() == 3:
        d2_output = d2_output.reshape(-1, d2_output.shape[-1])
        c25_output = c25_output.reshape(-1, c25_output.shape[-1])
        window_mask = window_mask.reshape(-1)

    if metric == "l1":
        distances = torch.abs(d2_output - c25_output).mean(dim=-1)
    elif metric == "l2":
        distances = torch.norm(d2_output - c25_output, p=2, dim=-1)
    elif metric == "max":
        distances = torch.abs(d2_output - c25_output).max(dim=-1)[0]
    elif metric == "cosine":
        sim = F.cosine_similarity(d2_output, c25_output, dim=-1)
        distances = 1.0 - sim
    else:
        raise ValueError(f"Unknown metric: {metric}")

    inside_mask = window_mask.float()
    outside_mask = (~window_mask).float()

    inside_dist = (distances * inside_mask).sum() / (inside_mask.sum() + 1e-6)
    outside_dist = (distances * outside_mask).sum() / (outside_mask.sum() + 1e-6)

    ratio = (inside_dist / (outside_dist + 1e-6)).item()

    return inside_dist.item(), outside_dist.item(), ratio
