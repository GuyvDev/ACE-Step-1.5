"""ACE payload contract for Phase D Timing v2.

Resolves ``attention_mask`` and enforces the non-lokr / no-leakage invariants
permanently. Every payload written or loaded passes through ``validate_payload``
and fails closed.
"""

from __future__ import annotations

from typing import Any

import torch

SCHEMA = "phase_d_timing_v2_ace_payload_v1"

# attention_mask rule, verified on 60/60 clean D2-1 payloads:
#   shape == (T,) | dtype bfloat16 | unique values == [1.0]
#   == context_latents[:, 64:][:, 0]  (60/60)
#   == ones(T)                        (60/60)
# Both candidate rules coincide; the chunk_masks form is used because it is
# derived from the payload rather than assumed.
ATTENTION_MASK_RULE = "context_latents[:, 64:][:, 0].to(torch.bfloat16)"

SILENCE_TOLERANCE = 0.05          # bf16 rounding on the silence latent
MAX_CONTEXT_TARGET_CORR = 0.5     # leakage tripwire


def normalise_silence(silence_latent: torch.Tensor, frames: int) -> torch.Tensor:
    """Return the first ``frames`` silence rows as ``[frames, 64]``.

    The handler exposes ``[1, T, 64]``; the raw ``silence_latent.pt`` file is
    ``[64, T]``. Both are accepted so the leakage check cannot be defeated by
    an orientation mismatch.
    """
    tensor = silence_latent.detach().to('cpu', torch.float32)
    if tensor.ndim == 3:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"silence_latent must be [T,64] or [64,T], got {tuple(silence_latent.shape)}")
    if tensor.shape[0] == 64 and tensor.shape[1] != 64:
        tensor = tensor.T
    if tensor.shape[1] != 64:
        raise ValueError(f"silence_latent channel dim must be 64, got {tuple(tensor.shape)}")
    if tensor.shape[0] < frames:
        raise ValueError(f"silence_latent has {tensor.shape[0]} frames, need {frames}")
    return tensor[:frames]


def derive_attention_mask(context_latents: torch.Tensor) -> torch.Tensor:
    """attention_mask from the chunk_masks half of context_latents."""
    if context_latents.ndim != 2 or context_latents.shape[1] != 128:
        raise ValueError(f"context_latents must be [T,128], got {tuple(context_latents.shape)}")
    return context_latents[:, 64:][:, 0].to(torch.bfloat16).contiguous()


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.detach().to('cpu', torch.float32).flatten()
    y = b.detach().to('cpu', torch.float32).flatten()
    n = min(len(x), len(y))
    x, y = x[:n] - x[:n].mean(), y[:n] - y[:n].mean()
    denominator = float(torch.sqrt((x**2).sum() * (y**2).sum()))
    return abs(float((x * y).sum()) / denominator) if denominator > 1e-12 else 0.0


def validate_payload(payload: dict[str, Any], silence_latent: torch.Tensor, mode: str) -> dict[str, Any]:
    """Fail closed on schema, shape, dtype and leakage violations."""
    if mode == "lokr":
        raise RuntimeError(
            "mode='lokr' routes target_latents into BOTH encoder reference-audio "
            "conditioning and context_latents. Forbidden for Phase D Timing v2."
        )

    for key in ("target_latents", "encoder_hidden_states", "encoder_attention_mask", "context_latents"):
        if key not in payload:
            raise RuntimeError(f"payload missing required tensor: {key}")

    target = payload["target_latents"]
    context = payload["context_latents"]
    frames = target.shape[0]

    if target.ndim != 2 or target.shape[1] != 64:
        raise RuntimeError(f"target_latents must be [T,64], got {tuple(target.shape)}")
    if context.shape != (frames, 128):
        raise RuntimeError(f"context_latents must be [{frames},128], got {tuple(context.shape)}")
    if payload["encoder_hidden_states"].ndim != 2 or payload["encoder_hidden_states"].shape[1] != 2048:
        raise RuntimeError(f"encoder_hidden_states must be [L,2048], got {tuple(payload['encoder_hidden_states'].shape)}")
    if payload["encoder_attention_mask"].shape[0] != payload["encoder_hidden_states"].shape[0]:
        raise RuntimeError("encoder mask/state length mismatch")

    for key in ("target_latents", "encoder_hidden_states", "context_latents"):
        tensor = payload[key]
        if not torch.isfinite(tensor.float()).all():
            raise RuntimeError(f"{key} contains NaN or Inf")

    # --- leakage invariants -------------------------------------------------
    chunk = context[:, 64:]
    if not bool((chunk.float() == 1.0).all()):
        raise RuntimeError("context_latents[:, 64:] must be all ones (chunk_masks)")

    src = context[:, :64].detach().to('cpu', torch.float32)
    reference = normalise_silence(silence_latent, frames)
    deviation = float((src - reference).abs().max())
    if deviation > SILENCE_TOLERANCE:
        raise RuntimeError(
            f"context_latents[:, :64] deviates from the approved silence latent by {deviation:.4f} "
            f"(> {SILENCE_TOLERANCE}); target audio may have leaked into the context branch"
        )

    correlation = _corr(src, target)
    if correlation >= MAX_CONTEXT_TARGET_CORR:
        raise RuntimeError(f"context/target correlation {correlation:.3f} >= {MAX_CONTEXT_TARGET_CORR}: leakage")

    mask = payload.get("attention_mask")
    if mask is None:
        mask = derive_attention_mask(context)
        payload["attention_mask"] = mask
    if mask.shape != (frames,):
        raise RuntimeError(f"attention_mask must be [{frames}], got {tuple(mask.shape)}")
    if mask.dtype != torch.bfloat16:
        raise RuntimeError(f"attention_mask must be bfloat16, got {mask.dtype}")
    if not bool((mask.float() == 1.0).all()):
        raise RuntimeError("attention_mask must be all ones for unpadded sections")

    return {
        "schema": SCHEMA,
        "frames": int(frames),
        "encoder_tokens": int(payload["encoder_hidden_states"].shape[0]),
        "encoder_valid_tokens": int(payload["encoder_attention_mask"].sum()),
        "attention_mask_rule": ATTENTION_MASK_RULE,
        "mode": mode,
        "silence_max_deviation": round(deviation, 6),
        "context_target_correlation": round(correlation, 6),
        "leakage_checks_passed": True,
    }


__all__ = ["SCHEMA", "ATTENTION_MASK_RULE", "derive_attention_mask", "validate_payload", "normalise_silence",
           "SILENCE_TOLERANCE", "MAX_CONTEXT_TARGET_CORR"]
