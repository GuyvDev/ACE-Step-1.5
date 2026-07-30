"""Exact model loader for the controlled actual-new-chain Phase-D experiment.

Unlike the generic loader, this module tries exactly one attention backend and never
falls through to another implementation. Any failure is fatal.
"""
from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path
from typing import Any

import torch


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_phase_d_model_exact(
    *,
    checkpoint_dir: str | Path,
    variant: str,
    device: str,
    precision: str,
    attention_backend: str,
) -> tuple[Any, dict[str, str]]:
    if variant != "base":
        raise RuntimeError(f"controlled Phase-D requires model variant 'base', got {variant!r}")
    if device != "cuda":
        raise RuntimeError(f"controlled Phase-D requires device='cuda', got {device!r}")
    if precision != "bf16":
        raise RuntimeError(f"controlled Phase-D requires precision='bf16', got {precision!r}")
    if attention_backend != "sdpa":
        raise RuntimeError(
            f"controlled Phase-D uses exactly attention_backend='sdpa'; got {attention_backend!r}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    model_dir = Path(checkpoint_dir).resolve() / "acestep-v15-base"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"base model directory is missing: {model_dir}")
    required = ("config.json", "modeling_acestep_v15_base.py")
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"base model directory is incomplete: missing={missing}")

    from transformers import AutoModel

    # One call, one backend, no retry/fallback list.
    model = AutoModel.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    )
    actual_backend = str(getattr(model.config, "_attn_implementation", ""))
    if actual_backend != "sdpa":
        raise RuntimeError(
            f"model did not honor exact SDPA backend: requested='sdpa' actual={actual_backend!r}"
        )
    for parameter in model.parameters():
        parameter.requires_grad = False
    model = model.to(device="cuda", dtype=torch.bfloat16)
    model.eval()
    dynamic_path = Path(inspect.getfile(model.__class__)).resolve()
    if not dynamic_path.is_file():
        raise FileNotFoundError(f"dynamic model source is missing: {dynamic_path}")
    cache_raw = os.environ.get("HF_MODULES_CACHE", "")
    if not cache_raw:
        raise RuntimeError("controlled training requires an isolated HF_MODULES_CACHE")
    cache_root = Path(cache_raw).resolve()
    if not dynamic_path.is_relative_to(cache_root):
        raise RuntimeError(
            f"dynamic model source escaped isolated cache: {dynamic_path} not under {cache_root}"
        )
    checkpoint_source = (model_dir / "modeling_acestep_v15_base.py").resolve()
    dynamic_sha = _sha256_file(dynamic_path)
    checkpoint_sha = _sha256_file(checkpoint_source)
    if dynamic_sha != checkpoint_sha:
        raise RuntimeError(
            f"dynamic model source differs from checkpoint source: {dynamic_sha} != {checkpoint_sha}"
        )
    return model, {
        "model_dir": str(model_dir),
        "attention_backend": actual_backend,
        "requested_attention_backend": attention_backend,
        "fallback_used": False,
        "load_attempt_count": 1,
        "dynamic_module_path": str(dynamic_path),
        "dynamic_module_sha256": dynamic_sha,
        "checkpoint_source_path": str(checkpoint_source),
        "checkpoint_source_sha256": checkpoint_sha,
        "dynamic_matches_checkpoint_source": True,
        "isolated_hf_modules_cache": str(cache_root),
    }
