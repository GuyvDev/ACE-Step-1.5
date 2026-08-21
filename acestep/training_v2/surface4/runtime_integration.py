"""Fail-closed attachment of Control Surface 4 to the frozen base runtime."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any

import torch

from acestep.training_v2.phase_d2.runtime_integration import (
    unwrap_real_dit,
    verify_exact_peft_adapter,
)
from acestep.training_v2.surface4.performance_regulator import PerformanceRegulator


def assert_base_runtime(model: torch.nn.Module) -> dict[str, Any]:
    """Assert that the loaded condition model and decoder come from models/base."""
    wrapped_decoder = getattr(model, "decoder", None)
    if wrapped_decoder is None:
        raise TypeError("runtime model has no decoder")
    get_base_model = getattr(wrapped_decoder, "get_base_model", None)
    decoder_candidate = (
        get_base_model() if callable(get_base_model) else wrapped_decoder
    )
    decoder = unwrap_real_dit(decoder_candidate)
    model_source = Path(inspect.getsourcefile(type(model)) or "").resolve()
    decoder_source = Path(inspect.getsourcefile(type(decoder)) or "").resolve()
    base_source = (
        Path(__file__).resolve().parents[2]
        / "models/base/modeling_acestep_v15_base.py"
    )
    base_hash = hashlib.sha256(base_source.read_bytes()).hexdigest()
    for source in (model_source, decoder_source):
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        is_direct_base = "/acestep/models/base/" in source.as_posix()
        is_synced_base_cache = (
            "transformers_modules/acestep_hyphen_v15_hyphen_base/"
            in source.as_posix()
            and source_hash == base_hash
        )
        if not (is_direct_base or is_synced_base_cache):
            raise RuntimeError(f"STOP Surface 4 wrong runtime class: {source}")
    runtime_hash = hashlib.sha256(decoder_source.read_bytes()).hexdigest()
    return {
        "runtime_variant": "models/base",
        "model_module": type(model).__module__,
        "decoder_module": type(decoder).__module__,
        "model_source": str(model_source),
        "decoder_source": str(decoder_source),
        "base_authority_source": str(base_source),
        "base_authority_sha256": base_hash,
        "runtime_source_sha256": runtime_hash,
        "runtime_source_byte_identical_to_base": runtime_hash == base_hash,
    }


def attach_frozen_c25_surface4(
    model: torch.nn.Module,
    regulator_dtype: torch.dtype = torch.float32,
) -> tuple[PerformanceRegulator, dict[str, Any]]:
    """Freeze C25 and attach only the 43,008-parameter frame regulator."""
    provenance = assert_base_runtime(model)
    if hasattr(model, "performance_regulator"):
        raise RuntimeError("Surface 4 regulator is already attached")
    dit = unwrap_real_dit(model.decoder)
    if int(dit.patch_size) != 2:
        raise RuntimeError(f"STOP unexpected DiT patch_size={dit.patch_size}")
    base_parameters = list(model.parameters())
    for parameter in base_parameters:
        parameter.requires_grad_(False)
    regulator = PerformanceRegulator()
    device = next(dit.parameters()).device
    regulator.to(device=device, dtype=regulator_dtype)
    model.performance_regulator = regulator
    for parameter in regulator.parameters():
        parameter.requires_grad_(True)
    base_ids = {id(parameter) for parameter in base_parameters}
    regulator_ids = {id(parameter) for parameter in regulator.parameters()}
    unintended = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in regulator_ids
    ]
    if base_ids & regulator_ids or unintended:
        raise RuntimeError(
            f"STOP Surface 4 parameter ownership failure; unintended={unintended[:8]}"
        )
    audit = {
        **provenance,
        "injection": (
            "after prepare_condition; residual added to existing context[..., :64] "
            "before decoder.proj_in"
        ),
        "prepare_condition_no_grad_preserved": True,
        "real_patch_size": int(dit.patch_size),
        "base_parameter_count": sum(parameter.numel() for parameter in base_parameters),
        "base_trainable_count": 0,
        "regulator_parameter_count": regulator.trainable_parameter_count,
        "regulator_trainable_count": sum(
            parameter.numel() for parameter in regulator.parameters()
            if parameter.requires_grad
        ),
        "parameter_id_overlap": len(base_ids & regulator_ids),
        "unintended_trainable_names": unintended,
    }
    if audit["regulator_parameter_count"] != 43008:
        raise RuntimeError(
            "STOP unexpected Surface 4 parameter count: "
            f"{audit['regulator_parameter_count']} != 43008"
        )
    return regulator, audit


__all__ = [
    "assert_base_runtime",
    "attach_frozen_c25_surface4",
    "verify_exact_peft_adapter",
]
