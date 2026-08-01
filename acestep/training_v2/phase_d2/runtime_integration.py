"""Fail-closed attachment of lightweight D2 controls to a real frozen C25 DiT."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from acestep.training_v2.phase_d2.config import PhaseD2Config
from acestep.training_v2.phase_d2.model_integration import PhaseD2Injector


def unwrap_real_dit(decoder: Any) -> torch.nn.Module:
    """Resolve an ACE DiT through Fabric and PEFT wrappers or raise."""
    current = decoder
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        while hasattr(current, "_forward_module"):
            current = current._forward_module
        if hasattr(current, "layers") and hasattr(current, "config"):
            return current
        get_base = getattr(current, "get_base_model", None)
        if callable(get_base):
            current = get_base()
            continue
        base_model = getattr(current, "base_model", None)
        if base_model is not None:
            current = getattr(base_model, "model", base_model)
            continue
        break
    raise TypeError(f"Could not resolve AceStepDiTModel from {type(decoder).__name__}")


def verify_exact_peft_adapter(
    decoder: Any,
    adapter_file: str | Path,
    adapter_name: str,
) -> dict[str, Any]:
    """Verify that every saved C25 PEFT tensor equals the active runtime tensor."""
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file

    source = load_file(str(adapter_file))
    runtime = get_peft_model_state_dict(decoder, adapter_name=adapter_name)
    missing = sorted(set(source) - set(runtime))
    common = sorted(set(source) & set(runtime))
    differences = {
        key: float((source[key].cpu() - runtime[key].detach().cpu()).abs().max())
        for key in common
        if not torch.equal(source[key].cpu(), runtime[key].detach().cpu())
    }
    if missing or differences:
        raise RuntimeError(
            f"C25 PEFT verification failed: missing={missing[:5]} "
            f"differences={dict(list(differences.items())[:5])}"
        )
    return {
        "adapter_name": adapter_name,
        "source_tensor_count": len(source),
        "matched_tensor_count": len(common),
        "missing_tensor_count": len(missing),
        "max_abs_difference": max(differences.values(), default=0.0),
    }


def attach_frozen_c25_d2(
    model: torch.nn.Module,
    config: PhaseD2Config,
    adapter_dtype: torch.dtype | None = None,
) -> tuple[PhaseD2Injector, dict[str, Any]]:
    """Freeze the loaded C25 model and attach adapter-only D2 hooks."""
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        raise TypeError("C25 model has no decoder")
    dit = unwrap_real_dit(decoder)
    if int(dit.config.hidden_size) != config.hidden_size:
        raise ValueError(
            f"D2 hidden size {config.hidden_size} != real DiT {dit.config.hidden_size}"
        )
    for parameter in model.parameters():
        parameter.requires_grad = False
    injector = PhaseD2Injector(dit.layers, config)
    device = next(dit.parameters()).device
    dtype = adapter_dtype or next(dit.parameters()).dtype
    injector.to(device=device, dtype=dtype)
    injector.attach()
    adapter_parameters = injector.get_adapter_parameters()
    for parameter in adapter_parameters:
        parameter.requires_grad = True
    base_ids = {id(parameter) for parameter in model.parameters()}
    adapter_ids = {id(parameter) for parameter in adapter_parameters}
    if base_ids & adapter_ids or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("C25/D2 parameter ownership or freeze audit failed")
    audit = {
        "real_dit_type": type(dit).__name__,
        "real_dit_layers": len(dit.layers),
        "real_hidden_size": int(dit.config.hidden_size),
        "real_patch_size": int(dit.patch_size),
        "injection_layer_indices": list(injector.injection_indices),
        "base_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "base_trainable_count": 0,
        "d2_parameter_count": sum(parameter.numel() for parameter in adapter_parameters),
        "d2_trainable_count": sum(parameter.numel() for parameter in adapter_parameters),
        "d2_parameter_dtype": str(dtype),
        "parameter_id_overlap": 0,
    }
    return injector, audit
