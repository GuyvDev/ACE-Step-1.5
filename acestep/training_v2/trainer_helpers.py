"""
Trainer helper functions for FixedLoRATrainer.

Contains checkpoint save/resume, adapter verification, memory
configuration, and module wrapper introspection -- extracted from
``trainer_fixed.py`` to keep it under the LOC limit.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple

import torch
import torch.nn as nn

from acestep.training.lora_checkpoint import (
    load_training_checkpoint,
    save_lora_weights,
)
from acestep.training.lokr_utils import (
    save_lokr_weights,
    load_lokr_weights,
)
from acestep.training_v2.ui import TrainingUpdate
from acestep.training_v2.mert_conditioning import load_saved_bridge
from acestep.training_v2.timing_conditioning import load_saved_timing_module

logger = logging.getLogger(__name__)


def _unwrap_runtime_decoder(decoder: Any) -> Any:
    """Strip Fabric wrappers while preserving the PEFT wrapper."""
    while hasattr(decoder, "_forward_module"):
        decoder = decoder._forward_module
    return decoder


def _collect_decoder_timing_state(module: Any) -> dict[str, torch.Tensor]:
    """Collect decoder-side timing-attention weights for checkpointing."""
    model = getattr(module, "model", None)
    decoder = getattr(model, "decoder", None) if model is not None else None
    if decoder is None:
        return {}
    raw_decoder = _unwrap_runtime_decoder(decoder)
    phrase_modulation = bool(getattr(module, "_enable_phrase_modulation", False))
    timing_state = {
        name: tensor.detach().cpu()
        for name, tensor in raw_decoder.state_dict().items()
        if (
            ("timing_" in name)
            and "lora_" not in name
            and "hada_" not in name
            and "lokr_" not in name
            and (
                phrase_modulation
                or "timing_cross_attn" in name
                or name.endswith("timing_attn_gate")
            )
        )
    }
    return timing_state


def _child_module(obj: Any, part: str) -> Any:
    if isinstance(obj, (nn.ModuleList, nn.Sequential)) and part.isdigit():
        return obj[int(part)]
    if isinstance(obj, (nn.ModuleDict, nn.ParameterDict)) and part in obj:
        return obj[part]
    return getattr(obj, part)


def _resolve_tensor_by_module_path(root: Any, key: str) -> tuple[torch.Tensor | None, str]:
    """Resolve one checkpoint tensor without permissive state-dict loading."""
    parts = key.split(".")
    target_paths = [parts]
    if key.startswith("base_model.model."):
        target_paths.append(key.removeprefix("base_model.model.").split("."))
    last_error = ""
    for candidate in target_paths:
        try:
            obj = root
            for part in candidate[:-1]:
                obj = _child_module(obj, part)
            target = getattr(obj, candidate[-1])
        except Exception as exc:
            last_error = f"path_not_found:{'.'.join(candidate)}:{exc}"
            continue
        if not isinstance(target, torch.Tensor):
            return None, f"target_not_tensor:{key}"
        return target, "resolved"
    return None, last_error or f"path_not_found:{key}"


def _copy_tensor_by_module_path(root: Any, key: str, tensor: torch.Tensor) -> tuple[bool, str]:
    """Copy one exact-shape tensor and verify an exact round trip."""
    target, reason = _resolve_tensor_by_module_path(root, key)
    if target is None:
        return False, reason
    if tuple(target.shape) != tuple(tensor.shape):
        return False, f"shape_mismatch:{key}:target={tuple(target.shape)}:source={tuple(tensor.shape)}"
    expected = tensor.to(device=target.device, dtype=target.dtype)
    with torch.no_grad():
        target.copy_(expected)
    max_diff = float((target.detach() - expected).abs().max().cpu()) if target.numel() else 0.0
    if max_diff != 0.0:
        return False, f"round_trip_difference:{key}:max_abs_difference={max_diff}"
    return True, "copied_exactly"


def _is_non_lora_decoder_timing_key(key: str) -> bool:
    return (
        "timing_" in key
        and ".lora_A." not in key
        and ".lora_B." not in key
        and ".lora_embedding_A." not in key
        and ".lora_embedding_B." not in key
    )


def _load_decoder_timing_state(module: Any, decoder_state: dict[str, torch.Tensor], source: Path) -> bool:
    """Restore every decoder timing tensor by exact name, shape, and value."""
    model = getattr(module, "model", None)
    decoder = getattr(model, "decoder", None) if model is not None else None
    if decoder is None or not decoder_state:
        return False
    raw_decoder = _unwrap_runtime_decoder(decoder)
    invalid = sorted(key for key in decoder_state if not _is_non_lora_decoder_timing_key(key))
    if invalid:
        raise RuntimeError(f"timing checkpoint contains forbidden keys: {invalid[:20]}")
    loaded: list[str] = []
    failed: dict[str, str] = {}
    for key in sorted(decoder_state):
        ok, reason = _copy_tensor_by_module_path(raw_decoder, key, decoder_state[key])
        if ok:
            loaded.append(key)
        else:
            failed[key] = reason
    expected = sorted(decoder_state)
    if failed or loaded != expected:
        raise RuntimeError(
            f"exact timing checkpoint load failed: expected={len(expected)} loaded={len(loaded)} "
            f"failed={dict(list(failed.items())[:20])}"
        )
    logger.info("[OK] Exact-loaded %d/%d decoder timing tensors from %s", len(loaded), len(expected), source)
    return True


def _load_module_state_shape_compatible(
    target_module: nn.Module,
    source_state: dict[str, torch.Tensor],
    source: Path,
    label: str,
) -> bool:
    """Exact-load an optional timing submodule; partial compatibility is forbidden."""
    target_state = target_module.state_dict()
    source_names = sorted(source_state)
    target_names = sorted(target_state)
    if source_names != target_names:
        missing = sorted(set(target_names) - set(source_names))
        unexpected = sorted(set(source_names) - set(target_names))
        raise RuntimeError(
            f"exact {label} key mismatch from {source}: missing={missing[:20]} unexpected={unexpected[:20]}"
        )
    mismatched = {
        key: (tuple(target_state[key].shape), tuple(source_state[key].shape))
        for key in source_names
        if tuple(target_state[key].shape) != tuple(source_state[key].shape)
    }
    if mismatched:
        raise RuntimeError(f"exact {label} shape mismatch from {source}: {dict(list(mismatched.items())[:20])}")
    target_module.load_state_dict(source_state, strict=True)
    loaded_state = target_module.state_dict()
    differences = {
        key: float((loaded_state[key].detach().cpu() - source_state[key].detach().cpu().to(loaded_state[key].dtype)).abs().max())
        for key in source_names
        if loaded_state[key].numel()
    }
    max_difference = max(differences.values(), default=0.0)
    if max_difference != 0.0:
        raise RuntimeError(f"exact {label} value mismatch from {source}: max_abs_difference={max_difference}")
    logger.info("[OK] Exact-loaded %s from %s (%d tensors, max_diff=0)", label, source, len(source_names))
    return True


def _save_bridge_module(module: Any, output_dir: str) -> None:
    voice_conditioner = getattr(module, "voice_conditioner", None)
    if voice_conditioner is None:
        return
    bridge_path = os.path.join(output_dir, "mert_bridge.pt")
    payload = {
        "state_dict": voice_conditioner.state_dict(),
        "config": voice_conditioner.export_config(),
    }
    torch.save(payload, bridge_path)
    logger.info("[OK] Saved MERT bridge weights to %s", bridge_path)


def _load_bridge_module(module: Any, checkpoint_dir: Path) -> bool:
    voice_conditioner = getattr(module, "voice_conditioner", None)
    bridge_path = checkpoint_dir / "mert_bridge.pt"
    if voice_conditioner is None or not bridge_path.exists():
        return False
    payload = load_saved_bridge(str(bridge_path), map_location=module.device)
    voice_conditioner.load_state_dict(payload["state_dict"], strict=True)
    logger.info("[OK] Loaded MERT bridge weights from %s", bridge_path)
    return True


def _save_identity_v5_module(module: Any, output_dir: str) -> None:
    """Persist V5 identity modules in the inference-compatible schema."""
    if not bool(getattr(module, "_identity_v5_enabled", False)):
        return
    projection = getattr(module, "v5_singer_projection", None)
    blocks = getattr(module, "v5_block_adapters", None)
    fragment = getattr(module, "v5_fragment_attention", None)
    prototype = getattr(module, "identity_v5_global_prototype", None)
    block_ids = [int(v) for v in getattr(module, "_identity_v5_block_ids", [])]
    if projection is None or blocks is None or fragment is None or prototype is None or not block_ids:
        raise RuntimeError("V5 checkpoint requested with incomplete runtime modules")
    payload = {
        "hidden_dim": int(projection.out_features),
        "singer_dim": int(projection.in_features),
        "singer_projection": projection.state_dict(),
        "block_ids": block_ids,
        "block_adapters": blocks.state_dict(),
        "crop_types": int(fragment.crop_embedding.num_embeddings),
        "fragment_attention": fragment.state_dict(),
        "global_prototype": prototype.detach().cpu(),
        "global_enabled": bool(getattr(module, "_identity_v5_global_enabled", False)),
        "local_enabled": bool(getattr(module, "_identity_v5_local_enabled", False)),
    }
    target = os.path.join(output_dir, "identity_v5_modules.pt")
    torch.save(payload, target)
    logger.info("[OK] Saved V5 identity modules to %s", target)


def _load_identity_v5_module(module: Any, checkpoint_dir: Path) -> bool:
    """Exact-load V5 identity modules into a configured V5 runtime."""
    path = checkpoint_dir / "identity_v5_modules.pt"
    enabled = bool(getattr(module, "_identity_v5_enabled", False))
    if not path.exists():
        if enabled:
            raise FileNotFoundError(f"V5 runtime requires missing payload: {path}")
        return False
    if not enabled:
        raise RuntimeError(f"checkpoint contains V5 state but runtime V5 is disabled: {path}")
    payload = torch.load(path, map_location=module.device, weights_only=False)
    required = {
        "hidden_dim", "singer_dim", "singer_projection", "block_ids",
        "block_adapters", "crop_types", "fragment_attention",
        "global_prototype", "global_enabled", "local_enabled",
    }
    missing = sorted(required - set(payload))
    unexpected = sorted(set(payload) - required)
    if missing or unexpected:
        raise RuntimeError(f"V5 schema mismatch: missing={missing} unexpected={unexpected}")
    projection = module.v5_singer_projection
    blocks = module.v5_block_adapters
    fragment = module.v5_fragment_attention
    prototype = module.identity_v5_global_prototype
    contract = {
        "hidden_dim": (int(projection.out_features), int(payload["hidden_dim"])),
        "singer_dim": (int(projection.in_features), int(payload["singer_dim"])),
        "block_ids": ([int(v) for v in module._identity_v5_block_ids], [int(v) for v in payload["block_ids"]]),
        "crop_types": (int(fragment.crop_embedding.num_embeddings), int(payload["crop_types"])),
        "global_enabled": (bool(module._identity_v5_global_enabled), bool(payload["global_enabled"])),
        "local_enabled": (bool(module._identity_v5_local_enabled), bool(payload["local_enabled"])),
    }
    mismatched = {k: v for k, v in contract.items() if v[0] != v[1]}
    if mismatched:
        raise RuntimeError(f"V5 contract mismatch: {mismatched}")
    _load_module_state_shape_compatible(projection, payload["singer_projection"], path, "V5 singer projection")
    _load_module_state_shape_compatible(blocks, payload["block_adapters"], path, "V5 block adapters")
    _load_module_state_shape_compatible(fragment, payload["fragment_attention"], path, "V5 fragment attention")
    source = payload["global_prototype"].to(device=prototype.device, dtype=prototype.dtype)
    if tuple(source.shape) != tuple(prototype.shape):
        raise RuntimeError(f"V5 prototype shape mismatch: runtime={tuple(prototype.shape)} source={tuple(source.shape)}")
    with torch.no_grad():
        prototype.copy_(source)
    max_diff = float((prototype.detach() - source).abs().max().cpu()) if prototype.numel() else 0.0
    if max_diff != 0.0:
        raise RuntimeError(f"V5 prototype value mismatch: max_abs_difference={max_diff}")
    logger.info("[OK] Exact-loaded V5 identity modules from %s", path)
    return True


def _save_timing_module(module: Any, output_dir: str) -> None:
    """Save timing state only for runs that explicitly enable the branch."""
    if not bool(getattr(module.training_config, "enable_timing_branch", False)):
        return
    timing_encoder = getattr(module, "timing_encoder", None)
    decoder_timing_supervisor = getattr(module, "decoder_timing_supervisor", None)
    timing_predictor = getattr(module, "performance_timing_predictor", None)
    decoder_expressivity_supervisor = getattr(module, "decoder_expressivity_supervisor", None)
    decoder_timing_state = _collect_decoder_timing_state(module)
    if (
        timing_encoder is None
        and decoder_timing_supervisor is None
        and timing_predictor is None
        and decoder_expressivity_supervisor is None
        and not decoder_timing_state
    ):
        return
    timing_path = os.path.join(output_dir, "timing_branch.pt")
    payload = {}
    if timing_encoder is not None:
        payload["state_dict"] = timing_encoder.state_dict()
        payload["config"] = timing_encoder.export_config()
    if decoder_timing_supervisor is not None:
        payload["decoder_state_dict"] = decoder_timing_supervisor.state_dict()
    if timing_predictor is not None:
        payload["predictor_state_dict"] = timing_predictor.state_dict()
        payload["predictor_config"] = timing_predictor.export_config()
    if decoder_expressivity_supervisor is not None:
        payload["expressivity_state_dict"] = decoder_expressivity_supervisor.state_dict()
    if decoder_timing_state:
        payload["decoder_model_state_dict"] = decoder_timing_state
    torch.save(payload, timing_path)
    logger.info("[OK] Saved timing branch weights to %s", timing_path)


def _load_timing_module(module: Any, checkpoint_dir: Path) -> bool:
    """Exact-load every timing component present in a checkpoint."""
    timing_path = checkpoint_dir / "timing_branch.pt"
    if not timing_path.exists():
        return False
    payload = load_saved_timing_module(str(timing_path), map_location=module.device)
    components = (
        ("state_dict", getattr(module, "timing_encoder", None), "timing branch"),
        ("decoder_state_dict", getattr(module, "decoder_timing_supervisor", None), "decoder timing supervisor"),
        ("predictor_state_dict", getattr(module, "performance_timing_predictor", None), "timing predictor"),
        ("expressivity_state_dict", getattr(module, "decoder_expressivity_supervisor", None), "decoder expressivity supervisor"),
    )
    loaded_any = False
    for state_key, target, label in components:
        if state_key not in payload:
            continue
        if target is None:
            raise RuntimeError(f"{timing_path} contains {state_key}, but runtime has no {label}")
        _load_module_state_shape_compatible(target, payload[state_key], timing_path, label)
        loaded_any = True
    if "decoder_model_state_dict" in payload:
        if not _load_decoder_timing_state(module, payload["decoder_model_state_dict"], timing_path):
            raise RuntimeError(f"{timing_path} contains decoder timing state, but runtime decoder is unavailable")
        loaded_any = True
    if not loaded_any:
        raise RuntimeError(f"timing checkpoint has no loadable state: {timing_path}")
    return True


def _verify_timing_module_exact(module: Any, checkpoint_dir: Path) -> Dict[str, Any]:
    """Verify every expected timing tensor after a controlled D1 resume."""
    timing_path = checkpoint_dir / "timing_branch.pt"
    if not timing_path.is_file():
        raise FileNotFoundError(f"required parent timing state is missing: {timing_path}")
    payload = load_saved_timing_module(str(timing_path), map_location=module.device)
    pairs = (
        ("state_dict", getattr(module, "timing_encoder", None), "timing_encoder"),
        ("decoder_state_dict", getattr(module, "decoder_timing_supervisor", None), "decoder_timing_supervisor"),
        ("predictor_state_dict", getattr(module, "performance_timing_predictor", None), "timing_predictor"),
        ("expressivity_state_dict", getattr(module, "decoder_expressivity_supervisor", None), "expressivity_supervisor"),
    )
    expected_component_keys = {key for key, target, _ in pairs if target is not None}
    decoder_runtime = _collect_decoder_timing_state(module)
    if decoder_runtime:
        expected_component_keys.add("decoder_model_state_dict")
    actual_component_keys = {
        key for key in payload
        if key.endswith("state_dict") or key == "state_dict"
    }
    if actual_component_keys != expected_component_keys:
        raise RuntimeError(
            f"timing component set mismatch: expected={sorted(expected_component_keys)} "
            f"actual={sorted(actual_component_keys)}"
        )
    source_flat: Dict[str, torch.Tensor] = {}
    runtime_flat: Dict[str, torch.Tensor] = {}
    for state_key, target, label in pairs:
        if target is None:
            continue
        source_state = payload.get(state_key)
        if not isinstance(source_state, dict):
            raise RuntimeError(f"required timing state {state_key} is invalid in {timing_path}")
        runtime_state = target.state_dict()
        if set(source_state) != set(runtime_state):
            raise RuntimeError(
                f"{label} key mismatch: missing={sorted(set(source_state)-set(runtime_state))[:8]} "
                f"unexpected={sorted(set(runtime_state)-set(source_state))[:8]}"
            )
        for name, source in source_state.items():
            if not isinstance(source, torch.Tensor) or not isinstance(runtime_state[name], torch.Tensor):
                raise RuntimeError(f"non-tensor timing state: {label}.{name}")
            source_flat[f"{state_key}.{name}"] = source.detach().cpu()
            runtime_flat[f"{state_key}.{name}"] = runtime_state[name].detach().cpu()
    if decoder_runtime:
        source_decoder = payload.get("decoder_model_state_dict")
        if not isinstance(source_decoder, dict) or set(source_decoder) != set(decoder_runtime):
            raise RuntimeError("decoder timing-state key set differs after D1 resume")
        for name, source in source_decoder.items():
            if not isinstance(source, torch.Tensor):
                raise RuntimeError(f"non-tensor decoder timing state: {name}")
            source_flat[f"decoder_model_state_dict.{name}"] = source.detach().cpu()
            runtime_flat[f"decoder_model_state_dict.{name}"] = decoder_runtime[name].detach().cpu()
    differences = []
    mismatched = []
    for name in sorted(source_flat):
        source = source_flat[name]
        runtime = runtime_flat[name]
        if source.shape != runtime.shape or source.dtype != runtime.dtype:
            mismatched.append(name)
            continue
        difference = float((source - runtime).abs().max().item()) if source.numel() else 0.0
        differences.append(difference)
        if difference != 0.0:
            mismatched.append(name)
    if mismatched:
        raise RuntimeError(f"D1 parent timing tensors differ after load: {mismatched[:8]}")
    return {
        "loaded": True,
        "path": str(timing_path.resolve()),
        "file_sha256": hashlib.sha256(timing_path.read_bytes()).hexdigest(),
        "component_keys": sorted(expected_component_keys),
        "tensor_count": len(source_flat),
        "tensor_names": sorted(source_flat),
        "max_abs_difference": max(differences, default=0.0),
        "source_sha256": _tensor_state_sha256(source_flat),
        "runtime_sha256": _tensor_state_sha256(runtime_flat),
    }


# ---------------------------------------------------------------------------
# Module introspection
# ---------------------------------------------------------------------------


def iter_module_wrappers(module: nn.Module) -> list:
    """Collect wrapper-chain modules (Fabric/PEFT/compile wrappers).

    Walks ``_forward_module``, ``_orig_mod``, ``base_model``, ``model``,
    and ``module`` attributes to find all wrapped layers.  Ported from
    ACE-Step's ``trainer.py`` to ensure parity.
    """
    modules: list = []
    stack = [module]
    visited: set = set()
    while stack:
        current = stack.pop()
        if not isinstance(current, nn.Module):
            continue
        mid = id(current)
        if mid in visited:
            continue
        visited.add(mid)
        modules.append(current)
        for attr in ("_forward_module", "_orig_mod", "base_model", "model", "module"):
            child = getattr(current, attr, None)
            if isinstance(child, nn.Module):
                stack.append(child)
    return modules


# ---------------------------------------------------------------------------
# Memory configuration
# ---------------------------------------------------------------------------


def configure_memory_features(decoder: nn.Module, *, strict: bool = False) -> tuple:
    """Enable gradient checkpointing, disable use_cache, and enable
    input_require_grads across all wrapper layers of *decoder*.

    Mirrors ACE-Step's ``_configure_training_memory_features()`` exactly
    so that VRAM usage is identical.

    Returns:
        ``(checkpointing_enabled, cache_disabled, input_grads_enabled)``
    """
    ckpt_enabled = False
    cache_disabled = False
    input_grads_enabled = False

    for mod in iter_module_wrappers(decoder):
        # 1. Gradient checkpointing
        if hasattr(mod, "gradient_checkpointing_enable"):
            try:
                mod.gradient_checkpointing_enable()
                ckpt_enabled = True
            except Exception as exc:
                if strict:
                    raise RuntimeError(
                        f"gradient_checkpointing_enable failed for {type(mod).__name__}"
                    ) from exc
        elif hasattr(mod, "gradient_checkpointing"):
            try:
                mod.gradient_checkpointing = True
                ckpt_enabled = True
            except Exception as exc:
                if strict:
                    raise RuntimeError(
                        f"setting gradient_checkpointing failed for {type(mod).__name__}"
                    ) from exc

        # 2. PEFT + checkpointing needs input embeddings to carry grads
        if hasattr(mod, "enable_input_require_grads"):
            try:
                mod.enable_input_require_grads()
                hook_ok = bool(getattr(mod, "_acestep_input_grads_hook_enabled", False))
                has_hook = getattr(mod, "_require_grads_hook", None) is not None
                if hook_ok or has_hook:
                    input_grads_enabled = True
            except Exception as exc:
                if strict:
                    raise RuntimeError(
                        f"enable_input_require_grads failed for {type(mod).__name__}"
                    ) from exc

        # 3. Disable use_cache (frees KV-cache memory)
        cfg = getattr(mod, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            try:
                if getattr(cfg, "use_cache", None) is not False:
                    cfg.use_cache = False
                    cache_disabled = True
            except Exception as exc:
                if strict:
                    raise RuntimeError(
                        f"disabling use_cache failed for {type(mod).__name__}"
                    ) from exc

    return ckpt_enabled, cache_disabled, input_grads_enabled


def offload_non_decoder(model: nn.Module) -> int:
    """Move encoder/VAE/non-decoder submodules to CPU. Returns count offloaded."""
    count = 0
    for name in (
        "music_encoder",
        "lyric_encoder",
        "timbre_encoder",
        "condition_projection",
        "vae",
        "text_encoder",
        "attention_pooler",
    ):
        sub = getattr(model, name, None)
        if sub is not None and isinstance(sub, nn.Module):
            sub.to("cpu")
            count += 1
    return count


# ---------------------------------------------------------------------------
# Adapter-aware save helpers
# ---------------------------------------------------------------------------


def save_adapter_flat(trainer: Any, output_dir: str) -> None:
    """Save adapter weights directly into *output_dir* (no nesting).

    Writes ``adapter_config.json`` and ``adapter_model.safetensors``
    (or LoKR equivalent) directly into *output_dir* so that
    inference tools can point straight at this directory.
    """
    module = trainer.module
    assert module is not None
    os.makedirs(output_dir, exist_ok=True)

    if trainer.adapter_type == "lokr":
        if module.lycoris_net is None:
            logger.error(
                "[BUG] adapter_type is 'lokr' but lycoris_net is None -- "
                "cannot save LoKR weights.  This indicates a configuration or "
                "injection error.  Refusing to silently save as LoRA."
            )
            raise RuntimeError(
                "LoKR adapter type was requested but no LyCORIS network is "
                "attached to the training module.  Cannot save weights."
            )
        lokr_meta = {"lokr_config": module.adapter_config.to_dict()}
        save_lokr_weights(module.lycoris_net, output_dir, metadata=lokr_meta)
    else:
        # Access the decoder directly (PeftModel after LoRA injection,
        # possibly wrapped by Fabric's _FabricModule after setup).
        # Do NOT use _unwrap_decoder here -- that function strips the PEFT
        # wrapper and returns the base DiT model, causing save_pretrained()
        # to write the full model instead of the adapter-only files.
        raw_decoder = module.model.decoder
        # Strip Fabric wrappers only (_forward_module chain).
        while hasattr(raw_decoder, "_forward_module"):
            raw_decoder = raw_decoder._forward_module
        controlled = getattr(module.training_config, "phase_d_resume_adapter", None) is not None
        if not hasattr(raw_decoder, "save_pretrained"):
            if controlled:
                raise RuntimeError(
                    "controlled Phase-D requires a PEFT decoder with save_pretrained; "
                    f"got {type(raw_decoder).__name__}"
                )
            save_lora_weights(module.model, output_dir)
        else:
            raw_decoder.save_pretrained(output_dir)
            logger.info("[OK] LoRA adapter saved to %s", output_dir)
        if controlled:
            required = (
                Path(output_dir) / "adapter_config.json",
                Path(output_dir) / "adapter_model.safetensors",
            )
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise RuntimeError(f"controlled adapter save is incomplete: {missing}")

    _save_bridge_module(module, output_dir)
    _save_identity_v5_module(module, output_dir)
    _save_timing_module(module, output_dir)


def save_checkpoint(
    trainer: Any,
    optimizer: Any,
    scheduler: Any,
    epoch: int,
    global_step: int,
    ckpt_dir: str,
    extra_state: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a resumable checkpoint that is also inference-ready.

    Adapter files (``adapter_config.json``, ``adapter_model.safetensors``)
    are saved flat in *ckpt_dir* (same layout as ``save_final``), so
    users can point inference tools directly at any checkpoint.
    ``training_state.pt`` is saved alongside for resume support.
    """
    save_adapter_flat(trainer, ckpt_dir)

    # Save optimizer / scheduler / progress for resume
    training_state = {
        "epoch": epoch,
        "global_step": global_step,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    if extra_state:
        training_state.update(extra_state)
    state_path = os.path.join(ckpt_dir, "training_state.pt")
    torch.save(training_state, state_path)

    # Also write a safetensors file with epoch/global_step so that
    # load_training_checkpoint (which reads .safetensors) can restore
    # training progress metadata.
    controlled = getattr(trainer.training_config, "phase_d_resume_adapter", None) is not None
    try:
        from safetensors.torch import save_file as _save_safetensors

        meta_tensors = {
            "epoch": torch.tensor([epoch], dtype=torch.int64),
            "global_step": torch.tensor([global_step], dtype=torch.int64),
        }
        sf_path = os.path.join(ckpt_dir, "training_state.safetensors")
        _save_safetensors(meta_tensors, sf_path)
    except Exception as exc:
        if controlled:
            raise RuntimeError(
                f"controlled checkpoint could not write training_state.safetensors in {ckpt_dir}"
            ) from exc
        logger.debug("Could not write training_state.safetensors: %s", exc)
    if controlled:
        required = (
            Path(ckpt_dir) / "training_state.pt",
            Path(ckpt_dir) / "training_state.safetensors",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"controlled checkpoint state save is incomplete: {missing}")

    logger.info(
        "Training checkpoint saved to %s (epoch %d, step %d)",
        ckpt_dir,
        epoch,
        global_step,
    )


def save_final(trainer: Any, output_dir: str) -> None:
    """Save final adapter weights (inference-ready, no training state)."""
    save_adapter_flat(trainer, output_dir)
    controlled = getattr(trainer.training_config, "phase_d_resume_adapter", None) is not None
    verify_saved_adapter(output_dir, strict=controlled)


def verify_saved_adapter(output_dir: str, *, strict: bool = False) -> None:
    """Check saved adapter weights exist and are non-trivial.

    Loads the safetensors file, counts non-zero parameters, and logs
    a warning if the weights appear to be all zeros (which would mean
    the LoRA has no effect during inference).
    """
    safetensors_path = os.path.join(output_dir, "adapter_model.safetensors")
    config_path = os.path.join(output_dir, "adapter_config.json")

    # LoKR uses a different file name
    if not os.path.exists(safetensors_path):
        lokr_path = os.path.join(output_dir, "lokr_weights.safetensors")
        if os.path.exists(lokr_path):
            logger.info("[OK] LoKR weights saved: %s", lokr_path)
            return
        message = f"No adapter weights found in {output_dir}"
        if strict:
            raise RuntimeError(message)
        logger.warning("[WARN] %s -- check save path", message)
        return

    try:
        from safetensors.torch import load_file

        weights = load_file(safetensors_path)
        total_params = 0
        nonzero_params = 0
        max_abs = 0.0
        with torch.no_grad():
            for tensor in weights.values():
                total_params += tensor.numel()
                nonzero_params += int((tensor != 0).sum().item())
                max_abs = max(max_abs, tensor.abs().max().item())

        if nonzero_params == 0:
            message = (
                "All saved LoRA weights are zero; the adapter would have no effect"
            )
            if strict:
                raise RuntimeError(message)
            logger.warning("[WARN] %s", message)
        else:
            pct = 100.0 * nonzero_params / max(total_params, 1)
            logger.info(
                "[OK] Adapter verified: %s params, %s non-zero (%.1f%%), max|w|=%.6f",
                f"{total_params:,}",
                f"{nonzero_params:,}",
                pct,
                max_abs,
            )

        if not os.path.exists(config_path):
            message = f"adapter_config.json missing in {output_dir}"
            if strict:
                raise RuntimeError(message)
            logger.warning("[WARN] %s -- inference tools cannot load this adapter", message)
    except Exception as exc:
        if strict:
            raise RuntimeError(f"controlled adapter verification failed for {output_dir}") from exc
        logger.warning("[WARN] Could not verify adapter: %s", exc)


def _tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

def _resume_phase_d_ablation(trainer: Any, ckpt_dir: Path, optimizer: Any, scheduler: Any) -> Tuple[int, int]:
    """Restore independently controlled Phase-D state and fail on any mismatch."""
    module = trainer.module
    cfg = trainer.training_config
    state_path = ckpt_dir / "training_state.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"parent training state missing: {state_path}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(f"parent training state must be a dict: {state_path}")
    for required_key in ("epoch", "global_step", "optimizer_state_dict", "scheduler_state_dict"):
        if required_key not in state:
            raise KeyError(f"parent training state missing {required_key!r}: {state_path}")
    optimizer_policy = str(cfg.phase_d_optimizer_policy)
    scheduler_policy = str(cfg.phase_d_scheduler_policy)
    if optimizer_policy == "inherit":
        optimizer.load_state_dict(state["optimizer_state_dict"])
    elif optimizer_policy != "fresh":
        raise ValueError(f"invalid optimizer policy: {optimizer_policy}")
    if scheduler_policy == "inherit":
        scheduler.load_state_dict(state["scheduler_state_dict"])
        inherited_lrs = list(scheduler.state_dict().get("_last_lr", scheduler.get_last_lr()))
        if len(inherited_lrs) != len(optimizer.param_groups):
            raise RuntimeError("inherited scheduler LR/group mismatch")
        for group, lr in zip(optimizer.param_groups, inherited_lrs):
            group["lr"] = float(lr)
    elif scheduler_policy != "fresh":
        raise ValueError(f"invalid scheduler policy: {scheduler_policy}")

    decoder = _unwrap_runtime_decoder(module.model.decoder)
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    runtime_before = get_peft_model_state_dict(decoder, adapter_name="default")
    adapter_policy = str(cfg.phase_d_resume_adapter)
    report = {
        "policy": adapter_policy,
        "source_count": 0,
        "matched_count": 0,
        "missing_count": 0,
        "unexpected_count": 0,
        "max_abs_difference": 0.0,
        "parent_adapter_loaded": False,
        "parent_adapter_verified": False,
        "initial_adapter_sha256": _tensor_state_sha256(runtime_before),
    }
    if adapter_policy == "verified":
        adapter_file = ckpt_dir / "adapter_model.safetensors"
        if not adapter_file.is_file():
            raise FileNotFoundError(f"parent adapter missing: {adapter_file}")
        from safetensors.torch import load_file
        source = load_file(str(adapter_file))
        set_peft_model_state_dict(decoder, source, adapter_name="default")
        runtime = get_peft_model_state_dict(decoder, adapter_name="default")
        missing = sorted(set(source) - set(runtime))
        runtime_only = sorted(set(runtime) - set(source))
        ignored_timing = [
            key for key in runtime_only
            if ".timing_cross_attn." in key and (".lora_A." in key or ".lora_B." in key)
        ]
        unexpected = sorted(set(runtime_only) - set(ignored_timing))
        common = sorted(set(source) & set(runtime))
        differences = [
            float((source[key].cpu() - runtime[key].detach().cpu()).abs().max())
            for key in common
        ]
        max_diff = max(differences, default=0.0)
        timing_adapter_parameters = [
            (name, parameter)
            for name, parameter in decoder.named_parameters()
            if ".timing_cross_attn." in name and ("lora_A" in name or "lora_B" in name)
        ]
        trainable_timing_adapter = [name for name, parameter in timing_adapter_parameters if parameter.requires_grad]
        nonzero_timing_b = [
            name for name, parameter in timing_adapter_parameters
            if "lora_B" in name and int(torch.count_nonzero(parameter.detach())) != 0
        ]
        if missing or unexpected or max_diff != 0.0 or trainable_timing_adapter or nonzero_timing_b:
            raise RuntimeError(
                "PEFT verification failed: "
                f"missing={missing[:5]} unexpected={unexpected[:5]} max_diff={max_diff} "
                f"trainable_timing_adapter={trainable_timing_adapter[:5]} "
                f"nonzero_timing_b={nonzero_timing_b[:5]}"
            )
        matched_runtime = {key: runtime[key] for key in common}
        source_matched = {key: source[key] for key in common}
        report.update({
            "source_count": len(source), "matched_count": len(common),
            "missing_count": len(missing), "unexpected_count": len(unexpected),
            "source_names": sorted(source), "runtime_names": sorted(runtime),
            "matched_names": common, "missing_names": missing, "unexpected_names": unexpected,
            "ignored_runtime_timing_adapter_count": len(ignored_timing),
            "ignored_runtime_timing_adapter_keys": ignored_timing,
            "ignored_runtime_timing_lora_b_all_zero": True,
            "ignored_runtime_timing_adapter_all_frozen": True,
            "max_abs_difference": max_diff,
            "source_sha256": _tensor_state_sha256(source),
            "source_matched_sha256": _tensor_state_sha256(source_matched),
            "matched_runtime_sha256": _tensor_state_sha256(matched_runtime),
            "runtime_full_sha256": _tensor_state_sha256(runtime),
            "parent_adapter_loaded": True,
            "parent_adapter_verified": True,
            "initial_adapter_sha256": _tensor_state_sha256(matched_runtime),
        })
    elif adapter_policy != "fresh":
        raise ValueError(f"invalid adapter policy: {adapter_policy}")

    timing_report = None
    if bool(getattr(cfg, "phase_d_require_parent_timing_state", False)):
        if not bool(getattr(cfg, "strict_timing_state_load", False)):
            raise RuntimeError("required parent timing state needs strict_timing_state_load=True")
        if not _load_timing_module(module, ckpt_dir):
            raise RuntimeError(f"required parent timing state was not loaded: {ckpt_dir}")
        timing_report = _verify_timing_module_exact(module, ckpt_dir)

    if bool(getattr(cfg, "use_mert_conditioning", False)):
        bridge_loaded = _load_bridge_module(module, ckpt_dir)
        if bool(getattr(cfg, "require_complete_mert", False)) and not bridge_loaded:
            raise RuntimeError("MERT conditioning was requested but the parent bridge was not loaded")
    else:
        # Do not even deserialize a bridge when the experiment says MERT is disabled.
        bridge_loaded = False
    trainer._phase_d_resume_report = {
        "adapter": report, "optimizer_loaded": optimizer_policy == "inherit",
        "scheduler_loaded": scheduler_policy == "inherit", "mert_bridge_loaded": bool(bridge_loaded),
        "parent_epoch": int(state.get("epoch", 0)), "parent_global_step": int(state.get("global_step", 0)),
        "timing_state": timing_report,
    }
    return int(state.get("epoch", 0)), int(state.get("global_step", 0))

def resume_checkpoint(
    trainer: Any,
    resume_path: str,
    optimizer: Any,
    scheduler: Any,
) -> Generator[TrainingUpdate, None, Optional[Tuple[int, int]]]:
    """Resume from a checkpoint directory. Returns (start_epoch, global_step) or None."""
    module = trainer.module
    assert module is not None
    ckpt_dir = Path(resume_path)
    restore_optimizer = bool(
        getattr(trainer.training_config, "resume_optimizer_state", True)
    )

    # Normalize: if user pointed to a file inside the checkpoint dir,
    # use the containing directory instead.
    if ckpt_dir.is_file():
        logger.info(
            "resume_from points to a file (%s) -- using parent directory %s",
            ckpt_dir.name,
            ckpt_dir.parent,
        )
        ckpt_dir = ckpt_dir.parent

    if getattr(trainer.training_config, "phase_d_resume_adapter", None) is not None:
        epoch, step = _resume_phase_d_ablation(trainer, ckpt_dir, optimizer, scheduler)
        report = trainer._phase_d_resume_report
        yield TrainingUpdate(0, 0.0, f"[OK] Controlled Phase-D resume: {report}", kind="info")
        return epoch, step

    # -- Detect format: LoKR uses lokr_weights.safetensors ---------------
    lokr_weights_path = ckpt_dir / "lokr_weights.safetensors"
    state_path = ckpt_dir / "training_state.pt"

    if lokr_weights_path.exists() and module.lycoris_net is not None:
        # LoKR resume
        if trainer.adapter_type != "lokr":
            raise RuntimeError(
                f"checkpoint is LoKR but configured adapter_type={trainer.adapter_type!r}"
            )
        load_lokr_weights(module.lycoris_net, str(lokr_weights_path))
        if state_path.exists():
            state = torch.load(
                str(state_path), map_location=module.device, weights_only=False
            )
            epoch = state.get("epoch", 0)
            step = state.get("global_step", 0)
            if restore_optimizer and "optimizer_state_dict" in state:
                optimizer.load_state_dict(state["optimizer_state_dict"])
            if restore_optimizer and "scheduler_state_dict" in state:
                scheduler.load_state_dict(state["scheduler_state_dict"])
            yield TrainingUpdate(
                0,
                0.0,
                f"[OK] Resumed LoKR from epoch {epoch}, step {step}",
                kind="info",
            )
            _load_bridge_module(module, ckpt_dir)
            _load_identity_v5_module(module, ckpt_dir)
            _load_timing_module(module, ckpt_dir)
            return (epoch, step)
        yield TrainingUpdate(
            0, 0.0, "[OK] LoKR weights loaded (no training state)", kind="info"
        )
        _load_bridge_module(module, ckpt_dir)
        _load_identity_v5_module(module, ckpt_dir)
        _load_timing_module(module, ckpt_dir)
        return None

    if trainer.adapter_type == "lokr":
        if not lokr_weights_path.exists():
            raise FileNotFoundError(
                f"adapter_type='lokr' but checkpoint lacks lokr_weights.safetensors: {resume_path}"
            )
        if module.lycoris_net is None:
            raise RuntimeError("adapter_type='lokr' but lycoris_net is unavailable")

    # LoRA resume (original logic)
    ckpt_info = load_training_checkpoint(
        str(ckpt_dir),
        optimizer=optimizer if restore_optimizer else None,
        scheduler=scheduler if restore_optimizer else None,
        device=module.device,
    )
    if ckpt_info["adapter_path"]:
        adapter_path = ckpt_info["adapter_path"]
        aw_path = os.path.join(adapter_path, "adapter_model.safetensors")
        if not os.path.exists(aw_path):
            aw_path = os.path.join(adapter_path, "adapter_model.bin")

        if os.path.exists(aw_path):
            from safetensors.torch import load_file

            state_dict = (
                load_file(aw_path)
                if aw_path.endswith(".safetensors")
                else torch.load(aw_path, map_location=module.device, weights_only=True)
            )
            decoder = module.model.decoder
            if hasattr(decoder, "_forward_module"):
                decoder = decoder._forward_module
            # PEFT save_pretrained() strips the runtime adapter slot from keys.
            # A permissive raw state-dict load can silently load zero LoRA tensors.
            from peft import get_peft_model_state_dict, set_peft_model_state_dict

            set_peft_model_state_dict(
                decoder, state_dict, adapter_name="default"
            )
            loaded_state = get_peft_model_state_dict(
                decoder, adapter_name="default"
            )
            missing_source = sorted(set(state_dict) - set(loaded_state))
            mismatched_source = sorted(
                key
                for key in set(state_dict).intersection(loaded_state)
                if not torch.equal(
                    state_dict[key].detach().cpu(),
                    loaded_state[key].detach().cpu(),
                )
            )
            if missing_source or mismatched_source:
                raise RuntimeError(
                    "PEFT adapter resume verification failed: "
                    f"missing_source={missing_source[:10]}, "
                    f"mismatched_source={mismatched_source[:10]}"
                )
            bridge_loaded = _load_bridge_module(module, ckpt_dir)
            v5_loaded = _load_identity_v5_module(module, ckpt_dir)
            timing_loaded = _load_timing_module(module, ckpt_dir)

            start_epoch = ckpt_info["epoch"]
            g_step = ckpt_info["global_step"]
            parts = [f"[OK] Resumed from epoch {start_epoch}, step {g_step}"]
            if not restore_optimizer:
                parts.append("fresh optimizer/scheduler")
            if ckpt_info["loaded_optimizer"]:
                parts.append("optimizer OK")
            if ckpt_info["loaded_scheduler"]:
                parts.append("scheduler OK")
            if bridge_loaded:
                parts.append("MERT bridge OK")
            if timing_loaded:
                parts.append("timing branch OK")
            yield TrainingUpdate(0, 0.0, ", ".join(parts), kind="info")
            return (start_epoch, g_step)
        yield TrainingUpdate(
            0, 0.0, f"[WARN] Adapter weights not found in {adapter_path}", kind="warn"
        )
        return None
    yield TrainingUpdate(
        0, 0.0, f"[WARN] No valid checkpoint in {ckpt_dir}", kind="warn"
    )
    return None
