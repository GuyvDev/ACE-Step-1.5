"""
Trainer helper functions for FixedLoRATrainer.

Contains checkpoint save/resume, adapter verification, memory
configuration, and module wrapper introspection -- extracted from
``trainer_fixed.py`` to keep it under the LOC limit.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Generator, Optional, Tuple

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
    timing_state = {
        name: tensor.detach().cpu()
        for name, tensor in raw_decoder.state_dict().items()
        if "timing_" in name
    }
    return timing_state


def _load_decoder_timing_state(module: Any, decoder_state: dict[str, torch.Tensor], source: Path) -> bool:
    """Restore decoder-side timing-attention weights from checkpoint payload."""
    model = getattr(module, "model", None)
    decoder = getattr(model, "decoder", None) if model is not None else None
    if decoder is None or not decoder_state:
        return False
    raw_decoder = _unwrap_runtime_decoder(decoder)
    try:
        missing, unexpected = raw_decoder.load_state_dict(decoder_state, strict=False)
        logger.info(
            "[OK] Loaded decoder timing weights from %s (%d tensors, missing=%d, unexpected=%d)",
            source,
            len(decoder_state),
            len(missing),
            len(unexpected),
        )
        return True
    except Exception as exc:
        logger.warning("[WARN] Failed to load decoder timing weights from %s: %s", source, exc)
        return False


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


def _save_timing_module(module: Any, output_dir: str) -> None:
    """Save timing encoder weights alongside LoRA adapter weights."""
    timing_encoder = getattr(module, "timing_encoder", None)
    decoder_timing_supervisor = getattr(module, "decoder_timing_supervisor", None)
    decoder_timing_state = _collect_decoder_timing_state(module)
    if timing_encoder is None and decoder_timing_supervisor is None and not decoder_timing_state:
        return
    timing_path = os.path.join(output_dir, "timing_branch.pt")
    payload = {}
    if timing_encoder is not None:
        payload["state_dict"] = timing_encoder.state_dict()
        payload["config"] = timing_encoder.export_config()
    if decoder_timing_supervisor is not None:
        payload["decoder_state_dict"] = decoder_timing_supervisor.state_dict()
    if decoder_timing_state:
        payload["decoder_model_state_dict"] = decoder_timing_state
    torch.save(payload, timing_path)
    logger.info("[OK] Saved timing branch weights to %s", timing_path)


def _load_timing_module(module: Any, checkpoint_dir: Path) -> bool:
    """Load timing encoder weights from a checkpoint directory."""
    timing_encoder = getattr(module, "timing_encoder", None)
    decoder_timing_supervisor = getattr(module, "decoder_timing_supervisor", None)
    timing_path = checkpoint_dir / "timing_branch.pt"
    if (timing_encoder is None and decoder_timing_supervisor is None) or not timing_path.exists():
        return False
    try:
        payload = load_saved_timing_module(str(timing_path), map_location=module.device)
        if timing_encoder is not None and "state_dict" in payload:
            try:
                timing_encoder.load_state_dict(payload["state_dict"], strict=True)
                logger.info("[OK] Loaded timing branch weights from %s", timing_path)
            except Exception as strict_err:
                logger.warning(
                    "[WARN] Strict timing-branch load failed for %s: %s; retrying non-strict",
                    timing_path,
                    strict_err,
                )
                timing_encoder.load_state_dict(payload["state_dict"], strict=False)
                logger.info("[OK] Loaded timing branch weights non-strictly from %s", timing_path)
        if decoder_timing_supervisor is not None and "decoder_state_dict" in payload:
            try:
                decoder_timing_supervisor.load_state_dict(payload["decoder_state_dict"], strict=True)
                logger.info("[OK] Loaded decoder timing supervisor from %s", timing_path)
            except Exception as strict_err:
                logger.warning(
                    "[WARN] Strict decoder timing-supervisor load failed for %s: %s; retrying non-strict",
                    timing_path,
                    strict_err,
                )
                decoder_timing_supervisor.load_state_dict(payload["decoder_state_dict"], strict=False)
                logger.info("[OK] Loaded decoder timing supervisor non-strictly from %s", timing_path)
        if "decoder_model_state_dict" in payload:
            _load_decoder_timing_state(module, payload["decoder_model_state_dict"], timing_path)
        return True
    except Exception as e:
        logger.warning("[WARN] Failed to load timing branch from %s: %s", timing_path, e)
        return False


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


def configure_memory_features(decoder: nn.Module) -> tuple:
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
            except Exception:
                pass
        elif hasattr(mod, "gradient_checkpointing"):
            try:
                mod.gradient_checkpointing = True
                ckpt_enabled = True
            except Exception:
                pass

        # 2. PEFT + checkpointing needs input embeddings to carry grads
        if hasattr(mod, "enable_input_require_grads"):
            try:
                mod.enable_input_require_grads()
                hook_ok = bool(getattr(mod, "_acestep_input_grads_hook_enabled", False))
                has_hook = getattr(mod, "_require_grads_hook", None) is not None
                if hook_ok or has_hook:
                    input_grads_enabled = True
            except Exception:
                pass

        # 3. Disable use_cache (frees KV-cache memory)
        cfg = getattr(mod, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            try:
                if getattr(cfg, "use_cache", None) is not False:
                    cfg.use_cache = False
                    cache_disabled = True
            except Exception:
                pass

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
        if hasattr(raw_decoder, "save_pretrained"):
            raw_decoder.save_pretrained(output_dir)
            logger.info("[OK] LoRA adapter saved to %s", output_dir)
        else:
            # Fallback for non-PEFT models
            save_lora_weights(module.model, output_dir)

    _save_bridge_module(module, output_dir)
    _save_timing_module(module, output_dir)


def save_checkpoint(
    trainer: Any,
    optimizer: Any,
    scheduler: Any,
    epoch: int,
    global_step: int,
    ckpt_dir: str,
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
    state_path = os.path.join(ckpt_dir, "training_state.pt")
    torch.save(training_state, state_path)

    # Also write a safetensors file with epoch/global_step so that
    # load_training_checkpoint (which reads .safetensors) can restore
    # training progress metadata.
    try:
        from safetensors.torch import save_file as _save_safetensors

        meta_tensors = {
            "epoch": torch.tensor([epoch], dtype=torch.int64),
            "global_step": torch.tensor([global_step], dtype=torch.int64),
        }
        sf_path = os.path.join(ckpt_dir, "training_state.safetensors")
        _save_safetensors(meta_tensors, sf_path)
    except Exception as exc:
        logger.debug("Could not write training_state.safetensors: %s", exc)

    logger.info(
        "Training checkpoint saved to %s (epoch %d, step %d)",
        ckpt_dir,
        epoch,
        global_step,
    )


def save_final(trainer: Any, output_dir: str) -> None:
    """Save final adapter weights (inference-ready, no training state)."""
    save_adapter_flat(trainer, output_dir)
    verify_saved_adapter(output_dir)


def verify_saved_adapter(output_dir: str) -> None:
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
        logger.warning(
            "[WARN] No adapter weights found in %s -- check save path",
            output_dir,
        )
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
            logger.warning(
                "[WARN] All saved LoRA weights are ZERO -- "
                "the adapter will have no effect during inference. "
                "Training may not have converged."
            )
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
            logger.warning(
                "[WARN] adapter_config.json missing in %s -- "
                "inference tools will not be able to load this adapter",
                output_dir,
            )
    except Exception as exc:
        logger.warning("[WARN] Could not verify adapter: %s", exc)


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

    # Normalize: if user pointed to a file inside the checkpoint dir,
    # use the containing directory instead.
    if ckpt_dir.is_file():
        logger.info(
            "resume_from points to a file (%s) -- using parent directory %s",
            ckpt_dir.name,
            ckpt_dir.parent,
        )
        ckpt_dir = ckpt_dir.parent

    # -- Detect format: LoKR uses lokr_weights.safetensors ---------------
    lokr_weights_path = ckpt_dir / "lokr_weights.safetensors"
    state_path = ckpt_dir / "training_state.pt"

    if lokr_weights_path.exists() and module.lycoris_net is not None:
        # LoKR resume
        if trainer.adapter_type != "lokr":
            logger.warning(
                "[WARN] Found lokr_weights.safetensors but adapter_type is '%s' "
                "-- loading as LoKR anyway",
                trainer.adapter_type,
            )
        load_lokr_weights(module.lycoris_net, str(lokr_weights_path))
        if state_path.exists():
            state = torch.load(
                str(state_path), map_location=module.device, weights_only=False
            )
            epoch = state.get("epoch", 0)
            step = state.get("global_step", 0)
            if "optimizer_state_dict" in state:
                optimizer.load_state_dict(state["optimizer_state_dict"])
            if "scheduler_state_dict" in state:
                scheduler.load_state_dict(state["scheduler_state_dict"])
            yield TrainingUpdate(
                0,
                0.0,
                f"[OK] Resumed LoKR from epoch {epoch}, step {step}",
                kind="info",
            )
            _load_bridge_module(module, ckpt_dir)
            _load_timing_module(module, ckpt_dir)
            return (epoch, step)
        yield TrainingUpdate(
            0, 0.0, "[OK] LoKR weights loaded (no training state)", kind="info"
        )
        _load_bridge_module(module, ckpt_dir)
        _load_timing_module(module, ckpt_dir)
        return None

    # Warn if LoKR was expected but checkpoint is LoRA-format
    if trainer.adapter_type == "lokr":
        if not lokr_weights_path.exists():
            logger.warning(
                "[WARN] adapter_type is 'lokr' but no lokr_weights.safetensors "
                "found in %s -- falling back to LoRA resume format",
                resume_path,
            )
        elif module.lycoris_net is None:
            logger.warning(
                "[WARN] adapter_type is 'lokr' and lokr_weights.safetensors exists "
                "but lycoris_net is None -- cannot load LoKR checkpoint",
            )

    # LoRA resume (original logic)
    ckpt_info = load_training_checkpoint(
        str(ckpt_dir),
        optimizer=optimizer,
        scheduler=scheduler,
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
            decoder.load_state_dict(state_dict, strict=False)
            bridge_loaded = _load_bridge_module(module, ckpt_dir)
            timing_loaded = _load_timing_module(module, ckpt_dir)

            start_epoch = ckpt_info["epoch"]
            g_step = ckpt_info["global_step"]
            parts = [f"[OK] Resumed from epoch {start_epoch}, step {g_step}"]
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
