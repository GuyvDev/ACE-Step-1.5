"""
FixedLoRATrainer -- Orchestration for ACE-Step V2 adapter fine-tuning.

The actual per-step training logic lives in ``fixed_lora_module.py``
(``FixedLoRAModule``).  The non-Fabric fallback loop lives in
``trainer_basic_loop.py``.  Checkpoint, memory, and verification helpers
live in ``trainer_helpers.py``.

Supports both adapter types:
    - **LoRA** via PEFT (``inject_lora_into_dit``)
    - **LoKR** via LyCORIS (``inject_lokr_into_dit``)

Uses shared utilities from ``acestep.training``.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import logging
import math
import os
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple

import torch
import torch.nn as nn
from acestep.training.data_module import PreprocessedDataModule as StandardPreprocessedDataModule
from acestep.training_v2.optim import (
    GroupRatioCosineAnnealingLR,
    build_optimizer,
    build_scheduler,
)
from acestep.training_v2.phase_d_strict_data import (
    PreprocessedDataModule as StrictPhaseDDataModule,
    strict_phase_d_collate,
)

# V2 modules
from acestep.training_v2.configs import TrainingConfigV2
from acestep.training_v2.tensorboard_utils import TrainingLogger
from acestep.training_v2.ui import TrainingUpdate

# Split-out modules
from acestep.training_v2.fixed_lora_module import (
    AdapterConfig,
    FixedLoRAModule,
    _normalize_device_type,
    _select_compute_dtype,
    _select_fabric_precision,
)
from acestep.training_v2.model_loader import _resolve_dtype
from acestep.training_v2.trainer_helpers import (
    configure_memory_features,
    offload_non_decoder,
    resume_checkpoint,
    save_adapter_flat,
    save_checkpoint,
    save_final,
    verify_saved_adapter,
)
from acestep.training_v2.trainer_basic_loop import run_basic_training_loop

logger = logging.getLogger(__name__)

# Try to import Lightning Fabric
try:
    from lightning.fabric import Fabric

    _FABRIC_AVAILABLE = True
except ImportError:
    _FABRIC_AVAILABLE = False
    logger.warning("[WARN] Lightning Fabric not installed. Training will use basic loop.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _record_controlled_failure(cfg: Any, exc: BaseException) -> None:
    manifest_value = getattr(cfg, "experiment_manifest_out", None)
    if not manifest_value:
        return
    path = Path(str(manifest_value)).resolve()
    existing: Dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except Exception:
            existing = {}
    existing.update({
        "status": "failed",
        "experiment_completed": False,
        "failure_type": type(exc).__name__,
        "failure_message": str(exc),
        "failed_at": time.strftime("%FT%T%z"),
    })
    _atomic_json(path, existing)


def _append_jsonl(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _parameter_grad_norm(items: list[tuple[str, nn.Parameter]]) -> float:
    total = 0.0
    for _, parameter in items:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().item())
    return math.sqrt(total)


def _base_adapter_tensor_sha256(module: nn.Module) -> str:
    """Hash the inherited non-timing PEFT tensors by name, shape, dtype, and value."""
    decoder = module.model.decoder
    while hasattr(decoder, "_forward_module"):
        decoder = decoder._forward_module
    from peft import get_peft_model_state_dict

    state = get_peft_model_state_dict(decoder, adapter_name="default")
    state = {key: value for key, value in state.items() if "timing_cross_attn" not in key}
    if not state:
        raise RuntimeError("base adapter hash matched zero PEFT tensors")
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()



def _resolve_requested_compute_dtype(precision: str, device_type: str) -> torch.dtype:
    if precision == "auto":
        return _select_compute_dtype(device_type)
    return _resolve_dtype(precision)


def _resolve_requested_fabric_precision(precision: str, device_type: str) -> str:
    if precision == "auto":
        return _select_fabric_precision(device_type)

    mapping = {
        "bf16": "bf16-mixed",
        "fp16": "16-mixed",
        "fp32": "32-true",
    }
    return mapping.get(precision, _select_fabric_precision(device_type))


def _timing_named_parameters(module: nn.Module) -> dict[str, list[tuple[str, nn.Parameter]]]:
    groups: dict[str, list[tuple[str, nn.Parameter]]] = {
        "timing_encoder": [],
        "timing_gate": [],
        "timing_consumer": [],
    }
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith("timing_attn_gate"):
            groups["timing_gate"].append((name, parameter))
        elif (
            name.startswith("timing_encoder.")
            or name.startswith("decoder_timing_supervisor.")
            or name.startswith("performance_timing_predictor.")
            or name.startswith("decoder_expressivity_supervisor.")
        ):
            groups["timing_encoder"].append((name, parameter))
        elif "timing_cross_attn" in name or "timing_global_" in name:
            groups["timing_consumer"].append((name, parameter))
    return groups


def _gradient_health(parameters: list[tuple[str, nn.Parameter]]) -> tuple[float, bool, int]:
    squared = 0.0
    finite = True
    present = 0
    for _, parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        present += 1
        finite = finite and bool(torch.isfinite(grad).all())
        squared += float(grad.square().sum().item())
    return math.sqrt(squared), finite, present


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _sanitize_nonfinite_gradients(
    model: nn.Module,
    *,
    max_names: int = 12,
) -> tuple[int, list[str]]:
    """Replace non-finite gradients with zeros and return a short report."""
    fixed = 0
    names: list[str] = []
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None or torch.isfinite(grad).all():
            continue
        param.grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        fixed += 1
        if len(names) < max_names:
            names.append(name)
    return fixed, names


def _sanitize_nonfinite_parameters(
    model: nn.Module,
    *,
    max_names: int = 12,
) -> tuple[int, list[str]]:
    """Replace non-finite values only when explicitly requested by a non-controlled run."""
    fixed = 0
    names: list[str] = []
    for name, param in model.named_parameters():
        if torch.isfinite(param).all():
            continue
        with torch.no_grad():
            param.copy_(torch.nan_to_num(param, nan=0.0, posinf=0.0, neginf=0.0))
        fixed += 1
        if len(names) < max_names:
            names.append(name)
    return fixed, names


def _ordered_dataset_paths(dataset: Any) -> list[str]:
    """Resolve the exact train-subset membership/order without discarding Subset indices."""
    indices = getattr(dataset, "indices", None)
    child = getattr(dataset, "dataset", None)
    if indices is not None and child is not None:
        base = _ordered_dataset_paths(child)
        resolved: list[str] = []
        for raw_index in indices:
            index = int(raw_index)
            if index < 0 or index >= len(base):
                raise RuntimeError(f"dataset subset index out of range: {index}/{len(base)}")
            resolved.append(base[index])
        return resolved
    valid_paths = getattr(dataset, "valid_paths", None)
    if valid_paths is not None:
        return [str(Path(value).resolve()) for value in valid_paths]
    if child is not None:
        return _ordered_dataset_paths(child)
    raise RuntimeError(f"cannot resolve ordered dataset paths from {type(dataset).__name__}")


def _summarize_batch_metadata(batch: Dict[str, Any], *, max_items: int = 3) -> str:
    """Build a short, human-readable identifier for the current batch."""
    metadata = batch.get("metadata")
    if not isinstance(metadata, list):
        return "<unknown>"

    labels: list[str] = []
    for item in metadata[:max_items]:
        if not isinstance(item, dict):
            continue
        label = (
            item.get("item_name")
            or item.get("audio_path")
            or item.get("source_path")
            or item.get("file_name")
        )
        if label:
            labels.append(str(label))

    if not labels:
        return "<unknown>"
    if len(metadata) > max_items:
        labels.append("...")
    return ", ".join(labels)


# ===========================================================================
# FixedLoRATrainer -- orchestration
# ===========================================================================

class FixedLoRATrainer:
    """High-level trainer for corrected ACE-Step adapter fine-tuning.

    Supports both LoRA (PEFT) and LoKR (LyCORIS) adapters.
    Uses Lightning Fabric for mixed precision and gradient scaling.
    Falls back to a basic PyTorch loop when Fabric is not installed.
    """

    def __init__(
        self,
        model: nn.Module,
        adapter_config: AdapterConfig,
        training_config: TrainingConfigV2,
    ) -> None:
        self.model = model
        self.adapter_config = adapter_config
        self.training_config = training_config
        self.adapter_type = training_config.adapter_type

        # Backward-compat alias
        self.lora_config = adapter_config

        self.module: Optional[FixedLoRAModule] = None
        self.fabric: Optional[Any] = None
        self.is_training = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        training_state: Optional[Dict[str, Any]] = None,
    ) -> Generator[Tuple[int, float, str], None, None]:
        """Run the full training loop.

        Yields ``(global_step, loss, status_message)`` tuples.
        """
        self.is_training = True
        cfg = self.training_config

        try:
            # -- Validate ---------------------------------------------------
            ds_dir = Path(cfg.dataset_dir)
            if not ds_dir.is_dir():
                message = f"Dataset directory not found: {ds_dir}"
                if getattr(cfg, "phase_d_resume_adapter", None) is not None:
                    raise FileNotFoundError(message)
                yield TrainingUpdate(0, 0.0, f"[FAIL] {message}", kind="fail")
                return

            # -- Seed -------------------------------------------------------
            torch.manual_seed(cfg.seed)
            random.seed(cfg.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg.seed)

            # Controlled Phase-D never repairs or skips numerical failures.
            if getattr(cfg, "phase_d_resume_adapter", None) is not None:
                if bool(getattr(cfg, "skip_nonfinite_gradients", False)):
                    raise RuntimeError("controlled Phase-D forbids skip_nonfinite_gradients")
                if not bool(getattr(cfg, "strict_sidecars", False)):
                    raise RuntimeError("controlled Phase-D requires strict_sidecars=True")
                if not bool(getattr(cfg, "strict_timing_state_load", False)):
                    raise RuntimeError("controlled Phase-D requires strict_timing_state_load=True")
                if bool(getattr(cfg, "use_mert_conditioning", False)):
                    raise RuntimeError("first actual-new-chain Phase-D test requires MERT disabled")
                if str(getattr(cfg, "timing_condition_source", "")) != "sidecar":
                    raise RuntimeError("controlled Phase-D requires timing_condition_source='sidecar'")
                if str(getattr(cfg, "device", "")) != "cuda":
                    raise RuntimeError("controlled Phase-D requires device='cuda'")
                if str(getattr(cfg, "precision", "")) != "bf16":
                    raise RuntimeError("controlled Phase-D requires precision='bf16'")
                if str(getattr(cfg, "optimizer_type", "")) != "adamw":
                    raise RuntimeError("controlled Phase-D requires optimizer_type='adamw'")
                if str(getattr(cfg, "scheduler_type", "")) != "cosine":
                    raise RuntimeError("controlled Phase-D requires scheduler_type='cosine'")
                if int(getattr(cfg, "warmup_steps", -1)) != 0:
                    raise RuntimeError("controlled Phase-D requires warmup_steps=0")
                if bool(getattr(cfg, "offload_encoder", False)):
                    raise RuntimeError("controlled Phase-D forbids training encoder offload")
                clarity_replay = bool(getattr(cfg, "phase_d_clarity_replay_enabled", False))
                d1_roles = bool(getattr(cfg, "phase_d_d1_alternating_roles_enabled", False))
                if clarity_replay and d1_roles:
                    raise RuntimeError("legacy clarity replay and D1 alternating roles are mutually exclusive")
                if clarity_replay and bool(getattr(cfg, "freeze_base_adapter_in_phase_d", False)):
                    raise RuntimeError("clarity replay and base-adapter freezing are mutually exclusive")
                if clarity_replay:
                    if abs(float(getattr(cfg, "f0_loss_weight", 0.0)) - 0.03) > 1e-12:
                        raise RuntimeError("clarity replay requires V4 C0 f0_loss_weight=0.03")
                    if abs(float(getattr(cfg, "speaker_loss_weight", 0.0)) - 0.01) > 1e-12:
                        raise RuntimeError("clarity replay requires V4 C0 speaker_loss_weight=0.01")
                    if abs(float(getattr(cfg, "learning_rate", 0.0)) - 6e-6) > 1e-12:
                        raise RuntimeError("clarity replay requires base LoRA learning_rate=6e-6")
                    if abs(float(getattr(cfg, "timing_encoder_learning_rate", 0.0)) - 5e-5) > 1e-12:
                        raise RuntimeError("clarity replay requires timing_encoder_learning_rate=5e-5")
                    if abs(float(getattr(cfg, "timing_gate_learning_rate", 0.0)) - 5e-5) > 1e-12:
                        raise RuntimeError("clarity replay requires timing_gate_learning_rate=5e-5")
                if d1_roles:
                    expected = {
                        "learning_rate": 1e-6,
                        "timing_encoder_learning_rate": 2e-5,
                        "timing_gate_learning_rate": 2e-5,
                        "parent_preservation_loss_weight": 1.0,
                        "f0_loss_weight": 0.03,
                        "speaker_loss_weight": 0.01,
                        "timing_decoder_loss_weight": 0.1,
                    }
                    actual = {name: float(getattr(cfg, name, -1.0)) for name in expected}
                    if any(abs(actual[name] - value) > 1e-12 for name, value in expected.items()):
                        raise RuntimeError(f"D1 loss/LR contract mismatch: expected={expected} actual={actual}")
                    if bool(getattr(cfg, "freeze_base_adapter_in_phase_d", False)):
                        raise RuntimeError("D1 requires the inherited base LoRA to be trainable")
                    if not bool(getattr(cfg, "phase_d_require_parent_timing_state", False)):
                        raise RuntimeError("D1 requires exact parent timing-state resume")
                    if int(cfg.batch_size) != 1 or int(cfg.gradient_accumulation_steps) != 1:
                        raise RuntimeError("D1 deterministic role schedule requires batch_size=1 and accumulation=1")
                    role_manifest = getattr(cfg, "phase_d_role_manifest_path", None)
                    if not role_manifest or not Path(role_manifest).is_file():
                        raise FileNotFoundError(f"D1 role manifest is missing: {role_manifest}")

            # -- Build module -----------------------------------------------
            device = torch.device(cfg.device)
            dtype = _resolve_requested_compute_dtype(
                getattr(cfg, "precision", "auto"),
                _normalize_device_type(device),
            )

            self.module = FixedLoRAModule(
                model=self.model,
                adapter_config=self.adapter_config,
                training_config=cfg,
                device=device,
                dtype=dtype,
            )

            # -- Data -------------------------------------------------------
            # Windows uses spawn for multiprocessing; default to 0 workers there
            num_workers = cfg.num_workers
            if sys.platform == "win32" and num_workers > 0:
                logger.info("[Side-Step] Windows detected -- setting num_workers=0 (spawn incompatible)")
                num_workers = 0

            # Controlled Phase D uses the fail-closed loader. Ordinary adapter
            # training must retain ACE-Steps standard tensor loader.
            controlled_phase_d = getattr(cfg, "phase_d_resume_adapter", None) is not None
            _timing_dir = getattr(cfg, "timing_dir", None) if getattr(cfg, "enable_timing_branch", False) else None
            data_kwargs = {
                "tensor_dir": cfg.dataset_dir,
                "batch_size": cfg.batch_size,
                "num_workers": num_workers,
                "pin_memory": cfg.pin_memory,
                "prefetch_factor": cfg.prefetch_factor if num_workers > 0 else None,
                "persistent_workers": cfg.persistent_workers if num_workers > 0 else False,
                "pin_memory_device": cfg.pin_memory_device,
                "val_split": getattr(cfg, "val_split", 0.0),
                "timing_dir": _timing_dir,
                "identity_sidecar_dir": getattr(cfg, "identity_sidecar_dir", None),
                "strict_timing_sidecars": getattr(cfg, "strict_sidecars", False),
            }
            if controlled_phase_d:
                data_module = StrictPhaseDDataModule(
                    **data_kwargs,
                    seed=int(cfg.seed),
                    d1_alternating_roles=bool(
                        getattr(cfg, "phase_d_d1_alternating_roles_enabled", False)
                    ),
                    role_manifest_path=getattr(cfg, "phase_d_role_manifest_path", None),
                )
            else:
                data_module = StandardPreprocessedDataModule(**data_kwargs)
            data_module.setup("fit")

            if len(data_module.train_dataset) == 0:
                if getattr(cfg, "phase_d_resume_adapter", None) is not None:
                    raise RuntimeError("No valid samples found in dataset directory")
                yield TrainingUpdate(0, 0.0, "[FAIL] No valid samples found in dataset directory", kind="fail")
                return

            yield TrainingUpdate(0, 0.0, f"[OK] Loaded {len(data_module.train_dataset)} preprocessed samples", kind="info")

            # -- Dispatch to Fabric or basic loop ---------------------------
            if getattr(cfg, "phase_d_resume_adapter", None) is not None and not _FABRIC_AVAILABLE:
                raise RuntimeError("Controlled Phase-D execution requires Lightning Fabric; basic loop is not contract-aware")
            if _FABRIC_AVAILABLE:
                yield from self._train_fabric(data_module, training_state)
            else:
                yield from run_basic_training_loop(self, data_module, training_state)

        except Exception as exc:
            logger.exception("Training failed")
            if getattr(cfg, "phase_d_resume_adapter", None) is not None:
                _record_controlled_failure(cfg, exc)
                raise
            yield TrainingUpdate(0, 0.0, f"[FAIL] Training failed: {exc}", kind="fail")
        finally:
            self.is_training = False

    def stop(self) -> None:
        self.is_training = False

    # ------------------------------------------------------------------
    # Delegate helpers (thin wrappers around trainer_helpers functions)
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_module_wrappers(module: nn.Module) -> list:
        from acestep.training_v2.trainer_helpers import iter_module_wrappers
        return iter_module_wrappers(module)

    @classmethod
    def _configure_memory_features(cls, decoder: nn.Module) -> tuple:
        return configure_memory_features(decoder)

    @staticmethod
    def _offload_non_decoder(model: nn.Module) -> int:
        return offload_non_decoder(model)

    def _save_adapter_flat(self, output_dir: str) -> None:
        save_adapter_flat(self, output_dir)

    def _save_checkpoint(
        self,
        optimizer: Any,
        scheduler: Any,
        epoch: int,
        global_step: int,
        ckpt_dir: str,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        save_checkpoint(
            self,
            optimizer,
            scheduler,
            epoch,
            global_step,
            ckpt_dir,
            extra_state=extra_state,
        )

    def _save_final(self, output_dir: str) -> None:
        save_final(self, output_dir)

    @staticmethod
    def _verify_saved_adapter(output_dir: str) -> None:
        verify_saved_adapter(output_dir)

    def _resume_checkpoint(
        self, resume_path: str, optimizer: Any, scheduler: Any,
    ) -> Generator[TrainingUpdate, None, Optional[Tuple[int, int]]]:
        return (yield from resume_checkpoint(self, resume_path, optimizer, scheduler))

    # ------------------------------------------------------------------
    # Fabric training loop
    # ------------------------------------------------------------------

    def _train_fabric(
        self,
        data_module: StandardPreprocessedDataModule | StrictPhaseDDataModule,
        training_state: Optional[Dict[str, Any]],
    ) -> Generator[TrainingUpdate, None, None]:
        cfg = self.training_config
        assert self.module is not None

        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        device_type = self.module.device_type
        precision = _resolve_requested_fabric_precision(
            getattr(cfg, "precision", "auto"),
            device_type,
        )
        accelerator = device_type if device_type in ("cuda", "xpu", "mps", "cpu") else "auto"

        # -- Fabric init ----------------------------------------------------
        # Always use devices=1 (integer).  Passing devices=[index] (a list)
        # causes Fabric on Windows to create a DistributedSampler wrapper
        # that yields 0 batches, silently breaking the training loop.
        # Instead, we set the default CUDA device so Fabric's single-device
        # mode picks up the correct GPU.
        if device_type == "cuda":
            device_idx = self.module.device.index or 0
            torch.cuda.set_device(device_idx)

        self.fabric = Fabric(
            accelerator=accelerator,
            devices=1,
            precision=precision,
        )
        self.fabric.launch()

        yield TrainingUpdate(0, 0.0, f"[INFO] Starting training (device: {device_type}, precision: {precision})", kind="info")

        # -- TensorBoard logger ---------------------------------------------
        tb = TrainingLogger(cfg.effective_log_dir)

        # -- Dataloader -----------------------------------------------------
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

        # -- Trainable params / optimizer -----------------------------------
        # Cast the model before optimizer construction, then restore the tiny
        # timing gates to FP32 so BF16 quantization cannot erase their updates.
        self.module.model = self.module.model.to(self.module.dtype)
        promoted_timing_gates = 0
        if getattr(self.module, "_use_timing_branch", False):
            promoted_timing_gates = self.module.promote_decoder_timing_gates_fp32()

        trainable_params = [p for p in self.module.parameters() if p.requires_grad]
        if not trainable_params:
            tb.close()
            if getattr(cfg, "phase_d_resume_adapter", None) is not None:
                raise RuntimeError("No trainable parameters found in controlled Phase-D run")
            yield TrainingUpdate(0, 0.0, "[FAIL] No trainable parameters found", kind="fail")
            return

        yield TrainingUpdate(0, 0.0, f"[INFO] Training {sum(p.numel() for p in trainable_params):,} parameters", kind="info")
        if getattr(self.module, "_use_timing_branch", False):
            timing_groups = _timing_named_parameters(self.module)
            gate_dtypes = {str(parameter.dtype) for _, parameter in timing_groups["timing_gate"]}
            if len(timing_groups["timing_gate"]) != 24 or gate_dtypes != {"torch.float32"}:
                raise RuntimeError(
                    "Timing gate precision contract failed: "
                    f"tensors={len(timing_groups['timing_gate'])}, dtypes={sorted(gate_dtypes)}"
                )
            yield TrainingUpdate(
                0,
                0.0,
                (
                    f"[TIMING] FP32 gate contract healthy: tensors=24, "
                    f"params={promoted_timing_gates:,}"
                ),
                kind="info",
            )

        optimizer_type = getattr(cfg, "optimizer_type", "adamw")
        dedicated_timing_lrs = (
            float(getattr(cfg, "timing_encoder_learning_rate", 0.0)),
            float(getattr(cfg, "timing_gate_learning_rate", 0.0)),
            float(getattr(cfg, "timing_consumer_learning_rate", 0.0)),
        )
        if getattr(cfg, "identity_v5_enabled", False):
            optimizer_params = self.module.v5_optimizer_groups()
        elif (
            getattr(self.module, "_use_timing_branch", False)
            and any(rate > 0.0 for rate in dedicated_timing_lrs)
        ):
            optimizer_params = self.module.timing_optimizer_groups(
                base_lr=cfg.learning_rate,
                timing_encoder_lr=dedicated_timing_lrs[0],
                timing_gate_lr=dedicated_timing_lrs[1],
                timing_consumer_lr=dedicated_timing_lrs[2],
            )
        else:
            # Historical Phase D used one optimizer group. Keep that exact
            # behavior unless dedicated timing rates are explicitly requested.
            optimizer_params = trainable_params
        optimizer = build_optimizer(
            optimizer_params,
            optimizer_type=optimizer_type,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            device_type=self.module.device.type,
        )
        if bool(getattr(cfg, "freeze_base_adapter_in_phase_d", False)):
            base_adapter_ids = {
                id(parameter)
                for name, parameter in self.module.named_parameters()
                if ("lora_" in name or "lokr_" in name or "hada_" in name)
                and "timing_cross_attn" not in name
            }
            optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
            overlap = base_adapter_ids & optimizer_ids
            if not base_adapter_ids or overlap:
                raise RuntimeError(
                    "frozen base-adapter optimizer exclusion failed: "
                    f"base_tensors={len(base_adapter_ids)} overlap={len(overlap)}"
                )
        if getattr(cfg, "phase_d_resume_adapter", None) is not None:
            if type(optimizer) is not torch.optim.AdamW:
                raise RuntimeError(f"controlled optimizer type mismatch: {type(optimizer)!r}")
            clarity_replay = bool(getattr(cfg, "phase_d_clarity_replay_enabled", False))
            d1_roles = bool(getattr(cfg, "phase_d_d1_alternating_roles_enabled", False))
            expected_groups = 3 if (clarity_replay or d1_roles) else 1
            if len(optimizer.param_groups) != expected_groups:
                raise RuntimeError(
                    f"controlled optimizer group mismatch: expected={expected_groups} actual={len(optimizer.param_groups)}"
                )
            if clarity_replay:
                actual_groups = {group.get("group_name"): float(group["lr"]) for group in optimizer.param_groups}
                expected_group_lrs = {"base": 6e-6, "timing_encoder": 5e-5, "timing_gate": 5e-5}
                if actual_groups != expected_group_lrs:
                    raise RuntimeError(
                        f"clarity-replay optimizer groups mismatch: expected={expected_group_lrs} actual={actual_groups}"
                    )
            if d1_roles:
                actual_groups = {group.get("group_name"): float(group["lr"]) for group in optimizer.param_groups}
                expected_group_lrs = {"base": 1e-6, "timing_encoder": 2e-5, "timing_gate": 2e-5}
                if actual_groups != expected_group_lrs:
                    raise RuntimeError(
                        f"D1 optimizer groups mismatch: expected={expected_group_lrs} actual={actual_groups}"
                    )
            if optimizer.defaults.get("fused") is not True:
                raise RuntimeError("controlled CUDA AdamW must use fused=True")
            if float(optimizer.param_groups[0]["lr"]) != float(cfg.learning_rate):
                raise RuntimeError("controlled optimizer initial LR differs from requested LR")
        yield TrainingUpdate(0, 0.0, f"[INFO] Optimizer: {optimizer_type}", kind="info")
        if getattr(self.module, "_use_timing_branch", False):
            group_summary = ", ".join(
                f"{group.get('group_name', index)}={float(group['lr']):.3e}"
                f"/{sum(parameter.numel() for parameter in group['params']):,}p"
                for index, group in enumerate(optimizer.param_groups)
            )
            yield TrainingUpdate(
                0,
                0.0,
                f"[TIMING] Optimizer groups: {group_summary}",
                kind="info",
            )

        # -- Scheduler -------------------------------------------------------
        steps_per_epoch = max(1, math.ceil(len(train_loader) / cfg.gradient_accumulation_steps))
        epoch_total_steps = steps_per_epoch * cfg.max_epochs
        controlled_steps = int(getattr(cfg, "max_optimizer_steps", 0))
        total_steps = (
            controlled_steps
            if getattr(cfg, "phase_d_resume_adapter", None) is not None
            else epoch_total_steps
        )
        if total_steps <= 0:
            raise RuntimeError(f"scheduler horizon must be positive: {total_steps}")

        scheduler_type = getattr(cfg, "scheduler_type", "cosine")
        scheduler = build_scheduler(
            optimizer,
            scheduler_type=scheduler_type,
            total_steps=total_steps,
            warmup_steps=cfg.warmup_steps,
            lr=cfg.learning_rate,
            optimizer_type=optimizer_type,
        )
        if getattr(cfg, "phase_d_resume_adapter", None) is not None:
            from torch.optim.lr_scheduler import CosineAnnealingLR
            expected_scheduler = (
                GroupRatioCosineAnnealingLR
                if len(optimizer.param_groups) > 1
                else CosineAnnealingLR
            )
            if type(scheduler) is not expected_scheduler:
                raise RuntimeError(f"controlled scheduler type mismatch: {type(scheduler)!r}")
            if int(getattr(scheduler, "T_max", -1)) != int(total_steps):
                raise RuntimeError("controlled cosine scheduler T_max mismatch")
            if isinstance(scheduler, GroupRatioCosineAnnealingLR):
                if float(scheduler.min_factor) != 0.01:
                    raise RuntimeError("controlled group-ratio cosine floor mismatch")
                expected_base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
                if [float(value) for value in scheduler.base_lrs] != expected_base_lrs:
                    raise RuntimeError("controlled group-ratio cosine base LR mismatch")
        yield TrainingUpdate(0, 0.0, f"[INFO] Scheduler: {scheduler_type}", kind="info")

        # -- Training memory features ----------------------------------------
        if getattr(cfg, "gradient_checkpointing", True):
            ckpt_ok, cache_off, grads_ok = configure_memory_features(
                self.module.model.decoder,
                strict=getattr(cfg, "phase_d_resume_adapter", None) is not None,
            )
            self.module.force_input_grads_for_checkpointing = ckpt_ok
            if getattr(cfg, "phase_d_resume_adapter", None) is not None:
                if not (ckpt_ok and cache_off and grads_ok):
                    raise RuntimeError(
                        "controlled gradient-checkpointing contract failed: "
                        f"enabled={ckpt_ok} cache_disabled={cache_off} input_grads={grads_ok}"
                    )
            if ckpt_ok:
                yield TrainingUpdate(
                    0, 0.0,
                    f"[INFO] Gradient checkpointing enabled "
                    f"(use_cache={not cache_off}, input_grads={grads_ok})",
                    kind="info",
                )
            else:
                yield TrainingUpdate(
                    0, 0.0, "[WARN] Gradient checkpointing not supported by this model",
                    kind="warn",
                )
        else:
            yield TrainingUpdate(
                0, 0.0,
                "[INFO] Gradient checkpointing OFF (faster but uses more VRAM)",
                kind="info",
            )

        # -- Encoder/VAE offloading ------------------------------------------
        if getattr(cfg, "offload_encoder", False):
            offloaded = offload_non_decoder(self.module.model)
            if offloaded:
                yield TrainingUpdate(0, 0.0, f"[INFO] Offloaded {offloaded} model components to CPU (saves VRAM)", kind="info")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # -- dtype / Fabric setup -------------------------------------------
        self.module.model.decoder, optimizer = self.fabric.setup(self.module.model.decoder, optimizer)
        train_loader = self.fabric.setup_dataloaders(train_loader)
        if val_loader is not None:
            val_loader = self.fabric.setup_dataloaders(val_loader)

        # -- Resume ---------------------------------------------------------
        start_epoch = 0
        global_step = 0

        if cfg.resume_from and Path(cfg.resume_from).exists():
            try:
                yield TrainingUpdate(0, 0.0, f"[INFO] Loading checkpoint from {cfg.resume_from}", kind="info")
                resumed = yield from self._resume_checkpoint(
                    cfg.resume_from, optimizer, scheduler,
                )
                if resumed is not None:
                    start_epoch, global_step = resumed
            except Exception as exc:
                logger.exception("Failed to load checkpoint")
                if getattr(cfg, "identity_v5_enabled", False) or getattr(cfg, "phase_d_resume_adapter", None) is not None:
                    raise RuntimeError(f"Controlled training refuses to continue after checkpoint resume failure: {exc}") from exc
                yield TrainingUpdate(0, 0.0, f"[WARN] Checkpoint load failed: {exc} -- starting fresh", kind="warn")
                start_epoch = 0
                global_step = 0
        elif getattr(cfg, "identity_v5_enabled", False):
            raise RuntimeError("V5 requires an existing exact Phase A resume checkpoint")

        self.module.initialize_v5_teacher()

        # -- Timing health monitor ------------------------------------------
        timing_telemetry_every = max(0, int(getattr(cfg, "timing_telemetry_every", 0)))
        timing_hazard_patience = max(1, int(getattr(cfg, "timing_hazard_patience", 5)))
        timing_fail_on_hazard = bool(getattr(cfg, "timing_fail_on_hazard", False))
        timing_parameters = _timing_named_parameters(self.module)
        initial_timing_gates = {
            name: parameter.detach().float().clone()
            for name, parameter in timing_parameters["timing_gate"]
        }
        timing_streaks = {
            "missing_conditioning": 0,
            "zero_gate_gradient": 0,
            "zero_consumer_gradient": 0,
            "zero_timing_signal": 0,
            "closed_encoder_output_gate": 0,
            "unchanged_gate": 0,
        }
        timing_telemetry_path = output_dir / "timing_telemetry.jsonl"
        timing_hazard_path = output_dir / "timing_hazards.jsonl"

        ablation_mode = getattr(cfg, "phase_d_resume_adapter", None) is not None
        local_step = 0
        max_local_steps = max(0, int(getattr(cfg, "max_optimizer_steps", 0)))
        probe_steps = {
            int(value)
            for value in str(getattr(cfg, "phase_d_probe_steps", "")).split(",")
            if value.strip()
        }
        probe_dir = Path(cfg.phase_d_probe_dir).resolve() if ablation_mode else None
        probe_checkpoints: Dict[str, str] = {}
        d1_roles_enabled = bool(getattr(cfg, "phase_d_d1_alternating_roles_enabled", False))
        d1_step_records: list[dict[str, Any]] = []
        pending_d1_record: dict[str, Any] | None = None
        experiment_manifest_path = (
            Path(cfg.experiment_manifest_out).resolve() if ablation_mode else None
        )
        experiment_manifest: Dict[str, Any] = {}

        if ablation_mode:
            if not bool(getattr(cfg, "strict_sidecars", False)):
                raise RuntimeError("controlled Phase-D execution requires --strict-sidecars")
            if not bool(getattr(cfg, "strict_timing_state_load", False)):
                raise RuntimeError("controlled Phase-D execution requires --strict-timing-state-load")
            if not getattr(cfg, "resume_from", None) or not Path(cfg.resume_from).exists():
                raise FileNotFoundError(f"controlled parent checkpoint is missing: {getattr(cfg, 'resume_from', None)}")
            if not getattr(cfg, "phase_d_probe_dir", None) or not getattr(cfg, "experiment_manifest_out", None):
                raise RuntimeError("controlled Phase-D probe and manifest paths are required")
            if max_local_steps <= 0:
                raise RuntimeError("controlled Phase-D max_optimizer_steps must be positive")
            if not probe_steps or 0 not in probe_steps or max_local_steps not in probe_steps:
                raise RuntimeError("probe steps must include 0 and max_optimizer_steps")
            if cfg.phase_d_optimizer_policy != "fresh":
                raise RuntimeError("main Phase-D ablation requires a fresh optimizer")
            startup = self.module.phase_d_ablation_snapshot()
            if startup["decoder_timing_trainable_count"] != 49_152:
                raise RuntimeError(f"timing topology mismatch: {startup}")
            if startup["timing_consumer_lora_b_nonzero"] != 0:
                raise RuntimeError("timing consumer LoRA B must start frozen and zero")
            if cfg.timing_init_profile == "historical_v4":
                resuming_parent_timing = bool(getattr(cfg, "phase_d_require_parent_timing_state", False))
                if not resuming_parent_timing:
                    if abs(startup["output_gate_logit"]) > 1e-8 or startup["global_gate_logit"] is not None:
                        raise RuntimeError(f"historical timing gate mismatch: {startup}")
                    if startup["projection_l2"] <= 0.0 or startup["projection_bias_l2"] != 0.0:
                        raise RuntimeError(f"historical projection mismatch: {startup}")
                    if not 0.015 <= startup["stream_embedding_std"] <= 0.025:
                        raise RuntimeError(f"historical stream initialization mismatch: {startup}")
                elif startup["global_gate_logit"] is not None:
                    raise RuntimeError(f"D1 resumed timing state unexpectedly contains a global gate: {startup}")
            elif cfg.timing_init_profile == "suppressed_new":
                if abs(startup["output_gate_logit"] + 8.0) > 1e-8:
                    raise RuntimeError(f"suppressed output gate mismatch: {startup}")
                if startup["projection_l2"] != 0.0 or startup["stream_embedding_l2"] != 0.0:
                    raise RuntimeError(f"suppressed source initialization mismatch: {startup}")
            else:
                raise RuntimeError(f"unsupported ablation timing profile: {cfg.timing_init_profile}")

            fixed_latent_path = Path(cfg.phase_d_fixed_latent_path).resolve()
            if not fixed_latent_path.is_file():
                raise FileNotFoundError(
                    f"fixed latent must be prepared before training; missing: {fixed_latent_path}"
                )
            fixed_payload = torch.load(fixed_latent_path, map_location="cpu", weights_only=True)
            if not isinstance(fixed_payload, dict):
                raise TypeError("fixed latent payload must be a metadata dictionary")
            if fixed_payload.get("schema_version") != "phase_d_fixed_latent_v1":
                raise RuntimeError(
                    f"fixed latent schema mismatch: {fixed_payload.get('schema_version')!r}"
                )
            fixed_tensor = fixed_payload.get("initial_latent")
            if not isinstance(fixed_tensor, torch.Tensor) or tuple(fixed_tensor.shape) != (1, 8500, 64):
                raise RuntimeError(f"fixed latent shape mismatch: {getattr(fixed_tensor, 'shape', None)}")
            if fixed_tensor.dtype != torch.float32:
                raise RuntimeError(f"fixed latent dtype mismatch: {fixed_tensor.dtype}")
            if not bool(torch.isfinite(fixed_tensor).all()):
                raise RuntimeError("fixed latent contains non-finite values")
            if int(fixed_payload.get("seed", -1)) != int(cfg.seed):
                raise RuntimeError(
                    f"fixed latent seed mismatch: {fixed_payload.get('seed')} vs {cfg.seed}"
                )
            if fixed_payload.get("shape") != [1, 8500, 64] or fixed_payload.get("dtype") != "torch.float32":
                raise RuntimeError("fixed latent payload shape/dtype declaration mismatch")
            fixed_tensor_sha256 = hashlib.sha256(
                fixed_tensor.contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            if fixed_payload.get("tensor_sha256") != fixed_tensor_sha256:
                raise RuntimeError("fixed latent embedded tensor hash mismatch")

            train_dataset = train_loader.dataset
            paths = _ordered_dataset_paths(train_dataset)
            order_digest = hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()
            base_dataset = train_dataset
            while hasattr(base_dataset, "dataset"):
                base_dataset = base_dataset.dataset
            sidecar_report = getattr(base_dataset, "timing_sidecar_report", None)
            if not sidecar_report or sidecar_report.get("valid") != sidecar_report.get("tensor_count"):
                raise RuntimeError(f"strict sidecar coverage mismatch: {sidecar_report}")
            # MERT is disabled for this experiment, but provenance still reports
            # coverage over the complete 98-sample dataset, not only the train split.
            full_dataset_paths = list(getattr(base_dataset, "valid_paths", []))
            if len(full_dataset_paths) != int(sidecar_report["tensor_count"]):
                raise RuntimeError(
                    "full-dataset provenance path count differs from strict sidecar report: "
                    f"paths={len(full_dataset_paths)} sidecars={sidecar_report['tensor_count']}"
                )
            mert_valid = 0
            mert_masked = []
            for tensor_path in full_dataset_paths:
                sample = torch.load(tensor_path, map_location="cpu", weights_only=True)
                features = sample.get("ref_voice_features")
                mask = sample.get("ref_voice_attention_mask")
                conditioned = (
                    isinstance(features, torch.Tensor) and features.numel() > 0
                    and isinstance(mask, torch.Tensor) and mask.numel() > 0
                    and bool(mask.bool().any())
                )
                if conditioned:
                    mert_valid += 1
                else:
                    mert_masked.append(tensor_path)

            runtime_decoder = self.module.model.decoder
            while hasattr(runtime_decoder, "_forward_module"):
                runtime_decoder = runtime_decoder._forward_module
            dynamic_decoder = (
                runtime_decoder.get_base_model()
                if hasattr(runtime_decoder, "get_base_model")
                else runtime_decoder
            )
            dynamic_path = Path(inspect.getfile(dynamic_decoder.__class__)).resolve()
            required_runtime_sources = [
                Path(__file__).resolve(),
                Path(inspect.getfile(FixedLoRAModule)).resolve(),
                Path(inspect.getfile(resume_checkpoint)).resolve(),
                Path(inspect.getfile(build_scheduler)).resolve(),
                Path(inspect.getfile(type(self.module.timing_encoder))).resolve(),
                Path(inspect.getfile(type(data_module))).resolve(),
            ]
            model_load_raw = os.environ.get("PHASE_D_STRICT_MODEL_LOAD_AUDIT")
            if not model_load_raw:
                raise RuntimeError("PHASE_D_STRICT_MODEL_LOAD_AUDIT is missing")
            model_load_audit = json.loads(model_load_raw)
            if not isinstance(model_load_audit, dict) or model_load_audit.get("attention_backend") != "sdpa":
                raise RuntimeError(f"invalid strict model-load audit: {model_load_audit}")
            if Path(str(model_load_audit.get("dynamic_module_path", ""))).resolve() != dynamic_path:
                raise RuntimeError("strict model-load dynamic source differs from runtime model source")
            guarded_raw = os.environ.get("PHASE_D_GUARDED_SOURCES")
            if not guarded_raw:
                raise RuntimeError("PHASE_D_GUARDED_SOURCES was not supplied by the controlled launcher")
            guarded_values = json.loads(guarded_raw)
            if not isinstance(guarded_values, list) or not guarded_values:
                raise RuntimeError("PHASE_D_GUARDED_SOURCES must be a non-empty JSON list")
            source_paths = [Path(str(value)).resolve() for value in guarded_values]
            if len(source_paths) != len(set(source_paths)):
                raise RuntimeError("controlled guarded source list contains duplicates")
            missing_guarded = [str(source) for source in source_paths if not source.is_file()]
            if missing_guarded:
                raise FileNotFoundError(f"guarded source files are missing: {missing_guarded}")
            missing_required = [
                str(source) for source in required_runtime_sources if source not in set(source_paths)
            ]
            if missing_required:
                raise RuntimeError(
                    f"controlled source guard omitted imported training modules: {missing_required}"
                )
            source_hashes = {str(source): _sha256_file(source) for source in source_paths}
            resume_report = getattr(self, "_phase_d_resume_report", None)
            if not resume_report:
                raise RuntimeError("controlled resume report is missing")
            adapter_report = resume_report["adapter"]
            freeze_base_adapter = bool(getattr(cfg, "freeze_base_adapter_in_phase_d", False))
            frozen_base_adapter_sha256 = _base_adapter_tensor_sha256(self.module)
            if freeze_base_adapter and not self.module._frozen_base_adapter_names:
                raise RuntimeError("frozen Phase-D base-adapter manifest is empty")
            first_lr = float(optimizer.param_groups[0]["lr"])
            if cfg.phase_d_scheduler_policy == "inherit" and not (4.0e-6 <= first_lr <= 6.0e-6):
                raise RuntimeError(f"inherited scheduler produced unexpected first LR: {first_lr}")
            if cfg.phase_d_scheduler_policy == "fresh":
                requested_lr = float(cfg.learning_rate)
                if not (0.99 * requested_lr <= first_lr <= 1.01 * requested_lr):
                    raise RuntimeError(
                        f"fresh scheduler first LR mismatch: actual={first_lr}, requested={requested_lr}"
                    )
            if resume_report["optimizer_loaded"] or (
                cfg.phase_d_scheduler_policy == "fresh" and resume_report["scheduler_loaded"]
            ):
                raise RuntimeError(f"fresh state restoration contract violated: {resume_report}")
            if not bool(getattr(cfg, "use_mert_conditioning", False)) and resume_report["mert_bridge_loaded"]:
                raise RuntimeError("MERT-disabled experiment unexpectedly loaded a MERT bridge")
            if d1_roles_enabled:
                timing_resume = resume_report.get("timing_state")
                if not isinstance(timing_resume, dict) or timing_resume.get("loaded") is not True:
                    raise RuntimeError(f"D1 exact parent timing-state report is missing: {timing_resume}")
                if float(timing_resume.get("max_abs_difference", -1.0)) != 0.0:
                    raise RuntimeError(f"D1 parent timing-state tensor mismatch: {timing_resume}")
                if timing_resume.get("source_sha256") != timing_resume.get("runtime_sha256"):
                    raise RuntimeError(f"D1 parent timing-state hash mismatch: {timing_resume}")
            teacher_parameters = list(self.module.phase_a_teacher_decoder.parameters()) if self.module.phase_a_teacher_decoder is not None else []
            teacher_trainable_count = sum(parameter.numel() for parameter in teacher_parameters if parameter.requires_grad)
            if float(getattr(cfg, "parent_preservation_loss_weight", 0.0)) > 0.0 and not teacher_parameters:
                raise RuntimeError("parent preservation requested but the frozen teacher is absent")
            if teacher_trainable_count != 0:
                raise RuntimeError(f"parent teacher has trainable parameters: {teacher_trainable_count}")
            experiment_manifest = {
                "resume_adapter": cfg.phase_d_resume_adapter,
                "parent_adapter_loaded": adapter_report["parent_adapter_loaded"],
                "parent_adapter_verified": adapter_report["parent_adapter_verified"],
                "adapter_load_report": adapter_report,
                "scheduler_policy": cfg.phase_d_scheduler_policy,
                "scheduler_horizon_steps": int(total_steps),
                "optimizer_policy": cfg.phase_d_optimizer_policy,
                "timing_init_profile": cfg.timing_init_profile,
                "seed": int(cfg.seed),
                "max_optimizer_steps": max_local_steps,
                "strict_sidecars": bool(cfg.strict_sidecars),
                "strict_timing_state_load": bool(cfg.strict_timing_state_load),
                "initial_adapter_sha256": adapter_report["initial_adapter_sha256"],
                "base_adapter_frozen": freeze_base_adapter,
                "phase_d_clarity_replay_enabled": bool(getattr(cfg, "phase_d_clarity_replay_enabled", False)),
                "phase_d_d1_alternating_roles_enabled": d1_roles_enabled,
                "d1_role_pattern": ["short_clarity", "short_clarity", "full_timing"] if d1_roles_enabled else None,
                "d1_role_manifest_path": str(Path(cfg.phase_d_role_manifest_path).resolve()) if d1_roles_enabled else None,
                "d1_role_manifest_sha256": _sha256_file(Path(cfg.phase_d_role_manifest_path).resolve()) if d1_roles_enabled else None,
                "optimizer_groups": [
                    {"name": group.get("group_name", str(index)), "lr": float(group["lr"]),
                     "parameter_count": sum(parameter.numel() for parameter in group["params"])}
                    for index, group in enumerate(optimizer.param_groups)
                ],
                "frozen_base_adapter_tensor_count": len(self.module._frozen_base_adapter_names),
                "frozen_base_adapter_names": list(self.module._frozen_base_adapter_names),
                "frozen_base_adapter_sha256": frozen_base_adapter_sha256,
                "parent_checkpoint": str(Path(cfg.resume_from).resolve()),
                "parent_adapter_file_sha256": _sha256_file(Path(cfg.resume_from).resolve() / "adapter_model.safetensors"),
                "adapter_contract": {
                    "r": int(self.adapter_config.r),
                    "lora_alpha": int(self.adapter_config.alpha),
                    "lora_dropout": float(self.adapter_config.dropout),
                    "target_modules": list(self.adapter_config.target_modules),
                },
                "mert_policy": {
                    "enabled": bool(getattr(cfg, "use_mert_conditioning", False)),
                    "require_complete": bool(getattr(cfg, "require_complete_mert", False)),
                    "bridge_loaded": bool(resume_report["mert_bridge_loaded"]),
                },
                "fixed_latent": {
                    "path": str(fixed_latent_path),
                    "sha256": _sha256_file(fixed_latent_path),
                    "tensor_sha256": fixed_tensor_sha256,
                    "shape": list(fixed_tensor.shape),
                    "dtype": str(fixed_tensor.dtype),
                    "seed": int(fixed_payload["seed"]),
                    "schema_version": str(fixed_payload["schema_version"]),
                },
                "probe_checkpoints": probe_checkpoints,
                "status": "running",
                "experiment_completed": False,
                "local_optimizer_steps_completed": 0,
                "started_at": time.strftime("%FT%T%z"),
                "source_hashes": source_hashes,
                "dynamic_module_path": str(dynamic_path),
                "dynamic_module_sha256": _sha256_file(dynamic_path),
                "model_load": model_load_audit,
                "package_versions": {
                    name: importlib.metadata.version(name)
                    for name in ("torch", "peft", "transformers", "lightning")
                },
                "torch_cuda_version": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "data_order_hash": order_digest,
                "data_order_paths": paths,
                "data_order_semantics": "actual optimizer-step role/path order" if d1_roles_enabled else "ordered train-subset membership before DataLoader shuffle",
                "sampler_class": type(getattr(train_loader, "sampler", None)).__name__,
                "sidecar_coverage": sidecar_report,
                "mert_coverage": {
                    "conditioned": mert_valid,
                    "total": len(full_dataset_paths),
                    "masked": len(mert_masked),
                    "masked_paths": mert_masked,
                },
                "actual_first_lr": first_lr,
                "trainable_parameter_summary": startup,
                "parent_resume_report": resume_report,
                "parent_teacher_present": bool(teacher_parameters),
                "parent_teacher_parameter_count": sum(parameter.numel() for parameter in teacher_parameters),
                "parent_teacher_trainable_count": teacher_trainable_count,
                "attention_backend": {
                    "sdpa_flash": bool(torch.backends.cuda.flash_sdp_enabled()),
                    "sdpa_mem_efficient": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
                    "sdpa_math": bool(torch.backends.cuda.math_sdp_enabled()),
                },
                "cache_policy": {"gradient_checkpointing": bool(cfg.gradient_checkpointing)},
                "offload_policy": {"encoder": bool(cfg.offload_encoder)},
                "probe_timing_condition_policy": "forced_on_for_probe_batches_only",
            }
            _atomic_json(experiment_manifest_path, experiment_manifest)

        def write_phase_d_probe(step: int, gradients: dict[str, dict[str, Any]]) -> None:
            if not ablation_mode or step not in probe_steps or str(step) in probe_checkpoints:
                return
            snapshot = self.module.phase_d_ablation_snapshot()
            aux = dict(getattr(self.module, "_last_aux_losses", {}))
            encoder = self.module.timing_encoder
            layer_rms = dict(getattr(self.module, "_timing_layer_residual_rms", {}))
            residual_rms = math.sqrt(
                sum(value * value for value in layer_rms.values()) / max(1, len(layer_rms))
            )
            normal_lora = [
                (name, parameter)
                for name, parameter in self.module.named_parameters()
                if parameter.requires_grad and "lora_" in name and "timing_cross_attn" not in name
            ]
            current_base_adapter_sha256 = _base_adapter_tensor_sha256(self.module)
            teacher_gradient_count = 0
            teacher_gradient_norm_sq = 0.0
            if self.module.phase_a_teacher_decoder is not None:
                for parameter in self.module.phase_a_teacher_decoder.parameters():
                    if parameter.grad is not None:
                        teacher_gradient_count += 1
                        teacher_gradient_norm_sq += float(parameter.grad.detach().float().square().sum().item())
            if teacher_gradient_count or teacher_gradient_norm_sq != 0.0:
                raise RuntimeError(
                    f"parent teacher received gradients at probe {step}: count={teacher_gradient_count} norm2={teacher_gradient_norm_sq}"
                )
            if freeze_base_adapter and current_base_adapter_sha256 != frozen_base_adapter_sha256:
                raise RuntimeError(
                    f"frozen base adapter changed at probe {step}: "
                    f"expected={frozen_base_adapter_sha256} actual={current_base_adapter_sha256}"
                )
            row = {
                "step": int(step),
                "base_adapter_sha256": current_base_adapter_sha256,
                "actual_lr": float(optimizer.param_groups[0]["lr"]),
                "output_gate_logit": snapshot["output_gate_logit"],
                "output_gate_strength": snapshot["output_gate_strength"],
                "projection_l2": snapshot["projection_l2"],
                "projection_std": snapshot["projection_std"],
                "projection_bias_l2": snapshot["projection_bias_l2"],
                "stream_embedding_std": snapshot["stream_embedding_std"],
                "timing_encoder_output_rms": float(getattr(encoder, "_last_output_rms", 0.0)),
                "timing_encoder_pre_projection_rms": float(getattr(encoder, "_last_pre_projection_rms", 0.0)),
                "post_decoder_gate_timing_residual_rms": residual_rms,
                "per_layer_timing_residual_rms": layer_rms,
                "decoder_timing_gate_grad_norm": float(gradients["timing_gate"]["norm"]),
                "normal_lora_grad_norm": _parameter_grad_norm(normal_lora),
                "parent_teacher_gradient_count": teacher_gradient_count,
                "parent_teacher_gradient_norm": math.sqrt(teacher_gradient_norm_sq),
                "timing_encoder_grad_norm": float(gradients["timing_encoder"]["norm"]),
                "timing_encoder_grad_finite": bool(gradients["timing_encoder"]["finite"]),
                "decoder_timing_gate_grad_finite": bool(gradients["timing_gate"]["finite"]),
                "diffusion_loss": float(aux.get("diffusion_loss", 0.0)),
                "timing_aux_loss": float(aux.get("timing_loss", 0.0)),
                "timing_aux_loss_weight": float(aux.get("timing_loss_weight", 0.0)),
                "timing_aux_weighted_loss": float(aux.get("timing_aux_weighted_loss", 0.0)),
                "timing_counterfactual_loss": float(aux.get("timing_counterfactual_loss", 0.0)),
                "timing_correct_flow_loss": float(aux.get("timing_correct_flow_loss", 0.0)),
                "timing_shifted_flow_loss": float(aux.get("timing_shifted_flow_loss", 0.0)),
                "timing_attention_local": float(aux.get("timing_attention_local", 0.0)),
                "decoder_timing_loss": float(aux.get("decoder_timing_loss", 0.0)),
                "decoder_timing_loss_weight": float(aux.get("decoder_timing_loss_weight", 0.0)),
                "decoder_timing_weighted_loss": float(aux.get("decoder_timing_weighted_loss", 0.0)),
                "parent_preservation_loss": float(aux.get("parent_preservation_loss", 0.0)),
                "f0_loss": float(aux.get("f0_loss", 0.0)),
                "batch_role": str(getattr(self.module, "_last_batch_role", "ordinary")),
                "learning_rates": {str(group.get("group_name", index)): float(group["lr"]) for index, group in enumerate(optimizer.param_groups)},
                "d1_data_order_prefix_sha256": hashlib.sha256(json.dumps(d1_step_records, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
                "d1_data_order_prefix_steps": len(d1_step_records),
                "decoder_gate_min": snapshot["decoder_gate_min"],
                "decoder_gate_mean": snapshot["decoder_gate_mean"],
                "decoder_gate_max": snapshot["decoder_gate_max"],
                "timing_consumer_lora_b_nonzero": snapshot["timing_consumer_lora_b_nonzero"],
            }
            required = [
                "actual_lr", "output_gate_logit", "output_gate_strength", "projection_l2", "projection_std",
                "timing_encoder_output_rms", "post_decoder_gate_timing_residual_rms",
                "timing_encoder_grad_norm", "decoder_timing_gate_grad_norm", "diffusion_loss", "timing_aux_loss",
                "timing_aux_loss_weight", "timing_aux_weighted_loss",
                "timing_counterfactual_loss", "timing_correct_flow_loss", "timing_shifted_flow_loss",
                "timing_attention_local",
                "decoder_timing_loss", "decoder_timing_loss_weight", "decoder_timing_weighted_loss",
                "parent_preservation_loss", "f0_loss",
            ]
            if any(not math.isfinite(float(row[key])) for key in required):
                raise RuntimeError(f"non-finite Phase-D probe metric: {row}")
            if not row["timing_encoder_grad_finite"] or not row["decoder_timing_gate_grad_finite"]:
                raise RuntimeError(f"non-finite Phase-D probe gradient: {row}")
            current_source_hashes = {
                str(source): _sha256_file(source)
                for source in source_paths
                if source.is_file()
            }
            if current_source_hashes != source_hashes:
                raise RuntimeError(
                    f"training source changed during run: expected={source_hashes}, actual={current_source_hashes}"
                )
            checkpoint = probe_dir / f"step_{step:06d}"
            self._save_checkpoint(
                optimizer, scheduler, start_epoch, global_step, str(checkpoint),
                extra_state={"phase_d_local_step": step},
            )
            adapter_file = checkpoint / "adapter_model.safetensors"
            if not adapter_file.is_file():
                raise RuntimeError(f"probe adapter was not saved: {adapter_file}")
            row["adapter_file_sha256"] = _sha256_file(adapter_file)
            probe_manifest = {
                "step": int(step),
                "metrics": dict(row),
                "trainable_parameter_manifest": snapshot,
                "source_hashes": current_source_hashes,
                "dynamic_module_path": str(dynamic_path),
                "dynamic_module_sha256": _sha256_file(dynamic_path),
                "parent_checkpoint": str(Path(cfg.resume_from).resolve()),
                "parent_adapter_file_sha256": experiment_manifest["parent_adapter_file_sha256"],
                "fixed_latent": dict(experiment_manifest["fixed_latent"]),
                "mert_policy": dict(experiment_manifest["mert_policy"]),
            }
            probe_manifest_path = checkpoint / "probe_manifest.json"
            _atomic_json(probe_manifest_path, probe_manifest)
            row["probe_manifest_path"] = str(probe_manifest_path.resolve())
            row["probe_manifest_sha256"] = _sha256_file(probe_manifest_path)
            probe_checkpoints[str(step)] = str(checkpoint.resolve())
            _append_jsonl(probe_dir / "metrics.jsonl", row)
            experiment_manifest["probe_checkpoints"] = probe_checkpoints
            experiment_manifest["local_optimizer_steps_completed"] = int(step)
            _atomic_json(experiment_manifest_path, experiment_manifest)

        def timing_gradient_snapshot() -> dict[str, dict[str, Any]]:
            return {
                name: dict(zip(("norm", "finite", "present"), _gradient_health(parameters)))
                for name, parameters in timing_parameters.items()
            }

        def timing_post_step(
            *,
            step: int,
            epoch_number: int,
            gradients: dict[str, dict[str, Any]],
        ) -> tuple[dict[str, Any] | None, list[str]]:
            if timing_telemetry_every <= 0 or not timing_parameters["timing_gate"]:
                return None, []

            aux = dict(getattr(self.module, "_last_aux_losses", {}))
            active = float(aux.get("timing_condition_active", 0.0)) >= 0.5
            dropped = float(aux.get("timing_condition_dropped", 0.0)) >= 0.5
            gate_values = torch.cat(
                [parameter.detach().float().reshape(-1) for _, parameter in timing_parameters["timing_gate"]]
            )
            gate_delta = max(
                float((parameter.detach().float() - initial_timing_gates[name]).abs().max().item())
                for name, parameter in timing_parameters["timing_gate"]
            )
            consumer_b = [
                parameter.detach()
                for name, parameter in timing_parameters["timing_consumer"]
                if "lora_B" in name
            ]
            consumer_b_nonzero = sum(int(torch.count_nonzero(value).item()) for value in consumer_b)
            timing_signal_rms = float(aux.get("timing_signal_rms", 0.0))
            encoder_output_gate = getattr(
                getattr(self.module, "timing_encoder", None),
                "output_gate",
                None,
            )
            encoder_output_gate_sigmoid = (
                float(torch.sigmoid(encoder_output_gate.detach().float()).item())
                if encoder_output_gate is not None
                else 0.0
            )
            decoder_gate_sigmoid_mean = float(torch.sigmoid(gate_values).mean().item())
            effective_timing_strength = timing_signal_rms * decoder_gate_sigmoid_mean

            checks = {
                "missing_conditioning": not active and not dropped,
                "zero_gate_gradient": active and float(gradients["timing_gate"]["norm"]) <= 0.0,
                "zero_consumer_gradient": (
                    active
                    and bool(timing_parameters["timing_consumer"])
                    and float(gradients["timing_consumer"]["norm"]) <= 0.0
                ),
                "zero_timing_signal": active and timing_signal_rms <= 1e-8,
                "closed_encoder_output_gate": (
                    active
                    and encoder_output_gate is not None
                    and encoder_output_gate_sigmoid <= 0.05
                ),
            }
            for name, failed in checks.items():
                timing_streaks[name] = timing_streaks[name] + 1 if failed else 0

            hazards = [
                name
                for name, streak in timing_streaks.items()
                if streak >= timing_hazard_patience
            ]
            if any(not bool(values["finite"]) for values in gradients.values()):
                hazards.append("nonfinite_timing_gradient")
            if not bool(torch.isfinite(gate_values).all()):
                hazards.append("nonfinite_timing_gate")
            if any(parameter.dtype != torch.float32 for _, parameter in timing_parameters["timing_gate"]):
                hazards.append("timing_gate_not_fp32")

            payload: dict[str, Any] = {
                "step": step,
                "epoch": epoch_number,
                "time": time.strftime("%FT%T%z"),
                "losses": {
                    key: float(value)
                    for key, value in aux.items()
                    if key in {
                        "total_loss",
                        "diffusion_loss",
                        "timing_loss",
                        "decoder_timing_loss",
                        "timing_predictor_loss",
                        "timing_expressivity_loss",
                        "timing_condition_active",
                        "timing_condition_dropped",
                        "timing_signal_rms",
                    }
                },
                "gradients": gradients,
                "learning_rates": {
                    str(group.get("group_name", index)): float(group["lr"])
                    for index, group in enumerate(optimizer.param_groups)
                },
                "gate": {
                    "dtype": str(timing_parameters["timing_gate"][0][1].dtype),
                    "min": float(gate_values.min().item()),
                    "mean": float(gate_values.mean().item()),
                    "max": float(gate_values.max().item()),
                    "sigmoid_mean": decoder_gate_sigmoid_mean,
                    "max_abs_delta_from_start": gate_delta,
                },
                "encoder_output_gate": {
                    "raw": (
                        float(encoder_output_gate.detach().float().item())
                        if encoder_output_gate is not None
                        else None
                    ),
                    "sigmoid": encoder_output_gate_sigmoid,
                },
                "effective_timing_strength": effective_timing_strength,
                "timing_consumer_lora_b_nonzero": consumer_b_nonzero,
                "streaks": dict(timing_streaks),
                "hazards": sorted(set(hazards)),
            }
            if step % timing_telemetry_every == 0 or hazards:
                _append_jsonl(timing_telemetry_path, payload)
            if hazards:
                _append_jsonl(timing_hazard_path, payload)
            return payload, sorted(set(hazards))

        d1_probe_batch: dict[str, Any] | None = None
        if d1_roles_enabled:
            train_dataset = data_module.train_dataset
            if train_dataset is None:
                raise RuntimeError("D1 probe cannot access the training dataset")
            for sample_index in range(len(train_dataset)):
                sample = train_dataset[sample_index]
                metadata = sample.get("metadata") if isinstance(sample, dict) else None
                if isinstance(metadata, dict) and metadata.get("phase_d_batch_role") == "full_timing":
                    d1_probe_batch = strict_phase_d_collate([sample])
                    break
            if d1_probe_batch is None:
                raise RuntimeError("D1 probe could not find a full_timing sample in the train split")

        def write_d1_diagnostic_probe(step: int) -> None:
            if not d1_roles_enabled or step not in probe_steps or str(step) in probe_checkpoints:
                return
            assert d1_probe_batch is not None
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            optimizer.zero_grad(set_to_none=True)
            self.module._force_timing_condition_for_probe = True
            try:
                probe_loss = self.module.training_step(d1_probe_batch, record_loss=False)
                if not torch.isfinite(probe_loss):
                    raise RuntimeError(f"D1 full-timing diagnostic probe {step} produced non-finite loss")
                self.fabric.backward(probe_loss)
                probe_gradients = timing_gradient_snapshot()
                if any(not bool(values["finite"]) for values in probe_gradients.values()):
                    raise RuntimeError(f"D1 full-timing diagnostic probe {step} produced non-finite gradients")
                write_phase_d_probe(step, probe_gradients)
            finally:
                self.module._force_timing_condition_for_probe = False
                optimizer.zero_grad(set_to_none=True)
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)

        if d1_roles_enabled:
            self.module.model.decoder.train()
            write_d1_diagnostic_probe(0)

        # -- Training loop --------------------------------------------------
        accumulation_step = 0
        accumulated_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        self.module.model.decoder.train()
        best_val_loss = float("inf")
        best_ckpt_dir: Optional[str] = None
        epochs_without_improvement = 0
        early_stop_output_dir = str(Path(cfg.output_dir).resolve())
        if cfg.resume_from and start_epoch > 0:
            resume_path = Path(cfg.resume_from)
            if resume_path.is_file():
                resume_path = resume_path.parent
            resume_state_path = resume_path / "training_state.pt"
            if resume_state_path.is_file():
                resume_state = torch.load(
                    resume_state_path, map_location="cpu", weights_only=True
                )
                if resume_state.get("early_stopping_output_dir") == early_stop_output_dir:
                    best_val_loss = float(
                        resume_state.get("best_val_loss", best_val_loss)
                    )
                    epochs_without_improvement = int(
                        resume_state.get(
                            "epochs_without_improvement",
                            epochs_without_improvement,
                        )
                    )
                    best_ckpt_dir = resume_state.get("best_ckpt_dir")
                    yield TrainingUpdate(
                        step=global_step,
                        loss=best_val_loss,
                        msg=(
                            "[OK] Restored early-stopping state: "
                            f"best={best_val_loss:.4f}, "
                            f"non_improving={epochs_without_improvement}"
                        ),
                        kind="info",
                    )
        validate_every = max(1, int(getattr(cfg, "validate_every_n_epochs", 1)))
        early_patience = max(0, int(getattr(cfg, "early_stopping_patience", 0)))
        early_min_delta = max(0.0, float(getattr(cfg, "early_stopping_min_delta", 0.0)))
        save_best_checkpoint = bool(getattr(cfg, "save_best_checkpoint", True))
        should_stop_early = False
        reached_max_steps = False

        def early_stop_state() -> Dict[str, Any]:
            return {
                "early_stopping_output_dir": early_stop_output_dir,
                "best_val_loss": best_val_loss,
                "epochs_without_improvement": epochs_without_improvement,
                "best_ckpt_dir": best_ckpt_dir,
            }

        for epoch in range(start_epoch, cfg.max_epochs):
            self.module.apply_v5_freeze_schedule(epoch - start_epoch + 1)
            if hasattr(self.module, "apply_identity_curriculum"):
                self.module.apply_identity_curriculum(epoch + 1)
            epoch_loss = 0.0
            num_updates = 0
            epoch_start = time.time()

            for _batch_idx, batch in enumerate(train_loader):
                # Stop signal
                if training_state and training_state.get("should_stop", False):
                    _stop_loss = (
                        accumulated_loss * cfg.gradient_accumulation_steps
                        / max(accumulation_step, 1)
                    )
                    tb.close()
                    if ablation_mode:
                        raise InterruptedError("Controlled Phase-D training was stopped before contract completion")
                    yield TrainingUpdate(global_step, _stop_loss, "[INFO] Training stopped by user", kind="complete")
                    return

                if d1_roles_enabled:
                    metadata = batch.get("metadata")
                    if not isinstance(metadata, list) or len(metadata) != 1 or not isinstance(metadata[0], dict):
                        raise RuntimeError(f"D1 optimizer step lacks one metadata record: {metadata}")
                    role = metadata[0].get("phase_d_batch_role")
                    expected_role = ("short_clarity", "short_clarity", "full_timing")[local_step % 3]
                    if role != expected_role:
                        raise RuntimeError(f"D1 role schedule mismatch at local step {local_step + 1}: expected={expected_role} actual={role}")
                    tensor_path = str(metadata[0].get("tensor_path", ""))
                    if not tensor_path or not Path(tensor_path).is_file():
                        raise FileNotFoundError(f"D1 batch tensor path is invalid: {tensor_path}")
                    pending_d1_record = {
                        "optimizer_step": local_step + 1,
                        "role": role,
                        "tensor_path": str(Path(tensor_path).resolve()),
                        "tensor_sha256": _sha256_file(Path(tensor_path).resolve()),
                    }

                if ablation_mode:
                    self.module._force_timing_condition_for_probe = bool(
                        local_step == 0 or (local_step + 1) in probe_steps
                    )
                try:
                    loss = self.module.training_step(batch)
                finally:
                    if ablation_mode:
                        self.module._force_timing_condition_for_probe = False
                if (
                    getattr(cfg, "identity_v5_enabled", False)
                    and accumulation_step == 0
                    and self.module._identity_v5_step % 100 <= cfg.gradient_accumulation_steps
                ):
                    self.module._record_v5_gradient_conflicts(
                        self.module._identity_v5_diagnostic_losses,
                        self.fabric.backward,
                    )
                if not torch.isfinite(loss):
                    batch_desc = _summarize_batch_metadata(batch)
                    if getattr(cfg, "skip_nonfinite_gradients", False):
                        logger.warning(
                            "[Side-Step] Skipping non-finite training loss at epoch %d batch %d: %s",
                            epoch + 1,
                            _batch_idx,
                            batch_desc,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        accumulated_loss = 0.0
                        accumulation_step = 0
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    raise RuntimeError(
                        f"Non-finite training loss at epoch {epoch + 1} batch {_batch_idx}: {batch_desc}"
                    )
                loss = loss / cfg.gradient_accumulation_steps
                self.fabric.backward(loss)
                accumulated_loss += loss.item()
                del loss  # free scalar tensor immediately
                accumulation_step += 1

                if accumulation_step >= cfg.gradient_accumulation_steps:
                    if getattr(cfg, "skip_nonfinite_gradients", False):
                        fixed_count, fixed_names = _sanitize_nonfinite_gradients(
                            self.module
                        )
                        if fixed_count:
                            logger.warning(
                                "[Side-Step] Replaced non-finite gradients in %d trainable params before clipping: %s",
                                fixed_count,
                                ", ".join(fixed_names),
                            )
                    timing_gradients = timing_gradient_snapshot()
                    if local_step == 0:
                        write_phase_d_probe(0, timing_gradients)
                    if getattr(cfg, "skip_nonfinite_gradients", False):
                        torch.nn.utils.clip_grad_norm_(
                            trainable_params,
                            cfg.max_grad_norm,
                            error_if_nonfinite=False,
                        )
                    else:
                        self.fabric.clip_gradients(
                            self.module, optimizer, max_norm=cfg.max_grad_norm,
                        )
                    optimizer.step()
                    if getattr(cfg, "skip_nonfinite_gradients", False):
                        fixed_param_count, fixed_param_names = _sanitize_nonfinite_parameters(
                            self.module
                        )
                        if fixed_param_count:
                            logger.warning(
                                "[Side-Step] Replaced non-finite parameter values in %d params after optimizer step: %s",
                                fixed_param_count,
                                ", ".join(fixed_param_names),
                            )
                    scheduler.step()
                    global_step += 1
                    local_step += 1
                    if d1_roles_enabled:
                        if pending_d1_record is None or pending_d1_record["optimizer_step"] != local_step:
                            raise RuntimeError(f"D1 step-order record missing at optimizer step {local_step}")
                        d1_step_records.append(pending_d1_record)
                        pending_d1_record = None
                    timing_payload, timing_hazards = timing_post_step(
                        step=global_step,
                        epoch_number=epoch + 1,
                        gradients=timing_gradients,
                    )
                    if d1_roles_enabled:
                        write_d1_diagnostic_probe(local_step)
                    else:
                        write_phase_d_probe(local_step, timing_gradients)
                    if timing_payload is not None and (
                        global_step % max(1, timing_telemetry_every) == 0
                        or timing_hazards
                    ):
                        gate = timing_payload["gate"]
                        losses = timing_payload["losses"]
                        timing_msg = (
                            f"[TIMING] step={global_step} "
                            f"active={int(losses.get('timing_condition_active', 0.0))} "
                            f"loss={losses.get('timing_loss', 0.0):.4f} "
                            f"decoder={losses.get('decoder_timing_loss', 0.0):.4f} "
                            f"gate_grad={timing_payload['gradients']['timing_gate']['norm']:.3e} "
                            f"gate_mean={gate['mean']:.6f} "
                            f"gate_delta={gate['max_abs_delta_from_start']:.3e} "
                            f"signal_rms={losses.get('timing_signal_rms', 0.0):.3e} "
                            f"effective={timing_payload['effective_timing_strength']:.3e} "
                            f"consumer_B_nz={timing_payload['timing_consumer_lora_b_nonzero']} "
                            f"hazards={','.join(timing_hazards) if timing_hazards else 'none'}"
                        )
                        yield TrainingUpdate(
                            step=global_step,
                            loss=float(losses.get("total_loss", 0.0)),
                            msg=timing_msg,
                            kind="warn" if timing_hazards else "info",
                            epoch=epoch + 1,
                            max_epochs=cfg.max_epochs,
                        )
                    if timing_hazards and timing_fail_on_hazard:
                        raise RuntimeError(
                            "Timing health hazard: " + ", ".join(timing_hazards)
                        )

                    avg_loss = accumulated_loss * cfg.gradient_accumulation_steps / accumulation_step
                    _lr = scheduler.get_last_lr()[0]
                    if global_step % cfg.log_every == 0:
                        tb.log_loss(avg_loss, global_step)
                        tb.log_lr(_lr, global_step)
                        yield TrainingUpdate(
                            step=global_step, loss=avg_loss,
                            msg=f"Epoch {epoch + 1}/{cfg.max_epochs}, Step {global_step}, Loss: {avg_loss:.4f}",
                            kind="step", epoch=epoch + 1, max_epochs=cfg.max_epochs, lr=_lr,
                            steps_per_epoch=steps_per_epoch,
                        )

                    if global_step % cfg.log_heavy_every == 0:
                        tb.log_per_layer_grad_norms(self.module.model, global_step)

                    optimizer.zero_grad(set_to_none=True)
                    epoch_loss += avg_loss
                    num_updates += 1
                    accumulated_loss = 0.0
                    accumulation_step = 0

                    # Periodic CUDA cache cleanup to prevent intra-epoch
                    # memory fragmentation on consumer GPUs.
                    if torch.cuda.is_available() and global_step % cfg.log_every == 0:
                        torch.cuda.empty_cache()
                    if max_local_steps > 0 and local_step >= max_local_steps:
                        reached_max_steps = True
                        break

            # Flush remainder
            if accumulation_step > 0:
                if getattr(cfg, "skip_nonfinite_gradients", False):
                    fixed_count, fixed_names = _sanitize_nonfinite_gradients(
                        self.module
                    )
                    if fixed_count:
                        logger.warning(
                            "[Side-Step] Replaced non-finite gradients in %d trainable params before clipping: %s",
                            fixed_count,
                            ", ".join(fixed_names),
                        )
                timing_gradients = timing_gradient_snapshot()
                if local_step == 0:
                    write_phase_d_probe(0, timing_gradients)
                if getattr(cfg, "skip_nonfinite_gradients", False):
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params,
                        cfg.max_grad_norm,
                        error_if_nonfinite=False,
                    )
                else:
                    self.fabric.clip_gradients(
                        self.module, optimizer, max_norm=cfg.max_grad_norm,
                    )
                optimizer.step()
                if getattr(cfg, "skip_nonfinite_gradients", False):
                    fixed_param_count, fixed_param_names = _sanitize_nonfinite_parameters(
                        self.module
                    )
                    if fixed_param_count:
                        logger.warning(
                            "[Side-Step] Replaced non-finite parameter values in %d params after optimizer step: %s",
                            fixed_param_count,
                            ", ".join(fixed_param_names),
                        )
                scheduler.step()
                global_step += 1
                local_step += 1
                if d1_roles_enabled:
                    if pending_d1_record is None or pending_d1_record["optimizer_step"] != local_step:
                        raise RuntimeError(f"D1 step-order record missing at optimizer step {local_step}")
                    d1_step_records.append(pending_d1_record)
                    pending_d1_record = None
                timing_payload, timing_hazards = timing_post_step(
                    step=global_step,
                    epoch_number=epoch + 1,
                    gradients=timing_gradients,
                )
                if d1_roles_enabled:
                    write_d1_diagnostic_probe(local_step)
                else:
                    write_phase_d_probe(local_step, timing_gradients)
                if timing_payload is not None and (
                    global_step % max(1, timing_telemetry_every) == 0
                    or timing_hazards
                ):
                    gate = timing_payload["gate"]
                    losses = timing_payload["losses"]
                    timing_msg = (
                        f"[TIMING] step={global_step} "
                        f"active={int(losses.get('timing_condition_active', 0.0))} "
                        f"loss={losses.get('timing_loss', 0.0):.4f} "
                        f"decoder={losses.get('decoder_timing_loss', 0.0):.4f} "
                        f"gate_grad={timing_payload['gradients']['timing_gate']['norm']:.3e} "
                        f"gate_mean={gate['mean']:.6f} "
                        f"gate_delta={gate['max_abs_delta_from_start']:.3e} "
                        f"signal_rms={losses.get('timing_signal_rms', 0.0):.3e} "
                        f"effective={timing_payload['effective_timing_strength']:.3e} "
                        f"consumer_B_nz={timing_payload['timing_consumer_lora_b_nonzero']} "
                        f"hazards={','.join(timing_hazards) if timing_hazards else 'none'}"
                    )
                    yield TrainingUpdate(
                        step=global_step,
                        loss=float(losses.get("total_loss", 0.0)),
                        msg=timing_msg,
                        kind="warn" if timing_hazards else "info",
                        epoch=epoch + 1,
                        max_epochs=cfg.max_epochs,
                    )
                if timing_hazards and timing_fail_on_hazard:
                    raise RuntimeError(
                        "Timing health hazard: " + ", ".join(timing_hazards)
                    )

                avg_loss = accumulated_loss * cfg.gradient_accumulation_steps / accumulation_step
                _lr = scheduler.get_last_lr()[0]
                if global_step % cfg.log_every == 0:
                    tb.log_loss(avg_loss, global_step)
                    tb.log_lr(_lr, global_step)
                    yield TrainingUpdate(
                        step=global_step, loss=avg_loss,
                        msg=f"Epoch {epoch + 1}/{cfg.max_epochs}, Step {global_step}, Loss: {avg_loss:.4f}",
                        kind="step", epoch=epoch + 1, max_epochs=cfg.max_epochs, lr=_lr,
                        steps_per_epoch=steps_per_epoch,
                    )

                optimizer.zero_grad(set_to_none=True)
                epoch_loss += avg_loss
                num_updates += 1
                accumulated_loss = 0.0
                accumulation_step = 0

            # End of epoch
            epoch_time = time.time() - epoch_start
            avg_epoch_loss = epoch_loss / max(num_updates, 1)
            tb.log_epoch_loss(avg_epoch_loss, epoch + 1)
            yield TrainingUpdate(
                step=global_step, loss=avg_epoch_loss,
                msg=f"[OK] Epoch {epoch + 1}/{cfg.max_epochs} in {epoch_time:.1f}s, Loss: {avg_epoch_loss:.4f}",
                kind="epoch", epoch=epoch + 1, max_epochs=cfg.max_epochs, epoch_time=epoch_time,
            )

            if val_loader is not None and ((epoch + 1) % validate_every == 0):
                self.module.model.decoder.eval()
                total_val_loss = 0.0
                n_val = 0
                with torch.no_grad():
                    for val_batch in val_loader:
                        v_loss = self.module.training_step(val_batch, record_loss=False)
                        if not torch.isfinite(v_loss):
                            batch_desc = _summarize_batch_metadata(val_batch)
                            if ablation_mode:
                                raise RuntimeError(
                                    f"Non-finite validation loss at epoch {epoch + 1}: {batch_desc}"
                                )
                            logger.warning(
                                "[Side-Step] Skipping non-finite validation loss at epoch %d: %s",
                                epoch + 1,
                                batch_desc,
                            )
                            continue
                        total_val_loss += float(v_loss.item())
                        n_val += 1
                self.module.model.decoder.train()

                if n_val == 0:
                    if ablation_mode:
                        raise RuntimeError(
                            f"No finite validation batches at epoch {epoch + 1}"
                        )
                    val_loss = float("inf")
                    logger.warning(
                        "[Side-Step] All validation batches were non-finite at epoch %d",
                        epoch + 1,
                    )
                else:
                    val_loss = total_val_loss / n_val
                tb.log_scalar("val/loss", val_loss, epoch + 1)
                yield TrainingUpdate(
                    step=global_step,
                    loss=val_loss,
                    msg=f"[INFO] Validation after epoch {epoch + 1}: loss={val_loss:.4f}",
                    kind="info",
                    epoch=epoch + 1,
                    max_epochs=cfg.max_epochs,
                )

                if val_loss < (best_val_loss - early_min_delta):
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    if save_best_checkpoint:
                        best_ckpt_dir = str(output_dir / "checkpoints" / "best")
                        self._save_checkpoint(
                            optimizer,
                            scheduler,
                            epoch + 1,
                            global_step,
                            best_ckpt_dir,
                            extra_state=early_stop_state(),
                        )
                        yield TrainingUpdate(
                            step=global_step,
                            loss=val_loss,
                            msg=f"[OK] New best checkpoint saved (val_loss={val_loss:.4f})",
                            kind="checkpoint",
                            epoch=epoch + 1,
                            max_epochs=cfg.max_epochs,
                            checkpoint_path=best_ckpt_dir,
                        )
                else:
                    epochs_without_improvement += 1
                    if early_patience > 0 and epochs_without_improvement >= early_patience:
                        should_stop_early = True
                        yield TrainingUpdate(
                            step=global_step,
                            loss=val_loss,
                            msg=(
                                f"[INFO] Early stopping triggered after epoch {epoch + 1} "
                                f"(best_val_loss={best_val_loss:.4f}, last_val_loss={val_loss:.4f})"
                            ),
                            kind="complete",
                            epoch=epoch + 1,
                            max_epochs=cfg.max_epochs,
                        )

            # Checkpoint
            if (epoch + 1) % cfg.save_every_n_epochs == 0:
                ckpt_dir = str(output_dir / "checkpoints" / f"epoch_{epoch + 1}_loss_{avg_epoch_loss:.4f}")
                self._save_checkpoint(
                    optimizer,
                    scheduler,
                    epoch + 1,
                    global_step,
                    ckpt_dir,
                    extra_state=early_stop_state(),
                )
                yield TrainingUpdate(
                    step=global_step, loss=avg_epoch_loss,
                    msg=f"[OK] Checkpoint saved at epoch {epoch + 1}",
                    kind="checkpoint", epoch=epoch + 1, max_epochs=cfg.max_epochs,
                    checkpoint_path=ckpt_dir,
                )
                eval_template = getattr(cfg, "identity_v5_epoch_eval_command", None)
                if getattr(cfg, "identity_v5_enabled", False) and eval_template:
                    command = shlex.split(eval_template.format(checkpoint=ckpt_dir, epoch=epoch + 1, local_epoch=epoch - start_epoch + 1))
                    result = subprocess.run(command, timeout=int(getattr(cfg, "identity_v5_epoch_eval_timeout_sec", 7200)), check=False)
                    if result.returncode == 3:
                        should_stop_early = True
                        yield TrainingUpdate(step=global_step, loss=avg_epoch_loss, msg=f"[INFO] V5 identity kill criterion stopped arm after epoch {epoch + 1}", kind="complete", epoch=epoch + 1, max_epochs=cfg.max_epochs)
                    elif result.returncode != 0:
                        raise RuntimeError(f"V5 epoch evaluator failed with exit code {result.returncode}: {command}")

            # Clear CUDA cache AFTER checkpoint save so serialization
            # temporaries are also freed.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if should_stop_early or reached_max_steps:
                break

        if ablation_mode and local_step != max_local_steps:
            raise RuntimeError(
                f"ablation stopped at local step {local_step}, expected {max_local_steps}"
            )
        if ablation_mode and set(probe_checkpoints) != {str(value) for value in probe_steps}:
            raise RuntimeError(
                f"missing probe checkpoints: expected={sorted(probe_steps)} actual={sorted(probe_checkpoints)}"
            )

        # -- Sanity check: did we actually train? ----------------------------
        if global_step == 0:
            tb.close()
            if ablation_mode:
                raise RuntimeError("Controlled training completed zero optimizer steps")
            yield TrainingUpdate(
                step=0, loss=0.0,
                msg=(
                    "[FAIL] Training completed 0 steps -- no batches were processed.\n"
                    "       Possible causes:\n"
                    "         - Dataset directory is empty or contains no valid .pt files\n"
                    "         - DataLoader failed to yield batches (device/platform issue)\n"
                    "       Check the dataset path and try again."
                ),
                kind="fail",
            )
            return

        # -- Final save -----------------------------------------------------
        final_path = str(output_dir / "final")
        self._save_final(final_path)
        final_loss = self.module.training_losses[-1] if self.module.training_losses else 0.0

        adapter_label = "LoKR" if self.adapter_type == "lokr" else "LoRA"
        if ablation_mode:
            final_adapter = Path(final_path) / "adapter_model.safetensors"
            final_base_adapter_sha256 = _base_adapter_tensor_sha256(self.module)
            if freeze_base_adapter and final_base_adapter_sha256 != frozen_base_adapter_sha256:
                raise RuntimeError(
                    "frozen base adapter changed before completion: "
                    f"expected={frozen_base_adapter_sha256} actual={final_base_adapter_sha256}"
                )
            if not final_adapter.is_file():
                raise RuntimeError(f"final adapter was not saved: {final_adapter}")
            current_source_hashes = {
                str(source): _sha256_file(source)
                for source in source_paths
                if source.is_file()
            }
            if current_source_hashes != source_hashes:
                raise RuntimeError(
                    f"training source changed before completion: expected={source_hashes}, actual={current_source_hashes}"
                )
            if d1_roles_enabled:
                if len(d1_step_records) != local_step:
                    raise RuntimeError(f"D1 data-order record count mismatch: records={len(d1_step_records)} steps={local_step}")
                expected_roles = [("short_clarity", "short_clarity", "full_timing")[index % 3] for index in range(local_step)]
                actual_roles = [record["role"] for record in d1_step_records]
                if actual_roles != expected_roles:
                    raise RuntimeError("D1 final role order differs from the immutable schedule")
                experiment_manifest["d1_data_order_records"] = d1_step_records
                experiment_manifest["d1_data_order_sha256"] = hashlib.sha256(json.dumps(d1_step_records, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
                experiment_manifest["data_order_hash"] = experiment_manifest["d1_data_order_sha256"]
                experiment_manifest["d1_role_counts"] = {role: actual_roles.count(role) for role in ("short_clarity", "full_timing")}
            experiment_manifest.update({
                "status": "completed",
                "experiment_completed": True,
                "local_optimizer_steps_completed": int(local_step),
                "final_global_step": int(global_step),
                "final_adapter_path": str(final_adapter.resolve()),
                "final_adapter_file_sha256": _sha256_file(final_adapter),
                "final_base_adapter_sha256": final_base_adapter_sha256,
                "final_trainable_parameter_summary": self.module.phase_d_ablation_snapshot(),
                "completed_at": time.strftime("%FT%T%z"),
            })
            _atomic_json(experiment_manifest_path, experiment_manifest)
        tb.flush()
        tb.close()
        yield TrainingUpdate(
            step=global_step, loss=final_loss,
            msg=(
                f"[OK] Training complete! {adapter_label} saved to {final_path}\n"
                f"     For inference, set your LoRA path to: {final_path}"
                + (f"\n     Best validation checkpoint: {best_ckpt_dir}" if best_ckpt_dir else "")
            ),
            kind="complete",
        )
