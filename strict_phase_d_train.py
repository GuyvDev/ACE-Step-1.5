#!/usr/bin/env python3
"""Dedicated fail-closed entrypoint for the actual-new-chain Phase-D test.

This deliberately bypasses the generic ACE-Step CLI so no hidden default, resume
shortcut, or compatibility fallback can alter the controlled contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _contract(config: dict[str, Any], arm_dir: Path) -> None:
    from acestep.training_v2.phase_d_contract import (
        ARM_ID,
        EXACT_ARM,
        expected_arm_dir,
        validate_config_exact,
    )

    validate_config_exact(config)
    role = str(config["experiment_role"])
    _require(arm_dir == expected_arm_dir(role).resolve(), "arm directory differs from exact shared contract")
    _require(Path(config["train_entrypoint"]).resolve() == Path(__file__).resolve(), "train entrypoint mismatch")
    _require(Path(config["parent_checkpoint"]).is_dir(), "parent checkpoint directory is missing")
    _require(Path(config["dataset_dir"]).is_dir(), "dataset directory is missing")
    _require(Path(config["timing_dir"]).is_dir(), "timing directory is missing")
    _require(Path(config["checkpoint_dir"]).is_dir(), "checkpoint directory is missing")
    _require(Path(config["fixed_latent_path"]).is_file(), "fixed latent is missing")
    _require(config["arms"] == [EXACT_ARM] and EXACT_ARM["id"] == ARM_ID, "arm contract mismatch")
    _require((arm_dir / "arm_manifest.json").is_file(), "arm manifest is missing")
    _require((arm_dir / "launch.sh").is_file(), "launch script is missing")
    _require(not (arm_dir / "trainer_manifest.json").exists(), "stale trainer manifest exists")
    _require(not (arm_dir / "lora_output").exists(), "stale lora_output exists")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    arm_dir = args.arm_dir.resolve()
    config = _read(config_path)
    _contract(config, arm_dir)

    ace_repo = Path(config["ace_repo"]).resolve()
    if Path.cwd().resolve() != ace_repo:
        raise RuntimeError(f"strict trainer must run from ACE repo: cwd={Path.cwd()}, expected={ace_repo}")
    guarded = os.environ.get("PHASE_D_GUARDED_SOURCES")
    if not guarded:
        raise RuntimeError("PHASE_D_GUARDED_SOURCES is required")
    guarded_paths = json.loads(guarded)
    if not isinstance(guarded_paths, list) or not guarded_paths:
        raise RuntimeError("PHASE_D_GUARDED_SOURCES must be a non-empty JSON list")
    guarded_resolved = [str(Path(str(raw)).resolve()) for raw in guarded_paths]
    if len(guarded_resolved) != len(set(guarded_resolved)):
        raise RuntimeError("PHASE_D_GUARDED_SOURCES contains duplicate paths")
    arm_manifest = _read(arm_dir / "arm_manifest.json")
    if arm_manifest.get("arm") != {
        "id": "actual_new_chain_c25_verified_historical_fresh",
        "resume_adapter": "verified",
        "scheduler_policy": "fresh",
        "timing_init_profile": "historical_v4",
        "optimizer_policy": "fresh",
    }:
        raise RuntimeError("arm manifest contract mismatch")
    if arm_manifest.get("config_sha256") != _sha256_file(config_path):
        raise RuntimeError("config changed after materialization")
    expected_hashes = arm_manifest.get("source_hashes")
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != set(guarded_resolved):
        raise RuntimeError("guarded source paths differ from arm manifest")
    for raw in guarded_resolved:
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(f"guarded source is missing: {path}")
        if _sha256_file(path) != expected_hashes[raw]:
            raise RuntimeError(f"guarded source changed before training: {path}")

    hf_cache = arm_dir / "hf_modules_cache"
    if hf_cache.exists():
        raise RuntimeError(f"stale training Hugging Face module cache exists: {hf_cache}")
    hf_cache.mkdir(parents=False, exist_ok=False)
    os.environ["HF_MODULES_CACHE"] = str(hf_cache)
    os.environ["TRANSFORMERS_DYNAMIC_MODULE_NAME"] = "transformers_modules"
    os.environ["PHASE_D_STRICT_ATTN_IMPLEMENTATION"] = "sdpa"

    from acestep.training_v2.configs import LoRAConfigV2, TrainingConfigV2
    from acestep.training_v2.phase_d_strict_model_loader import load_phase_d_model_exact
    from acestep.training_v2.trainer_fixed import FixedLoRATrainer

    role = str(config["experiment_role"])
    max_steps = 2 if role == "smoke" else 200
    probes = [0, 1, 2] if role == "smoke" else [0, 50, 100, 200]
    if int(config["max_optimizer_steps"]) != max_steps or config["probe_steps"] != probes:
        raise RuntimeError("step/probe contract mismatch")

    adapter_contract = config["parent_adapter_contract"]
    required_adapter_keys = {"r", "lora_alpha", "lora_dropout", "target_modules"}
    if not isinstance(adapter_contract, dict) or set(adapter_contract) != required_adapter_keys:
        raise RuntimeError(f"invalid immutable parent_adapter_contract: {adapter_contract!r}")
    target_modules = list(adapter_contract["target_modules"])
    if set(target_modules) != {"q_proj", "k_proj", "v_proj", "o_proj"} or len(target_modules) != 4:
        raise RuntimeError(f"invalid parent adapter target_modules contract: {target_modules!r}")
    adapter_cfg = LoRAConfigV2(
        r=int(adapter_contract["r"]),
        alpha=int(adapter_contract["lora_alpha"]),
        dropout=float(adapter_contract["lora_dropout"]),
        target_modules=target_modules,
        bias="none",
        attention_type="both",
    )
    train_cfg = TrainingConfigV2(
        shift=1.0,
        num_inference_steps=50,
        learning_rate=5e-5,
        batch_size=2,
        gradient_accumulation_steps=1,
        max_epochs=33,
        save_every_n_epochs=1,
        warmup_steps=0,
        weight_decay=0.01,
        max_grad_norm=1.0,
        mixed_precision="bf16",
        gradient_checkpointing=True,
        seed=42,
        output_dir=str(arm_dir / "lora_output"),
        num_workers=0,
        pin_memory=True,
        prefetch_factor=0,
        persistent_workers=False,
        val_split=0.1,
        optimizer_type="adamw",
        scheduler_type="cosine",
        cfg_ratio=0.15,
        model_variant=str(config["model_variant"]),
        checkpoint_dir=str(config["checkpoint_dir"]),
        dataset_dir=str(config["dataset_dir"]),
        device=str(config["device"]),
        precision=str(config["precision"]),
        resume_from=str(config["parent_checkpoint"]),
        resume_optimizer_state=False,
        phase_d_resume_adapter="verified",
        phase_d_scheduler_policy="fresh",
        phase_d_optimizer_policy="fresh",
        strict_timing_state_load=True,
        max_optimizer_steps=max_steps,
        phase_d_probe_steps=",".join(str(x) for x in probes),
        phase_d_probe_dir=str(arm_dir / "probes"),
        phase_d_fixed_latent_path=str(config["fixed_latent_path"]),
        experiment_manifest_out=str(arm_dir / "trainer_manifest.json"),
        enable_timing_branch=True,
        timing_init_profile="historical_v4",
        timing_dir=str(config["timing_dir"]),
        strict_sidecars=True,
        timing_hidden_size=256,
        timing_num_heads=4,
        timing_num_layers=2,
        timing_output_dim=0,
        timing_dropout=0.1,
        timing_loss_weight=1.0,
        timing_dur_weight=1.0,
        timing_onset_weight=0.5,
        timing_pause_weight=0.5,
        timing_phrase_weight=0.3,
        timing_tempo_weight=0.2,
        timing_terminal_weight=0.2,
        timing_condition_dropout=0.1,
        timing_condition_scale=1.0,
        timing_use_stream_type_embedding=True,
        timing_decoder_loss_weight=1.0,
        enable_timing_consumer_training=False,
        enable_phrase_modulation=False,
        enable_absolute_time_conditioning=False,
        timing_train_last_n_layers=0,
        timing_encoder_learning_rate=0.0,
        timing_gate_learning_rate=0.0,
        timing_consumer_learning_rate=0.0,
        timing_telemetry_every=1,
        timing_hazard_patience=5,
        timing_fail_on_hazard=True,
        enable_timing_predictor=bool(config["enable_timing_predictor"]),
        timing_condition_source=str(config["timing_condition_source"]),
        timing_predictor_loss_weight=0.0,
        timing_hybrid_mix=0.0,
        use_mert_conditioning=False,
        require_complete_mert=False,
        preserve_mert_init_rng_without_conditioning=False,
        validate_every_n_epochs=1,
        early_stopping_patience=0,
        early_stopping_min_delta=0.0,
        save_best_checkpoint=False,
        skip_nonfinite_gradients=False,
        log_every=1,
        log_heavy_every=50,
    )
    model, model_load_audit = load_phase_d_model_exact(
        checkpoint_dir=train_cfg.checkpoint_dir,
        variant=train_cfg.model_variant,
        device=train_cfg.device,
        precision=train_cfg.precision,
        attention_backend="sdpa",
    )
    os.environ["PHASE_D_STRICT_MODEL_LOAD_AUDIT"] = json.dumps(model_load_audit, sort_keys=True)
    trainer = FixedLoRATrainer(model, adapter_cfg, train_cfg)
    for update in trainer.train():
        print(str(update), flush=True)

    manifest_path = arm_dir / "trainer_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("training returned without a trainer manifest")
    manifest = _read(manifest_path)
    if manifest.get("status") != "completed" or manifest.get("experiment_completed") is not True:
        raise RuntimeError(f"training did not complete the strict contract: {manifest}")
    if int(manifest.get("local_optimizer_steps_completed", -1)) != max_steps:
        raise RuntimeError("training completed with wrong optimizer step count")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("ERROR: interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
