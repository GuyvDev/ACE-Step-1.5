"""Single source of truth for the actual-new-chain Phase-D controlled experiment."""
from __future__ import annotations

from pathlib import Path
from typing import Any

WORKSPACE = Path("/home/projects/sipl-prj10043")
PROJECT_ROOT = WORKSPACE / "sub-new_plan_2"
ACE_REPO = PROJECT_ROOT / "ACE-Step-1.5"
TOOLS_ROOT = WORKSPACE / "scripts/new_chain/phase_d_coupling_recovery"
PARENT_CHECKPOINT = (
    ACE_REPO
    / "training_runs/new_chain_pure_v4_c0_no_mert_v1/phase_c/lora_output/checkpoints/epoch_25_loss_0.8750"
)
DATASET_DIR = ACE_REPO / "training_runs/reviewed_identity_v1_phase_c_shared/train_tensors"
TIMING_DIR = ACE_REPO / "training_runs/reviewed_identity_v1_phase_c_shared/timing"
FIXED_LATENT_PATH = (
    PROJECT_ROOT
    / "outputs/new_chain/experiments/phase_d_actual_new_chain_clean_v3_shared/fixed_initial_latent.pt"
)
ARM_ID = "actual_new_chain_c25_verified_historical_fresh"
EXACT_ARM = {
    "id": ARM_ID,
    "resume_adapter": "verified",
    "scheduler_policy": "fresh",
    "timing_init_profile": "historical_v4",
    "optimizer_policy": "fresh",
}
EXACT_PARENT_ADAPTER = {
    "r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}
SOURCE_FILES = [
    "strict_phase_d_train.py",
    "acestep/training_v2/configs.py",
    "acestep/training_v2/fixed_lora_module.py",
    "acestep/training_v2/trainer_fixed.py",
    "acestep/training_v2/trainer_helpers.py",
    "acestep/training_v2/optim.py",
    "acestep/training_v2/timing_conditioning.py",
    "acestep/training_v2/phase_d_contract.py",
    "acestep/training_v2/phase_d_strict_runtime.py",
    "acestep/training_v2/phase_d_strict_data.py",
    "acestep/core/generation/handler/generate_music.py",
    "acestep/core/generation/handler/generate_music_execute.py",
    "acestep/core/generation/handler/service_generate.py",
    "acestep/core/generation/handler/service_generate_execute.py",
    "checkpoints/acestep-v15-base/modeling_acestep_v15_base.py",
    str(PROJECT_ROOT / "run_acestep_phase_d_strict_infer.py"),
    str(TOOLS_ROOT / "run_phase_d_ablation.py"),
    str(TOOLS_ROOT / "run_phase_d_probe_matrix.py"),
    str(TOOLS_ROOT / "verify_phase_d_ablation_outputs.py"),
    str(TOOLS_ROOT / "evaluate_actual_new_chain_phase_d_result.py"),
    str(TOOLS_ROOT / "phase_d_runtime_patcher.py"),
    str(TOOLS_ROOT / "prepare_fixed_latent.py"),
    str(TOOLS_ROOT / "run_actual_new_chain_200step_autopilot.sh"),
    str(TOOLS_ROOT / "verify_clean_phase_d_installation.py"),
    str(WORKSPACE / "scripts/new_chain/pure_v4_c0_no_mert_v1/diagnose_pianoman_timing.py"),
    str(WORKSPACE / "scripts/new_chain/pure_v4_c0_no_mert_v1/run_literal_d_vs_original_mert_benchmark.py"),
    "acestep/training_v2/phase_d_strict_model_loader.py",
    "acestep/core/generation/handler/init_service_loader.py",
]


def expected_config(role: str) -> dict[str, Any]:
    if role == "smoke":
        steps = 2
        probes = [0, 1, 2]
        output_root = PROJECT_ROOT / "outputs/new_chain/experiments/phase_d_actual_new_chain_clean_v3_smoke"
    elif role == "main":
        steps = 200
        probes = [0, 50, 100, 200]
        output_root = PROJECT_ROOT / "outputs/new_chain/experiments/phase_d_actual_new_chain_clean_v3_200step"
    else:
        raise RuntimeError(f"experiment_role must be exactly 'smoke' or 'main', got {role!r}")
    return {
        "project_root": str(PROJECT_ROOT),
        "python": str(PROJECT_ROOT / "acestep_env_clean/bin/python"),
        "ace_repo": str(ACE_REPO),
        "train_entrypoint": str(ACE_REPO / "strict_phase_d_train.py"),
        "dataset_dir": str(DATASET_DIR),
        "timing_dir": str(TIMING_DIR),
        "parent_checkpoint": str(PARENT_CHECKPOINT),
        "parent_adapter_contract": {
            "r": EXACT_PARENT_ADAPTER["r"],
            "lora_alpha": EXACT_PARENT_ADAPTER["lora_alpha"],
            "lora_dropout": EXACT_PARENT_ADAPTER["lora_dropout"],
            "target_modules": list(EXACT_PARENT_ADAPTER["target_modules"]),
        },
        "output_root": str(output_root),
        "checkpoint_dir": str(ACE_REPO / "checkpoints"),
        "model_variant": "base",
        "device": "cuda",
        "precision": "bf16",
        "seed": 42,
        "max_optimizer_steps": steps,
        "probe_steps": probes,
        "batch_size": 2,
        "gradient_accumulation": 1,
        "requested_learning_rate": 5e-5,
        "timing_condition_scale": 1.0,
        "timing_condition_dropout": 0.1,
        "timing_decoder_loss_weight": 1.0,
        "timing_condition_source": "sidecar",
        "enable_timing_predictor": False,
        "use_mert_conditioning": False,
        "require_complete_mert": False,
        "fixed_latent_path": str(FIXED_LATENT_PATH),
        "arms": [EXACT_ARM],
        "experiment_contract": "actual_new_chain_single",
        "experiment_role": role,
        "expected_tensor_count": 98,
        "inference_entrypoint": str(PROJECT_ROOT / "run_acestep_phase_d_strict_infer.py"),
        "runtime_verifier": str(TOOLS_ROOT / "verify_clean_phase_d_installation.py"),
        "source_files": SOURCE_FILES,
    }


def validate_config_exact(config: dict[str, Any]) -> dict[str, Any]:
    role = config.get("experiment_role")
    expected = expected_config(str(role))
    missing = sorted(set(expected) - set(config))
    extra = sorted(set(config) - set(expected))
    mismatches = {
        key: {"expected": expected[key], "actual": config.get(key)}
        for key in expected
        if config.get(key) != expected[key]
    }
    if missing or extra or mismatches:
        raise RuntimeError(
            "actual-new-chain config contract failed: "
            f"missing={missing}, extra={extra}, mismatches={mismatches}"
        )
    return expected



def resolve_guarded_sources(config: dict[str, Any]) -> list[Path]:
    """Resolve the immutable source set for the controlled run.

    The explicit list covers entrypoints, metrics, and checkpoint dynamic code.
    The recursive ACE-Step package scan prevents an unlisted imported helper from
    changing between preflight, training, and rendering. Symlinks and escapes are
    rejected rather than followed.
    """
    validate_config_exact(config)
    ace_repo = Path(str(config["ace_repo"])).resolve()
    package_root = (ace_repo / "acestep").resolve()
    if not package_root.is_dir():
        raise FileNotFoundError(f"ACE-Step Python package is missing: {package_root}")
    paths: list[Path] = []
    for raw in config["source_files"]:
        path = Path(str(raw))
        if not path.is_absolute():
            path = ace_repo / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"guarded source is missing: {path}")
        paths.append(path)
    guarded_roots = [
        package_root,
        (ace_repo / "checkpoints/acestep-v15-base").resolve(),
    ]
    for guarded_root in guarded_roots:
        if not guarded_root.is_dir():
            raise FileNotFoundError(f"guarded source root is missing: {guarded_root}")
        for candidate in sorted(guarded_root.rglob("*.py")):
            if candidate.is_symlink():
                raise RuntimeError(f"guarded source may not be a symlink: {candidate}")
            resolved = candidate.resolve()
            if not resolved.is_relative_to(guarded_root):
                raise RuntimeError(f"guarded source escapes root {guarded_root}: {resolved}")
            paths.append(resolved)
    unique = sorted(set(paths), key=str)
    if len(unique) != len(set(unique)):
        raise RuntimeError("guarded source resolution produced duplicates")
    return unique

def expected_arm_dir(role: str) -> Path:
    return Path(expected_config(role)["output_root"]) / ARM_ID
