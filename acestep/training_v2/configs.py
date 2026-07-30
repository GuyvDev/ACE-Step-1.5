"""
Extended Training Configuration for ACE-Step Training V2

Uses base configs from ``acestep.training.configs``.  Extends them with
corrected-training-specific fields (CFG dropout,
continuous timestep sampling parameters, estimation, TensorBoard, sample
generation, etc.).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

# Vendored base configs -- no base ACE-Step installation required
from acestep.training.configs import (  # noqa: F401
    LoRAConfig,
    LoKRConfig,
    TrainingConfig,
)


# ---------------------------------------------------------------------------
# Extended LoRA config (unchanged for now, but available for future extension)
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfigV2(LoRAConfig):
    """Extended LoRA configuration.

    Inherits all fields from the original LoRAConfig and adds:
    - attention_type: Which attention layers to target (self, cross, or both)
    """

    attention_type: str = "both"
    """Which attention layers to target: 'self', 'cross', or 'both'."""

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["attention_type"] = self.attention_type
        return base

    # --- Data loading (declared here for compatibility with base packages
    #     that may not include these fields in TrainingConfig) -----------------
    num_workers: int = 4
    """Number of DataLoader worker processes."""

    pin_memory: bool = True
    """Pin memory in DataLoader for faster host-to-device transfer."""

    prefetch_factor: int = 2
    """Number of batches to prefetch per DataLoader worker."""

    persistent_workers: bool = True
    """Keep DataLoader workers alive between epochs."""

    pin_memory_device: str = ""
    """Device for pinned memory ("" = default CUDA device)."""


# ---------------------------------------------------------------------------
# Extended LoKR config
# ---------------------------------------------------------------------------

@dataclass
class LoKRConfigV2(LoKRConfig):
    """Extended LoKR configuration.

    Inherits all fields from the original LoKRConfig and adds:
    - attention_type: Which attention layers to target (self, cross, or both)
    """

    attention_type: str = "both"
    """Which attention layers to target: 'self', 'cross', or 'both'."""

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["attention_type"] = self.attention_type
        return base


# ---------------------------------------------------------------------------
# Extended Training config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfigV2(TrainingConfig):
    """Extended training configuration with corrected-training fields.

    New fields compared to the original TrainingConfig:
    - CFG dropout (cfg_ratio)
    - Continuous timestep sampling parameters (timestep_mu, timestep_sigma,
      data_proportion)
    - Model variant selection
    - Device / precision auto-detection
    - Estimation parameters
    - Extended TensorBoard logging
    - Sample generation during training
    - Checkpoint resume
    - Preprocessing flags
    """

    # --- Data loading (declared here for compatibility with base packages
    #     that may not include these fields in TrainingConfig) -----------------
    num_workers: int = 4
    """Number of DataLoader worker processes."""

    pin_memory: bool = True
    """Pin memory in DataLoader for faster host-to-device transfer."""

    prefetch_factor: int = 2
    """Number of batches to prefetch per DataLoader worker."""

    persistent_workers: bool = True
    """Keep DataLoader workers alive between epochs."""

    pin_memory_device: str = ""
    """Device for pinned memory ("" = default CUDA device)."""

    # --- Optimizer / Scheduler ------------------------------------------------
    optimizer_type: str = "adamw"
    """Optimizer: 'adamw', 'adamw8bit', 'adafactor', 'prodigy'."""

    scheduler_type: str = "cosine"
    """LR scheduler: 'cosine', 'cosine_restarts', 'linear', 'constant', 'constant_with_warmup'."""

    # --- VRAM management ------------------------------------------------------
    gradient_checkpointing: bool = True
    """Trade compute for memory by recomputing activations during backward.
    Enabled by default to match ACE-Step's behaviour and save ~40-60%
    activation VRAM.  Adds ~10-30% training time overhead."""

    offload_encoder: bool = False
    """Move encoder/VAE to CPU after setup to free ~2-4 GB VRAM."""

    vram_profile: str = "auto"
    """VRAM preset: 'auto', 'comfortable', 'standard', 'tight', 'minimal'."""

    # --- Corrected training params ------------------------------------------
    cfg_ratio: float = 0.15
    """Classifier-free guidance dropout probability."""

    f0_loss_weight: float = 0.0
    """Weight for auxiliary F0-contour consistency loss (0 disables)."""

    speaker_loss_weight: float = 0.0
    """Weight for auxiliary speaker-consistency loss (0 disables)."""

    contrastive_identity_loss_weight: float = 0.0
    """Weight for MERT bridge InfoNCE identity loss (0 disables)."""

    contrastive_num_negatives: int = 4
    """Maximum number of queued in-dataset negatives for contrastive identity loss."""

    contrastive_temperature: float = 0.07
    """Temperature for MERT bridge InfoNCE identity loss."""

    verifier_aux_weight: float = 0.0
    """Backward-compatible weight for differentiable verifier-style auxiliary loss."""

    wavlm_aux_weight: float = 0.0
    """Weight for metric-distilled latent-to-WavLM cosine auxiliary loss."""

    ecapa_aux_weight: float = 0.0
    """Weight for metric-distilled latent-to-ECAPA cosine auxiliary loss."""

    require_identity_sidecars: bool = False
    """Fail training when enabled identity losses do not receive frozen sidecar targets."""

    identity_sidecar_dir: Optional[str] = None
    """Directory containing Phase B .identity.pt frozen target sidecars."""

    spectral_formant_aux_weight: float = 0.0
    """Weight for differentiable spectral/formant proxy auxiliary loss."""

    pitch_style_aux_weight: float = 0.0
    """Weight for differentiable pitch-style proxy auxiliary loss."""

    identity_v5_enabled: bool = False
    identity_v5_global_enabled: bool = True
    identity_v5_local_enabled: bool = True
    """Enable V5 fail-closed architecture/loss checks."""

    identity_v5_stage2_encoder_json: Optional[str] = None
    identity_v5_prototype_file: Optional[str] = None
    """Selected encoder manifest produced by Stage 2 diagnostics."""

    identity_v5_teacher_checkpoint: Optional[str] = None
    """Frozen Phase A teacher checkpoint for preservation loss."""

    identity_v5_teacher_loss_weight: float = 0.0
    """Weight for Phase A teacher preservation loss."""

    parent_preservation_loss_weight: float = 0.0
    """Matched-input output-preservation weight for the exactly resumed parent."""

    identity_v5_waveform_identity_weight: float = 0.0
    identity_v5_student_checkpoint: Optional[str] = None
    identity_v5_decoded_batch_fraction: float = 0.25
    identity_v5_epoch_eval_command: Optional[str] = None
    identity_v5_epoch_eval_timeout_sec: int = 7200
    """Weight for verified differentiable decoded identity loss."""

    identity_v5_supcon_weight: float = 0.0
    """Weight for distinct-singer supervised contrastive prototype loss."""

    use_mert_conditioning: bool = False
    """Enable optional precomputed MERT reference-voice conditioning."""

    require_complete_mert: bool = False
    """Require MERT on every sample when enabled; false in the first controlled Phase-D test."""

    preserve_mert_init_rng_without_conditioning: bool = False
    """Consume historical MERT-bridge initialization RNG without retaining MERT."""

    mert_model_name_or_path: str = "m-a-p/MERT-v1-330M"
    """Model name or local path used during preprocessing to extract MERT features."""

    mert_local_files_only: bool = True
    """Require MERT assets to exist locally instead of downloading at runtime."""

    mert_hidden_size: int = 1024
    """Hidden size of the precomputed MERT features."""

    mert_num_layers: int = 25
    """Expected number of MERT hidden-state layers in precomputed features."""

    voice_condition_dropout: float = 0.1
    """Dropout applied after projecting reference voice states."""

    voice_condition_scale: float = 1.0
    """Scale applied to projected voice states before concatenation."""

    max_ref_voice_duration: float = 3.0
    """Maximum duration for reference-voice clips during preprocessing."""

    use_multicrop_mert_conditioning: bool = False
    """Enable V3 multi-crop MERT reference conditioning."""

    top_k_reference_crops: int = 3
    """Number of reference crops extracted for V3 MERT conditioning."""

    crop_duration: float = 3.0
    """Duration in seconds for each V3 MERT reference crop."""

    timestep_mu: float = -0.4
    """Mean for logit-normal timestep sampling (from model config)."""

    timestep_sigma: float = 1.0
    """Std for logit-normal timestep sampling (from model config)."""

    data_proportion: float = 0.5
    """Data proportion for sample_t_r (from model config)."""

    # --- Adapter selection ----------------------------------------------------
    adapter_type: str = "lora"
    """Adapter type: 'lora' (PEFT) or 'lokr' (LyCORIS)."""

    # --- Model / paths ------------------------------------------------------
    model_variant: str = "turbo"
    """Model variant: 'turbo', 'base', or 'sft'."""

    checkpoint_dir: str = "./checkpoints"
    """Path to checkpoints root directory."""

    dataset_dir: str = ""
    """Directory containing preprocessed .pt tensor files."""

    # --- Device / precision -------------------------------------------------
    device: str = "auto"
    """Device selection: 'auto', 'cuda', 'cuda:0', 'mps', 'xpu', 'cpu'."""

    precision: str = "auto"
    """Precision: 'auto', 'bf16', 'fp16', 'fp32'."""

    # --- Checkpointing ------------------------------------------------------
    resume_from: Optional[str] = None
    """Path to checkpoint directory to resume training from."""

    resume_optimizer_state: bool = True
    """Restore optimizer/scheduler state when resuming; disable for a new training phase."""

    phase_d_resume_adapter: Optional[str] = None
    phase_d_scheduler_policy: Optional[str] = None
    phase_d_optimizer_policy: Optional[str] = None
    freeze_base_adapter_in_phase_d: bool = False
    """Freeze the inherited non-timing LoRA during controlled Phase D."""
    phase_d_clarity_replay_enabled: bool = False
    """Train the base LoRA at a low LR on a verified full-song/short-clip union."""
    phase_d_d1_alternating_roles_enabled: bool = False
    """Use the fail-closed D1 short, short, full optimizer-step schedule."""
    phase_d_role_manifest_path: Optional[str] = None
    """Immutable mixed-dataset provenance manifest used to assign D1 roles."""
    phase_d_require_parent_timing_state: bool = False
    """Require and exactly verify timing_branch.pt during controlled resume."""
    strict_timing_state_load: bool = False
    max_optimizer_steps: int = 0
    phase_d_probe_steps: str = ""
    phase_d_probe_dir: Optional[str] = None
    phase_d_fixed_latent_path: Optional[str] = None
    experiment_manifest_out: Optional[str] = None

    # --- Extended TensorBoard logging ---------------------------------------
    log_dir: Optional[str] = None
    """TensorBoard log directory.  Defaults to {output_dir}/runs."""

    log_every: int = 10
    """Log basic metrics (loss, LR) every N optimiser steps."""

    log_heavy_every: int = 50
    """Log per-layer gradient norms every N optimiser steps."""

    # --- Sample generation --------------------------------------------------
    sample_every_n_epochs: int = 0
    """Generate an audio sample every N epochs (0 = disabled)."""

    # --- Estimation params --------------------------------------------------
    estimate_batches: Optional[int] = None
    """Number of batches for gradient estimation (None = auto from GPU)."""

    top_k: int = 16
    """Number of top modules to select during estimation."""

    granularity: str = "module"
    """Estimation granularity: 'layer' or 'module'."""

    module_config: Optional[str] = None
    """Path to JSON module config produced by the estimate subcommand."""

    auto_estimate: bool = False
    """Run estimation inline before training."""

    estimate_output: Optional[str] = None
    """Path to write module config JSON (estimate subcommand only)."""

    # --- Preprocessing flags ------------------------------------------------
    preprocess: bool = False
    """Run preprocessing before training."""

    audio_dir: Optional[str] = None
    """Source audio directory for preprocessing."""

    dataset_json: Optional[str] = None
    """Labeled dataset JSON for preprocessing."""

    tensor_output: Optional[str] = None
    """Output directory for preprocessed .pt tensor files."""

    max_duration: float = 240.0
    """Maximum audio duration in seconds (preprocessing)."""

    # --- Phase D: Timing / prosody branch -----------------------------------
    enable_timing_branch: bool = False
    """Enable the beat-aware word-aligned timing conditioning branch (Phase D)."""

    timing_init_profile: str = "safe"
    """Fresh timing-branch initialization profile: safe or historical_v4."""

    timing_max_seq_len: int = 0
    """Timing encoder position-table length. 0 preserves profile defaults."""

    timing_gate_init_logit: float = -4.0
    """Initial decoder timing-gate logit. Historical Phase D used -4."""

    timing_dir: Optional[str] = None
    """Directory containing .timing.pt sidecars. Controlled Phase-D requires every sidecar."""

    strict_sidecars: bool = False

    timing_hidden_size: int = 256
    """Internal hidden size of the TimingEncoder transformer."""

    timing_num_heads: int = 4
    """Number of attention heads in the TimingEncoder."""

    timing_num_layers: int = 2
    """Number of transformer layers in the TimingEncoder."""

    timing_output_dim: int = 0
    """Projection dimension from TimingEncoder to DiT conditioning space.
    0 = auto (uses model hidden_size)."""

    timing_dropout: float = 0.1
    """Dropout inside TimingEncoder."""

    timing_loss_weight: float = 1.0
    """Global weight applied to the auxiliary timing supervision loss."""

    timing_dur_weight: float = 1.0
    """Weight for the duration prediction sub-loss."""

    timing_onset_weight: float = 0.5
    """Weight for the onset-deviation prediction sub-loss."""

    timing_pause_weight: float = 0.5
    """Weight for the pause prediction sub-loss."""

    timing_phrase_weight: float = 0.3
    """Weight for phrase-boundary / phrase-break supervision."""

    timing_tempo_weight: float = 0.2
    """Weight for local tempo / rubato supervision."""

    timing_terminal_weight: float = 0.2
    """Weight for phrase-final terminal shaping supervision."""

    timing_condition_dropout: float = 0.1
    """Probability of dropping timing conditioning during training (for robustness)."""

    timing_condition_scale: float = 1.0
    """Global scale applied to the projected timing stream before decoder timing attention."""

    timing_use_stream_type_embedding: bool = True
    """Add a learned stream-type embedding to mark timing states inside the decoder context."""

    timing_decoder_loss_weight: float = 1.0
    """Weight for decoder-coupled timing supervision on pooled output trajectories."""

    enable_timing_consumer_training: bool = True
    """Allow the decoder timing consumer weights, norms, and gates to train (Phase E0)."""

    enable_phrase_modulation: bool = False
    """Enable phrase/global timing-state modulation on top of event timing cross-attention (Phase E1)."""

    enable_absolute_time_conditioning: bool = False
    """Encode normalized absolute event start/end positions in the timing stream."""

    timing_attention_prior: str = "global"
    """Timing attention topology: global or local_monotonic."""

    timing_local_sigma_sec: float = 1.0
    """Gaussian distance scale for local monotonic timing attention."""

    timing_local_window_sec: float = 3.0
    """Hard event window around each audio query for local timing attention."""

    timing_event_flow_weight: float = 0.0
    """Additional flow-loss weight on frames covered by aligned lyric events."""

    timing_event_flow_margin_sec: float = 0.08
    """Seconds of context added around each event for event-weighted flow loss."""

    timing_counterfactual_weight: float = 0.0
    """Weight for correct-vs-shifted timing reconstruction ranking."""

    timing_counterfactual_shift_sec: float = 8.0
    """Absolute shift used to construct the counterfactual timing mask."""

    timing_counterfactual_margin: float = 0.002
    """Required per-sample flow-MSE advantage over shifted timing."""

    timing_global_condition_scale: float = 1.0
    """Scale applied to the phrase/global timing modulation vector before decoder use."""

    timing_global_bottleneck_dim: int = 8
    """Bottleneck width for phrase/global decoder modulation (smaller is safer on tiny datasets)."""

    timing_train_last_n_layers: int = 8
    """Train timing-consumer parameters only in the last N decoder layers (0 = all timing layers)."""

    timing_consumer_adapter_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "o_proj"]
    )
    """Timing cross-attention adapter modules to train for low-data Phase E runs."""

    timing_encoder_learning_rate: float = 0.0
    """Dedicated timing encoder/supervisor LR; 0 inherits the base LR."""

    timing_gate_learning_rate: float = 0.0
    """Dedicated FP32 decoder timing-gate LR; 0 inherits the base LR."""

    timing_consumer_learning_rate: float = 0.0
    """Dedicated timing cross-attention consumer LR; 0 inherits the base LR."""

    timing_telemetry_every: int = 0
    """Write timing health telemetry every N optimizer steps; 0 disables."""

    timing_hazard_patience: int = 5
    """Active timing steps tolerated before a stalled gradient/gate becomes a hazard."""

    timing_fail_on_hazard: bool = False
    """Abort immediately when timing health monitoring detects a hard hazard."""

    enable_timing_predictor: bool = False
    """Enable the Phase E2 timing predictor foundation."""

    timing_condition_source: str = "sidecar"
    """Timing conditioning source: sidecar, predictor, or hybrid."""

    timing_predictor_loss_weight: float = 1.0
    """Weight applied to the E2 timing-predictor supervision loss."""

    timing_hybrid_mix: float = 0.5
    """Hybrid predictor mixing ratio: 0.0 = sidecar only, 1.0 = predictor only."""

    enable_expressivity_supervision: bool = False
    """Enable Phase E3 expressivity supervision on pooled decoder states and predictor heads."""

    expressivity_loss_weight: float = 0.5
    """Global weight applied to the E3 expressivity supervision loss."""

    expressivity_f0_weight: float = 0.4
    """Weight for phrase-aware F0 / landing supervision."""

    expressivity_energy_weight: float = 0.4
    """Weight for energy / release-envelope supervision."""

    expressivity_terminal_decay_weight: float = 0.5
    """Weight for phrase-final terminal decay supervision."""

    expressivity_cv_ratio_weight: float = 0.3
    """Weight for consonant-vowel redistribution supervision."""

    expressivity_release_weight: float = 0.2
    """Weight for release-class supervision."""

    validate_every_n_epochs: int = 1
    """Run validation every N epochs when val_split > 0."""

    early_stopping_patience: int = 0
    """Stop after this many non-improving validation windows (0 disables early stopping)."""

    early_stopping_min_delta: float = 0.0
    """Minimum validation-loss improvement required to reset early stopping patience."""

    save_best_checkpoint: bool = True
    """Save a rolling best checkpoint when validation improves."""

    skip_nonfinite_gradients: bool = False
    """Replace non-finite gradients with zeros before clipping (low-data safety valve)."""

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @property
    def effective_log_dir(self) -> Path:
        """Return the resolved TensorBoard log directory."""
        if self.log_dir is not None:
            return Path(self.log_dir)
        return Path(self.output_dir) / "runs"

    def to_dict(self) -> dict:
        """Serialize every field, including new ones."""
        base = super().to_dict()
        base.update(
            {
                "num_workers": self.num_workers,
                "pin_memory": self.pin_memory,
                "prefetch_factor": self.prefetch_factor,
                "persistent_workers": self.persistent_workers,
                "pin_memory_device": self.pin_memory_device,
                "optimizer_type": self.optimizer_type,
                "scheduler_type": self.scheduler_type,
                "gradient_checkpointing": self.gradient_checkpointing,
                "offload_encoder": self.offload_encoder,
                "vram_profile": self.vram_profile,
                "adapter_type": self.adapter_type,
                "cfg_ratio": self.cfg_ratio,
                "contrastive_identity_loss_weight": self.contrastive_identity_loss_weight,
                "contrastive_num_negatives": self.contrastive_num_negatives,
                "contrastive_temperature": self.contrastive_temperature,
                "verifier_aux_weight": self.verifier_aux_weight,
                "wavlm_aux_weight": self.wavlm_aux_weight,
                "ecapa_aux_weight": self.ecapa_aux_weight,
                "require_identity_sidecars": self.require_identity_sidecars,
                "identity_sidecar_dir": self.identity_sidecar_dir,
                "spectral_formant_aux_weight": self.spectral_formant_aux_weight,
                "pitch_style_aux_weight": self.pitch_style_aux_weight,
                "identity_v5_enabled": self.identity_v5_enabled,
                "identity_v5_global_enabled": self.identity_v5_global_enabled,
                "identity_v5_local_enabled": self.identity_v5_local_enabled,
                "identity_v5_stage2_encoder_json": self.identity_v5_stage2_encoder_json,
                "identity_v5_prototype_file": self.identity_v5_prototype_file,
                "identity_v5_teacher_checkpoint": self.identity_v5_teacher_checkpoint,
                "identity_v5_teacher_loss_weight": self.identity_v5_teacher_loss_weight,
                "parent_preservation_loss_weight": self.parent_preservation_loss_weight,
                "identity_v5_waveform_identity_weight": self.identity_v5_waveform_identity_weight,
                "identity_v5_student_checkpoint": self.identity_v5_student_checkpoint,
                "identity_v5_decoded_batch_fraction": self.identity_v5_decoded_batch_fraction,
                "identity_v5_epoch_eval_command": self.identity_v5_epoch_eval_command,
                "identity_v5_epoch_eval_timeout_sec": self.identity_v5_epoch_eval_timeout_sec,
                "identity_v5_supcon_weight": self.identity_v5_supcon_weight,
                "use_mert_conditioning": self.use_mert_conditioning,
                "require_complete_mert": self.require_complete_mert,
                "preserve_mert_init_rng_without_conditioning": self.preserve_mert_init_rng_without_conditioning,
                "mert_model_name_or_path": self.mert_model_name_or_path,
                "mert_local_files_only": self.mert_local_files_only,
                "mert_hidden_size": self.mert_hidden_size,
                "mert_num_layers": self.mert_num_layers,
                "voice_condition_dropout": self.voice_condition_dropout,
                "voice_condition_scale": self.voice_condition_scale,
                "max_ref_voice_duration": self.max_ref_voice_duration,
                "use_multicrop_mert_conditioning": self.use_multicrop_mert_conditioning,
                "top_k_reference_crops": self.top_k_reference_crops,
                "crop_duration": self.crop_duration,
                "timestep_mu": self.timestep_mu,
                "timestep_sigma": self.timestep_sigma,
                "data_proportion": self.data_proportion,
                "model_variant": self.model_variant,
                "checkpoint_dir": self.checkpoint_dir,
                "dataset_dir": self.dataset_dir,
                "device": self.device,
                "precision": self.precision,
                "resume_from": self.resume_from,
                "resume_optimizer_state": self.resume_optimizer_state,
                "phase_d_resume_adapter": self.phase_d_resume_adapter,
                "phase_d_scheduler_policy": self.phase_d_scheduler_policy,
                "phase_d_optimizer_policy": self.phase_d_optimizer_policy,
                "freeze_base_adapter_in_phase_d": self.freeze_base_adapter_in_phase_d,
                "phase_d_clarity_replay_enabled": self.phase_d_clarity_replay_enabled,
                "phase_d_d1_alternating_roles_enabled": self.phase_d_d1_alternating_roles_enabled,
                "phase_d_role_manifest_path": self.phase_d_role_manifest_path,
                "phase_d_require_parent_timing_state": self.phase_d_require_parent_timing_state,
                "strict_timing_state_load": self.strict_timing_state_load,
                "max_optimizer_steps": self.max_optimizer_steps,
                "phase_d_probe_steps": self.phase_d_probe_steps,
                "phase_d_probe_dir": self.phase_d_probe_dir,
                "phase_d_fixed_latent_path": self.phase_d_fixed_latent_path,
                "experiment_manifest_out": self.experiment_manifest_out,
                "log_dir": self.log_dir,
                "log_every": self.log_every,
                "log_heavy_every": self.log_heavy_every,
                "sample_every_n_epochs": self.sample_every_n_epochs,
                "estimate_batches": self.estimate_batches,
                "top_k": self.top_k,
                "granularity": self.granularity,
                "module_config": self.module_config,
                "auto_estimate": self.auto_estimate,
                "estimate_output": self.estimate_output,
                "preprocess": self.preprocess,
                "audio_dir": self.audio_dir,
                "dataset_json": self.dataset_json,
                "tensor_output": self.tensor_output,
                "max_duration": self.max_duration,
                # Phase D: timing branch
                "enable_timing_branch": self.enable_timing_branch,
                "timing_init_profile": self.timing_init_profile,
                "timing_max_seq_len": self.timing_max_seq_len,
                "timing_gate_init_logit": self.timing_gate_init_logit,
                "timing_dir": self.timing_dir,
                "strict_sidecars": self.strict_sidecars,
                "timing_hidden_size": self.timing_hidden_size,
                "timing_num_heads": self.timing_num_heads,
                "timing_num_layers": self.timing_num_layers,
                "timing_output_dim": self.timing_output_dim,
                "timing_dropout": self.timing_dropout,
                "timing_loss_weight": self.timing_loss_weight,
                "timing_dur_weight": self.timing_dur_weight,
                "timing_onset_weight": self.timing_onset_weight,
                "timing_pause_weight": self.timing_pause_weight,
                "timing_phrase_weight": self.timing_phrase_weight,
                "timing_tempo_weight": self.timing_tempo_weight,
                "timing_terminal_weight": self.timing_terminal_weight,
                "timing_condition_dropout": self.timing_condition_dropout,
                "timing_condition_scale": self.timing_condition_scale,
                "timing_use_stream_type_embedding": self.timing_use_stream_type_embedding,
                "timing_decoder_loss_weight": self.timing_decoder_loss_weight,
                "enable_timing_consumer_training": self.enable_timing_consumer_training,
                "enable_phrase_modulation": self.enable_phrase_modulation,
                "enable_absolute_time_conditioning": self.enable_absolute_time_conditioning,
                "timing_attention_prior": self.timing_attention_prior,
                "timing_local_sigma_sec": self.timing_local_sigma_sec,
                "timing_local_window_sec": self.timing_local_window_sec,
                "timing_event_flow_weight": self.timing_event_flow_weight,
                "timing_event_flow_margin_sec": self.timing_event_flow_margin_sec,
                "timing_counterfactual_weight": self.timing_counterfactual_weight,
                "timing_counterfactual_shift_sec": self.timing_counterfactual_shift_sec,
                "timing_counterfactual_margin": self.timing_counterfactual_margin,
                "timing_global_condition_scale": self.timing_global_condition_scale,
                "timing_global_bottleneck_dim": self.timing_global_bottleneck_dim,
                "timing_train_last_n_layers": self.timing_train_last_n_layers,
                "timing_consumer_adapter_modules": self.timing_consumer_adapter_modules,
                "timing_encoder_learning_rate": self.timing_encoder_learning_rate,
                "timing_gate_learning_rate": self.timing_gate_learning_rate,
                "timing_consumer_learning_rate": self.timing_consumer_learning_rate,
                "timing_telemetry_every": self.timing_telemetry_every,
                "timing_hazard_patience": self.timing_hazard_patience,
                "timing_fail_on_hazard": self.timing_fail_on_hazard,
                "enable_timing_predictor": self.enable_timing_predictor,
                "timing_condition_source": self.timing_condition_source,
                "timing_predictor_loss_weight": self.timing_predictor_loss_weight,
                "timing_hybrid_mix": self.timing_hybrid_mix,
                "enable_expressivity_supervision": self.enable_expressivity_supervision,
                "expressivity_loss_weight": self.expressivity_loss_weight,
                "expressivity_f0_weight": self.expressivity_f0_weight,
                "expressivity_energy_weight": self.expressivity_energy_weight,
                "expressivity_terminal_decay_weight": self.expressivity_terminal_decay_weight,
                "expressivity_cv_ratio_weight": self.expressivity_cv_ratio_weight,
                "expressivity_release_weight": self.expressivity_release_weight,
                "validate_every_n_epochs": self.validate_every_n_epochs,
                "early_stopping_patience": self.early_stopping_patience,
                "early_stopping_min_delta": self.early_stopping_min_delta,
                "save_best_checkpoint": self.save_best_checkpoint,
                "skip_nonfinite_gradients": self.skip_nonfinite_gradients,
            }
        )
        return base

    def save_json(self, path: Path) -> None:
        """Persist the full config to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "TrainingConfigV2":
        """Load config from a JSON file."""
        data = json.loads(Path(path).read_text())
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
