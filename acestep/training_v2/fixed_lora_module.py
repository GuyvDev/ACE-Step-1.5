"""
FixedLoRAModule -- Corrected adapter training step for ACE-Step V2.

This module contains the ``FixedLoRAModule`` (nn.Module) responsible for
the per-step training logic: CFG dropout, logit-normal timestep sampling,
flow-matching interpolation, and the decoder forward pass.

Also includes small device/dtype/precision helpers used by both the
Fabric and basic training loops.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ACE-Step utilities
from acestep.training.lora_injection import inject_lora_into_dit
from acestep.training.lora_utils import check_peft_available
from acestep.training.lokr_utils import (
    check_lycoris_available,
    inject_lokr_into_dit,
)

# V2 modules
from acestep.training_v2.configs import LoRAConfigV2, LoKRConfigV2, TrainingConfigV2
from acestep.training_v2.mert_conditioning import PrecomputedMERTConditioner
from acestep.training_v2.identity_v5_losses import DecodedIdentityLoss, MelSingerStudent, OobleckDecodeAdapter, distinct_singer_supcon
from acestep.training_v2.model_loader import load_vae
from acestep.training_v2.performance_timing_predictor import (
    PerformanceTimingPredictor,
    PerformanceTimingPredictorConfig,
)
from acestep.training_v2.release_targets import DecoderExpressivitySupervisor
from acestep.training_v2.singer_identity_conditioning import GatedFragmentAttention, GatedSingerAdaLN
from acestep.training_v2.timestep_sampling import apply_cfg_dropout, sample_timesteps
from acestep.training_v2.timing_conditioning import (
    build_event_frame_weights,
    build_local_timing_attention_mask,
    DecoderTimingSupervisor,
    TimingEncoder,
    TimingEncoderConfig,
)
from acestep.training_v2.ui import TrainingUpdate

# Union type for adapter configs
AdapterConfig = Union[LoRAConfigV2, LoKRConfigV2]


class _LastLossAccessor:
    """Lightweight wrapper that provides ``[-1]`` and bool access.

    Avoids storing an unbounded list of floats while keeping backward
    compatibility with code that reads ``module.training_losses[-1]``
    or checks ``if module.training_losses:``.
    """

    def __init__(self, module: "FixedLoRAModule") -> None:
        self._module = module
        self._has_value = False

    def append(self, value: float) -> None:
        self._module.last_training_loss = value
        self._has_value = True

    def __getitem__(self, idx: int) -> float:
        if idx == -1 or idx == 0:
            return self._module.last_training_loss
        raise IndexError("only index -1 or 0 is supported")

    def __bool__(self) -> bool:
        return self._has_value

    def __len__(self) -> int:
        return 1 if self._has_value else 0


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_device_type(device: Any) -> str:
    if isinstance(device, torch.device):
        return device.type
    if isinstance(device, str):
        return device.split(":", 1)[0]
    return str(device)


def _validate_timing_source_contract(
    *,
    controlled_phase_d: bool,
    timing_condition_source: str,
    enable_timing_predictor: bool,
) -> str:
    """Normalize and fail closed on timing-source selection.

    The actual-new-chain controlled run is intentionally sidecar-only. Predictor
    and hybrid modes are different experiments and must never be selected through
    fallback, defaults, or a partially mutated configuration.
    """
    source = str(timing_condition_source).strip().lower()
    if source not in {"sidecar", "predictor", "hybrid"}:
        if controlled_phase_d:
            raise ValueError(
                f"invalid controlled timing_condition_source: {source!r}; expected 'sidecar'"
            )
        raise ValueError(f"invalid timing_condition_source: {source}")
    if controlled_phase_d:
        if source != "sidecar":
            raise ValueError(
                f"invalid controlled timing_condition_source: {source!r}; expected 'sidecar'"
            )
        if enable_timing_predictor:
            raise RuntimeError(
                "controlled run requested timing_condition_source='sidecar' "
                "but enable_timing_predictor=True"
            )
    elif not enable_timing_predictor and source != "sidecar":
        raise RuntimeError(
            f"timing_condition_source={source} requires enable_timing_predictor=True"
        )
    return source


def _select_compute_dtype(device_type: str) -> torch.dtype:
    if device_type in ("cuda", "xpu"):
        return torch.bfloat16
    if device_type == "mps":
        return torch.float16
    return torch.float32


def _select_fabric_precision(device_type: str) -> str:
    if device_type in ("cuda", "xpu"):
        return "bf16-mixed"
    if device_type == "mps":
        return "16-mixed"
    return "32-true"


def _latent_f0_proxy_contour(latents: torch.Tensor) -> torch.Tensor:
    """Compute a differentiable F0-proxy contour from latent trajectories.

    This is a lightweight proxy built from frame-wise spectral centroids on
    channel-averaged latent signals. It does not estimate absolute acoustic F0,
    but provides a stable contour target for pitch-aware regularization.
    """
    if latents.ndim != 3:
        raise ValueError(f"Expected latents shape [B, C, T], got {tuple(latents.shape)}")

    # [B, T]
    signal = latents.mean(dim=1)
    bsz, tlen = signal.shape

    # Keep framing safe for short sequences.
    frame = min(64, max(8, tlen))
    hop = max(1, frame // 4)
    if tlen < frame:
        signal = F.pad(signal, (0, frame - tlen), mode="replicate")

    frames = signal.unfold(dimension=1, size=frame, step=hop)  # [B, N, frame]
    # FFT kernels used by this proxy are not implemented for bf16 on this stack,
    # so keep the auxiliary contour computation in fp32 for stability.
    aux_dtype = torch.float32
    frames = frames.to(aux_dtype)
    window = torch.hann_window(frame, device=latents.device, dtype=aux_dtype)
    frames = frames * window

    spec = torch.fft.rfft(frames, dim=-1).abs()
    power = spec.square() + 1e-8
    # Drop DC bin to reduce loudness bias.
    power = power[..., 1:]

    bins = power.shape[-1]
    freqs = torch.linspace(0.0, 1.0, bins + 1, device=latents.device, dtype=aux_dtype)[1:]
    contour = (power * freqs).sum(dim=-1) / (power.sum(dim=-1) + 1e-8)
    return contour


def _speaker_proxy_embedding(latents: torch.Tensor) -> torch.Tensor:
    """Compute a simple speaker/timbre proxy embedding from latent statistics."""
    mean = latents.mean(dim=-1)
    std = latents.std(dim=-1, unbiased=False)
    emb = torch.cat([mean, std], dim=-1)
    return F.normalize(emb, p=2, dim=-1, eps=1e-8)


def _latent_spectral_formant_proxy(latents: torch.Tensor) -> torch.Tensor:
    """Differentiable spectral-envelope proxy used as a Singer ID v2 component."""
    signal = latents.float().mean(dim=1)
    frame = min(128, max(16, signal.shape[-1]))
    if signal.shape[-1] < frame:
        signal = F.pad(signal, (0, frame - signal.shape[-1]), mode="replicate")
    frames = signal.unfold(dimension=1, size=frame, step=max(1, frame // 2))
    spec = torch.fft.rfft(frames * torch.hann_window(frame, device=latents.device), dim=-1).abs() + 1e-6
    log_spec = torch.log(spec)
    return F.normalize(log_spec.mean(dim=1), p=2, dim=-1, eps=1e-8)


def _latent_pitch_style_proxy(latents: torch.Tensor) -> torch.Tensor:
    """Differentiable pitch-style summary proxy used as a Singer ID v2 component."""
    contour = _latent_f0_proxy_contour(latents).float()
    if contour.shape[-1] < 2:
        delta = torch.zeros_like(contour)
    else:
        delta = contour[..., 1:] - contour[..., :-1]
    stats = torch.stack(
        [
            contour.mean(dim=-1),
            contour.std(dim=-1, unbiased=False),
            delta.abs().mean(dim=-1),
            contour.amax(dim=-1) - contour.amin(dim=-1),
        ],
        dim=-1,
    )
    return F.normalize(stats, p=2, dim=-1, eps=1e-8)




def _resize_embedding_dim(emb: torch.Tensor, target_dim: int) -> torch.Tensor:
    if emb.shape[-1] == target_dim:
        return emb
    resized = F.interpolate(
        emb.float().unsqueeze(1),
        size=int(target_dim),
        mode="linear",
        align_corners=False,
    ).squeeze(1)
    return F.normalize(resized, p=2, dim=-1, eps=1e-8)

def _mean_pool_masked(states: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return states.mean(dim=1)
    m = mask.to(states.dtype).unsqueeze(-1)
    return (states * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


def _info_nce_with_queue(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative_queue: torch.Tensor,
    queue_count: int,
    temperature: float,
) -> torch.Tensor:
    anchor = F.normalize(anchor.float(), p=2, dim=-1, eps=1e-8)
    positive = F.normalize(positive.float(), p=2, dim=-1, eps=1e-8)
    pos_logits = (anchor * positive).sum(dim=-1, keepdim=True)
    candidates = [pos_logits]
    labels = torch.zeros(anchor.shape[0], dtype=torch.long, device=anchor.device)
    if anchor.shape[0] > 1:
        in_batch = anchor @ positive.t()
        mask = ~torch.eye(anchor.shape[0], dtype=torch.bool, device=anchor.device)
        candidates.append(in_batch[mask].view(anchor.shape[0], -1))
    if queue_count > 0:
        q = F.normalize(negative_queue[:queue_count].to(anchor.device).float(), p=2, dim=-1, eps=1e-8)
        candidates.append(anchor @ q.t())
    logits = torch.cat(candidates, dim=1) / max(float(temperature), 1e-4)
    if logits.shape[1] <= 1:
        return anchor.new_tensor(0.0)
    return F.cross_entropy(logits, labels)


# ===========================================================================
# FixedLoRAModule -- corrected training step
# ===========================================================================


class FixedLoRAModule(nn.Module):
    """Adapter training module with corrected timestep sampling and CFG dropout.

    Supports both LoRA (PEFT) and LoKR (LyCORIS) adapters.  The training
    step is identical for both -- only the injection and weight format differ.

    Training flow (per step):
        1. Load pre-computed tensors (from ``PreprocessedDataModule``).
        2. Apply **CFG dropout** on ``encoder_hidden_states``.
        3. Sample noise ``x1`` and continuous timestep ``t`` via
           ``sample_timesteps()`` (logit-normal).
        4. Interpolate ``x_t = t * x1 + (1 - t) * x0``.
        5. Forward through decoder, compute flow matching loss.
    """

    def __init__(
        self,
        model: nn.Module,
        adapter_config: AdapterConfig,
        training_config: TrainingConfigV2,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()

        self.adapter_config = adapter_config
        self.adapter_type = training_config.adapter_type
        self.training_config = training_config
        self.device = torch.device(device) if isinstance(device, str) else device
        self.device_type = _normalize_device_type(self.device)
        self.dtype = dtype
        self.transfer_non_blocking = self.device_type in ("cuda", "xpu")
        self._controlled_phase_d = getattr(training_config, "phase_d_resume_adapter", None) is not None
        self._phase_d_d1_alternating_roles = bool(
            getattr(training_config, "phase_d_d1_alternating_roles_enabled", False)
        )
        if self._phase_d_d1_alternating_roles and not self._controlled_phase_d:
            raise RuntimeError("D1 alternating roles require controlled Phase-D mode")
        self._last_batch_role = "ordinary"

        # LyCORIS network reference (only set for LoKR)
        self.lycoris_net: Any = None
        self.adapter_info: Dict[str, Any] = {}

        # -- Adapter injection -----------------------------------------------
        if self.adapter_type == "lokr":
            self._inject_lokr(model, adapter_config)  # type: ignore[arg-type]
        else:
            self._inject_lora(model, adapter_config)  # type: ignore[arg-type]

        # Backward-compat alias
        self.lora_info = self.adapter_info

        # Model config (for timestep params read at runtime)
        self.config = model.config

        # -- Null condition embedding for CFG dropout ------------------------
        # ``model.null_condition_emb`` is a Parameter on the top-level model
        # (not the decoder).
        if hasattr(model, "null_condition_emb"):
            self._null_cond_emb = model.null_condition_emb
        else:
            self._null_cond_emb = None
            if self._controlled_phase_d and float(training_config.cfg_ratio) > 0.0:
                raise RuntimeError(
                    "controlled Phase-D requires model.null_condition_emb because CFG dropout is enabled"
                )
            logger.warning(
                "[WARN] model.null_condition_emb not found -- CFG dropout disabled"
            )

        # -- Timestep sampling params ----------------------------------------
        self._timestep_mu = training_config.timestep_mu
        self._timestep_sigma = training_config.timestep_sigma
        self._data_proportion = training_config.data_proportion
        self._cfg_ratio = training_config.cfg_ratio
        self._f0_loss_weight = max(0.0, float(getattr(training_config, "f0_loss_weight", 0.0)))
        self._speaker_loss_weight = max(0.0, float(getattr(training_config, "speaker_loss_weight", 0.0)))
        self._contrastive_identity_loss_weight = max(0.0, float(getattr(training_config, "contrastive_identity_loss_weight", 0.0)))
        self._contrastive_num_negatives = max(0, int(getattr(training_config, "contrastive_num_negatives", 4)))
        self._contrastive_temperature = max(1e-4, float(getattr(training_config, "contrastive_temperature", 0.07)))
        self._verifier_aux_weight = max(0.0, float(getattr(training_config, "verifier_aux_weight", 0.0)))
        self._wavlm_aux_weight = max(0.0, float(getattr(training_config, "wavlm_aux_weight", 0.0)))
        self._ecapa_aux_weight = max(0.0, float(getattr(training_config, "ecapa_aux_weight", 0.0)))
        self._base_contrastive_identity_loss_weight = self._contrastive_identity_loss_weight
        self._base_verifier_aux_weight = self._verifier_aux_weight
        self._base_wavlm_aux_weight = self._wavlm_aux_weight
        self._base_ecapa_aux_weight = self._ecapa_aux_weight
        self._require_identity_sidecars = bool(getattr(training_config, "require_identity_sidecars", False))
        self._spectral_formant_aux_weight = max(0.0, float(getattr(training_config, "spectral_formant_aux_weight", 0.0)))
        self._pitch_style_aux_weight = max(0.0, float(getattr(training_config, "pitch_style_aux_weight", 0.0)))
        self._base_spectral_formant_aux_weight = self._spectral_formant_aux_weight
        self._base_pitch_style_aux_weight = self._pitch_style_aux_weight
        self._voice_condition_scale = float(getattr(training_config, "voice_condition_scale", 1.0))
        self._use_mert_conditioning = bool(getattr(training_config, "use_mert_conditioning", False))
        self._identity_v5_enabled = bool(getattr(training_config, "identity_v5_enabled", False))
        self._identity_v5_global_enabled = self._identity_v5_enabled and bool(getattr(training_config, "identity_v5_global_enabled", True))
        self._identity_v5_local_enabled = self._identity_v5_enabled and bool(getattr(training_config, "identity_v5_local_enabled", True))
        self._identity_v5_teacher_loss_weight = max(0.0, float(getattr(training_config, "identity_v5_teacher_loss_weight", 0.0)))
        self._parent_preservation_loss_weight = max(
            0.0,
            float(getattr(training_config, "parent_preservation_loss_weight", 0.0)),
        )
        if (
            self._identity_v5_teacher_loss_weight > 0.0
            and self._parent_preservation_loss_weight > 0.0
        ):
            raise RuntimeError(
                "identity_v5 teacher loss and generic parent preservation cannot be enabled together"
            )
        self._identity_v5_waveform_identity_weight = max(0.0, float(getattr(training_config, "identity_v5_waveform_identity_weight", 0.0)))
        self._identity_v5_supcon_weight = max(0.0, float(getattr(training_config, "identity_v5_supcon_weight", 0.0)))
        self._identity_v5_teacher_start = self._identity_v5_teacher_loss_weight
        self._identity_v5_waveform_target = self._identity_v5_waveform_identity_weight
        self._identity_v5_supcon_target = self._identity_v5_supcon_weight
        self._identity_v5_selected_encoder: Dict[str, Any] | None = None
        self.register_buffer("identity_v5_global_prototype", torch.empty(0), persistent=True)
        self.phase_a_teacher_decoder: nn.Module | None = None
        self.v5_decoded_identity: DecodedIdentityLoss | None = None
        self._identity_v5_decoded_batch_fraction = max(0.0, min(1.0, float(getattr(training_config, "identity_v5_decoded_batch_fraction", 0.25))))
        if self._identity_v5_enabled:
            self._validate_identity_v5_contract(training_config)
            prototype_payload = torch.load(str(training_config.identity_v5_prototype_file), map_location="cpu", weights_only=False)
            self.identity_v5_global_prototype = prototype_payload["global_billy"].detach().float()
        self._last_aux_losses: Dict[str, float] = {}
        self._identity_v5_step = 0
        self._identity_v5_gradient_diagnostics: Dict[str, float] = {}

        self.voice_conditioner: PrecomputedMERTConditioner | None = None
        self.identity_proxy_projector: nn.Linear | None = None
        self.v5_singer_projection: nn.Module | None = None
        self.v5_global_adapter: nn.Module | None = None  # legacy checkpoint compatibility
        self.v5_block_adapters: nn.ModuleDict | None = None
        self.v5_fragment_attention: nn.Module | None = None
        self.wavlm_aux_head: nn.Module | None = None
        self.ecapa_aux_head: nn.Module | None = None
        self.spectral_formant_aux_head: nn.Module | None = None
        self.pitch_style_aux_head: nn.Module | None = None
        if self._identity_v5_enabled:
            hidden_dim = int(getattr(model.config, "hidden_size", 2048))
            selected = self._identity_v5_selected_encoder or {}
            backend = str(selected.get("backend", ""))
            default_dims = {"ecapa": 192, "wavlm": 512, "differentiable_student": 192, "mert_pooled": 1024}
            singer_dim = int(selected.get("embedding_dim") or default_dims.get(backend, 0))
            if singer_dim <= 0:
                raise RuntimeError(f"V5 selected encoder has no known embedding dimension: {backend!r}")
            self.v5_singer_projection = nn.Linear(singer_dim, hidden_dim).to(self.device)
            num_blocks = int(getattr(model.config, "num_hidden_layers", 24))
            block_ids = sorted({num_blocks // 2, (3 * num_blocks) // 4, num_blocks - 1})
            self.v5_block_adapters = nn.ModuleDict({
                str(i): GatedSingerAdaLN(hidden_dim, hidden_dim) for i in block_ids
            }).to(self.device)
            self.v5_fragment_attention = GatedFragmentAttention(hidden_dim, hidden_dim, crop_types=8).to(self.device)
            self._identity_v5_block_ids = block_ids
            logger.info("[OK] V5 gated identity adapters initialized with zero gates")
            if self._identity_v5_waveform_identity_weight > 0.0:
                student_path = Path(str(getattr(training_config, "identity_v5_student_checkpoint", "")))
                payload = torch.load(student_path, map_location="cpu", weights_only=False)
                embedding_dim = int(payload.get("metadata", {}).get("embedding_dim", 192))
                student = MelSingerStudent(embedding_dim=embedding_dim)
                student.load_state_dict(payload["state_dict"], strict=True)
                vae = load_vae(training_config.checkpoint_dir, device=str(self.device), precision="bf16" if self.dtype == torch.bfloat16 else "fp32")
                for frozen in (student, vae):
                    frozen.eval()
                    for parameter in frozen.parameters(): parameter.requires_grad_(False)
                self.v5_decoded_identity = DecodedIdentityLoss(OobleckDecodeAdapter(vae), student).to(self.device)
        if self._wavlm_aux_weight > 0.0:
            self.wavlm_aux_head = nn.Linear(1536, int(getattr(training_config, "wavlm_aux_dim", 512))).to(self.device)
        if self._ecapa_aux_weight > 0.0:
            self.ecapa_aux_head = nn.Linear(1536, int(getattr(training_config, "ecapa_aux_dim", 192))).to(self.device)
        if self._spectral_formant_aux_weight > 0.0:
            self.spectral_formant_aux_head = nn.Linear(1536, int(getattr(training_config, "spectral_formant_aux_dim", 23))).to(self.device)
        if self._pitch_style_aux_weight > 0.0:
            self.pitch_style_aux_head = nn.Linear(1536, int(getattr(training_config, "pitch_style_aux_dim", 7))).to(self.device)
        preserve_mert_rng = bool(
            getattr(
                training_config,
                "preserve_mert_init_rng_without_conditioning",
                False,
            )
        )
        if preserve_mert_rng and self._use_mert_conditioning:
            raise RuntimeError(
                "MERT RNG parity mode requires MERT conditioning to be disabled"
            )
        if preserve_mert_rng:
            output_dim = int(getattr(model.config, "hidden_size", 2048))
            shadow_conditioner = PrecomputedMERTConditioner(
                input_dim=int(getattr(training_config, "mert_hidden_size", 1024)),
                output_dim=output_dim,
                num_layers=int(getattr(training_config, "mert_num_layers", 25)),
                dropout=float(getattr(training_config, "voice_condition_dropout", 0.1)),
                use_layer_aggregation=True,
                max_reference_crops=1,
                use_crop_attention=False,
            )
            shadow_tensor_count = len(shadow_conditioner.state_dict())
            if shadow_tensor_count != 3:
                raise RuntimeError(
                    f"Historical MERT RNG shadow expected 3 tensors, found {shadow_tensor_count}"
                )
            del shadow_conditioner
            logger.info(
                "[OK] Consumed historical MERT bridge RNG (%d tensors); "
                "no MERT module retained",
                shadow_tensor_count,
            )

        if self._use_mert_conditioning:
            output_dim = int(getattr(model.config, "hidden_size", 2048))
            self.voice_conditioner = PrecomputedMERTConditioner(
                input_dim=int(getattr(training_config, "mert_hidden_size", 1024)),
                output_dim=output_dim,
                num_layers=int(getattr(training_config, "mert_num_layers", 25)),
                dropout=float(getattr(training_config, "voice_condition_dropout", 0.1)),
                use_layer_aggregation=True,
                max_reference_crops=max(1, int(getattr(training_config, "top_k_reference_crops", 3))),
                use_crop_attention=bool(getattr(training_config, "use_multicrop_mert_conditioning", False)),
            ).to(self.device)
            self.identity_proxy_projector = nn.Linear(1506, output_dim).to(self.device)
            queue_size = max(1, self._contrastive_num_negatives * 8)
            self.register_buffer("_identity_negative_queue", torch.zeros(queue_size, output_dim), persistent=False)
            self._identity_negative_queue_ptr = 0
            self._identity_negative_queue_count = 0
            logger.info(
                "[OK] MERT conditioning enabled: input_dim=%d, layers=%d, output_dim=%d",
                int(getattr(training_config, "mert_hidden_size", 1024)),
                int(getattr(training_config, "mert_num_layers", 25)),
                output_dim,
            )
            logger.info(
                "[OK] Improved identity losses: contrastive=%.4f, wavlm_aux=%.4f, ecapa_aux=%.4f, spectral_aux=%.4f, pitch_aux=%.4f, require_sidecars=%s",
                self._contrastive_identity_loss_weight,
                self._wavlm_aux_weight,
                self._ecapa_aux_weight,
                self._spectral_formant_aux_weight,
                self._pitch_style_aux_weight,
                self._require_identity_sidecars,
            )

        # -- Phase D: timing/prosody branch -----------------------------------
        self._use_timing_branch = bool(getattr(training_config, "enable_timing_branch", False))
        self._timing_loss_weight = float(getattr(training_config, "timing_loss_weight", 1.0))
        self._timing_dur_weight = float(getattr(training_config, "timing_dur_weight", 1.0))
        self._timing_onset_weight = float(getattr(training_config, "timing_onset_weight", 0.5))
        self._timing_pause_weight = float(getattr(training_config, "timing_pause_weight", 0.5))
        self._timing_phrase_weight = float(getattr(training_config, "timing_phrase_weight", 0.3))
        self._timing_tempo_weight = float(getattr(training_config, "timing_tempo_weight", 0.2))
        self._timing_terminal_weight = float(getattr(training_config, "timing_terminal_weight", 0.2))
        self._timing_condition_dropout = float(getattr(training_config, "timing_condition_dropout", 0.1))
        self._timing_decoder_loss_weight = float(
            getattr(training_config, "timing_decoder_loss_weight", 1.0)
        )
        self._timing_attention_prior = str(
            getattr(training_config, "timing_attention_prior", "global")
        ).strip().lower()
        if self._timing_attention_prior not in {"global", "local_monotonic"}:
            raise RuntimeError(
                f"unsupported timing attention prior: {self._timing_attention_prior!r}"
            )
        self._timing_local_sigma_sec = float(
            getattr(training_config, "timing_local_sigma_sec", 1.0)
        )
        self._timing_local_window_sec = float(
            getattr(training_config, "timing_local_window_sec", 3.0)
        )
        self._timing_event_flow_weight = float(
            getattr(training_config, "timing_event_flow_weight", 0.0)
        )
        self._timing_event_flow_margin_sec = float(
            getattr(training_config, "timing_event_flow_margin_sec", 0.08)
        )
        self._timing_counterfactual_weight = float(
            getattr(training_config, "timing_counterfactual_weight", 0.0)
        )
        self._timing_counterfactual_shift_sec = float(
            getattr(training_config, "timing_counterfactual_shift_sec", 8.0)
        )
        self._timing_counterfactual_margin = float(
            getattr(training_config, "timing_counterfactual_margin", 0.002)
        )
        self._timing_gate_init_logit = float(
            getattr(training_config, "timing_gate_init_logit", -4.0)
        )
        if self._timing_attention_prior == "local_monotonic":
            if not bool(getattr(training_config, "enable_absolute_time_conditioning", False)):
                raise RuntimeError(
                    "local monotonic timing requires absolute-time conditioning"
                )
            if self._timing_local_sigma_sec <= 0.0 or self._timing_local_window_sec <= 0.0:
                raise RuntimeError("local timing sigma/window must be positive")
            if self._timing_event_flow_weight < 0.0 or self._timing_counterfactual_weight < 0.0:
                raise RuntimeError("generator-coupled timing weights must be non-negative")
        elif self._timing_event_flow_weight > 0.0 or self._timing_counterfactual_weight > 0.0:
            raise RuntimeError(
                "event-weighted or counterfactual timing objectives require local_monotonic attention"
            )
        self._phase_d_ablation_mode = self._controlled_phase_d
        self._enable_timing_consumer_training = bool(
            getattr(training_config, "enable_timing_consumer_training", True)
        ) and not self._phase_d_ablation_mode
        self._enable_phrase_modulation = bool(
            getattr(training_config, "enable_phrase_modulation", False)
        )
        self._timing_train_last_n_layers = max(
            0, int(getattr(training_config, "timing_train_last_n_layers", 0))
        )
        self._enable_timing_predictor = bool(
            getattr(training_config, "enable_timing_predictor", False)
        )
        self._timing_condition_source = _validate_timing_source_contract(
            controlled_phase_d=self._controlled_phase_d,
            timing_condition_source=getattr(training_config, "timing_condition_source", "sidecar"),
            enable_timing_predictor=self._enable_timing_predictor,
        )
        self._timing_predictor_loss_weight = max(
            0.0, float(getattr(training_config, "timing_predictor_loss_weight", 1.0))
        )
        self._timing_hybrid_mix = float(getattr(training_config, "timing_hybrid_mix", 0.5))
        self._timing_hybrid_mix = max(0.0, min(1.0, self._timing_hybrid_mix))
        self._enable_expressivity_supervision = bool(
            getattr(training_config, "enable_expressivity_supervision", False)
        )
        self._expressivity_loss_weight = max(
            0.0, float(getattr(training_config, "expressivity_loss_weight", 0.5))
        )
        self._expressivity_f0_weight = float(getattr(training_config, "expressivity_f0_weight", 0.4))
        self._expressivity_energy_weight = float(getattr(training_config, "expressivity_energy_weight", 0.4))
        self._expressivity_terminal_decay_weight = float(
            getattr(training_config, "expressivity_terminal_decay_weight", 0.5)
        )
        self._expressivity_cv_ratio_weight = float(
            getattr(training_config, "expressivity_cv_ratio_weight", 0.3)
        )
        self._expressivity_release_weight = float(
            getattr(training_config, "expressivity_release_weight", 0.2)
        )
        raw_consumer_modules = getattr(
            training_config,
            "timing_consumer_adapter_modules",
            ["q_proj", "o_proj"],
        )
        if isinstance(raw_consumer_modules, str):
            raw_consumer_modules = [
                part.strip() for part in raw_consumer_modules.split(",") if part.strip()
            ]
        self._timing_consumer_adapter_modules = tuple(raw_consumer_modules) or (
            "q_proj",
            "o_proj",
        )

        self.timing_encoder: TimingEncoder | None = None
        self.decoder_timing_supervisor: DecoderTimingSupervisor | None = None
        self.performance_timing_predictor: PerformanceTimingPredictor | None = None
        self.decoder_expressivity_supervisor: DecoderExpressivitySupervisor | None = None
        self._frozen_base_adapter_names: list[str] = []
        if self._use_timing_branch:
            dit_dim = int(getattr(model.config, "hidden_size", 2048))
            timing_output_dim = int(getattr(training_config, "timing_output_dim", 0)) or dit_dim
            timing_init_profile = str(
                getattr(training_config, "timing_init_profile", "safe")
            ).strip().lower()
            historical_v4 = timing_init_profile == "historical_v4"
            timing_cfg = TimingEncoderConfig(
                hidden_size=int(getattr(training_config, "timing_hidden_size", 256)),
                num_heads=int(getattr(training_config, "timing_num_heads", 4)),
                num_layers=int(getattr(training_config, "timing_num_layers", 2)),
                output_dim=timing_output_dim,
                dropout=float(getattr(training_config, "timing_dropout", 0.1)),
                condition_scale=float(getattr(training_config, "timing_condition_scale", 1.0)),
                use_stream_type_embedding=bool(
                    getattr(training_config, "timing_use_stream_type_embedding", True)
                ),
                max_seq_len=(
                    int(getattr(training_config, "timing_max_seq_len", 0))
                    or (1024 if historical_v4 else 2048)
                ),
                enable_phrase_features=not historical_v4,
                enable_phrase_modulation=self._enable_phrase_modulation,
                enable_absolute_time_conditioning=bool(
                    getattr(training_config, "enable_absolute_time_conditioning", False)
                ),
                global_condition_scale=float(
                    getattr(training_config, "timing_global_condition_scale", 1.0)
                ),
                init_profile=timing_init_profile,
            )
            self.timing_encoder = TimingEncoder(timing_cfg).to(self.device)
            decoder_state_dim = int(getattr(model.config, "audio_acoustic_hidden_dim", 64))
            self.decoder_timing_supervisor = DecoderTimingSupervisor(
                input_dim=decoder_state_dim,
                hidden_size=timing_cfg.hidden_size,
                dropout=float(getattr(training_config, "timing_dropout", 0.1)),
            ).to(self.device)
            if self._timing_loss_weight <= 0.0:
                self.timing_encoder.prediction_head.requires_grad_(False)
            if self._timing_decoder_loss_weight <= 0.0:
                self.decoder_timing_supervisor.requires_grad_(False)
            logger.info(
                "[OK] Timing branch enabled: hidden=%d, heads=%d, layers=%d, output_dim=%d, decoder_supervision=%d, init=%s, max_seq_len=%d",
                timing_cfg.hidden_size,
                timing_cfg.num_heads,
                timing_cfg.num_layers,
                timing_output_dim,
                decoder_state_dim,
                timing_cfg.init_profile,
                timing_cfg.max_seq_len,
            )
            if self._enable_timing_predictor:
                predictor_cfg = PerformanceTimingPredictorConfig(
                    hidden_size=timing_cfg.hidden_size,
                    num_heads=timing_cfg.num_heads,
                    num_layers=timing_cfg.num_layers,
                    output_dim=timing_output_dim,
                    dropout=timing_cfg.dropout,
                    condition_scale=float(getattr(training_config, "timing_condition_scale", 1.0)),
                    enable_global_condition=self._enable_phrase_modulation,
                )
                self.performance_timing_predictor = PerformanceTimingPredictor(predictor_cfg).to(self.device)
                logger.info(
                    "[OK] Timing predictor enabled: source=%s, loss_weight=%.3f, hybrid_mix=%.3f",
                    self._timing_condition_source,
                    self._timing_predictor_loss_weight,
                    self._timing_hybrid_mix,
                )
            if self._enable_expressivity_supervision:
                self.decoder_expressivity_supervisor = DecoderExpressivitySupervisor(
                    input_dim=decoder_state_dim,
                    hidden_size=timing_cfg.hidden_size,
                    dropout=float(getattr(training_config, "timing_dropout", 0.1)),
                ).to(self.device)
                logger.info(
                    "[OK] Expressivity supervision enabled: loss_weight=%.3f",
                    self._expressivity_loss_weight,
                )
            bootstrapped_layers = self._bootstrap_decoder_timing_parameters()
            logger.info(
                "[OK] Decoder timing-attention bootstrapped from text cross-attn for %d layers",
                bootstrapped_layers,
            )
            decoder_timing_params = self._enable_decoder_timing_parameters()
            if self._phase_d_ablation_mode:
                self._assert_controlled_timing_contract(
                    bootstrapped_layers=bootstrapped_layers,
                    decoder_timing_params=decoder_timing_params,
                )
            self._install_timing_residual_hooks()
            logger.info(
                "[OK] Decoder timing-attention params unfrozen: %s",
                f"{decoder_timing_params:,}",
            )
            if self._controlled_phase_d and bool(
                getattr(training_config, "freeze_base_adapter_in_phase_d", False)
            ):
                decoder = self.model.decoder
                while hasattr(decoder, "_forward_module"):
                    decoder = decoder._forward_module
                frozen = []
                for name, parameter in decoder.named_parameters():
                    is_base_adapter = (
                        ("lora_" in name or "lokr_" in name or "hada_" in name)
                        and "timing_cross_attn" not in name
                    )
                    if is_base_adapter:
                        parameter.requires_grad_(False)
                        frozen.append(name)
                if not frozen:
                    raise RuntimeError("requested Phase-D base-adapter freeze matched zero tensors")
                frozen_set = set(frozen)
                still_trainable = [
                    name for name, parameter in decoder.named_parameters()
                    if name in frozen_set and parameter.requires_grad
                ]
                if still_trainable:
                    raise RuntimeError(f"base-adapter freeze failed: {still_trainable[:8]}")
                self._frozen_base_adapter_names = sorted(frozen)
                logger.info("[OK] Frozen inherited base adapter for Phase D: %d tensors", len(frozen))

        # When gradient checkpointing is enabled via wrapper layers that don't
        # expose enable_input_require_grads(), force at least one forward input
        # to require grad so checkpointed segments keep a valid autograd graph.
        self.force_input_grads_for_checkpointing: bool = False

        # Book-keeping -- store only the most recent loss to avoid
        # unbounded memory growth over long training runs.
        self.last_training_loss: float = 0.0

        # Backward-compat: property provides list-like [-1] access
        # for callers that read ``training_losses[-1]``.
        self.training_losses = _LastLossAccessor(self)


    def _record_v5_gradient_conflicts(
        self,
        losses: Dict[str, torch.Tensor],
        backward_fn,
    ) -> None:
        """Measure component gradients through Fabric without contaminating training."""
        params = [p for name, p in self.named_parameters() if p.requires_grad and "v5_" not in name][:64]
        if not params:
            return
        gradients = {}
        self.zero_grad(set_to_none=True)
        for name, loss in losses.items():
            if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
                continue
            backward_fn(loss, retain_graph=True)
            gradients[name] = [
                p.grad.detach().float().clone() if p.grad is not None else None
                for p in params
            ]
            self.zero_grad(set_to_none=True)
        report = {}
        for name, values in gradients.items():
            report[f"grad_norm/{name}"] = math.sqrt(sum(float(g.pow(2).sum()) for g in values if g is not None))
        names = sorted(gradients)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                dot = sum(float(a.mul(b).sum()) for a, b in zip(gradients[left], gradients[right]) if a is not None and b is not None)
                denom = report[f"grad_norm/{left}"] * report[f"grad_norm/{right}"]
                report[f"grad_cos/{left}:{right}"] = dot / denom if denom > 0 else 0.0
        self._identity_v5_gradient_diagnostics = report

    def v5_optimizer_groups(self) -> list[dict[str, Any]]:
        """Return disjoint V5 optimizer groups with plan-specified learning rates."""
        groups = {"lora": [], "singer": [], "fragment": []}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad or name.startswith("phase_a_teacher_decoder"):
                continue
            if "v5_fragment_attention" in name:
                groups["fragment"].append(parameter)
            elif "v5_singer_projection" in name or "v5_block_adapters" in name:
                groups["singer"].append(parameter)
            else:
                groups["lora"].append(parameter)
        rates = {"lora": 4e-6, "singer": 3e-5, "fragment": 2e-5}
        return [{"params": values, "lr": rates[key], "group_name": key} for key, values in groups.items() if values]

    def apply_v5_freeze_schedule(self, epoch_number: int) -> None:
        if not self._identity_v5_enabled:
            return
        local_epoch = max(1, int(epoch_number))
        freeze_lora = local_epoch <= 2
        progress = min(1.0, max(0.0, (local_epoch - 1) / 7.0))
        teacher_floor = min(self._identity_v5_teacher_start, 0.03)
        self._identity_v5_teacher_loss_weight = self._identity_v5_teacher_start + progress * (teacher_floor - self._identity_v5_teacher_start)
        identity_warmup = min(1.0, max(0.0, (local_epoch - 1) / 2.0))
        self._identity_v5_waveform_identity_weight = self._identity_v5_waveform_target * identity_warmup
        self._identity_v5_supcon_weight = self._identity_v5_supcon_target * identity_warmup
        for name, parameter in self.named_parameters():
            if name.startswith("phase_a_teacher_decoder"):
                parameter.requires_grad_(False)
            elif "v5_" not in name and ("lora_" in name or "lycoris" in name.lower()):
                parameter.requires_grad_(not freeze_lora)

    def initialize_v5_teacher(self) -> None:
        """Snapshot the exactly resumed decoder for matched-input preservation."""
        use_v5_teacher = (
            self._identity_v5_enabled
            and self._identity_v5_teacher_loss_weight > 0.0
        )
        if not use_v5_teacher and self._parent_preservation_loss_weight <= 0.0:
            return
        if self.phase_a_teacher_decoder is not None:
            return
        self.phase_a_teacher_decoder = copy.deepcopy(self.model.decoder).eval()
        for parameter in self.phase_a_teacher_decoder.parameters():
            parameter.requires_grad_(False)
        self.phase_a_teacher_decoder.to(device=self.device, dtype=self.dtype)
        logger.info("[OK] Frozen parent teacher snapshotted after exact checkpoint resume")

    def _validate_identity_v5_contract(self, training_config: TrainingConfigV2) -> None:
        """Fail closed when a requested V5 training arm is not actually wired."""
        if self._f0_loss_weight > 0.0 or self._speaker_loss_weight > 0.0 or self._spectral_formant_aux_weight > 0.0 or self._pitch_style_aux_weight > 0.0:
            raise RuntimeError("V5 primary arms must not enable old latent F0/speaker/spectral/pitch proxy losses")
        prototype_path = getattr(training_config, "identity_v5_prototype_file", None)
        if not prototype_path or not Path(prototype_path).is_file():
            raise RuntimeError("V5 requires --identity-v5-prototype-file from frozen Stage 1")
        enc_path = getattr(training_config, "identity_v5_stage2_encoder_json", None)
        if not enc_path:
            raise RuntimeError("V5 training requires --identity-v5-stage2-encoder-json from Stage 2 diagnostics")
        payload = json.loads(Path(enc_path).read_text(encoding="utf-8"))
        if payload.get("status") != "healthy" or not payload.get("selected_encoder"):
            raise RuntimeError("V5 Stage 2 encoder manifest is not healthy")
        self._identity_v5_selected_encoder = payload["selected_encoder"]
        teacher_weight = max(0.0, float(getattr(training_config, "identity_v5_teacher_loss_weight", 0.0)))
        teacher_ckpt = getattr(training_config, "identity_v5_teacher_checkpoint", None)
        if teacher_weight > 0.0 and (not teacher_ckpt or not Path(teacher_ckpt).exists()):
            raise RuntimeError("V5 teacher loss requires an existing --identity-v5-teacher-checkpoint")
        if teacher_weight > 0.0:
            resume_ckpt = getattr(training_config, "resume_from", None)
            if not resume_ckpt or Path(teacher_ckpt).resolve() != Path(resume_ckpt).resolve():
                raise RuntimeError("V5 teacher checkpoint must exactly match the Phase A resume checkpoint")
        waveform_weight = max(0.0, float(getattr(training_config, "identity_v5_waveform_identity_weight", 0.0)))
        if waveform_weight > 0.0:
            selected = str(payload["selected_encoder"].get("backend", ""))
            if selected not in {"differentiable_student"}:
                raise RuntimeError("V5 waveform identity loss requires a verified differentiable singer student; verifier backends alone are not gradient-safe")
        supcon_weight = max(0.0, float(getattr(training_config, "identity_v5_supcon_weight", 0.0)))
        if supcon_weight > 0.0 and self._contrastive_num_negatives < 4:
            raise RuntimeError("V5 supervised contrastive loss requires at least four distinct negative singers")
        if supcon_weight > 0.0 and waveform_weight <= 0.0:
            raise RuntimeError("V5 supervised contrastive loss requires the decoded differentiable singer anchor")

    # -----------------------------------------------------------------------
    # Adapter injection helpers
    # -----------------------------------------------------------------------

    def _inject_lora(self, model: nn.Module, cfg: LoRAConfigV2) -> None:
        """Inject LoRA adapters via PEFT.

        Raises:
            RuntimeError: If PEFT is not installed.
        """
        if not check_peft_available():
            raise RuntimeError(
                "PEFT is required for LoRA training but is not installed.\n"
                "Install it with:  uv pip install peft"
            )
        self.model, self.adapter_info = inject_lora_into_dit(model, cfg)
        logger.info(
            "[OK] LoRA injected: %s trainable params",
            f"{self.adapter_info['trainable_params']:,}",
        )

    def _inject_lokr(self, model: nn.Module, cfg: LoKRConfigV2) -> None:
        """Inject LoKR adapters via LyCORIS.

        After injection, explicitly moves the model to the target device
        so that newly created LoKR parameters (which LyCORIS creates on
        CPU) end up on GPU before Fabric wraps the model.

        Raises:
            RuntimeError: If LyCORIS is not installed.
        """
        if not check_lycoris_available():
            raise RuntimeError(
                "LyCORIS is required for LoKR training but is not installed.\n"
                "Install it with:  uv pip install lycoris-lora"
            )
        self.model, self.lycoris_net, self.adapter_info = inject_lokr_into_dit(
            model,
            cfg,
        )
        # LyCORIS creates adapter parameters on CPU.  Move the entire
        # model to the target device so all parameters (including the
        # new LoKR ones) are co-located before Fabric setup.
        self.model = self.model.to(self.device)
        logger.info(
            "[OK] LoKR injected: %s trainable params (moved to %s)",
            f"{self.adapter_info['trainable_params']:,}",
            self.device,
        )

    def _enable_decoder_timing_parameters(self) -> int:
        """Unfreeze gates globally and consumers only in explicitly selected layers."""
        decoder = self.model.decoder
        while hasattr(decoder, "_forward_module"):
            decoder = decoder._forward_module

        selected_layers = self._selected_timing_layers(decoder)
        enabled_params = 0
        for name, param in decoder.named_parameters():
            if "timing_" not in name:
                continue

            is_gate = name.endswith("timing_attn_gate")
            in_consumer_layers = (
                selected_layers is None
                or self._timing_param_in_layers(name, selected_layers)
            )
            is_timing_adapter = (
                self._enable_timing_consumer_training
                and in_consumer_layers
                and ".timing_cross_attn." in name
                and ("lora_" in name or "lokr_" in name or "hada_" in name)
                and any(
                    f".{module_name}." in name
                    for module_name in self._timing_consumer_adapter_modules
                )
            )
            is_timing_norm = (
                self._enable_timing_consumer_training
                and in_consumer_layers
                and ".timing_cross_attn_norm." in name
            )
            is_timing_global = (
                self._enable_timing_consumer_training
                and self._enable_phrase_modulation
                and in_consumer_layers
                and ".timing_global_" in name
            )
            should_train = is_gate or is_timing_adapter or is_timing_norm or is_timing_global
            param.requires_grad = should_train
            if should_train:
                enabled_params += param.numel()
        return enabled_params

    def _assert_controlled_timing_contract(
        self,
        *,
        bootstrapped_layers: int,
        decoder_timing_params: int,
    ) -> None:
        """Fail before training when the historical-v4 timing contract differs."""
        if self.timing_encoder is None:
            raise RuntimeError("controlled Phase-D timing encoder is missing")
        encoder = self.timing_encoder
        cfg = encoder.config
        if cfg.init_profile != "historical_v4":
            raise RuntimeError(f"controlled timing profile mismatch: {cfg.init_profile!r}")
        expected_max_seq_len = int(
            getattr(self.training_config, "timing_max_seq_len", 0)
        ) or 1024
        if cfg.max_seq_len != expected_max_seq_len or cfg.enable_phrase_features or cfg.enable_phrase_modulation:
            raise RuntimeError(
                "controlled historical-v4 encoder has an invalid sequence/phrase topology: "
                f"max_seq_len={cfg.max_seq_len}, expected={expected_max_seq_len}, "
                f"phrase_features={cfg.enable_phrase_features}, "
                f"phrase_modulation={cfg.enable_phrase_modulation}"
            )
        if encoder.global_gate is not None:
            raise RuntimeError("controlled historical-v4 encoder must not contain a global gate")
        if float(encoder.output_gate.detach().float()) != 0.0:
            raise RuntimeError("controlled historical-v4 output gate must initialize exactly at 0")
        if float(encoder.proj.bias.detach().float().abs().max()) != 0.0:
            raise RuntimeError("controlled historical-v4 projection bias must initialize exactly at 0")
        projection_std = float(encoder.proj.weight.detach().float().std())
        if not (0.02 < projection_std < 0.04):
            raise RuntimeError(f"historical-v4 Xavier projection std is invalid: {projection_std}")
        if encoder.stream_type_embedding is None:
            raise RuntimeError("controlled historical-v4 stream embedding is missing")
        stream_std = float(encoder.stream_type_embedding.detach().float().std())
        if not (0.01 < stream_std < 0.03):
            raise RuntimeError(f"historical-v4 stream embedding std is invalid: {stream_std}")
        decoder = self.model.decoder
        while hasattr(decoder, "_forward_module"):
            decoder = decoder._forward_module
        gates = [
            parameter
            for name, parameter in decoder.named_parameters()
            if name.endswith("timing_attn_gate")
        ]
        if bootstrapped_layers != 24 or len(gates) != 24:
            raise RuntimeError(
                f"controlled decoder must contain 24 bootstrapped timing layers; "
                f"bootstrapped={bootstrapped_layers}, gates={len(gates)}"
            )
        if decoder_timing_params != 49_152 or sum(parameter.numel() for parameter in gates) != 49_152:
            raise RuntimeError(
                f"controlled decoder gate-only topology must contain 49,152 parameters; "
                f"enabled={decoder_timing_params}, gates={sum(parameter.numel() for parameter in gates)}"
            )
        gate_values = torch.cat([parameter.detach().float().reshape(-1) for parameter in gates])
        if not bool(torch.all(gate_values == self._timing_gate_init_logit)):
            raise RuntimeError(
                "controlled decoder timing gates have the wrong initialization: "
                f"expected={self._timing_gate_init_logit}"
            )
        unexpected_trainable = [
            name
            for name, parameter in decoder.named_parameters()
            if "timing_" in name and parameter.requires_grad and not name.endswith("timing_attn_gate")
        ]
        if unexpected_trainable:
            raise RuntimeError(
                f"controlled decoder has non-gate trainable timing parameters: {unexpected_trainable[:8]}"
            )
        nonzero_timing_b = [
            name
            for name, parameter in decoder.named_parameters()
            if ".timing_cross_attn." in name
            and "lora_B" in name
            and int(torch.count_nonzero(parameter.detach())) != 0
        ]
        if nonzero_timing_b:
            raise RuntimeError(
                f"controlled timing-attention LoRA-B tensors must initialize at zero: {nonzero_timing_b[:8]}"
            )

    def _install_timing_residual_hooks(self) -> None:
        self._timing_layer_residual_rms = {}
        decoder = self.model.decoder
        while hasattr(decoder, "_forward_module"):
            decoder = decoder._forward_module
        index = 0
        for layer in decoder.modules():
            if not hasattr(layer, "timing_cross_attn") or not hasattr(layer, "timing_attn_gate"):
                continue
            layer_index = index
            gate = layer.timing_attn_gate

            def hook(_module, _inputs, output, layer_index=layer_index, gate=gate):
                value = output[0] if isinstance(output, tuple) else output
                if isinstance(value, torch.Tensor):
                    residual = value.detach().float() * torch.sigmoid(gate.detach().float())
                    self._timing_layer_residual_rms[str(layer_index)] = float(
                        residual.square().mean().sqrt().item()
                    )

            layer.timing_cross_attn.register_forward_hook(hook)
            index += 1

    def phase_d_ablation_snapshot(self) -> Dict[str, Any]:
        if self.timing_encoder is None:
            raise RuntimeError("timing encoder is unavailable")
        gates = [(name, p) for name, p in self.named_parameters() if name.endswith("timing_attn_gate")]
        timing_trainable = [
            (name, p)
            for name, p in self.model.decoder.named_parameters()
            if "timing_" in name and p.requires_grad
        ]
        timing_b = [
            p
            for name, p in self.model.decoder.named_parameters()
            if "timing_cross_attn" in name and "lora_B" in name
        ]
        gate_values = torch.cat([p.detach().float().reshape(-1) for _, p in gates])
        encoder = self.timing_encoder
        global_gate = getattr(encoder, "global_gate", None)
        return {
            "output_gate_logit": float(encoder.output_gate.detach().float()),
            "output_gate_strength": float(torch.sigmoid(encoder.output_gate.detach().float())),
            "global_gate_logit": float(global_gate.detach().float()) if global_gate is not None else None,
            "global_gate_strength": float(torch.sigmoid(global_gate.detach().float())) if global_gate is not None else None,
            "projection_l2": float(encoder.proj.weight.detach().float().norm()),
            "projection_std": float(encoder.proj.weight.detach().float().std()),
            "projection_bias_l2": float(encoder.proj.bias.detach().float().norm()),
            "stream_embedding_l2": (
                float(encoder.stream_type_embedding.detach().float().norm())
                if encoder.stream_type_embedding is not None else 0.0
            ),
            "stream_embedding_std": (
                float(encoder.stream_type_embedding.detach().float().std())
                if encoder.stream_type_embedding is not None else 0.0
            ),
            "decoder_gate_tensor_count": len(gates),
            "decoder_gate_parameter_count": sum(p.numel() for _, p in gates),
            "decoder_gate_min": float(gate_values.min()),
            "decoder_gate_mean": float(gate_values.mean()),
            "decoder_gate_max": float(gate_values.max()),
            "decoder_timing_trainable_count": sum(p.numel() for _, p in timing_trainable),
            "decoder_timing_trainable_names": [name for name, _ in timing_trainable],
            "timing_consumer_lora_b_nonzero": sum(
                int(torch.count_nonzero(p.detach())) for p in timing_b
            ),
        }

    def promote_decoder_timing_gates_fp32(self) -> int:
        """Keep tiny gate updates representable during BF16 mixed-precision training."""
        promoted = 0
        for name, parameter in self.named_parameters():
            if parameter.requires_grad and name.endswith("timing_attn_gate"):
                parameter.data = parameter.data.float()
                promoted += parameter.numel()
        return promoted

    def timing_optimizer_groups(
        self,
        *,
        base_lr: float,
        timing_encoder_lr: float,
        timing_gate_lr: float,
        timing_consumer_lr: float,
    ) -> list[dict[str, Any]]:
        """Build disjoint optimizer groups for identity-preserving timing training."""
        groups: dict[str, list[nn.Parameter]] = {
            "base": [],
            "timing_encoder": [],
            "timing_gate": [],
            "timing_consumer": [],
        }
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.endswith("timing_attn_gate"):
                group = "timing_gate"
            elif (
                name.startswith("timing_encoder.")
                or name.startswith("decoder_timing_supervisor.")
                or name.startswith("performance_timing_predictor.")
                or name.startswith("decoder_expressivity_supervisor.")
            ):
                group = "timing_encoder"
            elif "timing_cross_attn" in name or "timing_global_" in name:
                group = "timing_consumer"
            else:
                group = "base"
            groups[group].append(parameter)

        rates = {
            "base": float(base_lr),
            "timing_encoder": float(timing_encoder_lr or base_lr),
            "timing_gate": float(timing_gate_lr or base_lr),
            "timing_consumer": float(timing_consumer_lr or base_lr),
        }
        return [
            {"params": parameters, "lr": rates[name], "group_name": name}
            for name, parameters in groups.items()
            if parameters
        ]

    def _selected_timing_layers(self, decoder: nn.Module) -> set[int] | None:
        if self._timing_train_last_n_layers <= 0:
            return None

        layer_pattern = re.compile(r"\.layers\.(\d+)\.")
        layer_ids: set[int] = set()
        for name, _ in decoder.named_parameters():
            if "timing_" not in name:
                continue
            match = layer_pattern.search(name)
            if match:
                layer_ids.add(int(match.group(1)))

        if not layer_ids:
            return None

        keep = min(self._timing_train_last_n_layers, len(layer_ids))
        return set(sorted(layer_ids)[-keep:])

    @staticmethod
    def _timing_param_in_layers(name: str, layer_ids: set[int]) -> bool:
        match = re.search(r"\.layers\.(\d+)\.", name)
        if match is None:
            return True
        return int(match.group(1)) in layer_ids

    def _bootstrap_decoder_timing_parameters(self) -> int:
        """Initialize timing cross-attention from the pretrained text cross-attention."""
        decoder = self.model.decoder
        while hasattr(decoder, "_forward_module"):
            decoder = decoder._forward_module

        bootstrapped_layers = 0
        for layer in decoder.modules():
            if not (
                hasattr(layer, "cross_attn")
                and hasattr(layer, "timing_cross_attn")
                and hasattr(layer, "cross_attn_norm")
                and hasattr(layer, "timing_cross_attn_norm")
                and hasattr(layer, "timing_attn_gate")
            ):
                continue
            layer.timing_cross_attn.load_state_dict(layer.cross_attn.state_dict(), strict=True)
            layer.timing_cross_attn_norm.load_state_dict(layer.cross_attn_norm.state_dict(), strict=True)
            layer.timing_attn_gate.data.fill_(self._timing_gate_init_logit)
            bootstrapped_layers += 1
        return bootstrapped_layers

    # -----------------------------------------------------------------------
    # Training step
    # -----------------------------------------------------------------------

    def _pool_decoder_states_for_timing(
        self,
        decoder_states: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        event_starts_sec: torch.Tensor,
        event_ends_sec: torch.Tensor,
        audio_durations_sec: torch.Tensor,
        timing_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Average decoder states over each aligned timing event span."""
        bsz, max_t, dim = decoder_states.shape
        _, max_events = event_starts_sec.shape
        pooled = decoder_states.new_zeros(bsz, max_events, dim)
        valid_mask = timing_mask if timing_mask is not None else torch.ones(
            bsz, max_events, dtype=torch.bool, device=decoder_states.device
        )

        for b in range(bsz):
            valid_len = int(decoder_attention_mask[b].sum().item())
            if valid_len <= 0:
                continue
            audio_dur = float(audio_durations_sec[b].item())
            if audio_dur <= 1e-4:
                audio_dur = float(event_ends_sec[b][valid_mask[b]].max().item()) if valid_mask[b].any() else 1.0
            for n in range(max_events):
                if not bool(valid_mask[b, n]):
                    continue
                start_s = max(float(event_starts_sec[b, n].item()), 0.0)
                end_s = max(float(event_ends_sec[b, n].item()), start_s + 1e-4)
                start_idx = int((start_s / audio_dur) * valid_len)
                end_idx = int(math.ceil((end_s / audio_dur) * valid_len))
                start_idx = max(0, min(valid_len - 1, start_idx))
                end_idx = max(start_idx + 1, min(valid_len, end_idx))
                pooled[b, n] = decoder_states[b, start_idx:end_idx].mean(dim=0)
        return pooled

    def training_step(self, batch: Dict[str, torch.Tensor], record_loss: bool = True) -> torch.Tensor:
        """Single training step with corrected timestep sampling + CFG dropout.

        Args:
            batch: Dict with keys ``target_latents``, ``attention_mask``,
                ``encoder_hidden_states``, ``encoder_attention_mask``,
                ``context_latents``.

        Returns:
            Scalar loss tensor (``float32`` for stable backward).
        """
        # Mixed-precision context
        if self.device_type in ("cuda", "xpu", "mps"):
            autocast_ctx = torch.autocast(
                device_type=self.device_type, dtype=self.dtype
            )
        else:
            autocast_ctx = nullcontext()

        with autocast_ctx:
            nb = self.transfer_non_blocking

            target_latents = batch["target_latents"].to(
                self.device, dtype=self.dtype, non_blocking=nb
            )
            attention_mask = batch["attention_mask"].to(
                self.device, dtype=self.dtype, non_blocking=nb
            )
            encoder_hidden_states = batch["encoder_hidden_states"].to(
                self.device, dtype=self.dtype, non_blocking=nb
            )
            encoder_attention_mask = batch["encoder_attention_mask"].to(
                self.device, dtype=self.dtype, non_blocking=nb
            )
            context_latents = batch["context_latents"].to(
                self.device, dtype=self.dtype, non_blocking=nb
            )
            ref_voice_features = batch.get("ref_voice_features")
            ref_voice_attention_mask = batch.get("ref_voice_attention_mask")

            bsz = target_latents.shape[0]
            batch_role = "ordinary"
            if self._phase_d_d1_alternating_roles:
                metadata = batch.get("metadata")
                if not isinstance(metadata, list) or len(metadata) != bsz:
                    raise RuntimeError("D1 batch metadata is missing or has the wrong batch size")
                roles = {
                    item.get("phase_d_batch_role")
                    for item in metadata
                    if isinstance(item, dict)
                }
                if len(roles) != 1 or roles.pop() not in {"short_clarity", "full_timing"}:
                    raise RuntimeError(f"D1 batch must contain one valid homogeneous role: {metadata}")
                batch_role = str(metadata[0]["phase_d_batch_role"])
                if bsz != 1:
                    raise RuntimeError(f"D1 role schedule requires batch size 1, got {bsz}")
            self._last_batch_role = batch_role
            short_clarity_batch = batch_role == "short_clarity"
            full_timing_batch = batch_role == "full_timing"

            # ---- CFG dropout (CORRECTED -- missing in original trainer) ----
            if self._null_cond_emb is not None and self._cfg_ratio > 0.0:
                encoder_hidden_states = apply_cfg_dropout(
                    encoder_hidden_states,
                    self._null_cond_emb,
                    cfg_ratio=self._cfg_ratio,
                )

            voice_identity_embedding = None
            if (
                self.voice_conditioner is not None
                and ref_voice_features is not None
                and ref_voice_attention_mask is not None
            ):
                ref_voice_features = ref_voice_features.to(
                    self.device, dtype=self.dtype, non_blocking=nb
                )
                ref_voice_attention_mask = ref_voice_attention_mask.to(
                    self.device, dtype=self.dtype, non_blocking=nb
                )
                voice_hidden_states, voice_attention_mask = self.voice_conditioner(
                    ref_voice_features=ref_voice_features,
                    ref_voice_attention_mask=ref_voice_attention_mask,
                    scale=self._voice_condition_scale,
                )
                voice_identity_embedding = _mean_pool_masked(voice_hidden_states, voice_attention_mask)
                if self._identity_v5_local_enabled and self.v5_fragment_attention is not None:
                    if ref_voice_features.ndim == 5:
                        crop_count, crop_time = ref_voice_features.shape[1], ref_voice_features.shape[3]
                        crop_ids = torch.arange(crop_count, device=self.device).repeat_interleave(crop_time)
                        crop_ids = crop_ids[:voice_hidden_states.shape[1]].unsqueeze(0).expand(bsz, -1)
                    else:
                        crop_ids = torch.zeros(voice_hidden_states.shape[:2], dtype=torch.long, device=self.device)
                    encoder_hidden_states = self.v5_fragment_attention(
                        encoder_hidden_states, voice_hidden_states, crop_ids, voice_attention_mask.bool()
                    )
                    # V5 never concatenates an ungated full reference bank.
                else:
                    encoder_hidden_states = torch.cat([encoder_hidden_states, voice_hidden_states], dim=1)
                    encoder_attention_mask = torch.cat([encoder_attention_mask, voice_attention_mask], dim=1)

            # ---- V5 global prototype conditioning --------------------------
            identity_wavlm_target = batch.get("identity_wavlm_embedding")
            identity_ecapa_target = batch.get("identity_ecapa_embedding")
            if identity_wavlm_target is not None:
                identity_wavlm_target = identity_wavlm_target.to(self.device, dtype=torch.float32, non_blocking=nb)
            if identity_ecapa_target is not None:
                identity_ecapa_target = identity_ecapa_target.to(self.device, dtype=torch.float32, non_blocking=nb)
            identity_v5_prototype = batch.get("identity_v5_prototype")
            identity_v5_singer_hidden = None
            if self._identity_v5_global_enabled and self.v5_singer_projection is not None:
                if identity_v5_prototype is None:
                    raise RuntimeError("V5 batch is missing leave-one-song-out identity_v5_prototype")
                identity_v5_prototype = identity_v5_prototype.to(self.device, dtype=torch.float32, non_blocking=nb)
                identity_v5_singer_hidden = self.v5_singer_projection(identity_v5_prototype.to(dtype=self.dtype))

            # ---- Phase D: timing branch -------------------------------------
            timing_loss = torch.tensor(0.0, device=self.device)
            decoder_timing_loss = torch.tensor(0.0, device=self.device)
            timing_tokens = batch.get("timing_tokens")
            timing_mask = batch.get("timing_mask")
            timing_targets = batch.get("timing_targets")
            predictor_inputs = batch.get("predictor_inputs")
            predictor_mask = batch.get("predictor_mask")
            phrase_features = batch.get("phrase_features")
            global_phrase_features = batch.get("global_phrase_features")
            phrase_ids = batch.get("phrase_ids")
            f0_targets = batch.get("f0_targets")
            identity_spectral_target = batch.get("identity_spectral_formant")
            identity_pitch_target = batch.get("identity_pitch_style")
            identity_negative_wavlm = batch.get("identity_negative_wavlm_embeddings")
            identity_negative_ecapa = batch.get("identity_negative_ecapa_embeddings")
            energy_targets = batch.get("energy_targets")
            terminal_decay = batch.get("terminal_decay")
            cv_ratio_targets = batch.get("cv_ratio_targets")
            alignment_confidence = batch.get("alignment_confidence")
            beat_confidence = batch.get("beat_confidence")
            release_targets = batch.get("release_targets")
            timing_event_starts = batch.get("timing_event_starts")
            timing_event_ends = batch.get("timing_event_ends")
            timing_audio_durations = batch.get("timing_audio_duration")
            timing_counterfactual_mask = None
            drop_timing = False
            timing_loss = torch.tensor(0.0, device=self.device)
            predictor_loss = torch.tensor(0.0, device=self.device)
            expressivity_loss = torch.tensor(0.0, device=self.device)
            if (
                self.timing_encoder is not None
                and (timing_tokens is not None or predictor_inputs is not None)
            ):
                if timing_tokens is not None:
                    timing_tokens = timing_tokens.to(self.device, non_blocking=nb)
                if timing_mask is not None:
                    timing_mask = timing_mask.to(self.device, non_blocking=nb)
                if predictor_inputs is not None:
                    predictor_inputs = predictor_inputs.to(self.device, non_blocking=nb)
                if predictor_mask is not None:
                    predictor_mask = predictor_mask.to(self.device, non_blocking=nb)
                if timing_targets is not None:
                    timing_targets = timing_targets.to(
                        self.device, dtype=torch.float32, non_blocking=nb
                    )
                if phrase_features is not None:
                    phrase_features = phrase_features.to(
                        self.device, dtype=torch.float32, non_blocking=nb
                    )
                if global_phrase_features is not None:
                    global_phrase_features = global_phrase_features.to(
                        self.device, dtype=torch.float32, non_blocking=nb
                    )
                if phrase_ids is not None:
                    phrase_ids = phrase_ids.to(
                        self.device, dtype=torch.long, non_blocking=nb
                    )
                if f0_targets is not None:
                    f0_targets = f0_targets.to(self.device, dtype=torch.float32, non_blocking=nb)
                if identity_wavlm_target is not None:
                    identity_wavlm_target = identity_wavlm_target.to(self.device, dtype=torch.float32, non_blocking=nb)
                if identity_ecapa_target is not None:
                    identity_ecapa_target = identity_ecapa_target.to(self.device, dtype=torch.float32, non_blocking=nb)
                if identity_spectral_target is not None:
                    identity_spectral_target = identity_spectral_target.to(self.device, dtype=torch.float32, non_blocking=nb)
                if identity_pitch_target is not None:
                    identity_pitch_target = identity_pitch_target.to(self.device, dtype=torch.float32, non_blocking=nb)
                if identity_negative_wavlm is not None:
                    identity_negative_wavlm = identity_negative_wavlm.to(self.device, dtype=torch.float32, non_blocking=nb)
                if identity_negative_ecapa is not None:
                    identity_negative_ecapa = identity_negative_ecapa.to(self.device, dtype=torch.float32, non_blocking=nb)
                if energy_targets is not None:
                    energy_targets = energy_targets.to(self.device, dtype=torch.float32, non_blocking=nb)
                if terminal_decay is not None:
                    terminal_decay = terminal_decay.to(self.device, dtype=torch.float32, non_blocking=nb)
                if cv_ratio_targets is not None:
                    cv_ratio_targets = cv_ratio_targets.to(self.device, dtype=torch.float32, non_blocking=nb)
                if alignment_confidence is not None:
                    alignment_confidence = alignment_confidence.to(
                        self.device, dtype=torch.float32, non_blocking=nb
                    )
                if beat_confidence is not None:
                    beat_confidence = beat_confidence.to(
                        self.device, dtype=torch.float32, non_blocking=nb
                    )
                if release_targets is not None:
                    release_targets = release_targets.to(
                        self.device, dtype=torch.long, non_blocking=nb
                    )
                if timing_event_starts is not None:
                    timing_event_starts = timing_event_starts.to(
                        self.device, dtype=torch.float32, non_blocking=nb
                    )
                if timing_event_ends is not None:
                    timing_event_ends = timing_event_ends.to(
                        self.device, dtype=torch.float32, non_blocking=nb
                    )
                if timing_audio_durations is not None:
                    timing_audio_durations = timing_audio_durations.to(
                        self.device, dtype=torch.float32, non_blocking=nb
                    )
                predictor_condition_mask = predictor_mask if predictor_mask is not None else timing_mask

                # Apply conditioning dropout during ordinary training. Required
                # diagnostic probe batches are explicitly timing-active so probe
                # activations are deterministic and cannot be invalidated by the
                # 10% stochastic conditioning dropout.
                force_probe_timing = bool(
                    self._controlled_phase_d
                    and getattr(self, "_force_timing_condition_for_probe", False)
                )
                if self._phase_d_d1_alternating_roles:
                    # D1 roles are objective controls, not stochastic hints.
                    # Short batches are exactly timing-off; full batches are exactly timing-on.
                    drop_timing = short_clarity_batch
                else:
                    drop_timing = (
                        not force_probe_timing
                        and self._timing_condition_dropout > 0.0
                        and torch.rand(1).item() < self._timing_condition_dropout
                    )
                if not drop_timing:
                    timing_out = None
                    predictor_out = None
                    if self._timing_condition_source in {"sidecar", "hybrid"} and timing_tokens is not None:
                        encoder_phrase_features = (
                            phrase_features if self.timing_encoder.config.enable_phrase_features else None
                        )
                        encoder_global_features = (
                            global_phrase_features if self.timing_encoder.config.enable_phrase_modulation else None
                        )
                        encoder_event_starts = (
                            timing_event_starts if self.timing_encoder.config.enable_absolute_time_conditioning else None
                        )
                        encoder_event_ends = (
                            timing_event_ends if self.timing_encoder.config.enable_absolute_time_conditioning else None
                        )
                        encoder_audio_durations = (
                            timing_audio_durations if self.timing_encoder.config.enable_absolute_time_conditioning else None
                        )
                        timing_out = self.timing_encoder(
                            timing_tokens=timing_tokens,
                            timing_mask=timing_mask,
                            timing_targets=timing_targets,
                            phrase_features=encoder_phrase_features,
                            global_features=encoder_global_features,
                            event_starts_sec=encoder_event_starts,
                            event_ends_sec=encoder_event_ends,
                            audio_durations_sec=encoder_audio_durations,
                            dur_weight=self._timing_dur_weight,
                            onset_weight=self._timing_onset_weight,
                            pause_weight=self._timing_pause_weight,
                            phrase_weight=self._timing_phrase_weight,
                            tempo_weight=self._timing_tempo_weight,
                            terminal_weight=self._timing_terminal_weight,
                        )
                        timing_loss = timing_out["timing_loss"]
                    if self.performance_timing_predictor is not None:
                        predictor_out = self.performance_timing_predictor(
                            predictor_inputs=predictor_inputs,
                            timing_tokens=timing_tokens,
                            timing_mask=predictor_condition_mask,
                            timing_targets=timing_targets,
                            phrase_features=phrase_features,
                            global_features=global_phrase_features,
                            phrase_ids=phrase_ids,
                            f0_targets=f0_targets,
                            energy_targets=energy_targets,
                            terminal_decay=terminal_decay,
                            cv_ratio_targets=cv_ratio_targets,
                            release_targets=release_targets,
                            alignment_confidence=alignment_confidence,
                            beat_confidence=beat_confidence,
                            dur_weight=self._timing_dur_weight,
                            onset_weight=self._timing_onset_weight,
                            pause_weight=self._timing_pause_weight,
                            phrase_weight=self._timing_phrase_weight,
                            tempo_weight=self._timing_tempo_weight,
                            terminal_weight=self._timing_terminal_weight,
                            expressivity_f0_weight=self._expressivity_f0_weight,
                            expressivity_energy_weight=self._expressivity_energy_weight,
                            expressivity_terminal_decay_weight=self._expressivity_terminal_decay_weight,
                            expressivity_cv_ratio_weight=self._expressivity_cv_ratio_weight,
                            expressivity_release_weight=self._expressivity_release_weight,
                        )
                        predictor_loss = predictor_out["predictor_loss"]
                        expressivity_loss = predictor_out.get("expressivity_loss", expressivity_loss)

                    selected = timing_out
                    if self._timing_condition_source == "predictor" and predictor_out is not None:
                        selected = predictor_out
                    elif (
                        self._timing_condition_source == "hybrid"
                        and timing_out is not None
                        and predictor_out is not None
                    ):
                        mix = self._timing_hybrid_mix
                        hybrid_global = None
                        if (
                            timing_out.get("global_condition") is not None
                            and predictor_out.get("global_condition") is not None
                        ):
                            hybrid_global = (
                                (1.0 - mix) * timing_out["global_condition"]
                                + mix * predictor_out["global_condition"]
                            )
                        elif predictor_out.get("global_condition") is not None:
                            hybrid_global = predictor_out["global_condition"]
                        else:
                            hybrid_global = timing_out.get("global_condition")
                        selected = {
                            "projected": (1.0 - mix) * timing_out["projected"] + mix * predictor_out["projected"],
                            "attn_mask": timing_out["attn_mask"],
                            "global_condition": hybrid_global,
                        }
                    elif selected is None:
                        selected = predictor_out

                    if selected is not None:
                        timing_hidden_states = selected["projected"].to(self.dtype)
                        timing_attention_mask = selected["attn_mask"].to(encoder_attention_mask.dtype)
                        if self._timing_attention_prior == "local_monotonic":
                            if (
                                timing_event_starts is None
                                or timing_event_ends is None
                                or timing_audio_durations is None
                                or timing_mask is None
                            ):
                                raise RuntimeError(
                                    "local monotonic timing batch lacks absolute event boundaries"
                                )
                            raw_decoder = self.model.decoder
                            while hasattr(raw_decoder, "_forward_module"):
                                raw_decoder = raw_decoder._forward_module
                            patch_size = max(1, int(getattr(raw_decoder, "patch_size", 1)))
                            timing_query_len = math.ceil(target_latents.shape[1] / patch_size)
                            mask_kwargs = {
                                "event_starts_sec": timing_event_starts,
                                "event_ends_sec": timing_event_ends,
                                "audio_durations_sec": timing_audio_durations,
                                "timing_mask": timing_mask.bool(),
                                "query_len": timing_query_len,
                                "dtype": self.dtype,
                                "sigma_sec": self._timing_local_sigma_sec,
                                "window_sec": self._timing_local_window_sec,
                            }
                            timing_attention_mask = build_local_timing_attention_mask(
                                **mask_kwargs
                            )
                            if self._timing_counterfactual_weight > 0.0:
                                shift_sign = -1.0 if torch.rand((), device=self.device).item() < 0.5 else 1.0
                                timing_counterfactual_mask = build_local_timing_attention_mask(
                                    **mask_kwargs,
                                    shift_sec=shift_sign * self._timing_counterfactual_shift_sec,
                                )
                        timing_global_states = selected.get("global_condition")
                        if timing_global_states is not None:
                            timing_global_states = timing_global_states.to(self.dtype)
                    else:
                        timing_hidden_states = None
                        timing_attention_mask = None
                        timing_global_states = None
                else:
                    timing_hidden_states = None
                    timing_attention_mask = None
                    timing_global_states = None
            else:
                timing_hidden_states = None
                timing_attention_mask = None
                timing_global_states = None

            # ---- Flow matching noise ----------------------------------------
            x1 = torch.randn_like(target_latents)  # noise
            x0 = target_latents  # data

            # ---- Continuous timestep sampling (CORRECTED) -------------------
            t, r = sample_timesteps(
                batch_size=bsz,
                device=self.device,
                dtype=self.dtype,
                data_proportion=self._data_proportion,
                timestep_mu=self._timestep_mu,
                timestep_sigma=self._timestep_sigma,
                use_meanflow=False,  # r = t for all ACE-Step variants
            )
            t_ = t.unsqueeze(-1).unsqueeze(-1)

            # ---- Interpolate x_t -------------------------------------------
            xt = t_ * x1 + (1.0 - t_) * x0
            if self.force_input_grads_for_checkpointing:
                xt = xt.requires_grad_(True)

            # ---- Decoder forward -------------------------------------------
            decoder_outputs = self.model.decoder(
                hidden_states=xt,
                timestep=t,
                timestep_r=t,  # r = t
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timing_hidden_states=timing_hidden_states,
                timing_attention_mask=timing_attention_mask,
                timing_global_states=timing_global_states,
                context_latents=context_latents,
                identity_block_adapters=self.v5_block_adapters if self._identity_v5_global_enabled else None,
                identity_singer_embedding=identity_v5_singer_hidden,
            )

            # ---- Flow matching and matched-input Phase A preservation ------
            flow = x1 - x0
            frame_weights = None
            if self._timing_event_flow_weight > 0.0 and timing_hidden_states is not None:
                if (
                    timing_event_starts is None
                    or timing_event_ends is None
                    or timing_audio_durations is None
                    or timing_mask is None
                ):
                    raise RuntimeError("event-weighted flow loss lacks timing boundaries")
                frame_weights = build_event_frame_weights(
                    event_starts_sec=timing_event_starts,
                    event_ends_sec=timing_event_ends,
                    audio_durations_sec=timing_audio_durations,
                    timing_mask=timing_mask.bool(),
                    query_len=flow.shape[1],
                    extra_weight=self._timing_event_flow_weight,
                    margin_sec=self._timing_event_flow_margin_sec,
                )

            def flow_error_per_sample(prediction: torch.Tensor) -> torch.Tensor:
                frame_error = torch.square(prediction.float() - flow.float()).mean(dim=-1)
                if frame_weights is None:
                    return frame_error.mean(dim=-1)
                return (frame_error * frame_weights).sum(dim=-1) / frame_weights.sum(dim=-1).clamp_min(1e-6)

            correct_flow_error = flow_error_per_sample(decoder_outputs[0])
            diffusion_loss = correct_flow_error.mean()
            timing_counterfactual_loss = torch.tensor(0.0, device=self.device)
            shifted_flow_error = torch.tensor(0.0, device=self.device)
            if self._timing_counterfactual_weight > 0.0 and timing_hidden_states is not None:
                if timing_counterfactual_mask is None:
                    raise RuntimeError("counterfactual timing objective lacks shifted mask")
                shifted_outputs = self.model.decoder(
                    hidden_states=xt,
                    timestep=t,
                    timestep_r=t,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timing_hidden_states=timing_hidden_states,
                    timing_attention_mask=timing_counterfactual_mask,
                    timing_global_states=timing_global_states,
                    context_latents=context_latents,
                    identity_block_adapters=None,
                    identity_singer_embedding=None,
                )
                shifted_per_sample = flow_error_per_sample(shifted_outputs[0])
                shifted_flow_error = shifted_per_sample.mean()
                timing_counterfactual_loss = F.relu(
                    self._timing_counterfactual_margin
                    + correct_flow_error
                    - shifted_per_sample
                ).mean()
            teacher_loss = torch.tensor(0.0, device=self.device)
            teacher_weight = (
                self._identity_v5_teacher_loss_weight
                + self._parent_preservation_loss_weight
            )
            if teacher_weight > 0.0:
                if self.phase_a_teacher_decoder is None:
                    raise RuntimeError(
                        "parent preservation weight is nonzero but the frozen parent teacher was not initialized"
                    )
                student_preservation_outputs = decoder_outputs
                if self._phase_d_d1_alternating_roles and full_timing_batch:
                    # The written D1 contract is v_D(..., tau=0) versus v_C on
                    # the exact same xt/t/conditioning.  Reusing timing-on
                    # decoder_outputs here would couple preservation to timing.
                    student_preservation_outputs = self.model.decoder(
                        hidden_states=xt, timestep=t, timestep_r=t, attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask,
                        timing_hidden_states=None, timing_attention_mask=None, timing_global_states=None,
                        context_latents=context_latents, identity_block_adapters=None,
                        identity_singer_embedding=None,
                    )
                with torch.no_grad():
                    teacher_outputs = self.phase_a_teacher_decoder(
                        hidden_states=xt, timestep=t, timestep_r=t, attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask,
                        timing_hidden_states=None, timing_attention_mask=None, timing_global_states=None,
                        context_latents=context_latents, identity_block_adapters=None,
                        identity_singer_embedding=None,
                    )
                teacher_loss = F.mse_loss(
                    student_preservation_outputs[0].float(),
                    teacher_outputs[0].detach().float(),
                )

            if (
                self.decoder_timing_supervisor is not None
                and self._timing_decoder_loss_weight > 0.0
                and not drop_timing
                and timing_targets is not None
                and timing_event_starts is not None
                and timing_event_ends is not None
                and timing_audio_durations is not None
            ):
                pooled_decoder_states = self._pool_decoder_states_for_timing(
                    decoder_states=decoder_outputs[0],
                    decoder_attention_mask=attention_mask,
                    event_starts_sec=timing_event_starts,
                    event_ends_sec=timing_event_ends,
                    audio_durations_sec=timing_audio_durations,
                    timing_mask=timing_mask,
                )
                decoder_timing_loss = self.decoder_timing_supervisor(
                    decoder_event_states=pooled_decoder_states.float(),
                    timing_targets=timing_targets,
                    timing_mask=timing_mask,
                    dur_weight=self._timing_dur_weight,
                    onset_weight=self._timing_onset_weight,
                    pause_weight=self._timing_pause_weight,
                    phrase_weight=self._timing_phrase_weight,
                    tempo_weight=self._timing_tempo_weight,
                    terminal_weight=self._timing_terminal_weight,
                )
                if (
                    self.decoder_expressivity_supervisor is not None
                    and self._expressivity_loss_weight > 0.0
                ):
                    expressivity_loss = expressivity_loss + self.decoder_expressivity_supervisor(
                        decoder_event_states=pooled_decoder_states.float(),
                        f0_targets=f0_targets,
                        energy_targets=energy_targets,
                        terminal_decay=terminal_decay,
                        cv_ratio_targets=cv_ratio_targets,
                        release_targets=release_targets,
                        alignment_confidence=alignment_confidence,
                        beat_confidence=beat_confidence,
                        timing_mask=timing_mask,
                        f0_weight=self._expressivity_f0_weight,
                        energy_weight=self._expressivity_energy_weight,
                        terminal_weight=self._expressivity_terminal_decay_weight,
                        cv_ratio_weight=self._expressivity_cv_ratio_weight,
                        release_weight=self._expressivity_release_weight,
                    )

            # ---- Optional pitch-aware auxiliaries --------------------------
            total_loss = (
                diffusion_loss
                + teacher_weight * teacher_loss
                + self._timing_counterfactual_weight * timing_counterfactual_loss
            )

            # D1 applies timing objectives only on full-song timing batches.
            timing_objectives_enabled = (
                self._use_timing_branch
                and (not self._phase_d_d1_alternating_roles or full_timing_batch)
            )
            if timing_objectives_enabled and self._timing_loss_weight > 0.0:
                total_loss = total_loss + self._timing_loss_weight * timing_loss
            if (
                timing_objectives_enabled
                and self.performance_timing_predictor is not None
                and self._timing_predictor_loss_weight > 0.0
            ):
                total_loss = total_loss + self._timing_predictor_loss_weight * predictor_loss
            if timing_objectives_enabled and self._timing_decoder_loss_weight > 0.0:
                total_loss = total_loss + self._timing_decoder_loss_weight * decoder_timing_loss
            if timing_objectives_enabled and self._expressivity_loss_weight > 0.0:
                total_loss = total_loss + self._expressivity_loss_weight * expressivity_loss

            effective_f0_weight = (
                self._f0_loss_weight
                if not self._phase_d_d1_alternating_roles or short_clarity_batch
                else 0.0
            )
            f0_loss = torch.tensor(0.0, device=self.device)
            if effective_f0_weight > 0.0:
                pred_contour = _latent_f0_proxy_contour(decoder_outputs[0])
                target_contour = _latent_f0_proxy_contour(flow)
                f0_loss = F.l1_loss(pred_contour, target_contour)
                total_loss = total_loss + effective_f0_weight * f0_loss

            pred_x0_for_identity = x1 - decoder_outputs[0]
            target_x0_for_identity = x0
            decoded_identity_loss = torch.tensor(0.0, device=self.device)
            v5_supcon_loss = torch.tensor(0.0, device=self.device)
            run_decoded = self.v5_decoded_identity is not None and torch.rand((), device=self.device).item() < self._identity_v5_decoded_batch_fraction
            if run_decoded:
                decoded_embedding = self.v5_decoded_identity.embed(pred_x0_for_identity)
                decoded_identity_loss = (1.0 - F.cosine_similarity(decoded_embedding.float(), identity_v5_prototype.float(), dim=-1)).mean()
                total_loss = total_loss + self._identity_v5_waveform_identity_weight * decoded_identity_loss
                if self._identity_v5_supcon_weight > 0.0:
                    negative_prototypes = batch.get("identity_v5_negative_prototypes")
                    singer_ids = batch.get("identity_v5_negative_singer_ids")
                    if negative_prototypes is None or not singer_ids or len(singer_ids[0]) < 4:
                        raise RuntimeError("V5 SupCon batch lacks four distinct singer prototypes")
                    negative_prototypes = negative_prototypes.to(self.device, dtype=torch.float32, non_blocking=nb)
                    mapping = {str(sid): negative_prototypes[:, index] for index, sid in enumerate(singer_ids[0])}
                    v5_supcon_loss = distinct_singer_supcon(decoded_embedding, identity_v5_prototype, mapping, self._contrastive_temperature)
                    total_loss = total_loss + self._identity_v5_supcon_weight * v5_supcon_loss

            effective_speaker_weight = (
                self._speaker_loss_weight
                if not self._phase_d_d1_alternating_roles or short_clarity_batch
                else 0.0
            )
            if effective_speaker_weight > 0.0:
                pred_emb = _speaker_proxy_embedding(pred_x0_for_identity)
                target_emb = _speaker_proxy_embedding(target_x0_for_identity)
                speaker_loss = (1.0 - F.cosine_similarity(pred_emb, target_emb, dim=-1)).mean()
                total_loss = total_loss + effective_speaker_weight * speaker_loss
            else:
                speaker_loss = torch.tensor(0.0, device=self.device)

            contrastive_identity_loss = torch.tensor(0.0, device=self.device)
            verifier_aux_loss = torch.tensor(0.0, device=self.device)
            latent_to_wavlm_cosine_loss = torch.tensor(0.0, device=self.device)
            latent_to_ecapa_cosine_loss = torch.tensor(0.0, device=self.device)
            latent_to_spectral_formant_loss = torch.tensor(0.0, device=self.device)
            latent_to_pitch_style_loss = torch.tensor(0.0, device=self.device)

            identity_weights_enabled = (
                self._contrastive_identity_loss_weight > 0.0
                or self._verifier_aux_weight > 0.0
                or self._wavlm_aux_weight > 0.0
                or self._ecapa_aux_weight > 0.0
                or self._spectral_formant_aux_weight > 0.0
                or self._pitch_style_aux_weight > 0.0
            )
            if self._require_identity_sidecars and identity_weights_enabled:
                missing = []
                if (self._wavlm_aux_weight > 0.0 or self._contrastive_identity_loss_weight > 0.0) and identity_wavlm_target is None:
                    missing.append("identity_wavlm_embedding")
                if self._ecapa_aux_weight > 0.0 and identity_ecapa_target is None:
                    missing.append("identity_ecapa_embedding")
                if self._spectral_formant_aux_weight > 0.0 and identity_spectral_target is None:
                    missing.append("identity_spectral_formant")
                if self._pitch_style_aux_weight > 0.0 and identity_pitch_target is None:
                    missing.append("identity_pitch_style")
                if self._contrastive_identity_loss_weight > 0.0 and identity_negative_wavlm is None:
                    missing.append("identity_negative_wavlm_embeddings")
                if missing:
                    raise RuntimeError("Missing required Phase B identity sidecar targets: " + ", ".join(missing))

            latent_identity_features = _resize_embedding_dim(_speaker_proxy_embedding(pred_x0_for_identity).float(), 1536)
            if self._wavlm_aux_weight > 0.0 and identity_wavlm_target is not None and self.wavlm_aux_head is not None:
                wavlm_pred = F.normalize(self.wavlm_aux_head(latent_identity_features), dim=-1)
                wavlm_target = F.normalize(identity_wavlm_target.float(), dim=-1)
                latent_to_wavlm_cosine_loss = (1.0 - F.cosine_similarity(wavlm_pred, wavlm_target, dim=-1)).mean()
                total_loss = total_loss + self._wavlm_aux_weight * latent_to_wavlm_cosine_loss
                if self._verifier_aux_weight > 0.0:
                    verifier_aux_loss = verifier_aux_loss + latent_to_wavlm_cosine_loss
            if self._ecapa_aux_weight > 0.0 and identity_ecapa_target is not None and self.ecapa_aux_head is not None:
                ecapa_pred = F.normalize(self.ecapa_aux_head(latent_identity_features), dim=-1)
                ecapa_target = F.normalize(identity_ecapa_target.float(), dim=-1)
                latent_to_ecapa_cosine_loss = (1.0 - F.cosine_similarity(ecapa_pred, ecapa_target, dim=-1)).mean()
                total_loss = total_loss + self._ecapa_aux_weight * latent_to_ecapa_cosine_loss
                if self._verifier_aux_weight > 0.0:
                    verifier_aux_loss = verifier_aux_loss + latent_to_ecapa_cosine_loss
            if self._spectral_formant_aux_weight > 0.0 and identity_spectral_target is not None and self.spectral_formant_aux_head is not None:
                spec_pred = self.spectral_formant_aux_head(latent_identity_features)
                latent_to_spectral_formant_loss = F.smooth_l1_loss(spec_pred, identity_spectral_target.float())
                total_loss = total_loss + self._spectral_formant_aux_weight * latent_to_spectral_formant_loss
            if self._pitch_style_aux_weight > 0.0 and identity_pitch_target is not None and self.pitch_style_aux_head is not None:
                pitch_pred = self.pitch_style_aux_head(latent_identity_features)
                latent_to_pitch_style_loss = F.smooth_l1_loss(pitch_pred, identity_pitch_target.float())
                total_loss = total_loss + self._pitch_style_aux_weight * latent_to_pitch_style_loss
            if self._verifier_aux_weight > 0.0 and isinstance(verifier_aux_loss, torch.Tensor) and verifier_aux_loss.requires_grad:
                total_loss = total_loss + self._verifier_aux_weight * verifier_aux_loss

            if self._contrastive_identity_loss_weight > 0.0 and self.wavlm_aux_head is not None and identity_wavlm_target is not None:
                anchor = F.normalize(self.wavlm_aux_head(latent_identity_features), dim=-1)
                positive = F.normalize(identity_wavlm_target.float(), dim=-1)
                if identity_negative_wavlm is not None and identity_negative_wavlm.ndim == 3 and identity_negative_wavlm.shape[1] > 0:
                    negatives = F.normalize(identity_negative_wavlm.float(), dim=-1)
                    pos_logits = (anchor * positive).sum(dim=-1, keepdim=True)
                    neg_logits = torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1)
                    logits = torch.cat([pos_logits, neg_logits], dim=-1) / self._contrastive_temperature
                    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)
                    contrastive_identity_loss = F.cross_entropy(logits, labels)
                    total_loss = total_loss + self._contrastive_identity_loss_weight * contrastive_identity_loss
                elif not self._require_identity_sidecars:
                    contrastive_identity_loss = _info_nce_with_queue(
                        anchor=anchor,
                        positive=positive,
                        negative_queue=self._identity_negative_queue,
                        queue_count=min(self._identity_negative_queue_count, self._contrastive_num_negatives),
                        temperature=self._contrastive_temperature,
                    )
                    total_loss = total_loss + self._contrastive_identity_loss_weight * contrastive_identity_loss
                    with torch.no_grad():
                        q = self._identity_negative_queue
                        for emb in positive.detach().float():
                            q[self._identity_negative_queue_ptr % q.shape[0]].copy_(emb.cpu())
                            self._identity_negative_queue_ptr += 1
                            self._identity_negative_queue_count = min(self._identity_negative_queue_count + 1, q.shape[0])

            self._identity_v5_step += 1
            self._identity_v5_diagnostic_losses = {
                "diffusion": diffusion_loss,
                "teacher": teacher_loss,
                "identity": decoded_identity_loss,
                "contrastive": v5_supcon_loss,
            }
            self._last_aux_losses = {
                "diffusion_loss": float(diffusion_loss.detach().float().item()),
                "total_loss": float(total_loss.detach().float().item()),
                "timing_loss": float(timing_loss.detach().float().item()),
                "timing_loss_weight": float(self._timing_loss_weight),
                "timing_aux_weighted_loss": (
                    float((self._timing_loss_weight * timing_loss).detach().float().item())
                    if timing_objectives_enabled else 0.0
                ),
                "decoder_timing_loss": float(decoder_timing_loss.detach().float().item()),
                "decoder_timing_loss_weight": float(self._timing_decoder_loss_weight),
                "decoder_timing_weighted_loss": (
                    float((self._timing_decoder_loss_weight * decoder_timing_loss).detach().float().item())
                    if timing_objectives_enabled else 0.0
                ),
                "timing_counterfactual_loss": float(timing_counterfactual_loss.detach().float().item()),
                "timing_shifted_flow_loss": float(shifted_flow_error.detach().float().item()),
                "timing_correct_flow_loss": float(correct_flow_error.detach().float().mean().item()),
                "timing_attention_local": float(self._timing_attention_prior == "local_monotonic"),
                "timing_predictor_loss": float(predictor_loss.detach().float().item()),
                "timing_expressivity_loss": float(expressivity_loss.detach().float().item()),
                "timing_condition_active": float(timing_hidden_states is not None),
                "timing_condition_dropped": float(drop_timing),
                "timing_signal_rms": (
                    float(timing_hidden_states.detach().float().square().mean().sqrt().item())
                    if timing_hidden_states is not None
                    else 0.0
                ),
                "v5_teacher_loss": float(teacher_loss.detach().float().item()),
                "parent_preservation_loss": float(teacher_loss.detach().float().item()),
                "f0_loss": float(f0_loss.detach().float().item()),
                "batch_role_short": float(short_clarity_batch),
                "batch_role_full": float(full_timing_batch),
                "v5_decoded_identity_loss": float(decoded_identity_loss.detach().float().item()),
                "v5_supcon_loss": float(v5_supcon_loss.detach().float().item()),
                "v5_global_gate_mean": float(torch.stack([m.gate.detach().float() for m in self.v5_block_adapters.values()]).mean().item()) if self.v5_block_adapters else 0.0,
                "v5_fragment_gate": float(self.v5_fragment_attention.gate.detach().float().item()) if self.v5_fragment_attention is not None else 0.0,
                **self._identity_v5_gradient_diagnostics,
                "speaker_loss": float(speaker_loss.detach().float().item()),
                "contrastive_identity_loss": float(contrastive_identity_loss.detach().float().item()),
                "verifier_aux_loss": float(verifier_aux_loss.detach().float().item()),
                "latent_to_wavlm_cosine_loss": float(latent_to_wavlm_cosine_loss.detach().float().item()),
                "latent_to_ecapa_cosine_loss": float(latent_to_ecapa_cosine_loss.detach().float().item()),
                "latent_to_spectral_formant_loss": float(latent_to_spectral_formant_loss.detach().float().item()),
                "latent_to_pitch_style_loss": float(latent_to_pitch_style_loss.detach().float().item()),
            }

        # fp32 for stable backward
        total_loss = total_loss.float()
        if record_loss:
            self.training_losses.append(total_loss.item())
        return total_loss
