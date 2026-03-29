"""
FixedLoRAModule -- Corrected adapter training step for ACE-Step V2.

This module contains the ``FixedLoRAModule`` (nn.Module) responsible for
the per-step training logic: CFG dropout, logit-normal timestep sampling,
flow-matching interpolation, and the decoder forward pass.

Also includes small device/dtype/precision helpers used by both the
Fabric and basic training loops.
"""

from __future__ import annotations

import logging
import math
import re
from contextlib import nullcontext
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
from acestep.training_v2.performance_timing_predictor import (
    PerformanceTimingPredictor,
    PerformanceTimingPredictorConfig,
)
from acestep.training_v2.release_targets import DecoderExpressivitySupervisor
from acestep.training_v2.timestep_sampling import apply_cfg_dropout, sample_timesteps
from acestep.training_v2.timing_conditioning import (
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
        self._voice_condition_scale = float(getattr(training_config, "voice_condition_scale", 1.0))
        self._use_mert_conditioning = bool(getattr(training_config, "use_mert_conditioning", False))

        self.voice_conditioner: PrecomputedMERTConditioner | None = None
        if self._use_mert_conditioning:
            output_dim = int(getattr(model.config, "hidden_size", 2048))
            self.voice_conditioner = PrecomputedMERTConditioner(
                input_dim=int(getattr(training_config, "mert_hidden_size", 1024)),
                output_dim=output_dim,
                num_layers=int(getattr(training_config, "mert_num_layers", 25)),
                dropout=float(getattr(training_config, "voice_condition_dropout", 0.1)),
                use_layer_aggregation=True,
            ).to(self.device)
            logger.info(
                "[OK] MERT conditioning enabled: input_dim=%d, layers=%d, output_dim=%d",
                int(getattr(training_config, "mert_hidden_size", 1024)),
                int(getattr(training_config, "mert_num_layers", 25)),
                output_dim,
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
        self._enable_timing_consumer_training = bool(
            getattr(training_config, "enable_timing_consumer_training", True)
        )
        self._enable_phrase_modulation = bool(
            getattr(training_config, "enable_phrase_modulation", False)
        )
        self._timing_train_last_n_layers = max(
            0, int(getattr(training_config, "timing_train_last_n_layers", 0))
        )
        self._enable_timing_predictor = bool(
            getattr(training_config, "enable_timing_predictor", False)
        )
        self._timing_condition_source = str(
            getattr(training_config, "timing_condition_source", "sidecar")
        ).strip().lower()
        if self._timing_condition_source not in {"sidecar", "predictor", "hybrid"}:
            self._timing_condition_source = "sidecar"
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
        if not self._enable_timing_predictor and self._timing_condition_source != "sidecar":
            logger.warning(
                "[WARN] timing_condition_source=%s requested without enable_timing_predictor; falling back to sidecar",
                self._timing_condition_source,
            )
            self._timing_condition_source = "sidecar"
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
        if self._use_timing_branch:
            dit_dim = int(getattr(model.config, "hidden_size", 2048))
            timing_output_dim = int(getattr(training_config, "timing_output_dim", 0)) or dit_dim
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
                enable_phrase_modulation=self._enable_phrase_modulation,
                global_condition_scale=float(
                    getattr(training_config, "timing_global_condition_scale", 1.0)
                ),
            )
            self.timing_encoder = TimingEncoder(timing_cfg).to(self.device)
            decoder_state_dim = int(getattr(model.config, "audio_acoustic_hidden_dim", 64))
            self.decoder_timing_supervisor = DecoderTimingSupervisor(
                input_dim=decoder_state_dim,
                hidden_size=timing_cfg.hidden_size,
                dropout=float(getattr(training_config, "timing_dropout", 0.1)),
            ).to(self.device)
            logger.info(
                "[OK] Timing branch enabled: hidden=%d, heads=%d, layers=%d, output_dim=%d, decoder_supervision=%d",
                timing_cfg.hidden_size,
                timing_cfg.num_heads,
                timing_cfg.num_layers,
                timing_output_dim,
                decoder_state_dim,
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
            logger.info(
                "[OK] Decoder timing-attention params unfrozen: %s",
                f"{decoder_timing_params:,}",
            )

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
        """Unfreeze the lightweight decoder-side timing controls for Phase D."""
        decoder = self.model.decoder
        while hasattr(decoder, "_forward_module"):
            decoder = decoder._forward_module

        selected_layers = self._selected_timing_layers(decoder)
        enabled_params = 0
        for name, param in decoder.named_parameters():
            if "timing_" not in name:
                continue
            if (
                selected_layers is not None
                and not self._timing_param_in_layers(name, selected_layers)
            ):
                param.requires_grad = False
                continue
            should_train = name.endswith("timing_attn_gate")
            if self._enable_timing_consumer_training:
                is_timing_adapter = (
                    ".timing_cross_attn." in name
                    and (
                        "lora_" in name or "lokr_" in name or "hada_" in name
                    )
                    and any(
                        f".{module_name}." in name
                        for module_name in self._timing_consumer_adapter_modules
                    )
                )
                should_train = (
                    should_train
                    or is_timing_adapter
                    or ".timing_cross_attn_norm." in name
                    or ".timing_global_" in name
                )
            param.requires_grad = should_train
            if should_train:
                enabled_params += param.numel()
        return enabled_params

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
            layer.timing_attn_gate.data.fill_(-4.0)
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

            # ---- CFG dropout (CORRECTED -- missing in original trainer) ----
            if self._null_cond_emb is not None and self._cfg_ratio > 0.0:
                encoder_hidden_states = apply_cfg_dropout(
                    encoder_hidden_states,
                    self._null_cond_emb,
                    cfg_ratio=self._cfg_ratio,
                )

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
                encoder_hidden_states = torch.cat(
                    [encoder_hidden_states, voice_hidden_states], dim=1
                )
                encoder_attention_mask = torch.cat(
                    [encoder_attention_mask, voice_attention_mask], dim=1
                )

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
            energy_targets = batch.get("energy_targets")
            terminal_decay = batch.get("terminal_decay")
            cv_ratio_targets = batch.get("cv_ratio_targets")
            alignment_confidence = batch.get("alignment_confidence")
            beat_confidence = batch.get("beat_confidence")
            release_targets = batch.get("release_targets")
            timing_event_starts = batch.get("timing_event_starts")
            timing_event_ends = batch.get("timing_event_ends")
            timing_audio_durations = batch.get("timing_audio_duration")
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

                # Apply conditioning dropout (for robustness without timing)
                drop_timing = (
                    self._timing_condition_dropout > 0.0
                    and torch.rand(1).item() < self._timing_condition_dropout
                )
                if not drop_timing:
                    timing_out = None
                    predictor_out = None
                    if self._timing_condition_source in {"sidecar", "hybrid"} and timing_tokens is not None:
                        timing_out = self.timing_encoder(
                            timing_tokens=timing_tokens,
                            timing_mask=timing_mask,
                            timing_targets=timing_targets,
                            phrase_features=phrase_features,
                            global_features=global_phrase_features,
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
            )

            # ---- Flow matching loss ----------------------------------------
            flow = x1 - x0
            diffusion_loss = F.mse_loss(decoder_outputs[0], flow)

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
            total_loss = diffusion_loss

            # Timing branch auxiliary loss (Phase D)
            if self._use_timing_branch and self._timing_loss_weight > 0.0:
                total_loss = total_loss + self._timing_loss_weight * timing_loss
            if (
                self._use_timing_branch
                and self.performance_timing_predictor is not None
                and self._timing_predictor_loss_weight > 0.0
            ):
                total_loss = total_loss + self._timing_predictor_loss_weight * predictor_loss
            if self._use_timing_branch and self._timing_decoder_loss_weight > 0.0:
                total_loss = total_loss + self._timing_decoder_loss_weight * decoder_timing_loss
            if self._use_timing_branch and self._expressivity_loss_weight > 0.0:
                total_loss = total_loss + self._expressivity_loss_weight * expressivity_loss

            if self._f0_loss_weight > 0.0:
                pred_contour = _latent_f0_proxy_contour(decoder_outputs[0])
                target_contour = _latent_f0_proxy_contour(flow)
                f0_loss = F.l1_loss(pred_contour, target_contour)
                total_loss = total_loss + self._f0_loss_weight * f0_loss

            if self._speaker_loss_weight > 0.0:
                # flow = x1 - x0 => x0_hat = x1 - pred_flow
                pred_x0 = x1 - decoder_outputs[0]
                target_x0 = x0
                pred_emb = _speaker_proxy_embedding(pred_x0)
                target_emb = _speaker_proxy_embedding(target_x0)
                speaker_loss = (1.0 - F.cosine_similarity(pred_emb, target_emb, dim=-1)).mean()
                total_loss = total_loss + self._speaker_loss_weight * speaker_loss

        # fp32 for stable backward
        total_loss = total_loss.float()
        if record_loss:
            self.training_losses.append(total_loss.item())
        return total_loss
