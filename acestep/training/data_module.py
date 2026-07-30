"""
PyTorch Lightning DataModule for LoRA Training

Handles data loading and preprocessing for training ACE-Step LoRA adapters.
Supports both raw audio loading and preprocessed tensor loading.
"""

import os
import json
import random
from typing import Optional, List, Dict, Any, Tuple
from loguru import logger

from acestep.training.path_safety import safe_path

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

try:
    from lightning.pytorch import LightningDataModule
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    logger.warning("Lightning not installed. Training module will not be available.")
    # Create a dummy class for type hints
    class LightningDataModule:
        pass


# ============================================================================
# Preprocessed Tensor Dataset (Recommended for Training)
# ============================================================================

class PreprocessedTensorDataset(Dataset):
    """Dataset that loads preprocessed tensor files.
    
    This is the recommended dataset for training as all tensors are pre-computed:
    - target_latents: VAE-encoded audio [T, 64]
    - encoder_hidden_states: Condition encoder output [L, D]
    - encoder_attention_mask: Condition mask [L]
    - context_latents: Source context [T, 65]
    - attention_mask: Audio latent mask [T]
    
    No VAE/text encoder needed during training - just load tensors directly!
    """
    
    def __init__(self, tensor_dir: str, timing_dir: Optional[str] = None, identity_sidecar_dir: Optional[str] = None, strict_timing_sidecars: bool = False):
        """Initialize from a directory of preprocessed .pt files.

        Args:
            tensor_dir: Directory containing preprocessed .pt files and manifest.json
            timing_dir: Optional directory containing .timing.pt sidecar files produced
                        by extract_timing_features.py (Phase D). When provided, timing
                        features are loaded and returned in each sample dict.
            identity_sidecar_dir: Optional directory containing .identity.pt sidecars
                        with frozen Phase B identity targets. This is separate from
                        timing and does not enable the timing branch.

        Raises:
            ValueError: If tensor_dir is not an existing directory or escapes safe root.
        """
        validated_dir = safe_path(tensor_dir)
        if not os.path.isdir(validated_dir):
            raise ValueError(f"Not an existing directory: {tensor_dir}")
        self.tensor_dir = validated_dir
        # Phase D: optional timing sidecar directory
        self.timing_dir: Optional[str] = timing_dir
        self.strict_timing_sidecars = bool(strict_timing_sidecars)
        # Phase B: optional frozen identity target sidecars.
        self.identity_sidecar_dir: Optional[str] = identity_sidecar_dir
        self.sample_paths: List[str] = []
        
        # Load manifest if exists
        manifest_path = safe_path("manifest.json", base=self.tensor_dir)
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            raw_paths = manifest.get("samples", [])
            for raw in raw_paths:
                resolved = self._resolve_manifest_path(raw)
                if resolved is not None:
                    self.sample_paths.append(resolved)
        else:
            # Fallback: scan directory for .pt files (already inside tensor_dir)
            for f in os.listdir(self.tensor_dir):
                if f.endswith('.pt') and f != "manifest.json":
                    self.sample_paths.append(
                        safe_path(f, base=self.tensor_dir)
                    )
        
        # Validate paths exist on disk
        self.valid_paths = [p for p in self.sample_paths if os.path.exists(p)]
        
        if len(self.valid_paths) != len(self.sample_paths):
            logger.warning(
                f"Some tensor files not found: "
                f"{len(self.sample_paths) - len(self.valid_paths)} missing"
            )
        
        logger.info(
            f"PreprocessedTensorDataset: {len(self.valid_paths)} samples "
            f"from {self.tensor_dir}"
        )
        self.timing_sidecar_report = self._audit_timing_sidecars() if self.timing_dir else None

    def _audit_timing_sidecars(self) -> Dict[str, Any]:
        report: Dict[str, Any] = {"tensor_count": len(self.valid_paths), "valid": 0, "invalid": []}
        for tensor_path in self.valid_paths:
            stem = os.path.splitext(os.path.basename(tensor_path))[0]
            parts = stem.rsplit("_", 1)
            if len(parts) == 2 and len(parts[1]) >= 8 and all(c in "0123456789abcdef" for c in parts[1]):
                stem = parts[0]
            sidecar = os.path.join(str(self.timing_dir), f"{stem}.timing.pt")
            try:
                value = torch.load(sidecar, map_location="cpu", weights_only=False)
                tokens = value.get("timing_tokens")
                mask = value.get("timing_mask")
                if not isinstance(tokens, torch.Tensor) or tokens.numel() == 0:
                    raise ValueError("timing_tokens missing or empty")
                if not isinstance(mask, torch.Tensor) or mask.numel() == 0 or not bool(mask.bool().any()):
                    raise ValueError("timing_mask missing, empty, or all false")
                report["valid"] += 1
            except Exception as exc:
                report["invalid"].append({"path": sidecar, "error": str(exc)})
        report["coverage"] = report["valid"] / max(1, report["tensor_count"])
        if self.strict_timing_sidecars and report["invalid"]:
            raise RuntimeError(f"strict timing-sidecar audit failed: {report}")
        return report
    
    def _resolve_manifest_path(self, raw: str) -> Optional[str]:
        """Resolve a single manifest sample path to a validated absolute path.

        Tries ``base=tensor_dir`` first (correct for new manifests that store
        paths relative to the tensor directory).  If the resulting path does
        not exist on disk, falls back to resolving against the global safe
        root (backward compat for legacy manifests that stored CWD-relative
        paths like ``./datasets/…/foo.pt``).

        Returns:
            Validated absolute path, or ``None`` if the path cannot be
            resolved safely.
        """
        # Primary: resolve relative to tensor_dir
        try:
            child = safe_path(raw, base=self.tensor_dir)
            if os.path.exists(child):
                return child
        except ValueError:
            pass

        # Legacy fallback: resolve relative to global safe root (CWD)
        try:
            child = safe_path(raw)
            if os.path.exists(child):
                logger.debug(
                    f"Resolved legacy manifest path via safe root: {raw}"
                )
                return child
        except ValueError:
            pass

        logger.warning(f"Skipping unresolvable manifest path: {raw}")
        return None

    def __len__(self) -> int:
        return len(self.valid_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a preprocessed tensor file.
        
        Returns:
            Dictionary containing all pre-computed tensors for training
        """
        tensor_path = self.valid_paths[idx]
        data = torch.load(tensor_path, map_location='cpu', weights_only=True)
        
        sample = {
            "target_latents": data["target_latents"],  # [T, 64]
            "attention_mask": data["attention_mask"],  # [T]
            "encoder_hidden_states": data["encoder_hidden_states"],  # [L, D]
            "encoder_attention_mask": data["encoder_attention_mask"],  # [L]
            "context_latents": data["context_latents"],  # [T, 65]
            "metadata": data.get("metadata", {}),
            "ref_voice_features": data.get("ref_voice_features"),
            "ref_voice_attention_mask": data.get("ref_voice_attention_mask"),
            "ref_voice_crop_metadata": data.get("ref_voice_crop_metadata"),
            "text_hidden_states": data.get("text_hidden_states"),
            "text_attention_mask": data.get("text_attention_mask"),
            "lyric_hidden_states": data.get("lyric_hidden_states"),
            "lyric_attention_mask": data.get("lyric_attention_mask"),
            # Phase D: timing features (None if not available)
            "timing_tokens": None,
            "timing_targets": None,
            "timing_mask": None,
            "predictor_inputs": None,
            "predictor_mask": None,
            "phrase_features": None,
            "global_phrase_features": None,
            "phrase_ids": None,
            "f0_targets": None,
            "energy_targets": None,
            "terminal_decay": None,
            "cv_ratio_targets": None,
            "alignment_confidence": None,
            "beat_confidence": None,
            "release_targets": None,
            "timing_event_starts": None,
            "timing_event_ends": None,
            "timing_audio_duration": None,
            "timing_metadata": None,
            # Phase B identity-only sidecar targets (None if not configured).
            "identity_wavlm_embedding": None,
            "identity_ecapa_embedding": None,
            "identity_spectral_formant": None,
            "identity_pitch_style": None,
            "identity_negative_wavlm_embeddings": None,
            "identity_negative_ecapa_embeddings": None,
            "identity_sidecar_metadata": None,
            "identity_v5_prototype": None,
            "identity_v5_negative_prototypes": None,
            "identity_v5_negative_singer_ids": None,
        }

        tensor_basename = os.path.splitext(os.path.basename(tensor_path))[0]
        base_stem = tensor_basename
        parts = tensor_basename.rsplit("_", 1)
        if len(parts) == 2 and len(parts[1]) >= 8 and all(c in "0123456789abcdef" for c in parts[1]):
            base_stem = parts[0]

        # Phase B: load identity sidecar if identity_sidecar_dir is configured.
        if self.identity_sidecar_dir is not None:
            identity_path = os.path.join(self.identity_sidecar_dir, f"{tensor_basename}.identity.pt")
            if not os.path.exists(identity_path):
                identity_path = os.path.join(self.identity_sidecar_dir, f"{base_stem}.identity.pt")
            if os.path.exists(identity_path):
                try:
                    idata = torch.load(identity_path, map_location="cpu", weights_only=False)
                    sample["identity_wavlm_embedding"] = idata.get("wavlm_embedding")
                    sample["identity_ecapa_embedding"] = idata.get("ecapa_embedding")
                    sample["identity_spectral_formant"] = idata.get("spectral_formant_vector")
                    sample["identity_pitch_style"] = idata.get("pitch_style_vector")
                    sample["identity_negative_wavlm_embeddings"] = idata.get("negative_wavlm_embeddings")
                    sample["identity_negative_ecapa_embeddings"] = idata.get("negative_ecapa_embeddings")
                    sample["identity_sidecar_metadata"] = idata.get("metadata")
                    sample["identity_v5_prototype"] = idata.get("v5_billy_prototype")
                    sample["identity_v5_negative_prototypes"] = idata.get("v5_negative_prototypes")
                    sample["identity_v5_negative_singer_ids"] = idata.get("v5_negative_singer_ids")
                except Exception as e:
                    logger.warning("Failed to load identity sidecar %s: %s", identity_path, e)

        # Phase D: load timing sidecar if timing_dir is configured
        if self.timing_dir is not None:
            # Derive sidecar filename from tensor filename
            timing_path = os.path.join(self.timing_dir, f"{base_stem}.timing.pt")
            if os.path.exists(timing_path):
                try:
                    tdata = torch.load(timing_path, map_location="cpu", weights_only=False)
                    sample["timing_tokens"] = tdata.get("timing_tokens")
                    sample["timing_targets"] = tdata.get("timing_targets")
                    sample["timing_mask"] = tdata.get("timing_mask")
                    sample["predictor_inputs"] = tdata.get("predictor_inputs")
                    sample["predictor_mask"] = tdata.get("predictor_mask")
                    sample["phrase_features"] = tdata.get("phrase_features")
                    sample["global_phrase_features"] = tdata.get("global_phrase_features")
                    sample["phrase_ids"] = tdata.get("phrase_ids")
                    sample["f0_targets"] = tdata.get("f0_targets")
                    sample["energy_targets"] = tdata.get("energy_targets")
                    sample["terminal_decay"] = tdata.get("terminal_decay")
                    sample["cv_ratio_targets"] = tdata.get("cv_ratio_targets")
                    sample["alignment_confidence"] = tdata.get("alignment_confidence")
                    sample["beat_confidence"] = tdata.get("beat_confidence")
                    sample["release_targets"] = tdata.get("release_targets")
                    sample["timing_event_starts"] = tdata.get("event_start_sec")
                    sample["timing_event_ends"] = tdata.get("event_end_sec")
                    sample["timing_audio_duration"] = tdata.get("audio_duration_sec")
                    sample["timing_metadata"] = tdata.get("timing_metadata")
                except Exception as e:
                    if self.strict_timing_sidecars:
                        raise RuntimeError(f"Failed to load timing sidecar {timing_path}: {e}") from e
                    logger.warning(
                        "Failed to load timing sidecar %s: %s", timing_path, e
                    )

        return sample


def collate_preprocessed_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for preprocessed tensor batches.
    
    Handles variable-length tensors by padding to the longest in the batch.
    
    Args:
        batch: List of sample dictionaries with pre-computed tensors
        
    Returns:
        Batched dictionary with all tensors stacked
    """
    # Get max lengths
    max_latent_len = max(s["target_latents"].shape[0] for s in batch)
    max_encoder_len = max(s["encoder_hidden_states"].shape[0] for s in batch)
    
    # Pad and stack tensors
    target_latents = []
    attention_masks = []
    encoder_hidden_states = []
    encoder_attention_masks = []
    context_latents = []
    any_ref_voice = any(
        sample.get("ref_voice_features") is not None and sample.get("ref_voice_attention_mask") is not None
        for sample in batch
    )
    keep_text_lyric_inputs = any_ref_voice and all(
        sample.get("text_hidden_states") is not None
        and sample.get("text_attention_mask") is not None
        and sample.get("lyric_hidden_states") is not None
        and sample.get("lyric_attention_mask") is not None
        for sample in batch
    )
    ref_voice_features = []
    ref_voice_attention_masks = []
    text_hidden_states_keep = []
    text_attention_masks_keep = []
    lyric_hidden_states_keep = []
    lyric_attention_masks_keep = []
    max_ref_len = 0
    ref_rank = None
    ref_crops = None
    ref_layers = None
    ref_dim = None
    max_text_len_keep = 0
    max_lyric_len_keep = 0

    if any_ref_voice:
        for sample in batch:
            rvf = sample.get("ref_voice_features")
            if rvf is None:
                continue
            if rvf.ndim == 4:
                ref_rank = 4
                ref_crops = rvf.shape[0]
                ref_layers = rvf.shape[1]
                max_ref_len = max(max_ref_len, rvf.shape[2])
                ref_dim = rvf.shape[3]
            elif rvf.ndim == 3:
                ref_rank = 3
                ref_layers = rvf.shape[0]
                max_ref_len = max(max_ref_len, rvf.shape[1])
                ref_dim = rvf.shape[2]
            elif rvf.ndim == 2:
                ref_rank = 2
                max_ref_len = max(max_ref_len, rvf.shape[0])
                ref_dim = rvf.shape[1]
            else:
                raise ValueError(f"Unsupported ref_voice_features shape: {tuple(rvf.shape)}")
        if keep_text_lyric_inputs:
            max_text_len_keep = max(sample["text_hidden_states"].shape[0] for sample in batch)
            max_lyric_len_keep = max(sample["lyric_hidden_states"].shape[0] for sample in batch)

    for sample in batch:
        # Pad target_latents [T, 64] -> [max_T, 64]
        tl = sample["target_latents"]
        if tl.shape[0] < max_latent_len:
            pad = tl.new_zeros(max_latent_len - tl.shape[0], tl.shape[1])
            tl = torch.cat([tl, pad], dim=0)
        target_latents.append(tl)
        
        # Pad attention_mask [T] -> [max_T]
        am = sample["attention_mask"]
        if am.shape[0] < max_latent_len:
            pad = am.new_zeros(max_latent_len - am.shape[0])
            am = torch.cat([am, pad], dim=0)
        attention_masks.append(am)
        
        # Pad context_latents [T, 65] -> [max_T, 65]
        cl = sample["context_latents"]
        if cl.shape[0] < max_latent_len:
            pad = cl.new_zeros(max_latent_len - cl.shape[0], cl.shape[1])
            cl = torch.cat([cl, pad], dim=0)
        context_latents.append(cl)
        
        # Pad encoder_hidden_states [L, D] -> [max_L, D]
        ehs = sample["encoder_hidden_states"]
        if ehs.shape[0] < max_encoder_len:
            pad = ehs.new_zeros(max_encoder_len - ehs.shape[0], ehs.shape[1])
            ehs = torch.cat([ehs, pad], dim=0)
        encoder_hidden_states.append(ehs)
        
        # Pad encoder_attention_mask [L] -> [max_L]
        eam = sample["encoder_attention_mask"]
        if eam.shape[0] < max_encoder_len:
            pad = eam.new_zeros(max_encoder_len - eam.shape[0])
            eam = torch.cat([eam, pad], dim=0)
        encoder_attention_masks.append(eam)

        if any_ref_voice:
            rvf = sample.get("ref_voice_features")
            rvm = sample.get("ref_voice_attention_mask")
            if rvf is None or rvm is None:
                if ref_rank == 4:
                    rvf = torch.zeros(ref_crops, ref_layers, max_ref_len, ref_dim, dtype=ehs.dtype)
                    rvm = torch.zeros(ref_crops, max_ref_len, dtype=eam.dtype)
                elif ref_rank == 3:
                    rvf = torch.zeros(ref_layers, max_ref_len, ref_dim, dtype=ehs.dtype)
                    rvm = torch.zeros(max_ref_len, dtype=eam.dtype)
                else:
                    rvf = torch.zeros(max_ref_len, ref_dim, dtype=ehs.dtype)
                    rvm = torch.zeros(max_ref_len, dtype=eam.dtype)
            else:
                if ref_rank == 4:
                    if rvf.shape[0] != ref_crops or rvf.shape[1] != ref_layers:
                        raise ValueError(f"Mismatched multi-crop MERT shape: {tuple(rvf.shape)}")
                    if rvf.shape[2] < max_ref_len:
                        pad = rvf.new_zeros(rvf.shape[0], rvf.shape[1], max_ref_len - rvf.shape[2], rvf.shape[3])
                        rvf = torch.cat([rvf, pad], dim=2)
                    if rvm.shape[-1] < max_ref_len:
                        pad = rvm.new_zeros(rvm.shape[0], max_ref_len - rvm.shape[-1])
                        rvm = torch.cat([rvm, pad], dim=-1)
                elif ref_rank == 3:
                    if rvf.shape[1] < max_ref_len:
                        pad = rvf.new_zeros(rvf.shape[0], max_ref_len - rvf.shape[1], rvf.shape[2])
                        rvf = torch.cat([rvf, pad], dim=1)
                    if rvm.shape[0] < max_ref_len:
                        pad = rvm.new_zeros(max_ref_len - rvm.shape[0])
                        rvm = torch.cat([rvm, pad], dim=0)
                else:
                    if rvf.shape[0] < max_ref_len:
                        pad = rvf.new_zeros(max_ref_len - rvf.shape[0], rvf.shape[1])
                        rvf = torch.cat([rvf, pad], dim=0)
                    if rvm.shape[0] < max_ref_len:
                        pad = rvm.new_zeros(max_ref_len - rvm.shape[0])
                        rvm = torch.cat([rvm, pad], dim=0)
            ref_voice_features.append(rvf)
            ref_voice_attention_masks.append(rvm)

            if keep_text_lyric_inputs:
                ths = sample["text_hidden_states"]
                tam = sample["text_attention_mask"]
                lhs = sample["lyric_hidden_states"]
                lam = sample["lyric_attention_mask"]

                if ths.shape[0] < max_text_len_keep:
                    pad = ths.new_zeros(max_text_len_keep - ths.shape[0], ths.shape[1])
                    ths = torch.cat([ths, pad], dim=0)
                if tam.shape[0] < max_text_len_keep:
                    pad = tam.new_zeros(max_text_len_keep - tam.shape[0])
                    tam = torch.cat([tam, pad], dim=0)

                if lhs.shape[0] < max_lyric_len_keep:
                    pad = lhs.new_zeros(max_lyric_len_keep - lhs.shape[0], lhs.shape[1])
                    lhs = torch.cat([lhs, pad], dim=0)
                if lam.shape[0] < max_lyric_len_keep:
                    pad = lam.new_zeros(max_lyric_len_keep - lam.shape[0])
                    lam = torch.cat([lam, pad], dim=0)

                text_hidden_states_keep.append(ths)
                text_attention_masks_keep.append(tam)
                lyric_hidden_states_keep.append(lhs)
                lyric_attention_masks_keep.append(lam)

    output = {
        "target_latents": torch.stack(target_latents),  # [B, T, 64]
        "attention_mask": torch.stack(attention_masks),  # [B, T]
        "encoder_hidden_states": torch.stack(encoder_hidden_states),  # [B, L, D]
        "encoder_attention_mask": torch.stack(encoder_attention_masks),  # [B, L]
        "context_latents": torch.stack(context_latents),  # [B, T, 65]
        "metadata": [s["metadata"] for s in batch],
    }
    if any_ref_voice:
        output["ref_voice_features"] = torch.stack(ref_voice_features)
        output["ref_voice_attention_mask"] = torch.stack(ref_voice_attention_masks)
        if any(s.get("ref_voice_crop_metadata") is not None for s in batch):
            output["ref_voice_crop_metadata"] = [s.get("ref_voice_crop_metadata") for s in batch]
        if keep_text_lyric_inputs:
            output["text_hidden_states"] = torch.stack(text_hidden_states_keep)
            output["text_attention_mask"] = torch.stack(text_attention_masks_keep)
            output["lyric_hidden_states"] = torch.stack(lyric_hidden_states_keep)
            output["lyric_attention_mask"] = torch.stack(lyric_attention_masks_keep)

    # Phase B: collate frozen identity target sidecars.
    identity_keys = (
        "identity_wavlm_embedding",
        "identity_ecapa_embedding",
        "identity_spectral_formant",
        "identity_pitch_style",
        "identity_negative_wavlm_embeddings",
        "identity_negative_ecapa_embeddings",
        "identity_v5_prototype",
        "identity_v5_negative_prototypes",
    )
    for key in identity_keys:
        first = next((s.get(key) for s in batch if s.get(key) is not None), None)
        if first is None:
            continue
        stacked = []
        for sample in batch:
            value = sample.get(key)
            if value is None:
                value = torch.zeros_like(first)
            stacked.append(value.to(torch.float32))
        output[key] = torch.stack(stacked)
    if any(s.get("identity_sidecar_metadata") is not None for s in batch):
        output["identity_sidecar_metadata"] = [s.get("identity_sidecar_metadata") for s in batch]
    if any(s.get("identity_v5_negative_singer_ids") is not None for s in batch):
        output["identity_v5_negative_singer_ids"] = [s.get("identity_v5_negative_singer_ids") for s in batch]

    # Phase E: collate timing, predictor, and expressivity sidecars.
    def _first_non_none(key: str):
        return next((s.get(key) for s in batch if s.get(key) is not None), None)

    def _event_count(sample: Dict) -> int:
        for key in (
            "timing_tokens",
            "predictor_inputs",
            "timing_targets",
            "phrase_features",
            "f0_targets",
            "timing_event_starts",
        ):
            value = sample.get(key)
            if value is not None:
                return int(value.shape[0])
        return 0

    any_event_sidecar = any(_event_count(s) > 0 for s in batch)
    if any_event_sidecar:
        max_words = max(_event_count(s) for s in batch)
        first_timing = _first_non_none("timing_tokens")
        first_targets = _first_non_none("timing_targets")
        first_predictor = _first_non_none("predictor_inputs")
        first_phrase = _first_non_none("phrase_features")
        first_global_phrase = _first_non_none("global_phrase_features")
        first_f0 = _first_non_none("f0_targets")
        first_energy = _first_non_none("energy_targets")
        first_terminal = _first_non_none("terminal_decay")
        first_cv = _first_non_none("cv_ratio_targets")
        n_features = int(first_timing.shape[1]) if first_timing is not None else 0
        n_targets = int(first_targets.shape[1]) if first_targets is not None else 0
        predictor_feature_dim = int(first_predictor.shape[1]) if first_predictor is not None else 0
        phrase_feature_dim = int(first_phrase.shape[1]) if first_phrase is not None else 0
        global_phrase_dim = int(first_global_phrase.shape[0]) if first_global_phrase is not None else 0
        f0_dim = int(first_f0.shape[1]) if first_f0 is not None else 0
        energy_dim = int(first_energy.shape[1]) if first_energy is not None else 0
        terminal_dim = int(first_terminal.shape[1]) if first_terminal is not None else 0
        cv_dim = int(first_cv.shape[1]) if first_cv is not None else 0

        tok_batch, tgt_batch, mask_batch = [], [], []
        predictor_batch, predictor_mask_batch = [], []
        phrase_batch, global_phrase_batch, phrase_id_batch = [], [], []
        f0_batch, energy_batch, terminal_batch, cv_batch = [], [], [], []
        align_conf_batch, beat_conf_batch, release_batch = [], [], []
        event_start_batch, event_end_batch, audio_durations = [], [], []
        for s in batch:
            N = _event_count(s)
            pad = max_words - N
            if n_features > 0:
                tt = s.get("timing_tokens")
                if tt is None:
                    tt = torch.zeros(N, n_features, dtype=torch.long)
                tok_batch.append(torch.cat([tt, torch.zeros(pad, n_features, dtype=tt.dtype)], dim=0))
            if n_targets > 0:
                tm = s.get("timing_targets")
                if tm is None:
                    tm = torch.zeros(N, n_targets, dtype=torch.float32)
                tgt_batch.append(torch.cat([tm, torch.zeros(pad, n_targets, dtype=tm.dtype)], dim=0))

            timing_mask = s.get("timing_mask")
            predictor_mask = s.get("predictor_mask")
            if timing_mask is None:
                timing_mask = predictor_mask
            if timing_mask is None:
                timing_mask = torch.ones(N, dtype=torch.bool)
            mask_batch.append(torch.cat([timing_mask, torch.zeros(pad, dtype=torch.bool)], dim=0))

            if predictor_feature_dim > 0:
                pi = s.get("predictor_inputs")
                if pi is None:
                    pi = torch.zeros(N, predictor_feature_dim, dtype=torch.long)
                predictor_batch.append(torch.cat([pi, torch.zeros(pad, predictor_feature_dim, dtype=pi.dtype)], dim=0))
                pm = predictor_mask if predictor_mask is not None else timing_mask[:N]
                predictor_mask_batch.append(torch.cat([pm, torch.zeros(pad, dtype=torch.bool)], dim=0))

            if phrase_feature_dim > 0:
                pf = s.get("phrase_features")
                if pf is None:
                    pf = torch.zeros(N, phrase_feature_dim, dtype=torch.float32)
                phrase_batch.append(torch.cat([pf, torch.zeros(pad, phrase_feature_dim, dtype=pf.dtype)], dim=0))
                phrase_ids = s.get("phrase_ids")
                if phrase_ids is None:
                    phrase_ids = torch.zeros(N, dtype=torch.long)
                phrase_id_batch.append(torch.cat([phrase_ids, torch.zeros(pad, dtype=phrase_ids.dtype)], dim=0))
            if global_phrase_dim > 0:
                gpf = s.get("global_phrase_features")
                if gpf is None:
                    gpf = torch.zeros(global_phrase_dim, dtype=torch.float32)
                global_phrase_batch.append(gpf.to(torch.float32))

            for key, dim, stack in (
                ("f0_targets", f0_dim, f0_batch),
                ("energy_targets", energy_dim, energy_batch),
                ("terminal_decay", terminal_dim, terminal_batch),
                ("cv_ratio_targets", cv_dim, cv_batch),
            ):
                if dim <= 0:
                    continue
                tensor = s.get(key)
                if tensor is None:
                    tensor = torch.zeros(N, dim, dtype=torch.float32)
                stack.append(torch.cat([tensor, torch.zeros(pad, dim, dtype=tensor.dtype)], dim=0))

            for key, stack, dtype in (
                ("alignment_confidence", align_conf_batch, torch.float32),
                ("beat_confidence", beat_conf_batch, torch.float32),
                ("release_targets", release_batch, torch.long),
            ):
                tensor = s.get(key)
                if tensor is None:
                    tensor = torch.zeros(N, dtype=dtype)
                stack.append(torch.cat([tensor, torch.zeros(pad, dtype=tensor.dtype)], dim=0))

            ev_start = s.get("timing_event_starts")
            if ev_start is None:
                ev_start = torch.zeros(N, dtype=torch.float32)
            ev_end = s.get("timing_event_ends")
            if ev_end is None:
                ev_end = torch.zeros(N, dtype=torch.float32)
            event_start_batch.append(torch.cat([ev_start, torch.zeros(pad, dtype=ev_start.dtype)], dim=0))
            event_end_batch.append(torch.cat([ev_end, torch.zeros(pad, dtype=ev_end.dtype)], dim=0))

            audio_dur = s.get("timing_audio_duration")
            if audio_dur is None:
                audio_dur = torch.tensor(0.0, dtype=torch.float32)
            elif not isinstance(audio_dur, torch.Tensor):
                audio_dur = torch.tensor(float(audio_dur), dtype=torch.float32)
            audio_durations.append(audio_dur.reshape(()).to(torch.float32))

        if n_features > 0:
            output["timing_tokens"] = torch.stack(tok_batch)
        if n_targets > 0:
            output["timing_targets"] = torch.stack(tgt_batch)
        output["timing_mask"] = torch.stack(mask_batch)
        if predictor_feature_dim > 0:
            output["predictor_inputs"] = torch.stack(predictor_batch)
            output["predictor_mask"] = torch.stack(predictor_mask_batch)
        if phrase_feature_dim > 0:
            output["phrase_features"] = torch.stack(phrase_batch)
            output["phrase_ids"] = torch.stack(phrase_id_batch)
        if global_phrase_dim > 0:
            output["global_phrase_features"] = torch.stack(global_phrase_batch)
        if f0_dim > 0:
            output["f0_targets"] = torch.stack(f0_batch)
        if energy_dim > 0:
            output["energy_targets"] = torch.stack(energy_batch)
        if terminal_dim > 0:
            output["terminal_decay"] = torch.stack(terminal_batch)
        if cv_dim > 0:
            output["cv_ratio_targets"] = torch.stack(cv_batch)
        if align_conf_batch:
            output["alignment_confidence"] = torch.stack(align_conf_batch)
        if beat_conf_batch:
            output["beat_confidence"] = torch.stack(beat_conf_batch)
        if release_batch:
            output["release_targets"] = torch.stack(release_batch)
        output["timing_event_starts"] = torch.stack(event_start_batch)
        output["timing_event_ends"] = torch.stack(event_end_batch)
        output["timing_audio_duration"] = torch.stack(audio_durations)
        output["timing_metadata"] = [s.get("timing_metadata") for s in batch]

    return output


class PreprocessedDataModule(LightningDataModule if LIGHTNING_AVAILABLE else object):
    """DataModule for preprocessed tensor files.
    
    This is the recommended DataModule for training. It loads pre-computed tensors
    directly without needing VAE, text encoder, or condition encoder at training time.
    """
    
    def __init__(
        self,
        tensor_dir: str,
        batch_size: int = 1,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        pin_memory_device: str = "",
        val_split: float = 0.0,
        timing_dir: Optional[str] = None,
        identity_sidecar_dir: Optional[str] = None,
        strict_timing_sidecars: bool = False,
    ):
        """Initialize the data module.

        Args:
            tensor_dir: Directory containing preprocessed .pt files
            batch_size: Training batch size
            num_workers: Number of data loading workers
            timing_dir: Optional directory with .timing.pt sidecars (Phase D)
            identity_sidecar_dir: Optional directory with .identity.pt sidecars (Phase B)
            pin_memory: Whether to pin memory for faster GPU transfer
            val_split: Fraction of data for validation (0 = no validation)
        """
        if LIGHTNING_AVAILABLE:
            super().__init__()
        
        self.tensor_dir = tensor_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.pin_memory_device = pin_memory_device
        self.val_split = val_split
        self.timing_dir = timing_dir  # Phase D
        self.identity_sidecar_dir = identity_sidecar_dir  # Phase B identity-only
        self.strict_timing_sidecars = bool(strict_timing_sidecars)

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        if stage == 'fit' or stage is None:
            # Create full dataset. Identity sidecars are independent of Phase D timing.
            full_dataset = PreprocessedTensorDataset(
                self.tensor_dir,
                timing_dir=self.timing_dir,
                identity_sidecar_dir=self.identity_sidecar_dir,
                strict_timing_sidecars=self.strict_timing_sidecars,
            )
            
            # Split if validation requested
            if self.val_split > 0 and len(full_dataset) > 1:
                n_val = max(1, int(len(full_dataset) * self.val_split))
                n_train = len(full_dataset) - n_val
                
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                    full_dataset, [n_train, n_val]
                )
            else:
                self.train_dataset = full_dataset
                self.val_dataset = None
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        prefetch_factor = None if self.num_workers == 0 else self.prefetch_factor
        persistent_workers = False if self.num_workers == 0 else self.persistent_workers
        kwargs = dict(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_preprocessed_batch,
            drop_last=False,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        if self.pin_memory_device:
            kwargs["pin_memory_device"] = self.pin_memory_device
        return DataLoader(**kwargs)
    
    def val_dataloader(self) -> Optional[DataLoader]:
        """Create validation dataloader."""
        if self.val_dataset is None:
            return None
        prefetch_factor = None if self.num_workers == 0 else self.prefetch_factor
        persistent_workers = False if self.num_workers == 0 else self.persistent_workers
        kwargs = dict(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_preprocessed_batch,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        if self.pin_memory_device:
            kwargs["pin_memory_device"] = self.pin_memory_device
        return DataLoader(**kwargs)


# ============================================================================
# Raw Audio Dataset (Legacy - for backward compatibility)
# ============================================================================

class AceStepTrainingDataset(Dataset):
    """Dataset for ACE-Step LoRA training from raw audio.
    
    DEPRECATED: Use PreprocessedTensorDataset instead for better performance.
    
    Audio Format Requirements (handled automatically):
    - Sample rate: 48kHz (resampled if different)
    - Channels: Stereo (2 channels, mono is duplicated)
    - Max duration: 240 seconds (4 minutes)
    - Min duration: 5 seconds (padded if shorter)
    """
    
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        dit_handler,
        max_duration: float = 240.0,
        target_sample_rate: int = 48000,
    ):
        """Initialize the dataset."""
        self.samples = samples
        self.dit_handler = dit_handler
        self.max_duration = max_duration
        self.target_sample_rate = target_sample_rate
        
        self.valid_samples = self._validate_samples()
        logger.info(f"Dataset initialized with {len(self.valid_samples)} valid samples")
    
    def _validate_samples(self) -> List[Dict[str, Any]]:
        """Validate and filter samples, resolving audio paths to safe paths."""
        valid = []
        for i, sample in enumerate(self.samples):
            audio_path = sample.get("audio_path", "")
            if not audio_path:
                logger.warning(f"Sample {i}: Missing audio_path")
                continue

            try:
                validated = safe_path(audio_path)
            except ValueError:
                logger.warning(f"Sample {i}: Rejected unsafe path: {audio_path}")
                continue

            if not os.path.isfile(validated):
                logger.warning(f"Sample {i}: Audio file not found: {audio_path}")
                continue
            
            if not sample.get("caption"):
                logger.warning(f"Sample {i}: Missing caption")
                continue
            
            # Store validated path so downstream code never uses raw user input
            sample = {**sample, "audio_path": validated}
            valid.append(sample)
        
        return valid
    
    def __len__(self) -> int:
        return len(self.valid_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample."""
        sample = self.valid_samples[idx]
        
        audio_path = sample["audio_path"]
        audio, sr = torchaudio.load(audio_path)
        
        # Resample to 48kHz
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            audio = resampler(audio)
        
        # Convert to stereo
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2, :]
        
        # Truncate/pad
        max_samples = int(self.max_duration * self.target_sample_rate)
        if audio.shape[1] > max_samples:
            audio = audio[:, :max_samples]
        
        min_samples = int(5.0 * self.target_sample_rate)
        if audio.shape[1] < min_samples:
            padding = min_samples - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, padding))
        
        return {
            "audio": audio,
            "caption": sample.get("caption", ""),
            "lyrics": sample.get("lyrics", "[Instrumental]"),
            "metadata": {
                "caption": sample.get("caption", ""),
                "lyrics": sample.get("lyrics", "[Instrumental]"),
                "bpm": sample.get("bpm"),
                "keyscale": sample.get("keyscale", ""),
                "timesignature": sample.get("timesignature", ""),
                "duration": sample.get("duration", audio.shape[1] / self.target_sample_rate),
                "language": sample.get("language", "unknown"),
                "is_instrumental": sample.get("is_instrumental", True),
            },
            "audio_path": audio_path,
        }


def collate_training_batch(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for raw audio batches (legacy)."""
    max_len = max(sample["audio"].shape[1] for sample in batch)
    
    padded_audio = []
    attention_masks = []
    
    for sample in batch:
        audio = sample["audio"]
        audio_len = audio.shape[1]
        
        if audio_len < max_len:
            padding = max_len - audio_len
            audio = torch.nn.functional.pad(audio, (0, padding))
        
        padded_audio.append(audio)
        
        mask = torch.ones(max_len)
        if audio_len < max_len:
            mask[audio_len:] = 0
        attention_masks.append(mask)
    
    return {
        "audio": torch.stack(padded_audio),
        "attention_mask": torch.stack(attention_masks),
        "captions": [s["caption"] for s in batch],
        "lyrics": [s["lyrics"] for s in batch],
        "metadata": [s["metadata"] for s in batch],
        "audio_paths": [s["audio_path"] for s in batch],
    }


class AceStepDataModule(LightningDataModule if LIGHTNING_AVAILABLE else object):
    """DataModule for raw audio loading (legacy).
    
    DEPRECATED: Use PreprocessedDataModule for better training performance.
    """
    
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        dit_handler,
        batch_size: int = 1,
        num_workers: int = 4,
        pin_memory: bool = True,
        max_duration: float = 240.0,
        val_split: float = 0.0,
    ):
        if LIGHTNING_AVAILABLE:
            super().__init__()
        
        self.samples = samples
        self.dit_handler = dit_handler
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_duration = max_duration
        self.val_split = val_split
        
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            if self.val_split > 0 and len(self.samples) > 1:
                n_val = max(1, int(len(self.samples) * self.val_split))
                
                indices = list(range(len(self.samples)))
                random.shuffle(indices)
                
                val_indices = indices[:n_val]
                train_indices = indices[n_val:]
                
                train_samples = [self.samples[i] for i in train_indices]
                val_samples = [self.samples[i] for i in val_indices]
                
                self.train_dataset = AceStepTrainingDataset(
                    train_samples, self.dit_handler, self.max_duration
                )
                self.val_dataset = AceStepTrainingDataset(
                    val_samples, self.dit_handler, self.max_duration
                )
            else:
                self.train_dataset = AceStepTrainingDataset(
                    self.samples, self.dit_handler, self.max_duration
                )
                self.val_dataset = None
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_training_batch,
            drop_last=True,
        )
    
    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_training_batch,
        )


def load_dataset_from_json(json_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load a dataset from JSON file.

    Args:
        json_path: Path to the JSON dataset file.

    Returns:
        Tuple of (samples list, metadata dict).

    Raises:
        ValueError: If json_path does not point to an existing file or escapes safe root.
    """
    validated = safe_path(json_path)
    if not os.path.isfile(validated):
        raise ValueError(f"Dataset JSON file not found: {json_path}")

    with open(validated, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metadata = data.get("metadata", {})
    samples = data.get("samples", [])
    
    return samples, metadata
