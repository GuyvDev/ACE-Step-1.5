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
    
    def __init__(self, tensor_dir: str, timing_dir: Optional[str] = None):
        """Initialize from a directory of preprocessed .pt files.

        Args:
            tensor_dir: Directory containing preprocessed .pt files and manifest.json
            timing_dir: Optional directory containing .timing.pt sidecar files produced
                        by extract_timing_features.py (Phase D). When provided, timing
                        features are loaded and returned in each sample dict.

        Raises:
            ValueError: If tensor_dir is not an existing directory or escapes safe root.
        """
        validated_dir = safe_path(tensor_dir)
        if not os.path.isdir(validated_dir):
            raise ValueError(f"Not an existing directory: {tensor_dir}")
        self.tensor_dir = validated_dir
        # Phase D: optional timing sidecar directory
        self.timing_dir: Optional[str] = timing_dir
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
            "text_hidden_states": data.get("text_hidden_states"),
            "text_attention_mask": data.get("text_attention_mask"),
            "lyric_hidden_states": data.get("lyric_hidden_states"),
            "lyric_attention_mask": data.get("lyric_attention_mask"),
            # Phase D: timing features (None if not available)
            "timing_tokens": None,
            "timing_targets": None,
            "timing_mask": None,
            "phrase_features": None,
            "global_phrase_features": None,
            "phrase_ids": None,
            "timing_event_starts": None,
            "timing_event_ends": None,
            "timing_audio_duration": None,
            "timing_metadata": None,
        }

        # Phase D: load timing sidecar if timing_dir is configured
        if self.timing_dir is not None:
            # Derive sidecar filename from tensor filename
            tensor_basename = os.path.splitext(os.path.basename(tensor_path))[0]
            # Strip hash suffix (_xxxxxxxx) if present (8 hex chars after last _)
            # e.g. 001_piano_man_s003_c40820d4be -> 001_piano_man_s003
            base_stem = tensor_basename
            parts = tensor_basename.rsplit("_", 1)
            if len(parts) == 2 and len(parts[1]) >= 8 and all(c in "0123456789abcdef" for c in parts[1]):
                base_stem = parts[0]
            timing_path = os.path.join(self.timing_dir, f"{base_stem}.timing.pt")
            if os.path.exists(timing_path):
                try:
                    tdata = torch.load(timing_path, map_location="cpu", weights_only=False)
                    sample["timing_tokens"] = tdata.get("timing_tokens")
                    sample["timing_targets"] = tdata.get("timing_targets")
                    sample["timing_mask"] = tdata.get("timing_mask")
                    sample["phrase_features"] = tdata.get("phrase_features")
                    sample["global_phrase_features"] = tdata.get("global_phrase_features")
                    sample["phrase_ids"] = tdata.get("phrase_ids")
                    sample["timing_event_starts"] = tdata.get("event_start_sec")
                    sample["timing_event_ends"] = tdata.get("event_end_sec")
                    sample["timing_audio_duration"] = tdata.get("audio_duration_sec")
                    sample["timing_metadata"] = tdata.get("timing_metadata")
                except Exception as e:
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
    ref_layers = None
    ref_dim = None
    max_text_len_keep = 0
    max_lyric_len_keep = 0

    if any_ref_voice:
        for sample in batch:
            rvf = sample.get("ref_voice_features")
            if rvf is None:
                continue
            if rvf.ndim == 3:
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
                if ref_rank == 3:
                    rvf = torch.zeros(ref_layers, max_ref_len, ref_dim, dtype=ehs.dtype)
                else:
                    rvf = torch.zeros(max_ref_len, ref_dim, dtype=ehs.dtype)
                rvm = torch.zeros(max_ref_len, dtype=eam.dtype)
            else:
                if ref_rank == 3:
                    if rvf.shape[1] < max_ref_len:
                        pad = rvf.new_zeros(rvf.shape[0], max_ref_len - rvf.shape[1], rvf.shape[2])
                        rvf = torch.cat([rvf, pad], dim=1)
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
        if keep_text_lyric_inputs:
            output["text_hidden_states"] = torch.stack(text_hidden_states_keep)
            output["text_attention_mask"] = torch.stack(text_attention_masks_keep)
            output["lyric_hidden_states"] = torch.stack(lyric_hidden_states_keep)
            output["lyric_attention_mask"] = torch.stack(lyric_attention_masks_keep)

    # Phase D: collate timing features (pad to longest word sequence in batch)
    any_timing = any(s.get("timing_tokens") is not None for s in batch)
    if any_timing:
        first_timing = next(s["timing_tokens"] for s in batch if s.get("timing_tokens") is not None)
        first_targets = next(
            (
                s.get("timing_targets")
                for s in batch
                if s.get("timing_targets") is not None
            ),
            None,
        )
        max_words = max(
            s["timing_tokens"].shape[0]
            for s in batch
            if s.get("timing_tokens") is not None
        )
        n_features = int(first_timing.shape[1])
        n_targets = int(first_targets.shape[1]) if first_targets is not None else 0
        first_phrase = next(
            (s.get("phrase_features") for s in batch if s.get("phrase_features") is not None),
            None,
        )
        first_global_phrase = next(
            (s.get("global_phrase_features") for s in batch if s.get("global_phrase_features") is not None),
            None,
        )
        phrase_feature_dim = int(first_phrase.shape[1]) if first_phrase is not None else 0
        global_phrase_dim = int(first_global_phrase.shape[0]) if first_global_phrase is not None else 0
        tok_batch, tgt_batch, mask_batch = [], [], []
        phrase_batch, global_phrase_batch, phrase_id_batch = [], [], []
        event_start_batch, event_end_batch, audio_durations = [], [], []
        for s in batch:
            tt = s.get("timing_tokens")
            if tt is None:
                # Sample has no timing data: pad with zeros
                tok_batch.append(torch.zeros(max_words, n_features, dtype=torch.long))
                if n_targets > 0:
                    tgt_batch.append(torch.zeros(max_words, n_targets, dtype=torch.float32))
                mask_batch.append(torch.zeros(max_words, dtype=torch.bool))
                if phrase_feature_dim > 0:
                    phrase_batch.append(torch.zeros(max_words, phrase_feature_dim, dtype=torch.float32))
                    phrase_id_batch.append(torch.zeros(max_words, dtype=torch.long))
                if global_phrase_dim > 0:
                    global_phrase_batch.append(torch.zeros(global_phrase_dim, dtype=torch.float32))
                event_start_batch.append(torch.zeros(max_words, dtype=torch.float32))
                event_end_batch.append(torch.zeros(max_words, dtype=torch.float32))
                audio_durations.append(torch.tensor(0.0, dtype=torch.float32))
            else:
                N = tt.shape[0]
                pad = max_words - N
                tok_batch.append(torch.cat([tt, torch.zeros(pad, n_features, dtype=torch.long)], dim=0))
                if n_targets > 0:
                    tm = s.get("timing_targets")
                    if tm is None:
                        tm = torch.zeros(N, n_targets, dtype=torch.float32)
                    tgt_batch.append(torch.cat([tm, torch.zeros(pad, n_targets, dtype=tm.dtype)], dim=0))
                msk = s.get("timing_mask", torch.ones(N, dtype=torch.bool))
                mask_batch.append(torch.cat([msk, torch.zeros(pad, dtype=torch.bool)], dim=0))
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
                ev_start = s.get("timing_event_starts")
                if ev_start is None:
                    ev_start = torch.zeros(N, dtype=torch.float32)
                ev_end = s.get("timing_event_ends")
                if ev_end is None:
                    ev_end = torch.zeros(N, dtype=torch.float32)
                event_start_batch.append(
                    torch.cat([ev_start, torch.zeros(pad, dtype=ev_start.dtype)], dim=0)
                )
                event_end_batch.append(
                    torch.cat([ev_end, torch.zeros(pad, dtype=ev_end.dtype)], dim=0)
                )
                audio_dur = s.get("timing_audio_duration")
                if audio_dur is None:
                    audio_dur = torch.tensor(0.0, dtype=torch.float32)
                elif not isinstance(audio_dur, torch.Tensor):
                    audio_dur = torch.tensor(float(audio_dur), dtype=torch.float32)
                audio_durations.append(audio_dur.reshape(()).to(torch.float32))
        output["timing_tokens"] = torch.stack(tok_batch)     # [B, N_words, 4]
        if n_targets > 0:
            output["timing_targets"] = torch.stack(tgt_batch)
        output["timing_mask"] = torch.stack(mask_batch)      # [B, N_words]
        if phrase_feature_dim > 0:
            output["phrase_features"] = torch.stack(phrase_batch)
            output["phrase_ids"] = torch.stack(phrase_id_batch)
        if global_phrase_dim > 0:
            output["global_phrase_features"] = torch.stack(global_phrase_batch)
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
    ):
        """Initialize the data module.

        Args:
            tensor_dir: Directory containing preprocessed .pt files
            batch_size: Training batch size
            num_workers: Number of data loading workers
            timing_dir: Optional directory with .timing.pt sidecars (Phase D)
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

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        if stage == 'fit' or stage is None:
            # Create full dataset (Phase D: pass timing_dir)
            full_dataset = PreprocessedTensorDataset(self.tensor_dir, timing_dir=self.timing_dir)
            
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
