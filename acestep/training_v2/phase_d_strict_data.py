"""Strict, deterministic data pipeline for the controlled Phase-D experiment.

This module intentionally has no missing-sidecar fallback. Every listed tensor and
its timing sidecar must exist, load, and satisfy the exact schema before training.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset, random_split

try:
    from lightning.pytorch import LightningDataModule
except ImportError:  # pragma: no cover - controlled trainer requires Lightning/Fabric
    class LightningDataModule:  # type: ignore[no-redef]
        pass

_HASH_SUFFIX = re.compile(r"_[0-9a-fA-F]{8,}$")
_CORE_KEYS = (
    "target_latents",
    "attention_mask",
    "encoder_hidden_states",
    "encoder_attention_mask",
    "context_latents",
)
_TIMING_KEYS = (
    "timing_tokens",
    "timing_targets",
    "timing_mask",
    "event_start_sec",
    "event_end_sec",
    "audio_duration_sec",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_finite(name: str, tensor: torch.Tensor, path: Path) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise RuntimeError(f"{name} contains non-finite values: {path}")


def _resolve_manifest_paths(tensor_dir: Path) -> List[Path]:
    manifest = tensor_dir / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"controlled dataset requires manifest.json: {manifest}")
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    samples = raw.get("samples") if isinstance(raw, dict) else None
    if not isinstance(samples, list) or not samples:
        raise RuntimeError(f"manifest samples must be a non-empty list: {manifest}")
    paths: List[Path] = []
    for item in samples:
        if not isinstance(item, str) or not item.strip():
            raise RuntimeError(f"manifest sample must be a non-empty string: {item!r}")
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = tensor_dir / candidate
        candidate = candidate.resolve()
        if not candidate.is_relative_to(tensor_dir):
            raise RuntimeError(f"manifest tensor escapes tensor directory: {candidate}")
        if not candidate.is_file():
            raise FileNotFoundError(f"manifest tensor is missing: {candidate}")
        if candidate.suffix != ".pt":
            raise RuntimeError(f"manifest tensor must be a .pt file: {candidate}")
        paths.append(candidate)
    if len(paths) != len(set(paths)):
        raise RuntimeError("dataset manifest contains duplicate tensor paths")
    return paths


def _timing_path(tensor_path: Path, timing_dir: Path) -> Path:
    stem = _HASH_SUFFIX.sub("", tensor_path.stem)
    return (timing_dir / f"{stem}.timing.pt").resolve()


def _validate_core(payload: Dict[str, Any], path: Path) -> None:
    for key in _CORE_KEYS:
        value = payload.get(key)
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"core tensor {key!r} missing from {path}")
        if value.numel() == 0:
            raise RuntimeError(f"core tensor {key!r} is empty in {path}")
        _require_finite(key, value, path)
    target = payload["target_latents"]
    context = payload["context_latents"]
    attention = payload["attention_mask"]
    encoder = payload["encoder_hidden_states"]
    encoder_mask = payload["encoder_attention_mask"]
    if target.ndim != 2 or target.shape[1] != 64:
        raise RuntimeError(f"target_latents must be [T,64], got {tuple(target.shape)}: {path}")
    if context.ndim != 2 or context.shape[0] != target.shape[0] or context.shape[1] != 128:
        raise RuntimeError(f"context_latents must be [T,128], got {tuple(context.shape)}: {path}")
    if attention.ndim != 1 or attention.shape[0] != target.shape[0]:
        raise RuntimeError(f"attention_mask shape mismatch in {path}: {tuple(attention.shape)}")
    if encoder.ndim != 2 or encoder_mask.ndim != 1 or encoder_mask.shape[0] != encoder.shape[0]:
        raise RuntimeError(f"encoder tensor/mask shape mismatch in {path}")
    if not bool(attention.bool().any()) or not bool(encoder_mask.bool().any()):
        raise RuntimeError(f"core attention mask is all false in {path}")


def _validate_timing(payload: Dict[str, Any], path: Path) -> None:
    for key in _TIMING_KEYS:
        if key not in payload:
            raise RuntimeError(f"timing key {key!r} missing from {path}")
    tokens = payload["timing_tokens"]
    targets = payload["timing_targets"]
    mask = payload["timing_mask"]
    starts = payload["event_start_sec"]
    ends = payload["event_end_sec"]
    duration = payload["audio_duration_sec"]
    for name, value in (
        ("timing_tokens", tokens),
        ("timing_targets", targets),
        ("timing_mask", mask),
        ("event_start_sec", starts),
        ("event_end_sec", ends),
    ):
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"{name} is not a tensor: {path}")
    if tokens.ndim != 2 or tokens.shape[0] <= 0 or tokens.shape[1] != 8:
        raise RuntimeError(f"timing_tokens must be non-empty [N,8]: {path}")
    if tokens.dtype != torch.long:
        raise RuntimeError(f"timing_tokens must use torch.long, got {tokens.dtype}: {path}")
    n_events = int(tokens.shape[0])
    if targets.ndim != 2 or targets.shape != (n_events, 8):
        raise RuntimeError(f"timing_targets must be [N,8], got {tuple(targets.shape)}: {path}")
    if mask.dtype != torch.bool or mask.ndim != 1 or mask.shape[0] != n_events or not bool(mask.any()):
        raise RuntimeError(f"timing_mask must be non-empty bool [N]: {path}")
    if starts.ndim != 1 or ends.ndim != 1 or starts.shape[0] != n_events or ends.shape[0] != n_events:
        raise RuntimeError(f"timing event span shape mismatch: {path}")
    if not isinstance(duration, torch.Tensor):
        try:
            duration = torch.tensor(float(duration), dtype=torch.float32)
        except Exception as exc:
            raise RuntimeError(f"audio_duration_sec is invalid: {path}") from exc
    duration = duration.reshape(()).float()
    for name, value in (
        ("timing_targets", targets.float()),
        ("event_start_sec", starts.float()),
        ("event_end_sec", ends.float()),
        ("audio_duration_sec", duration),
    ):
        _require_finite(name, value, path)
    valid = mask.bool()
    if bool((starts[valid] < 0).any()) or bool((ends[valid] < starts[valid]).any()):
        raise RuntimeError(f"invalid timing spans: {path}")
    if float(duration.item()) <= 0.0 or bool((ends[valid] > duration + 1e-3).any()):
        raise RuntimeError(f"timing spans exceed invalid audio duration: {path}")


def validate_timing_payload(payload: Dict[str, Any], path: str | Path) -> None:
    """Validate one timing payload with the exact controlled schema."""
    if not isinstance(payload, dict):
        raise RuntimeError(f"timing payload must be a dict: {path}")
    _validate_timing(payload, Path(path).resolve())



class StrictPhaseDTensorDataset(Dataset):
    """Dataset with mandatory, prevalidated timing sidecars."""

    def __init__(self, tensor_dir: str, timing_dir: str) -> None:
        self.tensor_dir = Path(tensor_dir).resolve()
        self.timing_dir = Path(timing_dir).resolve()
        if not self.tensor_dir.is_dir():
            raise FileNotFoundError(f"tensor directory missing: {self.tensor_dir}")
        if not self.timing_dir.is_dir():
            raise FileNotFoundError(f"timing directory missing: {self.timing_dir}")
        self.valid_paths = [str(path) for path in _resolve_manifest_paths(self.tensor_dir)]
        self._sidecars: List[Path] = []
        sidecar_records: List[Dict[str, Any]] = []
        optional_schema: Dict[str, tuple[bool, tuple[int, ...] | None]] | None = None
        for raw in self.valid_paths:
            tensor_path = Path(raw)
            timing_path = _timing_path(tensor_path, self.timing_dir)
            if not timing_path.is_file():
                raise FileNotFoundError(f"timing sidecar missing for {tensor_path}: {timing_path}")
            core = torch.load(tensor_path, map_location="cpu", weights_only=True)
            if not isinstance(core, dict):
                raise RuntimeError(f"tensor payload must be a dict: {tensor_path}")
            timing = torch.load(timing_path, map_location="cpu", weights_only=False)
            if not isinstance(timing, dict):
                raise RuntimeError(f"timing payload must be a dict: {timing_path}")
            _validate_core(core, tensor_path)
            _validate_timing(timing, timing_path)
            current_optional: Dict[str, tuple[bool, tuple[int, ...] | None]] = {}
            n_events = int(timing["timing_tokens"].shape[0])
            phrase_features = timing.get("phrase_features")
            if phrase_features is None:
                current_optional["phrase_features"] = (False, None)
            elif not isinstance(phrase_features, torch.Tensor):
                raise RuntimeError(f"optional timing field phrase_features is not a tensor: {timing_path}")
            else:
                if phrase_features.ndim != 2 or int(phrase_features.shape[0]) != n_events:
                    raise RuntimeError(
                        f"phrase_features must be [N,F] aligned to timing events, got {tuple(phrase_features.shape)}: {timing_path}"
                    )
                if not torch.is_floating_point(phrase_features) or not bool(torch.isfinite(phrase_features).all()):
                    raise RuntimeError(f"phrase_features must be finite floating point: {timing_path}")
                current_optional["phrase_features"] = (True, tuple(phrase_features.shape[1:]))

            global_phrase = timing.get("global_phrase_features")
            if global_phrase is None:
                current_optional["global_phrase_features"] = (False, None)
            elif not isinstance(global_phrase, torch.Tensor):
                raise RuntimeError(f"optional timing field global_phrase_features is not a tensor: {timing_path}")
            else:
                if global_phrase.ndim != 1 or global_phrase.numel() == 0:
                    raise RuntimeError(
                        f"global_phrase_features must be non-empty [F], got {tuple(global_phrase.shape)}: {timing_path}"
                    )
                if not torch.is_floating_point(global_phrase) or not bool(torch.isfinite(global_phrase).all()):
                    raise RuntimeError(f"global_phrase_features must be finite floating point: {timing_path}")
                current_optional["global_phrase_features"] = (True, tuple(global_phrase.shape))

            phrase_ids = timing.get("phrase_ids")
            if phrase_ids is None:
                current_optional["phrase_ids"] = (False, None)
            elif not isinstance(phrase_ids, torch.Tensor):
                raise RuntimeError(f"optional timing field phrase_ids is not a tensor: {timing_path}")
            else:
                if phrase_ids.ndim != 1 or int(phrase_ids.shape[0]) != n_events:
                    raise RuntimeError(
                        f"phrase_ids must be [N] aligned to timing events, got {tuple(phrase_ids.shape)}: {timing_path}"
                    )
                if phrase_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                    raise RuntimeError(f"phrase_ids must use an integer dtype, got {phrase_ids.dtype}: {timing_path}")
                current_optional["phrase_ids"] = (True, ())
            if optional_schema is None:
                optional_schema = current_optional
            else:
                for key, (present, shape) in current_optional.items():
                    expected_present, expected_shape = optional_schema[key]
                    if present != expected_present:
                        raise RuntimeError(f"optional timing field {key} is only partially present across dataset")
                    if present and shape != expected_shape:
                        raise RuntimeError(
                            f"optional timing field {key} schema changed: {shape} != {expected_shape}"
                        )
            self._sidecars.append(timing_path)
            sidecar_records.append({
                "tensor": str(tensor_path),
                "timing": str(timing_path),
                "tensor_sha256": _sha256_file(tensor_path),
                "timing_sha256": _sha256_file(timing_path),
                "events": int(timing["timing_tokens"].shape[0]),
            })
        self.timing_sidecar_report = {
            "strict": True,
            "tensor_count": len(self.valid_paths),
            "valid": len(self.valid_paths),
            "missing": 0,
            "invalid": 0,
            "coverage": 1.0,
            "optional_schema": {key: {"present": value[0], "shape": list(value[1]) if value[1] is not None else None} for key, value in (optional_schema or {}).items()},
            "records": sidecar_records,
        }
        self.sample_roles: Dict[int, str] = {}

    def install_sample_roles(self, role_manifest_path: str) -> Dict[str, int]:
        """Install exact D1 roles from the immutable union provenance manifest."""
        manifest_path = Path(role_manifest_path).resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"D1 role manifest is missing: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = payload.get("records") if isinstance(payload, dict) else None
        if not isinstance(records, list):
            raise RuntimeError(f"D1 role manifest has no records list: {manifest_path}")
        by_tensor: Dict[str, str] = {}
        source_to_role = {
            "short_aligned_v4_c0": "short_clarity",
            "full_song_phase_c": "full_timing",
        }
        for row in records:
            if not isinstance(row, dict):
                raise RuntimeError("D1 role manifest contains a non-object record")
            tensor = row.get("tensor")
            source_kind = row.get("source_kind")
            if not isinstance(tensor, str) or source_kind not in source_to_role:
                raise RuntimeError(f"invalid D1 role record: {row}")
            if tensor in by_tensor:
                raise RuntimeError(f"duplicate D1 role record: {tensor}")
            by_tensor[tensor] = source_to_role[source_kind]
        expected = {Path(path).name for path in self.valid_paths}
        if set(by_tensor) != expected:
            missing = sorted(expected - set(by_tensor))
            unexpected = sorted(set(by_tensor) - expected)
            raise RuntimeError(
                f"D1 role manifest mismatch: missing={missing[:8]} unexpected={unexpected[:8]}"
            )
        self.sample_roles = {
            index: by_tensor[Path(path).name]
            for index, path in enumerate(self.valid_paths)
        }
        counts = {
            role: sum(value == role for value in self.sample_roles.values())
            for role in ("short_clarity", "full_timing")
        }
        if counts != {"short_clarity": 125, "full_timing": 78}:
            raise RuntimeError(f"D1 full-dataset role counts mismatch: {counts}")
        return counts

    def __len__(self) -> int:
        return len(self.valid_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        tensor_path = Path(self.valid_paths[index])
        timing_path = self._sidecars[index]
        core = torch.load(tensor_path, map_location="cpu", weights_only=True)
        timing = torch.load(timing_path, map_location="cpu", weights_only=False)
        _validate_core(core, tensor_path)
        _validate_timing(timing, timing_path)
        metadata = dict(
            core.get("metadata") or {},
            tensor_path=str(tensor_path),
            timing_path=str(timing_path),
        )
        if self.sample_roles:
            role = self.sample_roles.get(index)
            if role not in {"short_clarity", "full_timing"}:
                raise RuntimeError(f"D1 sample role is unavailable for index {index}")
            metadata["phase_d_batch_role"] = role
        return {
            "target_latents": core["target_latents"],
            "attention_mask": core["attention_mask"],
            "encoder_hidden_states": core["encoder_hidden_states"],
            "encoder_attention_mask": core["encoder_attention_mask"],
            "context_latents": core["context_latents"],
            "metadata": metadata,
            "timing_tokens": timing["timing_tokens"],
            "timing_targets": timing["timing_targets"],
            "timing_mask": timing["timing_mask"].bool(),
            "phrase_features": timing.get("phrase_features"),
            "global_phrase_features": timing.get("global_phrase_features"),
            "phrase_ids": timing.get("phrase_ids"),
            "timing_event_starts": timing["event_start_sec"].float(),
            "timing_event_ends": timing["event_end_sec"].float(),
            "timing_audio_duration": torch.as_tensor(timing["audio_duration_sec"]).reshape(()).float(),
            "timing_metadata": timing.get("timing_metadata"),
        }


def _pad_2d(values: List[torch.Tensor], length: int) -> torch.Tensor:
    padded = []
    for value in values:
        if value.ndim != 2:
            raise RuntimeError(f"expected rank-2 tensor, got {tuple(value.shape)}")
        pad = length - value.shape[0]
        padded.append(torch.cat([value, value.new_zeros(pad, value.shape[1])], dim=0) if pad else value)
    return torch.stack(padded)


def _pad_1d(values: List[torch.Tensor], length: int) -> torch.Tensor:
    padded = []
    for value in values:
        if value.ndim != 1:
            raise RuntimeError(f"expected rank-1 tensor, got {tuple(value.shape)}")
        pad = length - value.shape[0]
        padded.append(torch.cat([value, value.new_zeros(pad)], dim=0) if pad else value)
    return torch.stack(padded)


def strict_phase_d_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        raise RuntimeError("cannot collate an empty batch")
    for index, sample in enumerate(batch):
        if sample.get("timing_tokens") is None or sample.get("timing_mask") is None:
            raise RuntimeError(f"timing data missing at collate index {index}")
    max_latent = max(int(item["target_latents"].shape[0]) for item in batch)
    max_encoder = max(int(item["encoder_hidden_states"].shape[0]) for item in batch)
    max_events = max(int(item["timing_tokens"].shape[0]) for item in batch)
    output: Dict[str, Any] = {
        "target_latents": _pad_2d([item["target_latents"] for item in batch], max_latent),
        "attention_mask": _pad_1d([item["attention_mask"] for item in batch], max_latent),
        "encoder_hidden_states": _pad_2d([item["encoder_hidden_states"] for item in batch], max_encoder),
        "encoder_attention_mask": _pad_1d([item["encoder_attention_mask"] for item in batch], max_encoder),
        "context_latents": _pad_2d([item["context_latents"] for item in batch], max_latent),
        "timing_tokens": _pad_2d([item["timing_tokens"] for item in batch], max_events),
        "timing_targets": _pad_2d([item["timing_targets"] for item in batch], max_events),
        "timing_mask": _pad_1d([item["timing_mask"].bool() for item in batch], max_events).bool(),
        "timing_event_starts": _pad_1d([item["timing_event_starts"] for item in batch], max_events),
        "timing_event_ends": _pad_1d([item["timing_event_ends"] for item in batch], max_events),
        "timing_audio_duration": torch.stack([item["timing_audio_duration"] for item in batch]),
        "metadata": [item["metadata"] for item in batch],
        "timing_metadata": [item.get("timing_metadata") for item in batch],
    }
    optional_2d = ("phrase_features",)
    optional_1d = ("phrase_ids",)
    for key in optional_2d:
        values = [item.get(key) for item in batch]
        if any(value is not None for value in values):
            if not all(isinstance(value, torch.Tensor) for value in values):
                raise RuntimeError(f"optional timing field {key} is only partially present")
            output[key] = _pad_2d(values, max_events)  # type: ignore[arg-type]
    for key in optional_1d:
        values = [item.get(key) for item in batch]
        if any(value is not None for value in values):
            if not all(isinstance(value, torch.Tensor) for value in values):
                raise RuntimeError(f"optional timing field {key} is only partially present")
            output[key] = _pad_1d(values, max_events)  # type: ignore[arg-type]
    global_values = [item.get("global_phrase_features") for item in batch]
    if any(value is not None for value in global_values):
        if not all(isinstance(value, torch.Tensor) for value in global_values):
            raise RuntimeError("global_phrase_features is only partially present")
        shapes = {tuple(value.shape) for value in global_values}  # type: ignore[union-attr]
        if len(shapes) != 1:
            raise RuntimeError(f"global_phrase_features shape mismatch: {sorted(shapes)}")
        output["global_phrase_features"] = torch.stack(global_values)  # type: ignore[arg-type]
    if not bool(output["timing_mask"].any()):
        raise RuntimeError("collated timing mask is all false")
    return output


class D1AlternatingRoleSampler(Sampler[int]):
    """Yield an exact short, short, full schedule over a deterministic split."""

    def __init__(self, dataset: Dataset[Any], *, seed: int) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        base = dataset
        local_to_full = list(range(len(dataset)))
        if isinstance(dataset, Subset):
            base = dataset.dataset
            local_to_full = [int(value) for value in dataset.indices]
        if not isinstance(base, StrictPhaseDTensorDataset) or not base.sample_roles:
            raise RuntimeError("D1 sampler requires strict dataset roles")
        self.role_indices: Dict[str, List[int]] = {
            "short_clarity": [],
            "full_timing": [],
        }
        for local_index, full_index in enumerate(local_to_full):
            role = base.sample_roles.get(full_index)
            if role not in self.role_indices:
                raise RuntimeError(f"D1 sampler found invalid role at index {full_index}: {role}")
            self.role_indices[role].append(local_index)
        if not all(self.role_indices.values()):
            raise RuntimeError(f"D1 train split lacks a required role: {self.role_counts}")

    @property
    def role_counts(self) -> Dict[str, int]:
        return {key: len(value) for key, value in self.role_indices.items()}

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        pools = {
            role: [indices[index] for index in torch.randperm(len(indices), generator=generator).tolist()]
            for role, indices in self.role_indices.items()
        }
        positions = {role: 0 for role in pools}
        pattern = ("short_clarity", "short_clarity", "full_timing")
        for offset in range(len(self)):
            role = pattern[offset % len(pattern)]
            if positions[role] >= len(pools[role]):
                source = self.role_indices[role]
                pools[role] = [
                    source[index]
                    for index in torch.randperm(len(source), generator=generator).tolist()
                ]
                positions[role] = 0
            yield pools[role][positions[role]]
            positions[role] += 1
        self.epoch += 1


class PreprocessedDataModule(LightningDataModule):
    """Drop-in strict data module for the controlled trainer."""

    def __init__(
        self,
        tensor_dir: str,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = True,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
        val_split: float = 0.0,
        timing_dir: Optional[str] = None,
        identity_sidecar_dir: Optional[str] = None,
        strict_timing_sidecars: bool = True,
        seed: int = 42,
        d1_alternating_roles: bool = False,
        role_manifest_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        if identity_sidecar_dir is not None:
            raise RuntimeError("identity sidecars are outside the controlled Phase-D contract")
        if not strict_timing_sidecars:
            raise RuntimeError("controlled Phase-D requires strict_timing_sidecars=True")
        if timing_dir is None:
            raise RuntimeError("controlled Phase-D requires a timing directory")
        if not 0.0 <= float(val_split) < 1.0:
            raise ValueError(f"val_split must be in [0,1), got {val_split}")
        self.tensor_dir = tensor_dir
        self.timing_dir = timing_dir
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = bool(persistent_workers) and self.num_workers > 0
        self.pin_memory_device = pin_memory_device
        self.val_split = float(val_split)
        self.seed = int(seed)
        self.d1_alternating_roles = bool(d1_alternating_roles)
        self.role_manifest_path = role_manifest_path
        self.train_dataset: Dataset[Any] | None = None
        self.val_dataset: Dataset[Any] | None = None
        self.full_dataset: StrictPhaseDTensorDataset | None = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage not in (None, "fit"):
            raise RuntimeError(f"unsupported controlled data stage: {stage}")
        full = StrictPhaseDTensorDataset(self.tensor_dir, self.timing_dir)
        if self.d1_alternating_roles:
            if self.batch_size != 1:
                raise RuntimeError("D1 alternating roles require batch_size=1")
            if not self.role_manifest_path:
                raise RuntimeError("D1 alternating roles require role_manifest_path")
            full.install_sample_roles(self.role_manifest_path)
        self.full_dataset = full
        if self.val_split > 0.0 and len(full) > 1:
            n_val = max(1, int(len(full) * self.val_split))
            n_train = len(full) - n_val
            if n_train <= 0:
                raise RuntimeError("validation split leaves no training samples")
            split_generator = torch.Generator().manual_seed(self.seed)
            self.train_dataset, self.val_dataset = random_split(full, [n_train, n_val], generator=split_generator)
        else:
            self.train_dataset, self.val_dataset = full, None

    def _loader(self, dataset: Dataset[Any], *, shuffle: bool) -> DataLoader[Any]:
        generator = torch.Generator().manual_seed(self.seed + (0 if shuffle else 1))
        sampler = None
        if shuffle and self.d1_alternating_roles:
            sampler = D1AlternatingRoleSampler(dataset, seed=self.seed)
        kwargs: Dict[str, Any] = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "shuffle": shuffle and sampler is None,
            "generator": generator,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "collate_fn": strict_phase_d_collate,
            "drop_last": False,
            "persistent_workers": self.persistent_workers,
        }
        if sampler is not None:
            kwargs["sampler"] = sampler
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor if self.prefetch_factor is not None else 2
        if self.pin_memory_device:
            kwargs["pin_memory_device"] = self.pin_memory_device
        return DataLoader(**kwargs)

    def train_dataloader(self) -> DataLoader[Any]:
        if self.train_dataset is None:
            raise RuntimeError("data module setup('fit') was not called")
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> Optional[DataLoader[Any]]:
        if self.val_dataset is None:
            return None
        return self._loader(self.val_dataset, shuffle=False)
