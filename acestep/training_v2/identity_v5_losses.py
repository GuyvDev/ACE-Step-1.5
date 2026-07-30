"""Identity V5 waveform-aligned loss helpers.

These helpers are deliberately fail-closed. They support supervised contrastive
prototype loss and teacher preservation directly, and require an explicitly
provided differentiable singer student before waveform identity loss can be used.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_loss(anchor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(F.normalize(anchor.float(), dim=-1), F.normalize(target.float(), dim=-1), dim=-1)).mean()


def distinct_singer_supcon(anchor: torch.Tensor, positive: torch.Tensor, negative_by_singer: Mapping[str, torch.Tensor], temperature: float = 0.07) -> torch.Tensor:
    if len(negative_by_singer) < 4:
        raise RuntimeError('V5 supcon requires at least four distinct negative singers')
    anchor=F.normalize(anchor.float(), dim=-1); positive=F.normalize(positive.float(), dim=-1)
    negatives=[]
    for singer, emb in sorted(negative_by_singer.items()):
        if emb.ndim == 2:
            emb=emb.mean(dim=0)
        negatives.append(F.normalize(emb.float(), dim=-1))
    neg=torch.stack(negatives, dim=0).to(anchor.device)
    pos=(anchor*positive).sum(dim=-1, keepdim=True)
    neg_logits=anchor @ neg.T
    logits=torch.cat([pos, neg_logits], dim=-1) / max(float(temperature), 1e-6)
    labels=torch.zeros(anchor.shape[0], dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, labels)


def phase_a_teacher_loss(v_pred: torch.Tensor, v_teacher: torch.Tensor) -> torch.Tensor:
    if v_pred.shape != v_teacher.shape:
        raise RuntimeError(f'Phase A teacher tensor shape mismatch: {tuple(v_pred.shape)} vs {tuple(v_teacher.shape)}')
    return F.mse_loss(v_pred.float(), v_teacher.detach().float())


@dataclass
class GradientDiagnostic:
    name: str
    norm: float


def gradient_norm(parameters, loss: torch.Tensor, retain_graph: bool = True) -> float:
    grads=torch.autograd.grad(loss, [p for p in parameters if p.requires_grad], retain_graph=retain_graph, allow_unused=True)
    total=0.0
    for g in grads:
        if g is not None:
            total += float(g.detach().float().pow(2).sum().item())
    return total ** 0.5


class DecodedIdentityLoss(nn.Module):
    def __init__(self, vae_decoder: nn.Module | None, singer_student: nn.Module | None, crop_frames: int = 256):
        super().__init__(); self.vae_decoder=vae_decoder; self.singer_student=singer_student; self.crop_frames=crop_frames
        if self.vae_decoder is None or self.singer_student is None:
            raise RuntimeError('V5 waveform identity loss requires explicit differentiable VAE decoder and singer student')
    def embed(self, pred_x0: torch.Tensor) -> torch.Tensor:
        if pred_x0.shape[1] > self.crop_frames:
            pred_x0=pred_x0[:, :self.crop_frames]
        wav=self.vae_decoder(pred_x0)
        return self.singer_student(wav)
    def forward(self, pred_x0: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
        emb=self.embed(pred_x0)
        if not emb.requires_grad:
            raise RuntimeError('V5 decoded identity encoder is not differentiable to generator output')
        return cosine_loss(emb, prototype)


class MelSingerStudent(nn.Module):
    """Small differentiable mel-domain student for ECAPA geometry distillation."""
    def __init__(self, embedding_dim: int = 192, sample_rate: int = 16000) -> None:
        super().__init__()
        import torchaudio
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=512, hop_length=160, n_mels=80)
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1), nn.SiLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Linear(96, embedding_dim)
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 3: waveform = waveform.mean(dim=1)
        if waveform.ndim != 2: raise ValueError(f'expected [B,T] waveform, got {tuple(waveform.shape)}')
        mel = torch.log1p(self.mel(waveform.float())).unsqueeze(1)
        return F.normalize(self.projection(self.network(mel).flatten(1)), dim=-1)


class OobleckDecodeAdapter(nn.Module):
    def __init__(self, vae: nn.Module):
        super().__init__(); self.vae=vae
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        # Diffusers Oobleck consumes [B,C,T]. Keep this path differentiable.
        decoded=self.vae.decode(latents.transpose(1,2))
        waveform=decoded.sample if hasattr(decoded,'sample') else decoded[0] if isinstance(decoded,(tuple,list)) else decoded
        return waveform
