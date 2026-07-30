"""V5 gated singer conditioning with identity-preserving initialization."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedSingerAdaLN(nn.Module):
    """Global singer FiLM adapter; exactly identity at initialization."""
    def __init__(self, singer_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.to_scale_shift=nn.Sequential(nn.Linear(singer_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,2*hidden_dim))
        # The zero gate guarantees an exact identity forward. Keeping the projection
        # nonzero avoids a zero-gradient deadlock between projection and gate.
        nn.init.xavier_uniform_(self.to_scale_shift[-1].weight, gain=0.02)
        nn.init.zeros_(self.to_scale_shift[-1].bias)
        self.gate=nn.Parameter(torch.zeros(()))
    def forward(self, hidden: torch.Tensor, singer: torch.Tensor) -> torch.Tensor:
        scale,shift=self.to_scale_shift(singer).chunk(2,dim=-1)
        update=scale[:,None]*F.layer_norm(hidden,hidden.shape[-1:])+shift[:,None]
        return hidden+torch.tanh(self.gate)*update

class GatedFragmentAttention(nn.Module):
    """Query-dependent local MERT attention with crop labels and zero gate."""
    def __init__(self, hidden_dim: int, mert_dim: int, crop_types: int=5) -> None:
        super().__init__()
        self.kv=nn.Linear(mert_dim,2*hidden_dim,bias=False)
        self.q=nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.crop_embedding=nn.Embedding(crop_types,mert_dim)
        self.gate=nn.Parameter(torch.zeros(())); self.scale=hidden_dim**-0.5
    def forward(self, hidden: torch.Tensor, tokens: torch.Tensor, crop_ids: torch.Tensor, mask: torch.Tensor|None=None) -> torch.Tensor:
        tokens=tokens+self.crop_embedding(crop_ids)
        key,value=self.kv(tokens).chunk(2,dim=-1); query=self.q(hidden)
        score=query@key.transpose(-1,-2)*self.scale
        if mask is not None:
            valid = mask.bool()
            if not torch.all(valid.any(dim=-1)):
                raise RuntimeError('V5 fragment attention received a sample with no valid reference tokens')
            score=score.masked_fill(~valid[:,None],torch.finfo(score.dtype).min)
        attended=torch.softmax(score,dim=-1)@value
        return hidden+torch.tanh(self.gate)*attended

def assert_identity_initialization(module: nn.Module, hidden: torch.Tensor, *args: torch.Tensor) -> None:
    with torch.no_grad():
        if not torch.equal(module(hidden,*args),hidden): raise AssertionError('fresh singer adapter must preserve Phase A exactly')
