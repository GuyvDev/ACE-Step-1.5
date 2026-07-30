"""Fail-closed helpers shared by controlled Phase-D inference.

The inference runner must call these helpers; they never use permissive state loading.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn

def tensor_sha256(t: torch.Tensor)->str:
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def file_sha256(path: Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()

def load_fixed_latent(path: str|Path, *, expected_seed:int)->tuple[torch.Tensor,dict[str,Any]]:
    p=Path(path).resolve(); payload=torch.load(p,map_location='cpu',weights_only=True)
    if not isinstance(payload,dict) or payload.get('schema_version')!='phase_d_fixed_latent_v1':
        raise RuntimeError(f'invalid fixed-latent schema: {p}')
    t=payload.get('initial_latent')
    if not isinstance(t,torch.Tensor) or tuple(t.shape)!=(1,8500,64) or t.dtype!=torch.float32:
        raise RuntimeError(f'invalid fixed latent tensor: {p}')
    if int(payload.get('seed',-1))!=int(expected_seed): raise RuntimeError('fixed latent seed mismatch')
    if payload.get('shape') != [1,8500,64] or payload.get('dtype') != 'torch.float32':
        raise RuntimeError('fixed latent payload shape/dtype declaration mismatch')
    if not bool(torch.isfinite(t).all()): raise RuntimeError('fixed latent is non-finite')
    actual_tensor_sha=tensor_sha256(t)
    if payload.get('tensor_sha256') != actual_tensor_sha: raise RuntimeError('fixed latent embedded tensor hash mismatch')
    return t,{'path':str(p),'file_sha256':file_sha256(p),'sha256':actual_tensor_sha,'tensor_sha256':actual_tensor_sha,'seed':expected_seed,'shape':list(t.shape),'dtype':str(t.dtype),'schema_version':payload['schema_version']}

def load_module_state_exact(module: nn.Module, state: dict[str,torch.Tensor], *, label:str)->dict[str,Any]:
    if not isinstance(state,dict) or not state: raise RuntimeError(f'{label} state is missing/empty')
    target=module.state_dict()
    missing=sorted(set(target)-set(state)); unexpected=sorted(set(state)-set(target))
    if missing or unexpected: raise RuntimeError(f'{label} names mismatch missing={missing[:10]} unexpected={unexpected[:10]}')
    bad_shapes=[k for k in state if tuple(state[k].shape)!=tuple(target[k].shape)]
    if bad_shapes: raise RuntimeError(f'{label} shape mismatch: {bad_shapes[:10]}')
    module.load_state_dict(state,strict=True)
    loaded=module.state_dict(); diffs=[float((loaded[k].detach().cpu()-state[k].detach().cpu().to(loaded[k].dtype)).abs().max()) for k in state]
    max_diff=max(diffs,default=0.0)
    if max_diff!=0.0: raise RuntimeError(f'{label} values differ after exact load: {max_diff}')
    return {'loading_policy':'strict_true_exact_names','names_exact':True,'expected_names':sorted(state),'loaded_names':sorted(loaded),'tensor_count':len(state),'max_abs_difference':max_diff}

def _child(obj:Any,part:str)->Any:
    if isinstance(obj,(nn.ModuleList,nn.Sequential)) and part.isdigit(): return obj[int(part)]
    if isinstance(obj,(nn.ModuleDict,nn.ParameterDict)) and part in obj: return obj[part]
    return getattr(obj,part)

def _resolve(root:Any,key:str)->torch.Tensor:
    candidates=[key.split('.')]
    if key.startswith('base_model.model.'): candidates.append(key.removeprefix('base_model.model.').split('.'))
    errors=[]
    for parts in candidates:
        try:
            obj=root
            for part in parts[:-1]: obj=_child(obj,part)
            value=getattr(obj,parts[-1])
            if isinstance(value,torch.Tensor): return value
            errors.append('target_not_tensor')
        except Exception as exc: errors.append(str(exc))
    raise KeyError(f'cannot resolve {key}: {errors}')

def load_decoder_timing_exact(
    decoder:nn.Module,
    state:dict[str,torch.Tensor],
    *,
    enable_global_condition:bool,
)->dict[str,Any]:
    if not isinstance(state, dict) or not state:
        raise RuntimeError('decoder timing state is missing/empty')
    forbidden=sorted(k for k in state if any(m in k for m in ('lora_','hada_','lokr_')))
    if forbidden: raise RuntimeError(f'timing state contains adapter tensors: {forbidden[:10]}')
    runtime_timing=sorted(
        key for key in decoder.state_dict()
        if 'timing_' in key and not any(marker in key for marker in ('lora_','hada_','lokr_'))
    )
    cross_or_gate=[
        key for key in runtime_timing
        if 'timing_cross_attn' in key or key.endswith('timing_attn_gate')
    ]
    global_names=[key for key in runtime_timing if 'timing_global_' in key]
    supported=set(cross_or_gate)|set(global_names)
    unsupported=sorted(set(runtime_timing)-supported)
    if unsupported:
        raise RuntimeError(f'unsupported runtime timing tensors: {unsupported[:10]}')
    runtime_expected=sorted(cross_or_gate + (global_names if enable_global_condition else []))
    disabled_global_names=sorted([] if enable_global_condition else global_names)
    expected=sorted(state)
    missing=sorted(set(runtime_expected)-set(expected))
    unexpected=sorted(set(expected)-set(runtime_expected))
    if missing or unexpected:
        raise RuntimeError(
            f'decoder timing names mismatch missing={missing[:10]} unexpected={unexpected[:10]}'
        )
    failed=[]; loaded=[]; max_diff=0.0
    for key in expected:
        try:
            target=_resolve(decoder,key); source=state[key]
            if tuple(target.shape)!=tuple(source.shape): raise RuntimeError(f'shape {tuple(source.shape)} != {tuple(target.shape)}')
            with torch.no_grad(): target.copy_(source.to(device=target.device,dtype=target.dtype))
            diff=float((target.detach()-source.to(device=target.device,dtype=target.dtype)).abs().max()) if target.numel() else 0.0
            max_diff=max(max_diff,diff)
            if diff!=0.0: raise RuntimeError(f'max diff {diff}')
            loaded.append(key)
        except Exception as exc: failed.append({'key':key,'error':str(exc)})
    if failed or loaded!=expected: raise RuntimeError(f'exact decoder timing load failed: {failed[:10]}')
    critical=[k for k in expected if 'timing_cross_attn' in k or k.endswith('timing_attn_gate')]
    if not critical: raise RuntimeError('no critical timing tensors found')
    return {'loading_policy':'exact_manual_profile_topology_no_permissive_load','critical_non_lora_timing_ok':True,'manual_non_lora_timing_failed':0,'missing_keys':0,'unexpected_keys':0,'names_exact':True,'enable_global_condition':bool(enable_global_condition),'disabled_global_tensor_count':len(disabled_global_names),'disabled_global_names':disabled_global_names,'runtime_expected_names':runtime_expected,'expected_names':expected,'loaded_names':loaded,'max_abs_difference':max_diff,'tensor_count':len(expected)}
