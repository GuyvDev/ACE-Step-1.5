"""Phase D timing control arms A and B over a frozen C25 backbone.

Arm A -- input fusion + plan-gated static LoRA
    The plan is projected into the DiT input sequence through a zero-initialised
    projection, so it is present before layer 0 and therefore already acting on
    the first step out of noise, where temporal structure is decided. The LoRA
    on q/k/v/o is static and scaled by a scalar plan_active.

Arm B -- conditional LoRA inside attention
    delta = B(A(hidden) * gate(plan, t, layer)) with gate shaped [B, T, rank].
    The gate is a per-rank VECTOR, not a scalar: a scalar could only rescale one
    fixed delta direction, which cannot relocate a word. A per-rank gate lets
    the plan reweight rank directions and so change the delta's direction.

Both arms share this module, the same encoder, the same rank and the same
target projections, so an A/B comparison differs only in where the plan acts.

Null route is exact by construction: every trainable contribution is multiplied
by plan_active, so plan_active = 0 reproduces frozen C25 bit for bit.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGET_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


def _is_projection(module) -> bool:
    """A linear projection, whether nn.Linear or a PEFT-wrapped one."""
    return (isinstance(module, nn.Module)
            and hasattr(module, "in_features") and hasattr(module, "out_features")
            and type(module).__name__ != "PlanLoRA")


class TemporalPlanEncoder(nn.Module):
    """[B, T, C_plan] -> [B, T, width]. Dilated convs give phrase-scale context."""

    def __init__(self, condition_dim: int, width: int = 256, layers: int = 4):
        super().__init__()
        self.input = nn.Conv1d(condition_dim, width, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size=3, padding=2 ** i, dilation=2 ** i)
            for i in range(layers)])
        self.norms = nn.ModuleList([nn.GroupNorm(8, width) for _ in range(layers)])

    def forward(self, plan: torch.Tensor) -> torch.Tensor:
        x = self.input(plan.transpose(1, 2))
        for block, norm in zip(self.blocks, self.norms):
            x = x + F.gelu(norm(block(x)))
        return x.transpose(1, 2)


def _resample(x: torch.Tensor, length: int) -> torch.Tensor:
    """Match the plan's frame count to the post-patchify sequence length."""
    if x.shape[1] == length:
        return x
    return F.interpolate(x.transpose(1, 2), size=length, mode="linear",
                         align_corners=False).transpose(1, 2)


class PlanState:
    """Active plan shared by the fusion hook and every LoRA site."""

    def __init__(self):
        self.encoded: torch.Tensor | None = None   # [B, T, width]
        self.active: float = 0.0
        self.timestep: torch.Tensor | None = None
        self.control_strength: float = 1.0

    def clear(self) -> None:
        self.encoded, self.active, self.timestep = None, 0.0, None

    @property
    def on(self) -> bool:
        return self.encoded is not None and self.active != 0.0


class InputFusion(nn.Module):
    """Zero-init projection of the plan added to the DiT input sequence."""

    def __init__(self, width: int, hidden_size: int):
        super().__init__()
        self.project = nn.Linear(width, hidden_size)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, hidden_states: torch.Tensor, state: PlanState) -> torch.Tensor:
        if not state.on:
            return hidden_states
        plan = _resample(state.encoded, hidden_states.shape[1])
        delta = self.project(plan.to(hidden_states.dtype))
        return hidden_states + delta * (state.active * state.control_strength)


class PlanLoRA(nn.Module):
    """LoRA on one frozen projection. Arm A static, Arm B plan-gated per rank."""

    def __init__(self, base: nn.Module, rank: int, width: int, conditional: bool,
                 num_layers: int, layer_index: int):
        super().__init__()
        self.base, self.rank, self.conditional = base, rank, conditional
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)          # zero-init: delta starts at 0
        if conditional:
            # gate -> [B, T, rank]; timestep and layer identity enter as scalars
            self.gate = nn.Sequential(
                nn.Linear(width + 2, width), nn.GELU(), nn.Linear(width, rank))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.ones_(self.gate[-1].bias)   # gate starts at 1, delta still 0
        self.layer_position = layer_index / max(num_layers - 1, 1)
        self.state: PlanState | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = self.base(hidden_states)
        state = self.state
        if state is None or not state.on:
            return out
        low = self.down(hidden_states)
        if self.conditional:
            plan = _resample(state.encoded, hidden_states.shape[1]).to(hidden_states.dtype)
            t = state.timestep
            t_col = (t.reshape(-1, 1, 1).expand(plan.shape[0], plan.shape[1], 1)
                     if t is not None else torch.zeros_like(plan[..., :1]))
            pos = torch.full_like(t_col, self.layer_position)
            low = low * self.gate(torch.cat([plan, t_col, pos], dim=-1))
        delta = self.up(low) * (state.active * state.control_strength)
        return out + delta.to(out.dtype)


class PlanController(nn.Module):
    """Attach one arm to a frozen DiT and own every trainable parameter."""

    def __init__(self, dit, condition_dim: int, arm: str = "A", rank: int = 16,
                 width: int = 256, layer_indices: Iterable[int] = range(8)):
        super().__init__()
        if arm not in ("A", "B"):
            raise ValueError(f"arm must be 'A' or 'B', got {arm!r}")
        self.arm, self.state = arm, PlanState()
        hidden_size = int(dit.config.hidden_size)
        self.encoder = TemporalPlanEncoder(condition_dim, width)
        self.fusion = InputFusion(width, hidden_size) if arm == "A" else None
        self.loras = nn.ModuleDict()
        self._handles = []
        layers = list(dit.layers)
        for index in layer_indices:
            if index >= len(layers):
                continue
            attn = getattr(layers[index], "self_attn", None)
            if attn is None:
                continue
            for name in TARGET_PROJECTIONS:
                base = getattr(attn, name, None)
                # C25 is a PEFT adapter, so these are peft.tuners.lora.layer.Linear
                # rather than nn.Linear. Duck-type on the projection interface so
                # the frozen C25 LoRA stays inside base and keeps applying.
                if not _is_projection(base):
                    continue
                lora = PlanLoRA(base, rank, width, conditional=(arm == "B"),
                                num_layers=len(layers), layer_index=index)
                lora.state = self.state
                device = next(base.parameters()).device
                lora.to(device, torch.float32)
                self.loras[f"l{index}_{name}"] = lora
                setattr(attn, name, lora)
        self.dit = [dit]                              # not a submodule: stays frozen

    def attach_input_fusion(self) -> None:
        """Add the plan to proj_in's output, i.e. before layer 0."""
        if self.fusion is None:
            return
        dit = self.dit[0]

        def hook(_module, _inputs, output):
            return self.fusion(output, self.state)

        self._handles.append(dit.proj_in.register_forward_hook(hook))

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def set_plan(self, plan: torch.Tensor, timestep: torch.Tensor | None = None,
                 control_strength: float = 1.0) -> None:
        if plan.dim() != 3:
            raise ValueError(f"plan must be [B, T, C], got {tuple(plan.shape)}")
        device = next(self.encoder.parameters()).device
        self.state.encoded = self.encoder(plan.to(device, torch.float32))
        self.state.active = 1.0
        self.state.timestep = timestep
        self.state.control_strength = control_strength

    def clear_plan(self) -> None:
        self.state.clear()

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def zero_delta_check(self) -> bool:
        """True when every output projection is still exactly zero."""
        zeros = [torch.count_nonzero(m.up.weight).item() == 0 for m in self.loras.values()]
        if self.fusion is not None:
            zeros.append(torch.count_nonzero(self.fusion.project.weight).item() == 0)
        return all(zeros)


def attach_plan_controller(model, condition_dim: int, arm: str = "A", rank: int = 16,
                           width: int = 256, layer_indices: Iterable[int] = range(8)):
    """Freeze the backbone, attach one arm, return (controller, audit)."""
    from acestep.training_v2.phase_d2.runtime_integration import unwrap_real_dit

    decoder = getattr(model, "decoder", None)
    if decoder is None:
        raise TypeError("model has no decoder")
    dit = unwrap_real_dit(decoder)
    for parameter in model.parameters():
        parameter.requires_grad = False

    controller = PlanController(dit, condition_dim, arm=arm, rank=rank, width=width,
                                layer_indices=layer_indices)
    controller.to(next(dit.parameters()).device, torch.float32)
    controller.attach_input_fusion()
    for parameter in controller.parameters():
        parameter.requires_grad = True
    # The wrapped frozen projections became children of the controller; keep the
    # backbone frozen even though it is now reachable from controller.parameters().
    for lora in controller.loras.values():
        for parameter in lora.base.parameters():
            parameter.requires_grad = False

    # Wrapping the projections makes controller parameters reachable from
    # model.parameters(), so the freeze audit must exclude them by identity
    # rather than counting every trainable tensor under the model.
    owned = {id(p) for p in controller.parameters()}
    backbone_trainable = [n for n, p in model.named_parameters()
                          if p.requires_grad and id(p) not in owned]
    trainable = sum(p.numel() for p in controller.parameters() if p.requires_grad)
    audit = {"arm": arm, "rank": rank, "encoder_width": width,
             "layer_indices": list(layer_indices),
             "lora_sites": len(controller.loras),
             "input_fusion": arm == "A",
             "trainable_parameters": trainable,
             "backbone_trainable": len(backbone_trainable),
             "zero_init_delta": controller.zero_delta_check()}
    if backbone_trainable:
        raise RuntimeError(
            f"backbone parameters remain trainable after attach: {backbone_trainable[:3]}")
    return controller, audit


__all__ = ["PlanController", "TemporalPlanEncoder", "InputFusion", "PlanLoRA",
           "PlanState", "attach_plan_controller", "TARGET_PROJECTIONS"]
