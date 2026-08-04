"""Contract tests for the Arm A / Arm B timing controllers."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from acestep.training_v2.phase_d_timing_v2.plan_control import (
    InputFusion, PlanController, PlanLoRA, PlanState, TARGET_PROJECTIONS,
    TemporalPlanEncoder,
)

HIDDEN, COND, RANK, WIDTH, T = 64, 14, 8, 32, 40


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        for name in TARGET_PROJECTIONS:
            setattr(self, name, nn.Linear(HIDDEN, HIDDEN))


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attn()


class Config:
    hidden_size = HIDDEN


class DiT(nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.layers = nn.ModuleList([Layer() for _ in range(n)])
        self.proj_in = nn.Linear(HIDDEN, HIDDEN)
        self.config = Config()


def controller(arm="A", n=4):
    dit = DiT(n)
    c = PlanController(dit, COND, arm=arm, rank=RANK, width=WIDTH,
                       layer_indices=range(n))
    c.attach_input_fusion()
    return dit, c


def plan(batch=1):
    return torch.randn(batch, T, COND)


def test_arm_must_be_a_or_b():
    with pytest.raises(ValueError, match="arm must be"):
        PlanController(DiT(), COND, arm="C")


def test_arm_a_has_input_fusion_and_arm_b_does_not():
    _, a = controller("A")
    _, b = controller("B")
    assert a.fusion is not None
    assert b.fusion is None


def test_every_target_projection_is_wrapped():
    dit, c = controller("A", n=4)
    assert len(c.loras) == 4 * len(TARGET_PROJECTIONS)
    for layer in dit.layers:
        for name in TARGET_PROJECTIONS:
            assert isinstance(getattr(layer.self_attn, name), PlanLoRA)


def test_delta_is_exactly_zero_at_init():
    _, c = controller("A")
    assert c.zero_delta_check()


def test_null_route_is_bitwise_frozen_backbone():
    dit, c = controller("A")
    x = torch.randn(1, T, HIDDEN)
    c.clear_plan()
    before = dit.proj_in(x).clone()
    lora = next(iter(c.loras.values()))
    assert torch.equal(lora(x), lora.base(x))
    c.clear_plan()
    assert torch.equal(dit.proj_in(x), before)


def test_plan_active_zero_restores_exact_output():
    dit, c = controller("A")
    x = torch.randn(1, T, HIDDEN)
    lora = next(iter(c.loras.values()))
    c.set_plan(plan())
    with torch.no_grad():
        lora.up.weight.fill_(0.1)
    conditioned = lora(x)
    c.clear_plan()
    assert torch.equal(lora(x), lora.base(x))
    assert not torch.equal(conditioned, lora.base(x))


def test_arm_b_gate_is_per_rank_vector_not_scalar():
    _, c = controller("B")
    lora = next(iter(c.loras.values()))
    c.set_plan(plan(), timestep=torch.tensor([0.5]))
    x = torch.randn(1, T, HIDDEN)
    gate_input = torch.cat([
        torch.nn.functional.interpolate(
            c.state.encoded.transpose(1, 2), size=T, mode="linear",
            align_corners=False).transpose(1, 2),
        torch.full((1, T, 1), 0.5), torch.full((1, T, 1), lora.layer_position)], dim=-1)
    gate = lora.gate(gate_input)
    assert gate.shape == (1, T, RANK)


def test_arm_a_lora_has_no_gate():
    _, c = controller("A")
    assert not hasattr(next(iter(c.loras.values())), "gate")


def test_input_fusion_is_zero_init_and_additive():
    fusion = InputFusion(WIDTH, HIDDEN)
    state = PlanState()
    x = torch.randn(1, T, HIDDEN)
    state.encoded, state.active = torch.randn(1, T, WIDTH), 1.0
    assert torch.equal(fusion(x, state), x)          # zero-init -> no-op
    with torch.no_grad():
        fusion.project.weight.fill_(0.01)
    assert not torch.equal(fusion(x, state), x)


def test_fusion_resamples_plan_to_sequence_length():
    fusion = InputFusion(WIDTH, HIDDEN)
    state = PlanState()
    state.encoded, state.active = torch.randn(1, T * 2, WIDTH), 1.0
    with torch.no_grad():
        fusion.project.weight.fill_(0.01)
    assert fusion(torch.randn(1, T, HIDDEN), state).shape == (1, T, HIDDEN)


def test_control_strength_scales_the_delta():
    _, c = controller("A")
    lora = next(iter(c.loras.values()))
    x = torch.randn(1, T, HIDDEN)
    with torch.no_grad():
        lora.up.weight.fill_(0.1)
    c.set_plan(plan(), control_strength=1.0)
    full = (lora(x) - lora.base(x)).abs().mean()
    c.set_plan(plan(), control_strength=0.5)
    half = (lora(x) - lora.base(x)).abs().mean()
    assert half < full


def test_encoder_preserves_time_and_maps_width():
    encoder = TemporalPlanEncoder(COND, WIDTH)
    assert encoder(torch.randn(2, T, COND)).shape == (2, T, WIDTH)


def test_plan_must_be_three_dimensional():
    _, c = controller("A")
    with pytest.raises(ValueError, match=r"plan must be \[B, T, C\]"):
        c.set_plan(torch.randn(T, COND))


def test_gradient_flows_only_to_controller():
    dit, c = controller("A")
    for p in dit.parameters():
        p.requires_grad = False
    for lora in c.loras.values():
        for p in lora.base.parameters():
            p.requires_grad = False
    for name, p in c.named_parameters():
        if ".base." not in name:
            p.requires_grad = True
    with torch.no_grad():
        for lora in c.loras.values():
            lora.up.weight.fill_(0.05)
        c.fusion.project.weight.fill_(0.05)
    c.set_plan(plan(), timestep=torch.tensor([0.5]))
    x = torch.randn(1, T, HIDDEN)
    out = dit.proj_in(x)
    for layer in dit.layers:
        out = layer.self_attn.q_proj(out)
    out.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for n, p in c.named_parameters() if ".base." not in n)
    assert all(p.grad is None or p.grad.abs().sum() == 0
               for lora in c.loras.values() for p in lora.base.parameters())


def test_detach_removes_the_fusion_hook():
    dit, c = controller("A")
    with torch.no_grad():
        c.fusion.project.weight.fill_(0.05)
    c.set_plan(plan())
    x = torch.randn(1, T, HIDDEN)
    fused = dit.proj_in(x).clone()
    c.detach()
    assert not torch.equal(fused, dit.proj_in(x))
