from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

ACE = Path(__file__).resolve().parents[3]
if str(ACE) not in sys.path:
    sys.path.insert(0, str(ACE))

from acestep.training_v2.fixed_lora_module import FixedLoRAModule
from acestep.training_v2.optim import (
    GroupRatioCosineAnnealingLR,
    build_optimizer,
    build_scheduler,
)
from acestep.training_v2.trainer_helpers import (
    _collect_decoder_timing_state,
    _save_timing_module,
)


class AdapterProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.ones(2, 2))
        self.lora_B = nn.Parameter(torch.zeros(2, 2))


class TimingAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = AdapterProjection()
        self.k_proj = AdapterProjection()
        self.v_proj = AdapterProjection()
        self.o_proj = AdapterProjection()


class TimingLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.timing_attn_gate = nn.Parameter(
            torch.full((2,), -4.0, dtype=torch.bfloat16)
        )
        self.timing_cross_attn = TimingAttention()
        self.timing_cross_attn_norm = nn.LayerNorm(2)
        self.timing_global_modulation = nn.Linear(2, 2)


class DecoderBody(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TimingLayer() for _ in range(24)])


class DecoderWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.model = DecoderBody()


def make_module() -> FixedLoRAModule:
    module = FixedLoRAModule.__new__(FixedLoRAModule)
    nn.Module.__init__(module)
    module.model = nn.Module()
    module.model.decoder = DecoderWrapper()
    module.timing_encoder = nn.Linear(2, 2)
    module.decoder_timing_supervisor = nn.Linear(2, 2)
    module.performance_timing_predictor = None
    module.decoder_expressivity_supervisor = None
    module._timing_train_last_n_layers = 8
    module._timing_consumer_adapter_modules = ("q_proj", "o_proj")
    module._enable_timing_consumer_training = True
    module._enable_phrase_modulation = False
    return module


class TimingTrainingContractTest(unittest.TestCase):
    def test_all_gates_and_only_last_eight_consumers_train(self) -> None:
        module = make_module()
        module._enable_decoder_timing_parameters()
        named = dict(module.model.decoder.named_parameters())
        gates = [
            parameter
            for name, parameter in named.items()
            if name.endswith("timing_attn_gate")
        ]
        self.assertEqual(len(gates), 24)
        self.assertTrue(all(parameter.requires_grad for parameter in gates))

        for layer in range(24):
            prefix = f"base_model.model.layers.{layer}"
            should_train_consumer = layer >= 16
            self.assertEqual(
                named[f"{prefix}.timing_cross_attn.q_proj.lora_B"].requires_grad,
                should_train_consumer,
            )
            self.assertEqual(
                named[f"{prefix}.timing_cross_attn.o_proj.lora_B"].requires_grad,
                should_train_consumer,
            )
            self.assertFalse(
                named[f"{prefix}.timing_cross_attn.k_proj.lora_B"].requires_grad
            )
            self.assertFalse(
                named[f"{prefix}.timing_global_modulation.weight"].requires_grad
            )

    def test_fp32_promotion_and_disjoint_optimizer_groups(self) -> None:
        module = make_module()
        module._enable_decoder_timing_parameters()
        self.assertEqual(module.promote_decoder_timing_gates_fp32(), 48)
        gates = [
            parameter
            for name, parameter in module.named_parameters()
            if name.endswith("timing_attn_gate")
        ]
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in gates))

        groups = module.timing_optimizer_groups(
            base_lr=5e-6,
            timing_encoder_lr=5e-5,
            timing_gate_lr=1e-3,
            timing_consumer_lr=1e-5,
        )
        by_name = {group["group_name"]: group for group in groups}
        self.assertEqual(set(by_name), {"timing_encoder", "timing_gate", "timing_consumer"})
        parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
        self.assertEqual(len(parameter_ids), len(set(parameter_ids)))
        self.assertEqual(by_name["timing_gate"]["lr"], 1e-3)
        self.assertEqual(by_name["timing_consumer"]["lr"], 1e-5)

    def test_clarity_replay_has_exact_three_optimizer_groups(self) -> None:
        module = make_module()
        module.base_adapter = AdapterProjection()
        module._enable_timing_consumer_training = False
        module._enable_decoder_timing_parameters()
        groups = module.timing_optimizer_groups(
            base_lr=6e-6,
            timing_encoder_lr=5e-5,
            timing_gate_lr=5e-5,
            timing_consumer_lr=0.0,
        )
        by_name = {group["group_name"]: group for group in groups}
        self.assertEqual(set(by_name), {"base", "timing_encoder", "timing_gate"})
        self.assertEqual(by_name["base"]["lr"], 6e-6)
        self.assertEqual(by_name["timing_encoder"]["lr"], 5e-5)
        self.assertEqual(by_name["timing_gate"]["lr"], 5e-5)
        parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
        self.assertEqual(len(parameter_ids), len(set(parameter_ids)))

    def test_checkpoint_excludes_disabled_global_modules(self) -> None:
        module = make_module()
        state = _collect_decoder_timing_state(module)
        self.assertTrue(state)
        self.assertFalse(any("timing_global_" in key for key in state))
        self.assertEqual(sum(key.endswith("timing_attn_gate") for key in state), 24)

    def test_disabled_timing_never_writes_checkpoint_state(self) -> None:
        module = make_module()
        module.training_config = SimpleNamespace(enable_timing_branch=False)
        with tempfile.TemporaryDirectory() as directory:
            _save_timing_module(module, directory)
            self.assertFalse((Path(directory) / "timing_branch.pt").exists())

    def test_zero_warmup_preserves_requested_initial_lr(self) -> None:
        parameter = nn.Parameter(torch.ones(1))
        optimizer = build_optimizer(
            [parameter],
            optimizer_type="adamw",
            lr=5e-5,
            device_type="cpu",
        )
        scheduler = build_scheduler(
            optimizer,
            scheduler_type="cosine",
            total_steps=200,
            warmup_steps=0,
            lr=5e-5,
        )
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 5e-5, places=12)
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 5e-5, places=12)

    def test_cosine_scheduler_preserves_group_lr_ratio(self) -> None:
        left = nn.Parameter(torch.ones(1))
        right = nn.Parameter(torch.ones(1))
        optimizer = build_optimizer(
            [
                {"params": [left], "lr": 5e-6, "group_name": "base"},
                {"params": [right], "lr": 1e-3, "group_name": "timing_gate"},
            ],
            optimizer_type="adamw",
            lr=5e-6,
            device_type="cpu",
        )
        scheduler = build_scheduler(
            optimizer,
            scheduler_type="cosine",
            total_steps=100,
            warmup_steps=10,
            lr=5e-6,
        )
        # Warmup wraps the named ratio-preserving cosine in SequentialLR.
        for _ in range(50):
            optimizer.step()
            scheduler.step()
        rates = [group["lr"] for group in optimizer.param_groups]
        self.assertAlmostEqual(rates[1] / rates[0], 200.0, places=6)

    def test_controlled_zero_warmup_uses_named_group_ratio_cosine(self) -> None:
        base = nn.Parameter(torch.ones(1))
        timing = nn.Parameter(torch.ones(1))
        optimizer = build_optimizer(
            [
                {"params": [base], "lr": 6e-6, "group_name": "base"},
                {"params": [timing], "lr": 5e-5, "group_name": "timing"},
            ],
            optimizer_type="adamw",
            lr=6e-6,
            device_type="cpu",
        )
        scheduler = build_scheduler(
            optimizer,
            scheduler_type="cosine",
            total_steps=200,
            warmup_steps=0,
            lr=6e-6,
        )
        self.assertIs(type(scheduler), GroupRatioCosineAnnealingLR)
        self.assertEqual(scheduler.T_max, 200)
        self.assertEqual(scheduler.min_factor, 0.01)
        self.assertEqual(scheduler.base_lrs, [6e-6, 5e-5])
        self.assertAlmostEqual(scheduler._factor(0), 1.0, places=12)
        self.assertAlmostEqual(scheduler._factor(100), 0.505, places=12)
        self.assertAlmostEqual(scheduler._factor(200), 0.01, places=12)
        for _ in range(200):
            optimizer.step()
            scheduler.step()
        rates = [group["lr"] for group in optimizer.param_groups]
        self.assertAlmostEqual(rates[0], 6e-8, places=14)
        self.assertAlmostEqual(rates[1], 5e-7, places=14)
        self.assertAlmostEqual(rates[1] / rates[0], 5e-5 / 6e-6, places=10)


if __name__ == "__main__":
    unittest.main()
