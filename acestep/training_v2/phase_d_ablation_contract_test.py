"""Focused CPU tests for the Phase-D causal-ablation contracts."""

import unittest

import torch
import torch.nn as nn

from acestep.training_v2.timing_conditioning import TimingEncoder, TimingEncoderConfig
from acestep.training_v2.trainer_helpers import _collect_decoder_timing_state


class _FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.timing_attn_gate = nn.Parameter(torch.full((4,), -4.0))
        self.timing_cross_attn = nn.Module()
        self.timing_cross_attn.q_proj = nn.Module()
        self.timing_cross_attn.q_proj.base_layer = nn.Linear(4, 4, bias=False)
        self.timing_cross_attn.q_proj.lora_A = nn.ModuleDict(
            {"default": nn.Linear(4, 2, bias=False)}
        )
        self.timing_cross_attn.q_proj.lora_B = nn.ModuleDict(
            {"default": nn.Linear(2, 4, bias=False)}
        )


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = _FakeDecoder()


class _FakeTrainingModule:
    def __init__(self):
        self.model = _FakeModel()
        self._enable_phrase_modulation = False


class PhaseDAblationContractTest(unittest.TestCase):
    def _config(self, profile: str) -> TimingEncoderConfig:
        return TimingEncoderConfig(
            hidden_size=16,
            num_heads=4,
            num_layers=1,
            output_dim=32,
            dropout=0.0,
            max_seq_len=32,
            enable_phrase_features=profile != "historical_v4",
            init_profile=profile,
        )

    def test_historical_profile_is_open_and_unsuppressed(self):
        torch.manual_seed(42)
        encoder = TimingEncoder(self._config("historical_v4"))
        self.assertEqual(float(encoder.output_gate), 0.0)
        self.assertIsNone(encoder.global_gate)
        self.assertGreater(float(encoder.proj.weight.norm()), 0.0)
        self.assertEqual(float(encoder.proj.bias.norm()), 0.0)
        self.assertAlmostEqual(float(encoder.stream_type_embedding.std()), 0.02, delta=0.01)

    def test_suppressed_profile_reproduces_zero_path(self):
        torch.manual_seed(42)
        encoder = TimingEncoder(self._config("suppressed_new"))
        self.assertEqual(float(encoder.output_gate), -8.0)
        self.assertEqual(float(encoder.global_gate), -8.0)
        self.assertEqual(float(encoder.proj.weight.norm()), 0.0)
        self.assertEqual(float(encoder.proj.bias.norm()), 0.0)
        self.assertEqual(float(encoder.stream_type_embedding.norm()), 0.0)

    def test_timing_checkpoint_excludes_adapter_tensors(self):
        state = _collect_decoder_timing_state(_FakeTrainingModule())
        self.assertTrue(any(key.endswith("timing_attn_gate") for key in state))
        self.assertTrue(any("base_layer.weight" in key for key in state))
        self.assertFalse(any("lora_" in key for key in state))


if __name__ == "__main__":
    unittest.main()
