from __future__ import annotations

import unittest

import torch

from acestep.training_v2.timing_conditioning import TimingEncoder, TimingEncoderConfig


class AbsoluteTimeConditioningTest(unittest.TestCase):
    def _encoder(self) -> TimingEncoder:
        torch.manual_seed(7)
        return TimingEncoder(
            TimingEncoderConfig(
                hidden_size=16,
                num_heads=4,
                num_layers=1,
                output_dim=16,
                max_seq_len=16,
                dropout=0.0,
                enable_phrase_features=False,
                enable_absolute_time_conditioning=True,
                init_profile="historical_v4",
            )
        ).eval()

    def test_missing_timestamps_fail_closed(self) -> None:
        encoder = self._encoder()
        tokens = torch.zeros(1, 3, 8, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "requires event starts"):
            encoder.encode(tokens)

    def test_absolute_position_changes_conditioning(self) -> None:
        encoder = self._encoder()
        tokens = torch.zeros(1, 3, 8, dtype=torch.long)
        mask = torch.ones(1, 3, dtype=torch.bool)
        duration = torch.tensor([100.0])
        ends_a = torch.tensor([[11.0, 21.0, 31.0]])
        ends_b = torch.tensor([[31.0, 41.0, 51.0]])
        _, projected_a, _, _ = encoder.encode(
            tokens,
            mask,
            event_starts_sec=ends_a - 1.0,
            event_ends_sec=ends_a,
            audio_durations_sec=duration,
        )
        _, projected_b, _, _ = encoder.encode(
            tokens,
            mask,
            event_starts_sec=ends_b - 1.0,
            event_ends_sec=ends_b,
            audio_durations_sec=duration,
        )
        self.assertGreater(float((projected_a - projected_b).abs().max()), 1e-5)

    def test_config_round_trip_preserves_opt_in(self) -> None:
        config = TimingEncoderConfig(enable_absolute_time_conditioning=True)
        restored = TimingEncoderConfig.from_dict(config.to_dict())
        self.assertTrue(restored.enable_absolute_time_conditioning)


if __name__ == "__main__":
    unittest.main()
