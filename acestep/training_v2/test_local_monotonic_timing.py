import unittest

import torch

from acestep.training_v2.timing_conditioning import (
    build_event_frame_weights,
    build_local_timing_attention_mask,
)


class LocalMonotonicTimingTest(unittest.TestCase):
    def setUp(self):
        self.starts = torch.tensor([[1.0, 4.0, 8.0]])
        self.ends = torch.tensor([[2.0, 5.0, 9.0]])
        self.duration = torch.tensor([10.0])
        self.valid = torch.tensor([[True, True, True]])

    def test_local_mask_is_finite_per_query_and_monotonic(self):
        mask = build_local_timing_attention_mask(
            event_starts_sec=self.starts,
            event_ends_sec=self.ends,
            audio_durations_sec=self.duration,
            timing_mask=self.valid,
            query_len=10,
            dtype=torch.float32,
            sigma_sec=0.5,
            window_sec=1.5,
        )
        self.assertEqual(tuple(mask.shape), (1, 1, 10, 3))
        self.assertFalse(torch.all(mask == torch.finfo(torch.float32).min, dim=-1).any())
        winners = mask[0, 0].argmax(dim=-1)
        self.assertTrue(bool(torch.all(winners[1:] >= winners[:-1])))

    def test_shift_changes_mask_but_not_shape(self):
        base = build_local_timing_attention_mask(
            event_starts_sec=self.starts,
            event_ends_sec=self.ends,
            audio_durations_sec=self.duration,
            timing_mask=self.valid,
            query_len=20,
            dtype=torch.float32,
            sigma_sec=1.0,
            window_sec=2.0,
        )
        shifted = build_local_timing_attention_mask(
            event_starts_sec=self.starts,
            event_ends_sec=self.ends,
            audio_durations_sec=self.duration,
            timing_mask=self.valid,
            query_len=20,
            dtype=torch.float32,
            sigma_sec=1.0,
            window_sec=2.0,
            shift_sec=2.0,
        )
        self.assertEqual(base.shape, shifted.shape)
        self.assertFalse(torch.equal(base, shifted))

    def test_event_weights_emphasize_events_and_normalize(self):
        weights = build_event_frame_weights(
            event_starts_sec=self.starts,
            event_ends_sec=self.ends,
            audio_durations_sec=self.duration,
            timing_mask=self.valid,
            query_len=100,
            extra_weight=2.0,
            margin_sec=0.0,
        )
        self.assertEqual(tuple(weights.shape), (1, 100))
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertGreater(float(weights.max()), float(weights.min()))

    def test_missing_events_fail_closed(self):
        with self.assertRaises(ValueError):
            build_local_timing_attention_mask(
                event_starts_sec=self.starts,
                event_ends_sec=self.ends,
                audio_durations_sec=self.duration,
                timing_mask=torch.zeros_like(self.valid),
                query_len=10,
                dtype=torch.float32,
                sigma_sec=1.0,
                window_sec=2.0,
            )


if __name__ == "__main__":
    unittest.main()
