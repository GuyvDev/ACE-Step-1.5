"""Tests for condition builder."""

import pytest
import torch
import numpy as np

from acestep.training_v2.phase_d_timing_v2.config import PhaseD2Config
from acestep.training_v2.phase_d_timing_v2.condition_builder import (
    build_dense_condition,
    build_condition_tensor,
)


@pytest.fixture
def config():
    """Fixture: default Phase D2 config."""
    return PhaseD2Config()


@pytest.fixture
def mock_events():
    """Fixture: mock event list."""
    return [
        {
            "section_id": 0,
            "phrase_id": 0,
            "word_id": 0,
            "phoneme_id": 0,
            "start_sec": 0.0,
            "end_sec": 1.0,
            "f0_hz": 100.0,
        },
        {
            "section_id": 0,
            "phrase_id": 0,
            "word_id": 1,
            "phoneme_id": 1,
            "start_sec": 1.0,
            "end_sec": 2.0,
            "f0_hz": 120.0,
        },
    ]


def test_build_dense_condition_basic(mock_events):
    """build_dense_condition returns valid sidecar."""
    sidecar = build_dense_condition(
        events=mock_events,
        beat_times=None,
        duration_sec=10.0,
        n_frames=250,  # 10 sec @ 25 Hz
    )

    assert "absolute_time" in sidecar
    assert len(sidecar["absolute_time"]) == 250
    assert sidecar["_schema_version"] == "phase_d2_sidecar_v1"
    assert sidecar["vocal_active"][:50].all()
    assert not sidecar["vocal_active"][50:].any()
    assert sidecar["pause"][50:].all()
    assert sidecar["word_start_pulse"].sum() == 2
    assert sidecar["word_end_pulse"].sum() == 2
    assert sidecar["target_word_duration"][0] == pytest.approx(1.0)
    assert sidecar["pos_in_word"][0] == pytest.approx(0.0)
    assert sidecar["pos_in_word"][24] == pytest.approx(1.0)


def test_build_dense_condition_beat_features(mock_events):
    """Beat inputs produce non-placeholder phase and tempo features."""
    sidecar = build_dense_condition(
        events=mock_events,
        beat_times=[0.0, 0.5, 1.0, 1.6, 2.2, 2.8],
        duration_sec=10.0,
        n_frames=250,
    )
    assert np.any(sidecar["beat_relative_time"] != 0)
    assert np.any(sidecar["bar_position"] != 0)
    assert np.any(sidecar["local_tempo"] != 1)
    assert sidecar["metadata"]["beat_alignment"] == "beat_times"


def test_build_dense_condition_rejects_invalid_event(mock_events):
    """Out-of-range events fail instead of being silently clipped."""
    invalid = [dict(mock_events[0], end_sec=11.0)]
    with pytest.raises(ValueError, match="Event interval"):
        build_dense_condition(invalid, None, duration_sec=10.0, n_frames=250)


def test_build_condition_tensor_shape(config, mock_events):
    """build_condition_tensor returns correct shape."""
    sidecar = build_dense_condition(
        events=mock_events,
        beat_times=None,
        duration_sec=10.0,
        n_frames=250,
        config=config,
    )

    cond_tensor, group_slices, audit = build_condition_tensor(sidecar, config)

    assert cond_tensor.shape == (1, 250, config.condition_dim)
    assert cond_tensor.dtype == torch.float32
    duration_start, duration_end = group_slices["explicit_duration"]
    assert duration_end - duration_start == 4
    assert torch.any(cond_tensor[:, :, duration_start:duration_end] != 0)


def test_condition_tensor_group_slices(config, mock_events):
    """Group slices match condition_dim."""
    sidecar = build_dense_condition(
        events=mock_events,
        beat_times=None,
        duration_sec=10.0,
        n_frames=100,
        config=config,
    )

    cond_tensor, group_slices, audit = build_condition_tensor(sidecar, config)

    # Reconstruct full dim from slices
    total_dim = sum(end - start for start, end in group_slices.values())
    assert total_dim == config.condition_dim


def test_condition_tensor_no_nan_inf(config, mock_events):
    """Condition tensor has no NaN or inf."""
    sidecar = build_dense_condition(
        events=mock_events,
        beat_times=None,
        duration_sec=10.0,
        n_frames=100,
        config=config,
    )

    cond_tensor, _, _ = build_condition_tensor(sidecar, config)

    assert not torch.isnan(cond_tensor).any()
    assert not torch.isinf(cond_tensor).any()


def test_disabled_groups_reduce_condition_dim(config, mock_events):
    """Disabling groups reduces condition_dim."""
    full_config = PhaseD2Config()
    full_dim = full_config.condition_dim

    # Disable some groups
    reduced_config = PhaseD2Config(
        linguistic=False,
        beat_relative_timing=False,
    )
    reduced_dim = reduced_config.condition_dim

    assert reduced_dim < full_dim


def test_condition_determinism(config, mock_events):
    """Same inputs produce identical condition tensors."""
    sidecar1 = build_dense_condition(
        events=mock_events,
        beat_times=None,
        duration_sec=10.0,
        n_frames=100,
        config=config,
    )

    sidecar2 = build_dense_condition(
        events=mock_events,
        beat_times=None,
        duration_sec=10.0,
        n_frames=100,
        config=config,
    )

    cond1, _, _ = build_condition_tensor(sidecar1, config)
    cond2, _, _ = build_condition_tensor(sidecar2, config)

    assert torch.allclose(cond1, cond2)
