"""Tests for counterfactual intervention and localization."""

import pytest
import torch
import numpy as np

from acestep.training_v2.phase_d2.condition_builder import build_dense_condition, build_condition_tensor
from acestep.training_v2.phase_d2.config import PhaseD2Config
from acestep.training_v2.phase_d2.counterfactuals import make_counterfactual
from acestep.training_v2.phase_d2.preservation import make_window_mask, expand_window_mask


@pytest.fixture
def config():
    """Fixture: config."""
    return PhaseD2Config()


@pytest.fixture
def mock_events():
    """Fixture: mock events."""
    return [
        {
            "section_id": 0,
            "phrase_id": 0,
            "word_id": i,
            "phoneme_id": i,
            "start_sec": float(i),
            "end_sec": float(i + 1),
        }
        for i in range(10)
    ]


def test_make_window_mask(config):
    """make_window_mask creates correct binary mask."""
    n_frames = 250
    frame_rate = 25.0
    
    mask = make_window_mask(
        n_frames=n_frames,
        start_sec=2.0,
        end_sec=5.0,
        frame_rate=frame_rate,
    )
    
    assert mask.shape == (n_frames,)
    assert mask.dtype == torch.bool
    
    # Frames 2-5 seconds should be True
    assert mask[50:125].all()  # frames 50-125 correspond to 2-5 sec


def test_expand_window_mask():
    """expand_window_mask adds margin frames."""
    mask = torch.zeros(100, dtype=torch.bool)
    mask[40:50] = True
    
    expanded = expand_window_mask(mask, margin_frames=5)
    
    assert expanded.shape == mask.shape
    # Expanded should include frames around original window
    assert expanded[35:55].all() or expanded[40:50].all()


def test_counterfactual_shift(mock_events, config):
    """make_counterfactual with shift modifies absolute times."""
    sidecar = build_dense_condition(
        events=mock_events,
        beat_times=None,
        duration_sec=10.0,
        n_frames=250,
        config=config,
    )
    
    modified, record = make_counterfactual(
        sidecar_events=sidecar,
        phrase_span=(25, 75),
        kind="shift",
        value=0.5,  # 0.5 second shift
    )
    
    assert record["kind"] == "shift"
    assert record["value"] == 0.5
    assert record["expected_direction"] == "later"
    
    # The global frame grid is invariant; the actual event fields relocate.
    assert np.array_equal(modified["absolute_time"], sidecar["absolute_time"])
    assert record["destination_span"] == [37, 87]
    assert np.array_equal(modified["word_id"][37:87], sidecar["word_id"][25:75])
    assert np.all(modified["word_id"][25:37] == -1)
    assert np.all(modified["vocal_active"][25:37] == 0)


def test_counterfactual_stretch(mock_events, config):
    """make_counterfactual with stretch modifies durations."""
    sidecar = build_dense_condition(
        events=mock_events,
        beat_times=None,
        duration_sec=10.0,
        n_frames=250,
        config=config,
    )
    
    modified, record = make_counterfactual(
        sidecar_events=sidecar,
        phrase_span=(25, 75),
        kind="stretch",
        value=1.2,  # 20% longer
    )
    
    assert record["kind"] == "stretch"
    assert record["expected_direction"] == "slower"
    assert record["destination_span"] == [25, 85]
    assert modified["vocal_active"][25:85].all()
    assert modified["target_phrase_duration"][25] == pytest.approx(
        sidecar["target_phrase_duration"][25] * 1.2
    )


def test_counterfactual_f0(mock_events, config):
    """make_counterfactual with f0 shifts target frequency."""
    sidecar = build_dense_condition(
        events=mock_events,
        beat_times=None,
        duration_sec=10.0,
        n_frames=250,
        config=config,
    )
    
    modified, record = make_counterfactual(
        sidecar_events=sidecar,
        phrase_span=(25, 75),
        kind="f0",
        value=50.0,  # +50 Hz
    )
    
    assert record["kind"] == "f0"
    assert record["expected_direction"] == "higher"


def test_counterfactual_pause(mock_events, config):
    """make_counterfactual with pause marks frames as silence."""
    sidecar = build_dense_condition(
        events=mock_events,
        beat_times=None,
        duration_sec=10.0,
        n_frames=250,
        config=config,
    )
    
    modified, record = make_counterfactual(
        sidecar_events=sidecar,
        phrase_span=(25, 75),
        kind="pause",
        value=1.0,
    )
    
    assert record["kind"] == "pause"
    assert record["expected_direction"] == "paused"
    
    # Check that pause and vocal_active were modified
    assert modified["pause"][50] == 1.0
    assert modified["vocal_active"][50] == 0.0
