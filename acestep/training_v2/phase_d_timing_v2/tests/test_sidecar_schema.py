"""Tests for sidecar schema validation."""

import pytest
import torch
import numpy as np

from acestep.training_v2.phase_d_timing_v2.sidecar_schema import validate_sidecar, REQUIRED_FIELDS


@pytest.fixture
def valid_sidecar():
    """Fixture: valid minimal sidecar dict."""
    n_frames = 100
    sidecar = {
        field: np.zeros(n_frames, dtype=np.float32)
        for field in REQUIRED_FIELDS
    }
    sidecar["_schema_version"] = "phase_d2_sidecar_v1"
    return sidecar


def test_valid_sidecar(valid_sidecar):
    """Valid sidecar passes validation."""
    validate_sidecar(valid_sidecar)  # Should not raise


def test_missing_required_field(valid_sidecar):
    """Missing required field raises KeyError."""
    del valid_sidecar["absolute_time"]
    with pytest.raises(KeyError, match="Missing required fields"):
        validate_sidecar(valid_sidecar)


def test_wrong_schema_version(valid_sidecar):
    """Wrong schema version raises ValueError."""
    valid_sidecar["_schema_version"] = "phase_d2_sidecar_v999"
    with pytest.raises(ValueError, match="Schema version mismatch"):
        validate_sidecar(valid_sidecar)


def test_nan_in_field(valid_sidecar):
    """NaN in field raises ValueError."""
    valid_sidecar["absolute_time"][0] = float("nan")
    with pytest.raises(ValueError, match="contains NaN"):
        validate_sidecar(valid_sidecar)


def test_inf_in_field(valid_sidecar):
    """Inf in field raises ValueError."""
    valid_sidecar["absolute_time"][0] = float("inf")
    with pytest.raises(ValueError, match="contains inf"):
        validate_sidecar(valid_sidecar)


def test_shape_mismatch(valid_sidecar):
    """Field with wrong first dimension raises ValueError."""
    valid_sidecar["absolute_time"] = np.zeros(50, dtype=np.float32)
    with pytest.raises(ValueError, match="has shape"):
        validate_sidecar(valid_sidecar, n_frames=100)


def test_tensor_input(valid_sidecar):
    """Sidecar with torch.Tensor fields passes validation."""
    for field in REQUIRED_FIELDS:
        valid_sidecar[field] = torch.from_numpy(valid_sidecar[field])
    validate_sidecar(valid_sidecar)


def test_list_input(valid_sidecar):
    """Sidecar with list fields passes validation."""
    for field in REQUIRED_FIELDS:
        valid_sidecar[field] = valid_sidecar[field].tolist()
    validate_sidecar(valid_sidecar)
