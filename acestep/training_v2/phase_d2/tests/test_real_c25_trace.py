"""Tests for matched real-C25 intermediate tracing."""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from acestep.training_v2.phase_d2.real_c25_trace import RealC25TraceRecorder


class TinyDecoder(nn.Module):
    """Real-DiT-shaped decoder returning a constant flow."""

    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 2})()
        self.patch_size = 1
        self.layers = nn.ModuleList([nn.Identity()])

    def forward(self, hidden_states, timestep, **kwargs):
        del kwargs
        return torch.ones_like(hidden_states), None


class TinyModel(nn.Module):
    """Generation-shaped model invoking its decoder once."""

    def __init__(self):
        super().__init__()
        self.decoder = TinyDecoder()

    def generate_audio(self, hidden_states, timestep):
        flow = self.decoder(hidden_states=hidden_states, timestep=timestep)[0]
        return {"target_latents": hidden_states - flow}


def test_trace_records_flow_clean_and_pre_vae_latent():
    """Recorder persists every tensor required by the D2 null gate."""
    model = TinyModel()
    recorder = RealC25TraceRecorder(model)
    hidden = torch.full((2, 3, 2), 4.0)
    output = model.generate_audio(hidden, torch.tensor([0.5, 0.5]))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.pt"
        audit = recorder.save(path)
        payload = torch.load(path, weights_only=True)
    recorder.close()
    assert torch.equal(payload["flow_predictions"], torch.ones(1, 2, 3, 2))
    assert torch.equal(payload["predicted_clean_latents"], torch.full((1, 2, 3, 2), 3.5))
    assert torch.equal(payload["pre_vae_latent"], output["target_latents"])
    assert audit["schema"] == "phase_d2_real_c25_trace_v1"
