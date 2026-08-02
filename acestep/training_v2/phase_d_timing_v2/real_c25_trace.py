"""Matched intermediate tracing for the real C25 inference runtime."""

from __future__ import annotations

import hashlib
import types
from pathlib import Path
from typing import Any

import torch

from acestep.training_v2.phase_d_timing_v2.runtime_integration import unwrap_real_dit


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash exact contiguous CPU tensor bytes."""
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _tensor_audit(tensor: torch.Tensor) -> dict[str, Any]:
    """Return shape, dtype, and exact-byte digest for one trace tensor."""
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": _tensor_sha256(tensor),
        "finite": bool(torch.isfinite(tensor).all()),
    }


class RealC25TraceRecorder:
    """Record decoder flows, implied clean latents, and the pre-VAE latent."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.decoder = unwrap_real_dit(model.decoder)
        self.flow_predictions: list[torch.Tensor] = []
        self.predicted_clean_latents: list[torch.Tensor] = []
        self.timesteps: list[torch.Tensor] = []
        self.pre_vae_latent: torch.Tensor | None = None
        self._hook = self.decoder.register_forward_hook(
            self._record_decoder_call,
            with_kwargs=True,
        )
        self._original_generate_audio = model.generate_audio
        model.generate_audio = types.MethodType(self._wrapped_generate_audio, model)

    def _record_decoder_call(
        self,
        module: torch.nn.Module,
        args: tuple,
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        """Capture raw conditional/unconditional decoder branches per ODE step."""
        del module, args
        flow = output[0] if isinstance(output, tuple) else output
        hidden = kwargs.get("hidden_states")
        timestep = kwargs.get("timestep")
        if not all(isinstance(value, torch.Tensor) for value in (flow, hidden, timestep)):
            raise RuntimeError("real C25 trace requires flow, hidden_states, and timestep tensors")
        scale = timestep.reshape(-1, *([1] * (hidden.ndim - 1)))
        predicted_clean = hidden - flow * scale
        self.flow_predictions.append(flow.detach().cpu())
        self.predicted_clean_latents.append(predicted_clean.detach().cpu())
        self.timesteps.append(timestep.detach().cpu())

    def _wrapped_generate_audio(
        self,
        model_self: torch.nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Capture the final diffusion latent immediately before VAE decoding."""
        del model_self
        output = self._original_generate_audio(*args, **kwargs)
        latent = output.get("target_latents")
        if not isinstance(latent, torch.Tensor):
            raise RuntimeError("real C25 trace did not receive target_latents")
        self.pre_vae_latent = latent.detach().cpu()
        return output

    def save(self, path: str | Path) -> dict[str, Any]:
        """Write all mandatory trace stages and return their audit metadata."""
        if not self.flow_predictions or self.pre_vae_latent is None:
            raise RuntimeError("real C25 trace is incomplete")
        payload = {
            "schema": "phase_d2_real_c25_trace_v1",
            "flow_predictions": torch.stack(self.flow_predictions),
            "predicted_clean_latents": torch.stack(self.predicted_clean_latents),
            "timesteps": torch.stack(self.timesteps),
            "pre_vae_latent": self.pre_vae_latent,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, target)
        audit = {
            key: _tensor_audit(value)
            for key, value in payload.items()
            if isinstance(value, torch.Tensor)
        }
        audit.update({"schema": payload["schema"], "path": str(target)})
        return audit

    def close(self) -> None:
        """Restore the model method and remove the decoder hook."""
        self._hook.remove()
        self.model.generate_audio = self._original_generate_audio
