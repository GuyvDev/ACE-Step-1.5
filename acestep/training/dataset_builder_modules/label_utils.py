from typing import Any, Optional

from loguru import logger
import torch


_CUDA_CAPTURE_ERROR_TOKENS = (
    "stream is capturing",
    "graph capture",
    "outside graph capture",
)


def _is_cuda_capture_failure(error_text: str) -> bool:
    """Return whether an error message looks like a CUDA graph-capture failure."""
    if not error_text:
        return False
    lowered = error_text.lower()
    return any(token in lowered for token in _CUDA_CAPTURE_ERROR_TOKENS)


def _convert_audio_to_codes_deterministic(audio_path: str, dit_handler) -> Optional[str]:
    """Retry audio-code extraction with deterministic VAE latents.

    This path avoids posterior sampling in VAE encode, which can trigger CUDA
    graph-capture runtime errors on some setups.
    """
    processed_audio = dit_handler.process_src_audio(audio_path)
    if processed_audio is None:
        return None

    if dit_handler.is_silence(processed_audio.unsqueeze(0)):
        return None

    with torch.inference_mode():
        with dit_handler._load_model_context("vae"):
            input_audio = processed_audio
            if input_audio.dim() == 2:
                input_audio = input_audio.unsqueeze(0)
            vae_input = input_audio.to(dit_handler.device).to(dit_handler.vae.dtype)
            latent_dist = dit_handler.vae.encode(vae_input).latent_dist
            latents = latent_dist.mode()
            latents = latents.to(dit_handler.device).to(dit_handler.dtype).transpose(1, 2)
            if processed_audio.dim() == 2:
                latents = latents.squeeze(0)

        attention_mask = torch.ones(latents.shape[0], dtype=torch.bool, device=dit_handler.device)
        with dit_handler._load_model_context("model"):
            hidden_states = latents.unsqueeze(0)
            _, indices, _ = dit_handler.model.tokenize(
                hidden_states,
                dit_handler.silence_latent,
                attention_mask.unsqueeze(0),
            )
            indices_flat = indices.flatten().cpu().tolist()
            if not indices_flat:
                return None
            return "".join([f"<|audio_code_{idx}|>" for idx in indices_flat])


def get_audio_codes(audio_path: str, dit_handler) -> Optional[str]:
    """Encode audio to get semantic codes for LLM understanding."""
    try:
        if not hasattr(dit_handler, "convert_src_audio_to_codes"):
            logger.error("DiT handler missing convert_src_audio_to_codes method")
            return None

        codes_string = dit_handler.convert_src_audio_to_codes(audio_path)

        if codes_string and not codes_string.startswith("❌"):
            return codes_string

        if isinstance(codes_string, str) and _is_cuda_capture_failure(codes_string):
            logger.warning(
                "CUDA graph-capture error detected while encoding audio; retrying with deterministic VAE mode"
            )
            retry_codes = _convert_audio_to_codes_deterministic(audio_path, dit_handler)
            if retry_codes:
                return retry_codes

        logger.warning(f"Failed to convert audio to codes: {codes_string}")
        return None
    except Exception:
        logger.exception(f"Error encoding audio {audio_path}")
        return None


def parse_int(value: Any) -> Optional[int]:
    """Safely parse an integer value."""
    if value is None or value == "N/A" or value == "":
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None
