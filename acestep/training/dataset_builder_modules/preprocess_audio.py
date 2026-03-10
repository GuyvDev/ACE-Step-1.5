import torch
import torchaudio
from loguru import logger

_FALLBACK_LOGGED = False


def load_audio_stereo(audio_path: str, target_sample_rate: int, max_duration: float):
    """Load audio, resample, convert to stereo, and truncate."""
    global _FALLBACK_LOGGED
    try:
        audio, sr = torchaudio.load(audio_path)
    except Exception:
        # torchcodec can fail on some CUDA/libnvrtc combinations; soundfile is a safe fallback.
        if not _FALLBACK_LOGGED:
            logger.warning("torchaudio.load unavailable in this env; using soundfile fallback for preprocessing")
            _FALLBACK_LOGGED = True
        import soundfile as sf

        samples, sr = sf.read(audio_path, always_2d=True)
        audio = torch.from_numpy(samples.T).float()

    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)

    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2, :]

    max_samples = int(max_duration * target_sample_rate)
    if audio.shape[1] > max_samples:
        audio = audio[:, :max_samples]

    return audio, sr
