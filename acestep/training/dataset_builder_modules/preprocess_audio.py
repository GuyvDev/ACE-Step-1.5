import torch
import torchaudio
from loguru import logger

_FALLBACK_LOGGED = False


def load_audio_stereo(audio_path: str, target_sample_rate: int, max_duration: float):
    """Load audio, resample, convert to stereo, and truncate."""
    global _FALLBACK_LOGGED
    try:
        import soundfile as sf

        samples, sr = sf.read(audio_path, always_2d=True)
        audio = torch.from_numpy(samples.T).float()
    except Exception:
        # Keep a last-resort fallback to torchaudio for uncommon soundfile failures.
        if not _FALLBACK_LOGGED:
            logger.warning("soundfile read failed; falling back to torchaudio.load for preprocessing")
            _FALLBACK_LOGGED = True
        audio, sr = torchaudio.load(audio_path)

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
