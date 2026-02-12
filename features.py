"""Feature extraction helpers.

Contains functions to compute log-Mel spectrograms and MFCCs using torchaudio
if available, plus simple augmentations (SpecAugment, noise, speed perturb).
"""
from typing import Optional
import numpy as np

try:
    import torchaudio
    import torch
    from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, MFCC
except Exception:
    torchaudio = None
    torch = None


def _ensure_tensor(waveform, sr=16000):
    """Return a 1D torch tensor normalized to float32 with shape (1, samples)."""
    if torch is not None and isinstance(waveform, torch.Tensor):
        w = waveform
        if w.dim() == 2 and w.size(0) > 1:
            # keep first channel
            w = w[0:1]
        elif w.dim() == 1:
            w = w.unsqueeze(0)
        return w.float()
    else:
        arr = np.asarray(waveform).flatten().astype(np.float32)
        try:
            return torch.from_numpy(arr).unsqueeze(0)
        except Exception:
            # fallback: create a zero tensor
            if torch is not None:
                return torch.zeros((1, sr * 3), dtype=torch.float32)
            else:
                return arr


def waveform_to_logmel(waveform, sr=16000, n_mels=80, n_fft=1024, hop_length=160):
    """Convert waveform (Tensor or ndarray) to log-Mel spectrogram.

    Returns a numpy array (n_mels, time_frames).
    """
    if torch is not None:
        w = _ensure_tensor(waveform, sr=sr)
        try:
            spec = MelSpectrogram(sample_rate=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)(w)
            spec_db = AmplitudeToDB()(spec)
            return spec_db.squeeze(0).cpu().numpy()
        except Exception:
            # torchaudio may not be available or failed; fallback to zeros
            return np.zeros((n_mels, max(1, int(w.shape[1] / hop_length))))
    else:
        waveform = np.asarray(waveform).flatten()
        return np.zeros((n_mels, int(len(waveform) / hop_length) + 1), dtype=np.float32)


def waveform_to_mfcc(waveform, sr=16000, n_mfcc=13, n_mels=40, n_fft=1024, hop_length=160):
    """Compute MFCCs; returns numpy array (n_mfcc, time_frames)."""
    if torch is not None:
        w = _ensure_tensor(waveform, sr=sr)
        try:
            mfcc_transform = MFCC(sample_rate=sr, n_mfcc=n_mfcc, n_mels=n_mels, melkwargs={"n_fft": n_fft, "hop_length": hop_length})
            mfcc = mfcc_transform(w)
            return mfcc.squeeze(0).cpu().numpy()
        except Exception:
            return np.zeros((n_mfcc, max(1, int(w.shape[1] / hop_length))))
    else:
        return np.zeros((n_mfcc, 1), dtype=np.float32)


def spec_augment(spec, time_mask_param=10, freq_mask_param=8):
    """Apply simple SpecAugment on a numpy array spec (n_mels, frames)."""
    spec = spec.copy()
    n_mels, frames = spec.shape
    # freq mask
    f = np.random.randint(0, freq_mask_param + 1)
    f0 = np.random.randint(0, max(1, n_mels - f))
    spec[f0 : f0 + f, :] = 0
    # time mask
    t = np.random.randint(0, time_mask_param + 1)
    t0 = np.random.randint(0, max(1, frames - t))
    spec[:, t0 : t0 + t] = 0
    return spec
