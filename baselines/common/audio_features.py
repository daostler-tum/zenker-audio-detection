from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import librosa
import numpy as np


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    clip_duration_s: float = 1.0


def load_audio_mono(path: str, *, cfg: AudioConfig) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=cfg.sample_rate, mono=True)
    target_len = int(round(cfg.clip_duration_s * cfg.sample_rate))
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    elif len(y) > target_len:
        y = y[:target_len]
    return y.astype(np.float32), cfg.sample_rate


def log_mel_spectrogram(
    y: np.ndarray,
    *,
    sr: int,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 256,
    fmin: float = 20.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def mfcc_features(
    y: np.ndarray,
    *,
    sr: int,
    n_mfcc: int = 20,
    n_fft: int = 1024,
    hop_length: int = 256,
    include_deltas: bool = True,
) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    feats = [mfcc]
    if include_deltas:
        feats.append(librosa.feature.delta(mfcc))
        feats.append(librosa.feature.delta(mfcc, order=2))
    return np.concatenate(feats, axis=0).astype(np.float32)


def aggregate_time_stats(feat_t: np.ndarray) -> np.ndarray:
    """Aggregate time-varying features into a fixed vector (mean + std over time)."""

    mean = np.mean(feat_t, axis=1)
    std = np.std(feat_t, axis=1)
    return np.concatenate([mean, std], axis=0).astype(np.float32)
