"""ARCH_1 -- Audio processing: raw audio -> mel-spectrogram (T, 64) in dB.

Implements the documented pipeline with NumPy/SciPy only (no librosa
dependency):

    load -> resample -> RMS normalise -> fade -> bandpass -> STFT
         -> mel filterbank -> dB -> clip to [-80, 0]

Output contract (DATA_FLOW_REFERENCE): ``(T, 64)`` float32, values in
[-80, 0] dB, one frame per 10 ms.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy import signal
from scipy.io import wavfile

from .config import AudioConfig

__all__ = [
    "load_audio",
    "normalize_rms",
    "fade_audio",
    "bandpass_filter",
    "frame_audio",
    "compute_stft",
    "hz_to_mel",
    "mel_to_hz",
    "create_mel_filterbank",
    "magnitude_to_db",
    "extract_mel_spectrogram",
    "validate_mel_spectrogram",
]


# ---------------------------------------------------------------------------
# Step 1: loading & resampling
# ---------------------------------------------------------------------------


def _read_wav(path: Path) -> Tuple[np.ndarray, int]:
    """Read a WAV file into float32 in [-1, 1]."""
    try:
        sr, data = wavfile.read(str(path))
    except Exception:  # pragma: no cover - fallback for exotic WAV headers
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
            data = np.frombuffer(raw, dtype=np.int16).reshape(-1, wf.getnchannels())

    data = np.asarray(data)
    if data.ndim > 1:  # mix down to mono
        data = data.mean(axis=1)

    if np.issubdtype(data.dtype, np.integer):
        max_val = float(np.iinfo(data.dtype).max)
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)

    return data, int(sr)


def load_audio(
    filepath, target_sr: int = 16_000, max_duration_s: Optional[float] = 10.0
) -> Tuple[np.ndarray, int]:
    """Load audio and resample to ``target_sr``.

    Returns ``(audio_float32_mono, target_sr)``.
    """
    audio, native_sr = _read_wav(Path(filepath))

    if native_sr != target_sr:
        audio = resample(audio, native_sr, target_sr)

    if max_duration_s is not None:
        max_samples = int(max_duration_s * target_sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

    return audio.astype(np.float32), target_sr


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Polyphase resampling (rational ratio, anti-aliased)."""
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    from math import gcd

    g = gcd(int(orig_sr), int(target_sr))
    up, down = target_sr // g, orig_sr // g
    return signal.resample_poly(audio, up, down).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 2-4: amplitude conditioning
# ---------------------------------------------------------------------------


def normalize_rms(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """Scale audio so its RMS equals ``target_rms``."""
    audio = np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    if rms > 0:
        audio = audio * (target_rms / rms)
    return audio.astype(np.float32)


def fade_audio(audio: np.ndarray, fade_ms: float = 50.0, sr: int = 16_000) -> np.ndarray:
    """Apply a linear fade in/out to suppress boundary clicks."""
    audio = np.array(audio, dtype=np.float32, copy=True)
    fade_samples = int(fade_ms * sr / 1000.0)
    fade_samples = min(fade_samples, len(audio) // 2)
    if fade_samples <= 0:
        return audio
    audio[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    audio[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    return audio


def bandpass_filter(
    audio: np.ndarray,
    low_hz: float = 100.0,
    high_hz: float = 8_000.0,
    sr: int = 16_000,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass (``filtfilt``)."""
    nyquist = sr / 2.0
    low = max(low_hz / nyquist, 1e-6)
    high = min(high_hz / nyquist, 0.999999)
    if low >= high:
        return np.asarray(audio, dtype=np.float32)

    sos = signal.butter(order, [low, high], btype="band", output="sos")
    padlen = 3 * (2 * order)
    if len(audio) <= padlen:
        return np.asarray(audio, dtype=np.float32)
    return signal.sosfiltfilt(sos, audio).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 5-6: framing & STFT
# ---------------------------------------------------------------------------


def frame_audio(
    audio: np.ndarray, frame_length: int, hop_length: int, center: bool = True
) -> np.ndarray:
    """Split audio into overlapping frames -> ``(n_frames, frame_length)``.

    With ``center=True`` the signal is reflect-padded by ``frame_length // 2`` so
    frame ``t`` is centred on sample ``t * hop_length`` and
    ``n_frames = 1 + len(audio) // hop_length``.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if center:
        pad = frame_length // 2
        mode = "reflect" if len(audio) > pad else "constant"
        audio = np.pad(audio, (pad, pad), mode=mode)

    n_frames = 1 + max(len(audio) - frame_length, 0) // hop_length
    if n_frames <= 0:
        return np.zeros((0, frame_length), dtype=np.float32)

    shape = (n_frames, frame_length)
    strides = (audio.strides[0] * hop_length, audio.strides[0])
    frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)
    return np.ascontiguousarray(frames)


def compute_stft(frames: np.ndarray, n_fft: int = 512, window: str = "hann") -> np.ndarray:
    """Magnitude STFT of pre-framed audio -> ``(n_frames, n_fft // 2 + 1)``."""
    if frames.size == 0:
        return np.zeros((0, n_fft // 2 + 1), dtype=np.float32)

    win = signal.get_window(window, frames.shape[1], fftbins=True).astype(np.float32)
    windowed = frames * win

    if windowed.shape[1] < n_fft:
        pad = n_fft - windowed.shape[1]
        windowed = np.pad(windowed, ((0, 0), (0, pad)), mode="constant")
    elif windowed.shape[1] > n_fft:
        windowed = windowed[:, :n_fft]

    spectrum = np.fft.rfft(windowed, n=n_fft, axis=1)
    return np.abs(spectrum).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 7: mel filterbank
# ---------------------------------------------------------------------------


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def create_mel_filterbank(
    n_mels: int = 64,
    sr: int = 16_000,
    n_fft: int = 512,
    f_min: float = 50.0,
    f_max: float = 8_000.0,
) -> np.ndarray:
    """Triangular mel filterbank -> ``(n_mels, n_fft // 2 + 1)``.

    Uses frequency-domain interpolation (rather than integer bin slicing) so no
    filter is empty even at 64 bands over a 257-bin spectrum.
    """
    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sr / 2.0, n_bins)

    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_points = np.asarray(mel_to_hz(mel_points))

    filterbank = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(n_mels):
        left, center, right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        if right <= left:
            continue
        rising = (fft_freqs - left) / max(center - left, 1e-9)
        falling = (right - fft_freqs) / max(right - center, 1e-9)
        filterbank[m] = np.maximum(0.0, np.minimum(rising, falling))

    # Slaney-style area normalisation keeps band energies comparable.
    enorm = 2.0 / np.maximum(hz_points[2 : n_mels + 2] - hz_points[:n_mels], 1e-9)
    filterbank *= enorm[:, None].astype(np.float32)
    return filterbank


# ---------------------------------------------------------------------------
# Step 8: dB conversion
# ---------------------------------------------------------------------------


def magnitude_to_db(
    mel_spec: np.ndarray, top_db: float = 80.0, amin: float = 1e-10
) -> np.ndarray:
    """Power -> dB, referenced to the per-clip peak and floored at ``-top_db``.

    Referencing to the peak is what makes the documented [-80, 0] dB output
    range hold for any recording level.
    """
    power = np.square(np.asarray(mel_spec, dtype=np.float32))
    ref = max(float(power.max()) if power.size else 0.0, amin)
    db = 10.0 * np.log10(np.maximum(power, amin) / ref)
    return np.maximum(db, -top_db).astype(np.float32)


def normalize_db_spec(db_spec: np.ndarray) -> np.ndarray:
    """Zero-mean/unit-variance standardisation (optional; see AudioConfig)."""
    mean = float(np.mean(db_spec))
    std = float(np.std(db_spec))
    return ((db_spec - mean) / (std + 1e-8)).astype(np.float32)


# ---------------------------------------------------------------------------
# Complete pipeline
# ---------------------------------------------------------------------------


def extract_mel_spectrogram(
    audio_or_path,
    cfg: Optional[AudioConfig] = None,
    sr: Optional[int] = None,
    n_frames: Optional[int] = None,
) -> np.ndarray:
    """Raw audio (array or file path) -> mel-spectrogram ``(T, 64)`` in dB.

    Parameters
    ----------
    audio_or_path:
        A 1-D float array of audio samples, or a path to a WAV file.
    sr:
        Sample rate of ``audio_or_path`` when it is an array.  Ignored (and
        re-derived) when a path is given.
    n_frames:
        If set, the output is padded with the dB floor or truncated to exactly
        this many frames.
    """
    cfg = cfg or AudioConfig()

    if isinstance(audio_or_path, (str, Path)):
        audio, sr = load_audio(audio_or_path, cfg.sample_rate, cfg.max_duration_s)
    else:
        audio = np.asarray(audio_or_path, dtype=np.float32)
        sr = sr or cfg.sample_rate
        if sr != cfg.sample_rate:
            audio = resample(audio, sr, cfg.sample_rate)
            sr = cfg.sample_rate

    audio = normalize_rms(audio, cfg.target_rms)
    audio = fade_audio(audio, cfg.fade_ms, sr)
    audio = bandpass_filter(
        audio, cfg.bandpass_low_hz, cfg.bandpass_high_hz, sr, cfg.bandpass_order
    )

    frames = frame_audio(audio, cfg.frame_length, cfg.hop_length, center=True)
    stft_mag = compute_stft(frames, n_fft=cfg.n_fft)

    fb = create_mel_filterbank(cfg.n_mels, sr, cfg.n_fft, cfg.f_min, cfg.f_max)
    mel = stft_mag @ fb.T  # (T, n_mels)

    mel_db = magnitude_to_db(mel, top_db=abs(cfg.db_floor))
    mel_db = np.clip(mel_db, cfg.db_floor, cfg.db_ceiling)

    if cfg.standardize_db:
        mel_db = normalize_db_spec(mel_db)

    if n_frames is not None:
        mel_db = fit_frames(mel_db, n_frames, pad_value=cfg.db_floor)

    return np.ascontiguousarray(mel_db, dtype=np.float32)


def fit_frames(x: np.ndarray, n_frames: int, pad_value: float = -80.0) -> np.ndarray:
    """Pad (with ``pad_value``) or truncate the time axis to ``n_frames``."""
    T = x.shape[0]
    if T == n_frames:
        return x
    if T > n_frames:
        return x[:n_frames]
    pad_width = [(0, n_frames - T)] + [(0, 0)] * (x.ndim - 1)
    return np.pad(x, pad_width, mode="constant", constant_values=pad_value)


# ---------------------------------------------------------------------------
# Quality assurance (ARCH_1 "Validation Metrics")
# ---------------------------------------------------------------------------


def validate_mel_spectrogram(mel_db: np.ndarray, cfg: Optional[AudioConfig] = None) -> dict:
    """Run the ARCH_1 QA checks; returns a dict of booleans/stats."""
    cfg = cfg or AudioConfig()
    checks = {
        "shape_ok": mel_db.ndim == 2 and mel_db.shape[1] == cfg.n_mels,
        "no_nan": not bool(np.any(np.isnan(mel_db))),
        "no_inf": not bool(np.any(np.isinf(mel_db))),
        "db_range_ok": bool(
            np.all(mel_db >= cfg.db_floor - 1e-3) and np.all(mel_db <= cfg.db_ceiling + 1e-3)
        ),
        "n_frames": int(mel_db.shape[0]),
        "min_db": float(mel_db.min()) if mel_db.size else 0.0,
        "max_db": float(mel_db.max()) if mel_db.size else 0.0,
        "mean_db": float(mel_db.mean()) if mel_db.size else 0.0,
    }
    checks["passed"] = all(
        checks[k] for k in ("shape_ok", "no_nan", "no_inf", "db_range_ok")
    )
    return checks
