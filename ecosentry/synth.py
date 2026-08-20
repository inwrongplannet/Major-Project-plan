"""Synthetic forest audio generator.

The design documents reference ESC-50, UrbanSound8K and three sets of custom
field recordings (Corbett / Seshachalam / Sundarbans).  None of those are
redistributable with this repository, so this module produces physically
plausible stand-ins with the documented spectro-temporal signatures:

* **gunshot**  -- sub-10 ms broadband transient, fast exponential decay,
  energy concentrated 200 Hz - 4 kHz, plus a forest echo tail.
* **chainsaw** -- ~110 Hz two-stroke fundamental with a rich harmonic stack and
  blade-contact amplitude modulation; sustained across the clip.
* **vehicle**  -- low-frequency engine rumble (60-200 Hz) with broadband tyre
  noise, slowly varying, sustained.
* **ambient**  -- forest background only.

Per-forest backgrounds follow ARCH_3 Component 1 ("characteristics"): Corbett is
noisy (birds/insects/wind + reverb), Seshachalam is clean, Sundarbans carries
sustained water noise and rain bursts.

Swap :func:`build_corpus` for a real loader when the field data is available --
everything downstream consumes plain ``(audio, label, forest, source)`` tuples.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import signal
from scipy.io import wavfile

from .config import CLASS_MAP, DatasetConfig

__all__ = [
    "SynthSample",
    "make_gunshot",
    "make_chainsaw",
    "make_vehicle",
    "forest_background",
    "synthesize",
    "build_corpus",
    "write_wav",
]


@dataclass
class SynthSample:
    audio: np.ndarray
    label: int
    forest: str
    source: str
    sample_rate: int


# ---------------------------------------------------------------------------
# Background beds
# ---------------------------------------------------------------------------


def _bandlimited_noise(
    n: int, low: float, high: float, sr: int, rng: np.random.Generator
) -> np.ndarray:
    noise = rng.standard_normal(n).astype(np.float32)
    nyq = sr / 2.0
    low_n = max(low / nyq, 1e-4)
    high_n = min(high / nyq, 0.999)
    if low_n >= high_n:
        return noise
    sos = signal.butter(4, [low_n, high_n], btype="band", output="sos")
    return signal.sosfilt(sos, noise).astype(np.float32)


def _chirp_bursts(
    n: int,
    sr: int,
    rng: np.random.Generator,
    count: int,
    f_lo: float,
    f_hi: float,
    dur_s: float,
) -> np.ndarray:
    """Bird-like frequency sweeps sprinkled through the clip."""
    out = np.zeros(n, dtype=np.float32)
    burst_len = int(dur_s * sr)
    if burst_len <= 1:
        return out
    t = np.arange(burst_len) / sr
    for _ in range(count):
        start = rng.integers(0, max(n - burst_len, 1))
        f0 = rng.uniform(f_lo, f_hi)
        f1 = f0 * rng.uniform(0.6, 1.6)
        sweep = signal.chirp(t, f0=f0, f1=f1, t1=t[-1], method="quadratic")
        envelope = np.hanning(burst_len)
        out[start : start + burst_len] += (sweep * envelope).astype(np.float32)
    return out


def forest_background(
    n: int, sr: int, forest: str, rng: np.random.Generator
) -> np.ndarray:
    """Ambient bed characteristic of each ecosystem."""
    bg = 0.02 * _bandlimited_noise(n, 80, 7_500, sr, rng)

    if forest == "corbett":
        # Dense forest: insects (high band), birds (chirps), wind, reverb.
        bg += 0.05 * _bandlimited_noise(n, 3_000, 7_000, sr, rng)
        bg += 0.04 * _chirp_bursts(n, sr, rng, count=14, f_lo=1_800, f_hi=5_000, dur_s=0.18)
        bg += 0.05 * _bandlimited_noise(n, 60, 300, sr, rng)
        ir = np.exp(-np.arange(int(0.25 * sr)) / (0.06 * sr)).astype(np.float32)
        ir[0] = 1.0
        bg = signal.fftconvolve(bg, ir / ir.sum(), mode="same").astype(np.float32)

    elif forest == "seshachalam":
        # Open terrain: quiet, occasional distant bird, light wind.
        bg += 0.015 * _bandlimited_noise(n, 2_000, 6_000, sr, rng)
        bg += 0.02 * _chirp_bursts(n, sr, rng, count=5, f_lo=1_500, f_hi=3_500, dur_s=0.15)

    elif forest == "sundarbans":
        # Wetlands: sustained water noise + intermittent rain bursts.
        bg += 0.07 * _bandlimited_noise(n, 150, 1_200, sr, rng)
        rain = _bandlimited_noise(n, 1_500, 7_500, sr, rng)
        env = np.zeros(n, dtype=np.float32)
        for _ in range(int(rng.integers(2, 5))):
            start = int(rng.integers(0, max(n - 1, 1)))
            length = int(rng.uniform(0.8, 2.5) * sr)
            end = min(start + length, n)
            env[start:end] = np.hanning(max(end - start, 2))[: end - start]
        bg += 0.06 * rain * env
        bg += 0.03 * _bandlimited_noise(n, 40, 200, sr, rng)

    else:  # "mixed" -- synthetic/urban corpora (ESC-50, UrbanSound8K)
        bg += 0.03 * _bandlimited_noise(n, 200, 4_000, sr, rng)

    return bg.astype(np.float32)


# ---------------------------------------------------------------------------
# Threat signatures
# ---------------------------------------------------------------------------


def make_gunshot(
    n: int, sr: int, rng: np.random.Generator, onset_s: Optional[float] = None
) -> np.ndarray:
    """Impulsive muzzle blast + forest echo tail."""
    out = np.zeros(n, dtype=np.float32)
    onset_frac = onset_s / (n / sr) if onset_s is not None else rng.uniform(0.15, 0.75)
    onset = int(np.clip(onset_frac, 0.0, 0.95) * n)

    tail = n - onset
    if tail <= 16:
        return out

    t = np.arange(tail) / sr
    # Muzzle blast: broadband crack with a very fast decay.
    crack = _bandlimited_noise(tail, 200, 6_000, sr, rng) * np.exp(-t / rng.uniform(0.02, 0.05))
    # Low-frequency "thump" from the pressure wave.
    thump = np.sin(2 * np.pi * rng.uniform(70, 130) * t) * np.exp(-t / 0.05)
    body = (0.9 * crack + 0.5 * thump).astype(np.float32)

    # Echo returns off canopy/terrain.
    for delay_s, atten in ((0.09, 0.35), (0.21, 0.18), (0.38, 0.08)):
        d = int(delay_s * sr)
        if d < tail:
            body[d:] += atten * body[: tail - d]

    out[onset:] = body * rng.uniform(0.7, 1.0)
    return out


def make_chainsaw(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Two-stroke engine harmonics with blade-load amplitude modulation."""
    t = np.arange(n) / sr
    f0 = rng.uniform(95, 135)
    # Slow RPM wander.
    f_inst = f0 * (1.0 + 0.06 * np.sin(2 * np.pi * rng.uniform(0.15, 0.4) * t))
    phase = 2 * np.pi * np.cumsum(f_inst) / sr

    sig = np.zeros(n, dtype=np.float32)
    for k in range(1, 26):
        if f0 * k > 7_500:
            break
        sig += (1.0 / k**0.85) * np.sin(k * phase + rng.uniform(0, 2 * np.pi))

    # Blade contact modulation (~8-14 Hz) and cutting hiss.
    am = 0.65 + 0.35 * np.sin(2 * np.pi * rng.uniform(8, 14) * t)
    hiss = 0.25 * _bandlimited_noise(n, 2_500, 7_000, sr, rng)
    sig = (sig * am + hiss).astype(np.float32)

    # The saw is not always cutting for the whole window.
    envelope = np.ones(n, dtype=np.float32)
    if rng.random() < 0.5:
        start = int(rng.uniform(0.0, 0.35) * n)
        stop = int(rng.uniform(0.65, 1.0) * n)
        envelope = np.zeros(n, dtype=np.float32)
        seg = stop - start
        envelope[start:stop] = np.minimum(1.0, np.hanning(max(seg, 2))[:seg] * 2.0)

    return (sig * envelope / (np.abs(sig).max() + 1e-9)).astype(np.float32)


def make_vehicle(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Engine rumble + tyre/road noise, sustained and slowly varying."""
    t = np.arange(n) / sr
    f0 = rng.uniform(45, 90)
    phase = 2 * np.pi * f0 * t * (1.0 + 0.03 * np.sin(2 * np.pi * 0.08 * t))

    sig = np.zeros(n, dtype=np.float32)
    for k in range(1, 12):
        sig += (1.0 / k**1.3) * np.sin(k * phase + rng.uniform(0, 2 * np.pi))

    tyre = 0.5 * _bandlimited_noise(n, 400, 3_500, sr, rng)
    # Doppler-ish pass-by envelope.
    center = rng.uniform(0.3, 0.7) * n
    width = rng.uniform(0.25, 0.6) * n
    envelope = np.exp(-((np.arange(n) - center) ** 2) / (2 * width**2)).astype(np.float32)

    out = (sig + tyre) * envelope
    return (out / (np.abs(out).max() + 1e-9)).astype(np.float32)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

_GENERATORS = {
    0: make_gunshot,
    1: make_chainsaw,
    2: make_vehicle,
}


def synthesize(
    label: int,
    forest: str,
    duration_s: float = 10.0,
    sr: int = 16_000,
    rng: Optional[np.random.Generator] = None,
    snr_db: Optional[float] = None,
) -> np.ndarray:
    """Render one labelled clip against the given forest background."""
    rng = rng or np.random.default_rng()
    n = int(duration_s * sr)
    background = forest_background(n, sr, forest, rng)

    if label == CLASS_MAP["ambient"]:
        return (background / (np.abs(background).max() + 1e-9)).astype(np.float32)

    event = _GENERATORS[label](n, sr, rng)

    if snr_db is None:
        # Harder SNR in noisy ecosystems; poaching events are often distant.
        base = {"corbett": 6.0, "seshachalam": 12.0, "sundarbans": 4.0}.get(forest, 9.0)
        snr_db = base + rng.uniform(-4.0, 6.0)

    p_event = float(np.mean(event**2)) + 1e-12
    p_bg = float(np.mean(background**2)) + 1e-12
    scale = np.sqrt(p_bg * (10.0 ** (snr_db / 10.0)) / p_event)

    mix = event * scale + background
    peak = float(np.abs(mix).max())
    return (mix / (peak + 1e-9)).astype(np.float32)


def build_corpus(
    cfg: Optional[DatasetConfig] = None,
    duration_s: float = 10.0,
    sr: int = 16_000,
    seed: Optional[int] = None,
    n_samples: Optional[int] = None,
) -> List[SynthSample]:
    """Build the 600-sample catalogue described in ARCH_3 Component 1.

    Class mix per source follows the documented statistics; ``n_samples`` scales
    the whole corpus down proportionally for fast smoke runs.
    """
    cfg = cfg or DatasetConfig()
    rng = np.random.default_rng(cfg.seed if seed is None else seed)

    # Class weights per source (ARCH_3 "Source Datasets" / "Dataset Statistics").
    source_class_mix: Dict[str, Dict[int, float]] = {
        "ESC-50": {0: 0.30, 1: 0.30, 2: 0.30, 3: 0.10},
        "UrbanSound8K": {0: 0.70, 1: 0.10, 2: 0.20, 3: 0.00},
        "Corbett": {0: 0.35, 1: 0.30, 2: 0.20, 3: 0.15},
        "Seshachalam": {0: 0.20, 1: 0.45, 2: 0.25, 3: 0.10},
        "Sundarbans": {0: 0.30, 1: 0.25, 2: 0.20, 3: 0.25},
    }

    total = sum(cfg.source_counts.values())
    scale = 1.0 if n_samples is None else max(n_samples / total, 0.0)

    corpus: List[SynthSample] = []
    for source, count in cfg.source_counts.items():
        forest = cfg.source_forest[source]
        mix = source_class_mix[source]
        n_src = max(int(round(count * scale)), 4 if scale < 1 else count)

        labels = np.array(list(mix.keys()))
        probs = np.array(list(mix.values()), dtype=np.float64)
        probs = probs / probs.sum()
        drawn = rng.choice(labels, size=n_src, p=probs)

        for label in drawn:
            audio = synthesize(int(label), forest, duration_s, sr, rng)
            corpus.append(
                SynthSample(
                    audio=audio,
                    label=int(label),
                    forest=forest,
                    source=source,
                    sample_rate=sr,
                )
            )

    rng.shuffle(corpus)
    return corpus


def write_wav(path, audio: np.ndarray, sr: int = 16_000) -> Path:
    """Write float audio to a 16-bit PCM WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    wavfile.write(str(path), sr, (clipped * 32767).astype(np.int16))
    return path
