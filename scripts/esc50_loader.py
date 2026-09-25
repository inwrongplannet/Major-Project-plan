"""Loads ESC-50 clips mapped to the eco-sentry class taxonomy.

ESC-50 has NO gunshot category -- confirmed by inspecting meta/esc50.csv,
which lists all 50 categories and none of them is gun_shot (that category
exists in UrbanSound8K, a different dataset, not ESC-50). This loader
therefore only ever produces samples labelled chainsaw, vehicle, or
ambient. Anything that imports CLASS_MAP["gunshot"] from this module's
output will simply never see it -- that is correct and expected, not a bug
to fix here.
"""
from __future__ import annotations

import csv
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ecosentry.config import CLASS_MAP
from ecosentry.synth import SynthSample

#: ESC-50 category -> eco-sentry class name. Every key here must be an
#: exact category string from meta/esc50.csv's "category" column.
CATEGORY_TO_CLASS: Dict[str, str] = {
    "chainsaw": "chainsaw",
    "engine": "vehicle",
    "car_horn": "vehicle",
    "train": "vehicle",
    "airplane": "vehicle",
    "helicopter": "vehicle",
    "wind": "ambient",
    "rain": "ambient",
    "crickets": "ambient",
    "chirping_birds": "ambient",
    "crackling_fire": "ambient",
    "thunderstorm": "ambient",
}


def _read_wav_mono_f32(path: Path) -> Tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV file as float32 mono in [-1, 1]."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            audio = audio.reshape(-1, w.getnchannels()).mean(axis=1)
    return audio, sr


def load_esc50_corpus(
    root: Path, max_per_category: Optional[int] = None
) -> Tuple[List[SynthSample], Dict[str, int]]:
    """Load every ESC-50 clip whose category is in CATEGORY_TO_CLASS.

    root must be the ESC-50 repository root (the directory containing
    meta/esc50.csv and audio/). Every returned SynthSample has
    forest="mixed" -- arch3_dataset.py's normalize_mixed_spikes already
    exists specifically for real-world corpora with no ecosystem prior
    (its docstring literally says "ESC-50 / UrbanSound8K"), so this is not
    a new concept being introduced, it is wiring up a hook that was already
    built for exactly this purpose.
    """
    root = Path(root)
    meta_path = root / "meta" / "esc50.csv"
    with open(meta_path) as f:
        rows = list(csv.DictReader(f))

    samples: List[SynthSample] = []
    counts: Dict[str, int] = {}
    for row in rows:
        category = row["category"]
        if category not in CATEGORY_TO_CLASS:
            continue
        if max_per_category is not None and counts.get(category, 0) >= max_per_category:
            continue
        wav_path = root / "audio" / row["filename"]
        audio, sr = _read_wav_mono_f32(wav_path)
        our_class = CATEGORY_TO_CLASS[category]
        samples.append(
            SynthSample(
                audio=audio,
                label=CLASS_MAP[our_class],
                forest="mixed",
                source=f"ESC-50:{category}:fold{row['fold']}",
                sample_rate=sr,
            )
        )
        counts[category] = counts.get(category, 0) + 1
    return samples, counts
