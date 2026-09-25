"""Loads UrbanSound8K clips mapped to the eco-sentry class taxonomy.

Unlike ESC-50, UrbanSound8K files are pre-sorted into ten fold subfolders
(fold1/ .. fold10/), and files vary in native sample rate, bit
depth, and channel count from file to file (the dataset's own
documentation states this explicitly). This loader therefore reuses
ecosentry.arch1_audio._read_wav -- the same robust reader the rest of the
pipeline already trusts for exactly this reason (it tries scipy.io.wavfile
first, falls back to the stdlib wave module for exotic headers, and
normalizes correctly regardless of source bit depth) -- rather than
re-implementing WAV reading with an assumption (like 16-bit-only) that
happened to work for ESC-50 but would silently misdecode a fraction of
UrbanSound8K's files.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ecosentry.arch1_audio import _read_wav
from ecosentry.config import CLASS_MAP
from ecosentry.synth import SynthSample

#: UrbanSound8K "class" column value -> eco-sentry class name. Every key
#: here must be an exact string from UrbanSound8K.csv's "class" column.
#: chainsaw has no entry -- UrbanSound8K has no chainsaw-like category,
#: same honest gap treatment as ESC-50's missing gun_shot category.
CATEGORY_TO_CLASS: Dict[str, str] = {
    "gun_shot": "gunshot",
    "car_horn": "vehicle",
    "engine_idling": "vehicle",
    "siren": "vehicle",
    "air_conditioner": "ambient",
    "street_music": "ambient",
}

#: Deliberately excluded UrbanSound8K categories and why -- listed so a
#: reader of this file doesn't wonder if they were forgotten.
EXCLUDED_CATEGORIES = {
    "children_playing": "discrete event, not ambient background or a threat class",
    "dog_bark": "discrete event, not ambient background or a threat class",
    "drilling": "mechanically closer to construction noise than any of our 4 classes",
    "jackhammer": "mechanically closer to construction noise than any of our 4 classes",
}


def load_urbansound8k_corpus(
    root: Path, max_per_category: Optional[int] = None
) -> Tuple[List[SynthSample], Dict[str, int], List[Tuple[str, str]]]:
    """Load every UrbanSound8K clip whose class is in CATEGORY_TO_CLASS.

    root must be the UrbanSound8K root directory containing fold1..fold10 and UrbanSound8K.csv.

    Returns (samples, counts, skipped) where skipped is a list of
    (filename, reason) pairs for any file that failed to read -- this is
    expected to happen occasionally on a real-world dataset this size and
    is handled by skipping, not crashing, but every skip is reported so
    nothing silently vanishes from the final counts.
    """
    root = Path(root)
    meta_path = root / "UrbanSound8K.csv"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No UrbanSound8K.csv found at {meta_path}. "
        )

    with open(meta_path) as f:
        rows = list(csv.DictReader(f))

    samples: List[SynthSample] = []
    counts: Dict[str, int] = {}
    skipped: List[Tuple[str, str]] = []
    for row in rows:
        category = row["class"]
        if category not in CATEGORY_TO_CLASS:
            continue
        if max_per_category is not None and counts.get(category, 0) >= max_per_category:
            continue
        fold = row["fold"]
        filename = row["slice_file_name"]
        wav_path = root / f"fold{fold}" / filename
        try:
            audio, sr = _read_wav(wav_path)
        except Exception as exc:  # real-world datasets have occasional bad files
            skipped.append((filename, f"{type(exc).__name__}: {exc}"))
            continue
        our_class = CATEGORY_TO_CLASS[category]
        samples.append(
            SynthSample(
                audio=audio,
                label=CLASS_MAP[our_class],
                forest="mixed",
                source=f"UrbanSound8K:{category}:fold{fold}",
                sample_rate=sr,
            )
        )
        counts[category] = counts.get(category, 0) + 1
    return samples, counts, skipped
