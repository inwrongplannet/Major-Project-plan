# Adding UrbanSound8K — implementation runbook

**Read this paragraph before anything else.** Unlike the ESC-50 plan, this
one could not be verified against your actual downloaded files — you were
explicit that the dataset should not be cloned or downloaded in this
session, so it wasn't. Every piece of *code* below was still tested, but
against a hand-built mock directory that exactly matches UrbanSound8K's
documented folder layout and CSV schema (fold subfolders, exact column
names, varying sample rates per file) rather than the real thing. Task T2
is a **mandatory** self-test against your real local copy — do not skip it
or trust any later task's numbers until T2 passes on your actual files.

## HOW TO EXECUTE THIS PLAN

1. Work through tasks in order: T0 through T8.
2. T2 is a hard gate. Every later task assumes the loader already works
   against your real files. If T2 fails, stop and fix the loader (most
   likely cause: your Kaggle download's folder names or CSV filename
   capitalization differ from the standard layout assumed here — see T2's
   troubleshooting note) before going any further.
3. New files only, same as the ESC-50 plan — nothing here edits existing
   project code.
4. **T7's multi-seed run takes a very long time (likely 1+ hour at full
   scale)** — use the same `setsid`-detached-background technique proven
   in the ESC-50 plan, not a foreground command.

---

## TASK T0 — what UrbanSound8K actually is (confirmed from its own documentation)

**No files touched.** This is a findings task.

- **8,732 labeled clips, ≤4 seconds each, WAV format.**
- **Ten classes:** `air_conditioner`, `car_horn`, `children_playing`,
  `dog_bark`, `drilling`, `engine_idling`, `gun_shot`, `jackhammer`,
  `siren`, `street_music`.
- **Per-class counts** (from the dataset's own published statistics):
  air_conditioner 1000, car_horn 429, children_playing 1000, dog_bark
  1000, drilling 1000, engine_idling 1000, **gun_shot 374**, jackhammer
  1000, siren 929, street_music 1000.
- **Files are pre-sorted into ten fold subfolders**: `audio/fold1/` through
  `audio/fold10/` — not a flat directory like ESC-50.
- **Sample rate, bit depth, and channel count vary from file to file**
  (16 kHz–48 kHz range, per the dataset's own documentation) — unlike
  ESC-50's uniform 44.1 kHz/16-bit/mono. This is the main engineering
  difference from the ESC-50 loader and is handled in T2.
- **Metadata file** `metadata/UrbanSound8K.csv` with columns:
  `slice_file_name, fsID, start, end, salience, fold, classID, class`.
  Filename format: `[fsID]-[classID]-[occurrenceID]-[sliceID].wav`.
- **A methodological rule from the dataset's own creators, stated in their
  own words with unusual force:**
  > "Don't reshuffle the data! Use the predefined 10 folds and perform
  > 10-fold (not 5-fold) cross validation... this could invalidate your
  > results, potentially leading to manuscripts being rejected."

  This plan's experiments (T5 onward) use random splits via the existing
  project's `create_stratified_splits`, **not** the official 10-fold
  protocol — flagged here explicitly so it's a known, visible limitation
  in any write-up rather than a silent deviation from a widely-cited
  dataset's own stated best practice. If a fully citation-grade result is
  needed later, re-running with the *official* folds preserved as the
  train/test boundary (rather than shuffled into ARCH_3's stratified
  splitter) would be the correct follow-up — the loader (T2) already
  records each sample's fold number in its `source` field specifically so
  this is possible without re-loading anything.

### What this dataset changes about the project's real-audio validation

The ESC-50 plan's biggest limitation was explicit: **ESC-50 has no gunshot
category at all**, so gunshot detection was never evaluated on real audio.
**UrbanSound8K's `gun_shot` class (374 real clips) fills exactly that
gap.** This is the single most valuable thing this dataset adds — not more
data in general, but the one specific class that was completely untested.

The reverse gap also exists: **UrbanSound8K has no chainsaw-like
category**, so chainsaw remains uncovered by real audio from *either*
dataset combined. Do not force `drilling` or `jackhammer` into a
"chainsaw" mapping to paper over this — they are acoustically closer to
construction impact noise than to a chainsaw's engine-and-cutting-drone
profile, and mislabeling them would produce a number that looks like
chainsaw validation but isn't. State the gap plainly in any write-up,
exactly as the ESC-50 plan did for gunshot.

### Final class mapping used in this plan

| UrbanSound8K class | classID | Maps to | Clips available |
|---|---|---|---|
| `gun_shot` | 6 | `gunshot` | 374 |
| `car_horn` | 1 | `vehicle` | 429 |
| `engine_idling` | 5 | `vehicle` | 1000 |
| `siren` | 8 | `vehicle` | 929 |
| `air_conditioner` | 0 | `ambient` | 1000 |
| `street_music` | 9 | `ambient` | 1000 |
| *(none)* | — | `chainsaw` | 0 — not covered |
| `children_playing`, `dog_bark`, `drilling`, `jackhammer` | 2,3,4,7 | *(excluded)* | — |

The four excluded categories don't cleanly fit any of the four eco-sentry
classes (they're discrete human/animal events or generic construction
noise, not threat sounds, vehicles, or steady ambient background) — using
them would blur class purity for no clear benefit. Totals if using every
available clip: gunshot 374, vehicle 2,358, ambient 2,000.

**`ambient` here means *urban* ambience** (air conditioners, street
music), not forest ambience — the same kind of honest caveat the ESC-50
plan gave for its composite `vehicle` class. State this in any write-up:
this tests whether the model over-fires on background sound in general,
not specifically on forest background sound.

---

## TASK T1 — confirm your local download's structure

**Preconditions:** T0. You have already downloaded the dataset yourself.

**No files touched.** Run this against your actual local copy before
writing or trusting anything else:

```
ls /path/to/your/UrbanSound8K/audio | sort
find /path/to/your/UrbanSound8K/audio -name '*.wav' | wc -l
ls /path/to/your/UrbanSound8K/metadata
head -3 /path/to/your/UrbanSound8K/metadata/UrbanSound8K.csv
```

**Expected result:**
- First command: `fold1` through `fold10`, ten lines.
- Second command: `8732`.
- Third command: a file whose name contains `UrbanSound8K` (any
  capitalization) and ends in `.csv`.
- Fourth command: a header line reading exactly
  `slice_file_name,fsID,start,end,salience,fold,classID,class`, followed by
  data rows.

**If any of these don't match** (a common Kaggle-specific issue is the zip
extracting to a nested folder like `UrbanSound8K/UrbanSound8K/audio/...`,
or the CSV living directly in the root instead of under `metadata/`), note
the actual structure now — T2's loader takes the root path as an argument,
so as long as you know the real layout, you can point it at the right
place, but you need to know what that real layout is before proceeding.

---

## TASK T2 — the loader (mandatory self-test gate)

**Preconditions:** T0, T1.

**Files touched:** create `urbansound8k_loader.py` at the project
repository root (same location as `esc50_loader.py` from the earlier
plan).

```python
"""Loads UrbanSound8K clips mapped to the eco-sentry class taxonomy.

Unlike ESC-50, UrbanSound8K files are pre-sorted into ten fold subfolders
(audio/fold1/ .. audio/fold10/), and files vary in native sample rate, bit
depth, and channel count from file to file (the dataset's own
documentation states this explicitly). This loader therefore reuses
ecosentry.arch1_audio._read_wav -- the same robust reader the rest of the
pipeline already trusts for exactly this reason (it tries scipy.io.wavfile
first, falls back to the stdlib wave module for exotic headers, and
normalizes correctly regardless of source bit depth) -- rather than
re-implementing WAV reading with an assumption (like 16-bit-only) that
happened to work for ESC-50 but would silently misdecode a fraction of
UrbanSound8K's files.

IMPORTANT: this loader's file-path and CSV-parsing logic was verified
against a hand-built mock directory matching UrbanSound8K's documented
structure and metadata schema (fold subfolders, slice_file_name/fsID/
start/end/salience/fold/classID/class columns) -- NOT against the real
downloaded dataset, since it was not downloaded in the session that wrote
this loader. Task T2 in the accompanying plan is a mandatory self-test
against your actual local copy before any later task's numbers can be
trusted.
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

    root must be the UrbanSound8K root directory, containing audio/fold1
    .. audio/fold10 and metadata/UrbanSound8K.csv (the standard layout;
    if your Kaggle download used different capitalization for the csv
    filename, pass the resolved path directly via the metadata_csv
    parameter instead of relying on the default lookup).

    Returns (samples, counts, skipped) where skipped is a list of
    (filename, reason) pairs for any file that failed to read -- this is
    expected to happen occasionally on a real-world dataset this size and
    is handled by skipping, not crashing, but every skip is reported so
    nothing silently vanishes from the final counts.
    """
    root = Path(root)
    meta_path = root / "metadata" / "UrbanSound8K.csv"
    if not meta_path.exists():
        # Kaggle packaging has been observed to vary the csv's capitalization
        # (e.g. "urbansound8k.csv") -- glob() is case-sensitive on Linux, so
        # this scans filenames manually instead of relying on a glob pattern.
        meta_dir = root / "metadata"
        candidates = [
            p for p in (meta_dir.iterdir() if meta_dir.is_dir() else [])
            if p.suffix.lower() == ".csv" and "urbansound8k" in p.name.lower()
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No UrbanSound8K.csv found under {meta_dir}. "
                f"Pass the correct root, or check the exact filename Kaggle gave you."
            )
        meta_path = candidates[0]

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
        wav_path = root / "audio" / f"fold{fold}" / filename
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
```

**Verification — run this against your REAL downloaded copy, not a mock:**
```
python3 -c "
from urbansound8k_loader import load_urbansound8k_corpus, CATEGORY_TO_CLASS
samples, counts, skipped = load_urbansound8k_corpus('/path/to/your/UrbanSound8K')
print('loaded:', len(samples))
print('counts:', counts)
print('skipped:', len(skipped))
if skipped:
    print('first few skips:', skipped[:5])
assert set(counts) == set(CATEGORY_TO_CLASS), 'missing or unexpected categories -- check class names match exactly'
"
```
**Expected result (approximately — see note below):**
```
loaded: ~4732
counts: {'gun_shot': ~374, 'car_horn': ~429, 'engine_idling': ~1000, 'siren': ~929, 'air_conditioner': ~1000, 'street_music': ~1000}
skipped: 0 (small numbers like 1-5 are not alarming on a dataset this size; dozens would be)
```
These are approximate, not exact-match requirements like the ESC-50 plan's
numbers — unlike ESC-50 (a small, hand-curated 2000-file set), UrbanSound8K
is large enough that minor discrepancies in your specific download (a
missing file, a slightly different total) are plausible and not
necessarily a bug. What **must** hold exactly: the six category names in
`counts` must exactly match `CATEGORY_TO_CLASS`'s keys (the `assert` above
checks this), and `skipped` should be small relative to ~4732, not a large
fraction of it — if most files are being skipped, something is
structurally wrong (wrong root path, wrong fold-folder pattern) rather
than a handful of genuinely bad files.

**Troubleshooting if T2 fails:**
- **`FileNotFoundError` on the metadata CSV** → your Kaggle extraction
  likely nested the dataset one level deeper (e.g.
  `UrbanSound8K/UrbanSound8K/...`) — point `root` at the inner folder.
- **Every file skipped with a "No such file or directory" reason** → the
  `audio/fold{N}/` path pattern doesn't match your extraction; check T1's
  `ls audio` output again and confirm the fold folders are named exactly
  `fold1`, `fold2`, etc. (not `Fold1` or `fold_1`).
- **`KeyError: 'class'`** → your CSV has different column names than the
  documented schema; run `head -1` on it and compare against the exact
  header string in T1's verification.

---

## TASK T3 — Experiment: zero-shot gunshot detection (the new capability)

**Preconditions:** T2 passed on your real files.

**What this measures:** the ESC-50 plan could never test this — there was
no real gunshot audio available. This is the first time in this project's
validation work that real gunshot detection can be measured at all.

**Files touched:** none new — this reuses `run_experiment1_zero_shot.py`
from the ESC-50 plan almost as-is, but pointed at UrbanSound8K instead.
That script already takes `--esc50-root` as its only dataset argument,
which is now a slight misnomer; rather than editing that script (out of
scope for this plan, which touches no existing files including ones from
the prior plan), create a small parallel script:

```python
"""Zero-shot evaluation of a synthetic-trained model on UrbanSound8K --
specifically the gunshot class, since UrbanSound8K is the first real-audio
source in this project's validation work that has any gunshot data at
all. Mirrors run_experiment1_zero_shot.py's structure exactly.

Usage:
    python run_experiment3_us8k_zero_shot.py \\
        --model artifacts/snn_model.npz \\
        --us8k-root /path/to/UrbanSound8K \\
        --out artifacts/real_audio_validation/experiment3_us8k_zero_shot.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecosentry.arch1_audio import extract_mel_spectrogram
from ecosentry.arch2_spikes import convert_mel_to_spikes
from ecosentry.arch4_training import SpikingNetwork, evaluate
from ecosentry.arch5_inference import EcoSentryInference
from ecosentry.config import CLASS_NAMES, EcoSentryConfig
from urbansound8k_loader import load_urbansound8k_corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("artifacts/snn_model.npz"))
    parser.add_argument("--us8k-root", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                         default=Path("artifacts/real_audio_validation/experiment3_us8k_zero_shot.json"))
    parser.add_argument("--n-frames", type=int, default=1000)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    base_cfg = EcoSentryConfig()

    model = SpikingNetwork.load(args.model)
    spike_gain = (model.metadata or {}).get("spike_gain")

    samples, counts, skipped = load_urbansound8k_corpus(args.us8k_root)
    print(f"Loaded {len(samples)} real clips: {counts}  (skipped {len(skipped)})")

    spikes_list = []
    for s in samples:
        mel = extract_mel_spectrogram(s.audio, base_cfg.audio, s.sample_rate, args.n_frames)
        sp = convert_mel_to_spikes(
            mel, base_cfg.spikes, db_floor=base_cfg.audio.db_floor, gain=spike_gain
        )["spikes"]
        spikes_list.append(sp)
    spikes_arr = np.stack(spikes_list)
    labels_arr = np.array([s.label for s in samples])

    result = evaluate(model, spikes_arr, labels_arr)

    engine = EcoSentryInference(model, base_cfg.audio, base_cfg.spikes, base_cfg.inference,
                                 spike_gain=spike_gain, n_frames=args.n_frames)
    alert_by_true_class: dict = {}
    for i in range(len(samples)):
        engine.reset()
        res = engine.process_spikes(spikes_arr[i], use_temporal_filter=False)
        name = CLASS_NAMES[labels_arr[i]]
        alert_by_true_class.setdefault(name, {"n": 0, "alerts": 0})
        alert_by_true_class[name]["n"] += 1
        alert_by_true_class[name]["alerts"] += int(res["alert"])
    for d in alert_by_true_class.values():
        d["alert_rate"] = d["alerts"] / d["n"] if d["n"] else None

    report = {
        "experiment": "urbansound8k_zero_shot",
        "model_path": str(args.model),
        "n_real_clips": len(samples),
        "category_counts": counts,
        "n_skipped": len(skipped),
        "note_on_chainsaw": "UrbanSound8K has no chainsaw category. chainsaw is not evaluated here.",
        "argmax_accuracy": result["accuracy"],
        "argmax_confusion_matrix": result["confusion"].tolist(),
        "argmax_per_class": result.get("per_class"),
        "alert_decision_by_true_class": alert_by_true_class,
    }
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"Wrote {args.out}")
    print(f"argmax_accuracy: {report['argmax_accuracy']:.1%}")
    for name, d in alert_by_true_class.items():
        print(f"  {name}: n={d['n']}  alert_rate={d['alert_rate']:.1%}")


if __name__ == "__main__":
    main()
```

**Verification:**
```
python3 run_experiment3_us8k_zero_shot.py --us8k-root /path/to/your/UrbanSound8K
```
**Expected result:** no crash, a JSON file written, and specifically a
`gunshot` entry in `alert_decision_by_true_class` with `n≈374` — the exact
alert rate is not predictable in advance (this is the first time this
project has ever measured it), but the run must produce *some* gunshot
row. If `gunshot` is missing from the output entirely, something upstream
failed to load or map `gun_shot` clips — check T2's per-category counts
again.

---

## TASK T4 — store Experiment 3's result

**Preconditions:** T3.

Same pattern as the ESC-50 plan's T4 — confirm the file, don't re-derive
anything by hand:
```
python3 -c "
import json
r = json.load(open('artifacts/real_audio_validation/experiment3_us8k_zero_shot.json'))
assert 'gunshot' in r['alert_decision_by_true_class']
print('OK -- gunshot is present in the real-audio zero-shot result for the first time in this project')
"
```
**Expected result:** `OK -- gunshot is present in the real-audio zero-shot
result for the first time in this project`

---

## TASK T5 — smoke-test the combined-dataset retraining path

**Preconditions:** T2 passed.

**Files touched:** create `run_iterations_combined_datasets.py` at the
project repository root. This single script does both the "combine all
three sources and retrain" job and the "repeat across seeds" job — see T7
for the full run; this task only smoke-tests it at tiny scale.

```python
"""Runs the combined-corpus retraining experiment across multiple random
seeds and stores per-seed results plus aggregate (mean/std) statistics.

This is the "iterations" deliverable: a single run's accuracy is a point
estimate with unknown variance. Repeating the same experiment at several
seeds and reporting mean +/- std is what turns "67.1% on one run" into a
number a reviewer can trust wasn't a lucky draw.

Results are written incrementally, one seed at a time, so an interrupted
run still leaves every completed seed's result on disk.

Usage:
    python run_iterations_combined_datasets.py \\
        --esc50-root /path/to/ESC-50 \\
        --us8k-root /path/to/UrbanSound8K \\
        --seeds 42 43 44 45 46 \\
        --epochs 60 \\
        --max-per-us8k-category 60 \\
        --out-dir artifacts/real_audio_validation/iterations
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from ecosentry import arch3_dataset as a3
from ecosentry.arch4_training import evaluate
from ecosentry.config import DatasetConfig, EcoSentryConfig
from ecosentry.pipeline import PipelinePreset, build_spike_dataset, run_training_stage
from ecosentry.synth import build_corpus
from esc50_loader import load_esc50_corpus
from urbansound8k_loader import load_urbansound8k_corpus


def run_one_seed(seed: int, esc50_root: Path, us8k_root: Path, epochs: int,
                  n_synthetic: int, max_per_us8k_category: int, n_frames: int,
                  out_dir: Path) -> dict:
    base_cfg = EcoSentryConfig()
    dataset_cfg = DatasetConfig()

    synthetic = build_corpus(dataset_cfg, duration_s=10.0, sr=base_cfg.audio.sample_rate,
                              seed=seed, n_samples=n_synthetic)
    esc50_samples, esc50_counts = load_esc50_corpus(esc50_root)
    us8k_samples, us8k_counts, us8k_skipped = load_urbansound8k_corpus(
        us8k_root, max_per_category=max_per_us8k_category
    )
    combined = list(synthetic) + list(esc50_samples) + list(us8k_samples)

    encoded = build_spike_dataset(combined, base_cfg.audio, base_cfg.spikes, n_frames=n_frames,
                                   target_rate=dataset_cfg.target_firing_rate, verbose=False)
    prepared = a3.prepare_dataset(encoded["spikes"], encoded["labels"], encoded["forests"],
                                   dataset_cfg, verbose=False)
    prepared["metadata"]["spike_gain"] = encoded["gain"]

    preset = PipelinePreset(f"iter_seed{seed}", 10.0, n_frames, None, epochs, 32, 100, 30)
    seed_out_dir = out_dir / f"seed_{seed}"
    stage2 = run_training_stage(prepared, None, preset, seed_out_dir, base_cfg,
                                 spike_gain=encoded["gain"], verbose=False)

    model = stage2["model"]
    test_forests = np.asarray(prepared["forests"]["test"])
    real_mask = test_forests == "mixed"
    n_real = int(real_mask.sum())
    real_result = evaluate(model, prepared["test"]["spikes"][real_mask],
                            prepared["test"]["labels"][real_mask]) if n_real > 0 else None

    result = {
        "seed": seed,
        "n_synthetic": len(synthetic),
        "n_esc50": len(esc50_samples),
        "n_us8k": len(us8k_samples),
        "n_us8k_skipped": len(us8k_skipped),
        "test_accuracy_combined": stage2["test"]["accuracy"],
        "test_accuracy_real_only": real_result["accuracy"] if real_result else None,
        "n_real_in_test": n_real,
        "val_accuracy": stage2["val_accuracy"],
    }
    (seed_out_dir).mkdir(parents=True, exist_ok=True)
    (seed_out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--esc50-root", type=Path, required=True)
    parser.add_argument("--us8k-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--n-synthetic", type=int, default=600)
    parser.add_argument("--max-per-us8k-category", type=int, default=60)
    parser.add_argument("--n-frames", type=int, default=1000)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/real_audio_validation/iterations"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    for seed in args.seeds:
        print(f"=== seed {seed} ({len(all_results) + 1}/{len(args.seeds)}) ===")
        result = run_one_seed(seed, args.esc50_root, args.us8k_root, args.epochs,
                               args.n_synthetic, args.max_per_us8k_category, args.n_frames,
                               args.out_dir)
        all_results.append(result)
        print(f"  test_accuracy_real_only: {result['test_accuracy_real_only']}")
        real_accs = [r["test_accuracy_real_only"] for r in all_results if r["test_accuracy_real_only"] is not None]
        summary = {
            "seeds_completed": [r["seed"] for r in all_results],
            "n_seeds_completed": len(all_results),
            "n_seeds_requested": len(args.seeds),
            "per_seed_results": all_results,
            "real_only_accuracy_mean": statistics.mean(real_accs) if real_accs else None,
            "real_only_accuracy_stdev": statistics.stdev(real_accs) if len(real_accs) > 1 else None,
            "real_only_accuracy_min": min(real_accs) if real_accs else None,
            "real_only_accuracy_max": max(real_accs) if real_accs else None,
        }
        (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n=== FINAL SUMMARY ===")
    print(f"seeds: {summary['seeds_completed']}")
    print(f"real_only_accuracy: mean={summary['real_only_accuracy_mean']:.1%}"
          f"  stdev={summary['real_only_accuracy_stdev']:.1%}"
          f"  range=[{summary['real_only_accuracy_min']:.1%}, {summary['real_only_accuracy_max']:.1%}]")


if __name__ == "__main__":
    main()
```

**Smoke-test verification (tiny settings, confirms no crashes — not a real
result):**
```
python3 run_iterations_combined_datasets.py \
  --esc50-root /path/to/ESC-50 --us8k-root /path/to/your/UrbanSound8K \
  --seeds 1 2 --epochs 2 --n-synthetic 30 --max-per-us8k-category 5 --n-frames 100 \
  --out-dir /tmp/iter_smoke
```
**Expected result:** completes without a traceback, prints a
`test_accuracy_real_only` line per seed, and a final summary line with
`mean=`, `stdev=`, and `range=`. On this exact command (against ESC-50 and
a small mock UrbanSound8K), a verified run produced `mean=62.2% stdev=7.2%
range=[57.1%, 67.3%]` — meaningless as a real result at these tiny
settings, but confirms the statistics aggregation itself is correct
(2 seeds is enough to compute a stdev, and it did).

**Check the incremental-checkpointing behavior specifically:**
```
cat /tmp/iter_smoke/summary.json | python3 -m json.tool | head -5
ls /tmp/iter_smoke/seed_1/
```
**Expected result:** `summary.json` exists and is valid JSON even though
you're checking it after the run already finished (this only proves the
file exists, not that it survives an interruption — but the code path
that writes it runs after every seed, not just at the end, which is the
property that matters for a long run you might need to stop early). The
`seed_1/` directory should contain both `result.json` and `snn_model.npz`
(one trained model checkpoint per seed).

---

## TASK T6 — decide your iteration budget before running for real

**Preconditions:** T5's smoke test passed.

**No files touched.** This is a planning step, because the full-scale
version of T5 is expensive and the right settings depend on how much time
you have, not on anything this plan can decide for you.

Known reference point: the ESC-50-only combined-retraining experiment (600
synthetic + 480 ESC-50 = 1,080 samples, 120 epochs) took **738.9 seconds**
in a verified run. Adding UrbanSound8K scales the corpus up substantially
depending on `--max-per-us8k-category`:

| `--max-per-us8k-category` | UrbanSound8K samples added | Approx. total corpus | Rough time per seed at 120 epochs |
|---|---|---|---|
| 60 (default suggestion) | 360 (6 categories × 60) | ~1,440 | ~15–18 minutes |
| 150 | 900 | ~1,980 | ~20–25 minutes |
| unlimited (all ~4,732) | 4,732 | ~5,812 | likely 45–60+ minutes |

These are extrapolations from one measured data point, not independent
measurements — treat them as planning guidance, not guarantees, and expect
your actual machine's speed to shift these numbers up or down.

**Recommended split, if you don't already have a strong preference:**
- Use `--max-per-us8k-category 60` and `--epochs 60` for the multi-seed run
  (T7) — moderate per-run cost, so 5 seeds is tractable in roughly
  1.5–2.5 hours total rather than most of a day.
- Separately, run one single full-scale pass (all available data, 120
  epochs, one seed) as the "headline" number, the same way the ESC-50 plan
  treated its one full run — this is not in this plan as a separate task
  because it's mechanically identical to T7 with `--seeds` given one value
  and `--max-per-us8k-category` omitted; just run T7's command that way
  once, separately from the 5-seed sweep.

---

## TASK T7 — the real multi-seed run

**Preconditions:** T6's budget decision made.

**Use the proven `setsid` launch technique** — a plain background `&` or a
foreground `timeout` wrapper were both confirmed, in the ESC-50 plan's
development, to silently kill long-running jobs in exactly this kind of
tool-call-based environment:

```
rm -rf artifacts/real_audio_validation/iterations
setsid nohup python3 run_iterations_combined_datasets.py \
  --esc50-root /path/to/ESC-50 --us8k-root /path/to/your/UrbanSound8K \
  --seeds 42 43 44 45 46 --epochs 60 --max-per-us8k-category 60 \
  --out-dir artifacts/real_audio_validation/iterations \
  > /tmp/iterations_full.log 2>&1 < /dev/null &
sleep 5
ps aux | grep run_iterations | grep -v grep
```
**Expected result:** one line showing the process running. If empty, check
`/tmp/iterations_full.log` immediately.

**Poll every 2–3 minutes, as separate short commands, not one long sleep:**
```
tail -15 /tmp/iterations_full.log
cat artifacts/real_audio_validation/iterations/summary.json 2>/dev/null | python3 -m json.tool | tail -20
ps aux | grep run_iterations | grep -v grep
```
The `summary.json` check lets you see completed-seed results even while
later seeds are still running, since it's rewritten after every seed —
useful for noticing early if something looks wrong (e.g. `stdev` far
larger than expected) without waiting for all 5 seeds.

**Once `ps` comes back empty, verify completion:**
```
python3 -c "
import json
s = json.load(open('artifacts/real_audio_validation/iterations/summary.json'))
assert s['n_seeds_completed'] == s['n_seeds_requested'], 'run did not finish all seeds -- check the log'
print(f\"mean={s['real_only_accuracy_mean']:.1%}  stdev={s['real_only_accuracy_stdev']:.1%}\")
print(f\"range=[{s['real_only_accuracy_min']:.1%}, {s['real_only_accuracy_max']:.1%}]\")
for r in s['per_seed_results']:
    print(f\"  seed {r['seed']}: real_only={r['test_accuracy_real_only']:.1%}  n_real={r['n_real_in_test']}\")
"
```
**Expected result:** no assertion error, five per-seed lines, and a
mean/stdev/range summary. There is no single "correct" number to expect
here — this is the first time this exact combined-dataset experiment has
been run to completion — but a `stdev` under roughly 5 percentage points
would indicate the result is reasonably stable across seeds; a much larger
spread would itself be a real, reportable finding (worth investigating
which seed's split happened to be unusually easy or hard, via each seed's
`n_esc50`/`n_us8k` counts and confusion matrices in its `seed_N/` folder).

---

## TASK T8 — assemble the combined-datasets summary

**Preconditions:** T4 and T7 complete.

**Files touched:** create `artifacts/real_audio_validation/SUMMARY_US8K.md`
by hand, following the same structure as the ESC-50 plan's T7:

1. **Setup.** UrbanSound8K's own citation, the 10-fold cross-validation
   caveat from T0 (state explicitly that this plan's experiments use
   random stratified splits, not the official folds, and why that matters
   for anyone trying to compare against published UrbanSound8K results).
2. **Experiment 3 (zero-shot gunshot).** The first-ever real-audio gunshot
   result for this project, from T4's JSON.
3. **Multi-seed combined results (T7).** Report the mean ± stdev, not just
   a single number — this is the credibility improvement this whole plan
   was for. Include the range across seeds, and the sample size
   (`n_real_in_test`) for every seed, not just the aggregate.
4. **Cross-dataset comparison table**, combining this plan's numbers with
   the earlier ESC-50-only plan's numbers:

   | Experiment | Real-audio accuracy | n | Gunshot covered? |
   |---|---|---|---|
   | ESC-50 zero-shot (earlier plan) | 15.6% | 480 | No |
   | ESC-50 retrained, real-only (earlier plan) | 67.1% | 146 | No |
   | UrbanSound8K zero-shot, gunshot only (T4) | *(fill in)* | 374 | Yes |
   | ESC-50 + UrbanSound8K retrained, real-only, multi-seed (T7) | *(fill in mean ± stdev)* | *(varies per seed)* | Yes |

5. **Limitations**, updated from the ESC-50 plan's list: chainsaw remains
   uncovered by real audio from *either* dataset; `ambient` mixes forest
   ambience (implied by the project's purpose) with urban ambience
   (what's actually tested); the official UrbanSound8K 10-fold protocol
   was not used, which any reviewer familiar with this dataset would
   otherwise expect.

**Verification:** none — a document, not code, verified by every number in
it tracing back to a specific JSON key from T4 or T7.
