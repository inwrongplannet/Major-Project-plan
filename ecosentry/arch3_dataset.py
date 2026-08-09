"""ARCH_3 -- Dataset preparation: consolidation, forest normalisation,
stratified splits, augmentation and HDF5 persistence.

Pipeline (ARCH_3 Component 5)::

    consolidate -> forest-specific normalisation -> stratified 60/20/20 split
                -> augmentation of the training split -> prepared_dataset.h5

A note on normalisation
-----------------------
The reference pseudocode normalises firing rate by multiplying a *binary* spike
tensor by ``target_rate / current_rate`` and clipping to [0, 1].  For
``current_rate < target_rate`` that is a no-op (1 * k clipped back to 1), so it
cannot actually raise a firing rate.  ``rate_normalize`` therefore adjusts the
spike *count* stochastically -- dropping spikes when the tensor is too dense and
recruiting the strongest sub-threshold candidates when it is too sparse -- which
is what the "25% +/- 5%" acceptance criterion requires.  The literal
multiply-and-clip behaviour is still available via ``mode="scale"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import CLASS_NAMES, DatasetConfig

__all__ = [
    "rate_normalize",
    "normalize_corbett_spikes",
    "normalize_seshachalam_spikes",
    "normalize_sundarbans_spikes",
    "normalize_mixed_spikes",
    "apply_forest_normalization",
    "create_stratified_splits",
    "verify_class_balance",
    "mixup_spikes",
    "time_shift_spikes",
    "add_spike_noise",
    "create_augmented_dataset",
    "prepare_dataset",
    "save_prepared_dataset",
    "load_prepared_dataset",
    "validate_normalized_spikes",
    "validate_splits",
]


# ---------------------------------------------------------------------------
# Firing-rate normalisation primitives
# ---------------------------------------------------------------------------


def _as2d(spikes: np.ndarray) -> Tuple[np.ndarray, bool]:
    arr = np.asarray(spikes)
    if arr.ndim == 3:
        return arr[:, :, 0].astype(np.float32), True
    return arr.astype(np.float32), False


def _restore(arr: np.ndarray, had_channel: bool) -> np.ndarray:
    return arr[:, :, None].astype(np.float32) if had_channel else arr.astype(np.float32)


def rate_normalize(
    spikes: np.ndarray,
    target_rate: float = 0.25,
    rng: Optional[np.random.Generator] = None,
    saliency: Optional[np.ndarray] = None,
    mode: str = "stochastic",
) -> np.ndarray:
    """Drive the global firing rate of a spike tensor to ``target_rate``.

    ``mode="stochastic"`` adds/removes individual spikes; ``mode="scale"``
    reproduces the literal multiply-and-clip formulation from the design doc.
    """
    arr2d, had_channel = _as2d(spikes)
    rng = rng or np.random.default_rng()
    total_cells = arr2d.size
    if total_cells == 0:
        return _restore(arr2d, had_channel)

    if mode == "scale":
        current = float(arr2d.mean())
        factor = target_rate / (current + 1e-8)
        return _restore(np.clip(arr2d * factor, 0.0, 1.0), had_channel)

    target_count = int(round(target_rate * total_cells))
    active = arr2d > 0
    current_count = int(active.sum())

    if current_count > target_count:
        idx = np.flatnonzero(active.ravel())
        weights = None
        if saliency is not None:
            sal = np.asarray(saliency, dtype=np.float64).ravel()[idx]
            weights = 1.0 / (sal - sal.min() + 1e-3)
            weights = weights / weights.sum()
        drop = rng.choice(idx, size=current_count - target_count, replace=False, p=weights)
        flat = arr2d.ravel().copy()
        flat[drop] = 0.0
        arr2d = flat.reshape(arr2d.shape)

    elif current_count < target_count:
        idx = np.flatnonzero(~active.ravel())
        need = min(target_count - current_count, len(idx))
        if need > 0:
            if saliency is not None:
                sal = np.asarray(saliency, dtype=np.float64).ravel()[idx]
                order = np.argsort(-sal)[:need]
                add = idx[order]
            else:
                add = rng.choice(idx, size=need, replace=False)
            flat = arr2d.ravel().copy()
            flat[add] = 1.0
            arr2d = flat.reshape(arr2d.shape)

    return _restore(arr2d, had_channel)


# ---------------------------------------------------------------------------
# Forest-specific normalisation (ARCH_3 Component 2)
# ---------------------------------------------------------------------------


def normalize_corbett_spikes(
    spikes: np.ndarray,
    target_rate: float = 0.25,
    rng: Optional[np.random.Generator] = None,
    background_threshold: float = 0.5,
    suppression: float = 0.5,
) -> np.ndarray:
    """Dense forest: suppress chronically-firing background neurons first."""
    arr2d, had_channel = _as2d(spikes)
    rng = rng or np.random.default_rng()

    per_neuron = arr2d.mean(axis=0)
    background = np.flatnonzero(per_neuron > background_threshold)
    for n in background:
        active_idx = np.flatnonzero(arr2d[:, n] > 0)
        if active_idx.size == 0:
            continue
        drop = rng.random(active_idx.size) < suppression
        arr2d[active_idx[drop], n] = 0.0

    # Local density is the saliency signal: transients survive, drone does not.
    saliency = _local_density(arr2d, window=20)
    out = rate_normalize(arr2d, target_rate, rng, saliency=saliency)
    return _restore(out if out.ndim == 2 else out[:, :, 0], had_channel)


def normalize_seshachalam_spikes(
    spikes: np.ndarray,
    target_rate: float = 0.25,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Open terrain: clean signals, so a plain global rate normalisation."""
    arr2d, had_channel = _as2d(spikes)
    out = rate_normalize(arr2d, target_rate, rng or np.random.default_rng())
    return _restore(out if out.ndim == 2 else out[:, :, 0], had_channel)


def normalize_sundarbans_spikes(
    spikes: np.ndarray,
    target_rate: float = 0.25,
    rng: Optional[np.random.Generator] = None,
    window: int = 20,
    sustained_threshold: float = 0.6,
) -> np.ndarray:
    """Wetlands: strip sustained water/rain patterns, keep transients."""
    arr2d, had_channel = _as2d(spikes)
    rng = rng or np.random.default_rng()

    smoothed = _local_density(arr2d, window=window)
    arr2d[smoothed > sustained_threshold] = 0.0

    # Contrast enhancement: prefer spikes whose neighbourhood is bursty.
    local_std = _local_std(arr2d, window=window)
    saliency = arr2d * (1.0 + local_std)

    out = rate_normalize(arr2d, target_rate, rng, saliency=saliency + local_std)
    return _restore(out if out.ndim == 2 else out[:, :, 0], had_channel)


def normalize_mixed_spikes(
    spikes: np.ndarray,
    target_rate: float = 0.25,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """ESC-50 / UrbanSound8K: no ecosystem prior, global normalisation only."""
    return normalize_seshachalam_spikes(spikes, target_rate, rng)


_NORMALIZERS = {
    "corbett": normalize_corbett_spikes,
    "seshachalam": normalize_seshachalam_spikes,
    "sundarbans": normalize_sundarbans_spikes,
    "mixed": normalize_mixed_spikes,
}


def apply_forest_normalization(
    spikes: np.ndarray,
    forest_assignment: Sequence[str],
    cfg: Optional[DatasetConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply the per-forest normaliser to every sample in ``spikes``."""
    cfg = cfg or DatasetConfig()
    rng = rng or np.random.default_rng(cfg.seed)

    out = np.zeros(spikes.shape, dtype=np.float32)
    for i, forest in enumerate(forest_assignment):
        fn = _NORMALIZERS.get(str(forest), normalize_mixed_spikes)
        out[i] = fn(spikes[i], cfg.target_firing_rate, rng)
    return out


def _local_density(arr2d: np.ndarray, window: int = 20) -> np.ndarray:
    """Centred moving average along the time axis (vectorised)."""
    if window <= 1:
        return arr2d.astype(np.float32)
    kernel = np.ones(window, dtype=np.float32) / window
    padded = np.pad(arr2d, ((window // 2, window // 2), (0, 0)), mode="edge")
    out = np.empty_like(arr2d, dtype=np.float32)
    for c in range(arr2d.shape[1]):
        out[:, c] = np.convolve(padded[:, c], kernel, mode="same")[
            window // 2 : window // 2 + arr2d.shape[0]
        ]
    return out


def _local_std(arr2d: np.ndarray, window: int = 20) -> np.ndarray:
    mean = _local_density(arr2d, window)
    mean_sq = _local_density(arr2d**2, window)
    return np.sqrt(np.maximum(mean_sq - mean**2, 0.0)).astype(np.float32)


# ---------------------------------------------------------------------------
# Splits (ARCH_3 Component 3)
# ---------------------------------------------------------------------------


def create_stratified_splits(
    spikes: np.ndarray,
    labels: np.ndarray,
    forest_assignment: Sequence[str],
    cfg: Optional[DatasetConfig] = None,
) -> Dict:
    """Stratified 60/20/20 split, balanced by (class, forest) stratum."""
    cfg = cfg or DatasetConfig()
    rng = np.random.default_rng(cfg.seed)

    labels = np.asarray(labels)
    forests = np.asarray(forest_assignment)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    strata = {}
    for i, (lab, forest) in enumerate(zip(labels, forests)):
        strata.setdefault((int(lab), str(forest)), []).append(i)

    for key in sorted(strata):
        idx = np.array(strata[key])
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * cfg.train_ratio))
        n_val = int(round(n * cfg.val_ratio))
        # Guarantee at least one sample per split when the stratum allows it.
        if n >= 3:
            n_train = max(min(n_train, n - 2), 1)
            n_val = max(min(n_val, n - n_train - 1), 1)
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train : n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val :].tolist())

    for split in (train_idx, val_idx, test_idx):
        rng.shuffle(split)

    return {
        "train": {"spikes": spikes[train_idx], "labels": labels[train_idx]},
        "val": {"spikes": spikes[val_idx], "labels": labels[val_idx]},
        "test": {"spikes": spikes[test_idx], "labels": labels[test_idx]},
        "split_indices": {"train": train_idx, "val": val_idx, "test": test_idx},
        "forests": {
            "train": forests[train_idx],
            "val": forests[val_idx],
            "test": forests[test_idx],
        },
    }


def verify_class_balance(split_data: Dict) -> Dict[str, Dict[str, float]]:
    """Per-split class histogram, as percentages."""
    report: Dict[str, Dict[str, float]] = {}
    for split in ("train", "val", "test"):
        labels = np.asarray(split_data[split]["labels"])
        entry: Dict[str, float] = {}
        for class_id in range(len(CLASS_NAMES)):
            count = int(np.sum(labels == class_id))
            if count:
                entry[CLASS_NAMES[class_id]] = round(100.0 * count / max(len(labels), 1), 1)
        entry["_n"] = int(len(labels))
        report[split] = entry
    return report


# ---------------------------------------------------------------------------
# Augmentation (ARCH_3 Component 4)
# ---------------------------------------------------------------------------


def mixup_spikes(
    spikes_a: np.ndarray,
    spikes_b: np.ndarray,
    label_a: int,
    label_b: int,
    alpha: float = 0.2,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, int, float]:
    """Interpolate two spike tensors.

    Returns ``(mixed, dominant_label, lam)``.  The hard label of the dominant
    component is kept so the tensor stays compatible with the sparse
    cross-entropy loss used in ARCH_4.
    """
    rng = rng or np.random.default_rng()
    lam = float(rng.beta(alpha, alpha))
    mixed = (lam * spikes_a + (1.0 - lam) * spikes_b).astype(np.float32)
    dominant = int(label_a) if lam >= 0.5 else int(label_b)
    return mixed, dominant, lam


def time_shift_spikes(
    spikes: np.ndarray,
    max_shift_frames: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Circularly shift by +/- ``max_shift_frames`` frames (10 ms each)."""
    rng = rng or np.random.default_rng()
    shift = int(rng.integers(-max_shift_frames, max_shift_frames + 1))
    return np.roll(spikes, shift, axis=0).astype(np.float32)


def add_spike_noise(
    spikes: np.ndarray,
    noise_rate: float = 0.01,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Flip a small fraction of cells to model sensor noise."""
    rng = rng or np.random.default_rng()
    mask = rng.random(spikes.shape) < noise_rate
    out = np.array(spikes, dtype=np.float32, copy=True)
    out[mask] = 1.0 - out[mask]
    return out


def create_augmented_dataset(
    train_spikes: np.ndarray,
    train_labels: np.ndarray,
    cfg: Optional[DatasetConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Expand the training split by ``augmentation_factor`` (default 2x)."""
    cfg = cfg or DatasetConfig()
    rng = np.random.default_rng(cfg.seed + 1)

    aug_spikes = [train_spikes]
    aug_labels = [np.asarray(train_labels)]

    n = len(train_spikes)
    for _ in range(max(cfg.augmentation_factor - 1, 0)):
        batch = np.empty_like(train_spikes)
        batch_labels = np.empty(n, dtype=np.int64)
        for i in range(n):
            sample = train_spikes[i]
            label = int(train_labels[i])

            if rng.random() < cfg.p_mixup:
                partner = int(rng.integers(n))
                sample, label, _ = mixup_spikes(
                    sample,
                    train_spikes[partner],
                    label,
                    int(train_labels[partner]),
                    cfg.mixup_alpha,
                    rng,
                )
            if rng.random() < cfg.p_time_shift:
                sample = time_shift_spikes(sample, cfg.max_shift_frames, rng)
            if rng.random() < cfg.p_noise:
                sample = add_spike_noise(sample, cfg.noise_rate, rng)

            batch[i] = sample
            batch_labels[i] = label
        aug_spikes.append(batch)
        aug_labels.append(batch_labels)

    return (
        np.concatenate(aug_spikes, axis=0).astype(np.float32),
        np.concatenate(aug_labels, axis=0).astype(np.int64),
    )


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------


def prepare_dataset(
    spikes: np.ndarray,
    labels: np.ndarray,
    forest_assignment: Sequence[str],
    cfg: Optional[DatasetConfig] = None,
    verbose: bool = True,
) -> Dict:
    """consolidate -> normalise -> split -> augment.

    ``spikes`` is ``(N, T, 64, 1)`` (or ``(N, T, 64)``) as produced by ARCH_2.
    """
    cfg = cfg or DatasetConfig()
    spikes = np.asarray(spikes, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    if verbose:
        print(f"[ARCH_3 1/4] consolidated {len(spikes)} samples, shape {spikes.shape}")

    normalized = apply_forest_normalization(spikes, forest_assignment, cfg)
    if verbose:
        rate = float((normalized > 0).mean())
        print(f"[ARCH_3 2/4] forest normalisation done, firing rate {rate:.1%}")

    split = create_stratified_splits(normalized, labels, forest_assignment, cfg)
    if verbose:
        print(f"[ARCH_3 3/4] split {verify_class_balance(split)}")

    aug_spikes, aug_labels = create_augmented_dataset(
        split["train"]["spikes"], split["train"]["labels"], cfg
    )
    if verbose:
        print(
            f"[ARCH_3 4/4] augmented train {len(split['train']['spikes'])} "
            f"-> {len(aug_spikes)} samples"
        )

    return {
        "train": {"spikes": aug_spikes, "labels": aug_labels},
        "val": split["val"],
        "test": split["test"],
        "forests": split["forests"],
        "metadata": {
            "total_original": int(len(spikes)),
            "total_train": int(len(aug_spikes)),
            "total_val": int(len(split["val"]["labels"])),
            "total_test": int(len(split["test"]["labels"])),
            "class_names": CLASS_NAMES,
            "target_firing_rate": cfg.target_firing_rate,
            "n_frames": int(spikes.shape[1]),
            "n_neurons": int(spikes.shape[2]),
        },
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_prepared_dataset(dataset: Dict, path) -> Path:
    """Write the prepared dataset to HDF5 (gzip-compressed)."""
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        for split in ("train", "val", "test"):
            grp = f.create_group(split)
            grp.create_dataset(
                "spikes",
                data=np.asarray(dataset[split]["spikes"], dtype=np.float32),
                compression="gzip",
                compression_opts=4,
            )
            grp.create_dataset(
                "labels", data=np.asarray(dataset[split]["labels"], dtype=np.int64)
            )
            forests = dataset.get("forests", {}).get(split)
            if forests is not None:
                grp.create_dataset(
                    "forests", data=np.asarray(forests, dtype="S16")
                )
        meta = f.create_group("metadata")
        for key, value in dataset["metadata"].items():
            if isinstance(value, (list, tuple)):
                meta.attrs[key] = np.asarray([str(v) for v in value], dtype="S32")
            else:
                meta.attrs[key] = value
    return path


def load_prepared_dataset(path) -> Dict:
    """Read a dataset written by :func:`save_prepared_dataset`."""
    import h5py

    out: Dict = {"forests": {}}
    with h5py.File(Path(path), "r") as f:
        for split in ("train", "val", "test"):
            out[split] = {
                "spikes": f[split]["spikes"][:],
                "labels": f[split]["labels"][:],
            }
            if "forests" in f[split]:
                out["forests"][split] = np.array(
                    [s.decode() for s in f[split]["forests"][:]]
                )
        out["metadata"] = {}
        for key, value in f["metadata"].attrs.items():
            if isinstance(value, np.ndarray) and value.dtype.kind == "S":
                out["metadata"][key] = [v.decode() for v in value]
            else:
                out["metadata"][key] = value
    return out


# ---------------------------------------------------------------------------
# Quality assurance (ARCH_3 Component 6)
# ---------------------------------------------------------------------------


def validate_normalized_spikes(
    spikes: np.ndarray, cfg: Optional[DatasetConfig] = None
) -> Dict:
    cfg = cfg or DatasetConfig()
    arr = np.asarray(spikes, dtype=np.float32)
    lo = cfg.target_firing_rate - cfg.firing_rate_tolerance
    hi = cfg.target_firing_rate + cfg.firing_rate_tolerance

    per_sample = (arr > 0).reshape(arr.shape[0], -1).mean(axis=1) if arr.ndim > 2 else None
    checks = {
        "has_nans": bool(np.any(np.isnan(arr))),
        "has_infs": bool(np.any(np.isinf(arr))),
        "in_range_0_1": bool(np.all(arr >= 0.0) and np.all(arr <= 1.0)),
        "global_firing_rate": float((arr > 0).mean()),
    }
    if per_sample is not None:
        checks["firing_rate_in_range"] = float(
            np.mean((per_sample >= lo) & (per_sample <= hi))
        )
        per_neuron_var = arr.reshape(-1, arr.shape[2]).var(axis=0)
        checks["neurons_have_variance"] = float(np.mean(per_neuron_var > 0))
    checks["passed"] = (
        not checks["has_nans"]
        and not checks["has_infs"]
        and checks["in_range_0_1"]
        and lo <= checks["global_firing_rate"] <= hi
    )
    return checks


def validate_splits(split_data: Dict) -> Dict:
    idx = split_data["split_indices"]
    sets = {k: set(v) for k, v in idx.items()}
    checks = {
        "total_samples": sum(len(v) for v in idx.values()),
        "no_overlap": (
            not (sets["train"] & sets["val"])
            and not (sets["train"] & sets["test"])
            and not (sets["val"] & sets["test"])
        ),
    }
    for split in ("train", "val", "test"):
        labels = np.asarray(split_data[split]["labels"])
        if labels.size:
            _, counts = np.unique(labels, return_counts=True)
            checks[f"{split}_class_balance_ratio"] = float(counts.max() / counts.min())
        checks[f"{split}_n"] = int(labels.size)
    checks["passed"] = bool(checks["no_overlap"])
    return checks
