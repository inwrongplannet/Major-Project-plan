"""ARCH_2 -- Spike conversion: mel-spectrogram (T, 64) -> spike train (T, 64, 1).

Per the ARCH_2 design decision there is *no* Gammatone filterbank: the mel
bands from ARCH_1 already provide the cochlear-like decomposition, so one LIF
neuron is attached directly to each mel band.

LIF dynamics at frame resolution (dt = hop_ms = 10 ms):

    V[t] = alpha * V[t-1] + (1 - alpha) * gain * I[t]
    spike if V >= v_th, then V <- 0 and the neuron is refractory for 2 ms

``alpha = exp(-dt / tau_m) = exp(-1) ~ 0.3679``.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .config import SpikeConfig

__all__ = [
    "normalize_mel_input",
    "lif_neurons",
    "compute_spike_latencies",
    "convert_mel_to_spikes",
    "calibrate_input_gain",
    "firing_rate",
    "spike_to_sparse",
    "sparse_to_spike",
    "validate_spikes",
]


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------


def normalize_mel_input(mel_spec: np.ndarray, db_floor: float = -80.0) -> np.ndarray:
    """Map dB mel-spectrogram [-80, 0] -> [0, 1] drive current."""
    mel_norm = (np.asarray(mel_spec, dtype=np.float32) - db_floor) / abs(db_floor)
    return np.clip(mel_norm, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# LIF layer
# ---------------------------------------------------------------------------


def lif_neurons(
    drive: np.ndarray,
    cfg: Optional[SpikeConfig] = None,
    gain: Optional[float] = None,
) -> np.ndarray:
    """Simulate one LIF neuron per channel over ``drive`` of shape ``(T, C)``.

    Returns a ``(T, C)`` boolean spike matrix.
    """
    cfg = cfg or SpikeConfig()
    gain = cfg.input_gain if gain is None else gain

    drive = np.asarray(drive, dtype=np.float32)
    T, C = drive.shape
    alpha = cfg.alpha
    one_minus_alpha = 1.0 - alpha
    refractory_frames = cfg.refractory_frames

    v = np.zeros(C, dtype=np.float32)
    refractory = np.zeros(C, dtype=np.int32)
    spikes = np.zeros((T, C), dtype=bool)

    for t in range(T):
        active = refractory <= 0
        i_t = drive[t] * gain
        v = np.where(active, alpha * v + one_minus_alpha * i_t, 0.0).astype(np.float32)

        fired = active & (v >= cfg.v_threshold)
        spikes[t] = fired
        v = np.where(fired, cfg.v_reset, v).astype(np.float32)
        refractory = np.where(fired, refractory_frames, np.maximum(refractory - 1, 0))

    return spikes


def firing_rate(spikes: np.ndarray) -> float:
    """Fraction of (frame, neuron) cells that carry a spike."""
    spikes = np.asarray(spikes)
    return float(spikes.astype(np.float32).mean()) if spikes.size else 0.0


def calibrate_input_gain(
    drive: np.ndarray,
    target_rate: float = 0.25,
    cfg: Optional[SpikeConfig] = None,
    lo: float = 0.5,
    hi: float = 20.0,
    iterations: int = 14,
) -> float:
    """Binary-search the input gain that lands the firing rate on ``target_rate``.

    The LIF threshold is fixed at 1.0 by the spec, so gain is the free knob used
    to hit the documented 25% +/- 5% sparsity target.
    """
    cfg = cfg or SpikeConfig()
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        rate = firing_rate(lif_neurons(drive, cfg, gain=mid))
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Time-to-first-spike encoding
# ---------------------------------------------------------------------------


def compute_spike_latencies(spikes: np.ndarray, hop_ms: float = 10.0):
    """Per-neuron time (ms) to first spike, plus the rank order.

    Returns ``(latencies_ms, rank_order, spike_indices)``.  Silent neurons get
    ``inf`` latency and index ``-1``.
    """
    spikes = np.asarray(spikes)
    T, C = spikes.shape
    latencies = np.full(C, np.inf, dtype=np.float32)
    indices = np.full(C, -1, dtype=np.int32)

    any_spike = spikes.any(axis=0)
    first_idx = spikes.argmax(axis=0)
    indices[any_spike] = first_idx[any_spike]
    latencies[any_spike] = first_idx[any_spike] * hop_ms

    rank_order = np.argsort(latencies, kind="stable").astype(np.int32)
    return latencies, rank_order, indices


# ---------------------------------------------------------------------------
# Complete conversion
# ---------------------------------------------------------------------------


def convert_mel_to_spikes(
    mel_spec: np.ndarray,
    cfg: Optional[SpikeConfig] = None,
    db_floor: float = -80.0,
    gain: Optional[float] = None,
    auto_gain: bool = False,
    target_rate: float = 0.25,
) -> Dict[str, np.ndarray]:
    """Mel-spectrogram -> spike train dictionary.

    Keys: ``spikes`` (T, 64, 1) uint8, ``latencies`` (64,), ``rank_order`` (64,),
    ``spike_indices`` (64,), ``mel_normalized`` (T, 64), ``gain``, ``firing_rate``.
    """
    cfg = cfg or SpikeConfig()
    drive = normalize_mel_input(mel_spec, db_floor)

    if auto_gain:
        gain = calibrate_input_gain(drive, target_rate, cfg)
    elif gain is None:
        gain = cfg.input_gain

    spikes_2d = lif_neurons(drive, cfg, gain=gain)
    latencies, rank_order, spike_indices = compute_spike_latencies(spikes_2d, cfg.hop_ms)

    return {
        "spikes": spikes_2d[:, :, None].astype(np.uint8),
        "latencies": latencies,
        "rank_order": rank_order,
        "spike_indices": spike_indices,
        "mel_normalized": drive,
        "gain": np.float32(gain),
        "firing_rate": np.float32(firing_rate(spikes_2d)),
    }


# ---------------------------------------------------------------------------
# Sparse storage
# ---------------------------------------------------------------------------


def spike_to_sparse(spikes: np.ndarray) -> Dict[str, np.ndarray]:
    """Dense ``(T, 64, 1)`` (or ``(T, 64)``) spikes -> COO index arrays."""
    dense = np.asarray(spikes)
    shape = dense.shape
    if dense.ndim == 3:
        dense = dense[:, :, 0]
    frames, neurons = np.where(dense > 0)
    return {
        "spike_frames": frames.astype(np.int32),
        "spike_neurons": neurons.astype(np.int16),
        "total_spikes": np.int32(len(frames)),
        "shape": np.array(shape, dtype=np.int32),
    }


def sparse_to_spike(sparse: Dict[str, np.ndarray]) -> np.ndarray:
    """Inverse of :func:`spike_to_sparse`."""
    shape = tuple(int(v) for v in sparse["shape"])
    dense = np.zeros(shape[:2], dtype=np.uint8)
    dense[sparse["spike_frames"], sparse["spike_neurons"]] = 1
    if len(shape) == 3:
        dense = dense[:, :, None]
    return dense


# ---------------------------------------------------------------------------
# Quality assurance (ARCH_2 "Validation metrics")
# ---------------------------------------------------------------------------


def validate_spikes(
    spikes: np.ndarray, min_rate: float = 0.05, max_rate: float = 0.45
) -> dict:
    """ARCH_2 QA: binary values, sane firing rate, refractory spacing."""
    arr = np.asarray(spikes)
    dense = arr[:, :, 0] if arr.ndim == 3 else arr
    rate = firing_rate(dense)

    per_neuron = dense.astype(np.float32).mean(axis=0)
    checks = {
        "binary": bool(np.all((dense == 0) | (dense == 1))),
        "firing_rate": float(rate),
        "firing_rate_in_range": bool(min_rate <= rate <= max_rate),
        "active_neuron_fraction": float(np.mean(per_neuron > 0)),
        "shape": tuple(int(s) for s in arr.shape),
    }
    checks["passed"] = checks["binary"] and checks["firing_rate_in_range"]
    return checks
