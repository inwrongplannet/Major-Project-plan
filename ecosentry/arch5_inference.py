"""ARCH_5 -- Real-time SNN inference on the edge device.

Wraps the trained ARCH_4 weights with the runtime concerns the design document
specifies: streaming windows, softmax confidence, per-class adaptive
thresholds, temporal debouncing, alert rate limiting and INT8 weight
quantisation for embedded deployment.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional, Sequence

import numpy as np

from .arch1_audio import extract_mel_spectrogram
from .arch2_spikes import convert_mel_to_spikes
from .arch4_training import SpikingNetwork, softmax
from .config import AudioConfig, InferenceConfig, SpikeConfig, THREAT_CLASSES

__all__ = [
    "classify_with_confidence",
    "detect_threat",
    "adaptive_thresholding",
    "temporal_filter",
    "quantize_weights",
    "dequantize_weights",
    "EcoSentryInference",
]


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------


def classify_with_confidence(
    logits: np.ndarray,
    class_names: Sequence[str],
    temperature: float = 1.0,
) -> Dict:
    """Logits -> class id, name, confidence, full distribution and entropy."""
    probs = softmax(np.atleast_2d(logits), temperature)[0]
    class_id = int(np.argmax(probs))
    entropy = float(-np.sum(probs * np.log(probs + 1e-8)))
    return {
        "class_id": class_id,
        "class_name": class_names[class_id],
        "confidence": float(probs[class_id]),
        "probabilities": probs.astype(np.float32),
        "entropy": entropy,
    }


def detect_threat(result: Dict, threshold: float = 0.85) -> bool:
    """Static threshold rule: threat class above ``threshold`` triggers."""
    return result["class_id"] in THREAT_CLASSES and result["confidence"] > threshold


def adaptive_thresholding(
    result: Dict, class_thresholds: Optional[Dict[int, float]] = None, default: float = 0.85
):
    """Per-class thresholds. Returns ``(alert, confidence, threshold_used)``."""
    class_thresholds = class_thresholds or {}
    threshold = class_thresholds.get(result["class_id"], default)
    return result["confidence"] > threshold, result["confidence"], threshold


def temporal_filter(history: Sequence[int], window_size: int = 5):
    """Majority vote over the last ``window_size`` classifications."""
    recent = list(history)[-window_size:]
    if not recent:
        return None, 0.0
    counts: Dict[int, int] = {}
    for class_id in recent:
        counts[class_id] = counts.get(class_id, 0) + 1
    winner = max(counts, key=lambda k: counts[k])
    return winner, counts[winner] / len(recent)


# ---------------------------------------------------------------------------
# Quantisation (ARCH_5 Component 1)
# ---------------------------------------------------------------------------


def quantize_weights(model: SpikingNetwork) -> Dict[str, Dict]:
    """Symmetric per-tensor INT8 quantisation of the weight matrices."""
    quantized: Dict[str, Dict] = {}
    for key in ("W1", "W2", "W3"):
        w = model.params[key]
        scale = float(np.max(np.abs(w))) / 127.0 if np.any(w) else 1.0
        quantized[key] = {
            "weights": np.clip(np.round(w / scale), -127, 127).astype(np.int8),
            "scale": scale,
        }
    return quantized


def dequantize_weights(entry: Dict) -> np.ndarray:
    return (entry["weights"].astype(np.float32) * entry["scale"]).astype(np.float32)


def quantization_error(model: SpikingNetwork) -> Dict[str, float]:
    """Relative reconstruction error introduced by INT8 quantisation."""
    q = quantize_weights(model)
    out = {}
    for key, entry in q.items():
        original = model.params[key]
        error = np.abs(dequantize_weights(entry) - original)
        out[key] = float(error.mean() / (np.abs(original).mean() + 1e-12))
    return out


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------


class EcoSentryInference:
    """Edge inference engine: audio in, alert decision out.

    Example
    -------
    >>> engine = EcoSentryInference.from_checkpoint("models/snn.npz")   # doctest: +SKIP
    >>> engine.process_audio(audio_10s)                                  # doctest: +SKIP
    {'alert': True, 'class_name': 'chainsaw', 'confidence': 0.91, ...}
    """

    def __init__(
        self,
        model: SpikingNetwork,
        audio_cfg: Optional[AudioConfig] = None,
        spike_cfg: Optional[SpikeConfig] = None,
        infer_cfg: Optional[InferenceConfig] = None,
        spike_gain: Optional[float] = None,
        n_frames: Optional[int] = None,
    ):
        self.model = model
        self.audio_cfg = audio_cfg or AudioConfig()
        self.spike_cfg = spike_cfg or SpikeConfig()
        self.cfg = infer_cfg or InferenceConfig()
        self.n_frames = n_frames
        meta_gain = (model.metadata or {}).get("spike_gain")
        self.spike_gain = spike_gain if spike_gain is not None else meta_gain

        self.history: Deque[int] = deque(maxlen=self.cfg.history_size)
        self.last_alert_time: Optional[float] = None
        self.sequence_number = 0

    # -- construction -------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, path, **kwargs) -> "EcoSentryInference":
        return cls(SpikingNetwork.load(Path(path)), **kwargs)

    # -- front-end ----------------------------------------------------------

    def audio_to_spikes(self, audio: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """ARCH_1 + ARCH_2 front-end -> spike tensor ``(T, 64, 1)``."""
        mel = extract_mel_spectrogram(
            audio, self.audio_cfg, sr=sr, n_frames=self.n_frames
        )
        return convert_mel_to_spikes(
            mel, self.spike_cfg, db_floor=self.audio_cfg.db_floor, gain=self.spike_gain
        )["spikes"]

    # -- inference ----------------------------------------------------------

    def classify_spikes(self, spikes: np.ndarray) -> Dict:
        logits = self.model.predict_logits(spikes)
        return classify_with_confidence(
            logits[0], self.model.class_names, self.cfg.softmax_temperature
        )

    def process_audio(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None,
        timestamp: Optional[float] = None,
        use_temporal_filter: bool = True,
    ) -> Dict:
        """Full edge path for one audio window.

        Returns the ARCH_5 output record, including the alert decision.
        """
        started = time.perf_counter()
        timestamp = time.time() if timestamp is None else timestamp

        t0 = time.perf_counter()
        mel = extract_mel_spectrogram(audio, self.audio_cfg, sr=sr, n_frames=self.n_frames)
        t_mel = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        spikes = convert_mel_to_spikes(
            mel, self.spike_cfg, db_floor=self.audio_cfg.db_floor, gain=self.spike_gain
        )["spikes"]
        t_spike = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        result = self.classify_spikes(spikes)
        t_snn = (time.perf_counter() - t0) * 1000.0

        return self._decide(
            result,
            timestamp,
            use_temporal_filter,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            breakdown={"mel_ms": t_mel, "spike_ms": t_spike, "snn_ms": t_snn},
            firing_rate=float(spikes.mean()),
        )

    def process_spikes(
        self,
        spikes: np.ndarray,
        timestamp: Optional[float] = None,
        use_temporal_filter: bool = True,
    ) -> Dict:
        """Inference path when spikes are already available (e.g. a dataset)."""
        started = time.perf_counter()
        timestamp = time.time() if timestamp is None else timestamp
        result = self.classify_spikes(spikes)
        return self._decide(
            result,
            timestamp,
            use_temporal_filter,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            breakdown={"snn_ms": (time.perf_counter() - started) * 1000.0},
            firing_rate=float(np.asarray(spikes).mean()),
        )

    # -- decision logic -----------------------------------------------------

    def _decide(
        self,
        result: Dict,
        timestamp: float,
        use_temporal_filter: bool,
        latency_ms: float,
        breakdown: Dict[str, float],
        firing_rate: float,
    ) -> Dict:
        self.history.append(result["class_id"])

        class_id = result["class_id"]
        confidence = result["confidence"]
        agreement = 1.0

        if use_temporal_filter and len(self.history) > 1:
            window = min(self.cfg.temporal_window, len(self.history))
            voted, agreement = temporal_filter(self.history, window)
            if voted is not None and voted != class_id:
                # The majority disagrees with this frame: down-weight it.
                confidence = confidence * agreement
                class_id = voted

        threshold = self.cfg.class_thresholds.get(class_id, self.cfg.default_threshold)
        alert = class_id in THREAT_CLASSES and confidence > threshold

        if alert:
            if (
                self.last_alert_time is not None
                and (timestamp - self.last_alert_time) < self.cfg.min_alert_interval_s
            ):
                alert = False  # rate limited
            else:
                self.last_alert_time = timestamp
                self.sequence_number = (self.sequence_number + 1) % 256

        return {
            "alert": bool(alert),
            "class_id": int(class_id),
            "class_name": self.model.class_names[class_id],
            "confidence": float(confidence),
            "raw_class_id": int(result["class_id"]),
            "raw_confidence": float(result["confidence"]),
            "threshold": float(threshold),
            "temporal_agreement": float(agreement),
            "probabilities": {
                name: float(p)
                for name, p in zip(self.model.class_names, result["probabilities"])
            },
            "entropy": float(result["entropy"]),
            "timestamp": float(timestamp),
            "timestamp_ms": int(timestamp * 1000),
            "sequence": int(self.sequence_number),
            "latency_ms": float(latency_ms),
            "latency_breakdown_ms": {k: float(v) for k, v in breakdown.items()},
            "input_firing_rate": firing_rate,
        }

    def reset(self) -> None:
        self.history.clear()
        self.last_alert_time = None


def safe_inference(engine: EcoSentryInference, audio: np.ndarray, **kwargs) -> Dict:
    """ARCH_5 error-handling wrapper: never raises, never falsely alerts."""
    try:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            raise ValueError("empty audio buffer")
        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            raise ValueError("audio contains NaN/Inf")
        return engine.process_audio(audio, **kwargs)
    except Exception as exc:  # noqa: BLE001 - deliberately broad on the edge device
        return {
            "alert": False,
            "class_id": -1,
            "class_name": "error",
            "confidence": 0.0,
            "error": str(exc),
            "timestamp": time.time(),
        }
