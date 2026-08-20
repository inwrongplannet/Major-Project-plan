"""ARCH_4 -- SNN training with surrogate-gradient BPTT (pure NumPy).

Architecture (ARCH_4 Component 1)::

    spikes (T, 64) -> LIF 128 -> LIF 64 -> non-spiking readout (3)

Forward dynamics per layer, at 10 ms frame resolution::

    V[t] = alpha * V[t-1] + (1 - alpha) * (s[t] @ W + b)
    s[t] = 1 if V[t] >= v_th else 0        (hidden layers only)
    V[t] <- 0 where s[t] == 1              (reset)

with ``alpha = exp(-dt / tau_m) = exp(-1) ~ 0.3679``.  The readout layer
integrates without a threshold and the final membrane potential is the logit
vector.

Backward pass is exact BPTT with a detached reset and an ArcTan surrogate for
the Heaviside derivative::

    ds/dV ~ a / (pi * (1 + (a * (V - v_th))^2)),   a = 2.0

Everything is batched over the sample axis, so one Python-level loop per time
step covers the whole mini-batch.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import CLASS_NAMES, SNNConfig

__all__ = [
    "SpikingNetwork",
    "Adam",
    "softmax",
    "cross_entropy",
    "train_snn",
    "evaluate",
    "TrainingHistory",
]


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row-wise softmax with max subtraction for numerical stability."""
    z = np.asarray(logits, dtype=np.float64) / max(temperature, 1e-8)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def cross_entropy(logits: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
    """Mean sparse cross-entropy loss and its gradient w.r.t. ``logits``."""
    probs = softmax(logits).astype(np.float64)
    n = len(labels)
    picked = np.clip(probs[np.arange(n), labels], 1e-12, 1.0)
    loss = float(-np.mean(np.log(picked)))

    grad = probs.copy()
    grad[np.arange(n), labels] -= 1.0
    return loss, (grad / n).astype(np.float32)


def _surrogate(v: np.ndarray, v_th: float, a: float) -> np.ndarray:
    """ArcTan surrogate derivative of the spike function."""
    x = a * (v - v_th)
    return (a / (np.pi * (1.0 + x * x))).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))).astype(np.float32)


def xavier_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SpikingNetwork:
    """Three-layer SNN with LIF hidden layers and an integrating readout."""

    def __init__(
        self,
        cfg: Optional[SNNConfig] = None,
        seed: Optional[int] = None,
        soft_spikes: bool = False,
        use_reset: bool = True,
    ):
        self.cfg = cfg or SNNConfig()
        #: Gradient-check aid: replace the Heaviside with a sigmoid in the
        #: *forward* pass so finite differences can validate the BPTT chain.
        #: Never enabled in production inference or training.
        self.soft_spikes = soft_spikes
        self.use_reset = use_reset
        rng = np.random.default_rng(self.cfg.seed if seed is None else seed)

        c = self.cfg
        self.params: Dict[str, np.ndarray] = {
            "W1": xavier_init(c.n_input, c.n_hidden1, rng),
            "b1": np.full(c.n_hidden1, 0.01, dtype=np.float32),
            "W2": xavier_init(c.n_hidden1, c.n_hidden2, rng),
            "b2": np.full(c.n_hidden2, 0.01, dtype=np.float32),
            "W3": xavier_init(c.n_hidden2, c.n_classes, rng),
            "b3": np.zeros(c.n_classes, dtype=np.float32),
        }
        self.class_names: List[str] = list(CLASS_NAMES[: c.n_classes])
        self.metadata: Dict = {}

    # -- shape helpers ------------------------------------------------------

    @staticmethod
    def _as_batch(x: np.ndarray) -> np.ndarray:
        """Accept ``(T, 64)``, ``(T, 64, 1)``, ``(B, T, 64)`` or ``(B, T, 64, 1)``."""
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[..., 0]
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0][None, ...]
        elif arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3:
            raise ValueError(f"unsupported spike tensor shape {np.shape(x)}")
        return arr

    @property
    def n_parameters(self) -> int:
        return int(sum(p.size for p in self.params.values()))

    def _spike(self, v: np.ndarray, v_th: float) -> np.ndarray:
        if self.soft_spikes:
            return _sigmoid(self.cfg.surrogate_alpha * (v - v_th))
        return (v >= v_th).astype(np.float32)

    def _spike_grad(self, v: np.ndarray, v_th: float) -> np.ndarray:
        if self.soft_spikes:
            s = _sigmoid(self.cfg.surrogate_alpha * (v - v_th))
            return (self.cfg.surrogate_alpha * s * (1.0 - s)).astype(np.float32)
        return _surrogate(v, v_th, self.cfg.surrogate_alpha)

    # -- forward ------------------------------------------------------------

    def forward(self, x: np.ndarray, record: bool = False) -> Dict:
        """Run the SNN forward.

        Returns a dict with ``logits`` ``(B, 3)`` and, when ``record`` is set,
        the per-timestep tensors required by :meth:`backward`.
        """
        c = self.cfg
        p = self.params
        alpha = np.float32(c.alpha)
        beta = np.float32(1.0 - c.alpha)
        readout_alpha = np.float32(c.readout_alpha)
        v_th = np.float32(c.v_threshold)

        xb = self._as_batch(x)
        B, T, _ = xb.shape

        v1 = np.zeros((B, c.n_hidden1), dtype=np.float32)
        v2 = np.zeros((B, c.n_hidden2), dtype=np.float32)
        v3 = np.zeros((B, c.n_classes), dtype=np.float32)

        if record:
            v1_pre = np.empty((T, B, c.n_hidden1), dtype=np.float32)
            v2_pre = np.empty((T, B, c.n_hidden2), dtype=np.float32)
            s1_all = np.empty((T, B, c.n_hidden1), dtype=np.float32)
            s2_all = np.empty((T, B, c.n_hidden2), dtype=np.float32)
        else:
            v1_pre = v2_pre = s1_all = s2_all = None

        spike_count_1 = 0.0
        spike_count_2 = 0.0

        for t in range(T):
            xt = xb[:, t, :]

            v1 = alpha * v1 + beta * (xt @ p["W1"] + p["b1"])
            s1 = self._spike(v1, v_th)
            if record:
                v1_pre[t] = v1
                s1_all[t] = s1
            if self.use_reset:
                v1 = v1 * (1.0 - s1)

            v2 = alpha * v2 + beta * (s1 @ p["W2"] + p["b2"])
            s2 = self._spike(v2, v_th)
            if record:
                v2_pre[t] = v2
                s2_all[t] = s2
            if self.use_reset:
                v2 = v2 * (1.0 - s2)

            v3 = readout_alpha * v3 + beta * (s2 @ p["W3"] + p["b3"])

            spike_count_1 += float(s1.sum())
            spike_count_2 += float(s2.sum())

        logits = v3 / T if c.readout_normalize else v3

        out = {
            "logits": logits,
            "hidden1_rate": spike_count_1 / max(B * T * c.n_hidden1, 1),
            "hidden2_rate": spike_count_2 / max(B * T * c.n_hidden2, 1),
        }
        if record:
            out["cache"] = {
                "x": xb,
                "v1_pre": v1_pre,
                "v2_pre": v2_pre,
                "s1": s1_all,
                "s2": s2_all,
            }
        return out

    # -- backward -----------------------------------------------------------

    def backward(self, cache: Dict, logits: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict]:
        """BPTT with surrogate gradients. Returns ``(loss, grads)``."""
        c = self.cfg
        p = self.params
        alpha = np.float32(c.alpha)
        beta = np.float32(1.0 - c.alpha)
        readout_alpha = np.float32(c.readout_alpha)
        v_th = c.v_threshold

        loss, dv3 = cross_entropy(logits, labels)

        x = cache["x"]
        v1_pre, v2_pre = cache["v1_pre"], cache["v2_pre"]
        s1_all, s2_all = cache["s1"], cache["s2"]
        T, B, _ = s1_all.shape

        if c.readout_normalize:
            dv3 = dv3 / np.float32(T)

        grads = {k: np.zeros_like(v) for k, v in p.items()}
        carry1 = np.zeros((B, c.n_hidden1), dtype=np.float32)
        carry2 = np.zeros((B, c.n_hidden2), dtype=np.float32)

        for t in range(T - 1, -1, -1):
            # --- readout layer (no threshold, no reset)
            di3 = beta * dv3
            grads["W3"] += s2_all[t].T @ di3
            grads["b3"] += di3.sum(axis=0)
            ds2 = di3 @ p["W3"].T
            dv3 = readout_alpha * dv3  # membrane carries to the previous frame

            # --- hidden layer 2
            dv2 = ds2 * self._spike_grad(v2_pre[t], v_th)
            if self.use_reset:
                dv2 = dv2 + carry2 * (1.0 - s2_all[t])  # detached reset
            else:
                dv2 = dv2 + carry2
            di2 = beta * dv2
            grads["W2"] += s1_all[t].T @ di2
            grads["b2"] += di2.sum(axis=0)
            ds1 = di2 @ p["W2"].T
            carry2 = alpha * dv2

            # --- hidden layer 1
            dv1 = ds1 * self._spike_grad(v1_pre[t], v_th)
            if self.use_reset:
                dv1 = dv1 + carry1 * (1.0 - s1_all[t])  # detached reset
            else:
                dv1 = dv1 + carry1
            di1 = beta * dv1
            grads["W1"] += x[:, t, :].T @ di1
            grads["b1"] += di1.sum(axis=0)
            carry1 = alpha * dv1

        # L2 regularisation on weights only.
        if c.weight_decay:
            for key in ("W1", "W2", "W3"):
                grads[key] += c.weight_decay * p[key]

        return loss, grads

    # -- convenience --------------------------------------------------------

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x, record=False)["logits"]

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns ``(class_ids, probabilities)``."""
        probs = softmax(self.predict_logits(x))
        return probs.argmax(axis=1).astype(np.int64), probs

    # -- persistence --------------------------------------------------------

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self.params)
        payload["_meta"] = np.frombuffer(
            json.dumps(
                {
                    "config": {
                        k: v
                        for k, v in self.cfg.__dict__.items()
                        if isinstance(v, (int, float, str, bool))
                    },
                    "class_names": self.class_names,
                    "metadata": _jsonable(self.metadata),
                    "created": time.time(),
                }
            ).encode("utf-8"),
            dtype=np.uint8,
        )
        np.savez_compressed(path, **payload)
        return path if path.suffix else path.with_suffix(".npz")

    @classmethod
    def load(cls, path) -> "SpikingNetwork":
        path = Path(path)
        if not path.exists() and path.with_suffix(".npz").exists():
            path = path.with_suffix(".npz")
        data = np.load(path, allow_pickle=False)

        meta = {}
        if "_meta" in data:
            meta = json.loads(bytes(data["_meta"]).decode("utf-8"))

        cfg_kwargs = {
            k: v
            for k, v in meta.get("config", {}).items()
            if k in SNNConfig.__dataclass_fields__
        }
        model = cls(SNNConfig(**cfg_kwargs))
        for key in model.params:
            model.params[key] = data[key].astype(np.float32)
        model.class_names = meta.get("class_names", model.class_names)
        model.metadata = meta.get("metadata", {})
        return model


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------


class Adam:
    """Adam with bias correction (ARCH_4 Component 5 hyperparameters)."""

    def __init__(self, params: Dict[str, np.ndarray], cfg: SNNConfig):
        self.cfg = cfg
        self.lr = cfg.learning_rate
        self.beta1 = cfg.adam_beta1
        self.beta2 = cfg.adam_beta2
        self.eps = cfg.adam_eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        self.t += 1
        bc1 = 1.0 - self.beta1**self.t
        bc2 = 1.0 - self.beta2**self.t
        for key, grad in grads.items():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad * grad)
            m_hat = self.m[key] / bc1
            v_hat = self.v[key] / bc2
            params[key] -= (self.lr * m_hat / (np.sqrt(v_hat) + self.eps)).astype(np.float32)


def calibrate_initialization(
    model: SpikingNetwork,
    sample_batch: np.ndarray,
    target_rate: float = 0.20,
    iterations: int = 14,
    verbose: bool = False,
) -> Dict[str, float]:
    """Scale each hidden weight matrix so its layer starts near ``target_rate``.

    Layers are calibrated front to back (layer 2 sees the already-scaled layer
    1).  Returns the scale factor applied to each matrix.
    """
    scales: Dict[str, float] = {}

    for key, rate_key in (("W1", "hidden1_rate"), ("W2", "hidden2_rate")):
        base = model.params[key].copy()
        lo, hi = 0.05, 200.0
        best = 1.0
        for _ in range(iterations):
            mid = float(np.sqrt(lo * hi))
            model.params[key] = (base * mid).astype(np.float32)
            rate = model.forward(sample_batch, record=False)[rate_key]
            if rate < target_rate:
                lo = mid
            else:
                hi = mid
            best = mid
        model.params[key] = (base * best).astype(np.float32)
        scales[key] = best
        if verbose:
            achieved = model.forward(sample_batch, record=False)[rate_key]
            print(f"  init calibration {key}: scale x{best:.2f} -> rate {achieved:.1%}")

    # Readout: bring the logits to unit scale.  The T-normalised accumulator
    # otherwise starts at ~1e-2, which leaves the softmax flat and the
    # cross-entropy gradient tiny.
    logits = model.forward(sample_batch, record=False)["logits"]
    spread = float(np.std(logits))
    if spread > 1e-9:
        scale = 1.0 / spread
        model.params["W3"] = (model.params["W3"] * scale).astype(np.float32)
        model.params["b3"] = (model.params["b3"] * scale).astype(np.float32)
        scales["W3"] = scale
        if verbose:
            print(f"  init calibration W3: scale x{scale:.2f} -> logit std 1.00")

    return scales


def clip_gradients(grads: Dict[str, np.ndarray], max_norm: float) -> float:
    """Global-norm gradient clipping.  Returns the pre-clip norm."""
    total = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads.values())))
    if max_norm > 0 and total > max_norm:
        scale = max_norm / (total + 1e-12)
        for key in grads:
            grads[key] *= scale
    return total


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@dataclass
class TrainingHistory:
    epoch_losses: List[float]
    train_accuracies: List[float]
    val_losses: List[float]
    val_accuracies: List[float]
    best_val_loss: float
    best_epoch: int
    epochs_run: int
    seconds: float

    def to_dict(self) -> Dict:
        return {
            "epoch_losses": self.epoch_losses,
            "train_accuracies": self.train_accuracies,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "epochs_run": self.epochs_run,
            "seconds": self.seconds,
        }


def evaluate(
    model: SpikingNetwork,
    spikes: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
) -> Dict:
    """Loss / accuracy / confusion matrix over a dataset."""
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    if n == 0:
        return {"loss": float("nan"), "accuracy": float("nan"), "n": 0}

    total_loss = 0.0
    preds = np.empty(n, dtype=np.int64)
    all_probs = np.empty((n, model.cfg.n_classes), dtype=np.float32)

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        logits = model.predict_logits(spikes[start:stop])
        loss, _ = cross_entropy(logits, labels[start:stop])
        total_loss += loss * (stop - start)
        probs = softmax(logits)
        all_probs[start:stop] = probs
        preds[start:stop] = probs.argmax(axis=1)

    k = model.cfg.n_classes
    confusion = np.zeros((k, k), dtype=np.int64)
    for true, pred in zip(labels, preds):
        if 0 <= true < k:
            confusion[true, pred] += 1

    per_class = {}
    for c in range(k):
        tp = int(confusion[c, c])
        support = int(confusion[c].sum())
        predicted = int(confusion[:, c].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[model.class_names[c]] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    return {
        "loss": total_loss / n,
        "accuracy": float(np.mean(preds == labels)),
        "n": int(n),
        "confusion": confusion,
        "per_class": per_class,
        "predictions": preds,
        "probabilities": all_probs,
    }


def train_snn(
    train_spikes: np.ndarray,
    train_labels: np.ndarray,
    val_spikes: np.ndarray,
    val_labels: np.ndarray,
    cfg: Optional[SNNConfig] = None,
    model: Optional[SpikingNetwork] = None,
    verbose: bool = True,
    log_every: int = 5,
) -> Tuple[SpikingNetwork, TrainingHistory]:
    """Mini-batch surrogate-gradient training with early stopping.

    Returns the best-validation-loss model and its training history.
    """
    cfg = cfg or SNNConfig()
    model = model or SpikingNetwork(cfg)

    if cfg.calibrate_init:
        probe = train_spikes[: min(cfg.batch_size, len(train_spikes))]
        calibrate_initialization(model, probe, cfg.init_target_rate, verbose=verbose)

    optimizer = Adam(model.params, cfg)
    rng = np.random.default_rng(cfg.seed)

    train_labels = np.asarray(train_labels, dtype=np.int64)
    val_labels = np.asarray(val_labels, dtype=np.int64)
    n = len(train_labels)

    history = TrainingHistory([], [], [], [], float("inf"), -1, 0, 0.0)
    best_params = {k: v.copy() for k, v in model.params.items()}
    patience = 0
    started = time.time()

    for epoch in range(cfg.epochs):
        order = rng.permutation(n)
        epoch_loss = 0.0
        correct = 0

        for start in range(0, n, cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            xb = train_spikes[idx]
            yb = train_labels[idx]

            out = model.forward(xb, record=True)
            loss, grads = model.backward(out["cache"], out["logits"], yb)
            clip_gradients(grads, cfg.grad_clip)
            optimizer.step(model.params, grads)

            epoch_loss += loss * len(idx)
            correct += int(np.sum(softmax(out["logits"]).argmax(axis=1) == yb))

        train_loss = epoch_loss / max(n, 1)
        train_acc = correct / max(n, 1)
        val = evaluate(model, val_spikes, val_labels, cfg.batch_size)

        history.epoch_losses.append(train_loss)
        history.train_accuracies.append(train_acc)
        history.val_losses.append(val["loss"])
        history.val_accuracies.append(val["accuracy"])
        history.epochs_run = epoch + 1

        if (epoch + 1) % cfg.lr_decay_every == 0:
            optimizer.lr *= cfg.lr_decay

        if val["loss"] < history.best_val_loss - 1e-6:
            history.best_val_loss = val["loss"]
            history.best_epoch = epoch
            best_params = {k: v.copy() for k, v in model.params.items()}
            patience = 0
        else:
            patience += 1

        if verbose and ((epoch + 1) % log_every == 0 or epoch == 0):
            print(
                f"  epoch {epoch + 1:3d}/{cfg.epochs}  "
                f"loss={train_loss:.4f} acc={train_acc:.3f}  "
                f"val_loss={val['loss']:.4f} val_acc={val['accuracy']:.3f}  "
                f"lr={optimizer.lr:.2e}"
            )

        if patience >= cfg.early_stopping_patience:
            if verbose:
                print(f"  early stopping at epoch {epoch + 1} (patience exhausted)")
            break

    model.params = best_params
    history.seconds = time.time() - started
    model.metadata = {
        "training_history": history.to_dict(),
        "n_train": int(n),
        "n_val": int(len(val_labels)),
        "n_parameters": model.n_parameters,
    }
    return model, history
