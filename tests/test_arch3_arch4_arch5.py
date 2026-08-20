"""ARCH_3 / ARCH_4 / ARCH_5 tests: dataset prep, gradients, training, inference."""

from __future__ import annotations

import numpy as np
import pytest

from ecosentry.arch3_dataset import (
    add_spike_noise,
    apply_forest_normalization,
    create_augmented_dataset,
    create_stratified_splits,
    load_prepared_dataset,
    mixup_spikes,
    prepare_dataset,
    rate_normalize,
    save_prepared_dataset,
    time_shift_spikes,
    validate_normalized_spikes,
    validate_splits,
)
from ecosentry.arch4_training import (
    SpikingNetwork,
    calibrate_initialization,
    clip_gradients,
    cross_entropy,
    evaluate,
    softmax,
    train_snn,
)
from ecosentry.arch5_inference import (
    EcoSentryInference,
    adaptive_thresholding,
    classify_with_confidence,
    detect_threat,
    quantization_error,
    quantize_weights,
    safe_inference,
    temporal_filter,
)
from ecosentry.config import DatasetConfig, InferenceConfig, SNNConfig


@pytest.fixture(scope="module")
def toy_dataset():
    """Small, linearly separable spike dataset: class k fires in its own band."""
    rng = np.random.default_rng(0)
    n_per_class, T, C = 24, 40, 64
    spikes, labels, forests = [], [], []
    bands = {0: (0, 20), 1: (20, 42), 2: (42, 64)}
    for label, (lo, hi) in bands.items():
        for i in range(n_per_class):
            x = (rng.random((T, C)) < 0.05).astype(np.float32)
            x[:, lo:hi] = (rng.random((T, hi - lo)) < 0.55).astype(np.float32)
            spikes.append(x[:, :, None])
            labels.append(label)
            forests.append(["corbett", "seshachalam", "sundarbans", "mixed"][i % 4])
    return (
        np.array(spikes, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        np.array(forests),
    )


# --- ARCH_3 -----------------------------------------------------------------


def test_rate_normalize_hits_target_up_and_down():
    rng = np.random.default_rng(1)
    dense = (rng.random((100, 64)) < 0.60).astype(np.float32)
    sparse = (rng.random((100, 64)) < 0.05).astype(np.float32)
    assert abs(rate_normalize(dense, 0.25, rng).mean() - 0.25) < 0.01
    assert abs(rate_normalize(sparse, 0.25, rng).mean() - 0.25) < 0.01


def test_forest_normalization_reaches_target_rate(toy_dataset):
    spikes, _, forests = toy_dataset
    cfg = DatasetConfig()
    out = apply_forest_normalization(spikes, forests, cfg)
    assert out.shape == spikes.shape
    assert abs(float((out > 0).mean()) - cfg.target_firing_rate) < 0.02


def test_normalization_preserves_binary_values(toy_dataset):
    spikes, _, forests = toy_dataset
    out = apply_forest_normalization(spikes, forests, DatasetConfig())
    assert set(np.unique(out)).issubset({0.0, 1.0})


def test_splits_have_no_leakage(toy_dataset):
    spikes, labels, forests = toy_dataset
    split = create_stratified_splits(spikes, labels, forests, DatasetConfig())
    checks = validate_splits(split)
    assert checks["no_overlap"]
    assert checks["total_samples"] == len(labels)


def test_splits_are_stratified(toy_dataset):
    spikes, labels, forests = toy_dataset
    split = create_stratified_splits(spikes, labels, forests, DatasetConfig())
    for name in ("train", "val", "test"):
        present = set(np.unique(split[name]["labels"]).tolist())
        assert present == {0, 1, 2}, f"{name} split is missing a class"


def test_split_ratios_roughly_60_20_20(toy_dataset):
    spikes, labels, forests = toy_dataset
    split = create_stratified_splits(spikes, labels, forests, DatasetConfig())
    n = len(labels)
    assert 0.5 <= len(split["train"]["labels"]) / n <= 0.7
    assert 0.1 <= len(split["val"]["labels"]) / n <= 0.3
    assert 0.1 <= len(split["test"]["labels"]) / n <= 0.3


def test_augmentation_doubles_training_set(toy_dataset):
    spikes, labels, _ = toy_dataset
    aug_x, aug_y = create_augmented_dataset(spikes, labels, DatasetConfig())
    assert len(aug_x) == 2 * len(spikes)
    assert len(aug_y) == 2 * len(labels)
    assert np.array_equal(aug_x[: len(spikes)], spikes), "originals must be preserved"


def test_time_shift_preserves_spike_count():
    rng = np.random.default_rng(2)
    x = (rng.random((50, 64, 1)) < 0.25).astype(np.float32)
    shifted = time_shift_spikes(x, 5, rng)
    assert shifted.sum() == x.sum()


def test_mixup_interpolates():
    a = np.ones((10, 4, 1), dtype=np.float32)
    b = np.zeros((10, 4, 1), dtype=np.float32)
    mixed, label, lam = mixup_spikes(a, b, 0, 1, 0.2, np.random.default_rng(3))
    assert np.allclose(mixed, lam)
    assert label == (0 if lam >= 0.5 else 1)


def test_spike_noise_flips_expected_fraction():
    rng = np.random.default_rng(4)
    x = np.zeros((200, 64, 1), dtype=np.float32)
    noisy = add_spike_noise(x, 0.05, rng)
    assert 0.03 < noisy.mean() < 0.07


def test_prepare_and_persist_dataset(tmp_path, toy_dataset):
    spikes, labels, forests = toy_dataset
    prepared = prepare_dataset(spikes, labels, forests, DatasetConfig(), verbose=False)
    path = save_prepared_dataset(prepared, tmp_path / "d.h5")
    loaded = load_prepared_dataset(path)
    assert loaded["train"]["spikes"].shape == prepared["train"]["spikes"].shape
    assert np.array_equal(loaded["test"]["labels"], prepared["test"]["labels"])
    assert validate_normalized_spikes(loaded["val"]["spikes"], DatasetConfig())["in_range_0_1"]


# --- ARCH_4 -----------------------------------------------------------------


def test_network_shapes_match_spec():
    m = SpikingNetwork(SNNConfig())
    assert m.params["W1"].shape == (64, 128)
    assert m.params["W2"].shape == (128, 64)
    assert m.params["W3"].shape == (64, 3)
    assert m.params["b1"].shape == (128,)
    # 64*128 + 128 + 128*64 + 64 + 64*3 + 3 = 16,579 with the ARCH_4 layer sizes
    assert m.n_parameters == 64 * 128 + 128 + 128 * 64 + 64 + 64 * 3 + 3


def test_alpha_matches_documented_value():
    assert abs(SNNConfig().alpha - np.exp(-1.0)) < 1e-6


def test_forward_accepts_all_documented_shapes():
    m = SpikingNetwork(SNNConfig())
    T = 20
    for shape in [(T, 64), (T, 64, 1), (4, T, 64), (4, T, 64, 1)]:
        x = np.zeros(shape, dtype=np.float32)
        logits = m.forward(x)["logits"]
        assert logits.shape[1] == 3


def test_softmax_sums_to_one():
    probs = softmax(np.array([[3.2, -1.5, -0.8]], dtype=np.float32))
    assert np.isclose(probs.sum(), 1.0)
    assert np.argmax(probs) == 0


def test_cross_entropy_gradient_is_probs_minus_onehot():
    logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    loss, grad = cross_entropy(logits, np.array([2]))
    expected = softmax(logits)[0].copy()
    expected[2] -= 1.0
    assert np.allclose(grad[0], expected, atol=1e-5)
    assert loss > 0


def test_bptt_gradients_match_numerical():
    """Validate the BPTT chain with a smooth forward pass.

    With the real Heaviside spike the true derivative is zero almost
    everywhere, so finite differences cannot check the hidden layers -- that is
    exactly why surrogate gradients exist.  Using a sigmoid forward (and no
    reset, which is intentionally detached in the backward pass) makes the
    network smooth so the chain itself becomes verifiable.
    """
    cfg = SNNConfig(n_input=8, n_hidden1=6, n_hidden2=5, n_classes=3, seed=0, weight_decay=0.0)
    m = SpikingNetwork(cfg, soft_spikes=True, use_reset=False)
    rng = np.random.default_rng(3)
    x = (rng.random((4, 12, 8)) < 0.3).astype(np.float32)
    y = np.array([0, 1, 2, 1])

    out = m.forward(x, record=True)
    _, grads = m.backward(out["cache"], out["logits"], y)

    eps = 1e-3
    worst = 0.0
    checked = 0
    for key in ("W1", "W2", "W3", "b1", "b2", "b3"):
        P = m.params[key]
        for _ in range(6):
            idx = tuple(int(rng.integers(0, s)) for s in P.shape)
            original = float(P[idx])
            P[idx] = original + eps
            loss_plus, _ = cross_entropy(m.forward(x)["logits"], y)
            P[idx] = original - eps
            loss_minus, _ = cross_entropy(m.forward(x)["logits"], y)
            P[idx] = original

            numeric = (loss_plus - loss_minus) / (2 * eps)
            if abs(numeric) < 1e-3:  # below float32 finite-difference noise
                continue
            checked += 1
            worst = max(worst, abs(numeric - float(grads[key][idx])) / abs(numeric))

    assert checked >= 10, "not enough significant gradients sampled"
    assert worst < 0.05, f"analytic vs numerical gradient mismatch: {worst:.3f}"


def test_surrogate_gradient_is_nonzero_at_threshold():
    from ecosentry.arch4_training import _surrogate

    assert _surrogate(np.array([1.0]), 1.0, 2.0)[0] > 0.5


def test_initialization_calibration_wakes_up_hidden_layers():
    rng = np.random.default_rng(5)
    x = (rng.random((8, 30, 64)) < 0.25).astype(np.float32)
    m = SpikingNetwork(SNNConfig())

    before = m.forward(x)
    assert before["hidden2_rate"] < 0.01, "precondition: raw Xavier init is silent"

    calibrate_initialization(m, x, target_rate=0.20)
    after = m.forward(x)
    assert 0.10 < after["hidden1_rate"] < 0.35
    assert 0.10 < after["hidden2_rate"] < 0.35


def test_gradient_clipping():
    grads = {"a": np.array([3.0, 4.0], dtype=np.float32)}
    norm = clip_gradients(grads, 1.0)
    assert np.isclose(norm, 5.0)
    assert np.isclose(np.linalg.norm(grads["a"]), 1.0, atol=1e-5)


def test_training_learns_separable_data(toy_dataset):
    spikes, labels, forests = toy_dataset
    split = create_stratified_splits(spikes, labels, forests, DatasetConfig())
    cfg = SNNConfig(epochs=30, batch_size=8, learning_rate=3e-3, n_hidden1=32, n_hidden2=16)

    model, history = train_snn(
        split["train"]["spikes"],
        split["train"]["labels"],
        split["val"]["spikes"],
        split["val"]["labels"],
        cfg,
        verbose=False,
    )
    result = evaluate(model, split["test"]["spikes"], split["test"]["labels"])

    assert history.epoch_losses[-1] < history.epoch_losses[0], "loss must decrease"
    assert result["accuracy"] > 0.8, f"separable data should train: {result['accuracy']:.2f}"
    assert result["confusion"].sum() == len(split["test"]["labels"])


def test_model_save_load_roundtrip(tmp_path):
    m = SpikingNetwork(SNNConfig())
    m.metadata = {"spike_gain": 1.7, "n_frames": 250}
    path = m.save(tmp_path / "model.npz")

    restored = SpikingNetwork.load(path)
    for key in m.params:
        assert np.array_equal(restored.params[key], m.params[key])
    assert restored.metadata["spike_gain"] == 1.7
    assert restored.class_names == m.class_names


def test_evaluate_reports_per_class_metrics(toy_dataset):
    spikes, labels, _ = toy_dataset
    m = SpikingNetwork(SNNConfig())
    result = evaluate(m, spikes[:12], labels[:12])
    assert set(result["per_class"]) == {"gunshot", "chainsaw", "vehicle"}
    assert result["confusion"].shape == (3, 3)


# --- ARCH_5 -----------------------------------------------------------------


def test_classify_with_confidence():
    result = classify_with_confidence(
        np.array([-0.5, 2.8, 0.3]), ["gunshot", "chainsaw", "vehicle"]
    )
    assert result["class_id"] == 1
    assert result["class_name"] == "chainsaw"
    assert 0.85 < result["confidence"] < 0.95
    assert result["entropy"] > 0


def test_detect_threat_respects_threshold():
    threat = {"class_id": 0, "confidence": 0.91}
    non_threat = {"class_id": 2, "confidence": 0.99}
    assert detect_threat(threat, 0.85)
    assert not detect_threat({"class_id": 0, "confidence": 0.5}, 0.85)
    assert not detect_threat(non_threat, 0.85), "vehicle is not a threat class"


def test_adaptive_thresholds_are_per_class():
    cfg = InferenceConfig()
    alert_g, _, thr_g = adaptive_thresholding({"class_id": 0, "confidence": 0.85}, cfg.class_thresholds)
    alert_c, _, thr_c = adaptive_thresholding({"class_id": 1, "confidence": 0.85}, cfg.class_thresholds)
    assert thr_g == 0.90 and thr_c == 0.80
    assert not alert_g and alert_c


def test_temporal_filter_majority_vote():
    winner, agreement = temporal_filter([1, 1, 0, 1, 1], 5)
    assert winner == 1
    assert agreement == 0.8


def test_alert_rate_limiting():
    m = SpikingNetwork(SNNConfig())
    engine = EcoSentryInference(m, infer_cfg=InferenceConfig(min_alert_interval_s=10.0))
    # Force a confident gunshot readout.
    m.params["b3"] = np.array([12.0, 0.0, 0.0], dtype=np.float32)
    spikes = np.zeros((20, 64, 1), dtype=np.float32)

    first = engine.process_spikes(spikes, timestamp=1_000.0, use_temporal_filter=False)
    second = engine.process_spikes(spikes, timestamp=1_001.0, use_temporal_filter=False)
    third = engine.process_spikes(spikes, timestamp=1_100.0, use_temporal_filter=False)

    assert first["alert"]
    assert not second["alert"], "second alert within the interval must be suppressed"
    assert third["alert"]


def test_int8_quantization_error_is_small():
    m = SpikingNetwork(SNNConfig())
    q = quantize_weights(m)
    assert q["W1"]["weights"].dtype == np.int8
    for key, error in quantization_error(m).items():
        assert error < 0.05, f"{key} INT8 error too high: {error:.3f}"


def test_safe_inference_never_raises():
    engine = EcoSentryInference(SpikingNetwork(SNNConfig()))
    result = safe_inference(engine, np.array([np.nan, 1.0], dtype=np.float32))
    assert result["alert"] is False
    assert "error" in result


def test_end_to_end_audio_path_produces_a_decision():
    from ecosentry.synth import synthesize

    m = SpikingNetwork(SNNConfig())
    m.metadata = {"spike_gain": 2.0}
    engine = EcoSentryInference(m, n_frames=100)
    audio = synthesize(1, "seshachalam", 2.0, 16_000, np.random.default_rng(7))

    result = engine.process_audio(audio, 16_000, use_temporal_filter=False)
    assert result["class_name"] in ("gunshot", "chainsaw", "vehicle")
    assert 0.0 <= result["confidence"] <= 1.0
    assert 0.0 < result["input_firing_rate"] < 1.0
    assert result["latency_ms"] > 0
