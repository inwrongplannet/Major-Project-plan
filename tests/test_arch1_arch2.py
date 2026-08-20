"""ARCH_1 / ARCH_2 contract tests: shapes, ranges, latency and sparsity."""

from __future__ import annotations

import time

import numpy as np
import pytest

from ecosentry.arch1_audio import (
    bandpass_filter,
    create_mel_filterbank,
    extract_mel_spectrogram,
    fit_frames,
    frame_audio,
    hz_to_mel,
    load_audio,
    mel_to_hz,
    normalize_rms,
    validate_mel_spectrogram,
)
from ecosentry.arch2_spikes import (
    calibrate_input_gain,
    compute_spike_latencies,
    convert_mel_to_spikes,
    firing_rate,
    lif_neurons,
    normalize_mel_input,
    sparse_to_spike,
    spike_to_sparse,
    validate_spikes,
)
from ecosentry.config import AudioConfig, SpikeConfig
from ecosentry.synth import synthesize, write_wav


@pytest.fixture(scope="module")
def audio() -> np.ndarray:
    return synthesize(0, "corbett", 10.0, 16_000, np.random.default_rng(0))


@pytest.fixture(scope="module")
def mel(audio) -> np.ndarray:
    return extract_mel_spectrogram(audio, AudioConfig())


# --- ARCH_1 -----------------------------------------------------------------


def test_mel_shape_is_T_by_64(mel):
    assert mel.ndim == 2
    assert mel.shape[1] == 64
    # 10 s at a 10 ms hop -> ~1000 frames (documented T).
    assert 990 <= mel.shape[0] <= 1010


def test_mel_values_in_documented_db_range(mel):
    assert mel.min() >= -80.0 - 1e-3
    assert mel.max() <= 0.0 + 1e-3
    assert not np.any(np.isnan(mel))
    assert not np.any(np.isinf(mel))


def test_mel_validation_passes(mel):
    assert validate_mel_spectrogram(mel)["passed"]


def test_mel_latency_under_100ms(audio):
    start = time.perf_counter()
    extract_mel_spectrogram(audio, AudioConfig())
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert elapsed_ms < 100.0, f"ARCH_1 latency budget exceeded: {elapsed_ms:.1f} ms"


def test_rms_normalization_hits_target():
    rng = np.random.default_rng(1)
    raw = rng.standard_normal(16_000).astype(np.float32) * 0.01
    out = normalize_rms(raw, 0.1)
    assert np.isclose(np.sqrt(np.mean(out**2)), 0.1, atol=1e-4)


def test_mel_hz_roundtrip():
    hz = np.array([50.0, 440.0, 1_000.0, 8_000.0])
    assert np.allclose(mel_to_hz(hz_to_mel(hz)), hz, rtol=1e-6)


def test_mel_filterbank_has_no_empty_filters():
    fb = create_mel_filterbank(64, 16_000, 512, 50.0, 8_000.0)
    assert fb.shape == (64, 257)
    assert np.all(fb.sum(axis=1) > 0), "some mel band has no FFT bin support"


def test_bandpass_attenuates_out_of_band():
    sr = 16_000
    t = np.arange(sr) / sr
    low_tone = np.sin(2 * np.pi * 20 * t).astype(np.float32)   # below 100 Hz
    in_band = np.sin(2 * np.pi * 1_000 * t).astype(np.float32)
    assert np.std(bandpass_filter(low_tone, sr=sr)) < 0.1 * np.std(low_tone)
    assert np.std(bandpass_filter(in_band, sr=sr)) > 0.5 * np.std(in_band)


def test_frame_audio_frame_count():
    audio = np.zeros(16_000, dtype=np.float32)
    frames = frame_audio(audio, 512, 160, center=True)
    assert frames.shape == (1 + 16_000 // 160, 512)


def test_fit_frames_pads_and_truncates():
    x = np.zeros((10, 64), dtype=np.float32)
    assert fit_frames(x, 20, -80.0).shape == (20, 64)
    assert fit_frames(x, 20, -80.0)[15, 0] == -80.0
    assert fit_frames(x, 5).shape == (5, 64)


def test_wav_roundtrip(tmp_path, audio):
    path = write_wav(tmp_path / "clip.wav", audio, 16_000)
    loaded, sr = load_audio(path, 16_000, 10.0)
    assert sr == 16_000
    assert len(loaded) == len(audio)
    assert np.corrcoef(loaded, audio)[0, 1] > 0.99


def test_resampling_from_44100(audio):
    from ecosentry.arch1_audio import resample

    upsampled = resample(audio, 16_000, 44_100)
    mel = extract_mel_spectrogram(upsampled, AudioConfig(), sr=44_100)
    assert mel.shape[1] == 64
    assert 990 <= mel.shape[0] <= 1010


# --- ARCH_2 -----------------------------------------------------------------


def test_spike_tensor_shape_is_frame_level(mel):
    out = convert_mel_to_spikes(mel, auto_gain=True)
    spikes = out["spikes"]
    assert spikes.shape == (mel.shape[0], 64, 1)
    # The classic mistake the docs call out: sample-level, not frame-level.
    assert spikes.shape[0] < 2_000, "spikes must be frame-level, not 16 kHz"


def test_spikes_are_binary(mel):
    spikes = convert_mel_to_spikes(mel, auto_gain=True)["spikes"]
    assert set(np.unique(spikes)).issubset({0, 1})


def test_calibrated_firing_rate_hits_25_percent(mel):
    out = convert_mel_to_spikes(mel, auto_gain=True, target_rate=0.25)
    assert 0.20 <= float(out["firing_rate"]) <= 0.30


def test_spike_validation_passes(mel):
    spikes = convert_mel_to_spikes(mel, auto_gain=True)["spikes"]
    assert validate_spikes(spikes)["passed"]


def test_lif_alpha_matches_documented_value():
    cfg = SpikeConfig()
    assert abs(cfg.alpha - np.exp(-1.0)) < 1e-6


def test_refractory_period_enforced():
    cfg = SpikeConfig(refractory_ms=20.0, hop_ms=10.0)  # 2 frames refractory
    drive = np.ones((50, 4), dtype=np.float32)
    spikes = lif_neurons(drive, cfg, gain=10.0)
    for neuron in range(4):
        idx = np.flatnonzero(spikes[:, neuron])
        if len(idx) > 1:
            assert np.all(np.diff(idx) > cfg.refractory_frames)


def test_higher_drive_gives_more_spikes():
    drive = np.linspace(0, 1, 200, dtype=np.float32)[:, None].repeat(8, axis=1)
    low = firing_rate(lif_neurons(drive, SpikeConfig(), gain=1.0))
    high = firing_rate(lif_neurons(drive, SpikeConfig(), gain=4.0))
    assert high > low


def test_silent_input_produces_no_spikes():
    drive = np.zeros((100, 64), dtype=np.float32)
    assert firing_rate(lif_neurons(drive, SpikeConfig(), gain=5.0)) == 0.0


def test_mel_input_normalization_range(mel):
    drive = normalize_mel_input(mel, -80.0)
    assert drive.min() >= 0.0 and drive.max() <= 1.0


def test_ttfs_latencies(mel):
    spikes = convert_mel_to_spikes(mel, auto_gain=True)["spikes"][:, :, 0]
    latencies, rank, idx = compute_spike_latencies(spikes, 10.0)
    assert latencies.shape == (64,)
    assert rank.shape == (64,)
    finite = np.isfinite(latencies)
    assert np.all(latencies[finite] == idx[finite] * 10.0)
    # Rank order must be sorted by latency.
    assert np.all(np.diff(latencies[rank][np.isfinite(latencies[rank])]) >= 0)


def test_sparse_roundtrip(mel):
    spikes = convert_mel_to_spikes(mel, auto_gain=True)["spikes"]
    restored = sparse_to_spike(spike_to_sparse(spikes))
    assert np.array_equal(restored, spikes)


def test_calibrate_gain_is_monotone_effective(mel):
    drive = normalize_mel_input(mel, -80.0)
    gain_low = calibrate_input_gain(drive, 0.10, SpikeConfig())
    gain_high = calibrate_input_gain(drive, 0.35, SpikeConfig())
    assert gain_high > gain_low
