"""End-to-end integration: every architecture in one run."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ecosentry.pipeline import (
    QUICK,
    build_spike_dataset,
    run_dataset_stage,
    run_full_pipeline,
    run_simulation_stage,
    run_training_stage,
)
from ecosentry.synth import build_corpus, synthesize
from ecosentry.config import DatasetConfig, EcoSentryConfig

#: Tiny preset so the integration test stays inside a normal test budget.
TINY = replace(QUICK, name="tiny", duration_s=2.0, n_frames=200, n_samples=60, epochs=8)


def test_synth_corpus_covers_all_forests_and_classes():
    corpus = build_corpus(DatasetConfig(), 2.0, 16_000, seed=0, n_samples=80)
    forests = {s.forest for s in corpus}
    labels = {s.label for s in corpus}
    assert forests == {"corbett", "seshachalam", "sundarbans", "mixed"}
    assert labels.issuperset({0, 1, 2})
    assert all(np.abs(s.audio).max() <= 1.0 for s in corpus)


def test_synth_classes_are_spectrally_distinguishable():
    """Sanity check that the generator is not producing three flavours of noise."""
    from ecosentry.arch1_audio import extract_mel_spectrogram

    rng = np.random.default_rng(0)
    centroids = {}
    for label in (0, 1, 2):
        spectra = []
        for _ in range(4):
            audio = synthesize(label, "seshachalam", 3.0, 16_000, rng, snr_db=25.0)
            mel = extract_mel_spectrogram(audio)
            spectra.append(mel.mean(axis=0))
        centroids[label] = np.mean(spectra, axis=0)

    for a in (0, 1, 2):
        for b in (0, 1, 2):
            if a < b:
                distance = float(np.linalg.norm(centroids[a] - centroids[b]))
                assert distance > 1.0, f"classes {a}/{b} are spectrally identical"


def test_build_spike_dataset_calibrates_to_target_rate():
    corpus = build_corpus(DatasetConfig(), 2.0, 16_000, seed=1, n_samples=24)
    out = build_spike_dataset(corpus, n_frames=200, verbose=False)
    assert out["spikes"].shape == (len(corpus), 200, 64, 1)
    assert 0.18 <= out["firing_rate"] <= 0.32
    assert out["latency_ms"]["mel_mean"] < 100.0
    assert out["latency_ms"]["spike_mean"] < 50.0


def test_dataset_stage_writes_artifacts(tmp_path):
    result = run_dataset_stage(TINY, tmp_path, EcoSentryConfig(), seed=2, verbose=False)
    assert Path(result["dataset_path"]).exists()
    assert result["n_train"] > 0 and result["n_val"] > 0 and result["n_test"] > 0
    assert result["qa"]["mel"]["passed"]
    assert result["qa"]["spikes_normalized"]["passed"]


def test_training_stage_produces_a_loadable_model(tmp_path):
    stage1 = run_dataset_stage(TINY, tmp_path, EcoSentryConfig(), seed=2, verbose=False)
    stage2 = run_training_stage(
        stage1["_prepared"], None, TINY, tmp_path, spike_gain=stage1["spike_gain"], verbose=False
    )

    from ecosentry.arch4_training import SpikingNetwork

    model = SpikingNetwork.load(stage2["model_path"])
    assert model.metadata["spike_gain"] == pytest.approx(stage1["spike_gain"])
    assert model.metadata["n_frames"] == TINY.n_frames
    assert 0.0 <= stage2["test"]["accuracy"] <= 1.0
    assert stage2["history"]["epoch_losses"][-1] < stage2["history"]["epoch_losses"][0]


def test_simulation_stage_meets_network_targets(tmp_path):
    result = run_simulation_stage(TINY, tmp_path, EcoSentryConfig(), verbose=False)
    assert set(result["energy"]) == {"corbett", "seshachalam", "sundarbans"}
    for scenario, report in result["network"].items():
        baseline = report["adaptive_sf_baseline"]
        assert baseline["delivery_rate"] >= 0.95, scenario
        assert baseline["latency_p99_ms"] <= 1_500.0, scenario
    assert Path(tmp_path / "energy_report.json").exists()
    assert Path(tmp_path / "network_report.json").exists()


def test_full_pipeline_runs_and_reports(tmp_path):
    """Smoke test of every stage, plus the acceptance table."""
    import ecosentry.pipeline as pipeline

    pipeline.PRESETS["tiny"] = TINY
    run_full_pipeline("tiny", tmp_path, "corbett", seed=3, verbose=False)

    assert (tmp_path / "pipeline_report.json").exists()
    assert (tmp_path / "alert_events.json").exists()
    assert (tmp_path / "snn_model.npz").exists()

    written = json.loads((tmp_path / "pipeline_report.json").read_text())
    assert set(written["acceptance"]) >= {
        "snn_test_accuracy",
        "spike_firing_rate",
        "alert_payload_size",
        "network_delivery_rate",
        "power_reduction",
    }
    # Criteria that must hold regardless of how well the tiny model trained.
    for name in ("spike_firing_rate", "network_delivery_rate", "network_latency_p99"):
        assert written["acceptance"][name]["passed"], f"{name}: {written['acceptance'][name]}"


def test_alert_events_carry_a_verified_decryption(tmp_path):
    import ecosentry.pipeline as pipeline

    pipeline.PRESETS["tiny"] = TINY
    run_full_pipeline("tiny", tmp_path, "seshachalam", seed=4, verbose=False)

    events = json.loads((tmp_path / "alert_events.json").read_text())
    assert events, "no windows were processed"
    for event in events:
        if event.get("alert") and "decrypt_ok" in event:
            assert event["decrypt_ok"], "command centre could not decode the alert"
            assert event["payload_bytes"] < 1_000
