"""End-to-end orchestration: raw audio -> alert delivered at the command centre.

Stage map::

    synth/field audio
        -> ARCH_1 mel-spectrogram      (arch1_audio)
        -> ARCH_2 spike trains         (arch2_spikes)
        -> ARCH_3 normalise/split/aug  (arch3_dataset)
        -> ARCH_4 train the SNN        (arch4_training)
        -> ARCH_5 inference + alerting (arch5_inference)
        -> ARCH_6 encrypted payload    (arch6_payload)
        -> ARCH_8 LoRa mesh delivery   (arch8_network)
        -> gateway priority proxy      (gateway)
        -> ARCH_7 energy profile       (arch7_energy)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import arch3_dataset as a3
from .arch1_audio import extract_mel_spectrogram, validate_mel_spectrogram
from .arch2_spikes import (
    calibrate_input_gain,
    convert_mel_to_spikes,
    normalize_mel_input,
    validate_spikes,
)
from .arch4_training import SpikingNetwork, evaluate, train_snn
from .arch5_inference import EcoSentryInference
from .arch6_payload import DeviceConfig, generate_alert_payload, parse_alert_message
from .arch7_energy import compare_scenarios, plot_energy_reports
from .arch8_network import (
    LoRaPHY,
    compare_network_scenarios,
    create_topology,
    simulate_message_delivery,
)
from .config import (
    CLASS_NAMES,
    SCENARIOS,
    AudioConfig,
    EcoSentryConfig,
    SpikeConfig,
)
from .gateway import BackhaulLink, MqttSink, PriorityGateway
from .synth import SynthSample, build_corpus

__all__ = [
    "PipelinePreset",
    "QUICK",
    "FULL",
    "build_spike_dataset",
    "run_dataset_stage",
    "run_training_stage",
    "run_alert_stage",
    "run_simulation_stage",
    "run_full_pipeline",
]

TRAIN_CLASSES = (0, 1, 2)  # ARCH_4/ARCH_5 head; "ambient" is a held-out negative


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelinePreset:
    """Knobs that trade fidelity against wall-clock time."""

    name: str
    duration_s: float
    n_frames: int
    n_samples: Optional[int]
    epochs: int
    batch_size: int
    network_messages: int
    mission_days: int


#: ~1 minute end to end -- 2.5 s clips, 120 samples, 25 epochs.
QUICK = PipelinePreset("quick", 2.5, 250, 120, 60, 16, 100, 30)

#: The documented configuration -- 10 s clips, 600 samples, 120 epochs.
FULL = PipelinePreset("full", 10.0, 1000, None, 120, 32, 200, 30)

PRESETS = {"quick": QUICK, "full": FULL}


def _configs(preset: PipelinePreset, base: Optional[EcoSentryConfig] = None):
    base = base or EcoSentryConfig()
    audio = replace(base.audio, max_duration_s=preset.duration_s)
    dataset = replace(base.dataset, n_frames=preset.n_frames)
    snn = replace(base.snn, epochs=preset.epochs, batch_size=preset.batch_size)
    energy = replace(base.energy, mission_days=preset.mission_days)
    return audio, dataset, snn, energy


# ---------------------------------------------------------------------------
# ARCH_1 + ARCH_2: corpus -> spike tensors
# ---------------------------------------------------------------------------


def build_spike_dataset(
    corpus: Sequence[SynthSample],
    audio_cfg: Optional[AudioConfig] = None,
    spike_cfg: Optional[SpikeConfig] = None,
    n_frames: int = 1_000,
    target_rate: float = 0.25,
    calibration_samples: int = 12,
    verbose: bool = True,
) -> Dict:
    """Run ARCH_1 + ARCH_2 over a corpus.

    The LIF input gain is calibrated once on a small subset so the corpus-wide
    firing rate lands on ``target_rate`` -- the ARCH_3 acceptance criterion.
    """
    audio_cfg = audio_cfg or AudioConfig()
    spike_cfg = spike_cfg or SpikeConfig()

    started = time.time()

    # -- gain calibration
    cal_idx = np.linspace(0, len(corpus) - 1, min(calibration_samples, len(corpus))).astype(int)
    gains = []
    for i in cal_idx:
        mel = extract_mel_spectrogram(corpus[i].audio, audio_cfg, corpus[i].sample_rate, n_frames)
        drive = normalize_mel_input(mel, audio_cfg.db_floor)
        gains.append(calibrate_input_gain(drive, target_rate, spike_cfg))
    gain = float(np.median(gains))
    if verbose:
        print(f"[ARCH_2] calibrated LIF input gain = {gain:.3f} (target rate {target_rate:.0%})")

    spikes = np.zeros((len(corpus), n_frames, audio_cfg.n_mels, 1), dtype=np.uint8)
    labels = np.zeros(len(corpus), dtype=np.int64)
    forests: List[str] = []
    sources: List[str] = []
    mel_checks = None
    latency_ms = {"mel": [], "spike": []}

    for i, sample in enumerate(corpus):
        t0 = time.perf_counter()
        mel = extract_mel_spectrogram(sample.audio, audio_cfg, sample.sample_rate, n_frames)
        latency_ms["mel"].append((time.perf_counter() - t0) * 1000.0)
        if mel_checks is None:
            mel_checks = validate_mel_spectrogram(mel, audio_cfg)

        t0 = time.perf_counter()
        spikes[i] = convert_mel_to_spikes(
            mel, spike_cfg, db_floor=audio_cfg.db_floor, gain=gain
        )["spikes"]
        latency_ms["spike"].append((time.perf_counter() - t0) * 1000.0)

        labels[i] = sample.label
        forests.append(sample.forest)
        sources.append(sample.source)

        if verbose and (i + 1) % 50 == 0:
            print(f"[ARCH_1+2] {i + 1}/{len(corpus)} samples encoded")

    rate = float(spikes.mean())
    if verbose:
        print(
            f"[ARCH_1+2] done in {time.time() - started:.1f}s -- shape {spikes.shape}, "
            f"firing rate {rate:.1%}"
        )

    return {
        "spikes": spikes,
        "labels": labels,
        "forests": np.array(forests),
        "sources": np.array(sources),
        "gain": gain,
        "firing_rate": rate,
        "mel_validation": mel_checks,
        "spike_validation": validate_spikes(spikes[0]),
        "latency_ms": {
            "mel_mean": float(np.mean(latency_ms["mel"])),
            "mel_p95": float(np.percentile(latency_ms["mel"], 95)),
            "spike_mean": float(np.mean(latency_ms["spike"])),
            "spike_p95": float(np.percentile(latency_ms["spike"], 95)),
        },
        "seconds": time.time() - started,
    }


# ---------------------------------------------------------------------------
# Stage 1: dataset
# ---------------------------------------------------------------------------


def run_dataset_stage(
    preset: PipelinePreset = QUICK,
    out_dir: Path = Path("artifacts"),
    base_cfg: Optional[EcoSentryConfig] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """Synthesize the corpus, encode it and write ``prepared_dataset.h5``."""
    audio_cfg, dataset_cfg, _, _ = _configs(preset, base_cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n=== Stage 1: dataset ({preset.name}) ===")

    corpus = build_corpus(
        dataset_cfg, preset.duration_s, audio_cfg.sample_rate, seed, preset.n_samples
    )
    if verbose:
        print(f"[corpus] {len(corpus)} clips of {preset.duration_s}s")

    encoded = build_spike_dataset(
        corpus,
        audio_cfg,
        (base_cfg or EcoSentryConfig()).spikes,
        preset.n_frames,
        dataset_cfg.target_firing_rate,
        verbose=verbose,
    )

    # The trained head covers 3 classes; ambient clips are kept aside as
    # negatives for the false-alert measurement in the alert stage.
    keep = np.isin(encoded["labels"], TRAIN_CLASSES)
    ambient_idx = np.flatnonzero(~keep)

    prepared = a3.prepare_dataset(
        encoded["spikes"][keep],
        encoded["labels"][keep],
        encoded["forests"][keep],
        dataset_cfg,
        verbose=verbose,
    )
    prepared["metadata"]["spike_gain"] = encoded["gain"]
    prepared["metadata"]["preset"] = preset.name
    prepared["metadata"]["duration_s"] = preset.duration_s

    dataset_path = a3.save_prepared_dataset(prepared, out_dir / "prepared_dataset.h5")

    ambient_path = None
    if ambient_idx.size:
        ambient_path = out_dir / "ambient_negatives.npz"
        np.savez_compressed(
            ambient_path,
            spikes=encoded["spikes"][ambient_idx],
            forests=encoded["forests"][ambient_idx].astype("S16"),
        )

    qa = {
        "mel": encoded["mel_validation"],
        "spikes_raw": encoded["spike_validation"],
        # The 25% +/- 5% criterion applies to the normalised data.  The
        # validation split is normalised but NOT augmented, so it is the right
        # thing to measure -- mixup blends two binary tensors into fractional
        # values, which inflates the "any activity" count in the train split.
        "spikes_normalized": a3.validate_normalized_spikes(
            prepared["val"]["spikes"], dataset_cfg
        ),
        "train_augmented_firing_rate": float((prepared["train"]["spikes"] > 0).mean()),
        "class_balance": a3.verify_class_balance(prepared),
    }

    if verbose:
        print(f"[ARCH_3] wrote {dataset_path}")
        print(f"[QA] mel passed={qa['mel']['passed']}  spikes passed={qa['spikes_raw']['passed']}")

    return {
        "dataset_path": str(dataset_path),
        "ambient_path": str(ambient_path) if ambient_path else None,
        "n_corpus": len(corpus),
        "n_train": int(len(prepared["train"]["labels"])),
        "n_val": int(len(prepared["val"]["labels"])),
        "n_test": int(len(prepared["test"]["labels"])),
        "spike_gain": encoded["gain"],
        "firing_rate": encoded["firing_rate"],
        "latency_ms": encoded["latency_ms"],
        "qa": qa,
        "seconds": encoded["seconds"],
        "_prepared": prepared,
    }


# ---------------------------------------------------------------------------
# Stage 2: training
# ---------------------------------------------------------------------------


def run_training_stage(
    dataset: Optional[Dict] = None,
    dataset_path: Optional[Path] = None,
    preset: PipelinePreset = QUICK,
    out_dir: Path = Path("artifacts"),
    base_cfg: Optional[EcoSentryConfig] = None,
    spike_gain: Optional[float] = None,
    verbose: bool = True,
) -> Dict:
    """ARCH_4 training + held-out test evaluation."""
    _, _, snn_cfg, _ = _configs(preset, base_cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if dataset is None:
        dataset = a3.load_prepared_dataset(dataset_path or out_dir / "prepared_dataset.h5")

    if verbose:
        print(f"\n=== Stage 2: SNN training ({preset.name}) ===")
        print(
            f"[ARCH_4] train={len(dataset['train']['labels'])} "
            f"val={len(dataset['val']['labels'])} test={len(dataset['test']['labels'])} "
            f"T={dataset['train']['spikes'].shape[1]} epochs={snn_cfg.epochs}"
        )

    model, history = train_snn(
        dataset["train"]["spikes"],
        dataset["train"]["labels"],
        dataset["val"]["spikes"],
        dataset["val"]["labels"],
        snn_cfg,
        verbose=verbose,
    )

    test = evaluate(model, dataset["test"]["spikes"], dataset["test"]["labels"], snn_cfg.batch_size)
    val = evaluate(model, dataset["val"]["spikes"], dataset["val"]["labels"], snn_cfg.batch_size)

    meta = dataset.get("metadata", {})
    gain = spike_gain if spike_gain is not None else meta.get("spike_gain")
    model.metadata.update(
        {
            "spike_gain": float(gain) if gain is not None else None,
            "n_frames": int(dataset["train"]["spikes"].shape[1]),
            "test_accuracy": test["accuracy"],
            "val_accuracy": val["accuracy"],
            "preset": preset.name,
        }
    )
    model_path = model.save(out_dir / "snn_model.npz")

    if verbose:
        print(
            f"[ARCH_4] best epoch {history.best_epoch + 1}, "
            f"val acc {val['accuracy']:.1%}, test acc {test['accuracy']:.1%} "
            f"({history.seconds:.1f}s)"
        )
        print("[ARCH_4] confusion matrix (rows=true, cols=pred):")
        for i, row in enumerate(test["confusion"]):
            print(f"    {CLASS_NAMES[i]:>9}: {row.tolist()}")

    return {
        "model_path": str(model_path),
        "model": model,
        "history": history.to_dict(),
        "test": {
            "accuracy": test["accuracy"],
            "loss": test["loss"],
            "n": test["n"],
            "confusion": test["confusion"].tolist(),
            "per_class": test["per_class"],
        },
        "val_accuracy": val["accuracy"],
        "meets_accuracy_target": bool(test["accuracy"] >= 0.85),
        "n_parameters": model.n_parameters,
        "seconds": history.seconds,
    }


# ---------------------------------------------------------------------------
# Stage 3: alert path (ARCH_5 -> ARCH_6 -> ARCH_8 -> gateway)
# ---------------------------------------------------------------------------


def run_alert_stage(
    model: SpikingNetwork,
    dataset: Dict,
    preset: PipelinePreset = QUICK,
    out_dir: Path = Path("artifacts"),
    base_cfg: Optional[EcoSentryConfig] = None,
    scenario_key: str = "corbett",
    max_alerts: int = 40,
    ambient_path: Optional[Path] = None,
    verbose: bool = True,
) -> Dict:
    """Drive real test samples through inference, payload, mesh and gateway."""
    base_cfg = base_cfg or EcoSentryConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n=== Stage 3: alert path ({scenario_key}) ===")

    scenario = SCENARIOS[scenario_key]
    engine = EcoSentryInference(
        model,
        base_cfg.audio,
        base_cfg.spikes,
        base_cfg.inference,
        spike_gain=(model.metadata or {}).get("spike_gain"),
        n_frames=preset.n_frames,
    )

    device = DeviceConfig(
        device_id=f"SENTRY_{scenario_key.upper()[:3]}_01",
        latitude=scenario.latitude,
        longitude=scenario.longitude,
    )
    network = create_topology(scenario_key, base_cfg.network)
    phy = LoRaPHY(base_cfg.network)
    gw = PriorityGateway(
        device.encryption_key,
        base_cfg.gateway,
        BackhaulLink(base_cfg.gateway, seed=7),
        MqttSink(),
    )
    rng = np.random.default_rng(11)

    test_spikes = dataset["test"]["spikes"]
    test_labels = np.asarray(dataset["test"]["labels"])
    n = min(max_alerts, len(test_labels))

    events: List[Dict] = []
    e2e_latencies: List[float] = []
    payload_sizes: List[int] = []
    delivered = 0
    alerts = 0
    correct_alerts = 0

    for i in range(n):
        engine.reset()  # each sample is an independent 10 s window
        result = engine.process_spikes(
            test_spikes[i], timestamp=time.time() + i * 60.0, use_temporal_filter=False
        )
        record = {
            "index": i,
            "true_class": CLASS_NAMES[int(test_labels[i])],
            "predicted": result["class_name"],
            "confidence": round(result["confidence"], 4),
            "alert": result["alert"],
            "inference_ms": round(result["latency_ms"], 2),
        }

        if result["alert"]:
            alerts += 1
            if int(test_labels[i]) == result["class_id"]:
                correct_alerts += 1

            packet = generate_alert_payload(result, device, priority=None)
            payload_sizes.append(packet["sizes"]["wire_bytes"])

            source = str(rng.choice(network.sensors()))
            tx = simulate_message_delivery(
                network,
                source,
                packet["sizes"]["wire_bytes"],
                cfg=base_cfg.network,
                rng=rng,
            )

            record.update(
                {
                    "payload_bytes": packet["sizes"]["wire_bytes"],
                    "payload_ms": round(packet["timings_ms"]["total"], 3),
                    "priority": packet["priority"],
                    "lora_source": source,
                    "lora_hops": tx["hops"],
                    "lora_delivered": tx["delivered"],
                    "lora_latency_ms": round(tx["latency_ms"], 1) if tx["delivered"] else None,
                }
            )

            if tx["delivered"]:
                uplink = gw.handle_uplink(packet["message"], source)
                record["gateway_topic"] = (
                    uplink["publish"].topic if uplink.get("publish") else None
                )
                record["backhaul_ms"] = (
                    round(uplink["publish"].latency_ms, 1) if uplink.get("publish") else None
                )
                if uplink.get("publish") and uplink["publish"].delivered:
                    delivered += 1
                    e2e = (
                        result["latency_ms"]
                        + packet["timings_ms"]["total"]
                        + tx["latency_ms"]
                        + uplink["publish"].latency_ms
                    )
                    e2e_latencies.append(e2e)
                    record["end_to_end_ms"] = round(e2e, 1)

                # Verify the command centre can decrypt what it received.
                decoded = parse_alert_message(packet["message"], device.encryption_key)
                record["decrypt_ok"] = (
                    decoded.class_id == result["class_id"]
                    and abs(decoded.confidence - round(result["confidence"], 2)) < 0.02
                )

        events.append(record)

    # Ambient false-alert rate: clips with no threat present at all.
    ambient_alerts = 0
    ambient_n = 0
    if ambient_path and Path(ambient_path).exists():
        blob = np.load(Path(ambient_path))
        ambient_spikes = blob["spikes"]
        ambient_n = min(len(ambient_spikes), 40)
        for i in range(ambient_n):
            engine.reset()
            res = engine.process_spikes(ambient_spikes[i], use_temporal_filter=False)
            ambient_alerts += int(res["alert"])

    has_latency = bool(e2e_latencies)
    lat = np.array(e2e_latencies) if has_latency else np.array([float("nan")])
    summary = {
        "scenario": scenario.name,
        "samples_processed": n,
        "alerts_triggered": alerts,
        "alert_precision": correct_alerts / alerts if alerts else float("nan"),
        "alerts_delivered": delivered,
        "delivery_rate": delivered / alerts if alerts else float("nan"),
        "payload_bytes_mean": float(np.mean(payload_sizes)) if payload_sizes else None,
        "payload_bytes_max": int(np.max(payload_sizes)) if payload_sizes else None,
        "end_to_end_ms_mean": float(np.mean(lat)) if has_latency else None,
        "end_to_end_ms_p95": float(np.percentile(lat, 95)) if has_latency else None,
        "end_to_end_ms_max": float(np.max(lat)) if has_latency else None,
        "meets_latency_target": (
            bool(np.percentile(lat, 95) <= 1500.0) if has_latency else False
        ),
        "ambient_clips": ambient_n,
        "ambient_false_alerts": ambient_alerts,
        "ambient_false_alert_rate": ambient_alerts / ambient_n if ambient_n else None,
        "time_on_air_ms": {
            sf: round(phy.time_on_air_ms(payload_sizes[0] if payload_sizes else 132, sf), 1)
            for sf in range(7, 13)
        },
        "gateway": gw.metrics(),
        "events": events,
    }

    if verbose:
        print(
            f"[ARCH_5] {alerts}/{n} windows raised an alert "
            f"(precision {summary['alert_precision']:.1%})"
            if alerts
            else f"[ARCH_5] no alerts across {n} windows"
        )
        if payload_sizes:
            print(
                f"[ARCH_6] payload {summary['payload_bytes_mean']:.0f} B mean "
                f"(limit {base_cfg.payload.max_message_bytes} B)"
            )
            print(
                f"[ARCH_8] mesh+backhaul delivered {delivered}/{alerts}, "
                f"end-to-end p95 {summary['end_to_end_ms_p95']:.0f} ms"
            )
        if ambient_n:
            print(
                f"[QA] ambient false-alert rate "
                f"{summary['ambient_false_alert_rate']:.1%} over {ambient_n} clips"
            )

    (out_dir / "alert_events.json").write_text(json.dumps(events, indent=2, default=str))
    return summary


# ---------------------------------------------------------------------------
# Stage 4: simulations (ARCH_7 + ARCH_8)
# ---------------------------------------------------------------------------


def run_simulation_stage(
    preset: PipelinePreset = QUICK,
    out_dir: Path = Path("artifacts"),
    base_cfg: Optional[EcoSentryConfig] = None,
    verbose: bool = True,
) -> Dict:
    """ARCH_7 energy projections and ARCH_8 network reports for all forests."""
    base_cfg = base_cfg or EcoSentryConfig()
    _, _, _, energy_cfg = _configs(preset, base_cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n=== Stage 4: energy + network simulations ===")

    energy = compare_scenarios(energy_cfg, preset.mission_days)
    network = compare_network_scenarios(preset.network_messages, 132, base_cfg.network)

    plot_path = plot_energy_reports(energy, out_dir / "energy_profiles.png")

    if verbose:
        for key, report in energy.items():
            print(
                f"[ARCH_7] {report['scenario']:<24} "
                f"avg {report['average_power_mw']:7.2f} mW | "
                f"net {report['daily_net_wh']:+6.2f} Wh/day | "
                f"solar {report['operational_days_with_solar']}d, "
                f"battery-only {report['operational_days_battery_only']}d"
            )
        for key, report in network.items():
            b = report["adaptive_sf_baseline"]
            print(
                f"[ARCH_8] {key:<13} delivery {b['delivery_rate']:6.1%} | "
                f"p99 {b['latency_p99_ms']:7.0f} ms | hops {b['avg_hop_count']:.1f}"
            )

    serialisable_energy = {
        k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
        for k, v in energy.items()
    }
    (out_dir / "energy_report.json").write_text(
        json.dumps(serialisable_energy, indent=2, default=str)
    )
    (out_dir / "network_report.json").write_text(json.dumps(network, indent=2, default=str))

    return {
        "energy": serialisable_energy,
        "network": network,
        "energy_plot": plot_path,
    }


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------


def run_full_pipeline(
    preset_name: str = "quick",
    out_dir: Path = Path("artifacts"),
    scenario: str = "corbett",
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """Run every stage and write ``artifacts/pipeline_report.json``."""
    preset = PRESETS[preset_name]
    base_cfg = EcoSentryConfig()
    out_dir = Path(out_dir)
    started = time.time()

    if verbose:
        print("=" * 72)
        print(f"Eco-Sentry full pipeline -- preset '{preset.name}'")
        print("=" * 72)

    stage1 = run_dataset_stage(preset, out_dir, base_cfg, seed, verbose)
    stage2 = run_training_stage(
        stage1["_prepared"], None, preset, out_dir, base_cfg, stage1["spike_gain"], verbose
    )
    stage3 = run_alert_stage(
        stage2["model"],
        stage1["_prepared"],
        preset,
        out_dir,
        base_cfg,
        scenario,
        ambient_path=stage1["ambient_path"],
        verbose=verbose,
    )
    stage4 = run_simulation_stage(preset, out_dir, base_cfg, verbose)

    report = {
        "preset": preset.name,
        "scenario": scenario,
        "seed": seed,
        "seconds_total": time.time() - started,
        "dataset": {k: v for k, v in stage1.items() if not k.startswith("_")},
        "training": {k: v for k, v in stage2.items() if k != "model"},
        "alerts": {k: v for k, v in stage3.items() if k != "events"},
        "simulations": stage4,
        "acceptance": acceptance_summary(stage1, stage2, stage3, stage4),
    }
    (out_dir / "pipeline_report.json").write_text(json.dumps(report, indent=2, default=str))

    if verbose:
        print("\n" + "=" * 72)
        print("Acceptance criteria (SUMMARY_HIGH_LEVEL_ARCHITECTURE.md)")
        print("=" * 72)
        for name, item in report["acceptance"].items():
            mark = "PASS" if item["passed"] else "FAIL"
            print(f"  [{mark}] {name:<26} {item['actual']:<38} target {item['target']}")
            if item.get("note"):
                for line in _wrap(item["note"], 66):
                    print(f"         | {line}")
        print(f"\nTotal wall clock: {report['seconds_total']:.1f}s")
        print(f"Artifacts written to: {out_dir.resolve()}")

    return report


def _wrap(text: str, width: int) -> List[str]:
    import textwrap

    return textwrap.wrap(text, width)


def acceptance_summary(stage1: Dict, stage2: Dict, stage3: Dict, stage4: Dict) -> Dict:
    """Check the headline targets from the summary document."""
    energy = stage4["energy"]
    network = stage4["network"]

    worst_delivery = min(
        r["adaptive_sf_baseline"]["delivery_rate"] for r in network.values()
    )
    worst_p99 = max(r["adaptive_sf_baseline"]["latency_p99_ms"] for r in network.values())
    min_endurance = min(r["operational_days_with_solar"] for r in energy.values())
    power = list(energy.values())[0]["power_reduction"]
    reduction = power["reduction_vs_cnn_average_x"]
    normalized_rate = float(
        stage1["qa"]["spikes_normalized"].get("global_firing_rate", stage1["firing_rate"])
    )

    def entry(actual, target, passed, note=None):
        out = {"actual": actual, "target": target, "passed": bool(passed)}
        if note:
            out["note"] = note
        return out

    payload_max = stage3.get("payload_bytes_max")
    return {
        "snn_test_accuracy": entry(
            f"{stage2['test']['accuracy']:.1%}", ">85%", stage2["test"]["accuracy"] >= 0.85
        ),
        "spike_firing_rate": entry(
            f"{normalized_rate:.1%} normalized ({stage1['firing_rate']:.1%} raw from ARCH_2)",
            "25% +/- 5%",
            0.20 <= normalized_rate <= 0.30,
            note=(
                "The rate that reaches ARCH_4 is the ARCH_3-normalised one; the "
                "raw ARCH_2 rate varies per clip and per forest, which is what "
                "forest normalisation exists to remove."
            ),
        ),
        "alert_payload_size": entry(
            f"{payload_max} B" if payload_max else "n/a", "<1000 B",
            payload_max is not None and payload_max < 1000,
        ),
        "end_to_end_latency_p95": entry(
            f"{stage3['end_to_end_ms_p95']:.0f} ms"
            if stage3.get("end_to_end_ms_p95") is not None
            else "n/a (no alert crossed the threshold)",
            "<1500 ms",
            stage3["meets_latency_target"],
        ),
        "network_delivery_rate": entry(
            f"{worst_delivery:.1%} (worst forest)", ">95%", worst_delivery >= 0.95
        ),
        "network_latency_p99": entry(
            f"{worst_p99:.0f} ms (worst forest)", "<1500 ms", worst_p99 <= 1500.0
        ),
        "power_reduction": entry(
            f"{reduction:.0f}x system avg / "
            f"{power['reduction_vs_cnn_active_x']:.0f}x compute path",
            "50-75x",
            50 <= reduction,
            note=(
                "Unreachable by construction: the 50-75x headline in "
                "SUMMARY_HIGH_LEVEL_ARCHITECTURE assumes 0.5 mW quiescent power, "
                "which ARCH_7's own 'CRITICAL CORRECTIONS' table supersedes with "
                "50 mW. At 50 mW quiescent the ceiling is 600/(50+3) ~ 11x -- "
                "ARCH_7's own corrected formula. The SNN compute path itself "
                "still beats the CNN by ~18x."
            ),
        ),
        "operational_endurance": entry(
            f"{min_endurance} days (worst forest)", ">=30 days", min_endurance >= 30
        ),
    }
