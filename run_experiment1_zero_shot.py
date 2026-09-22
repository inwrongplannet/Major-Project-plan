"""Experiment 1: zero-shot transfer -- evaluate a synthetic-trained SNN on
real, never-seen ESC-50 audio. No training happens in this script; it only
measures how a model that has only ever heard synthetic audio behaves on
real recordings.

Usage:
    python run_experiment1_zero_shot.py \\
        --model artifacts/snn_model.npz \\
        --esc50-root /path/to/ESC-50 \\
        --out artifacts/real_audio_validation/experiment1_zero_shot.json
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
from esc50_loader import load_esc50_corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("artifacts/snn_model.npz"))
    parser.add_argument("--esc50-root", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                         default=Path("artifacts/real_audio_validation/experiment1_zero_shot.json"))
    parser.add_argument("--n-frames", type=int, default=1000,
                         help="Must match the n_frames the model was trained with -- 1000 for the 'full' preset.")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    base_cfg = EcoSentryConfig()

    print(f"Loading model from {args.model} ...")
    model = SpikingNetwork.load(args.model)
    spike_gain = (model.metadata or {}).get("spike_gain")
    print(f"model.cfg.n_classes={model.cfg.n_classes}  class_names={model.class_names}  spike_gain={spike_gain}")

    print(f"Loading real ESC-50 audio from {args.esc50_root} ...")
    samples, counts = load_esc50_corpus(args.esc50_root)
    print(f"Loaded {len(samples)} real clips: {counts}")

    print("Encoding real audio through the model's own mel/spike pipeline ...")
    spikes_list = []
    for s in samples:
        mel = extract_mel_spectrogram(s.audio, base_cfg.audio, s.sample_rate, args.n_frames)
        sp = convert_mel_to_spikes(
            mel, base_cfg.spikes, db_floor=base_cfg.audio.db_floor, gain=spike_gain
        )["spikes"]
        spikes_list.append(sp)
    spikes_arr = np.stack(spikes_list)
    labels_arr = np.array([s.label for s in samples])
    print(f"spikes_arr shape: {spikes_arr.shape}  overall firing rate: {(spikes_arr > 0).mean():.1%}")

    print("Running argmax classification evaluation ...")
    result = evaluate(model, spikes_arr, labels_arr)

    print("Running alert-decision evaluation (the real system behavior, not just argmax) ...")
    engine = EcoSentryInference(
        model, base_cfg.audio, base_cfg.spikes, base_cfg.inference,
        spike_gain=spike_gain, n_frames=args.n_frames,
    )
    alert_by_true_class: dict = {}
    for i in range(len(samples)):
        engine.reset()
        res = engine.process_spikes(spikes_arr[i], use_temporal_filter=False)
        true_name = CLASS_NAMES[labels_arr[i]]
        alert_by_true_class.setdefault(true_name, {"n": 0, "alerts": 0})
        alert_by_true_class[true_name]["n"] += 1
        alert_by_true_class[true_name]["alerts"] += int(res["alert"])

    for name, d in alert_by_true_class.items():
        d["alert_rate"] = d["alerts"] / d["n"] if d["n"] else None

    report = {
        "experiment": "zero_shot_synthetic_to_real",
        "model_path": str(args.model),
        "esc50_root": str(args.esc50_root),
        "n_real_clips": len(samples),
        "category_counts": counts,
        "classes_covered": sorted(set(CLASS_NAMES[l] for l in labels_arr)),
        "classes_not_covered": sorted(set(CLASS_NAMES) - set(CLASS_NAMES[l] for l in labels_arr)),
        "note_on_gunshot": (
            "ESC-50 has no gun_shot category. gunshot is not evaluated in this "
            "experiment and is absent from classes_covered by design, not by error."
        ),
        "argmax_accuracy": result["accuracy"],
        "argmax_confusion_matrix": result["confusion"].tolist(),
        "argmax_per_class": result.get("per_class"),
        "alert_decision_by_true_class": alert_by_true_class,
    }

    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {args.out}")
    print(f"argmax_accuracy: {report['argmax_accuracy']:.1%}")
    for name, d in alert_by_true_class.items():
        print(f"  {name}: n={d['n']}  alert_rate={d['alert_rate']:.1%}")


if __name__ == "__main__":
    main()
