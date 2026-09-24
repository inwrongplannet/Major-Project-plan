"""Zero-shot evaluation of a synthetic-trained model on UrbanSound8K --
specifically the gunshot class, since UrbanSound8K is the first real-audio
source in this project's validation work that has any gunshot data at
all. Mirrors run_experiment1_zero_shot.py's structure exactly.

Usage:
    python run_experiment3_us8k_zero_shot.py \
        --model artifacts/snn_model.npz \
        --us8k-root /path/to/UrbanSound8K \
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
