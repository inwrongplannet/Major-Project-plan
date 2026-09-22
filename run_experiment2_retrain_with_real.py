"""Experiment 2: retrain including real ESC-50 audio in the training corpus
itself, then re-evaluate on a held-out real-audio test split, to check
whether mixing in real data during training closes the domain gap
Experiment 1 measures.

This trains a brand-new model -- it does not fine-tune the existing
synthetic-only model -- because train_snn (ARCH_4) has no fine-tuning
entry point, only train-from-scratch. That is a real, existing constraint
of the codebase, not a simplification made for this experiment.

Usage:
    python run_experiment2_retrain_with_real.py \\
        --esc50-root /path/to/ESC-50 \\
        --epochs 120 \\
        --out artifacts/real_audio_validation/experiment2_retrain.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecosentry import arch3_dataset as a3
from ecosentry.arch4_training import evaluate
from ecosentry.config import CLASS_NAMES, DatasetConfig, EcoSentryConfig
from ecosentry.pipeline import PipelinePreset, build_spike_dataset, run_training_stage
from ecosentry.synth import build_corpus
from esc50_loader import load_esc50_corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--esc50-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--n-synthetic", type=int, default=600,
                         help="Matches the documented ARCH_3 corpus size for the synthetic half.")
    parser.add_argument("--n-frames", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/real_audio_validation"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = EcoSentryConfig()
    dataset_cfg = DatasetConfig()

    print("Building synthetic half of the corpus ...")
    synthetic_samples = build_corpus(
        dataset_cfg, duration_s=10.0, sr=base_cfg.audio.sample_rate,
        seed=args.seed, n_samples=args.n_synthetic,
    )
    print(f"  {len(synthetic_samples)} synthetic samples")

    print(f"Loading real ESC-50 audio from {args.esc50_root} ...")
    real_samples, counts = load_esc50_corpus(args.esc50_root)
    print(f"  {len(real_samples)} real samples: {counts}")

    combined = list(synthetic_samples) + list(real_samples)
    print(f"Combined corpus: {len(combined)} samples "
          f"({len(synthetic_samples)} synthetic + {len(real_samples)} real)")

    print("Encoding combined corpus (this is the slow step) ...")
    encoded = build_spike_dataset(
        combined, base_cfg.audio, base_cfg.spikes, n_frames=args.n_frames,
        target_rate=dataset_cfg.target_firing_rate, verbose=True,
    )

    print("Running stratified split + augmentation (ARCH_3) ...")
    prepared = a3.prepare_dataset(
        encoded["spikes"], encoded["labels"], encoded["forests"], dataset_cfg, verbose=True
    )
    prepared["metadata"]["spike_gain"] = encoded["gain"]

    preset = PipelinePreset("real_mix", 10.0, args.n_frames, None, args.epochs, 32, 100, 30)
    print(f"Training for {args.epochs} epochs on {len(prepared['train']['labels'])} samples ...")
    stage2 = run_training_stage(prepared, None, preset, args.out_dir, base_cfg,
                                 spike_gain=encoded["gain"], verbose=True)

    model = stage2["model"]
    test_result = evaluate(model, prepared["test"]["spikes"], prepared["test"]["labels"])

    # Only ESC-50-derived rows carry forest=="mixed" -- synthetic rows are
    # always corbett/seshachalam/sundarbans (see synth.py's
    # DatasetConfig.source_forest). This isolates the held-out REAL test
    # rows specifically, which is the number to compare against
    # Experiment 1's zero-shot accuracy.
    test_forests = np.asarray(prepared["forests"]["test"])
    real_test_mask = test_forests == "mixed"
    n_real_in_test = int(real_test_mask.sum())
    if n_real_in_test > 0:
        real_only_result = evaluate(
            model,
            prepared["test"]["spikes"][real_test_mask],
            prepared["test"]["labels"][real_test_mask],
        )
    else:
        real_only_result = None

    report = {
        "experiment": "retrain_with_real_mixed_in",
        "n_synthetic": len(synthetic_samples),
        "n_real": len(real_samples),
        "real_category_counts": counts,
        "epochs": args.epochs,
        "test_accuracy_combined": test_result["accuracy"],
        "test_confusion_matrix": test_result["confusion"].tolist(),
        "test_per_class": test_result.get("per_class"),
        "val_accuracy": stage2["val_accuracy"],
        "n_real_in_test_split": n_real_in_test,
        "test_accuracy_real_only": real_only_result["accuracy"] if real_only_result else None,
        "test_confusion_matrix_real_only": real_only_result["confusion"].tolist() if real_only_result else None,
    }

    out_path = args.out_dir / "experiment2_retrain.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    model.save(args.out_dir / "snn_model_real_mix.npz")
    print(f"\nWrote {out_path}")
    print(f"test_accuracy_combined: {report['test_accuracy_combined']:.1%}")
    if real_only_result:
        print(f"test_accuracy_real_only (n={n_real_in_test}): {report['test_accuracy_real_only']:.1%}"
              f"  <-- compare this to Experiment 1's zero-shot accuracy")


if __name__ == "__main__":
    main()
