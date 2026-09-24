"""Runs the combined-corpus retraining experiment across multiple random
seeds and stores per-seed results plus aggregate (mean/std) statistics.

This is the "iterations" deliverable: a single run's accuracy is a point
estimate with unknown variance. Repeating the same experiment at several
seeds and reporting mean +/- std is what turns "67.1% on one run" into a
number a reviewer can trust wasn't a lucky draw.

Results are written incrementally, one seed at a time, so an interrupted
run still leaves every completed seed's result on disk.

Usage:
    python run_iterations_combined_datasets.py \
        --esc50-root /path/to/ESC-50 \
        --us8k-root /path/to/UrbanSound8K \
        --seeds 42 43 44 45 46 \
        --epochs 60 \
        --max-per-us8k-category 60 \
        --out-dir artifacts/real_audio_validation/iterations
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from ecosentry import arch3_dataset as a3
from ecosentry.arch4_training import evaluate
from ecosentry.config import DatasetConfig, EcoSentryConfig
from ecosentry.pipeline import PipelinePreset, build_spike_dataset, run_training_stage
from ecosentry.synth import build_corpus
from esc50_loader import load_esc50_corpus
from urbansound8k_loader import load_urbansound8k_corpus


def run_one_seed(seed: int, esc50_root: Path, us8k_root: Path, epochs: int,
                  n_synthetic: int, max_per_us8k_category: int, n_frames: int,
                  out_dir: Path) -> dict:
    base_cfg = EcoSentryConfig()
    dataset_cfg = DatasetConfig()

    synthetic = build_corpus(dataset_cfg, duration_s=10.0, sr=base_cfg.audio.sample_rate,
                              seed=seed, n_samples=n_synthetic)
    esc50_samples, esc50_counts = load_esc50_corpus(esc50_root)
    us8k_samples, us8k_counts, us8k_skipped = load_urbansound8k_corpus(
        us8k_root, max_per_category=max_per_us8k_category
    )
    combined = list(synthetic) + list(esc50_samples) + list(us8k_samples)

    encoded = build_spike_dataset(combined, base_cfg.audio, base_cfg.spikes, n_frames=n_frames,
                                   target_rate=dataset_cfg.target_firing_rate, verbose=False)
    prepared = a3.prepare_dataset(encoded["spikes"], encoded["labels"], encoded["forests"],
                                   dataset_cfg, verbose=False)
    prepared["metadata"]["spike_gain"] = encoded["gain"]

    preset = PipelinePreset(f"iter_seed{seed}", 10.0, n_frames, None, epochs, 32, 100, 30)
    seed_out_dir = out_dir / f"seed_{seed}"
    stage2 = run_training_stage(prepared, None, preset, seed_out_dir, base_cfg,
                                 spike_gain=encoded["gain"], verbose=False)

    model = stage2["model"]
    test_forests = np.asarray(prepared["forests"]["test"])
    real_mask = test_forests == "mixed"
    n_real = int(real_mask.sum())
    real_result = evaluate(model, prepared["test"]["spikes"][real_mask],
                            prepared["test"]["labels"][real_mask]) if n_real > 0 else None

    result = {
        "seed": seed,
        "n_synthetic": len(synthetic),
        "n_esc50": len(esc50_samples),
        "n_us8k": len(us8k_samples),
        "n_us8k_skipped": len(us8k_skipped),
        "test_accuracy_combined": stage2["test"]["accuracy"],
        "test_accuracy_real_only": real_result["accuracy"] if real_result else None,
        "n_real_in_test": n_real,
        "val_accuracy": stage2["val_accuracy"],
    }
    (seed_out_dir).mkdir(parents=True, exist_ok=True)
    (seed_out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--esc50-root", type=Path, required=True)
    parser.add_argument("--us8k-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--n-synthetic", type=int, default=600)
    parser.add_argument("--max-per-us8k-category", type=int, default=60)
    parser.add_argument("--n-frames", type=int, default=1000)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/real_audio_validation/iterations"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = args.out_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary_data = json.load(f)
        all_results = summary_data.get("per_seed_results", [])
    else:
        all_results = []
        
    completed_seeds = {r["seed"] for r in all_results}
    
    for seed in args.seeds:
        if seed in completed_seeds:
            print(f"=== seed {seed} (already completed, skipping) ===")
            continue
            
        print(f"=== seed {seed} ({len(all_results) + 1}/{len(args.seeds)}) ===")
        result = run_one_seed(seed, args.esc50_root, args.us8k_root, args.epochs,
                               args.n_synthetic, args.max_per_us8k_category, args.n_frames,
                               args.out_dir)
        all_results.append(result)
        print(f"  test_accuracy_real_only: {result['test_accuracy_real_only']}")
        real_accs = [r["test_accuracy_real_only"] for r in all_results if r["test_accuracy_real_only"] is not None]
        summary = {
            "seeds_completed": [r["seed"] for r in all_results],
            "n_seeds_completed": len(all_results),
            "n_seeds_requested": len(args.seeds),
            "per_seed_results": all_results,
            "real_only_accuracy_mean": statistics.mean(real_accs) if real_accs else None,
            "real_only_accuracy_stdev": statistics.stdev(real_accs) if len(real_accs) > 1 else None,
            "real_only_accuracy_min": min(real_accs) if real_accs else None,
            "real_only_accuracy_max": max(real_accs) if real_accs else None,
        }
        (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n=== FINAL SUMMARY ===")
    print(f"seeds: {summary['seeds_completed']}")
    print(f"real_only_accuracy: mean={summary['real_only_accuracy_mean']:.1%}"
          f"  stdev={summary['real_only_accuracy_stdev']:.1%}"
          f"  range=[{summary['real_only_accuracy_min']:.1%}, {summary['real_only_accuracy_max']:.1%}]")


if __name__ == "__main__":
    main()
