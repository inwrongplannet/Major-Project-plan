"""Command-line interface.

    python -m ecosentry run     --preset quick        # everything, ~1 minute
    python -m ecosentry dataset --preset full
    python -m ecosentry train   --preset full
    python -m ecosentry infer   --audio clip.wav --model artifacts/snn_model.npz
    python -m ecosentry energy  --scenario sundarbans
    python -m ecosentry network --scenario corbett
    python -m ecosentry synth   --out samples/ --count 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import CLASS_NAMES, SCENARIOS, EcoSentryConfig
from .pipeline import (
    PRESETS,
    run_dataset_stage,
    run_full_pipeline,
    run_training_stage,
)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--preset", choices=sorted(PRESETS), default="quick", help="fidelity preset"
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts"), help="output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ecosentry", description="Eco-Sentry neuromorphic anti-poaching pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the full pipeline (all 8 architectures)")
    _add_common(p_run)
    p_run.add_argument("--scenario", choices=sorted(SCENARIOS), default="corbett")

    p_dataset = sub.add_parser("dataset", help="ARCH_1-3: build the prepared spike dataset")
    _add_common(p_dataset)

    p_train = sub.add_parser("train", help="ARCH_4: train the SNN on a prepared dataset")
    _add_common(p_train)
    p_train.add_argument("--dataset", type=Path, default=None)

    p_infer = sub.add_parser("infer", help="ARCH_1/2/5/6: classify a WAV file")
    p_infer.add_argument("--audio", type=Path, required=True)
    p_infer.add_argument("--model", type=Path, default=Path("artifacts/snn_model.npz"))
    p_infer.add_argument("--payload", action="store_true", help="also build the encrypted alert")

    p_energy = sub.add_parser("energy", help="ARCH_7: energy profile")
    p_energy.add_argument("--scenario", choices=sorted(SCENARIOS) + ["all"], default="all")
    p_energy.add_argument("--days", type=int, default=30)
    p_energy.add_argument("--out", type=Path, default=Path("artifacts"))

    p_net = sub.add_parser("network", help="ARCH_8: LoRa mesh simulation")
    p_net.add_argument("--scenario", choices=sorted(SCENARIOS) + ["all"], default="all")
    p_net.add_argument("--messages", type=int, default=200)
    p_net.add_argument("--payload-bytes", type=int, default=132)

    p_delivery = sub.add_parser(
        "delivery", help="Priority delivery pipeline: gateway + officer ack simulation"
    )
    p_delivery.add_argument("--scenario", choices=sorted(SCENARIOS), default="corbett")
    p_delivery.add_argument("--messages", type=int, default=300)
    p_delivery.add_argument("--seed", type=int, default=42)
    p_delivery.add_argument("--out", type=Path, default=Path("artifacts"))


    p_synth = sub.add_parser("synth", help="write synthetic forest audio to WAV files")
    p_synth.add_argument("--out", type=Path, default=Path("samples"))
    p_synth.add_argument("--count", type=int, default=6)
    p_synth.add_argument("--duration", type=float, default=10.0)
    p_synth.add_argument("--forest", choices=sorted(SCENARIOS) + ["mixed"], default="corbett")

    return parser


def cmd_run(args) -> int:
    run_full_pipeline(args.preset, args.out, args.scenario, args.seed, not args.quiet)
    return 0


def cmd_dataset(args) -> int:
    result = run_dataset_stage(
        PRESETS[args.preset], args.out, EcoSentryConfig(), args.seed, not args.quiet
    )
    print(json.dumps({k: v for k, v in result.items() if not k.startswith("_")}, indent=2, default=str))
    return 0


def cmd_train(args) -> int:
    result = run_training_stage(
        None,
        args.dataset or args.out / "prepared_dataset.h5",
        PRESETS[args.preset],
        args.out,
        EcoSentryConfig(),
        verbose=not args.quiet,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "model"}, indent=2, default=str))
    return 0


def cmd_infer(args) -> int:
    from .arch1_audio import load_audio
    from .arch5_inference import EcoSentryInference
    from .arch6_payload import DeviceConfig, generate_alert_payload

    if not args.model.exists():
        print(f"model not found: {args.model} -- run `python -m ecosentry run` first", file=sys.stderr)
        return 1

    engine = EcoSentryInference.from_checkpoint(args.model)
    engine.n_frames = (engine.model.metadata or {}).get("n_frames")

    audio, sr = load_audio(args.audio, engine.audio_cfg.sample_rate, engine.audio_cfg.max_duration_s)
    result = engine.process_audio(audio, sr, use_temporal_filter=False)

    print(json.dumps(result, indent=2, default=str))

    if args.payload and result["alert"]:
        device = DeviceConfig()
        packet = generate_alert_payload(result, device)
        print(
            json.dumps(
                {
                    "wire_bytes": packet["sizes"]["wire_bytes"],
                    "sizes": packet["sizes"],
                    "timings_ms": packet["timings_ms"],
                    "priority": packet["priority"],
                    "hex_preview": packet["message"][:32].hex(),
                },
                indent=2,
            )
        )
    return 0


def cmd_energy(args) -> int:
    from .arch7_energy import compare_scenarios, generate_energy_report, plot_energy_reports
    from .config import EnergyConfig

    cfg = EnergyConfig()
    if args.scenario == "all":
        reports = compare_scenarios(cfg, args.days)
        plot_energy_reports(reports, Path(args.out) / "energy_profiles.png")
    else:
        reports = {args.scenario: generate_energy_report(SCENARIOS[args.scenario], cfg, args.days)}

    for report in reports.values():
        print(f"\n--- {report['scenario']} ---")
        print(f"  average power        : {report['average_power_mw']:.1f} mW")
        print(f"  daily consumption    : {report['daily_consumption_wh']:.2f} Wh")
        print(f"  daily solar harvest  : {report['daily_harvest_wh']:.2f} Wh")
        print(f"  daily net            : {report['daily_net_wh']:+.2f} Wh")
        print(f"  endurance (solar)    : {report['operational_days_with_solar']} days")
        print(f"  endurance (battery)  : {report['operational_days_battery_only']} days")
        print(f"  final SoC            : {report['final_soc_percent']:.1f}%")
        print(
            f"  power reduction      : "
            f"{report['power_reduction']['reduction_vs_cnn_average_x']:.0f}x vs 600 mW CNN"
        )
        for rec in report["recommendations"]:
            print(f"  * {rec}")
    return 0


def cmd_network(args) -> int:
    from .arch8_network import compare_network_scenarios, generate_network_report

    if args.scenario == "all":
        reports = compare_network_scenarios(args.messages, args.payload_bytes)
    else:
        reports = {
            args.scenario: generate_network_report(
                args.scenario, args.messages, args.payload_bytes
            )
        }

    for key, report in reports.items():
        base = report["adaptive_sf_baseline"]
        print(f"\n--- {key} ---")
        print(f"  topology            : {report['topology']}")
        print(f"  delivery rate       : {base['delivery_rate']:.1%}")
        print(f"  latency mean/p95/p99: "
              f"{base['latency_mean_ms']:.0f} / {base['latency_p95_ms']:.0f} / "
              f"{base['latency_p99_ms']:.0f} ms")
        print(f"  average hops        : {base['avg_hop_count']:.2f}")
        print(f"  congestion delivery : {report['congestion']['delivery_rate']:.1%}")
        print(f"  interference deliv. : {report['interference']['delivery_rate']:.1%}")
        print(f"  time-on-air (ms)    : {report['time_on_air_ms']}")
        for rec in report["recommendations"]:
            print(f"  * {rec}")
    return 0


def cmd_delivery(args) -> int:
    from .pipeline import run_delivery_stage

    report = run_delivery_stage(
        scenario=args.scenario, messages=args.messages, seed=args.seed, out_dir=args.out
    )
    print(json.dumps(report, indent=2, default=str))
    return 0


def cmd_synth(args) -> int:
    from .synth import synthesize, write_wav

    rng = np.random.default_rng(0)
    args.out.mkdir(parents=True, exist_ok=True)
    written = []
    for i in range(args.count):
        label = i % 4
        audio = synthesize(label, args.forest, args.duration, 16_000, rng)
        path = write_wav(
            args.out / f"{args.forest}_{CLASS_NAMES[label]}_{i:02d}.wav", audio, 16_000
        )
        written.append(str(path))
    print("\n".join(written))
    return 0


_COMMANDS = {
    "run": cmd_run,
    "dataset": cmd_dataset,
    "train": cmd_train,
    "infer": cmd_infer,
    "energy": cmd_energy,
    "network": cmd_network,
    "delivery": cmd_delivery,
    "synth": cmd_synth,
}



def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return _COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
