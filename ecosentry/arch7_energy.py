"""ARCH_7 -- Virtual energy profiler.

Models battery discharge, solar harvesting and per-operation consumption for the
Eco-Sentry edge node across the three forest scenarios, and projects a 30-day
mission.

The document's "CRITICAL CORRECTIONS" table is authoritative here: quiescent
power is **50 mW** (ARM Cortex-M7 + always-on audio codec), not the 0.5 mW used
in the earlier optimistic projection.  That single number is why the realistic
answer is "solar-sustained or ~10-20 days on battery alone", not "60+ days".
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .config import EnergyConfig, ForestScenario, SCENARIOS

__all__ = [
    "DeviceEnergyModel",
    "battery_voltage_at_soc",
    "solar_power_at_time",
    "solar_profile_24h",
    "seasonal_solar_factor",
    "battery_temperature_derating",
    "simulate_24h_cycle",
    "simulate_mission",
    "generate_energy_report",
    "compare_scenarios",
    "power_reduction_table",
]


# ---------------------------------------------------------------------------
# Device model (ARCH_7 Component 1)
# ---------------------------------------------------------------------------


class DeviceEnergyModel:
    """Per-operation energy accounting for one alert cycle."""

    def __init__(self, cfg: Optional[EnergyConfig] = None):
        self.cfg = cfg or EnergyConfig()

    # -- per-operation energy (mJ) -----------------------------------------

    def energy_audio_mj(self) -> float:
        return self.cfg.power_audio_mw * self.cfg.t_audio_s

    def energy_spike_mj(self) -> float:
        return self.cfg.power_spike_mw * self.cfg.t_spike_s

    def energy_inference_mj(self) -> float:
        return self.cfg.power_inference_mw * self.cfg.t_inference_s

    def energy_encryption_mj(self) -> float:
        return self.cfg.power_encryption_mw * self.cfg.t_encryption_s

    def energy_transmission_mj(self, time_on_air_s: Optional[float] = None) -> float:
        toa = self.cfg.t_transmission_s if time_on_air_s is None else time_on_air_s
        return self.cfg.power_transmission_mw * toa

    def energy_per_detection_mj(self) -> float:
        """Sense + classify, with no transmission (every processed window)."""
        return self.energy_audio_mj() + self.energy_spike_mj() + self.energy_inference_mj()

    def energy_per_alert_mj(self, time_on_air_s: Optional[float] = None) -> float:
        """Full alert cycle: sense, classify, encrypt, transmit."""
        return (
            self.energy_per_detection_mj()
            + self.energy_encryption_mj()
            + self.energy_transmission_mj(time_on_air_s)
        )

    def quiescent_energy_mj(self, seconds: float) -> float:
        return self.cfg.power_quiescent_mw * seconds

    # -- averages -----------------------------------------------------------

    def average_power_mw(
        self, alerts_per_day: float, detections_per_hour: float = 6.0,
        time_on_air_s: Optional[float] = None,
    ) -> float:
        """Mean power draw including duty-cycled processing bursts."""
        day_s = 24 * 3600.0
        detection_mj = self.energy_per_detection_mj() * detections_per_hour * 24
        alert_extra_mj = (
            self.energy_encryption_mj() + self.energy_transmission_mj(time_on_air_s)
        ) * alerts_per_day
        quiescent_mj = self.quiescent_energy_mj(day_s)
        return (quiescent_mj + detection_mj + alert_extra_mj) / day_s

    def breakdown_mj_per_day(
        self,
        alerts_per_day: float,
        detections_per_hour: float = 6.0,
        beacon_transmission_mj: float = 0.0,
    ) -> Dict[str, float]:
        """beacon_transmission_mj defaults to 0.0 so every existing caller
        that doesn't pass it gets the exact pre-Phase-5 numbers (all
        transmission energy attributed to "transmission"). Pass a non-zero
        value (see generate_energy_report below) to split it into
        beacon_transmission and payload_transmission instead."""
        payload_transmission_mj = self.energy_transmission_mj() * alerts_per_day
        result = {
            "quiescent": self.quiescent_energy_mj(24 * 3600.0),
            "audio": self.energy_audio_mj() * detections_per_hour * 24,
            "spike_conversion": self.energy_spike_mj() * detections_per_hour * 24,
            "snn_inference": self.energy_inference_mj() * detections_per_hour * 24,
            "encryption": self.energy_encryption_mj() * alerts_per_day,
        }
        if beacon_transmission_mj > 0.0:
            result["beacon_transmission"] = beacon_transmission_mj * alerts_per_day
            result["payload_transmission"] = payload_transmission_mj
        else:
            result["transmission"] = payload_transmission_mj
        return result



# ---------------------------------------------------------------------------
# Battery model (ARCH_7 Component 2)
# ---------------------------------------------------------------------------


def battery_voltage_at_soc(soc_percent: float) -> float:
    """Piecewise-linear Li-Ion discharge curve (4.2 V full -> 2.8 V empty)."""
    soc = float(np.clip(soc_percent, 0.0, 100.0))
    if soc >= 80:
        return 4.0 + (soc - 80) / 20 * 0.2
    if soc >= 50:
        return 3.7 + (soc - 50) / 30 * 0.3
    if soc >= 20:
        return 3.3 + (soc - 20) / 30 * 0.4
    return 2.8 + (soc / 20) * 0.5


def battery_temperature_derating(temp_celsius: float) -> float:
    """Usable-capacity multiplier vs. temperature (1.0 at 25 C)."""
    if temp_celsius >= 25:
        derate = 1.0 - 0.003 * (temp_celsius - 25)
    else:
        derate = 1.0 - 0.004 * (25 - temp_celsius)
    return float(max(derate, 0.5))


# ---------------------------------------------------------------------------
# Solar model (ARCH_7 Component 3)
# ---------------------------------------------------------------------------


def solar_power_at_time(
    hour_of_day: float, cfg: Optional[EnergyConfig] = None
) -> float:
    """Panel output (W) at a given hour; Gaussian day curve, 0 at night."""
    cfg = cfg or EnergyConfig()
    if hour_of_day < cfg.sunrise_hour or hour_of_day > cfg.sunset_hour:
        return 0.0
    peak_hour = 0.5 * (cfg.sunrise_hour + cfg.sunset_hour)
    power = cfg.solar_peak_w * float(
        np.exp(-(((hour_of_day - peak_hour) / cfg.solar_sigma_h) ** 2))
    )
    return max(power, cfg.solar_min_w)


def solar_profile_24h(cfg: Optional[EnergyConfig] = None) -> np.ndarray:
    cfg = cfg or EnergyConfig()
    return np.array([solar_power_at_time(h, cfg) for h in range(24)], dtype=np.float64)


def seasonal_solar_factor(day_of_year: int) -> float:
    """+/-10% seasonal swing for Indian latitudes (13-29 N)."""
    day_angle = (day_of_year - 1) * 360.0 / 365.0
    return float(1.0 + 0.1 * np.sin(np.radians(day_angle)))


# ---------------------------------------------------------------------------
# 24-hour simulation (ARCH_7 Component 5)
# ---------------------------------------------------------------------------


def simulate_24h_cycle(
    scenario: ForestScenario,
    start_energy_j: Optional[float] = None,
    cfg: Optional[EnergyConfig] = None,
    device: Optional[DeviceEnergyModel] = None,
    weather_factor: float = 1.0,
    seasonal_factor: float = 1.0,
    detections_per_hour: float = 6.0,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """Hour-by-hour energy balance for a single day."""
    cfg = cfg or EnergyConfig()
    device = device or DeviceEnergyModel(cfg)
    rng = rng or np.random.default_rng()

    max_energy_j = cfg.battery_energy_j
    energy_j = max_energy_j if start_energy_j is None else min(start_energy_j, max_energy_j)

    solar = solar_profile_24h(cfg)
    hourly_alert_rate = scenario.avg_alerts_per_day / 24.0

    energy_history: List[float] = []
    soc_history: List[float] = []
    alerts: List[int] = []
    consumption: List[float] = []
    harvest: List[float] = []

    for hour in range(24):
        n_alerts = int(rng.poisson(hourly_alert_rate))

        consumed_j = (
            device.quiescent_energy_mj(3600.0)
            + device.energy_per_detection_mj() * detections_per_hour
            + (device.energy_encryption_mj() + device.energy_transmission_mj()) * n_alerts
        ) / 1000.0

        harvested_j = (
            solar[hour]
            * 3600.0
            * cfg.solar_system_efficiency
            * weather_factor
            * seasonal_factor
            * scenario.solar_derate
        )

        energy_j = float(np.clip(energy_j + harvested_j - consumed_j, 0.0, max_energy_j))

        energy_history.append(energy_j)
        soc_history.append(100.0 * energy_j / max_energy_j)
        alerts.append(n_alerts)
        consumption.append(consumed_j)
        harvest.append(harvested_j)

    return {
        "hours": np.arange(24),
        "energy_j": np.array(energy_history),
        "soc_percent": np.array(soc_history),
        "voltage_v": np.array([battery_voltage_at_soc(s) for s in soc_history]),
        "alerts_per_hour": np.array(alerts),
        "consumption_j": np.array(consumption),
        "harvest_j": np.array(harvest),
        "daily_consumption_wh": float(np.sum(consumption) / 3600.0),
        "daily_harvest_wh": float(np.sum(harvest) / 3600.0),
        "end_energy_j": energy_j,
    }


# ---------------------------------------------------------------------------
# Mission projection (ARCH_7 Component 6)
# ---------------------------------------------------------------------------


def simulate_mission(
    scenario: ForestScenario,
    days: Optional[int] = None,
    cfg: Optional[EnergyConfig] = None,
    random_weather: bool = True,
    solar_enabled: bool = True,
    detections_per_hour: float = 6.0,
    seed: int = 42,
) -> Dict:
    """Multi-day projection with cloud-cover and seasonal variation."""
    cfg = cfg or EnergyConfig()
    days = days or cfg.mission_days
    device = DeviceEnergyModel(cfg)
    rng = np.random.default_rng(seed)

    energy_j = cfg.battery_energy_j
    trajectory: List[float] = []
    soc_curve: List[float] = []
    daily: List[Dict] = []
    operational_days = days

    for day in range(days):
        weather = float(rng.uniform(*cfg.weather_factor_range)) if random_weather else 1.0
        if not solar_enabled:
            weather = 0.0
        season = seasonal_solar_factor(day + 1)

        profile = simulate_24h_cycle(
            scenario,
            start_energy_j=energy_j,
            cfg=cfg,
            device=device,
            weather_factor=weather,
            seasonal_factor=season,
            detections_per_hour=detections_per_hour,
            rng=rng,
        )

        energy_j = profile["end_energy_j"]
        trajectory.append(energy_j)
        soc_curve.append(100.0 * energy_j / cfg.battery_energy_j)
        daily.append(
            {
                "day": day + 1,
                "weather_factor": weather,
                "alerts": int(profile["alerts_per_hour"].sum()),
                "consumption_wh": profile["daily_consumption_wh"],
                "harvest_wh": profile["daily_harvest_wh"],
                "end_soc": soc_curve[-1],
                "soc_hourly": profile["soc_percent"],
            }
        )

        if energy_j <= 0.0 and operational_days == days:
            operational_days = day + 1

    return {
        "scenario": scenario.name,
        "scenario_key": scenario.key,
        "days": days,
        "battery_trajectory_j": np.array(trajectory),
        "soc_trajectory": np.array(soc_curve),
        "daily": daily,
        "operational_days": operational_days,
        "survived": bool(trajectory[-1] > 0) if trajectory else False,
        "solar_enabled": solar_enabled,
    }


# ---------------------------------------------------------------------------
# Reporting (ARCH_7 Component 7-8)
# ---------------------------------------------------------------------------


def power_reduction_table(cfg: Optional[EnergyConfig] = None) -> Dict:
    """SNN vs. always-on CNN baseline (ARCH_7 "Power Reduction Validation")."""
    cfg = cfg or EnergyConfig()
    device = DeviceEnergyModel(cfg)

    rows = {
        "listening": {"cnn_mw": 200.0, "snn_mw": cfg.power_quiescent_mw},
        "audio_processing": {"cnn_mw": 150.0, "snn_mw": cfg.power_audio_mw},
        "inference": {"cnn_mw": 250.0, "snn_mw": cfg.power_inference_mw},
    }
    for row in rows.values():
        row["reduction_x"] = round(row["cnn_mw"] / max(row["snn_mw"], 1e-9), 1)

    # Duty-cycled average: the SNN only runs on triggered windows.
    snn_avg = device.average_power_mw(alerts_per_day=5.0, detections_per_hour=6.0)
    return {
        "rows": rows,
        "cnn_total_mw": cfg.cnn_baseline_mw,
        "snn_active_mw": cfg.power_audio_mw + cfg.power_spike_mw + cfg.power_inference_mw,
        "snn_average_mw": round(snn_avg, 3),
        "reduction_vs_cnn_active_x": round(
            cfg.cnn_baseline_mw
            / max(cfg.power_audio_mw + cfg.power_spike_mw + cfg.power_inference_mw, 1e-9),
            1,
        ),
        "reduction_vs_cnn_average_x": round(cfg.cnn_baseline_mw / max(snn_avg, 1e-9), 1),
    }


def generate_energy_report(
    scenario: ForestScenario,
    cfg: Optional[EnergyConfig] = None,
    days: Optional[int] = None,
    random_weather: bool = True,
    seed: int = 42,
) -> Dict:
    """Complete per-scenario energy analysis with a deployment recommendation."""
    cfg = cfg or EnergyConfig()
    device = DeviceEnergyModel(cfg)

    solar_run = simulate_mission(
        scenario, days, cfg, random_weather, solar_enabled=True, seed=seed
    )
    battery_only = simulate_mission(
        scenario, days or cfg.mission_days, cfg, False, solar_enabled=False, seed=seed
    )

    from .arch8_network import LoRaPHY
    from .config import BeaconConfig, NetworkConfig

    net_cfg = NetworkConfig()
    beacon_cfg = BeaconConfig()
    phy = LoRaPHY(net_cfg)
    beacon_toa_s = phy.time_on_air_ms(beacon_cfg.size_bytes, net_cfg.beacon_spreading_factor) / 1000.0
    beacon_transmission_mj = cfg.power_transmission_mw * beacon_toa_s
    breakdown = device.breakdown_mj_per_day(
        scenario.avg_alerts_per_day, beacon_transmission_mj=beacon_transmission_mj
    )

    daily_consumption_wh = sum(breakdown.values()) / 1000.0 / 3600.0
    daily_harvest_wh = float(
        np.sum(solar_profile_24h(cfg))
        * cfg.solar_system_efficiency
        * scenario.solar_derate
        * np.mean(cfg.weather_factor_range)
    )

    avg_power_mw = device.average_power_mw(scenario.avg_alerts_per_day)
    net_wh = daily_harvest_wh - daily_consumption_wh

    recommendations: List[str] = []
    if net_wh > 0.5:
        recommendations.append(
            "Solar surplus covers consumption -- deploy with the standard 5000 mAh pack."
        )
    elif net_wh > 0:
        recommendations.append(
            "Solar barely covers consumption -- add margin (larger panel or MPPT)."
        )
    else:
        recommendations.append(
            "Daily deficit: add a second battery or a larger panel, or duty-cycle listening."
        )
    if scenario.false_positive_rate > 0.05:
        recommendations.append(
            "High false-positive rate inflates transmission energy -- raise the "
            "confidence threshold or add temporal debouncing."
        )
    if battery_only["operational_days"] < 20:
        recommendations.append(
            f"Battery-only endurance is {battery_only['operational_days']} days; "
            "solar harvesting is mandatory for a 30-day deployment."
        )

    return {
        "scenario": scenario.name,
        "scenario_key": scenario.key,
        "battery_capacity_wh": cfg.battery_energy_wh,
        "average_power_mw": round(avg_power_mw, 3),
        "energy_breakdown_mj_per_day": {k: round(v, 3) for k, v in breakdown.items()},
        "daily_consumption_wh": round(daily_consumption_wh, 3),
        "daily_harvest_wh": round(daily_harvest_wh, 3),
        "daily_net_wh": round(net_wh, 3),
        "energy_per_alert_mj": round(device.energy_per_alert_mj(), 3),
        "energy_per_detection_mj": round(device.energy_per_detection_mj(), 3),
        "operational_days_with_solar": solar_run["operational_days"],
        "operational_days_battery_only": battery_only["operational_days"],
        "final_soc_percent": round(float(solar_run["soc_trajectory"][-1]), 1),
        "min_soc_percent": round(float(solar_run["soc_trajectory"].min()), 1),
        "survived_mission": solar_run["survived"],
        "power_reduction": power_reduction_table(cfg),
        "recommendations": recommendations,
        "soc_trajectory_solar": [round(float(x), 2) for x in solar_run["soc_trajectory"]],
        "soc_trajectory_battery_only": [round(float(x), 2) for x in battery_only["soc_trajectory"]],
        "daily_series": [
            {
                "day": d["day"],
                "harvest_wh": round(float(d["harvest_wh"]), 3),
                "consumption_wh": round(float(d["consumption_wh"]), 3),
                "end_soc": round(float(d["end_soc"]), 1),
            }
            for d in solar_run["daily"]
        ],
        "_solar_run": solar_run,
        "_battery_only": battery_only,
    }


def compare_scenarios(
    cfg: Optional[EnergyConfig] = None, days: Optional[int] = None, seed: int = 42
) -> Dict[str, Dict]:
    """Run :func:`generate_energy_report` for all three forests."""
    return {
        key: generate_energy_report(scenario, cfg, days, seed=seed)
        for key, scenario in SCENARIOS.items()
    }


def plot_energy_reports(reports: Dict[str, Dict], out_path) -> Optional[str]:
    """Battery trajectories + daily balance chart (requires matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return None

    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for key, report in reports.items():
        soc = report["_solar_run"]["soc_trajectory"]
        axes[0].plot(np.arange(1, len(soc) + 1), soc, label=report["scenario"], linewidth=2)
    axes[0].set_xlabel("Mission day")
    axes[0].set_ylabel("State of charge (%)")
    axes[0].set_title("30-day battery trajectory (solar enabled)")
    axes[0].set_ylim(0, 105)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    keys = list(reports)
    x = np.arange(len(keys))
    consumption = [reports[k]["daily_consumption_wh"] for k in keys]
    harvest = [reports[k]["daily_harvest_wh"] for k in keys]
    axes[1].bar(x - 0.18, consumption, width=0.36, label="consumption Wh/day")
    axes[1].bar(x + 0.18, harvest, width=0.36, label="harvest Wh/day")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([reports[k]["scenario"].split()[0] for k in keys])
    axes[1].set_ylabel("Wh per day")
    axes[1].set_title("Daily energy balance")
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return str(out_path)
