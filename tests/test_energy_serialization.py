"""Tests for the trajectory-serialization fix (Phase 0, Task P0.1)."""

from __future__ import annotations

import json

from ecosentry.arch7_energy import generate_energy_report
from ecosentry.config import EnergyConfig, SCENARIOS


def test_energy_report_has_trajectory_series():
    cfg = EnergyConfig(mission_days=10)
    report = generate_energy_report(SCENARIOS["corbett"], cfg, days=10, seed=1)
    assert len(report["soc_trajectory_solar"]) == 10
    assert len(report["soc_trajectory_battery_only"]) == 10
    assert len(report["daily_series"]) == 10
    for entry in report["daily_series"]:
        assert set(entry) == {"day", "harvest_wh", "consumption_wh", "end_soc"}


def test_energy_report_public_fields_are_json_serializable():
    cfg = EnergyConfig(mission_days=5)
    report = generate_energy_report(SCENARIOS["sundarbans"], cfg, days=5, seed=1)
    public = {k: v for k, v in report.items() if not k.startswith("_")}
    # Must not raise -- this is exactly what pipeline.py does before writing
    # energy_report.json, so this test catches the same failure mode.
    json.dumps(public)
