"""Tests for 4-class ambient rejection fix."""

from __future__ import annotations

import numpy as np

from ecosentry.config import CLASS_NAMES, SNNConfig, EcoSentryConfig
from ecosentry.pipeline import (
    PipelinePreset,
    run_dataset_stage,
)


def test_snn_config_trains_four_classes():
    assert SNNConfig().n_classes == 4
    assert CLASS_NAMES[:4] == ["gunshot", "chainsaw", "vehicle", "ambient"]


def test_dataset_stage_no_longer_holds_out_ambient(tmp_path):
    preset = PipelinePreset("tiny", 2.0, 100, 60, 2, 16, 100, 30)
    result = run_dataset_stage(preset, tmp_path, EcoSentryConfig(), seed=1, verbose=False)
    assert result["ambient_path"] is None


def test_ambient_false_alert_rate_is_measured_end_to_end(tmp_path):
    """Runs stages 1-3 on a tiny preset and asserts the new metric is
    well-formed.  Deliberately does not assert a specific rate here --
    too few epochs/samples on the 'tiny' preset for a stable number --
    only that the measurement pipeline itself works now that ambient
    is trained instead of held out."""
    from ecosentry.pipeline import run_training_stage, run_alert_stage
    from ecosentry.arch3_dataset import load_prepared_dataset

    preset = PipelinePreset("tiny", 2.0, 100, 60, 2, 16, 100, 30)
    cfg = EcoSentryConfig()
    stage1 = run_dataset_stage(preset, tmp_path, cfg, seed=1, verbose=False)
    stage2 = run_training_stage(
        None, stage1["dataset_path"], preset, tmp_path, cfg, verbose=False
    )
    dataset = load_prepared_dataset(stage1["dataset_path"])
    stage3 = run_alert_stage(
        stage2["model"],
        dataset,
        preset,
        tmp_path,
        cfg,
        ambient_path=stage1.get("ambient_path"),
        verbose=False,
    )
    assert 0.0 <= stage3["ambient_false_alert_rate"] <= 1.0
    assert 0.0 <= stage3["ambient_ood_false_alert_rate"] <= 1.0
