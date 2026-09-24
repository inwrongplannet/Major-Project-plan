"""Tests for the templated dashboard (Phase 0, Task P0.3)."""

from __future__ import annotations

import json

from ecosentry.pipeline import render_dashboard


def test_render_dashboard_embeds_report_json(tmp_path):
    # render_dashboard now attaches report["real_audio_validation"]
    # (None, in tmp_path, since no real_audio_validation/ subdirectory
    # exists there) alongside whatever the caller passed in -- so the
    # embedded JSON is the caller's report PLUS that one extra key, not
    # byte-identical to the caller's report on its own. This is a
    # deliberate behavior change (see ecosentry/pipeline.py's
    # render_dashboard docstring), not a regression -- this test checks
    # the original keys survive and the new key is present, rather than
    # requiring exact equality with the input.
    report = {"hello": "world", "n": 3}
    out_path = render_dashboard(report, tmp_path)
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "__REPORT_JSON__" not in content
    assert '"hello": "world"' in content
    assert '"n": 3' in content
    assert '"real_audio_validation": null' in content


def test_render_dashboard_output_differs_for_different_reports(tmp_path):
    # tmp_path already exists (pytest creates it) -- render_dashboard does
    # not create its out_dir argument, only the dashboard.html file inside
    # it, so both calls below reuse the same, already-existing directory.
    p1 = render_dashboard({"n": 1}, tmp_path)
    text1 = p1.read_text(encoding="utf-8")
    p2 = render_dashboard({"n": 2}, tmp_path)
    text2 = p2.read_text(encoding="utf-8")
    assert text1 != text2
    assert '"n": 2' in text2
    assert '"n": 1' not in text2
