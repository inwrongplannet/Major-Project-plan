"""Tests for the templated dashboard (Phase 0, Task P0.3)."""

from __future__ import annotations

import json

from ecosentry.pipeline import render_dashboard


def test_render_dashboard_embeds_report_json(tmp_path):
    report = {"hello": "world", "n": 3}
    out_path = render_dashboard(report, tmp_path)
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "__REPORT_JSON__" not in content
    assert json.dumps(report) in content


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
