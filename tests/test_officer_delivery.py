"""Tests for reliable ACK-based delivery to the forest officer (Phase 4)."""

from __future__ import annotations

import numpy as np

from ecosentry.officer_delivery import (
    OfficerDeliveryConfig,
    OfficerDeliveryTracker,
)


def test_immediate_ack_on_first_channel():
    rng = np.random.default_rng(1)
    tracker = OfficerDeliveryTracker(rng=rng)
    tracker.dispatch(sequence=1, officer_id="ranger_a", now=0.0)
    result = tracker.tick(sequence=1, now=1.0, ack_probability_per_channel=1.0)
    assert result is not None
    assert result.acked is True
    assert result.acked_via == "push"
    assert result.ack_latency_s == 1.0


def test_never_acks_escalates_through_every_channel_then_exhausted():
    rng = np.random.default_rng(2)
    cfg = OfficerDeliveryConfig(ack_timeout_s=5.0)
    tracker = OfficerDeliveryTracker(cfg=cfg, rng=rng)
    tracker.dispatch(sequence=2, officer_id="ranger_b", now=0.0)
    now = 0.0
    result = None
    seen_channels = set()
    for _ in range(200):
        now += 1.0
        entry = tracker.pending[2]
        if not entry["resolved"]:
            seen_channels.add(cfg.escalation_channels[entry["channel_idx"]])
        result = tracker.tick(sequence=2, now=now, ack_probability_per_channel=0.0)
        if result is not None:
            break
    assert result is not None
    assert result.exhausted is True
    assert result.acked is False
    assert seen_channels == {"push", "sms", "radio"}


def test_times_out_on_first_channel_then_acks_on_second():
    rng = np.random.default_rng(3)
    cfg = OfficerDeliveryConfig(ack_timeout_s=5.0)
    tracker = OfficerDeliveryTracker(cfg=cfg, rng=rng)
    tracker.dispatch(sequence=3, officer_id="ranger_c", now=0.0)
    now = 0.0
    result = None
    for _ in range(50):
        now += 1.0
        prob = 0.0 if tracker.pending[3]["channel_idx"] == 0 else 1.0
        result = tracker.tick(sequence=3, now=now, ack_probability_per_channel=prob)
        if result is not None:
            break
    assert result is not None
    assert result.acked is True
    assert result.acked_via == "sms"


def test_unknown_sequence_returns_none():
    tracker = OfficerDeliveryTracker(rng=np.random.default_rng(4))
    assert tracker.tick(sequence=999, now=0.0) is None


def test_resolved_sequence_is_idempotent():
    rng = np.random.default_rng(5)
    tracker = OfficerDeliveryTracker(rng=rng)
    tracker.dispatch(sequence=5, officer_id="ranger_e", now=0.0)
    first = tracker.tick(sequence=5, now=1.0, ack_probability_per_channel=1.0)
    second = tracker.tick(sequence=5, now=2.0, ack_probability_per_channel=1.0)
    assert first is not None and first.acked
    assert second is None
