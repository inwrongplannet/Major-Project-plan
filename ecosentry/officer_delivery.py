"""New architecture -- reliable ACK-based delivery to the forest officer.

Extends the existing ACK chain (device -> gateway local ACK, gateway ->
cloud backhaul retry) one hop further: cloud -> officer, with a
human-acknowledgement requirement and escalation through a configured list
of channels if nobody acks in time. Nothing in gateway.py or arch6_payload.py
is modified by this module -- it is a new, independent stage that consumes
PriorityGateway's successful deliveries as input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "OfficerDeliveryConfig",
    "OfficerAck",
    "OfficerDeliveryTracker",
]


@dataclass(frozen=True)
class OfficerDeliveryConfig:
    ack_timeout_s: float = 75.0
    escalation_channels: Tuple[str, ...] = ("push", "sms", "radio")


@dataclass(frozen=True)
class OfficerAck:
    alert_sequence: int
    officer_id: str
    acked: bool
    acked_via: Optional[str]
    ack_latency_s: Optional[float]
    exhausted: bool = False


class OfficerDeliveryTracker:
    """Simulates channel delivery and human acknowledgement for one alert,
    escalating through cfg.escalation_channels if no ack arrives within
    cfg.ack_timeout_s on the current channel."""

    def __init__(
        self,
        cfg: Optional[OfficerDeliveryConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.cfg = cfg or OfficerDeliveryConfig()
        self.rng = rng or np.random.default_rng()
        self.pending: Dict[int, Dict] = {}

    def dispatch(self, sequence: int, officer_id: str, now: float) -> None:
        """Call once per alert that the gateway successfully delivered to
        the cloud/backhaul. Starts the officer-side ack clock on the first
        escalation channel."""
        self.pending[sequence] = {
            "officer_id": officer_id,
            "dispatched_at": now,
            "channel_sent_at": now,
            "channel_idx": 0,
            "resolved": False,
        }

    def tick(
        self, sequence: int, now: float, ack_probability_per_channel: float = 0.9
    ) -> Optional[OfficerAck]:
        """Call periodically (e.g. once per simulated second, or driven by a
        Monte-Carlo campaign) to resolve whether this alert has been acked,
        needs escalation, or has exhausted every channel. Returns None while
        the alert is still pending on its current channel; returns an
        OfficerAck exactly once, the moment it resolves (acked or
        exhausted). Returns None immediately for an unknown or
        already-resolved sequence -- never raises."""
        entry = self.pending.get(sequence)
        if entry is None or entry["resolved"]:
            return None

        if self.rng.random() < ack_probability_per_channel:
            entry["resolved"] = True
            channel = self.cfg.escalation_channels[entry["channel_idx"]]
            return OfficerAck(
                alert_sequence=sequence,
                officer_id=entry["officer_id"],
                acked=True,
                acked_via=channel,
                ack_latency_s=now - entry["dispatched_at"],
            )

        elapsed_on_channel = now - entry["channel_sent_at"]
        if elapsed_on_channel >= self.cfg.ack_timeout_s:
            if entry["channel_idx"] < len(self.cfg.escalation_channels) - 1:
                entry["channel_idx"] += 1
                entry["channel_sent_at"] = now
            else:
                entry["resolved"] = True
                return OfficerAck(
                    alert_sequence=sequence,
                    officer_id=entry["officer_id"],
                    acked=False,
                    acked_via=None,
                    ack_latency_s=None,
                    exhausted=True,
                )
        return None
