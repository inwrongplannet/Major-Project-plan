"""Gateway priority proxy (PRIORITY_PAYLOAD_DELIVERY.md).

Sits between LoRa ingest and the cloud broker.  For alerts that satisfy the
ARCH_5 priority rule (``class_id in {0, 1}`` and ``confidence >= 0.85``) it:

1. sends a local ACK so the device stops retrying (saves airtime and battery),
2. appends the raw packet to a durable local queue,
3. forwards it over the cellular QoS backhaul (DSCP AF41) and publishes to
   ``priority/alerts`` with MQTT QoS 1,
4. retries with exponential backoff on backhaul failure.

Everything else takes the existing best-effort path (``alerts``, QoS 0).
ARCH_1..ARCH_5 are untouched, and the encrypted payload schema is unchanged --
priority lives in the transport header (Option A).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .arch6_payload import AlertPayload, AlertQueue, parse_alert_message, parse_message_envelope
from .config import GatewayConfig, THREAT_CLASSES

__all__ = [
    "PublishRecord",
    "BackhaulLink",
    "MqttSink",
    "PriorityGateway",
]


@dataclass
class PublishRecord:
    topic: str
    payload: bytes
    qos: int
    dscp: Optional[str]
    latency_ms: float
    attempts: int
    delivered: bool
    timestamp: float


class BackhaulLink:
    """Cellular backhaul with a QoS lane and a best-effort lane."""

    def __init__(self, cfg: Optional[GatewayConfig] = None, seed: int = 42):
        self.cfg = cfg or GatewayConfig()
        self.rng = np.random.default_rng(seed)
        self.congestion = 1.0

    def send(self, priority: bool) -> Tuple[bool, float]:
        """Returns ``(delivered, latency_ms)`` for one backhaul attempt."""
        if priority:
            mean = self.cfg.qos_backhaul_ms
            loss = self.cfg.qos_backhaul_loss
            # A QoS/pre-emption profile largely insulates the lane from load.
            jitter = 0.15
            load = 1.0 + 0.1 * (self.congestion - 1.0)
        else:
            mean = self.cfg.best_effort_backhaul_ms
            loss = self.cfg.best_effort_backhaul_loss
            jitter = 0.45
            load = self.congestion

        latency = float(self.rng.normal(mean * load, mean * jitter))
        latency = max(latency, 10.0)
        delivered = self.rng.random() > min(loss * load, 0.95)
        return delivered, latency


class MqttSink:
    """In-memory stand-in for the MQTT broker; records every publish."""

    def __init__(self) -> None:
        self.published: List[PublishRecord] = []

    def publish(self, record: PublishRecord) -> None:
        self.published.append(record)

    def by_topic(self, topic: str) -> List[PublishRecord]:
        return [r for r in self.published if r.topic == topic]


class PriorityGateway:
    """LoRa ingest -> priority classification -> backhaul publish."""

    def __init__(
        self,
        decryption_key: bytes,
        cfg: Optional[GatewayConfig] = None,
        backhaul: Optional[BackhaulLink] = None,
        sink: Optional[MqttSink] = None,
        store: Optional[AlertQueue] = None,
    ):
        self.cfg = cfg or GatewayConfig()
        self.key = decryption_key
        self.backhaul = backhaul or BackhaulLink(self.cfg)
        self.sink = sink or MqttSink()
        self.store = store or AlertQueue(retention_hours=self.cfg.retention_hours)

        self.acks_sent: List[Dict] = []
        self.seen: set = set()  # cloud dedupe: (device_id, sequence, time bucket)
        self.duplicates = 0
        self.parse_failures = 0
        self.stats = {"priority": 0, "normal": 0, "delivered": 0, "dropped": 0}

    # -- rules --------------------------------------------------------------

    def is_priority(self, alert: AlertPayload) -> bool:
        """ARCH_5 semantics, unchanged."""
        return alert.class_id in THREAT_CLASSES and alert.confidence >= self.cfg.confidence_threshold

    def dedupe_key(self, alert: AlertPayload, bucket_s: int = 10) -> Tuple:
        return (
            alert.device_id,
            alert.sequence_number,
            int(alert.timestamp_ms / 1000 / bucket_s),
        )

    # -- ingest -------------------------------------------------------------

    def handle_uplink(self, raw: bytes, source_node: str = "S1", now: Optional[float] = None) -> Dict:
        """Process one LoRa uplink packet."""
        now = time.time() if now is None else now

        try:
            alert = parse_alert_message(raw, self.key)
        except Exception as exc:  # noqa: BLE001 - malformed radio traffic is expected
            self.parse_failures += 1
            return {"accepted": False, "reason": f"parse_error: {exc}"}

        key = self.dedupe_key(alert)
        if key in self.seen:
            self.duplicates += 1
            return {"accepted": False, "reason": "duplicate", "alert": alert}
        self.seen.add(key)

        priority = self.is_priority(alert)
        # The transport header flag set by the device is advisory; the gateway
        # re-derives priority from the decrypted alert so a stripped or spoofed
        # header cannot promote/demote traffic.
        try:
            _, header_priority, _ = parse_message_envelope(raw)
        except Exception:  # noqa: BLE001
            header_priority = False

        if priority:
            self.send_local_ack(source_node, alert, now)
            self.stats["priority"] += 1
            result = self._forward(raw, priority=True, now=now)
        else:
            self.stats["normal"] += 1
            result = self._forward(raw, priority=False, now=now)

        return {
            "accepted": True,
            "alert": alert,
            "priority": priority,
            "header_priority": header_priority,
            "publish": result,
        }

    def send_local_ack(self, node_id: str, alert: AlertPayload, now: float) -> None:
        """Minimal downlink ACK -- priority traffic only (duty-cycle budget)."""
        self.acks_sent.append(
            {"node": node_id, "sequence": alert.sequence_number, "timestamp": now}
        )

    # -- egress -------------------------------------------------------------

    def _forward(self, raw: bytes, priority: bool, now: float) -> PublishRecord:
        topic = self.cfg.priority_topic if priority else self.cfg.normal_topic
        qos = self.cfg.priority_qos if priority else self.cfg.normal_qos
        dscp = self.cfg.priority_dscp if priority else None

        self.store.enqueue(raw, priority=priority, timestamp=now)

        total_latency = 0.0
        attempts = 0
        delivered = False
        max_attempts = len(self.cfg.backoff_schedule_s) if priority else 1

        while attempts < max_attempts:
            ok, latency = self.backhaul.send(priority)
            total_latency += latency
            attempts += 1
            if ok:
                delivered = True
                break
            if attempts < max_attempts:
                total_latency += self.cfg.backoff_schedule_s[attempts - 1] * 1000.0

        record = PublishRecord(
            topic=topic,
            payload=raw,
            qos=qos,
            dscp=dscp,
            latency_ms=total_latency,
            attempts=attempts,
            delivered=delivered,
            timestamp=now,
        )
        self.sink.publish(record)

        if delivered:
            self.stats["delivered"] += 1
            # Message acknowledged downstream; release it from the local store.
            self.store.queue = [m for m in self.store.queue if m.data != raw]
        else:
            self.stats["dropped"] += 1

        return record

    # -- reporting ----------------------------------------------------------

    def metrics(self) -> Dict:
        """ARCH_8 sync metrics from PRIORITY_PAYLOAD_DELIVERY section 'Sync Changes'."""
        priority_records = self.sink.by_topic(self.cfg.priority_topic)
        normal_records = self.sink.by_topic(self.cfg.normal_topic)

        def _summary(records: List[PublishRecord]) -> Dict:
            if not records:
                return {"count": 0, "delivery_rate": 0.0}
            delivered_latencies = [r.latency_ms for r in records if r.delivered]
            lat = np.array(delivered_latencies) if delivered_latencies else np.array([np.nan])
            return {
                "count": len(records),
                "delivery_rate": float(np.mean([r.delivered for r in records])),
                "latency_mean_ms": float(np.nanmean(lat)),
                "latency_p95_ms": float(np.nanpercentile(lat, 95)),
                "latency_p99_ms": float(np.nanpercentile(lat, 99)),
                "mean_attempts": float(np.mean([r.attempts for r in records])),
            }

        # Each local ACK removes the device-side retransmissions it would
        # otherwise have made (LoRaWAN class-A default is 2 unconfirmed retries).
        airtime_saved_tx = len(self.acks_sent) * 2

        return {
            "priority": _summary(priority_records),
            "normal": _summary(normal_records),
            "priority_delivery_rate": _summary(priority_records).get("delivery_rate", 0.0),
            "priority_latency_p95_ms": _summary(priority_records).get("latency_p95_ms"),
            "priority_latency_p99_ms": _summary(priority_records).get("latency_p99_ms"),
            "local_acks": len(self.acks_sent),
            "airtime_saved_transmissions": airtime_saved_tx,
            "duplicates_suppressed": self.duplicates,
            "parse_failures": self.parse_failures,
            "queued": len(self.store),
            "counters": dict(self.stats),
        }
