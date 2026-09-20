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

from .arch6_beacon import decode_beacon
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


@dataclass
class LLQState:
    """Adaptive low-latency queue for the priority lane (new architecture).

    Priority traffic is never dropped and never queued behind best-effort
    traffic -- that guarantee is unchanged by this class. What "adaptive"
    controls is whether a priority forward happens at the fast, uncontended
    backhaul latency, or at a penalized latency representing the priority
    lane itself being saturated by an unusual burst (several sensors
    triggering near-simultaneously on one real incident, or a storm causing
    a cluster of false positives).

    Two time constants:
      - A short-horizon token bucket (`credits` / `max_credits` /
        `refill_per_s`) absorbs bursts up to `max_credits` messages with no
        penalty, then penalizes further messages until credits refill.
      - `baseline_rate_per_day`, updated by calling `rebase()` with a
        trailing multi-day average alert rate, re-derives `max_credits` and
        `refill_per_s` so a gateway that has been busier than usual gets a
        proportionally bigger burst allowance, and a quiet gateway shrinks
        back down. Clamped to [llq_max_credits_floor, llq_max_credits_ceiling]
        from GatewayConfig.
    """

    max_credits: int = 3
    refill_per_s: float = 3 / (24 * 3600.0)
    credits: float = 3.0
    last_refill_s: float = 0.0
    baseline_rate_per_day: float = 5.0
    overflow_latency_multiplier: float = 2.5
    credits_floor: int = 2
    credits_ceiling: int = 20

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self.last_refill_s)
        self.credits = min(float(self.max_credits), self.credits + elapsed * self.refill_per_s)
        self.last_refill_s = now

    def consume(self, now: float) -> float:
        """Call once per priority forward. Returns the latency multiplier
        to apply to that forward: 1.0 if a credit was available, otherwise
        overflow_latency_multiplier. This never blocks and never drops --
        it only affects simulated latency."""
        self._refill(now)
        if self.credits >= 1.0:
            self.credits -= 1.0
            return 1.0
        return self.overflow_latency_multiplier

    def rebase(self, observed_rate_per_day: float) -> None:
        """Call periodically (e.g. once per simulated day) with the
        trailing multi-day average alert rate. Grows or shrinks max_credits
        and refill_per_s proportionally, clamped to [credits_floor,
        credits_ceiling]."""
        self.baseline_rate_per_day = observed_rate_per_day
        self.max_credits = int(
            np.clip(round(observed_rate_per_day * 0.5), self.credits_floor, self.credits_ceiling)
        )
        self.refill_per_s = self.max_credits / (24 * 3600.0)


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
        self.llq = LLQState(
            credits_floor=self.cfg.llq_max_credits_floor,
            credits_ceiling=self.cfg.llq_max_credits_ceiling,
            overflow_latency_multiplier=self.cfg.llq_overflow_latency_multiplier,
        )
        self.beacon_seen: Dict[int, float] = {}  # sequence_number -> time seen, for orphan detection

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

    def handle_uplink(
        self,
        raw: bytes,
        source_node: str = "S1",
        now: Optional[float] = None,
        kind: str = "payload",
    ) -> Dict:
        """Process one LoRa uplink packet. kind is "payload" (default,
        matches pre-Phase-1 behavior exactly) or "beacon" (new architecture
        -- see handle_beacon_uplink for the beacon-specific path, which this
        method delegates to)."""
        now = time.time() if now is None else now
        if kind == "beacon":
            return self.handle_beacon_uplink(raw, source_node, now)

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

    def handle_beacon_uplink(self, raw: bytes, source_node: str, now: float) -> Dict:
        """Process one 16-byte beacon packet (Phase 1/3, new architecture).

        mac_key is self.key -- the same key material used to decrypt the
        paired full AlertPayload, per arch6_beacon.py's module docstring.
        """
        beacon = decode_beacon(raw, self.key)
        if beacon is None:
            self.parse_failures += 1
            return {"accepted": False, "reason": "beacon_parse_error"}

        self.beacon_seen[beacon.sequence_number] = now
        record = PublishRecord(
            topic=self.cfg.beacon_topic,
            payload=raw,
            qos=self.cfg.beacon_qos,
            dscp=self.cfg.priority_dscp,
            latency_ms=0.0,
            attempts=1,
            delivered=True,
            timestamp=now,
        )
        self.sink.publish(record)
        return {"accepted": True, "beacon": beacon, "publish": record}

    def check_orphaned_beacons(self, now: float) -> List[Dict]:
        """Call periodically. Returns a degraded-alert dict for every beacon
        whose paired full payload has not arrived within
        cfg.beacon_orphan_timeout_s -- something happened, forensic detail
        is missing, but the event must not be silently dropped."""
        orphaned = []
        for sequence, seen_at in list(self.beacon_seen.items()):
            if sequence in self.seen:
                del self.beacon_seen[sequence]  # matched -- not orphaned
                continue
            if now - seen_at >= self.cfg.beacon_orphan_timeout_s:
                orphaned.append(
                    {"sequence": sequence, "beacon_seen_at": seen_at, "degraded": True}
                )
                del self.beacon_seen[sequence]
        return orphaned

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

        llq_multiplier = self.llq.consume(now) if priority else 1.0

        while attempts < max_attempts:
            ok, latency = self.backhaul.send(priority)
            total_latency += latency * llq_multiplier
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
            "llq_max_credits": self.llq.max_credits,
            "llq_current_credits": round(self.llq.credits, 2),
            "llq_baseline_rate_per_day": self.llq.baseline_rate_per_day,
        }
