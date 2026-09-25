# Priority Payload Delivery - Final Synced Plan

Purpose: final, cleaned, and implementation-safe priority delivery design that is fully compatible with existing architectures, especially `ARCH_1` to `ARCH_5` (and payload stage in `ARCH_6`).

## Final Decision (Power-first)
- Keep end devices LoRa-only (no cellular modem on device).
- Add priority handling at gateway/base-station layer.
- Forward priority alerts on a cellular QoS backhaul lane (private APN or operator QoS profile), while preserving existing LoRa path for normal traffic.

This gives the best reliability-to-power tradeoff.

## 3-Line Solution
- Do not change audio or SNN compute path; keep your current edge pipeline as-is.
- Add a gateway priority proxy: local ACK, persistent queue, immediate priority forward over cellular QoS.
- Keep existing JSON payload schema compatible; add priority metadata in transport/header (or optional JSON field) without breaking current parser.

## Compatibility Matrix (Synced)

| Architecture | Current behavior (implemented) | Required change | Compatibility impact |
|---|---|---|---|
| ARCH_1 Audio Processing | DSP to mel-spectrogram pipeline | None | Fully compatible |
| ARCH_2 Spike Conversion | LIF spike encoding | None | Fully compatible |
| ARCH_3 Dataset Prep | Normalization + augmentation | None | Fully compatible |
| ARCH_4 SNN Training | Trained classifier weights | None | Fully compatible |
| ARCH_5 SNN Inference | Real-time class + confidence thresholding | None (only map high-confidence threat alerts to priority lane) | Fully compatible |
| ARCH_6 JSON Payload | Compact encrypted payload | Non-breaking: optional priority metadata or transport header only | Backward compatible |
| ARCH_8 Network Simulator | LoRa mesh + base station path | Add gateway priority proxy + cellular backhaul model | Planned enhancement |

## Exact Sync Rules with ARCH_1 to ARCH_5

To keep your existing implementation stable:
1. No change to sampling rate, mel features, spike encoding, or SNN architecture.
2. No change to class mapping used by inference:
   - `0 = gunshot`
   - `1 = chainsaw`
   - `2 = vehicle`
3. Preserve inference confidence logic from ARCH_5:
   - Default threshold remains `0.85` unless you explicitly tune it.
4. Priority routing trigger should use already-generated inference outputs:
   - `class_id in {0,1}` and `confidence >= threshold`.
5. Existing payload encryption/compression flow in ARCH_6 remains unchanged.

## Non-Breaking Payload Strategy (Important)

Because implementation is already done, use one of these in order:

### Option A (recommended): transport header only
- Add a tiny network header before encrypted payload at gateway ingress only.
- Device payload schema stays unchanged.
- Cloud parser strips header then processes existing payload unchanged.

### Option B: optional JSON field
- Add optional field `"pr":1` in plaintext JSON before compression/encryption.
- Old parser must treat missing field as `pr=0`.
- Keep all existing fields unchanged (`t,d,c,p,l,v,s,x`).

Default recommendation: Option A for minimal application impact.

## Gateway Priority Proxy Design

### Core behavior
- Receive LoRa alert packet from existing path.
- Detect priority condition from inference metadata or optional flag.
- Send local ACK to reduce retries (airtime and battery savings).
- Persist message in local durable queue.
- Forward priority message via cellular backhaul with QoS and DSCP marking.
- Publish to `priority/alerts` topic (MQTT QoS 1 by default).
- Retry on backhaul failure with exponential backoff.

### Pseudocode (final)
```python
class PriorityGateway:
    def __init__(self, lora, mqtt, backhaul, store, threshold=0.85):
        self.lora = lora
        self.mqtt = mqtt
        self.backhaul = backhaul
        self.store = store
        self.threshold = threshold

    def is_priority(self, alert):
        # Keep ARCH_5 semantics intact
        return alert.class_id in (0, 1) and alert.confidence >= self.threshold

    def handle_uplink(self, packet):
        alert = self.parse_existing_payload(packet)  # unchanged payload schema

        if self.is_priority(alert):
            self.send_local_ack(packet.src)
            msg_id = self.store.append(packet.raw)
            try:
                self.forward_priority(packet.raw)
                self.store.mark_sent(msg_id)
            except Exception:
                self.store.schedule_retry(msg_id)
        else:
            self.forward_normal(packet.raw)

    def forward_priority(self, raw_bytes):
        sock = self.backhaul.open_socket()
        sock.set_dscp("AF41")
        self.mqtt.publish("priority/alerts", raw_bytes, qos=1)

    def forward_normal(self, raw_bytes):
        self.mqtt.publish("alerts", raw_bytes, qos=0)
```

## Operator QoS Request Template

Use this directly with operator/private-core team:

- Service: Private APN or managed QoS profile for gateway cellular backhaul
- Traffic type: low-bandwidth critical IoT alerts
- Requirement: priority/pre-emption for priority flow during congestion
- SLA target: priority one-way backhaul latency < 200 ms (p95)
- Reliability target: > 99% priority delivery (gateway to cloud)
- DSCP policy: preserve AF41 (or operator equivalent)
- Security: private APN + VPN/IPsec or private routing to broker

## Default Config Values (Safe Start)
- Priority topic: `priority/alerts`
- Normal topic: `alerts`
- MQTT QoS:
  - priority path: `1`
  - normal path: `0`
- DSCP:
  - priority: `AF41`
  - normal: default best effort
- Gateway retry backoff: `1s,2s,4s,...,60s` max
- Local retention for priority queue: `24h`
- Dedupe key at cloud: `<device_id, sequence_number, timestamp_bucket>`

## ARCH_8 Simulator Sync Changes

Update only network simulation layer; do not touch ARCH_1 to ARCH_5 logic.

1. Replace current `BASE` behavior with `Gateway` behavior:
   - `receive_lora(pkt)`
   - `send_local_ack(node_id)`
   - `forward_to_backhaul(pkt, priority)`
2. Add backhaul model with two modes:
   - `qos_enabled`: lower latency/loss for priority
   - `best_effort`: baseline path
3. Add persistent queue model and retry scheduling.
4. Add metrics:
   - `priority_delivery_rate`
   - `priority_latency_p95`
   - `priority_latency_p99`
   - `airtime_saved`
   - `battery_delta`

## Risks and Mitigations
- Risk: DSCP not honored end-to-end.
  - Mitigation: use operator QoS/APN contract and validate with packet captures and SLA probes.
- Risk: Local ACK may cause duplicate processing if cloud path retries.
  - Mitigation: enforce cloud dedupe using sequence numbers.
- Risk: LoRa downlink duty-cycle limitations.
  - Mitigation: only ACK priority traffic, keep ACK payload minimal.

## Proof Map (Claim -> Source)
- Cellular QoS/pre-emption capability -> 3GPP TS 23.501 / 23.503 / 38.300
- LoRaWAN contention and class behavior -> LoRaWAN specification (LoRa Alliance)
- DSCP depends on domain enforcement -> RFC 2474 / RFC 2475
- MQTT QoS semantics -> OASIS MQTT 5.0 specification
- Deterministic scheduled MAC alternative -> IEEE 802.15.4e / IETF 6TiSCH

## Final Execution Plan

Phase 1 (No-risk integration)
1. Keep existing `ARCH_1` to `ARCH_5` code unchanged.
2. Add gateway priority proxy as a separate layer.
3. Route only threat classes (0/1) over priority topic.

Phase 2 (Network hardening)
1. Provision private APN/QoS with operator.
2. Enable DSCP and validate preservation.
3. Run load tests under simulated congestion.

Phase 3 (Production validation)
1. Track p95/p99 latency and delivery for priority alerts.
2. Verify battery impact remains within budget using retry reduction metrics.
3. Incrementally rollout to all gateways.

## Acceptance Checklist
- [ ] No code changes required in ARCH_1 to ARCH_5 processing path
- [ ] Existing ARCH_6 payload parser remains backward compatible
- [ ] Priority alerts route to `priority/alerts`
- [ ] Local ACK reduces retries in congestion tests
- [ ] Priority delivery SLO met under load
- [ ] Battery consumption unchanged or improved at device side

---
This document is the final cleaned and synced version for priority delivery planning.

## 10 Implementation Deltas (Execution Checklist)

1. Keep `ARCH_1` to `ARCH_5` code unchanged.
2. Add a gateway priority proxy service between LoRa ingest and cloud publish.
3. Implement priority rule using existing inference outputs: `class_id in {0,1}` and `confidence >= 0.85`.
4. Keep `ARCH_6` payload schema unchanged by default; use transport header tagging at gateway.
5. Add local ACK for priority alerts only (to reduce retries and device airtime).
6. Add gateway persistent queue (append-only log) before forwarding priority packets.
7. Forward priority packets over cellular QoS path (private APN/operator QoS), publish to `priority/alerts` with MQTT QoS 1.
8. Keep normal packets on current path/topic (`alerts`, MQTT QoS 0).
9. Add cloud dedupe using `<device_id, sequence_number, timestamp_bucket>` to prevent duplicate processing.
10. Update ARCH_8 simulator with gateway QoS backhaul model and validate targets (`priority_delivery_rate`, p95/p99 latency, airtime_saved, battery_delta).
