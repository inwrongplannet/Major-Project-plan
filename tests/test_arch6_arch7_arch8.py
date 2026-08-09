"""ARCH_6 / ARCH_7 / ARCH_8 + gateway tests: payload, energy, mesh, priority."""

from __future__ import annotations

import time

import numpy as np
import pytest

from ecosentry.arch6_payload import (
    AlertEncryption,
    AlertQueue,
    DeviceConfig,
    derive_key_from_password,
    generate_alert_payload,
    generate_encryption_key,
    parse_alert_message,
    parse_message_envelope,
    validate_payload,
)
from ecosentry.arch7_energy import (
    DeviceEnergyModel,
    battery_temperature_derating,
    battery_voltage_at_soc,
    compare_scenarios,
    generate_energy_report,
    power_reduction_table,
    seasonal_solar_factor,
    simulate_24h_cycle,
    simulate_mission,
    solar_power_at_time,
    solar_profile_24h,
)
from ecosentry.arch8_network import (
    LoRaPHY,
    MessageQueue,
    calculate_link_quality,
    connectivity_check,
    create_topology,
    find_route_to_base,
    generate_network_report,
    simulate_alert_delivery_campaign,
    simulate_message_delivery,
    time_on_air_table,
)
from ecosentry.config import EnergyConfig, GatewayConfig, SCENARIOS
from ecosentry.gateway import BackhaulLink, MqttSink, PriorityGateway


@pytest.fixture()
def device() -> DeviceConfig:
    return DeviceConfig(device_id="SENTRY_TEST_01", latitude=29.2452, longitude=79.1234)


@pytest.fixture()
def inference_result() -> dict:
    return {
        "class_id": 1,
        "confidence": 0.89,
        "timestamp": 1_705_329_900.123,
        "timestamp_ms": 1_705_329_900_123,
        "sequence": 42,
    }


# --- ARCH_6 -----------------------------------------------------------------


def test_payload_roundtrip(device, inference_result):
    packet = generate_alert_payload(inference_result, device)
    decoded = parse_alert_message(packet["message"], device.encryption_key)

    assert decoded.class_id == 1
    assert abs(decoded.confidence - 0.89) < 0.01  # uint8 percent quantisation
    assert decoded.timestamp_ms == 1_705_329_900_123
    assert decoded.sequence_number == 42
    assert decoded.location_hash == device.location_hash


def test_payload_fits_lora_and_budget(device, inference_result):
    packet = generate_alert_payload(inference_result, device)
    size = packet["sizes"]["wire_bytes"]
    assert size < 1_000, "exceeds the ARCH_6 message budget"
    assert size <= 242, "exceeds the LoRa maximum payload"


def test_payload_generation_under_8ms(device, inference_result):
    packet = generate_alert_payload(inference_result, device)
    assert packet["timings_ms"]["total"] < 8.0


def test_envelope_framing(device, inference_result):
    packet = generate_alert_payload(inference_result, device)
    message = packet["message"]
    assert message[:2] == b"\xec\xea"
    sequence, priority, blob = parse_message_envelope(message)
    assert sequence == 42
    assert priority is True  # chainsaw at 0.89 >= 0.85
    assert len(blob) % 16 == 0


def test_priority_flag_does_not_change_the_payload_schema(device, inference_result):
    """PRIORITY_PAYLOAD_DELIVERY Option A: the schema must stay byte-compatible."""
    high = generate_alert_payload(inference_result, device, priority=True)
    low = generate_alert_payload(inference_result, device, priority=False)
    assert high["sizes"]["json_bytes"] == low["sizes"]["json_bytes"]

    decoded_high = parse_alert_message(high["message"], device.encryption_key)
    decoded_low = parse_alert_message(low["message"], device.encryption_key)
    assert decoded_high.class_id == decoded_low.class_id
    assert decoded_high.priority and not decoded_low.priority


def test_wrong_key_fails_to_decrypt(device, inference_result):
    packet = generate_alert_payload(inference_result, device)
    with pytest.raises(ValueError):
        parse_alert_message(packet["message"], generate_encryption_key())


def test_tampered_message_rejected(device, inference_result):
    packet = generate_alert_payload(inference_result, device)
    corrupted = bytearray(packet["message"])
    corrupted[20] ^= 0xFF
    with pytest.raises(Exception):
        parse_alert_message(bytes(corrupted), device.encryption_key)


def test_bad_magic_bytes_rejected():
    with pytest.raises(ValueError, match="magic"):
        parse_message_envelope(b"\x00\x00\x01\x00" + b"0" * 48)


def test_aes_requires_256_bit_key():
    with pytest.raises(ValueError):
        AlertEncryption(b"tooshort")


def test_iv_is_unique_per_message():
    enc = AlertEncryption(generate_encryption_key())
    blobs = {enc.encrypt_payload(b"same plaintext")[:16] for _ in range(20)}
    assert len(blobs) == 20, "CBC IV must be random per message"


def test_pbkdf2_is_deterministic():
    salt = b"0123456789abcdef"
    assert derive_key_from_password("hunter2", salt) == derive_key_from_password("hunter2", salt)
    assert derive_key_from_password("hunter2", salt) != derive_key_from_password("other", salt)


def test_validate_payload_structural_checks(device, inference_result):
    packet = generate_alert_payload(inference_result, device)
    ok, checks = validate_payload(packet["message"])
    assert ok and checks["magic_bytes"] and checks["version"]
    assert not validate_payload(b"\x00\x01")[0]


def test_alert_queue_prefers_priority_and_evicts_normal():
    queue = AlertQueue(max_size_mb=1_000 / 1024 / 1024)  # ~1000 bytes
    for i in range(6):
        queue.enqueue(b"n" * 200, priority=False, timestamp=float(i))
    queue.enqueue(b"p" * 200, priority=True, timestamp=99.0)

    batch = queue.dequeue_batch(1)
    assert batch[0].priority, "priority messages must be dequeued first"
    assert queue.dropped > 0


def test_alert_queue_persistence(tmp_path):
    queue = AlertQueue()
    queue.enqueue(b"alpha", priority=True)
    queue.enqueue(b"beta", priority=False)
    path = queue.persist(tmp_path / "queue.log")

    restored = AlertQueue.restore(path)
    assert [m.data for m in restored.queue] == [b"alpha", b"beta"]
    assert restored.queue[0].priority


def test_alert_queue_expiry():
    queue = AlertQueue(retention_hours=1)
    queue.enqueue(b"old", timestamp=time.time() - 7_200)
    queue.enqueue(b"new", timestamp=time.time())
    assert queue.expire() == 1
    assert len(queue) == 1


# --- ARCH_7 -----------------------------------------------------------------


def test_battery_energy_matches_spec():
    cfg = EnergyConfig()
    assert abs(cfg.battery_energy_wh - 18.5) < 0.01
    assert abs(cfg.battery_energy_j - 66_600) < 100


def test_discharge_curve_is_monotone():
    socs = np.linspace(0, 100, 50)
    voltages = [battery_voltage_at_soc(s) for s in socs]
    assert all(b >= a - 1e-9 for a, b in zip(voltages, voltages[1:]))
    assert abs(battery_voltage_at_soc(100) - 4.2) < 0.01
    assert abs(battery_voltage_at_soc(50) - 3.7) < 0.01


def test_temperature_derating():
    assert battery_temperature_derating(25) == 1.0
    assert battery_temperature_derating(40) < 1.0
    assert battery_temperature_derating(0) < 1.0
    assert battery_temperature_derating(-40) >= 0.5


def test_solar_is_zero_at_night_and_peaks_at_noon():
    assert solar_power_at_time(2.0) == 0.0
    assert solar_power_at_time(23.0) == 0.0
    profile = solar_profile_24h()
    assert profile.argmax() == 12
    assert profile.max() <= EnergyConfig().solar_peak_w + 1e-9


def test_seasonal_factor_within_10_percent():
    factors = [seasonal_solar_factor(d) for d in range(1, 366)]
    assert 0.89 < min(factors) and max(factors) < 1.11


def test_energy_per_alert_is_transmission_dominated():
    model = DeviceEnergyModel()
    breakdown = model.breakdown_mj_per_day(alerts_per_day=5)
    assert model.energy_transmission_mj() > model.energy_inference_mj()
    assert breakdown["quiescent"] > sum(
        v for k, v in breakdown.items() if k != "quiescent"
    ), "50 mW always-on listening should dominate the daily budget"


def test_24h_cycle_shape_and_bounds():
    profile = simulate_24h_cycle(SCENARIOS["corbett"], rng=np.random.default_rng(0))
    assert len(profile["soc_percent"]) == 24
    assert profile["soc_percent"].min() >= 0.0
    assert profile["soc_percent"].max() <= 100.0 + 1e-6


def test_mission_without_solar_drains_faster_than_with_solar():
    with_solar = simulate_mission(SCENARIOS["corbett"], 30, solar_enabled=True, seed=1)
    without = simulate_mission(SCENARIOS["corbett"], 30, solar_enabled=False, seed=1)
    assert without["soc_trajectory"][-1] < with_solar["soc_trajectory"][-1]
    assert without["operational_days"] <= with_solar["operational_days"]


def test_power_reduction_reflects_corrected_quiescent():
    table = power_reduction_table()
    # ARCH_7's own corrected formula: 600 / (50 + 3) ~ 11x, NOT the 50-75x
    # headline that assumed 0.5 mW quiescent.
    assert 8 < table["reduction_vs_cnn_average_x"] < 20
    assert table["reduction_vs_cnn_active_x"] > 15


def test_energy_report_has_recommendations():
    report = generate_energy_report(SCENARIOS["sundarbans"], days=30)
    assert report["recommendations"]
    assert report["daily_consumption_wh"] > 0
    assert 0 <= report["final_soc_percent"] <= 100


def test_all_scenarios_profile():
    reports = compare_scenarios(days=10)
    assert set(reports) == {"corbett", "seshachalam", "sundarbans"}
    # Sundarbans has the highest alert rate and worst solar -> worst net budget.
    assert reports["sundarbans"]["daily_net_wh"] <= reports["corbett"]["daily_net_wh"]


# --- ARCH_8 -----------------------------------------------------------------


def test_time_on_air_increases_with_spreading_factor():
    table = time_on_air_table(116)
    values = [table[sf] for sf in range(7, 13)]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_time_on_air_matches_documented_magnitudes():
    table = time_on_air_table(116)
    # ARCH_8 quotes ~56 ms at SF7 and ~1.5 s at SF12 for a 116-byte payload.
    assert 100 < table[7] < 300
    assert 3_000 < table[12] < 8_000


def test_path_loss_increases_with_distance():
    phy = LoRaPHY()
    near = phy.path_loss_db(1.0, include_fading=False)
    far = phy.path_loss_db(10.0, include_fading=False)
    assert far > near


def test_link_quality_degrades_with_distance():
    close = calculate_link_quality(0.5, 7)
    distant = calculate_link_quality(30.0, 7)
    assert close > distant


def test_higher_sf_extends_range():
    assert calculate_link_quality(15.0, 12) >= calculate_link_quality(15.0, 7)


@pytest.mark.parametrize("scenario", ["corbett", "seshachalam", "sundarbans"])
def test_every_sensor_reaches_the_base(scenario):
    network = create_topology(scenario)
    check = connectivity_check(network)
    assert check["connected"], f"{scenario} orphans: {check['orphaned_sensors']}"
    assert 1 <= check["max_hops"] <= 4


@pytest.mark.parametrize("scenario", ["corbett", "seshachalam", "sundarbans"])
def test_delivery_and_latency_targets(scenario):
    network = create_topology(scenario)
    stats = simulate_alert_delivery_campaign(network, 200, 132, seed=3)
    assert stats["delivery_rate"] >= 0.95, f"{scenario}: {stats['delivery_rate']:.2%}"
    assert stats["latency_p99_ms"] <= 1_500.0, f"{scenario}: {stats['latency_p99_ms']:.0f} ms"


def test_route_from_base_is_zero_hops():
    network = create_topology("corbett")
    path, hops, distance = find_route_to_base(network, "BASE")
    assert hops == 0 and distance == 0.0


def test_unreachable_source_reported():
    network = create_topology("corbett")
    network.add_device("FAR", 0.0, 0.0, "sensor")
    network.compute_distances()
    result = simulate_message_delivery(network, "FAR", 132, rng=np.random.default_rng(0))
    assert not result["delivered"]
    assert result["reason"] == "unreachable"


def test_message_queue_drops_oldest_normal_first():
    queue = MessageQueue(max_size_bytes=400)
    for i in range(3):
        queue.enqueue(b"x" * 150, timestamp=float(i), priority=False)
    assert queue.dropped >= 1
    assert queue.current_size <= 400


def test_network_report_structure():
    report = generate_network_report("corbett", num_messages=60, message_bytes=132)
    assert report["topology"]["n_sensors"] == 3
    assert report["recommendations"]
    assert report["connectivity"]["connected"]
    assert "adaptive_sf_baseline" in report and "congestion" in report


# --- Gateway priority proxy -------------------------------------------------


def test_gateway_routes_threats_to_priority_topic(device):
    gw = PriorityGateway(device.encryption_key, GatewayConfig(), BackhaulLink(seed=1), MqttSink())
    threat = generate_alert_payload(
        {"class_id": 0, "confidence": 0.93, "timestamp": 1_700_000_000.0, "sequence": 1}, device
    )
    result = gw.handle_uplink(threat["message"], "S1")

    assert result["accepted"] and result["priority"]
    assert result["publish"].topic == "priority/alerts"
    assert result["publish"].qos == 1
    assert result["publish"].dscp == "AF41"
    assert len(gw.acks_sent) == 1, "priority alerts must get a local ACK"


def test_gateway_routes_non_threats_to_best_effort(device):
    gw = PriorityGateway(device.encryption_key, GatewayConfig(), BackhaulLink(seed=1), MqttSink())
    vehicle = generate_alert_payload(
        {"class_id": 2, "confidence": 0.99, "timestamp": 1_700_000_000.0, "sequence": 2}, device
    )
    result = gw.handle_uplink(vehicle["message"], "S1")

    assert not result["priority"]
    assert result["publish"].topic == "alerts"
    assert result["publish"].qos == 0
    assert not gw.acks_sent, "only priority traffic is ACKed (duty-cycle budget)"


def test_gateway_low_confidence_threat_is_not_priority(device):
    gw = PriorityGateway(device.encryption_key, GatewayConfig(), BackhaulLink(seed=1), MqttSink())
    weak = generate_alert_payload(
        {"class_id": 0, "confidence": 0.60, "timestamp": 1_700_000_000.0, "sequence": 3},
        device,
        priority=False,
    )
    assert not gw.handle_uplink(weak["message"], "S1")["priority"]


def test_gateway_deduplicates(device):
    gw = PriorityGateway(device.encryption_key, GatewayConfig(), BackhaulLink(seed=1), MqttSink())
    packet = generate_alert_payload(
        {"class_id": 1, "confidence": 0.91, "timestamp": 1_700_000_000.0, "sequence": 7}, device
    )
    assert gw.handle_uplink(packet["message"], "S1")["accepted"]
    second = gw.handle_uplink(packet["message"], "S1")
    assert not second["accepted"] and second["reason"] == "duplicate"
    assert gw.duplicates == 1


def test_gateway_rejects_garbage(device):
    gw = PriorityGateway(device.encryption_key, GatewayConfig(), BackhaulLink(seed=1), MqttSink())
    assert not gw.handle_uplink(b"not a real packet", "S1")["accepted"]
    assert gw.parse_failures == 1


def test_priority_lane_beats_best_effort(device):
    gw = PriorityGateway(device.encryption_key, GatewayConfig(), BackhaulLink(seed=5), MqttSink())
    for i in range(40):
        threat = generate_alert_payload(
            {"class_id": 0, "confidence": 0.95, "timestamp": 1_700_000_000.0 + i * 60, "sequence": i},
            device,
        )
        gw.handle_uplink(threat["message"], "S1")
        vehicle = generate_alert_payload(
            {
                "class_id": 2,
                "confidence": 0.99,
                "timestamp": 1_700_000_500.0 + i * 60,
                "sequence": 100 + i,
            },
            device,
        )
        gw.handle_uplink(vehicle["message"], "S2")

    metrics = gw.metrics()
    assert metrics["priority"]["delivery_rate"] >= metrics["normal"]["delivery_rate"]
    assert metrics["priority"]["latency_p95_ms"] < metrics["normal"]["latency_p95_ms"]
    assert metrics["airtime_saved_transmissions"] > 0
