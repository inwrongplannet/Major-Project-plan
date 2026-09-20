"""Tests for the 16-byte critical event beacon (Phase 1, arch6_beacon.py)."""

from __future__ import annotations

import pytest

from ecosentry.arch6_beacon import (
    BeaconEncodeError,
    BeaconPayload,
    decode_beacon,
    encode_beacon,
)

KEY = b"x" * 32


def _make(**overrides) -> BeaconPayload:
    defaults = dict(
        device_id=1,
        event_type=0,
        confidence=0.91,
        sequence_number=7,
        relative_timestamp_s=42,
        lat_delta_millideg=1234,
        lon_delta_millideg=-5678,
    )
    defaults.update(overrides)
    return BeaconPayload(**defaults)


def test_encoded_packet_is_exactly_16_bytes():
    packet = encode_beacon(_make(), KEY)
    assert len(packet) == 16


def test_golden_vector_matches_verified_output():
    # This exact hex string was produced by the verified prototype in
    # TASK P1.2's Verification block -- if this ever changes, the wire
    # format itself changed and every downstream consumer must be updated.
    packet = encode_beacon(_make(), KEY)
    assert packet.hex() == "1000013907002a0004d2ffe9d2042d18"


def test_round_trip_preserves_all_fields_within_quantization_tolerance():
    original = _make()
    packet = encode_beacon(original, KEY)
    decoded = decode_beacon(packet, KEY)
    assert decoded is not None
    assert decoded.device_id == original.device_id
    assert decoded.event_type == original.event_type
    assert abs(decoded.confidence - original.confidence) <= (1 / 63)
    assert decoded.sequence_number == original.sequence_number
    assert decoded.relative_timestamp_s == original.relative_timestamp_s
    assert decoded.lat_delta_millideg == original.lat_delta_millideg
    assert decoded.lon_delta_millideg == original.lon_delta_millideg


@pytest.mark.parametrize(
    "overrides",
    [
        dict(device_id=65535, sequence_number=255, relative_timestamp_s=65535,
             lat_delta_millideg=8388607, lon_delta_millideg=-8388608, confidence=1.0),
        dict(device_id=0, sequence_number=0, relative_timestamp_s=0,
             lat_delta_millideg=0, lon_delta_millideg=0, confidence=0.0),
    ],
)
def test_boundary_values_round_trip(overrides):
    original = _make(**overrides)
    packet = encode_beacon(original, KEY)
    assert len(packet) == 16
    decoded = decode_beacon(packet, KEY)
    assert decoded is not None
    assert decoded.device_id == original.device_id
    assert decoded.lat_delta_millideg == original.lat_delta_millideg
    assert decoded.lon_delta_millideg == original.lon_delta_millideg


def test_tampered_packet_is_rejected():
    packet = bytearray(encode_beacon(_make(), KEY))
    packet[4] ^= 0x01  # flip a bit in the sequence_number field
    assert decode_beacon(bytes(packet), KEY) is None


def test_wrong_key_is_rejected():
    packet = encode_beacon(_make(), KEY)
    assert decode_beacon(packet, b"y" * 32) is None


def test_wrong_length_is_rejected():
    packet = encode_beacon(_make(), KEY)
    assert decode_beacon(packet[:15], KEY) is None
    assert decode_beacon(packet + b"\x00", KEY) is None


@pytest.mark.parametrize(
    "overrides",
    [
        dict(device_id=70000),
        dict(event_type=4),
        dict(confidence=1.5),
        dict(sequence_number=256),
        dict(relative_timestamp_s=70000),
        dict(lat_delta_millideg=9_000_000),
        dict(lon_delta_millideg=-9_000_000),
    ],
)
def test_out_of_range_fields_raise(overrides):
    with pytest.raises(BeaconEncodeError):
        encode_beacon(_make(**overrides), KEY)
