"""New architecture -- 16-byte critical event beacon.

Sent as a fast, cheap first signal ahead of the full encrypted AlertPayload
(arch6_payload.py), which still carries the complete forensic record. The
existing wire format spends its entire 16-byte budget on the IV alone
(magic(2B) | version(1B) | sequence(1B) | IV(16B) | ciphertext), so a true
16-byte packet cannot carry a random IV. This format instead authenticates
with a truncated HMAC over the plaintext fields -- it is deliberately not
encrypted (there is no room), and deliberately not the record anything gets
acted on beyond "go look now": the full AlertPayload that follows carries
the complete, encrypted, forensic-grade data.

Wire format (16 bytes total):

    byte 0      : version (upper 4 bits) | reserved (lower 4 bits, always 0)
    bytes 1-2   : device_id            (uint16, big-endian)
    byte 3      : event_type (bits 7-6) | confidence_quantized (bits 5-0)
    byte 4      : sequence_number      (uint8, shared with the paired AlertPayload)
    bytes 5-6   : relative_timestamp_s (uint16, big-endian)
    bytes 7-9   : lat_delta_millideg   (int24, big-endian, signed)
    bytes 10-12 : lon_delta_millideg   (int24, big-endian, signed)
    bytes 13-15 : truncated_mac        (first 3 bytes of HMAC-SHA256 over bytes 0-12)
"""

from __future__ import annotations

import hashlib
import hmac
import struct
from dataclasses import dataclass
from typing import Optional

from .config import BeaconConfig

__all__ = [
    "BeaconPayload",
    "BeaconEncodeError",
    "encode_beacon",
    "decode_beacon",
]


class BeaconEncodeError(ValueError):
    """Raised by encode_beacon when a field is out of range for the wire format."""


@dataclass(frozen=True)
class BeaconPayload:
    device_id: int             # uint16, 0-65535
    event_type: int            # 2-bit field, 0-3 valid on the wire (0=gunshot, 1=chainsaw)
    confidence: float          # 0.0-1.0 inclusive
    sequence_number: int       # uint8, 0-255, wraps -- shared with the paired AlertPayload
    relative_timestamp_s: int  # uint16, 0-65535 seconds since the gateway's last sync beacon
    lat_delta_millideg: int    # signed 24-bit, -8388608..8388607, relative to DeviceConfig home position
    lon_delta_millideg: int    # signed 24-bit, -8388608..8388607, relative to DeviceConfig home position


def _validate(b: BeaconPayload) -> None:
    if not (0 <= b.device_id <= 0xFFFF):
        raise BeaconEncodeError(f"device_id {b.device_id} out of range 0-65535")
    if not (0 <= b.event_type <= 3):
        raise BeaconEncodeError(f"event_type {b.event_type} out of range 0-3")
    if not (0.0 <= b.confidence <= 1.0):
        raise BeaconEncodeError(f"confidence {b.confidence} out of range 0.0-1.0")
    if not (0 <= b.sequence_number <= 0xFF):
        raise BeaconEncodeError(f"sequence_number {b.sequence_number} out of range 0-255")
    if not (0 <= b.relative_timestamp_s <= 0xFFFF):
        raise BeaconEncodeError(
            f"relative_timestamp_s {b.relative_timestamp_s} out of range 0-65535"
        )
    if not (-8_388_608 <= b.lat_delta_millideg <= 8_388_607):
        raise BeaconEncodeError(
            f"lat_delta_millideg {b.lat_delta_millideg} out of range for signed int24"
        )
    if not (-8_388_608 <= b.lon_delta_millideg <= 8_388_607):
        raise BeaconEncodeError(
            f"lon_delta_millideg {b.lon_delta_millideg} out of range for signed int24"
        )


def encode_beacon(
    b: BeaconPayload, mac_key: bytes, cfg: Optional[BeaconConfig] = None
) -> bytes:
    """Encode a BeaconPayload into exactly cfg.size_bytes (16) bytes.

    Raises BeaconEncodeError if any field is out of range. mac_key must be
    bytes -- use the same key material as the paired AlertPayload's
    encryption key (DeviceConfig.encryption_key).
    """
    cfg = cfg or BeaconConfig()
    _validate(b)
    conf_q = min(63, max(0, round(b.confidence * 63)))
    byte0 = (cfg.version << 4) & 0xF0
    byte3 = ((b.event_type & 0x3) << 6) | (conf_q & 0x3F)
    body = struct.pack(
        ">B H B B H",
        byte0,
        b.device_id,
        byte3,
        b.sequence_number,
        b.relative_timestamp_s,
    )
    body += b.lat_delta_millideg.to_bytes(3, "big", signed=True)
    body += b.lon_delta_millideg.to_bytes(3, "big", signed=True)
    assert len(body) == cfg.body_bytes, f"body is {len(body)} bytes, expected {cfg.body_bytes}"
    mac = hmac.new(mac_key, body, hashlib.sha256).digest()[: cfg.mac_bytes]
    packet = body + mac
    assert len(packet) == cfg.size_bytes, f"packet is {len(packet)} bytes, expected {cfg.size_bytes}"
    return packet


def decode_beacon(
    raw: bytes, mac_key: bytes, cfg: Optional[BeaconConfig] = None
) -> Optional[BeaconPayload]:
    """Decode a 16-byte beacon packet.

    Returns None if the length is wrong or the MAC does not match (tampered,
    corrupted, or wrong key) -- never raises for malformed input.
    """
    cfg = cfg or BeaconConfig()
    if len(raw) != cfg.size_bytes:
        return None
    body, mac = raw[: cfg.body_bytes], raw[cfg.body_bytes :]
    expected_mac = hmac.new(mac_key, body, hashlib.sha256).digest()[: cfg.mac_bytes]
    if not hmac.compare_digest(mac, expected_mac):
        return None
    byte0, device_id, byte3, sequence_number, relative_timestamp_s = struct.unpack(
        ">B H B B H", body[:7]
    )
    version = (byte0 & 0xF0) >> 4
    if version != cfg.version:
        return None
    event_type = (byte3 & 0xC0) >> 6
    conf_q = byte3 & 0x3F
    confidence = conf_q / 63.0
    lat_delta = int.from_bytes(body[7:10], "big", signed=True)
    lon_delta = int.from_bytes(body[10:13], "big", signed=True)
    return BeaconPayload(
        device_id=device_id,
        event_type=event_type,
        confidence=confidence,
        sequence_number=sequence_number,
        relative_timestamp_s=relative_timestamp_s,
        lat_delta_millideg=lat_delta,
        lon_delta_millideg=lon_delta,
    )
