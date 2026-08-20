"""ARCH_6 -- Alert payload: JSON -> zlib -> AES-256 -> wire envelope.

Wire format (ARCH_6 Component 4)::

    +--------+---------+----------+------------------------------+
    | magic  | version | sequence | IV (16 B) || AES-256-CBC CT  |
    | 0xEC EA| 0x01    | 0x00-FF  |                              |
    +--------+---------+----------+------------------------------+

The design document mentions both AES-256-CBC (Component 3, with a random
per-message IV) and AES-256-ECB (cross-reference table).  CBC is implemented:
ECB leaks equality between identical plaintexts, which for a fixed-schema alert
payload would let an eavesdropper fingerprint repeated alerts.  The extra 16
bytes of IV keep the message at ~132 B -- still far inside both the 1000 B
budget and the 242 B LoRa limit.

Priority tagging follows PRIORITY_PAYLOAD_DELIVERY.md **Option A**: the flag
lives in the transport header's version byte, so the encrypted payload schema is
byte-for-byte unchanged and old parsers keep working.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .config import PayloadConfig, THREAT_CLASSES

__all__ = [
    "AlertPayload",
    "AlertEncryption",
    "DeviceConfig",
    "generate_encryption_key",
    "derive_key_from_password",
    "location_hash",
    "device_hash",
    "create_message_envelope",
    "parse_message_envelope",
    "generate_alert_payload",
    "parse_alert_message",
    "validate_payload",
    "AlertQueue",
]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass
class DeviceConfig:
    """Per-device identity carried in every alert."""

    device_id: str = "SENTRY_01"
    latitude: float = 29.2452
    longitude: float = 79.1234
    firmware_version: str = "v1.0.0"
    encryption_key: bytes = b""

    def __post_init__(self) -> None:
        if not self.encryption_key:
            self.encryption_key = generate_encryption_key()

    @property
    def location_hash(self) -> str:
        return location_hash(self.latitude, self.longitude)

    @property
    def device_hash(self) -> str:
        return device_hash(self.device_id)


@dataclass
class AlertPayload:
    """Decoded alert, in the receiver-facing representation."""

    timestamp_ms: int = 0
    device_id: str = ""
    class_id: int = 0
    confidence: float = 0.0  # 0..1
    location_hash: str = ""
    firmware_version: str = "v1.0.0"
    sequence_number: int = 0
    priority: bool = False
    checksum: str = ""

    def to_wire_dict(self) -> Dict:
        """Compact, abbreviated field names (ARCH_6 Component 1)."""
        return {
            "t": int(self.timestamp_ms),
            "d": self.device_id,
            "c": int(self.class_id),
            "p": int(round(self.confidence * 100)),  # uint8 percent
            "l": self.location_hash,
            "v": self.firmware_version,
            "s": int(self.sequence_number) % 256,
        }

    @classmethod
    def from_wire_dict(cls, data: Dict, priority: bool = False) -> "AlertPayload":
        return cls(
            timestamp_ms=int(data["t"]),
            device_id=str(data["d"]),
            class_id=int(data["c"]),
            confidence=float(data["p"]) / 100.0,
            location_hash=str(data.get("l", "")),
            firmware_version=str(data.get("v", "")),
            sequence_number=int(data.get("s", 0)),
            priority=priority,
            checksum=str(data.get("x", "")),
        )

    @property
    def is_threat(self) -> bool:
        return self.class_id in THREAT_CLASSES


# ---------------------------------------------------------------------------
# Key management (ARCH_6 Component 7)
# ---------------------------------------------------------------------------


def generate_encryption_key(n_bytes: int = 32) -> bytes:
    """Cryptographically secure AES-256 key."""
    return os.urandom(n_bytes)


def derive_key_from_password(password: str, salt: bytes, iterations: int = 100_000) -> bytes:
    """PBKDF2-HMAC-SHA256 key derivation (ARCH_6 encryption parameters)."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations,
        backend=default_backend(),
    )
    return kdf.derive(password.encode("utf-8"))


def location_hash(latitude: float, longitude: float, n_bytes: int = 4) -> str:
    """SHA-256 of the GPS fix, truncated -- coarse location without exposing it."""
    raw = f"{latitude:.5f},{longitude:.5f}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[: n_bytes * 2]


def device_hash(device_id: str, n_bytes: int = 4) -> str:
    return hashlib.sha256(device_id.encode("utf-8")).hexdigest()[: n_bytes * 2]


# ---------------------------------------------------------------------------
# Encryption
# ---------------------------------------------------------------------------


class AlertEncryption:
    """AES-256-CBC with PKCS7 padding and a fresh random IV per message."""

    def __init__(self, key: bytes):
        if len(key) != 32:
            raise ValueError(f"AES-256 needs a 32-byte key, got {len(key)}")
        self.key = key

    def encrypt_payload(self, plaintext: bytes) -> bytes:
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(self._pad(plaintext)) + encryptor.finalize()
        return iv + ciphertext

    def decrypt_payload(self, blob: bytes) -> bytes:
        if len(blob) < 32 or (len(blob) - 16) % 16:
            raise ValueError("malformed ciphertext")
        iv, ciphertext = blob[:16], blob[16:]
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        return self._unpad(decryptor.update(ciphertext) + decryptor.finalize())

    @staticmethod
    def _pad(data: bytes) -> bytes:
        pad_len = 16 - (len(data) % 16)
        return data + bytes([pad_len] * pad_len)

    @staticmethod
    def _unpad(data: bytes) -> bytes:
        if not data:
            raise ValueError("empty plaintext")
        pad_len = data[-1]
        if pad_len < 1 or pad_len > 16 or data[-pad_len:] != bytes([pad_len] * pad_len):
            raise ValueError("invalid PKCS7 padding (wrong key?)")
        return data[:-pad_len]


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------


def create_message_envelope(
    encrypted: bytes,
    sequence_number: int = 0,
    priority: bool = False,
    cfg: Optional[PayloadConfig] = None,
) -> bytes:
    """magic || version(+priority bit) || sequence || IV||ciphertext."""
    cfg = cfg or PayloadConfig()
    version = cfg.protocol_version | (cfg.priority_header_flag if priority else 0)
    return cfg.magic + bytes([version, sequence_number % 256]) + encrypted


def parse_message_envelope(
    message: bytes, cfg: Optional[PayloadConfig] = None
) -> Tuple[int, bool, bytes]:
    """Returns ``(sequence, priority, encrypted_blob)``; raises on bad framing."""
    cfg = cfg or PayloadConfig()
    if len(message) < 4:
        raise ValueError("message too short")
    if message[:2] != cfg.magic:
        raise ValueError("invalid magic bytes")

    version_byte = message[2]
    priority = bool(version_byte & cfg.priority_header_flag)
    version = version_byte & ~cfg.priority_header_flag
    if version != cfg.protocol_version:
        raise ValueError(f"unsupported protocol version: {version}")

    return message[3], priority, message[4:]


# ---------------------------------------------------------------------------
# End-to-end payload generation / parsing
# ---------------------------------------------------------------------------


def generate_alert_payload(
    inference_result: Dict,
    device: DeviceConfig,
    sequence_number: Optional[int] = None,
    cfg: Optional[PayloadConfig] = None,
    priority: Optional[bool] = None,
    priority_threshold: float = 0.85,
) -> Dict:
    """Inference result -> encrypted wire message.

    Returns ``{"message": bytes, "payload": AlertPayload, "sizes": {...},
    "timings_ms": {...}, "priority": bool}``.
    """
    cfg = cfg or PayloadConfig()

    timestamp = inference_result.get("timestamp", time.time())
    timestamp_ms = int(inference_result.get("timestamp_ms", timestamp * 1000))
    seq = (
        int(inference_result.get("sequence", 0))
        if sequence_number is None
        else int(sequence_number)
    )
    confidence = float(inference_result["confidence"])
    class_id = int(inference_result["class_id"])

    if priority is None:
        priority = class_id in THREAT_CLASSES and confidence >= priority_threshold

    payload = AlertPayload(
        timestamp_ms=timestamp_ms,
        device_id=device.device_hash,
        class_id=class_id,
        confidence=confidence,
        location_hash=device.location_hash,
        firmware_version=device.firmware_version,
        sequence_number=seq,
        priority=bool(priority),
    )

    t0 = time.perf_counter()
    wire = payload.to_wire_dict()
    wire["x"] = format(zlib.crc32(json.dumps(wire, separators=(",", ":")).encode()), "08x")
    payload.checksum = wire["x"]
    plaintext = json.dumps(wire, separators=(",", ":")).encode("utf-8")
    t_json = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    compressed = zlib.compress(plaintext, level=cfg.zlib_level)
    t_zip = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    encrypted = AlertEncryption(device.encryption_key).encrypt_payload(compressed)
    t_aes = (time.perf_counter() - t0) * 1000.0

    message = create_message_envelope(encrypted, seq, bool(priority), cfg)

    if len(message) > cfg.max_message_bytes:
        raise ValueError(f"message too large: {len(message)} > {cfg.max_message_bytes} B")

    return {
        "message": message,
        "payload": payload,
        "priority": bool(priority),
        "sizes": {
            "json_bytes": len(plaintext),
            "compressed_bytes": len(compressed),
            "encrypted_bytes": len(encrypted),
            "wire_bytes": len(message),
        },
        "timings_ms": {
            "serialize": t_json,
            "compress": t_zip,
            "encrypt": t_aes,
            "total": t_json + t_zip + t_aes,
        },
    }


def parse_alert_message(
    message: bytes, key: bytes, cfg: Optional[PayloadConfig] = None
) -> AlertPayload:
    """Decrypt, decompress and validate a received wire message."""
    cfg = cfg or PayloadConfig()
    sequence, priority, encrypted = parse_message_envelope(message, cfg)

    compressed = AlertEncryption(key).decrypt_payload(encrypted)
    plaintext = zlib.decompress(compressed)
    data = json.loads(plaintext.decode("utf-8"))

    expected = data.pop("x", None)
    if expected is not None:
        actual = format(
            zlib.crc32(json.dumps(data, separators=(",", ":")).encode()), "08x"
        )
        if actual != expected:
            raise ValueError("checksum mismatch -- payload corrupted")
        data["x"] = expected

    payload = AlertPayload.from_wire_dict(data, priority=priority)
    payload.sequence_number = sequence
    return payload


def validate_payload(message: bytes, cfg: Optional[PayloadConfig] = None) -> Tuple[bool, Dict]:
    """Cheap structural validation, before attempting decryption."""
    cfg = cfg or PayloadConfig()
    checks = {
        "min_length": len(message) >= 4 + 32,
        "size_ok": len(message) <= cfg.max_message_bytes,
        "magic_bytes": message[:2] == cfg.magic if len(message) >= 2 else False,
        "version": (
            (message[2] & ~cfg.priority_header_flag) == cfg.protocol_version
            if len(message) >= 3
            else False
        ),
        "block_aligned": (len(message) - 4 - 16) % 16 == 0 if len(message) > 20 else False,
        "size_bytes": len(message),
    }
    ok = all(v for k, v in checks.items() if isinstance(v, bool))
    return ok, checks


# ---------------------------------------------------------------------------
# Store-and-forward queue (ARCH_6 "Message Queue")
# ---------------------------------------------------------------------------


@dataclass
class QueuedMessage:
    data: bytes
    timestamp: float
    priority: bool = False
    attempts: int = 0


class AlertQueue:
    """Durable-ish local queue for intermittent connectivity (Sundarbans).

    Priority messages are dequeued first and are the last to be evicted when the
    queue overflows.
    """

    def __init__(self, max_size_mb: float = 10.0, retention_hours: int = 24):
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.retention_s = retention_hours * 3600
        self.queue: List[QueuedMessage] = []
        self.dropped = 0

    @property
    def current_size(self) -> int:
        return sum(len(m.data) for m in self.queue)

    def __len__(self) -> int:
        return len(self.queue)

    def enqueue(self, message: bytes, priority: bool = False, timestamp: Optional[float] = None) -> bool:
        """Append a message.  Returns ``False`` if something had to be dropped."""
        entry = QueuedMessage(message, timestamp or time.time(), priority)
        clean = True

        while self.current_size + len(message) > self.max_size_bytes and self.queue:
            victim = next(
                (i for i, m in enumerate(self.queue) if not m.priority), 0
            )
            self.queue.pop(victim)
            self.dropped += 1
            clean = False

        self.queue.append(entry)
        return clean

    def dequeue_batch(self, max_messages: int = 10) -> List[QueuedMessage]:
        """Pop up to ``max_messages``, priority first then FIFO."""
        order = sorted(
            range(len(self.queue)),
            key=lambda i: (not self.queue[i].priority, self.queue[i].timestamp),
        )[:max_messages]
        batch = [self.queue[i] for i in order]
        for i in sorted(order, reverse=True):
            self.queue.pop(i)
        return batch

    def expire(self, now: Optional[float] = None) -> int:
        """Drop messages older than the retention window. Returns count removed."""
        now = now or time.time()
        before = len(self.queue)
        self.queue = [m for m in self.queue if now - m.timestamp <= self.retention_s]
        return before - len(self.queue)

    def persist(self, path) -> Path:
        """Append-only binary log of the queued messages."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            for m in self.queue:
                f.write(len(m.data).to_bytes(4, "big"))
                f.write(b"\x01" if m.priority else b"\x00")
                f.write(m.data)
        return path

    @classmethod
    def restore(cls, path, **kwargs) -> "AlertQueue":
        queue = cls(**kwargs)
        raw = Path(path).read_bytes()
        offset = 0
        while offset + 5 <= len(raw):
            size = int.from_bytes(raw[offset : offset + 4], "big")
            priority = raw[offset + 4] == 1
            offset += 5
            queue.queue.append(QueuedMessage(raw[offset : offset + size], time.time(), priority))
            offset += size
        return queue
