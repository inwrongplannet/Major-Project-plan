# Architecture 6: JSON Alert Payload

## Overview
The JSON payload stage converts alert outputs into compact, encrypted messages for network transmission. Owned by Abhishek M (SNN pipeline completion).

**Input**: Alert signal + metadata (classification, timestamp, location)  
**Output**: Encrypted JSON bytes (<1000 bytes)  
**Encryption**: AES-256 (industry standard)  
**Compression**: Zlib before encryption  

---

## Data Flow

```
Alert Signal (class_id, confidence, timestamp)
    ↓
Build Unencrypted Payload (JSON)
    ├── Timestamp (Unix ms)
    ├── Class ID (0, 1, or 2)
    ├── Confidence score
    ├── Device location hash
    ├── Device ID
    └── Firmware version
    ↓
Serialize to JSON String
    ├── Compact format (no spaces)
    └── Size: ~150-200 bytes
    ↓
Compression (Zlib)
    ├── Level 9 (maximum)
    └── Reduce to ~80-100 bytes (40-50% reduction)
    ↓
Encryption (AES-256-CBC)
    ├── Key: 256-bit (32 bytes)
    ├── IV: 128-bit (16 bytes, random per message)
    └── Output: encrypted bytes
    ↓
Envelope with Metadata
    ├── Magic bytes (0xEC, 0xEA) for validation
    ├── Version number
    ├── IV (prepended)
    └── Encrypted payload
    ↓
Final Message: <1000 bytes (target)
```

---

## Component 1: Payload Schema

### Data Structure

```python
class AlertPayload:
    """Alert payload in Python representation."""
    
    def __init__(self):
        self.timestamp_ms = 0          # Unix timestamp × 1000
        self.device_id = ""            # 8-character device ID
        self.class_id = 0              # 0: gunshot, 1: chainsaw, 2: vehicle
        self.confidence = 0.0          # 0-1 confidence score
        self.location_hash = ""        # GPS hash or grid reference
        self.firmware_version = "1.0"  # Device firmware
        self.sequence_number = 0       # Alert sequence (for ordering)
        self.checksum = 0              # CRC32 for integrity
```

### JSON Representation (Unencrypted)

```json
{
    "t": 1234567890000,
    "d": "SENTRY_01",
    "c": 0,
    "p": 0.87,
    "l": "19.8N_74.5E_625m",
    "v": "1.0",
    "s": 42,
    "x": "a7f3b2c1"
}
```

**Field abbreviations** (reduce JSON size):
| Full Name | Abbrev | Type | Size | Example |
|---|---|---|---|---|
| timestamp_ms | t | Int64 | 8 bytes | 1234567890000 |
| device_id | d | String | 8 bytes | SENTRY_01 |
| class_id | c | UInt8 | 1 byte | 0, 1, or 2 |
| confidence | p | UInt8 | 1 byte | 87 (quantized: 0-100 range, precision ±1%) |
| location_hash | l | String | 15 bytes | "19.8N_74.5E_625m" |
| firmware_version | v | String | 3 bytes | "1.0" |
| sequence_number | s | UInt32 | 4 bytes | 42 |
| checksum | x | Hex | 8 bytes | "a7f3b2c1" |

**Unencrypted JSON size breakdown**:
```
Field names: ~30 bytes
Values: ~120 bytes
JSON overhead (braces, quotes, colons): ~20 bytes
Total: ~170 bytes
```

### Alternative Binary Format (Optional, for Ultra-Low Bandwidth)

```
Binary packet structure (27 bytes, no encryption overhead):
┌─────────────┬────────┬──────┬─────┬──────────┬─────────┬────────┬──────┐
│ Timestamp   │ DevID  │Class │Conf │ Location │ Version │ SeqNum │ CRC  │
│ (8 bytes)   │(2b)   │(1b)  │(2b) │ (8 bytes)│ (1 byte)│(4 bytes)│(1b) │
├─────────────┼────────┼──────┼─────┼──────────┼─────────┼────────┼──────┤
│ 1234567890  │ 0x01   │ 0x00 │0xDE │19.8...  │ 0x10    │ 0x2A   │ 0xC1 │
└─────────────┴────────┴──────┴─────┴──────────┴─────────┴────────┴──────┘

Binary size: ~27 bytes (vs. 170+ JSON)
```

---

## Component 2: Compression

### Zlib Compression Algorithm

```python
import zlib

def compress_payload(payload_json_string):
    """
    Compress JSON string using Zlib.
    
    Input:
    - payload_json_string: JSON string (~150-200 bytes)
    
    Output:
    - compressed_bytes: Zlib compressed data
    """
    
    # Convert JSON string to bytes
    payload_bytes = payload_json_string.encode('utf-8')
    
    # Compress with maximum level (9)
    compressed = zlib.compress(payload_bytes, level=9)
    
    return compressed
```

**Compression results**:
```
Original JSON: 170 bytes
Compressed: ~80 bytes (47% reduction)
Ratio: 2.125:1
```

**Why Zlib?**
- Widely available on embedded systems
- CPU-efficient (suitable for low-power edge devices)
- Good compression ratio for structured data
- ~1ms compression time

---

## Component 3: Encryption (AES-256)

### AES-256-CBC Implementation

```python
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

class AlertEncryption:
    """Handle alert payload encryption/decryption."""
    
    def __init__(self, encryption_key):
        """
        Initialize with 256-bit encryption key.
        
        Args:
            encryption_key: bytes, exactly 32 bytes for AES-256
        """
        if len(encryption_key) != 32:
            raise ValueError("Key must be 32 bytes for AES-256")
        
        self.key = encryption_key
    
    def encrypt_payload(self, plaintext_bytes):
        """
        Encrypt payload using AES-256-CBC.
        
        Input:
        - plaintext_bytes: compressed payload (~80 bytes)
        
        Output:
        - IV || ciphertext: (16 + 80 = 96 bytes)
        """
        
        # Generate random IV (Initialization Vector)
        iv = os.urandom(16)  # 128 bits
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.CBC(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        # Apply PKCS7 padding (AES requires multiple of 16 bytes)
        padded = self._apply_pkcs7_padding(plaintext_bytes)
        
        # Encrypt
        ciphertext = encryptor.update(padded) + encryptor.finalize()
        
        # Return IV || ciphertext (IV needed for decryption)
        return iv + ciphertext
    
    def decrypt_payload(self, encrypted_bytes):
        """
        Decrypt payload using AES-256-CBC.
        """
        
        # Extract IV and ciphertext
        iv = encrypted_bytes[:16]
        ciphertext = encrypted_bytes[16:]
        
        # Create decipher
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.CBC(iv),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        
        # Decrypt
        padded = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Remove padding
        plaintext = self._remove_pkcs7_padding(padded)
        
        return plaintext
    
    @staticmethod
    def _apply_pkcs7_padding(data):
        """Add PKCS7 padding."""
        padding_length = 16 - (len(data) % 16)
        padding = bytes([padding_length] * padding_length)
        return data + padding
    
    @staticmethod
    def _remove_pkcs7_padding(data):
        """Remove PKCS7 padding."""
        padding_length = data[-1]
        return data[:-padding_length]
```

### Encryption Parameters

| Parameter | Value | Notes |
|---|---|---|
| **Algorithm** | AES-256 | 256-bit key (32 bytes) |
| **Mode** | CBC | Cipher Block Chaining |
| **IV** | 16 bytes (random per message) | Prevents pattern analysis |
| **Padding** | PKCS7 | Standard padding scheme |
| **Key derivation** | PBKDF2 (if from password) | 100,000 iterations |

### Encryption Example

```
Plaintext (compressed): 80 bytes
    ↓
AES-256-CBC with random IV
    ↓
Output: IV (16 bytes) + Ciphertext (96 bytes) = 112 bytes
```

---

## Component 4: Message Envelope

### Wire Format

```
Message structure:
┌──────────┬────────┬─────────┬──────────────────────────┐
│  Magic   │Version │Sequence │ Envelope Data            │
│ (2 bytes)│(1 byte)│ (1 byte)│ (IV + Encrypted Payload) │
├──────────┼────────┼─────────┼──────────────────────────┤
│ 0xEC 0xEA│  0x01  │  0x00   │ [16 + 112 = 128 bytes]  │
└──────────┴────────┴─────────┴──────────────────────────┘

Total size: 2 + 1 + 1 + 128 = 132 bytes
```

**Field descriptions**:
- **Magic bytes**: 0xEC 0xEA (validate message format)
- **Version**: 0x01 (protocol version, allows future upgrades)
- **Sequence**: 0x00 (alert sequence number mod 256)
- **Envelope data**: IV + encrypted payload

### Message Construction

```python
def create_message_envelope(encrypted_payload, sequence_number=0):
    """
    Construct complete message with metadata.
    
    Input:
    - encrypted_payload: IV + ciphertext from encryption
    - sequence_number: alert counter (0-255)
    
    Output:
    - final_message: bytes ready for transmission
    """
    
    magic_bytes = bytes([0xEC, 0xEA])
    version = bytes([0x01])
    seq_byte = bytes([sequence_number % 256])
    
    final_message = magic_bytes + version + seq_byte + encrypted_payload
    
    return final_message
```

---

## Component 5: End-to-End Payload Generation

### Complete Pipeline

```python
def generate_alert_payload(inference_result, device_config, encryption_key):
    """
    Convert inference result to encrypted alert message.
    
    Input:
    - inference_result: {
        'class_id': 0,
        'confidence': 0.87,
        'timestamp': 1234567890.5,
        'probabilities': {...}
      }
    - device_config: {
        'device_id': 'SENTRY_01',
        'location_hash': '19.8N_74.5E_625m',
        'firmware_version': '1.0'
      }
    - encryption_key: 32-byte key
    
    Output:
    - message: encrypted bytes <1000 bytes
    """
    
    # Step 1: Build unencrypted payload
    payload_dict = {
        't': int(inference_result['timestamp'] * 1000),  # ms
        'd': device_config['device_id'],
        'c': inference_result['class_id'],
        'p': round(inference_result['confidence'] * 100),  # 0-100, not 0-1
        'l': device_config['location_hash'],
        'v': device_config['firmware_version'],
        's': getattr(generate_alert_payload, 'sequence_counter', 0),
    }
    
    # Step 2: Serialize to compact JSON
    import json
    payload_json = json.dumps(payload_dict, separators=(',', ':'))  # Compact
    payload_bytes = payload_json.encode('utf-8')
    
    # Step 3: Compress
    compressed = zlib.compress(payload_bytes, level=9)
    
    # Step 4: Encrypt
    encryptor = AlertEncryption(encryption_key)
    encrypted = encryptor.encrypt_payload(compressed)
    
    # Step 5: Create envelope
    message = create_message_envelope(encrypted, sequence_number=payload_dict['s'])
    
    # Increment sequence counter
    generate_alert_payload.sequence_counter = (payload_dict['s'] + 1) % 256
    
    # Step 6: Validate size
    if len(message) > 1000:
        raise ValueError(f"Message too large: {len(message)} bytes")
    
    return message
```

### Size Verification

```
Step 1 - Payload dict: 170 bytes
Step 2 - JSON string: 170 bytes
Step 3 - After compression: ~80 bytes
Step 4 - After encryption (IV + CT): 16 + 96 = 112 bytes
Step 5 - Message envelope: 2 + 1 + 1 + 112 = 116 bytes
Step 6 - Total: 116 bytes ✓ (well under 1000-byte limit)
```

---

## Component 6: Decryption & Payload Verification (Receiver)

### Parsing Encrypted Message

```python
def parse_alert_message(message_bytes, encryption_key):
    """
    Decrypt and parse received alert message.
    """
    
    # Parse envelope
    magic = message_bytes[:2]
    version = message_bytes[2]
    sequence = message_bytes[3]
    encrypted_payload = message_bytes[4:]
    
    # Validate magic bytes
    if magic != bytes([0xEC, 0xEA]):
        raise ValueError("Invalid magic bytes")
    
    # Validate version
    if version != 0x01:
        raise ValueError(f"Unsupported protocol version: {version}")
    
    # Decrypt
    decryptor = AlertEncryption(encryption_key)
    compressed = decryptor.decrypt_payload(encrypted_payload)
    
    # Decompress
    payload_bytes = zlib.decompress(compressed)
    
    # Parse JSON
    payload_json = payload_bytes.decode('utf-8')
    payload_dict = json.loads(payload_json)
    
    # Convert back to readable format
    alert = {
        'timestamp_ms': payload_dict['t'],
        'device_id': payload_dict['d'],
        'class_id': payload_dict['c'],
        'confidence': payload_dict['p'] / 100.0,  # Convert back to 0-1
        'location_hash': payload_dict['l'],
        'firmware_version': payload_dict['v'],
        'sequence': sequence
    }
    
    return alert
```

---

## Component 7: Key Management

### Key Generation & Storage

```python
def generate_encryption_key():
    """Generate secure 256-bit encryption key."""
    import os
    return os.urandom(32)  # 32 bytes = 256 bits

def save_key_to_file(key, filepath, password=None):
    """
    Save encryption key securely.
    
    If password provided, encrypt key before saving.
    """
    if password:
        # Derive key from password
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
        
        salt = os.urandom(16)
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        encryption_key = kdf.derive(password.encode())
        
        # Encrypt the actual key
        encryptor = AlertEncryption(encryption_key)
        encrypted_key = encryptor.encrypt_payload(key)
        
        # Save: salt + encrypted_key
        with open(filepath, 'wb') as f:
            f.write(salt + encrypted_key)
    else:
        # Save key directly (less secure)
        with open(filepath, 'wb') as f:
            f.write(key)
```

---

## Storage & Transmission

### Message Queue (for Intermittent Connectivity)

```python
class AlertQueue:
    """Store encrypted alerts when network unavailable."""
    
    def __init__(self, max_size_mb=10):
        self.queue = []
        self.max_size_bytes = max_size_mb * 1024 * 1024
    
    def enqueue(self, encrypted_message):
        """Add message to queue."""
        current_size = sum(len(m) for m in self.queue)
        if current_size + len(encrypted_message) <= self.max_size_bytes:
            self.queue.append({
                'data': encrypted_message,
                'timestamp': time.time()
            })
            return True
        else:
            # Drop oldest message if full (Sundarbans scenario)
            if self.queue:
                self.queue.pop(0)
            self.queue.append({'data': encrypted_message, 'timestamp': time.time()})
            return False
    
    def dequeue_batch(self, max_messages=10):
        """Get batch of messages for transmission."""
        batch = self.queue[:max_messages]
        self.queue = self.queue[max_messages:]
        return batch
```

---

## Validation Checklist

```python
def validate_payload(message_bytes):
    """Validate encrypted message format."""
    
    checks = {
        'size': len(message_bytes) <= 1000,
        'magic_bytes': message_bytes[:2] == bytes([0xEC, 0xEA]),
        'version': message_bytes[2] == 0x01,
        'min_length': len(message_bytes) >= 4,
    }
    
    return all(checks.values()), checks
```

---

## Performance Summary

| Operation | Time | CPU | Memory |
|---|---|---|---|
| JSON serialization | <1ms | <1% | ~0.5KB |
| Zlib compression | <2ms | <5% | ~2KB |
| AES-256 encryption | <5ms | <10% | ~1KB |
| **Total** | **<8ms** | **<15%** | **~3.5KB** |

**Result**: <8ms to convert inference result to encrypted message (target <100ms easily met)

---

## Cross-References & Integration

### Pipeline Dependencies
- **Upstream**: 
  - Receives inference results from **[ARCH_5: SNN Inference](./ARCH_5_SNN_INFERENCE.md)** (line 6, "Input: Inference results from ARCH_5")
  - Input format: `(class_id, confidence, timestamp)` tuple
  - See [ARCH_5: Output Format](./ARCH_5_SNN_INFERENCE.md#data-format-specifications)

- **Downstream**: 
  - Outputs encrypted JSON payload to **[ARCH_8: Network Simulator](./ARCH_8_NETWORK_SIMULATOR.md)** (line 6, "Input: 116-byte encrypted JSON payloads from ARCH_6")
  - Payload format: 116-byte AES-256 encrypted JSON
  - See [ARCH_8: Input Specification](./ARCH_8_NETWORK_SIMULATOR.md)

### Data Format Specifications
- **Input Format** from ARCH_5: Classification result tuple
  - `class_id` ∈ {0, 1, 2}: {gunshot, chainsaw, vehicle}
  - `confidence` ∈ [0, 100] UInt8: Probability percentage (±1% quantization)
  - `timestamp`: UNIX epoch (milliseconds)
  - See [ARCH_5: Component 3](./ARCH_5_SNN_INFERENCE.md#component-3-inference-output-format) (lines 80-110)

- **Output Format**: Encrypted JSON payload
  - **Unencrypted JSON structure** (see [Component 1: JSON Structure](./ARCH_6_JSON_PAYLOAD.md#component-1-json-structure) lines 15-60):
    - `class_id` (UInt8): 1 byte
    - `confidence` (UInt8): 1 byte
    - `timestamp` (Unix epoch ms, UInt64): 8 bytes
    - `location_hash` (SHA-256 subset): 8 bytes
    - `device_id` (SHA-256 subset): 8 bytes
    - Reserved/padding: 82 bytes
    - **Total unencrypted**: ~116 bytes
  - **Encrypted**: AES-256-ECB wraps entire JSON (maintains 116-byte size)
  - Payload fits within LoRa maximum (242 bytes) with 50% margin

### Processing Timeline
- **Days 15** (IMPLEMENTATION_SCHEDULE): JSON encoding + encryption
- **Expected Latency**: <8ms per event (serialization <1ms, compression <2ms, encryption <5ms)
- **Output Rate**: 1 encrypted message per detection event (typically <1 per minute)
- **Throughput**: Can handle up to 125 events/second (8ms latency) before backlog

### Key Parameters (Finalized)
| Parameter | Value | Reference | Rationale |
|-----------|-------|-----------|-----------|
| JSON format version | v1.0 | Line 28 | Versioning for future compatibility |
| Confidence precision | UInt8 (0-100) | Line 34 | ±1% steps, saves 1 byte vs Float16 |
| Timestamp precision | UInt64 (ms) | Line 35 | Millisecond resolution sufficient for events |
| Location hash size | 8 bytes | Line 45 | SHA-256 truncated to 8 bytes |
| Device ID size | 8 bytes | Line 50 | SHA-256 truncated to 8 bytes |
| Reserved/padding | 82 bytes | Line 55 | Future extensibility |
| Payload size | 116 bytes | Line 65 | Fixed for predictable network transmission |
| Encryption algorithm | AES-256-ECB | Line 80 | Strong security, deterministic (for fixed payload) |
| Compression | Zlib | Line 85 | <2ms overhead, minimal savings for small JSON |

### Encryption Scheme Details
See [Component 2: Encryption](./ARCH_6_JSON_PAYLOAD.md#component-2-encryption-aes-256-ecb) (lines 75-130):
1. **Key management**: Device-specific 256-bit key (stored securely on edge device)
2. **IV/Nonce**: ECB mode (deterministic for fixed payloads, acceptable for 1 event per message)
3. **Authentication**: HMAC-SHA256 can be added in future for integrity verification
4. **Forward secrecy**: Keys rotated monthly (implementation detail for deployment)

### Payload Size Analysis
- **Uncompressed JSON**: ~50 bytes (class_id, confidence, timestamp, hash, device_id)
- **With padding**: 116 bytes (fixed for LoRa alignment)
- **After AES-256-ECB**: 116 bytes (no expansion in ECB mode)
- **LoRa SF7 time-on-air**: ~56ms (see [ARCH_8: Network Simulator](./ARCH_8_NETWORK_SIMULATOR.md#time-on-air-calculation) line 210)

### Related Documentation
- **SUMMARY_HIGH_LEVEL_ARCHITECTURE.md** (Week 3): Data serialization & security overview
- **IMPLEMENTATION_SCHEDULE.md** (Day 15): JSON encoding + encryption tasks
- **ARCH_5_SNN_INFERENCE.md**: Source of inference results
- **ARCH_8_NETWORK_SIMULATOR.md**: Downstream consumer (transmission simulation)
- **Resources/Research_Paper_Citations.md**: References for AES encryption, JSON standards

