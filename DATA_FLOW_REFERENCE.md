# Data Flow Reference: Complete Tensor Dimensions & Transformations

## Overview

This document provides a comprehensive reference for all tensor shapes, data formats, and transformations throughout the Eco-Sentry pipeline. Use this as a quick lookup table when implementing or debugging any architecture stage.

---

## End-to-End Pipeline with Dimensions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ FOREST AUDIO (RAW)                                          ARCH_1          │
│ Format: WAV/MP3, Variable sample rate                                       │
│ Typical duration: 10 seconds                                                │
│ ↓                                                                           │
│ PROCESS: Load → Resample to 16kHz → RMS normalize → Bandpass filter        │
│          → STFT (window=512, hop=160) → Mel-scale (64 bands) → dB scale    │
│ ↓                                                                           │
│ OUTPUT: MEL-SPECTROGRAM TENSOR                                             │
│         Shape: (T, 64)                                                     │
│         - T = 1000 for 10s audio (100 frames/second at 10ms hop)           │
│         - 64 = mel-frequency bands (50Hz to 8kHz perceptual scale)         │
│         - Values: [-80, 0] dB (log-magnitude)                              │
│         - Dtype: float32                                                   │
│         - Size: ~32KB uncompressed                                         │
│         - Example shape: (1000, 64)                                        │
└─────────────────────────────────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ SPIKE CONVERSION (NEUROMORPHIC ENCODING)               ARCH_2              │
│ Input: Mel-spectrogram (T, 64) from ARCH_1                                 │
│ ↓                                                                           │
│ PROCESS: Normalize input [-80,0]→[0,1] → Apply LIF neurons (64 neurons)    │
│          → Binary spike generation → Output at frame-level resolution      │
│ ↓                                                                           │
│ OUTPUT: SPIKE TRAIN TENSOR                                                 │
│         Shape: (T, 64, 1)                                                  │
│         - T = 1000 frames (frame-level, NOT sample-level!)                 │
│         - 64 = LIF neurons (one per mel-band)                              │
│         - 1 = binary spike dimension                                       │
│         - Values: {0, 1} binary (0=no spike, 1=spike occurred)             │
│         - Dtype: uint8 or bool                                             │
│         - Sparsity: ~25% neurons fire per frame (on average)               │
│         - Size: ~64KB (1000×64×1 bytes)                                    │
│         - Per-sample latency: 15ms (frame-level processing)                │
│         - Example shape: (1000, 64, 1)                                     │
│                                                                             │
│ KEY INSIGHT: Frame-level (10ms) resolution, NOT 16kHz sample-level!        │
│ Frame interval: 10ms (matches hop_length from ARCH_1)                     │
│ Firing rate target: 25% ±5% (tuned in ARCH_3)                             │
└─────────────────────────────────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ DATASET PREPARATION & FOREST NORMALIZATION           ARCH_3                │
│ Input: 600 raw spike tensors from ARCH_2                                   │
│ ↓                                                                           │
│ STAGE 1: Load 600 spike samples                                            │
│          - ESC-50: 100 samples                                             │
│          - UrbanSound8K: 150 samples                                       │
│          - Corbett (Custom): 150 samples                                   │
│          - Seshachalam (Custom): 100 samples                               │
│          - Sundarbans (Custom): 100 samples                                │
│          Shape: (600, T, 64, 1)                                            │
│          - 600 = number of training samples                                │
│          - T ≈ 1000 frames per sample                                      │
│          - 64 = mel-band neurons                                           │
│          - 1 = binary spike dimension                                      │
│                                                                             │
│ STAGE 2: Forest-specific normalization                                     │
│          - Corbett (dense forest): High noise suppression                   │
│          - Seshachalam (open terrain): Simple global normalization          │
│          - Sundarbans (wetlands): Sustained pattern removal                 │
│          - Target firing rate: 25% ±5% across all forests                  │
│          Shape remains: (600, T, 64, 1)                                    │
│                                                                             │
│ STAGE 3: Train/Validation/Test split (60/20/20)                            │
│          - Train: 360 samples (60%)                                        │
│          - Validation: 120 samples (20%)                                   │
│          - Test: 120 samples (20%)                                         │
│          - Stratified by class and forest                                  │
│                                                                             │
│ STAGE 4: Data augmentation (600 → 1200 samples)                            │
│          - Mixup (50% probability): Interpolate 2 spike tensors            │
│          - Time-shift (75% probability): ±5 frames (±50ms)                 │
│          - Noise (25% probability): Dropout-based augmentation             │
│          Total: 600 original + 600 augmented = 1200 samples                │
│          Final shapes:                                                     │
│          - Training: 720 samples after augmentation                        │
│          - Validation: 120 samples (no augmentation)                       │
│          - Test: 120 samples (no augmentation)                             │
│          - Each sample: (T, 64, 1) with T ≈ 1000 frames                   │
│                                                                             │
│ OUTPUT: Balanced, normalized, augmented spike dataset                      │
│         Shape: (1200, T, 64, 1) total after augmentation                   │
│         Size: ~76.8MB (1200×1000×64×1 bytes)                               │
│         Training/Val/Test: 720 + 240 + 240 = 1200 samples                  │
└─────────────────────────────────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ SNN TRAINING                                           ARCH_4               │
│ Input: Training dataset from ARCH_3 (720 samples after augmentation)        │
│ ↓                                                                           │
│ MODEL ARCHITECTURE:                                                         │
│ Layer 0 (Input): 64 LIF neurons (receives spike input)                     │
│ Layer 1 (Hidden): 128 LIF neurons                                          │
│         Weights: W1 shape (64, 128)                                        │
│         Biases: b1 shape (128,)                                            │
│ Layer 2 (Hidden): 64 LIF neurons                                           │
│         Weights: W2 shape (128, 64)                                        │
│         Biases: b2 shape (64,)                                             │
│ Layer 3 (Output): 3 neurons (dense, non-spiking)                           │
│         Weights: W3 shape (64, 3)                                          │
│         Biases: b3 shape (3,)                                              │
│                                                                             │
│ TRAINING DATA BATCHES:                                                      │
│ Batch size: 32 samples per batch                                           │
│ Input batch: (32, T, 64, 1) where T ≈ 1000 frames                          │
│ Label batch: (32,) with class_id ∈ {0, 1, 2}                              │
│ - 0 = gunshot                                                              │
│ - 1 = chainsaw                                                             │
│ - 2 = vehicle                                                              │
│                                                                             │
│ LOSS FUNCTION: Sparse categorical cross-entropy                            │
│ Output logits: (32, 3) per batch                                           │
│ Softmax: (32, 3) normalized probabilities                                  │
│                                                                             │
│ TRAINING DURATION: 120 epochs                                              │
│ Per-epoch time: ~3 seconds on GPU                                          │
│ Total training: ~6 minutes                                                 │
│                                                                             │
│ OUTPUT: Trained weights                                                    │
│ Weight matrices:                                                            │
│ - W1: (64, 128) ~ 8KB at float32                                           │
│ - b1: (128,) ~ 512B                                                        │
│ - W2: (128, 64) ~ 32KB at float32                                          │
│ - b2: (64,) ~ 256B                                                         │
│ - W3: (64, 3) ~ 768B                                                       │
│ - b3: (3,) ~ 12B                                                           │
│ Total: ~42KB at float32, ~21KB at float16                                  │
│ Model test accuracy: >85% on held-out test set                             │
└─────────────────────────────────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ SNN INFERENCE (REAL-TIME DETECTION)                   ARCH_5               │
│ Input: Trained model weights from ARCH_4 + live audio stream               │
│ ↓                                                                           │
│ AT INFERENCE TIME:                                                          │
│ 1. Audio capture: 10-second window                                         │
│ 2. Process through ARCH_1: Raw audio → Mel-spectrogram (T, 64)             │
│    where T ≈ 1000 frames for 10s audio                                     │
│ 3. Process through ARCH_2: Mel-spectrogram → Spike train (T, 64, 1)        │
│ 4. Forward pass: (1, T, 64, 1) through trained SNN                         │
│    - Reshape to (1, T*64) or process frame-by-frame                        │
│    - Pass through Layer 1: (1, T*64) → (1, 128×T or 128) depending on impl │
│    - Pass through Layer 2: (1, 128) → (1, 64)                              │
│    - Pass through Layer 3: (1, 64) → (1, 3) logits                         │
│ 5. Softmax: (1, 3) → class probabilities                                   │
│ 6. Argmax: class_id ∈ {0, 1, 2}                                            │
│ 7. Confidence: max(softmax(logits)) ∈ [0, 1]                               │
│                                                                             │
│ OUTPUT: (class_id, confidence, timestamp) tuple                            │
│ - class_id: uint8 ∈ {0, 1, 2}                                              │
│ - confidence: float32 ∈ [0, 1], converted to uint8 [0, 100] for ARCH_6     │
│ - timestamp: UNIX epoch (ms), uint64                                       │
│                                                                             │
│ INFERENCE LATENCY: ~800ms per 10s audio                                    │
│ - ARCH_1 (mel-spec): ~100ms                                                │
│ - ARCH_2 (spike conv): ~15ms                                               │
│ - ARCH_5 (forward pass): ~5-20ms                                           │
│ - Total end-to-end: ~120-135ms compute, ~660ms I/O overhead                │
└─────────────────────────────────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ JSON PAYLOAD GENERATION & ENCRYPTION                 ARCH_6                │
│ Input: (class_id, confidence, timestamp) from ARCH_5                       │
│ ↓                                                                           │
│ STAGE 1: JSON Serialization                                                │
│ Python dict (uncompressed):                                                │
│ {                                                                           │
│   "version": "1.0",                                                        │
│   "class_id": 0,                        # uint8: {0,1,2}                   │
│   "confidence": 95,                     # uint8: [0,100]                   │
│   "timestamp": 1704067200000,           # uint64: UNIX epoch (ms)          │
│   "location_hash": "abc123...",         # str: 8 bytes from SHA-256         │
│   "device_id": "device_0001",           # str: 8 bytes from SHA-256         │
│   "firmware": "v1.0.0"                  # str: ~8 bytes                    │
│ }                                                                           │
│ JSON string size: ~100-150 bytes                                           │
│                                                                             │
│ STAGE 2: Compression (Zlib)                                                │
│ Compressed size: ~50-80 bytes (40-50% compression)                         │
│ Compression overhead: <2ms                                                 │
│                                                                             │
│ STAGE 3: Encryption (AES-256-ECB)                                          │
│ Plaintext: 116 bytes (padded to AES block size)                            │
│ Key: 256-bit device-specific key                                           │
│ Ciphertext: 116 bytes (ECB mode preserves size)                            │
│ Encryption time: <5ms                                                      │
│                                                                             │
│ OUTPUT: Encrypted 116-byte payload                                         │
│ Format: Binary bytes suitable for LoRa transmission                        │
│ Size: 116 bytes (fixed)                                                    │
│ Fits in LoRa maximum: 242 bytes ✓                                          │
└─────────────────────────────────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ NETWORK TRANSMISSION (LORA MESH)                     ARCH_8                │
│ Input: 116-byte encrypted JSON from ARCH_6                                 │
│ ↓                                                                           │
│ TRANSMISSION PARAMETERS:                                                    │
│ Payload size: 116 bytes                                                    │
│ Spreading Factor (SF): 7-12 (configurable for range vs. speed)             │
│ Bandwidth: 125 kHz (standard LoRa US915 ISM band)                          │
│ Coding Rate: 4/5 (error correction)                                        │
│ TX Power: 14 dBm (25mW)                                                    │
│                                                                             │
│ TIME-ON-AIR (ToA) Calculations:                                            │
│ SF7:  56ms (shortest, ~5-10km range)                                       │
│ SF9:  352ms (medium)                                                       │
│ SF10: 711ms (medium-long)                                                  │
│ SF12: 1500ms (longest, ~10-15km range in forest)                           │
│                                                                             │
│ MESH ROUTING:                                                               │
│ Forest scenario determines path:                                           │
│ Corbett (dense): 2-3 hops through relay nodes, SF12 likely                 │
│ Seshachalam (open): 1-2 hops, SF9-10 typical                               │
│ Sundarbans (wet): 1-3 hops, SF10-11 typical                                │
│                                                                             │
│ NETWORK DELIVERY:                                                           │
│ Target success rate: >95%                                                  │
│ Target latency: <1.5s (end-to-end from TX to gateway receipt)              │
│ Expected congestion: Low (1 alert per minute average)                      │
└─────────────────────────────────────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ COMMAND CENTER RECEIVES ALERT                                              │
│ Output: Decrypted alert with threat classification                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference: Tensor Shapes by Architecture

### ARCH_1: Audio Processing → Mel-Spectrogram
| Item | Value | Notes |
|------|-------|-------|
| **Input** | WAV/MP3 variable | Resampled to 16kHz |
| **Output Shape** | (T, 64) | T ≈ 1000 for 10s audio |
| **Data Type** | float32 | |
| **Value Range** | [-80, 0] dB | Log-magnitude scale |
| **Memory Size** | ~32KB | 1000 × 64 × 4 bytes |
| **Latency** | <100ms | Per 10-second sample |
| **Frame Duration** | 10ms | hop_length=160 @ 16kHz SR |
| **Num Mel Bands** | 64 | 50Hz to 8kHz |

### ARCH_2: Spike Conversion → Spike Train
| Item | Value | Notes |
|------|-------|-------|
| **Input** | (T, 64) mel-spec | From ARCH_1 |
| **Output Shape** | (T, 64, 1) | **Frame-level**, NOT sample-level |
| **Data Type** | uint8 or bool | |
| **Value Range** | {0, 1} | Binary spikes |
| **Memory Size** | ~64KB | 1000 × 64 × 1 byte |
| **Latency** | ~15ms | Per 10-second sample |
| **Sparsity** | ~25% | Firing rate average |
| **Frame Duration** | 10ms | Inherited from ARCH_1 |
| **Num Neurons** | 64 | One per mel-band |

### ARCH_3: Dataset Preparation → Normalized Dataset
| Item | Value | Notes |
|------|-------|-------|
| **Input** | (600, T, 64, 1) raw spike tensors | 600 samples from ARCH_2 |
| **Output (Train)** | (720, T, 64, 1) | After augmentation |
| **Output (Val)** | (240, T, 64, 1) | Stratified 20% |
| **Output (Test)** | (240, T, 64, 1) | Stratified 20% |
| **Data Type** | uint8 or float32 | |
| **Memory Size (Total)** | ~76.8MB | 1200 × 1000 × 64 × 1 |
| **Memory Size (Train)** | ~46.1MB | 720 samples |
| **Num Raw Samples** | 600 | ESC-50, UrbanSound8K, 3 forests |
| **Augmentation Ratio** | 600 → 1200 | 2× expansion |
| **Target Firing Rate** | 25% ±5% | Normalized across forests |

### ARCH_4: SNN Training → Trained Model
| Item | Value | Notes |
|------|-------|-------|
| **Input Batch** | (32, T, 64, 1) | 32 samples, T ≈ 1000 frames |
| **Layer 0 (Input)** | 64 neurons | Receives spike input |
| **Layer 1 (Hidden)** | 128 neurons | W1: (64, 128), b1: (128,) |
| **Layer 2 (Hidden)** | 64 neurons | W2: (128, 64), b2: (64,) |
| **Layer 3 (Output)** | 3 neurons | W3: (64, 3), b3: (3,) |
| **Output Logits** | (batch, 3) | Class logits |
| **Output Classes** | {0, 1, 2} | {gunshot, chainsaw, vehicle} |
| **Total Parameters** | ~12,547 | ~42KB at float32 |
| **Training Samples** | 720 | 60% after augmentation |
| **Validation Samples** | 240 | 20% stratified |
| **Test Samples** | 240 | 20% stratified |
| **Training Time** | ~6 minutes | 120 epochs @ 3s/epoch on GPU |
| **Target Accuracy** | >85% | On test set |

### ARCH_5: SNN Inference → Classification Result
| Item | Value | Notes |
|------|-------|-------|
| **Input (Weights)** | W1, W2, W3, b1, b2, b3 | From ARCH_4 |
| **Input (Audio)** | (1, T, 64, 1) spikes | From ARCH_2 at runtime |
| **Output class_id** | uint8 ∈ {0, 1, 2} | {gunshot, chainsaw, vehicle} |
| **Output confidence** | float32 ∈ [0, 1] | Will be converted to uint8 [0, 100] |
| **Output timestamp** | uint64 | UNIX epoch (milliseconds) |
| **Output Size** | 3 floats + 1 uint8 | ~20 bytes |
| **Inference Latency** | ~800ms | Per 10s audio (includes ARCH_1+2 overhead) |
| **Latency Target** | <3s | End-to-end from audio to alert |

### ARCH_6: JSON Payload → Encrypted Message
| Item | Value | Notes |
|------|-------|-------|
| **Input** | (class_id, confidence, timestamp) | From ARCH_5 |
| **JSON Uncompressed** | ~100-150 bytes | Includes device_id, location_hash |
| **JSON Compressed** | ~50-80 bytes | Zlib compression |
| **JSON Encrypted** | 116 bytes | AES-256-ECB (fixed size) |
| **Encryption Key** | 256 bits | Device-specific |
| **Payload Size** | 116 bytes | Fixed for LoRa transmission |
| **LoRa Max Size** | 242 bytes | 116-byte payload fits with 50% margin |
| **Encoding Time** | <8ms total | <1ms JSON + <2ms compress + <5ms encrypt |

### ARCH_8: Network Simulator → Delivery Metrics
| Item | Value | Notes |
|------|-------|-------|
| **Input Payload** | 116 bytes | From ARCH_6 |
| **Time-on-Air (SF7)** | 56ms | Shortest range, fastest transmission |
| **Time-on-Air (SF12)** | 1500ms | Longest range, slowest transmission |
| **Delivery Success** | >95% | Target success rate |
| **Latency (p99)** | <1500ms | End-to-end network latency |
| **Expected Congestion** | Low | ~1 alert per minute average |
| **Mesh Hops** | 1-3 | Forest-dependent |

---

## Implementation Timing Table

| Stage | Architecture | Input Type | Output Type | Latency | Power | Notes |
|-------|--------------|-----------|------------|---------|-------|-------|
| Audio Processing | ARCH_1 | WAV/MP3 (10s) | Mel-spec (1000,64) | <100ms | 2mW | Includes DSP overhead |
| Spike Conversion | ARCH_2 | Mel-spec | Spikes (1000,64,1) | ~15ms | 1mW | Frame-level resolution |
| Dataset Prep | ARCH_3 | 600 spike tensors | 1200 augmented | ~5-10min | - | Batch processing |
| SNN Training | ARCH_4 | 720 training samples | Trained weights | ~6min | 20W (GPU) | 120 epochs |
| SNN Inference | ARCH_5 | Trained weights + spikes | (class, conf, ts) | ~800ms | 3mW | Real-time on edge |
| JSON Payload | ARCH_6 | (class, conf, ts) | 116-byte payload | <8ms | <1mW | Serialization + encryption |
| Network TX | ARCH_8 | 116-byte payload | Gateway receipt | 56-1500ms | 140mW | SF-dependent |
| **End-to-End** | **ARCH_1→8** | **Raw audio** | **Decrypted alert** | **<1.5s** | **5-50mW** | **From detection to command center** |

---

## Data Type Reference

| Format | Bytes | Range | Use Case | Tradeoff |
|--------|-------|-------|----------|----------|
| **uint8** | 1 | [0, 255] | class_id, confidence [0-100], binary spikes | Limited range but compact |
| **uint16** | 2 | [0, 65535] | Frame counts, sample indices | Moderate overhead |
| **uint32** | 4 | [0, 4.3B] | Sample counts in datasets | Overkill for typical sizes |
| **uint64** | 8 | [0, 18.4E18] | UNIX timestamps (ms) | Standard for time |
| **float16** | 2 | ±65504 | Model weights (quantized) | 50% memory vs. float32, slight accuracy loss |
| **float32** | 4 | ±3.4E38 | Mel-specs, logits, probabilities | Standard precision for DSP/ML |
| **bool** | 1 | {0, 1} | Binary spikes (optimal) | Most compact for spike trains |

---

## Dimension Constraints & Validations

### Critical: Must Match Across Stages
- **T (time frames)**: Must be ~1000 for 10s audio across ARCH_1 → ARCH_2 → ARCH_3 (if not, pad/truncate)
- **64 mel bands**: Consistent across ARCH_1 → ARCH_2 → ARCH_3 → ARCH_4 input layer
- **Batch size**: During training (ARCH_4), typically 32 samples per batch
- **3 output classes**: {gunshot, chainsaw, vehicle} across ARCH_4 output → ARCH_5 softmax → ARCH_6 encoding

### Memory Budget Sanity Checks
- **ARCH_2 spike tensor**: 1000 × 64 × 1 = 64KB (not 160,000 × 64 × 1 = 10MB!) ← Common mistake
- **ARCH_3 dataset**: 1200 × 1000 × 64 × 1 = 76.8MB fits in typical system RAM
- **ARCH_4 trained model**: ~42KB at float32 (~20KB at float16) - easily fits on edge device
- **ARCH_6 payload**: 116 bytes - fits in single LoRa packet with margin

### Validation Checklist Before Each Stage
```
ARCH_1 → ARCH_2:
☑ Output mel-spec is (T, 64) where T ≈ 1000
☑ Values are in [-80, 0] dB range
☑ No NaN or Inf values
☑ Frame duration is 10ms (hop_length = 160 samples @ 16kHz)

ARCH_2 → ARCH_3:
☑ Spike tensor is (T, 64, 1) - NOT (160000, 64, 1)
☑ Values are {0, 1} binary
☑ Sparsity is ~25% (firing rate)
☑ Frame duration is 10ms (inherited from ARCH_1)

ARCH_3 → ARCH_4:
☑ Training set is 720 samples (after augmentation from 600)
☑ Each sample is (T, 64, 1) where T ≈ 1000
☑ Firing rate is 25% ±5% after normalization
☑ Classes are stratified {gunshot, chainsaw, vehicle}

ARCH_4 → ARCH_5:
☑ Weights loaded: W1(64×128), W2(128×64), W3(64×3)
☑ Model in .eval() mode (no backprop)
☑ Test accuracy >85% before deployment

ARCH_5 → ARCH_6:
☑ class_id is uint8 ∈ {0, 1, 2}
☑ confidence is converted to uint8 [0, 100]
☑ timestamp is UNIX epoch in milliseconds

ARCH_6 → ARCH_8:
☑ Payload is exactly 116 bytes
☑ Encrypted with AES-256-ECB
☑ Fits in LoRa maximum (242 bytes)

ARCH_8 (Network):
☑ >95% delivery success rate
☑ <1.5s end-to-end latency
☑ <5% packet loss
```

---

## Example Data Flow for Single 10-Second Audio Sample

### Input: Forest Audio File (gunshot)
```
gunshot_recording.wav
- Duration: 10 seconds
- Sample rate: 16 kHz (after resampling if needed)
- Total samples: 10 × 16,000 = 160,000 audio samples
- File size: ~320 KB (16-bit PCM)
```

### Step 1: ARCH_1 Processing
```
Process: Load → Resample → Normalize → Bandpass → STFT → Mel-scale → dB
Output mel-spectrogram:
- Shape: (1000, 64)
- T = 1000 frames (10 seconds ÷ 10ms frame duration)
- 64 mel bands from 50 Hz to 8 kHz
- Values: dB magnitude [-80, 0]
- Example: [[-45.2, -50.1, ..., -78.0], ... (1000 rows)]
```

### Step 2: ARCH_2 Processing
```
Process: Normalize [−80,0]→[0,1] → Apply 64 LIF neurons → Generate spikes
Output spike tensor:
- Shape: (1000, 64, 1)
- T = 1000 frames (matching ARCH_1)
- Binary: 0 (no spike) or 1 (spike fired)
- Sparsity: ~25% of values are 1, ~75% are 0
- Example frame t=150: [[0], [1], [0], [1], [0], ..., [0]] (64 neurons)
```

### Step 3: ARCH_3 Processing (Normalize + Augment)
```
Step 3a: Forest Normalization
- Gunshot sample from Corbett (dense forest)
- Apply Corbett-specific noise suppression (suppress bottom 10% percentile)
- Target firing rate: 25% ±5%
- Output: Normalized (1000, 64, 1) spike tensor

Step 3b: Augmentation (create 2 copies)
- Original: (1000, 64, 1) gunshot
- Augmented #1 (time-shift +3 frames): Shift spikes forward by 30ms
- Augmented #2 (mixup with vehicle): Interpolate 50% gunshot + 50% vehicle spikes
- Result: 3 samples instead of 1 (for training data expansion)
```

### Step 4: ARCH_4 Training (Batch Processing)
```
Multiple samples processed together in batches:
- Batch size: 32 samples per batch
- Each sample: (1000, 64, 1) spike tensor + label ∈ {0, 1, 2}
- Batch input: (32, 1000, 64, 1)
- Forward pass: 
  - Input (32, 1000, 64, 1) → flatten to (32, 64000)
  - Layer 1: (32, 64000) @ W1(64×128) → (32, 128)
  - Layer 2: (32, 128) @ W2(128×64) → (32, 64)
  - Output Layer: (32, 64) @ W3(64×3) → (32, 3) logits
- Loss: Cross-entropy between logits and true labels
- Backprop: Update W1, W2, W3, biases
```

### Step 5: ARCH_5 Inference (Real-Time)
```
New forest audio arrives at edge device:
1. ARCH_1: Audio → Mel-spec (1000, 64)
2. ARCH_2: Mel-spec → Spikes (1000, 64, 1)
3. ARCH_5 forward pass:
   - Input (1, 1000, 64, 1) or flattened (1, 64000)
   - Output logits: (1, 3)
   - Softmax: [0.05, 0.88, 0.07] (high confidence on class 1: chainsaw)
   - Argmax: class_id = 1
   - Confidence: max([0.05, 0.88, 0.07]) = 0.88
   - Timestamp: 1704067200123 (UNIX ms)
```

### Step 6: ARCH_6 Encoding
```
Input: (class_id=1, confidence=0.88, timestamp=1704067200123)

JSON encoding:
{
  "version": "1.0",
  "class_id": 1,
  "confidence": 88,          # uint8 [0-100]
  "timestamp": 1704067200123,
  "location_hash": "deadbeef",
  "device_id": "sensor_042",
  "firmware": "v1.0.0"
}
Size: ~120 bytes

AES-256-ECB Encryption:
- Key: device-specific 256-bit key
- Plaintext: 116 bytes (padded)
- Ciphertext: 116 bytes (ECB preserves size)
```

### Step 7: ARCH_8 Network Transmission
```
Transmit 116-byte encrypted payload via LoRa:

Spreading Factor 9 (medium):
- Time-on-air: 352ms
- Range: ~5-10km in forest
- Power: 140mW peak

Mesh routing (example):
- Device → Relay Node 1 (1 hop, 1km away): 352ms
- Relay Node 1 → Gateway (1 hop, 2km away): 352ms
- Total latency: 704ms (well within 1.5s target)

Gateway receives, decrypts, forwards to command center.
Ranger receives alert: "⚠️ CHAINSAW detected 5 minutes ago at GPS [29.2N, 79.1E]"
```

---

## Summary

This reference document consolidates all tensor shapes, transformations, and data formats across the 8-architecture Eco-Sentry pipeline. Use this as a quick lookup during implementation and debugging to ensure dimensional consistency and catch common mistakes (especially the frame-level vs. sample-level confusion in ARCH_2-3!).

**Key Takeaway**: Frame-level processing (10ms resolution, ~1000 frames per 10s audio) reduces memory and compute by 160× compared to sample-level (16kHz, 160,000 samples per 10s audio), while maintaining sufficient temporal resolution for threat detection.
