# Implementation Examples: Realistic Data at Each Pipeline Stage

## Overview

This document provides concrete, realistic examples of data flowing through the Eco-Sentry pipeline. Each section shows actual expected values, shapes, and formats that will occur during implementation.

---

## Example 1: Gunshot Detection Pipeline

### Scenario
- **Forest Location**: Corbett National Park, Uttarakhand
- **Date/Time**: 2024-01-15 14:30:00 UTC
- **Audio**: Poacher's 0.22 caliber rifle gunshot + forest background
- **Duration**: 10 seconds (includes pre-gunshot and post-gunshot ambience)

---

### ARCH_1: Audio Processing - Input to Output

#### Input: Raw Forest Audio
```
File: corbett_gunshot_20240115_143000.wav
Duration: 10 seconds
Sample rate: 16 kHz (mono)
Bit depth: 16-bit PCM
File size: ~320 KB
Format: WAV (standard audio codec)

First 5 samples (at 16 kHz):
[156, 142, -89, 201, -345, ...]  # Raw int16 values
Time: 0ms → 0.3125ms
```

#### Processing Steps Breakdown

**Step 1a: Load & Resample**
```
Input audio array:
- Length: 160,000 samples (10s × 16,000 Hz)
- dtype: int16 (after decoding from WAV)
- Range: [-32768, 32767]

After normalization to float32:
[0.00476, 0.00433, -0.00271, 0.00613, -0.01050, ...]
Range: [-1.0, 1.0]
```

**Step 1b: RMS Normalization**
```
Original RMS value:
rms = sqrt(mean(audio²)) = 0.087

Target RMS: 0.1

Normalized audio (first 10 samples):
[0.00547, 0.00496, -0.00311, 0.00703, -0.01204, 
 0.00118, -0.00225, 0.00641, -0.00782, 0.00398]

Adjustment factor: 0.1 / 0.087 ≈ 1.149
```

**Step 1c: Bandpass Filter (100Hz - 8kHz)**
```
Filter type: Butterworth, order=5
Low cutoff: 100 Hz
High cutoff: 8000 Hz
Removes: Wind noise (<100Hz), electronic noise (>8kHz)

After filtering (first 10 samples):
[-0.00011, 0.00198, -0.00156, 0.00789, -0.01023,
  0.00223, -0.00198, 0.00612, -0.00701, 0.00412]
(Note: Slight attenuation, phase shift applied)
```

**Step 1d: STFT Computation**
```
Window: Hann window, 512 samples
Hop length: 160 samples (10ms intervals)
Number of frames: floor((160000 - 512) / 160) + 1 ≈ 999 frames

Frame 1 (t=0-32ms):
- STFT input: 512 samples (32ms duration)
- FFT output: 257 frequency bins (512/2 + 1)
- Magnitude spectrum: [0.002, 0.001, 0.0015, ..., 0.00001]

Frame 150 (t=1.5s - 1.532s):
- Contains part of gunshot event
- Magnitude spectrum shows spike in low-mid frequencies
  (gunshot has energy ~500Hz-4kHz typically)

Frame 999 (t=9.99s):
- Near end of audio, mostly forest ambient
- Lower magnitude, broader frequency spread
```

**Step 1e: Mel-Scale Conversion**
```
Convert 257 frequency bins → 64 mel-frequency bands

Mel-band frequencies (centers):
Band 0: 50 Hz
Band 1: 56 Hz
Band 2: 63 Hz
...
Band 32: 1000 Hz (middle band)
...
Band 63: 8000 Hz

Example: Frame 150 (gunshot event)
Raw frequency magnitudes (first 10 bins):
[0.012, 0.011, 0.015, 0.009, 0.008, 0.010, 0.011, 0.009, 0.007, 0.006]

After mel-scale filtering (64 bands):
[0.045, 0.052, 0.061, 0.058, 0.055, 0.049, 0.042, 0.038,
 0.035, 0.031, ..., 0.001]  # 64 values per frame
```

**Step 1f: dB Conversion (Log-Scale)**
```
Formula: mel_db = 10 * log10(mel_magnitude + epsilon)
epsilon = 1e-10 (avoid log(0))

Example: Frame 150, Mel-bands
Linear: [0.045, 0.052, 0.061, 0.058, 0.055, ...]
dB: [-13.46, -12.84, -12.15, -12.37, -12.60, ...]

Clipping to [-80, 0] dB range:
- Values < -80 dB are clipped to -80 (noise floor)
- Values > 0 dB are clipped to 0 (max amplitude)

Final for frame 150:
[-13.46, -12.84, -12.15, -12.37, -12.60, -14.11, -15.52, -16.42,
 -16.85, -17.70, -18.01, -19.23, -21.45, -24.67, -28.90, -35.12,
 -42.34, -56.78, -70.12, -79.45, -80.00, -80.00, ..., -80.00]
 
(Note: Last values are near silence/noise floor)
```

#### ARCH_1 Output: Complete Mel-Spectrogram

```python
output_mel_spec = numpy.ndarray
shape: (1000, 64)
dtype: float32

Full spectrogram (showing frames 148-152 around gunshot):

Frame 148 (t=1.48s, before gunshot):
[-45.12, -46.89, -43.21, -50.12, -52.34, -55.67, -60.12, -65.34,
 -68.90, -70.12, -72.45, -75.67, -76.89, -78.34, -79.45, -80.00,
 ..., -80.00]  # Mostly forest ambient

Frame 149 (t=1.49s, gunshot onset):
[-23.45, -24.67, -22.90, -28.12, -30.34, -32.67, -37.12, -42.34,
 -45.90, -48.12, -50.45, -55.67, -58.89, -68.34, -75.45, -79.34,
 ..., -80.00]  # Gunshot energy rising

Frame 150 (t=1.50s, peak gunshot):
[-13.46, -12.84, -12.15, -12.37, -12.60, -14.11, -15.52, -16.42,
 -16.85, -17.70, -18.01, -19.23, -21.45, -24.67, -28.90, -35.12,
 -42.34, -56.78, -70.12, -79.45, -80.00, -80.00, ..., -80.00]
 # Peak gunshot energy across multiple mel-bands

Frame 151 (t=1.51s, gunshot tail):
[-28.34, -29.45, -27.67, -33.12, -35.34, -37.67, -42.12, -47.34,
 -50.90, -53.12, -55.45, -60.67, -63.89, -73.34, -80.00, -80.00,
 ..., -80.00]  # Energy decaying

Frame 152 (t=1.52s, post-gunshot):
[-48.12, -49.34, -47.89, -52.12, -54.34, -57.67, -62.12, -67.34,
 -70.90, -72.12, -74.45, -77.67, -78.89, -80.00, -80.00, -80.00,
 ..., -80.00]  # Back to forest ambient

... (frames 153-999 continue with forest ambient similar to frames 148)

Statistics:
- Min value: -80.0 dB (noise floor)
- Max value: -12.15 dB (gunshot peak)
- Mean value: -65.3 dB (mostly ambient)
- Median value: -72.1 dB
```

---

### ARCH_2: Spike Conversion - Input to Output

#### Input: Mel-Spectrogram from ARCH_1
```
Shape: (1000, 64)
Same as ARCH_1 output above
```

#### Processing: LIF Neuron Dynamics

**Step 2a: Input Normalization**
```
Convert mel-spec [-80, 0] dB → [0, 1] for LIF input

Formula: I_normalized = (mel_spec_dB + 80) / 80

Frame 150 input (mel-spec):
[-13.46, -12.84, -12.15, -12.37, -12.60, -14.11, -15.52, -16.42,
 -16.85, -17.70, -18.01, -19.23, -21.45, -24.67, -28.90, -35.12,
 -42.34, -56.78, -70.12, -79.45, -80.00, ...]

Frame 150 normalized input (for LIF):
[0.8319, 0.8398, 0.8481, 0.8455, 0.8425, 0.8261, 0.8061, 0.7948,
 0.7894, 0.7788, 0.7750, 0.7597, 0.7269, 0.6791, 0.6388, 0.5611,
 0.4796, 0.2903, 0.1236, 0.0069, 0.0000, ..., 0.0000]
```

**Step 2b: LIF Neuron State Evolution**

```
For each neuron n = 0 to 63:
For each time frame t = 0 to 999:

LIF equations:
V(t) = α * V(t-1) + I(t)
spike(t) = 1 if V(t) > V_th else 0
V(t) = 0 if spike(t) == 1  (reset on spike)

Parameters:
- α (decay constant) = exp(-dt / τ_m) = exp(-10ms / 10ms) ≈ 0.3679
- V_th (threshold) = 1.0
- Initial V(0) = 0 for all neurons

Neuron 5 example trace (frames 148-152):
Time(ms) | I(t)  | V(t-1) | V(t) calc     | Spike | V(t) after
1480     | 0.577 | 0      | 0.577         | 0     | 0.577
1490     | 0.606 | 0.577  | 0.212 + 0.606 | 0     | 0.818
1500     | 0.843 | 0.818  | 0.301 + 0.843 | 1     | 0       # SPIKE!
1510     | 0.606 | 0      | 0.000 + 0.606 | 0     | 0.606
1520     | 0.573 | 0.606  | 0.223 + 0.573 | 0     | 0.796

Frame 150: Neuron 5 produces spike (1) because V(t) exceeded threshold
Frames around gunshot: Higher input current → more spikes
Frames in ambient: Lower input current → fewer spikes
```

#### ARCH_2 Output: Spike Train Tensor

```python
spike_train = numpy.ndarray
shape: (1000, 64, 1)
dtype: uint8 (values: 0 or 1)

Displaying spike tensor T[t, n, 0] for frame 150 (gunshot peak):
Frame 150 spikes across 64 neurons:
[1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  
Firing rate: 9 neurons active / 64 total ≈ 14% (lower than usual)
(Peak gunshot has strong energy in low-mid frequencies, not all 64 bands)

Frame 148 (pre-gunshot ambient):
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
 
Firing rate: 0 neurons active / 64 total = 0% (forest ambient is quiet)

Frame 152 (post-gunshot):
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
 
Firing rate: 0% (returning to ambient baseline)

Overall spike statistics:
- Total spikes in 1000 frames × 64 neurons: ~25,000 events
- Average firing rate: 25,000 / (1000 × 64) ≈ 0.391 = 39% (higher than target!)
  This will be normalized down to 25% by ARCH_3
- Spike concentration: Frames 149-151 contain ~60% of all spikes
  (Temporal concentration during gunshot event - desired for discrimination)
```

---

### ARCH_3: Dataset Preparation - Normalization Example

#### Input: 1 Raw Spike Tensor
```
Shape: (1000, 64, 1)
From: Corbett gunshot (from ARCH_2)
Firing rate before normalization: 39%
```

#### Processing: Forest-Specific Normalization

**Step 3a: Corbett Strategy (Dense Forest)**
```
Goal: High ambient noise suppression (Corbett has bird calls, insects)

Algorithm:
1. Compute firing rate per neuron: sum(spikes along time) / T
2. For each neuron n:
   - If firing_rate[n] < 10th percentile: mark as noise
   - Suppress (set to 0) for low-amplitude activations
3. Target: Reduce to 25% ±5%

Per-neuron firing rates (before suppression):
Neuron 0: 18 spikes / 1000 frames = 1.8%
Neuron 1: 22 spikes / 1000 frames = 2.2%
Neuron 2: 35 spikes / 1000 frames = 3.5%  # Active in gunshot
...
Neuron 32: 450 spikes / 1000 frames = 45%  # Very active (mid-freq peak)
...
Neuron 63: 2 spikes / 1000 frames = 0.2%

10th percentile threshold: ~5 spikes per neuron
Suppress neurons with <5 spikes (noise):
- Neurons 0, 1, ..., 63: Keep 60 neurons, suppress 4 (very sparse)

After suppression:
- Total remaining spikes: ~24,000 (from 25,000)
- New firing rate: 24,000 / (1000 × 64) ≈ 37.5%

Need to reduce further to 25%:
2. Per-frame re-scaling:
   - Scale down frame-by-frame spikes by factor 0.67
   - Each spike becomes spike * 0.67 (with rounding)
   - Result: ~25% firing rate target

Final normalized spike tensor:
- Shape: (1000, 64, 1)
- Firing rate: 25% ±2%
- Temporal structure preserved (gunshot peak still visible)
```

---

### ARCH_4: SNN Training - Batch Example

#### Training Batch
```
Batch size: 32 samples
Each sample: (1000, 64, 1) spike tensor
Batch shape: (32, 1000, 64, 1)

Class distribution in batch:
- Gunshots: 11 samples (class 0)
- Chainsaws: 10 samples (class 1)
- Vehicles: 11 samples (class 2)

Memory size: 32 × 1000 × 64 × 1 = 2,048,000 values
Storage: ~2MB per batch at uint8

Example labels:
labels = [0, 1, 2, 0, 1, 1, 2, 0, 0, 1, 0, 2, 1, 1, 0, 2,
          1, 0, 2, 1, 0, 0, 1, 2, 1, 0, 2, 1, 0, 1, 2, 0]
```

#### Forward Pass Through SNN

```
Input: (32, 1000, 64, 1) spikes

Reshape for processing: (32, 64000) or process frame-by-frame

Layer 1 (Input → Hidden):
- Input: (32, 64000)
- Weight W1: (64, 128) pre-flattened dimension handling
- Linear transformation: (32, 64) @ W1 → (32, 128)
- LIF activation applied
- Output: (32, 128) spiking hidden layer

Layer 2 (Hidden → Hidden):
- Input: (32, 128)
- Weight W2: (128, 64)
- Linear transformation: (32, 128) @ W2 → (32, 64)
- LIF activation applied
- Output: (32, 64) spiking hidden layer

Layer 3 (Output):
- Input: (32, 64)
- Weight W3: (64, 3)
- Linear transformation: (32, 64) @ W3 → (32, 3) logits
- NO spike activation (dense output layer)
- Softmax applied
- Output: (32, 3) probabilities
```

#### Loss Calculation

```
For one sample in batch (sample 0: true class = 0, gunshot):

Forward output logits: [3.2, -1.5, -0.8]

Softmax computation:
exp_logits = exp([3.2, -1.5, -0.8]) = [24.5, 0.223, 0.449]
sum_exp = 25.172
softmax = [24.5/25.172, 0.223/25.172, 0.449/25.172]
       = [0.974, 0.0089, 0.0178]

Cross-entropy loss (true class = 0):
loss = -log(softmax[0]) = -log(0.974) ≈ 0.0265

Batch loss (average over 32 samples):
batch_loss = mean([0.0265, 0.031, 0.025, ...]) ≈ 0.0285
```

#### Backpropagation & Weight Update

```
Learning rate: lr = 0.0005 (with decay schedule)

Gradient for one weight in W3:
dL/dW3[i,j] ≈ 0.0018 (computed via surrogate gradient)

Weight update:
W3_new = W3_old - lr * dL/dW3
W3_new[i,j] = W3_old[i,j] - 0.0005 * 0.0018
            = W3_old[i,j] - 0.0000009

(Changes accumulate over many batches/epochs)
```

#### Training Progress Example

```
Epoch 1, Batch 1:
  Loss: 1.245
  Accuracy (batch): 28/32 = 87.5%

Epoch 1, Batch 10:
  Loss: 0.876
  Accuracy: 30/32 = 93.8%

Epoch 1 Average:
  Loss: 0.945
  Accuracy: 90.2%
  
Epoch 10:
  Loss: 0.387
  Accuracy: 96.1%

Epoch 60 (convergence):
  Loss: 0.142
  Accuracy: 98.5%

Epoch 120 (final):
  Training loss: 0.098
  Training accuracy: 99.1%
  Validation loss: 0.156
  Validation accuracy: 96.2%
  Test accuracy: 95.8% ✓ (meets >85% requirement)
```

---

### ARCH_5: SNN Inference - Real-Time Example

#### Input: New Forest Audio (unknown class)
```
File: corbett_unknown_20240115_144500.wav
Duration: 10 seconds
(Will be processed through ARCH_1 & ARCH_2 in-line)

Suppose it's actually a chainsaw (class 1)
```

#### Processing Path

```
ARCH_1 Output: Mel-spec (1000, 64)
ARCH_2 Output: Spike tensor (1000, 64, 1)

Loaded trained weights from ARCH_4:
W1: (64, 128)
b1: (128,)
W2: (128, 64)
b2: (64,)
W3: (64, 3)
b3: (3,)
```

#### Forward Pass

```
Input spike tensor: (1000, 64, 1)

Layer 1:
- Input: (64000,)  # Flattened
- Output: (128,) logits from W1
- LIF spikes: (128,) binary

Layer 2:
- Input: (128,)
- Output: (64,) logits from W2
- LIF spikes: (64,) binary

Layer 3:
- Input: (64,)
- Logits: (3,) = [-0.5, 2.8, 0.3]
- This is a chainsaw! Strong activation on class 1
```

#### Output Generation

```
Softmax:
exp_logits = exp([-0.5, 2.8, 0.3]) = [0.606, 16.445, 1.350]
sum_exp = 18.401
probabilities = [0.0329, 0.8941, 0.0734]

Classification:
- class_id = argmax([0.0329, 0.8941, 0.0734]) = 1
- class_name = "chainsaw"
- confidence = 0.8941 ≈ 89%
- timestamp = 1705329900000 (UNIX ms)

Output tuple:
(class_id=1, confidence=89, timestamp=1705329900000)
```

---

### ARCH_6: JSON Payload - Encryption Example

#### Input: Classification Result
```
class_id = 1 (chainsaw)
confidence = 89
timestamp = 1705329900000
location_hash = sha256_truncate(29.2452,-79.1234) = "deadbeef"
device_id = sha256_truncate("sensor_corbett_01") = "cafe1234"
```

#### JSON Serialization

```json
{
  "version": "1.0",
  "class_id": 1,
  "confidence": 89,
  "timestamp": 1705329900000,
  "location_hash": "deadbeef",
  "device_id": "cafe1234",
  "firmware": "v1.0.0"
}
```

**Size**: 117 bytes (including quotes, colons, commas, newlines)

#### Encryption

```
Plaintext (padded to 128-byte AES block):
deadbeef cafe1234 1705329900000 89 1 v1.0.0 ... (padding)
116 bytes → padded to 128 bytes with PKCS7 padding

AES-256-ECB with device-specific key:
key = 256-bit (32 bytes) device key stored on sensor
ciphertext = AES_Encrypt(plaintext, key)

Result: 128-byte ciphertext (ECB mode doesn't reduce size)
Use only first 116 bytes for LoRa transmission
```

#### Final Payload

```
116-byte encrypted binary blob (hex dump):
3d 2c 8f a1 e2 9b 4f 7c d1 5a 9e 2b 3f 6d 8a c4
7e 1c 9d b5 4a 6f 8e 2c a3 5b 9f 1e d4 7a 3c 5b
... (110 more hex pairs)

This payload gets transmitted via LoRa to gateway
```

---

### ARCH_8: Network Transmission - Simulation

#### Message Queuing

```
Device generates alert: chainsaw detected at 14:45:00 UTC

LoRa transmission attempt:
Spreading Factor: 9 (medium range)
Payload: 116 bytes
Time-on-air: 352ms

Transmission timeline:
14:45:00.000 - Message queued
14:45:00.100 - TX starts
14:45:00.352 - TX ends
14:45:00.500 - Gateway receives (assuming immediate reception)
14:45:00.520 - Gateway decrypts & validates
14:45:00.530 - Forward to command center via internet
```

#### Network Performance Metrics

```
Single transmission:
- Success: Yes ✓
- ToA: 352ms
- Latency from TX to receipt: 30ms
- Signal quality: -104 dBm (good signal)
- SNR: 8dB (above noise floor)

Simulated 24-hour performance (Corbett scenario):
- Alerts generated: 47 events over 24h (~2 per hour)
- Successful transmissions: 45/47 = 95.7% ✓
- Failed transmissions: 2/47 (interference window)
- Average latency: 156ms
- Max latency: 1200ms (relayed through 2 hops)
- Min latency: 78ms (direct transmission)
- Median latency: 125ms
- p99 latency: 890ms ✓ (well within 1500ms target)
```

---

## Example 2: Chainsaw Detection in Seshachalam

### Scenario
- **Forest**: Seshachalam Hills, Andhra Pradesh (open terrain)
- **Alert**: Chainsaw detected during logging attempt
- **Signal**: Chainsaw has strong 500Hz fundamental + harmonics

### Output Comparison

```
ARCH_1 Mel-spectrogram (Seshachalam):
- More energy in mid-frequency bands (chainsaws: 500Hz-2kHz dominant)
- Cleaner signal (less ambient noise than Corbett)
- Example peak frame:
  Mel-bands 20-30 (approx 500-1000Hz): [-8.5, -7.2, -6.1, -5.3, -4.8, -5.1, -6.0, -7.2, -8.5, -10.2]
  Vs. Corbett gunshot: more energy spread, lower frequencies dominate

ARCH_2 Spikes:
- Concentrated in mid-frequency neurons (20-35)
- Higher firing rate in temporal pattern
- Shows periodic spiking (chainsaw blade periodicity ~50-100Hz)

ARCH_5 Classification:
- class_id: 1 (chainsaw) ✓
- confidence: 92% (higher than gunshot due to cleaner signal)
- Certainty: Very high

ARCH_8 Network:
- SF7 used (good signal, open terrain)
- ToA: 56ms (fastest)
- Latency: 89ms total
```

---

## Example 3: False Positive Case - Vehicle Pass-By

### Scenario
- **Forest**: Sundarbans Wetlands
- **Event**: Truck passing on forest road (4km away, barely audible)
- **Challenge**: Similar energy to gunshot, but sustained duration

### Processing

```
ARCH_1 Mel-spectrogram:
- Broad frequency spread (engine noise: 200Hz-4kHz)
- Lower amplitude than gunshot/chainsaw
- Duration: Full 10-second clip (not transient like gunshot)
- Mel-band energies: -35 to -50 dB (softer than threats)

ARCH_2 Spikes:
- Firing rate: ~12% (below 25% threshold - system de-sensitized)
- Pattern: Sustained across all 1000 frames (not bursty)
- Low concentration: No strong temporal peak

ARCH_5 Classification:
- Logits: [-0.3, -0.8, 1.2]
- Softmax: [0.31, 0.18, 0.51]
- class_id: 2 (vehicle) ✓ Correct!
- confidence: 51%

ALERT DECISION:
- Confidence threshold: >85% (configured for high specificity)
- Actual confidence: 51% < 85%
- ACTION: No alert generated ✓ (false positive suppressed)
```

---

## Summary: Realistic End-to-End Flow

```
Raw audio (10s, 16kHz)
    ↓ ARCH_1 (100ms)
Mel-spectrogram (1000, 64)
    ↓ ARCH_2 (15ms)
Spike tensor (1000, 64, 1)
    ↓ ARCH_3 (during training only)
Normalized spikes → Training dataset
    ↓ ARCH_4 (training: 6min total)
Trained SNN model
    ↓ ARCH_5 (inference: 800ms)
Classification: (class_id=1, confidence=89%, timestamp)
    ↓ ARCH_6 (<8ms)
116-byte encrypted JSON
    ↓ ARCH_8 (56-1500ms LoRa ToA)
Gateway receipt & decryption
    ↓
Ranger alert: "⚠️ CHAINSAW at Corbett [29.24°N, 79.12°E] at 14:45 UTC"
```

**Total end-to-end latency**: ~900ms to 1500ms (within <3s requirement) ✓

---

