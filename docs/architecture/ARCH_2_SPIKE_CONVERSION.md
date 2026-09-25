# Architecture 2: Spike Conversion Pipeline

## Overview
The spike conversion pipeline transforms mel-spectrograms into sparse spike trains using Leaky Integrate-and-Fire (LIF) neuron models. This is the neuromorphic encoding stage that bridges traditional DSP and spiking neural networks. **Key insight**: Mel-scale frequency decomposition already models cochlear processing, so we apply LIF neurons directly to mel-bands rather than redundantly applying Gammatone filters.

**Input**: Mel-spectrogram (T, 64) from ARCH_1  
**Output**: Spike train (T, 64, 1) - binary spike tensor at frame-level time resolution  
**Latency Target**: <50ms per sample  
**Power Budget**: 1mW average consumption

---

## Data Flow

```
Mel-Spectrogram (T, 64)
    ├── Each of 64 mel-bands represents a frequency channel
    │   (already models cochlear decomposition from ARCH_1)
    │
    ↓
LIF Neuron Layer (64 neurons, 1 per mel-band)
    ├── Membrane potential dynamics: V(t) = exp(-dt/τ_m) * V(t-1) + I(t)
    ├── Threshold comparison: V_th = 1.0
    ├── Reset on spike: V → 0
    ├── Refractory period: 2ms
    └── Input normalization: Mel-band values → [0, 1]
    │
    ↓
Spike Train Output (T, 64, 1)
    ├── T = number of time frames (≈1000 for 10s audio at 10ms hop)
    ├── 64 = number of neurons (one per mel-band)
    ├── Binary: 0 or 1 at each time frame
    │
    ↓
Time-to-First-Spike (TTFS) Encoding (optional, for loss functions)
    ├── Rank neurons by spike latency
    └── Preserve temporal information in ranking
    │
    ↓
Output: Spike Train (T, 64, 1) - binary tensor
```

**Rationale for skipping Gammatone**: Mel-scale frequency bands already provide perceptually-motivated, cochlear-like decomposition. Applying Gammatone filters would redundantly re-decompose the already-decomposed signal. Instead, we directly apply LIF neurons to mel-band channels.

---

## Component 1: Mel-Spectrogram as Pseudo-Cochlear Input

### Rationale: Why No Gammatone Filterbank?

Both Gammatone filters and Mel-scale spectrograms attempt to model **cochlear frequency decomposition**. However:

| Aspect | Gammatone | Mel-Spectrogram |
|--------|-----------|-----------------|
| **Purpose** | Time-domain cochlear model | Perceptual frequency scale |
| **Operation** | Convolve time-domain audio with IR | FFT → triangular mel-scale filters |
| **Output** | Temporal envelope per channel | Log-magnitude per channel per frame |
| **Redundancy** | Would double-filter already-decomposed signal | Already provides cochlear-like decomposition |

**Decision**: Apply LIF neurons directly to mel-spectrogram channels. This avoids redundant filtering and simplifies the pipeline while preserving frequency selectivity.

### Mel-Band as LIF Input

Each mel-spectrogram column (T frames) becomes input to one LIF neuron:
- Channel 0 (50Hz): Input current to Neuron 0
- Channel 1 (56Hz): Input current to Neuron 1
- ...
- Channel 63 (8000Hz): Input current to Neuron 63

**Input normalization**:
```
function normalize_mel_input(mel_spec):
    """
    Normalize mel-spectrogram for LIF neurons.
    
    Input: mel_spec (T, 64) - dB-scaled from ARCH_1
    Output: normalized_input (T, 64) - scaled to [0, 1]
    """
    # mel_spec is in dB range [-80, 0] from ARCH_1
    # Rescale to [0, 1] for LIF input
    mel_norm = (mel_spec + 80) / 80  # Shift: [-80, 0] → [0, 1]
    mel_norm = np.clip(mel_norm, 0, 1)  # Clip to valid range
    
    return mel_norm
```

---

## Component 2: Leaky Integrate-and-Fire (LIF) Neurons

### Overview
LIF neuron model simulates spiking behavior. Each of 64 neurons receives input from one mel-spectrogram channel (frequency band) and generates sparse binary spike output at frame-level time resolution.

### LIF Equations

**Membrane potential dynamics**:
```
dV/dt = -V/τ_m + I(t)

Discretized (Euler method):
V[t+1] = α * V[t] + (1 - α) * I[t]

where:
α = exp(-dt / τ_m)  = leakage decay factor
dt = time step = 1/16000 ≈ 62.5 microseconds
τ_m = membrane time constant = 10ms
```

**Spike generation rule**:
```
if V[t] ≥ V_th:
    spike[t] = 1 (neuron fires)
    V[t+1] = V_reset = 0
    (enter refractory period)
else:
    spike[t] = 0 (no spike)
    V[t+1] = α * V[t] + I[t]

where:
V_th = 1.0 (threshold voltage)
V_reset = 0.0 (membrane resets to ground)
```

**Refractory period**:
```
if neuron in refractory period:
    spike[t] = 0 (forced to 0)
    V[t] = 0 (membrane clamped to ground)
    refractory_countdown -= dt
else if refractory_countdown ≤ 0:
    neuron exits refractory period

Parameters:
refractory_period = 2ms
```

### LIF Implementation

**Efficient vectorized implementation**:
```
function lif_neurons(mel_spec_normalized, tau_m=10e-3, v_th=1.0, 
                     refractory_ms=2, hop_ms=10, sr=16000):
    """
    Simulate 64 LIF neurons (one per mel-band).
    
    Input:
    - mel_spec_normalized: (T, 64) normalized mel-spectrogram [0, 1]
    - tau_m: membrane time constant (seconds)
    - v_th: spike threshold (normalized units)
    - refractory_ms: refractory period (milliseconds)
    - hop_ms: time per frame in mel-spec (milliseconds)
    - sr: sampling rate (for discretization reference)
    
    Output:
    - spikes: (T, 64) binary spike trains at frame level
    
    Note: Time step dt = hop_ms / 1000 seconds (frame interval, not sample interval)
    """
    
    T, n_channels = mel_spec_normalized.shape
    dt = hop_ms / 1000.0  # Time step between frames (e.g., 10ms)
    alpha = np.exp(-dt / tau_m)
    refractory_frames = int(refractory_ms / hop_ms)
    
    # Initialize state
    V = np.zeros(n_channels)  # Membrane potentials
    refractory_countdown = np.zeros(n_channels, dtype=int)  # Refractory timers
    spikes = np.zeros((T, n_channels), dtype=np.bool_)
    
    # Simulate over time frames
    for t in range(T):
        # Current input (mel-spectrogram value at this frame)
        I = mel_spec_normalized[t, :]  # (64,) - one value per mel-band
        
        # Leaky integration (only if not in refractory period)
        not_refractory = refractory_countdown <= 0
        V[not_refractory] = alpha * V[not_refractory] + (1 - alpha) * I[not_refractory]
        
        # Spike generation
        spike_mask = (V >= v_th) & (refractory_countdown <= 0)
        spikes[t, spike_mask] = True
        V[spike_mask] = 0  # Reset membrane
        refractory_countdown[spike_mask] = refractory_frames
        
        # Decrement refractory timers
        refractory_countdown = np.maximum(refractory_countdown - 1, 0)
    
    return spikes  # (T, 64) binary tensor
```

### Parameter Selection

| Parameter | Value | Rationale |
|---|---|---|
| **Membrane time constant (τ_m)** | 10ms | ~2x input time step; captures input dynamics |
| **Spike threshold (V_th)** | 1.0 | Normalized scale; tuned for firing rate ~20-30% |
| **Refractory period** | 2ms | Prevents unrealistic double-spikes within short intervals |
| **Number of neurons** | 64 | One per Gammatone channel; preserves frequency selectivity |
| **Input normalization** | Gammatone response clipped to [0, 1] | Ensures consistent input scaling |

---

## Component 3: Time-to-First-Spike (TTFS) Encoding (Optional)

### Overview
Convert spike times into latency-based representation (rank-order code). Optional feature used for certain loss functions in SNN training. First neuron to spike gets highest priority.

### Algorithm

**Spike latency calculation**:
```
function compute_spike_latencies(spikes, sr=16000):
    """
    For each neuron, compute time (ms) to first spike.
    
    Input:
    - spikes: (T, 64) binary spike trains
    - sr: sampling rate
    
    Output:
    - latencies: (64,) time to first spike for each neuron (ms)
    - spike_indices: (64,) indices of first spike in time
    """
    
    latencies = np.full(64, np.inf)  # Initialize to infinity
    spike_indices = np.full(64, -1, dtype=int)
    dt_ms = 1000.0 / sr  # Time per sample in ms
    
    for neuron_idx in range(64):
        spike_times = np.where(spikes[:, neuron_idx])[0]
        if len(spike_times) > 0:
            first_spike_idx = spike_times[0]
            latencies[neuron_idx] = first_spike_idx * dt_ms
            spike_indices[neuron_idx] = first_spike_idx
    
    # Rank neurons by latency (earliest spike = rank 0)
    rank_order = np.argsort(latencies)
    
    return latencies, rank_order, spike_indices
```

**Rank-order representation**:
```
Example output for 64 neurons:
latencies = [5.0, 10.0, 3.2, ∞, 7.5, ...]  (ms to first spike)
rank_order = [2, 0, 4, 1, 3, ...]  (neuron indices sorted by latency)

Interpretation:
- Neuron 2 spikes first (3.2ms) → rank 0 (highest priority)
- Neuron 0 spikes second (5.0ms) → rank 1
- Neuron 4 spikes third (7.5ms) → rank 2
- Neuron 1 spikes fourth (10.0ms) → rank 3
- Remaining neurons don't spike → rank 4+
```

### Why TTFS?
1. **Sparse**: Only active neurons contribute (many have latency=∞)
2. **Latency-based**: Temporal information preserved
3. **Noise robust**: Single spike per neuron less affected by noise
4. **Training friendly**: Natural loss for SNNs (penalize wrong order)

---

## Complete Spike Conversion Pipeline

**End-to-end algorithm**:
```
function convert_mel_to_spikes(mel_spec, tau_m=10e-3, v_th=1.0, 
                               refractory_ms=2, hop_ms=10, sr=16000):
    """
    Convert mel-spectrogram to spike trains.
    
    Input:
    - mel_spec: (T, 64) mel-spectrogram from ARCH_1
      (T ≈ 1000 frames for 10s audio at 10ms hop, values in dB [-80, 0])
    
    Output:
    - spikes: (T, 64, 1) binary spike trains at frame-level resolution
    - latencies: (64,) time to first spike (frame index)
    - rank_order: (64,) neuron indices sorted by latency
    """
    
    # Step 1: Normalize mel-spectrogram from dB scale to [0, 1]
    mel_norm = (mel_spec + 80) / 80  # Shift: [-80, 0] → [0, 1]
    mel_norm = np.clip(mel_norm, 0, 1)
    
    # Step 2: LIF neuron simulation (frame-level resolution)
    spikes_2d = lif_neurons(mel_norm, tau_m, v_th, refractory_ms, hop_ms, sr)
    
    # Step 3: Expand to (T, 64, 1) format for compatibility
    spikes = np.expand_dims(spikes_2d, axis=2)
    
    # Step 4: TTFS encoding (optional)
    latencies, rank_order, spike_indices = compute_spike_latencies(spikes_2d, hop_ms)
    
    return {
        'spikes': spikes,  # (T, 64, 1)
        'latencies': latencies,  # (64,)
        'rank_order': rank_order,  # (64,)
        'spike_indices': spike_indices,  # (64,)
        'mel_normalized': mel_norm  # (T, 64) for debugging
    }
```

---

## Output Specifications

**Spike train tensor**:
- Shape: (T, 64, 1) where T is number of mel-spectrogram frames
- Dtype: bool or uint8 (0 or 1)
- T for 10s audio: 10 × (16000 / 160) = 1000 frames (at 10ms hop from ARCH_1)
- Total size: 1000 × 64 × 1 = 64,000 values per sample (not 160,000!)

**Frame-level timing**:
- Each frame represents 10ms of audio (ARCH_1 hop_length)
- One spike per frame per neuron (binary: fired or didn't fire)
- Output directly compatible with ARCH_4 SNN training (which expects frame-level input)

**Compressed storage** (for batch SNN training):
```
function spike_to_sparse(spikes):
    """Convert dense spike tensor to sparse representation (COO format)."""
    # Remove singleton dimension for sparse conversion
    spikes_2d = np.squeeze(spikes, axis=2)
    
    # HDF5 storage of sparse spikes
    spike_frames, spike_neurons = np.where(spikes_2d[:, :])
    
    return {
        'spike_frames': spike_frames,     # Frame indices where spikes occur
        'spike_neurons': spike_neurons,   # Neuron indices
        'total_spikes': len(spike_frames),
        'shape': spikes.shape             # (T, 64, 1)
    }
```

**Storage efficiency**:
- Dense: 1000 × 64 × 1 bit ≈ 8 KB per sample
- Sparse: Depends on firing rate (typically 20-40% active) ≈ 2-4 KB per sample
- 600 samples: ~5-25 MB total (minimal storage impact)

---

## Quality Assurance

**Validation metrics**:
- Verify spike output is binary (0 or 1 only)
- Check firing rate per neuron (should be 15-40%, not 0% or 100%)
- Verify first-spike latencies are correlated with input mel-band energy
- Validate mel-bands show frequency selectivity (higher energy in relevant bands)
- Confirm refractory period prevents double-spikes (spike gaps ≥ refractory_ms)

**Example test**:
```
# Test on synthetic chirp signal
chirp_signal = scipy.signal.chirp(t, f0=100, f1=8000, t1=1.0, sr=16000)
mel_spec = extract_mel_spectrogram(chirp_signal, ...)  # ARCH_1
spikes_dict = convert_mel_to_spikes(mel_spec, ...)     # ARCH_2

# Validate:
# - Early frames should have high-freq spikes (high-CF neurons fire first)
# - Later frames should have low-freq spikes (low-CF neurons fire second)
# - Spike density should vary with chirp energy over time
```

---

## Integration with Next Stage (ARCH_4 SNN Training)

**Output format for ARCH_4**:
```python
spike_dataset = {
    'spike_trains': spikes,           # (600, T, 64, 1) - dense or sparse
    'labels': labels,                 # (600,) - class IDs
    'latencies': latencies_all,       # (600, 64) - optional TTFS encoding
    'metadata': {
        'n_neurons': 64,
        'n_frames': T,                # ~1000 for 10s audio
        'frame_hop_ms': 10,           # From ARCH_1
        'tau_m': 0.01,
        'v_th': 1.0,
        'refractory_ms': 2,
        'firing_rate_target': '20-30%'
    }
}
```

**Key differences from previous (corrected)**:
- Spike tensor is **(T_frames, 64, 1)**, not (160000, 64, 1)
- Frame-level resolution matches ARCH_1 output
- Directly compatible with ARCH_4 training pipeline

---

## Cross-References & Integration

### Pipeline Dependencies
- **Upstream**: 
  - Receives mel-spectrograms from **[ARCH_1: Audio Processing](./ARCH_1_AUDIO_PROCESSING.md)** (line 34, output format)
  - Input format specification: `(T, 64)` from ARCH_1 lines 34, 387
  
- **Downstream**: 
  - Outputs spike trains to **[ARCH_3: Dataset Preparation](./ARCH_3_DATASET_PREPARATION.md)** (line 23, "Spike Conversion (ARCH_2)")
  - Spike format: `(T, 64, 1)` at frame-level resolution (10ms per frame)

### Data Format Specifications
- **Input Format** (from ARCH_1): `(T, 64)` mel-spectrogram in dB range [-80, 0]
  - See [ARCH_1: Output Format](./ARCH_1_AUDIO_PROCESSING.md#quality-assurance) (lines 379-386)
- **Output Format**: Spike train `(T, 64, 1)` 
  - `T` = ~1000 frames for 10-second audio (matching 10ms hop from ARCH_1)
  - `64` = 64 spiking neurons (one per mel-band)
  - `1` = binary spike dimension (0 or 1)
  - **Value Range**: {0, 1} (binary spike events)
  - **Frame Duration**: 10ms (inherited from ARCH_1 hop_length=160 samples)
  - See [ARCH_3: Input Specification](./ARCH_3_DATASET_PREPARATION.md#component-1-data-sources--consolidation) (lines 23) for how spike trains are validated

### Processing Timeline
- **Day 4** (IMPLEMENTATION_SCHEDULE): Spike conversion implementation + LIF neuron testing
- **Expected Latency**: <50ms per 10-second audio sample (16ms theoretical at 10Hz frame rate)
- **Expected Sparsity**: ~20-30% neurons active per frame (tunable via firing threshold)
- **Output Size**: 1000 × 64 × 1 = 64,000 binary values per 10s audio (~8KB uncompressed)

### Key Parameters (Finalized)
| Parameter | Value | Reference | Rationale |
|-----------|-------|-----------|-----------|
| Input Range | [-80, 0] dB | Line 81 | From ARCH_1 mel-spec output |
| Normalized Range | [0, 1] | Line 81 | LIF input normalization |
| LIF Threshold | 1.0 | Line 25 | Binary spike threshold |
| LIF Decay (τ_m) | 10ms | Line 106 | Matches frame duration |
| LIF Decay (α) | exp(-1) ≈ 0.3679 | Line 107 | Frame-level discretization |
| Refractory Period | 2ms | Line 25 | Prevents repeated firing |
| Mel Bands (neurons) | 64 | Line 20 | Inherits from ARCH_1 |

### Critical Architectural Decisions
1. **No Gammatone Filterbank**: Mel-spectrogram already provides cochlear-like decomposition
   - See [Component 1: Rationale](./ARCH_2_SPIKE_CONVERSION.md#rationale-why-no-gammatone-filterbank) (lines 49-60)
2. **Frame-Level Resolution**: Spikes generated at 10ms intervals (100 Hz), not 16kHz sample rate
   - Reduces tensor size from 160,000 to ~1,000 frames (160x compression)
   - See [Data Flow](./ARCH_2_SPIKE_CONVERSION.md#data-flow) (lines 13-41)

### Related Documentation
- **SUMMARY_HIGH_LEVEL_ARCHITECTURE.md** (Week 1): Neuromorphic encoding overview
- **IMPLEMENTATION_SCHEDULE.md** (Day 4): Spike conversion tasks
- **Resources/Research_Paper_Citations.md**: References for LIF neurons, neuromorphic encoding

