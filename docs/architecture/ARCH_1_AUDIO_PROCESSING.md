# Architecture 1: Audio Processing Pipeline

## Overview
The audio processing pipeline converts raw forest audio recordings into mel-scaled spectrograms suitable for spike conversion. This is the first stage of the Eco-Sentry SNN pipeline, owned by Abhishek M.

**Input**: Raw audio files (gunshots, chainsaws, vehicles, ambient sounds)  
**Output**: Mel-spectrogram tensors (64 mel bands × T time frames)  
**Latency Target**: <100ms per sample  
**Power Budget**: 2mW average consumption

---

## Data Flow

```
Raw Audio File (WAV/MP3)
    ↓
Load & Resample (if needed)
    ↓
Normalize (RMS normalization)
    ↓
Fade In/Out (avoid clicks)
    ↓
Bandpass Filter (100Hz - 8kHz)
    ↓
Frame into overlapping windows
    ↓
Compute STFT (Short-Time Fourier Transform)
    ↓
Convert to Mel-scale
    ↓
Log-scale (dB conversion)
    ↓
Output: Mel-spectrogram (64, T)
```

---

## Processing Pipeline - Detailed Steps

### Step 1: Audio Loading & Resampling
**Purpose**: Load raw audio and ensure consistent sampling rate  
**Requirement**: Support multiple formats (WAV, MP3, FLAC)

**Algorithm**:
```
function load_audio(filepath, target_sr=16000):
    audio_data, native_sr = librosa.load(filepath, sr=None)
    if native_sr != target_sr:
        audio_data = librosa.resample(audio_data, orig_sr=native_sr, target_sr=target_sr)
    return audio_data, target_sr

Parameters:
- target_sr = 16000 Hz (standard for gunshot detection)
- Maximum duration = 10 seconds (enough for gunshot + chainsaw)
```

**Why this matters**: Different devices record at different sample rates. 16kHz is standard for audio classification and provides sufficient frequency coverage (Nyquist limit = 8kHz).

---

### Step 2: RMS Normalization
**Purpose**: Standardize audio amplitude across different devices and recording conditions

**Algorithm**:
```
function normalize_rms(audio, target_rms=0.1):
    rms_value = sqrt(mean(audio^2))
    if rms_value > 0:
        audio_normalized = audio * (target_rms / rms_value)
    return audio_normalized

Parameters:
- target_rms = 0.1 (prevents clipping while maintaining headroom)
```

**Formula**:
```
RMS = sqrt((1/N) * Σ|x[n]|²)  where x[n] is audio sample n
normalized_x[n] = x[n] * (target_RMS / RMS)
```

**Why this matters**: Gun shots can have vastly different recording levels depending on distance and microphone. RMS normalization equalizes these, improving consistency for ML model.

---

### Step 3: Fade In/Out
**Purpose**: Remove sudden amplitude changes at audio boundaries that create artifacts in spectrogram

**Algorithm**:
```
function fade_audio(audio, fade_length_ms=50, sr=16000):
    fade_samples = int(fade_length_ms * sr / 1000)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    audio[:fade_samples] *= fade_in
    audio[-fade_samples:] *= fade_out
    return audio

Parameters:
- fade_length_ms = 50 (typical for audio processing)
```

**Why this matters**: Sudden amplitude changes cause ringing artifacts in frequency domain (spectral leakage).

---

### Step 4: Bandpass Filter
**Purpose**: Remove frequencies outside gunshot/chainsaw range, reduce wind noise and low-frequency rumble

**Algorithm** (Butterworth IIR filter):
```
function bandpass_filter(audio, low_freq=100, high_freq=8000, sr=16000, order=4):
    nyquist = sr / 2
    low_normalized = low_freq / nyquist
    high_normalized = high_freq / nyquist
    
    b, a = scipy.signal.butter(order, [low_normalized, high_normalized], btype='band')
    filtered = scipy.signal.filtfilt(b, a, audio)
    return filtered

Parameters:
- low_freq = 100 Hz (removes DC offset and very low rumble)
- high_freq = 8000 Hz (Nyquist limit at 16kHz SR)
- order = 4 (2nd order = 12dB/octave rolloff)
```

**Why this matters**: Gunshots have energy in 100-8000Hz range. Wind noise is <100Hz, high-frequency noise often is not relevant for detection.

---

### Step 5: Frame Windowing
**Purpose**: Divide continuous audio into overlapping frames for STFT computation

**Algorithm**:
```
function frame_audio(audio, frame_length_ms=32, hop_length_ms=10, sr=16000):
    frame_samples = int(frame_length_ms * sr / 1000)     # 512 samples at 16kHz
    hop_samples = int(hop_length_ms * sr / 1000)         # 160 samples at 16kHz
    
    # Pad audio to ensure we capture all content
    pad_length = frame_samples
    audio_padded = np.pad(audio, (pad_length, pad_length), mode='reflect')
    
    frames = []
    for i in range(0, len(audio_padded) - frame_samples, hop_samples):
        frame = audio_padded[i:i+frame_samples]
        frames.append(frame)
    
    return np.array(frames)  # Shape: (num_frames, frame_samples)

Parameters:
- frame_length = 32ms (window size)
- hop_length = 10ms (stride)
- Overlap ratio = (32 - 10) / 32 = 68.75% (standard)
```

**Output dimensions**:
- For 10s audio: (10 * 1000 - 32) / 10 + 1 ≈ 1000 frames

**Why this matters**: STFT requires fixed-size windows. 32ms captures gunshot transients while 10ms hop provides temporal resolution.

---

### Step 6: Short-Time Fourier Transform (STFT)
**Purpose**: Convert time-domain audio frames into frequency-domain representation

**Algorithm**:
```
function compute_stft(frames, window_type='hann', n_fft=512):
    # Apply window function to each frame
    window = scipy.signal.get_window(window_type, len(frames[0]))
    windowed_frames = frames * window
    
    # Compute FFT for each frame
    stft_matrix = []
    for frame in windowed_frames:
        # Zero-pad to n_fft length
        padded = np.pad(frame, (0, n_fft - len(frame)), mode='constant')
        fft = np.fft.rfft(padded)
        magnitude = np.abs(fft)
        stft_matrix.append(magnitude)
    
    return np.array(stft_matrix)  # Shape: (num_frames, n_fft//2 + 1)

Parameters:
- window_type = 'hann' (Hann window reduces spectral leakage)
- n_fft = 512 (frequency resolution = 16000/512 ≈ 31Hz)
```

**Formula**:
```
STFT[k, m] = Σ x[n] * w[n - m*hop] * exp(-j*2π*k*n/N)
where:
  k = frequency bin (0 to N/2)
  m = frame index
  N = FFT size
  w = window function
```

**Output dimensions**:
- Input frames: (1000, 512)
- Output STFT: (1000, 257) - 512/2 + 1 frequency bins

---

### Step 7: Mel-Scale Conversion
**Purpose**: Convert linear frequency scale to perceptually-motivated mel-scale, mimicking human hearing

**Algorithm**:
```
function hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

function mel_to_hz(mel):
    return 700 * (10^(mel / 2595) - 1)

function create_mel_filterbank(n_mels=64, sr=16000, n_fft=512, f_min=50, f_max=8000):
    nyquist = sr / 2
    
    # Create mel-spaced frequencies
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    
    # Convert to FFT bin indices
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    
    # Create triangular filters
    filterbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(n_mels):
        left = bin_points[m]
        center = bin_points[m + 1]
        right = bin_points[m + 2]
        
        # Left slope (0 to 1)
        filterbank[m, left:center] = np.linspace(0, 1, center - left)
        # Right slope (1 to 0)
        filterbank[m, center:right] = np.linspace(1, 0, right - center)
    
    return filterbank

function stft_to_mel(stft_magnitude, mel_filterbank):
    mel_spec = np.dot(mel_filterbank, stft_magnitude.T)
    return mel_spec.T  # Shape: (num_frames, n_mels)

Parameters:
- n_mels = 64 (number of mel bands - chosen for gunshot frequency distribution)
- f_min = 50 Hz
- f_max = 8000 Hz (Nyquist at 16kHz)
```

**Mel-scale formula**:
```
mel = 2595 * log₁₀(1 + f/700)
```

**Why this matters**: Mel-scale matches human perception of pitch. Lower frequencies have higher resolution (e.g., 100-200Hz separation) while higher frequencies are compressed (e.g., 7-8kHz treated as similar).

---

### Step 8: Log-Scale Conversion (dB)
**Purpose**: Convert magnitude spectrogram to logarithmic dB scale, matching human perception of loudness

**Algorithm**:
```
function magnitude_to_db(mel_spectrogram, ref_power=1.0, amin=1e-5):
    # Ensure non-zero values to avoid log(0)
    mel_spec_safe = np.maximum(mel_spectrogram, amin)
    
    # Convert to dB (power, not amplitude)
    db_spec = 10 * np.log10(mel_spec_safe / ref_power)
    
    return db_spec

Parameters:
- ref_power = 1.0 (reference power for dB calculation)
- amin = 1e-5 (minimum amplitude floor, prevents -inf values)

Optional normalization (standardization):
function normalize_db_spec(db_spec):
    mean = np.mean(db_spec)
    std = np.std(db_spec)
    return (db_spec - mean) / (std + 1e-8)  # +1e-8 prevents division by zero
```

**Formula**:
```
dB = 10 * log₁₀(P / P_ref)
where P is power (magnitude²) and P_ref is reference power
```

**Output range**: Typically -80dB to 0dB for normalized audio

---

## Complete Mel-Spectrogram Extraction

**Combined Algorithm**:
```
function extract_mel_spectrogram(audio_path, sr=16000, n_mels=64, n_fft=512, 
                                  hop_ms=10, frame_ms=32, f_min=50, f_max=8000):
    # Load and prepare
    audio, sr = load_audio(audio_path, sr)
    audio = normalize_rms(audio, target_rms=0.1)
    audio = fade_audio(audio, fade_length_ms=50, sr=sr)
    audio = bandpass_filter(audio, low_freq=f_min, high_freq=f_max, sr=sr)
    
    # Frame
    hop_samples = int(hop_ms * sr / 1000)
    frame_samples = int(frame_ms * sr / 1000)
    frames = frame_audio(audio, frame_ms, hop_ms, sr)
    
    # STFT
    stft_matrix = compute_stft(frames, window_type='hann', n_fft=n_fft)
    
    # Mel conversion
    mel_filterbank = create_mel_filterbank(n_mels, sr, n_fft, f_min, f_max)
    mel_spec = stft_to_mel(stft_matrix, mel_filterbank)
    
    # Log-scale
    mel_spec_db = magnitude_to_db(mel_spec)
    mel_spec_db = normalize_db_spec(mel_spec_db)  # Standardize
    
    return mel_spec_db  # Shape: (num_frames, 64)

Typical output for 1s audio: (100, 64) - 100 frames × 64 mel bands
Typical output for 10s audio: (1000, 64) - 1000 frames × 64 mel bands
```

---

## Requirements Summary

| Requirement | Value | Rationale |
|---|---|---|
| **Input Sample Rate** | 16000 Hz | Standard for speech/gunshot detection; provides 8kHz Nyquist |
| **Frame Length** | 32 ms | Captures gunshot transient (typical ~20-50ms) |
| **Hop Length** | 10 ms | Provides 68.75% overlap for smooth temporal transitions |
| **FFT Size** | 512 | Frequency resolution = 31.25Hz; balance between time/freq |
| **Mel Bands** | 64 | Standard for sound classification; covers 50Hz-8kHz range |
| **Min Frequency** | 50 Hz | Removes DC and wind noise |
| **Max Frequency** | 8000 Hz | Nyquist limit for 16kHz sample rate |
| **Normalization** | RMS + dB standardization | Handles device/distance variations in gunshot recordings |
| **Latency** | <100ms per sample | Enables real-time processing on edge device |
| **Output Shape** | (T, 64) where T ≈ 100-1000 | Depends on audio duration; fixed frequency dimension |

---

## Power Consumption

**Estimated power by operation** (on Cortex-M4 @ 100MHz):
- Load audio: 0.5mW (mostly SD card I/O)
- Normalize & filter: 0.3mW
- Frame & STFT: 0.8mW (FFT is compute-heavy)
- Mel conversion & dB: 0.4mW
- **Total average**: ~2mW

---

## Storage & Format

**HDF5 Dataset Format** (for batch processing):
```
dataset/
├── mel_specs/          (shape: 600 × 1000 × 64, dtype: float32)
├── labels/             (shape: 600, dtype: int32) - 0:gunshot, 1:chainsaw, 2:vehicle, 3:ambient
├── sample_rates/       (shape: 600, dtype: int32) - all 16000
├── durations_ms/       (shape: 600, dtype: int32) - duration of each sample
└── metadata.json       (sample sources, creation date, preprocessing params)

File size for 600 samples: ~154MB (600 × 1000 × 64 × 4 bytes)
```

---

## Quality Assurance

**Validation Metrics**:
- Verify no NaN or Inf values in output
- Check output shape matches expectations (num_frames, 64)
- Verify dB range is within [-80, 0] dB
- Confirm mel bands are perceptually spaced (validate filter bank coverage)
- Spot-check spectrograms visually (should show noise peaks for gunshots)

---

## Cross-References & Integration

### Pipeline Dependencies
- **Upstream**: None - this is the entry point to the Eco-Sentry pipeline
- **Downstream**: 
  - Outputs feed directly to **[ARCH_2: Spike Conversion](./ARCH_2_SPIKE_CONVERSION.md)** (line 6, "Input: Mel-spectrogram (T, 64) from ARCH_1")

### Data Format Specifications
- **Output Format**: Mel-spectrogram tensor `(T, 64)`
  - `T` = number of time frames (e.g., ~1000 for 10-second audio at 10ms hop_length)
  - `64` = number of mel-frequency bands (50Hz to 8kHz perceptual scale)
  - **Value Range**: [-80, 0] dB (log-magnitude)
  - **Frame Duration**: 10ms (hop_length = 160 samples at 16kHz SR)
  - See [ARCH_2: Input Normalization](./ARCH_2_SPIKE_CONVERSION.md#mel-band-as-lif-input) (line 71) for how this is normalized for LIF neurons

### Processing Timeline
- **Day 1** (IMPLEMENTATION_SCHEDULE): Audio codec setup + librosa integration
- **Day 2** (IMPLEMENTATION_SCHEDULE): RMS normalization + fade testing
- **Day 3** (IMPLEMENTATION_SCHEDULE): STFT + Mel-scale implementation
- **Expected Latency**: <100ms per 10-second audio sample
- **Expected Throughput**: Process 3600 seconds of audio per hour (at real-time playback speed)

### Key Parameters (Finalized)
| Parameter | Value | Reference |
|-----------|-------|-----------|
| Sample Rate | 16 kHz | Line 54 |
| Frame Length (window) | 512 samples | Line 115 |
| Hop Length | 160 samples (10ms) | Line 116 |
| Mel Bands | 64 | Line 135 |
| Frequency Range | 50–8000 Hz | Line 137 |
| Power Basis | dB (log10) | Line 160 |
| Normalization | RMS to 0.1 | Line 74 |

### Related Documentation
- **SUMMARY_HIGH_LEVEL_ARCHITECTURE.md** (Week 1): Overview of all 8 architectures and their roles
- **IMPLEMENTATION_SCHEDULE.md** (Days 1–3): Detailed implementation tasks for ARCH_1
- **Resources/Research_Paper_Citations.md**: References for mel-scale motivation and Gammatone vs. mel-spectrogram tradeoffs

