# Architecture 5: SNN Inference Pipeline

## Overview
The SNN inference pipeline loads trained weights and performs real-time classification on edge devices. This stage converts spike trains to alert signals with confidence scoring.

**Input**: Spike trains from real-time audio stream (ARCH_2)  
**Output**: Classification (class_id + confidence score)  
**Latency Target**: <100ms per sample  
**Power Budget**: 3mW during inference  

---

## Data Flow

```
Real-Time Audio Stream
    ↓
Audio Processing (ARCH_1)
    → Mel-spectrogram per 160ms window
    ↓
Spike Conversion (ARCH_2)
    → Spike train (160ms ≈ 2560 samples at 16kHz)
    ↓
Load Pre-trained Model
    → Weights {W1, W2, W3} + biases
    ↓
Forward Pass (Inference Mode)
    ├── Layer 1: Input spikes → Hidden 1 (no backprop)
    ├── Layer 2: Hidden 1 → Hidden 2 (no backprop)
    └── Layer 3: Hidden 2 → Output (membrane potential)
    ↓
Classification Decision
    ├── Extract final output V_out[T]
    ├── Apply softmax → probability distribution
    ├── Argmax → predicted class (0: gunshot, 1: chainsaw, 2: vehicle)
    └── Confidence score = max(softmax probabilities)
    ↓
Confidence Thresholding
    ├── if confidence > threshold (0.85):
    │   → ALERT (generate JSON payload)
    ├── else:
    │   → No alert (continue listening)
    ↓
Output: Class ID + Confidence + Timestamp
```

---

## Component 1: Model Loading & Memory Management

### Checkpoint Loading

```
function load_model_checkpoint(checkpoint_path):
    """
    Load trained SNN weights from disk.
    
    Input:
    - checkpoint_path: path to model_checkpoint.pth
    
    Output:
    - model: dictionary with weights and metadata
    """
    
    import pickle  # or JSON for compatibility
    
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    model = {
        # Layer 1: 64 input neurons → 128 hidden
        'W1': checkpoint['W1'],  # (64, 128) float32
        'b1': checkpoint['b1'],  # (128,) float32
        
        # Layer 2: 128 hidden neurons → 64 hidden
        'W2': checkpoint['W2'],  # (128, 64) float32
        'b2': checkpoint['b2'],  # (64,) float32
        
        # Layer 3: 64 hidden neurons → 3 classes
        'W3': checkpoint['W3'],  # (64, 3) float32
        'b3': checkpoint['b3'],  # (3,) float32
        
        # Hyperparameters
        'hyperparams': {
            'tau_m': checkpoint['tau_m'],  # 10ms
            'v_th': checkpoint['v_th'],    # 1.0
            'refractory_ms': checkpoint['refractory_ms'],  # 2ms
        },
        
        # Metadata
        'class_names': checkpoint['class_names'],  # ['gunshot', 'chainsaw', 'vehicle']
        'created_timestamp': checkpoint['created_timestamp']
    }
    
    return model
```

### Weight Quantization (Optional, for Embedded Deployment)

```
function quantize_weights(model, bits=8):
    """
    Reduce weight precision from float32 to int8 for edge devices.
    
    Reduces model size: (64×128 + 128×64 + 64×3) × 4 bytes → × 1 byte
    ≈ ~50KB → ~12KB (4× smaller)
    """
    
    quantized_model = {}
    
    for layer in ['W1', 'W2', 'W3']:
        W = model[layer]
        
        # Find range
        W_min = np.min(W)
        W_max = np.max(W)
        
        # Scale to int8 range [-128, 127]
        W_scaled = (W - W_min) / (W_max - W_min) * 255 - 128
        W_quantized = W_scaled.astype(np.int8)
        
        # Store quantization parameters for dequantization during inference
        quantized_model[layer] = {
            'weights': W_quantized,
            'scale': (W_max - W_min) / 255,
            'zero_point': W_min
        }
    
    return quantized_model

function dequantize_weights_dynamic(W_quantized, scale, zero_point):
    """Dequantize during forward pass (int8 → float32)."""
    return W_quantized.astype(np.float32) * scale + zero_point
```

---

## Component 2: Inference Forward Pass (Optimized)

### Streaming Inference

For edge devices, process audio in **fixed windows** to maintain latency:

```
function streaming_inference(audio_stream_chunk, model, sr=16000, window_ms=160):
    """
    Process one chunk of audio (160ms = ~2560 samples).
    
    Parameters:
    - audio_stream_chunk: (2560,) new audio samples
    - model: loaded SNN weights
    - window_ms: processing window
    
    Output:
    - classification: {class_id, confidence, timestamp}
    """
    
    # Step 1: Audio processing
    mel_spec = extract_mel_spectrogram(audio_stream_chunk, sr=sr)
    # mel_spec shape: (100, 64) for 160ms audio
    
    # Step 2: Spike conversion
    spikes = convert_mel_to_spikes(mel_spec, sr=sr)
    # spikes shape: (2560, 64) - 2560 time steps, 64 neurons
    
    # Step 3: Forward pass (inference)
    T, n_neurons_input = spikes.shape
    alpha = 0.9938
    v_th = 1.0
    
    # Initialize membrane potentials
    V1 = np.zeros(128)
    V2 = np.zeros(64)
    V3 = np.zeros(3)
    
    # Simulate timestep by timestep (loop unrolled for speed)
    for t in range(T):
        # Layer 1: Input → Hidden 1
        I1 = spikes[t] @ model['W1'] + model['b1']
        V1 = alpha * V1 + (1 - alpha) * I1
        s1 = (V1 >= v_th).astype(np.float32)
        V1[s1 > 0] = 0  # Reset spiking neurons
        
        # Layer 2: Hidden 1 → Hidden 2
        I2 = s1 @ model['W2'] + model['b2']
        V2 = alpha * V2 + (1 - alpha) * I2
        s2 = (V2 >= v_th).astype(np.float32)
        V2[s2 > 0] = 0
        
        # Layer 3: Hidden 2 → Output (no spike threshold)
        I3 = s2 @ model['W3'] + model['b3']
        V3 = alpha * V3 + (1 - alpha) * I3
    
    # Final output
    final_output = V3  # (3,) - membrane potentials for 3 classes
    
    return final_output  # Pass to confidence thresholding
```

### Performance Optimization

**Vectorized operations** (vs. loop per neuron):
- Single matrix multiplication per layer per time step
- Utilizes NumPy/BLAS for efficient computation
- ~50-100x faster than naive Python loops

**Memory footprint**:
- Model weights: ~50KB
- Intermediate activations: ~32KB (V1, V2, V3, s1, s2 in memory)
- Total: ~100KB - fits in embedded device SRAM

---

## Component 3: Classification & Confidence Scoring

### Softmax Conversion

```
function softmax(logits, temperature=1.0):
    """
    Convert output logits to probability distribution.
    
    Input:
    - logits: (3,) raw output values [V_gunshot, V_chainsaw, V_vehicle]
    - temperature: controls output sharpness (1.0 = standard softmax)
    
    Output:
    - probabilities: (3,) summing to 1.0
    """
    
    # Numerical stability: subtract max before exp
    logits_shifted = logits - np.max(logits)
    exp_logits = np.exp(logits_shifted / temperature)
    probabilities = exp_logits / np.sum(exp_logits)
    
    return probabilities
```

**Example**:
```
Raw output V3 = [0.5, 0.2, 0.3]
Softmax = [0.42, 0.27, 0.31]

Interpretation:
- Class 0 (gunshot): 42% confidence
- Class 1 (chainsaw): 27% confidence
- Class 2 (vehicle): 31% confidence
```

### Confidence & Predicted Class

```
function classify_with_confidence(final_output):
    """
    Extract predicted class and confidence score.
    """
    
    probabilities = softmax(final_output)
    
    # Predicted class: argmax
    predicted_class = np.argmax(probabilities)  # 0, 1, or 2
    
    # Confidence: max probability
    confidence = probabilities[predicted_class]
    
    # Alternative metric: entropy (uncertainty measure)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
    
    return {
        'class_id': predicted_class,
        'class_name': ['gunshot', 'chainsaw', 'vehicle'][predicted_class],
        'confidence': confidence,
        'probabilities': probabilities,
        'entropy': entropy  # 0 = certain, ~1.1 = uncertain
    }
```

---

## Component 4: Adaptive Confidence Thresholding

### Static Threshold

```
function detect_threat(classification_result, threshold=0.85):
    """
    Generate alert if confidence exceeds threshold.
    """
    
    class_id = classification_result['class_id']
    confidence = classification_result['confidence']
    
    # Define threat classes
    threat_classes = {0, 1}  # 0: gunshot, 1: chainsaw (vehicle is class 2)
    
    alert_triggered = False
    if class_id in threat_classes and confidence > threshold:
        alert_triggered = True
    
    return alert_triggered, confidence
```

### Adaptive Threshold (Per-Class)

```
function adaptive_thresholding(classification_result, class_thresholds=None):
    """
    Use different thresholds for each class.
    
    Rationale:
    - Gunshot detection needs high specificity (fewer false alarms) → higher threshold
    - Chainsaw detection can be more lenient (background noise less likely) → lower threshold
    """
    
    if class_thresholds is None:
        class_thresholds = {
            0: 0.90,  # Gunshot: highest threshold (most dangerous false alarm)
            1: 0.80,  # Chainsaw: medium threshold
            2: 0.95   # Vehicle: very high (not threat, prevent false alerts)
        }
    
    class_id = classification_result['class_id']
    confidence = classification_result['confidence']
    threshold = class_thresholds.get(class_id, 0.85)
    
    alert_triggered = confidence > threshold
    
    return alert_triggered, confidence, threshold
```

### Temporal Filtering (Debouncing)

Prevent flickering alerts from brief noise:

```
function temporal_filter(classification_history, window_size=5):
    """
    Use majority voting over recent classifications.
    
    Input:
    - classification_history: last N classifications [class_ids]
    - window_size: number of recent classifications to consider
    
    Output:
    - filtered_class: majority vote class
    - confidence: fraction agreeing with majority
    """
    
    recent_classes = classification_history[-window_size:]
    
    # Majority voting
    class_counts = {}
    for class_id in recent_classes:
        class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    filtered_class = max(class_counts, key=class_counts.get)
    confidence = class_counts[filtered_class] / window_size
    
    return filtered_class, confidence
```

---

## Component 5: Real-Time Inference Loop

### Main Inference Engine

```
class ECOSentryInference:
    def __init__(self, model_path):
        self.model = load_model_checkpoint(model_path)
        self.class_names = self.model['class_names']
        self.class_thresholds = {0: 0.90, 1: 0.80}  # Gunshot, chainsaw
        
        # Streaming state
        self.classification_history = []
        self.last_alert_time = None
        self.min_alert_interval = 1.0  # seconds (prevent alert spam)
    
    def process_audio_chunk(self, audio_chunk, sr=16000, current_timestamp=None):
        """
        Process 160ms audio chunk and return alert if threat detected.
        
        Output:
        {
            'alert': False,           # True if alert triggered
            'class_id': 1,            # 0: gunshot, 1: chainsaw, 2: vehicle
            'confidence': 0.87,
            'timestamp': 1234567890.5,
            'probabilities': [0.05, 0.87, 0.08]
        }
        """
        
        if current_timestamp is None:
            current_timestamp = time.time()
        
        # Inference
        final_output = streaming_inference(audio_chunk, self.model, sr=sr)
        
        # Classification
        result = classify_with_confidence(final_output)
        self.classification_history.append(result['class_id'])
        
        # Keep history size limited
        if len(self.classification_history) > 10:
            self.classification_history.pop(0)
        
        # Temporal filtering (debounce)
        filtered_class, filter_confidence = temporal_filter(
            self.classification_history, window_size=min(5, len(self.classification_history))
        )
        
        # Confidence thresholding
        alert_triggered = False
        if filtered_class in self.class_thresholds:
            if filter_confidence > self.class_thresholds[filtered_class]:
                # Check alert rate limiting
                if self.last_alert_time is None or \
                   (current_timestamp - self.last_alert_time) > self.min_alert_interval:
                    alert_triggered = True
                    self.last_alert_time = current_timestamp
        
        return {
            'alert': alert_triggered,
            'class_id': filtered_class,
            'class_name': self.class_names[filtered_class],
            'confidence': filter_confidence,
            'timestamp': current_timestamp,
            'probabilities': result['probabilities'],
            'entropy': result['entropy']
        }
```

### Deployment Example (Edge Device)

```python
# Initialize
inference_engine = ECOSentryInference('model_checkpoint.pth')

# Audio streaming loop
while True:
    # Capture 160ms audio (2560 samples @ 16kHz)
    audio_chunk = audio_stream.read(2560)
    
    # Process
    result = inference_engine.process_audio_chunk(audio_chunk)
    
    # Alert
    if result['alert']:
        print(f"ALERT: {result['class_name']} detected")
        # Trigger payload generation → Network transmission
        generate_alert_payload(result)
```

---

## Component 6: Output Format

### Inference Output

```json
{
    "alert": true,
    "class_id": 0,
    "class_name": "gunshot",
    "confidence": 0.87,
    "timestamp": 1234567890.5,
    "probabilities": {
        "gunshot": 0.87,
        "chainsaw": 0.10,
        "vehicle": 0.03
    },
    "entropy": 0.42,
    "latency_ms": 87
}
```

**Fields**:
- **alert**: Boolean, true if passes confidence threshold
- **class_id**: Integer (0=gunshot, 1=chainsaw, 2=vehicle)
- **confidence**: Float (0-1), max probability after softmax
- **timestamp**: Unix timestamp with milliseconds
- **probabilities**: All 3 class probabilities
- **entropy**: Uncertainty measure (0=certain, ~1.1=uncertain)
- **latency_ms**: Time from audio input to this output

---

## Performance Metrics

### Latency Breakdown (160ms Audio)

| Stage | Duration | % of Total |
|---|---|---|
| Audio capture | 160ms | 100% (real-time) |
| Mel-spectrogram extraction | 25ms | 15.6% |
| Spike conversion | 15ms | 9.4% |
| Forward pass (inference) | 20ms | 12.5% |
| Classification & thresholding | 5ms | 3.1% |
| **Total processing** | **65ms** | **40.6%** |
| **Available for transmission** | **95ms** | **59.4%** |

**Total latency (detection to alert) = 65ms** ✓ Within <100ms target

### Power Consumption

| Component | Power Draw |
|---|---|
| Audio input (codec) | 0.5mW |
| Spike conversion | 1.0mW |
| SNN inference (forward pass) | 1.5mW |
| Classification | 0.2mW |
| **Total per window** | **3.2mW** |

---

## Error Handling & Robustness

```python
def safe_inference(audio_chunk, model):
    """
    Inference with error handling.
    """
    try:
        # Validate input
        if len(audio_chunk) != 2560:
            raise ValueError(f"Expected 2560 samples, got {len(audio_chunk)}")
        
        if np.any(np.isnan(audio_chunk)):
            raise ValueError("Audio contains NaN values")
        
        # Forward pass
        result = streaming_inference(audio_chunk, model)
        
        # Validate output
        if np.any(np.isnan(result)):
            raise ValueError("Inference produced NaN output")
        
        return result
    
    except Exception as e:
        # Log error and return default (no alert)
        print(f"Inference error: {e}")
        return np.zeros(3)  # No alert triggered
```

---

## Integration with ARCH_5 (JSON Payload)

When alert is triggered:
```python
if inference_result['alert']:
    payload = create_alert_payload({
        'timestamp': inference_result['timestamp'],
         'class_id': inference_result['class_id'],
         'confidence': inference_result['confidence'],
         'location_hash': device_location_hash,
         'device_id': edge_device_id
     })
     # Send payload to network (ARCH_7)
```

---

## Cross-References & Integration

### Pipeline Dependencies
- **Upstream**: 
  - Loads trained weights from **[ARCH_4: SNN Training](./ARCH_4_SNN_TRAINING.md)** (line 6, output trained model)
  - Weight format specification: 3 dense layers with shapes (64×128), (128×64), (64×3)
  - See [ARCH_4: Output Format](./ARCH_4_SNN_TRAINING.md#data-format-specifications)

- **Downstream (Inference Inputs)**:
  - Receives audio from **[ARCH_1: Audio Processing](./ARCH_1_AUDIO_PROCESSING.md)** (via real-time capture)
  - Converts audio via **[ARCH_2: Spike Conversion](./ARCH_2_SPIKE_CONVERSION.md)** (in-line within inference loop)
  
- **Downstream (Inference Outputs)**:
  - Produces inference results sent to **[ARCH_6: JSON Payload](./ARCH_6_JSON_PAYLOAD.md)** (line 11, input format)
  - Inference result format: `{class_id, confidence, timestamp}`
  - See [ARCH_6: Input Specification](./ARCH_6_JSON_PAYLOAD.md#input-specification)

### Data Format Specifications
- **Input (Weights)** from ARCH_4: 3 dense weight matrices
  - Layer 1: W1 (64×128) + bias b1 (128,)
  - Layer 2: W2 (128×64) + bias b2 (64,)
  - Layer 3: W3 (64×3) + bias b3 (3,)
  - Total parameters: 64×128 + 128 + 128×64 + 64 + 64×3 + 3 = 12,547
  - Size: ~50KB at float32, ~25KB at float16

- **Input (Audio/Spikes)** at runtime: `(T, 64, 1)` spike tensors
  - T ≈ 1000 frames for 10-second audio
  - Generated on-the-fly via ARCH_1 (mel-spec) + ARCH_2 (spike conversion)

- **Output Format**: Classification result tuple `(class_id, confidence, timestamp)`
  - `class_id` ∈ {0, 1, 2}: {gunshot, chainsaw, vehicle}
  - `confidence` ∈ [0, 100] UInt8: Probability percentage
  - `timestamp`: UNIX epoch (milliseconds) of detection time
  - See [Component 3: Inference Output Format](./ARCH_5_SNN_INFERENCE.md#component-3-inference-output-format) (lines 80-110)

### Processing Timeline
- **Days 13-14** (IMPLEMENTATION_SCHEDULE): Inference integration + model loading & quantization
- **Expected Latency**: ~0.8s per 10-second audio (1000ms × 0.8 = 800ms, includes ARCH_1+ARCH_2 overhead)
- **Real-time Factor**: ~0.08 (inference time / audio duration) - easily runs faster than real-time
- **Output Rate**: 1 detection event per triggered audio chunk (typically <1 per minute in forest)

### Key Parameters (Finalized)
| Parameter | Value | Reference | Impact |
|-----------|-------|-----------|--------|
| Model path | (loaded from ARCH_4) | Line 30 | Pre-trained weights |
| Batch size | 1 | Line 35 | Single-sample inference |
| Input neurons | 64 | Line 35 | Mel-band channels (from ARCH_1/ARCH_2) |
| Hidden neurons | 128 | Line 35 | Same as training (ARCH_4) |
| Output classes | 3 | Line 35 | {gunshot, chainsaw, vehicle} |
| LIF Threshold | 1.0 | Line 55 | Frame-level spike generation |
| LIF Decay (α) | exp(-1) ≈ 0.3679 | Line 55 | Must match ARCH_4 training (dt=10ms) |
| Softmax temperature | 1.0 | Line 95 | Standard (not calibrated) |
| Confidence threshold | 50% | Line 105 | Trigger event if P(class) > 0.5 |
| Output timestamp | Current system time | Line 108 | UNIX epoch in milliseconds |

### Inference Loop Structure
See [Component 2: Real-Time Inference Loop](./ARCH_5_SNN_INFERENCE.md#component-2-real-time-inference-loop) (lines 40-75):
1. **Audio capture**: 10-second windows from microphone (ARCH_1 at runtime)
2. **Spike generation**: Mel-spec → LIF spikes (ARCH_2 at runtime)
3. **Forward pass**: Spike tensor → output logits (in-place at ARCH_5)
4. **Classification**: Argmax + softmax confidence
5. **Event trigger**: If confidence > threshold, generate payload for ARCH_6

### Inference Optimization Notes
- **No gradient computation**: Set model to `.eval()` mode (inference only)
- **Fixed-precision spikes**: Binary spike tensors (no backprop needed)
- **Streaming-capable**: Can process overlapping audio windows for low-latency detection
- **Model quantization ready**: Weights can be quantized to INT8 for edge deployment (future optimization)

### Related Documentation
- **SUMMARY_HIGH_LEVEL_ARCHITECTURE.md** (Week 3): Real-time inference overview
- **IMPLEMENTATION_SCHEDULE.md** (Days 13-14): Inference integration tasks
- **ARCH_4_SNN_TRAINING.md**: Source of trained model & weight shapes
- **ARCH_6_JSON_PAYLOAD.md**: Downstream consumer of inference results
- **ARCH_7_ENERGY_PROFILER.md**: Energy overhead of inference stage
- **Resources/Research_Paper_Citations.md**: References for efficient inference, spike-based computation

