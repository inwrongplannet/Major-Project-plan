# Architecture 4: SNN Training Pipeline

## Overview
The SNN training pipeline converts spike trains into a trained classifier using surrogate gradient descent. This stage bridges neuromorphic spike generation and inference, owned by Abhishek M.

**Input**: Spike trains from ARCH_3 (1200 augmented samples, 64 neurons each) 
**Output**: Trained SNN weights + inference model  
**Training duration**: ~2-3 hours on GPU  
**Accuracy target**: >85% on 3-class problem (gunshot, chainsaw, vehicle)

---

## Data Flow

```
Spike Dataset from ARCH_3 (720 samples after augmentation, 64 neurons, ~1000 time steps each)
    ↓
Train/Val/Test Split (60/20/20 on augmented data)
    ├── Training: 720 samples
    ├── Validation: 240 samples
    └── Test: 240 samples
    ↓
SNN Architecture Definition
    ├── Layer 1: 64 input neurons → 128 hidden neurons
    ├── Layer 2: 128 hidden neurons → 64 hidden neurons
    └── Output: 64 neurons → 3 class outputs (gunshot, chainsaw, vehicle)
    ↓
Forward Pass (BPTT + Surrogate Gradients)
    ├── Leaky integrate-and-fire dynamics
    ├── Spike output at each time step
    └── Loss computation
    ↓
Backward Pass (Surrogate Gradient Descent)
    ├── Compute gradients using surrogate function
    ├── Update weights (Adam optimizer)
    └── Clipping & normalization
    ↓
Validation & Hyperparameter Tuning
    ├── Per-epoch accuracy tracking
    ├── Learning rate decay
    └── Early stopping on validation loss
    ↓
Trained Model (weights + state dict)
```

---

## Component 1: SNN Architecture

### Network Structure

**2-layer SNN for 3-class classification**:

```
Input: Spike train (T, 64)
  ↓ Fully connected layer 1 (weight matrix 64×128)
Spiking Hidden Layer 1 (T, 128 neurons)
  ├─ LIF dynamics (τ_m = 10ms)
  ├─ Spike threshold V_th = 1.0
  └─ Output: spikes h₁[t] ∈ {0, 1}¹²⁸
  ↓ Fully connected layer 2 (weight matrix 128×64)
Spiking Hidden Layer 2 (T, 64 neurons)
  ├─ LIF dynamics (τ_m = 10ms)
  ├─ Spike threshold V_th = 1.0
  └─ Output: spikes h₂[t] ∈ {0, 1}⁶⁴
  ↓ Fully connected layer 3 (weight matrix 64×3)
Output Layer (T, 3 neurons)
  ├─ LIF dynamics (NO spike threshold on output)
  ├─ Membrane potential integrated over time
  └─ Output: V_out[T] ∈ ℝ³ (real-valued, not spiking)
```

### Layer Design Rationale

| Layer | Input Size | Output Size | Type | Rationale |
|---|---|---|---|---|
| **Input** | 64 neurons | - | Spike input from Gammatone | Cochlear encoding |
| **Hidden 1** | 64 | 128 | Spiking LIF | 2× expansion for feature extraction |
| **Hidden 2** | 128 | 64 | Spiking LIF | Dimensionality reduction |
| **Output** | 64 | 3 | Non-spiking (integrated) | Stable rate-based readout |

### Weight Initialization

**Xavier initialization** (prevents saturation):
```
function initialize_weights(input_size, output_size):
    limit = sqrt(6.0 / (input_size + output_size))
    W = random.uniform(-limit, limit, size=(input_size, output_size))
    return W
```

**Bias initialization**:
```
Biases initialized to small positive values (0.01) to encourage initial spiking
b = np.ones(output_size) * 0.01
```

---

## Component 2: Forward Pass (Inference during Training)

### LIF Forward Dynamics

**For each time step t and each neuron i**:

```
Current input to neuron i:
I_i[t] = Σ_j W_ij * x_j[t] + b_i

Leaky integration (LIF):
V_i[t+1] = α * V_i[t] + (1 - α) * I_i[t]

where:
α = exp(-dt / τ_m)  [dt = frame interval = 10ms, τ_m = 10ms]
  = exp(-10/10) = exp(-1) ≈ 0.3679

Spike generation (for hidden layers only):
if V_i[t] ≥ V_th:
    s_i[t] = 1 (spike fires)
    V_i[t+1] = 0 (reset)
else:
    s_i[t] = 0
    V_i[t+1] = (computed above)

Output layer (no spike threshold):
V_out[t] remains as real value (integrated membrane potential)
```

### Forward Pass Algorithm

```
function forward_pass(spike_input, weights_dict, biases_dict, durations_ms=1000):
    """
    Simulate SNN forward pass.
    
    Input:
    - spike_input: (T, 64) spike train
    - weights_dict: {'W1': (64, 128), 'W2': (128, 64), 'W3': (64, 3)}
    - biases_dict: {'b1': (128,), 'b2': (64,), 'b3': (3,)}
    
    Output:
    - output_membrane: (T, 3) membrane potentials at output
    - hidden_spikes_1: (T, 128) hidden layer 1 spikes
    - hidden_spikes_2: (T, 64) hidden layer 2 spikes
    """
    
    T = len(spike_input)
    alpha = np.exp(-10/10)  # α = exp(-frame_dt / τ_m) ≈ 0.3679
    # Each frame = 10ms (from ARCH_1/ARCH_2 hop_length)
    # τ_m = 10ms (membrane time constant)
    v_th = 1.0
    
    # Initialize membrane potentials and spike trains
    V1 = np.zeros(128)
    V2 = np.zeros(64)
    V3 = np.zeros(3)
    
    hidden_spikes_1 = np.zeros((T, 128), dtype=np.bool_)
    hidden_spikes_2 = np.zeros((T, 64), dtype=np.bool_)
    output_membrane = np.zeros((T, 3))
    
    # Timestep-by-timestep simulation
    for t in range(T):
        # Layer 1: Input → Hidden 1
        I1 = spike_input[t] @ weights_dict['W1'] + biases_dict['b1']
        V1 = alpha * V1 + (1 - alpha) * I1
        s1 = (V1 >= v_th).astype(np.float32)
        V1[s1.astype(bool)] = 0  # Reset spiking neurons
        hidden_spikes_1[t] = s1
        
        # Layer 2: Hidden 1 → Hidden 2
        I2 = s1 @ weights_dict['W2'] + biases_dict['b2']
        V2 = alpha * V2 + (1 - alpha) * I2
        s2 = (V2 >= v_th).astype(np.float32)
        V2[s2.astype(bool)] = 0  # Reset
        hidden_spikes_2[t] = s2
        
        # Layer 3: Hidden 2 → Output (NO spike threshold, rate-based)
        I3 = s2 @ weights_dict['W3'] + biases_dict['b3']
        V3 = alpha * V3 + (1 - alpha) * I3
        # NO reset for output layer - accumulate membrane potential
        output_membrane[t] = V3
    
    return {
        'output_membrane': output_membrane,
        'hidden_spikes_1': hidden_spikes_1,
        'hidden_spikes_2': hidden_spikes_2,
        'final_output': V3  # Final membrane potential (T=end)
    }
```

**Key insight**: By time T (end of sequence), output layer integrates total spike counts from all hidden neurons, forming a rate-based classifier at the final timestep.

---

## Component 3: Surrogate Gradient Descent

### The Problem: Non-Differentiable Spike Function

Standard spike function is step function:
```
s[t] = 1 if V[t] ≥ V_th else 0
```

**Gradient is 0 everywhere** (except at threshold where it's undefined):
```
ds/dV = 0 (can't train!)
```

### Solution: Surrogate Gradient Function

Replace spike function's derivative with smooth approximation during backpropagation:

**ArcTan surrogate** (recommended for SNNs):
```
True forward pass:
s[t] = heaviside(V[t] - V_th)  [step function, 0 or 1]

Training backward pass (surrogate):
ds/dV ≈ (1/π) * α / (1 + (α*(V[t] - V_th))²)

where α = 2.0 (steepness parameter)

Compared to true derivative:
∂heaviside(x)/∂x = δ(x)  [Dirac delta - essentially 0]
∂surrogate(x)/∂x ≠ 0  [smooth & trainable!]
```

**Visualization**:
```
Heaviside (true):        ArcTan Surrogate (training):
    1 ┐                       1 ┐
      │        ___              │     ___
      │       |                 │    /
      │       |                 │   /
      │_______|                 │  /
      0 1 2 3 4 (V)         0 1 2 3 4 (V)
    
True: Discontinuous         Surrogate: Smooth & differentiable
      gradient is zero       gradient is non-zero near threshold
```

### Backward Pass (Backpropagation Through Time)

**BPTT with Surrogate Gradients**:

```
function backward_pass(output_membrane, target_label, hidden_spikes_1, 
                       hidden_spikes_2, spike_input, weights_dict):
    """
    Compute loss and gradients using BPTT.
    
    Input:
    - output_membrane: (T, 3) output layer membrane potentials
    - target_label: int in {0, 1, 2}
    - hidden_spikes_1, hidden_spikes_2: spike trains from forward pass
    - spike_input: (T, 64) input spikes
    - weights_dict: network weights
    
    Output:
    - loss: scalar cross-entropy loss
    - gradients: {dW1, dW2, dW3, db1, db2, db3}
    """
    
    T = len(output_membrane)
    alpha = np.exp(-10/10)  # α = exp(-frame_dt / τ_m) ≈ 0.3679 at frame level
    v_th = 1.0
    alpha_surrogate = 2.0  # Steepness of ArcTan
    
    # Loss: Cross-entropy on final output
    final_output = output_membrane[-1]  # (3,) - final membrane potential
    softmax_output = softmax(final_output)
    loss = -log(softmax_output[target_label])
    
    # Gradient of loss w.r.t. final output
    dL_dout = softmax_output
    dL_dout[target_label] -= 1  # Gradient of cross-entropy
    
    # Initialize gradients for hidden layers
    dL_dV3 = np.zeros((T, 3))
    dL_dV3[-1] = dL_dout  # Backprop from loss at final time
    
    # Backpropagate through output layer (no spike threshold)
    # For non-spiking layer: ds/dV = 1 (identity)
    # dL/dI3[t] = dL/dV3[t] (direct backprop)
    
    # Initialize gradient accumulators
    grad_W3 = np.zeros_like(weights_dict['W3'])
    grad_W2 = np.zeros_like(weights_dict['W2'])
    grad_W1 = np.zeros_like(weights_dict['W1'])
    
    # Backward through time (t = T-1 down to 0)
    for t in range(T-1, -1, -1):
        # Backprop to hidden layer 2
        dL_dI3 = dL_dV3[t]
        grad_W3 += np.outer(hidden_spikes_2[t], dL_dI3)
        dL_dh2 = dL_dI3 @ weights_dict['W3'].T
        
        # Surrogate gradient for hidden layer 2 (spiking layer)
        # dL/dV2[t] = dL/dh2[t] * (surrogate derivative)
        surrogate_grad_2 = alpha_surrogate / (np.pi * (1 + (alpha_surrogate * (V2[t] - v_th))**2))
        dL_dV2 = dL_dh2 * surrogate_grad_2
        
        # Backprop to hidden layer 1
        dL_dI2 = dL_dV2
        grad_W2 += np.outer(hidden_spikes_1[t], dL_dI2)
        dL_dh1 = dL_dI2 @ weights_dict['W2'].T
        
        # Surrogate gradient for hidden layer 1
        surrogate_grad_1 = alpha_surrogate / (np.pi * (1 + (alpha_surrogate * (V1[t] - v_th))**2))
        dL_dV1 = dL_dh1 * surrogate_grad_1
        
        # Backprop to input
        dL_dI1 = dL_dV1
        grad_W1 += np.outer(spike_input[t], dL_dI1)
        
        # Backprop through time for V2 and V1
        if t > 0:
            dL_dV3[t-1] += dL_dV2 * alpha  # BPTT: influence on previous time step
            dL_dV2[t-1] += dL_dV1 * alpha
    
    # Normalize gradients by sequence length
    grad_W3 /= T
    grad_W2 /= T
    grad_W1 /= T
    
    return {
        'loss': loss,
        'gradients': {
            'dW1': grad_W1,
            'dW2': grad_W2,
            'dW3': grad_W3
        }
    }
```

**Note**: In practice, use **gradient clipping** to prevent exploding gradients:
```
grad_norm = sqrt(Σ grad²)
if grad_norm > threshold:
    grad *= threshold / grad_norm
```

---

## Component 4: Data Augmentation

### Augmentation Techniques

**1. Mixup augmentation** (interpolate between samples):
```
function mixup_spikes(spike_a, spike_b, label_a, label_b, alpha=0.2):
    """
    Linear interpolation between two spike samples.
    
    lambda ~ Beta(alpha, alpha)
    spike_mixed = lambda * spike_a + (1 - lambda) * spike_b
    label_mixed = lambda * one_hot(label_a) + (1 - lambda) * one_hot(label_b)
    """
    
    lambda_ = np.random.beta(alpha, alpha)
    spike_mixed = lambda_ * spike_a + (1 - lambda_) * spike_b
    label_mixed = np.array([
        lambda_ if label_a == i else (1 - lambda_) if label_b == i else 0
        for i in range(3)
    ])
    
    return spike_mixed, label_mixed
```

**2. Time-shift augmentation** (temporal jitter):
```
function time_shift_spikes(spikes, max_shift_ms=50, sr=16000):
    """Shift spike train by random amount (±50ms)."""
    
    max_shift_samples = int(max_shift_ms * sr / 1000)
    shift_amount = np.random.randint(-max_shift_samples, max_shift_samples)
    
    return np.roll(spikes, shift_amount, axis=0)  # Circular shift
```

**3. Noise injection** (add low-level spike noise):
```
function add_spike_noise(spikes, noise_rate=0.01):
    """Randomly flip (1→0 or 0→1) spike values."""
    
    noise_mask = np.random.rand(*spikes.shape) < noise_rate
    noisy_spikes = spikes.copy()
    noisy_spikes[noise_mask] = 1 - noisy_spikes[noise_mask]
    
    return noisy_spikes
```

### Augmentation Pipeline

```
function create_augmented_dataset(spike_dataset, augmentation_factor=2):
    """
    Expand dataset from 600 to 1200 samples.
    
    For each original sample:
    1. Keep original
    2. Create augmented version (random combination of techniques)
    """
    
    augmented_spikes = []
    augmented_labels = []
    
    for i, (spikes, label) in enumerate(spike_dataset):
        # Original
        augmented_spikes.append(spikes)
        augmented_labels.append(label)
        
        # Augmented version
        augmented = spikes.copy()
        
        # Apply random augmentations
        if np.random.rand() > 0.3:
            augmented = time_shift_spikes(augmented, max_shift_ms=50)
        
        if np.random.rand() > 0.4:
            augmented = add_spike_noise(augmented, noise_rate=0.01)
        
        augmented_spikes.append(augmented)
        augmented_labels.append(label)
    
    return augmented_spikes, augmented_labels
```

**Result**: 600 samples → 1200 samples (2× expansion)

---

## Component 5: Training Loop

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| **Learning rate** | 1e-3 (0.001) | Standard for neural networks |
| **Optimizer** | Adam (β₁=0.9, β₂=0.999) | Handles sparse updates well |
| **LR decay** | 0.95 every 10 epochs | Stabilize training after convergence |
| **Batch size** | 16 | GPU memory budget (~4GB) |
| **Epochs** | 100 | Typical convergence for SNNs |
| **Gradient clip** | 5.0 | Prevent exploding gradients |
| **Weight decay (L2)** | 1e-4 | Regularization |

### Training Loop Algorithm

```
function train_snn(augmented_spikes, augmented_labels, val_spikes, val_labels):
    """
    Main training loop (100 epochs).
    """
    
    model = {
        'W1': xavier_init(64, 128),
        'W2': xavier_init(128, 64),
        'W3': xavier_init(64, 3),
        'b1': np.ones(128) * 0.01,
        'b2': np.ones(64) * 0.01,
        'b3': np.zeros(3)
    }
    
    optimizer = Adam(learning_rate=1e-3)
    best_val_loss = float('inf')
    patience = 20  # Early stopping
    patience_counter = 0
    
    for epoch in range(100):
        # Training phase
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        # Shuffle training data
        indices = np.random.permutation(len(augmented_spikes))
        
        # Mini-batch training
        for batch_start in range(0, len(augmented_spikes), batch_size=16):
            batch_indices = indices[batch_start:batch_start+16]
            batch_loss = 0
            batch_gradients = None
            
            for idx in batch_indices:
                spikes = augmented_spikes[idx]
                label = augmented_labels[idx]
                
                # Forward pass
                output = forward_pass(spikes, model, ...)
                
                # Backward pass
                loss, grads = backward_pass(output, label, ...)
                
                # Accumulate gradients
                if batch_gradients is None:
                    batch_gradients = grads
                else:
                    for key in batch_gradients:
                        batch_gradients[key] += grads[key]
                
                batch_loss += loss
            
            # Average gradients by batch size
            for key in batch_gradients:
                batch_gradients[key] /= 16
            
            # Clip gradients
            grad_norm = sqrt(sum(norm(g)**2 for g in batch_gradients.values()))
            if grad_norm > 5.0:
                for key in batch_gradients:
                    batch_gradients[key] *= 5.0 / grad_norm
            
            # Update weights (Adam optimizer)
            model = optimizer.step(model, batch_gradients)
            
            epoch_loss += batch_loss
        
        # Validation phase
        val_loss = 0
        val_correct = 0
        
        for spikes, label in zip(val_spikes, val_labels):
            output = forward_pass(spikes, model, ...)
            final_output = output['final_output']
            
            # Loss
            softmax_out = softmax(final_output)
            val_loss -= log(softmax_out[label])
            
            # Accuracy
            pred = argmax(final_output)
            if pred == label:
                val_correct += 1
        
        val_acc = val_correct / len(val_spikes)
        
        # Decay learning rate every 10 epochs
        if (epoch + 1) % 10 == 0:
            optimizer.learning_rate *= 0.95
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy(model)
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={epoch_loss/1200:.4f}, val_acc={val_acc:.3f}")
    
    return best_model
```

---

## Component 6: Loss Function

### Cross-Entropy Loss

```
function cross_entropy_loss(output_logits, target_label):
    """
    Standard cross-entropy loss.
    
    Input:
    - output_logits: (3,) raw output values from network
    - target_label: int in {0, 1, 2}
    
    Output:
    - loss: scalar
    """
    
    # Softmax normalization
    exp_logits = exp(output_logits - max(output_logits))  # Subtract max for numerical stability
    softmax_probs = exp_logits / sum(exp_logits)
    
    # Cross-entropy
    loss = -log(softmax_probs[target_label])
    
    # Gradient for backprop
    grad = softmax_probs
    grad[target_label] -= 1  # Standard cross-entropy gradient
    
    return loss, grad
```

---

## Training Output

**Saved model checkpoint** (after 100 epochs):
```
model_checkpoint.pth:
├── W1: (64, 128) weight matrix
├── W2: (128, 64) weight matrix
├── W3: (64, 3) weight matrix
├── b1, b2, b3: bias vectors
├── hyperparams: tau_m, v_th, etc.
├── training_history: {
│   ├── epoch_losses: [loss₀, loss₁, ..., loss₉₉]
│   ├── val_accuracies: [acc₀, acc₁, ..., acc₉₉]
│   └── best_val_loss: X.XXX
│   }
└── class_names: ['gunshot', 'chainsaw', 'vehicle']
```

**Expected performance**:
- Training accuracy: >90%
- Validation accuracy: >85%
- Test accuracy: >85% (target)
- Training time: ~2-3 hours on V100 GPU

---

## Next Stage (ARCH_4: Inference)

The trained model weights are loaded into inference pipeline:
```python
# Load trained model
model = load_checkpoint('model_checkpoint.pth')

# Inference: raw audio → classification
audio = load_audio('forest_recording.wav')
mel_spec = extract_mel_spectrogram(audio)
spikes = convert_mel_to_spikes(mel_spec)
output = forward_pass(spikes, model)  # No backward pass
class_id = argmax(output['final_output'])  # 0, 1, or 2
```

---

## Cross-References & Integration

### Pipeline Dependencies
- **Upstream**: 
  - Receives training dataset from **[ARCH_3: Dataset Preparation](./ARCH_3_DATASET_PREPARATION.md)** (line 43, output dataset)
  - Input format specification: `(720, T, 64, 1)` spike tensors for training, `(240, T, 64, 1)` for validation, `(240, T, 64, 1)` for test
  - See [ARCH_3: Output Format](./ARCH_3_DATASET_PREPARATION.md#data-format-specifications)

- **Downstream**: 
  - Outputs trained weights to **[ARCH_5: SNN Inference](./ARCH_5_SNN_INFERENCE.md)** (line 6, "Input: Trained SNN model from ARCH_4")
  - Weight format: 3 dense layers with shapes (64×128), (128×64), (64×3)
  - See [ARCH_5: Input Specification](./ARCH_5_SNN_INFERENCE.md)

### Data Format Specifications
- **Input Format** (from ARCH_3): `(batch_size, T, 64, 1)` spike tensors
  - Training: batch_size=32, T≈1000 frames, 64 mel-band neurons, 1 spike dimension
  - Val/Test: batch_size=32 (or full validation set of 240 samples)
  - Value range: {0, 1} binary spikes
  - See [ARCH_3: Output Format](./ARCH_3_DATASET_PREPARATION.md#data-format-specifications)

- **Output Format**: Class predictions and final layer activations
  - `(batch_size, 3)` logits for 3 classes: {gunshot, chainsaw, vehicle}
  - Uses sparse softmax cross-entropy loss
  - See [Component 2: Loss Function](./ARCH_4_SNN_TRAINING.md#loss-function-sparse-cross-entropy-for-multilayer-snn) (lines 137-160)

### Processing Timeline
- **Days 10-12** (IMPLEMENTATION_SCHEDULE): SNN training (120 epochs, ~3 seconds per epoch on GPU)
- **Expected Training Time**: ~6 minutes total (120 epochs × 3s with checkpoint + validation)
- **Expected Convergence**: ~60 epochs (validation accuracy plateau)
- **Output Size**: Trained weights (64×128 + 128×64 + 64×3 = 12,544 parameters) ≈ 50KB at float32

### Key Parameters (Finalized)
| Parameter | Value | Reference | Impact |
|-----------|-------|-----------|--------|
| Input neurons (I) | 64 | Line 20 | Mel-band frequency channels from ARCH_2 |
| Hidden neurons (H) | 128 | Line 22 | Balances expressivity vs. energy |
| Output classes (O) | 3 | Line 24 | {gunshot, chainsaw, vehicle} |
| LIF Threshold | 1.0 | Line 106 | Frame-level spike generation |
| LIF Decay (α) | exp(-1) ≈ 0.3679 | Line 107 | Frame-level discretization (dt=10ms, τ_m=10ms) |
| Batch Size | 32 | Line 68 | Balances gradient stability vs. memory |
| Learning Rate (initial) | 5e-4 | Line 80 | Adam optimizer decay schedule |
| Epochs | 120 | Line 85 | Stopping criteria + validation plateau |
| Loss Function | Sparse Cross-Entropy | Line 137 | Multi-class classification |
| Target Accuracy | >85% | Line 7 | Across Corbett, Seshachalam, Sundarbans |

### Model Architecture Details
See [Component 1: Model Architecture](./ARCH_4_SNN_TRAINING.md#component-1-model-architecture-for-multilayer-snn) (lines 15-35):
- **Input Layer**: 64 LIF neurons (inherits from ARCH_1 mel-bands, ARCH_2 spike conversion)
- **Hidden Layer**: 128 LIF neurons with weight matrix W1 (64×128)
- **Output Layer**: 3 output neurons (dense, non-spiking) with weight matrix W2 (128×3), bias vector b2
- **Output Activation**: Softmax over 3 classes

### Critical Data Validation
Before training, verify:
1. **Input shape**: (720, ~1000, 64, 1) for training set
2. **Class distribution**: Each class ≥200 training samples (from 3-way stratified split)
3. **Firing rate**: 25% ±5% across training dataset (from ARCH_3 normalization)
4. **No NaN/Inf**: All spike tensors are 0 or 1
5. **Time dimension**: All samples have T≈1000 frames (pad/truncate if needed)

### Training Stability Features
- **Gradient clipping**: max norm = 1.0 (prevents exploding gradients)
- **Batch normalization**: Applied post-hoc via normalization in ARCH_3
- **Early stopping**: Monitor validation loss, stop if plateau for 10 epochs
- **Checkpoint strategy**: Save best model (lowest validation loss) for ARCH_5 inference

### Related Documentation
- **SUMMARY_HIGH_LEVEL_ARCHITECTURE.md** (Week 2-3): SNN training overview
- **IMPLEMENTATION_SCHEDULE.md** (Days 10-12): Training tasks
- **ARCH_3_DATASET_PREPARATION.md**: Input dataset preparation (ARCH_3 normalization + augmentation)
- **ARCH_5_SNN_INFERENCE.md**: Trained model consumer
- **Resources/Research_Paper_Citations.md**: References for SNN training algorithms, LIF neurons

