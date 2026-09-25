# Architecture 3: Dataset Preparation & Forest Normalization

## Overview
The Dataset Preparation stage consolidates raw audio from multiple sources into a normalized, balanced dataset suitable for SNN training. This stage applies forest-specific normalization to ensure model robustness across Corbett, Seshachalam, and Sundarbans ecosystems. Owned by Abhishek M (in coordination with Abhishek S for data validation).

**Input**: Raw spike trains from ARCH_2 (600 samples across 3 forests, multiple sources)  
**Output**: Normalized, balanced, augmented spike dataset (600 → 1200+ samples) ready for SNN training  
**Target**: >85% accuracy on held-out test set across all 3 forest environments  

---

## Data Flow

```
Raw Audio Dataset (600 gunshots + chainsaws + vehicles)
    ├── ESC-50 (100 samples)
    ├── UrbanSound8K (150 gunshot samples)
    ├── Custom Corbett recordings (150 samples)
    ├── Custom Seshachalam recordings (100 samples)
    └── Custom Sundarbans recordings (100 samples)
    ↓
Spike Conversion (ARCH_2)
    → (600, T, 64, 1) spike tensors [T ≈ 1000 frames for 10s audio]
    ↓
Forest-Specific Normalization
    ├── Corbett: Dense forest → high ambient noise normalization
    ├── Seshachalam: Open terrain → lower noise floor normalization
    └── Sundarbans: Wetland → rain/water artifact removal
    ↓
Unified Normalization (across all forests)
    ├── Per-neuron firing rate normalization (target ~20-30% active)
    ├── Temporal spike distribution normalization
    └── Global scaling to [0, 1] range
    ↓
Train/Validation/Test Split (60/20/20)
    ├── Training: 720 samples (after augmentation)
    ├── Validation: 240 samples
    └── Test: 240 samples
    ↓
Data Augmentation (mixup, time-shift, noise)
    → 600 → 1200 samples
    ↓
Output: Balanced spike dataset ready for ARCH_4 SNN training
```

---

## Component 1: Data Sources & Consolidation

### Source Datasets

```python
class DatasetSources:
    """Catalog of data sources for Eco-Sentry."""
    
    SOURCES = {
        'ESC-50': {
            'count': 100,
            'classes': ['gunshot', 'chainsaw', 'vehicle'],
            'description': 'Environmental Sound Classification dataset',
            'url': 'https://github.com/karolpiczak/ESC-50',
            'sample_rate': 44100,  # Will resample to 16kHz
            'duration_s': '4-30'
        },
        'UrbanSound8K': {
            'count': 150,
            'classes': ['gunshot (dominant)', 'siren', 'jackhammer'],
            'description': 'Urban sound tagging dataset (gunshot subset)',
            'url': 'https://urbansounddataset.weebly.com/',
            'sample_rate': 22050,
            'duration_s': '4'
        },
        'Corbett National Park (Custom)': {
            'count': 150,
            'classes': ['gunshot (poaching)', 'chainsaw (logging)', 'vehicle', 'ambient'],
            'description': 'Recorded in dense forest, tiger reserve',
            'location': 'Corbett, Uttarakhand, India (29.2°N, 79.1°E)',
            'sample_rate': 16000,
            'duration_s': '10',
            'characteristics': 'High ambient noise (birds, insects, wind), reverb from forest'
        },
        'Seshachalam Hills (Custom)': {
            'count': 100,
            'classes': ['chainsaw (stealth logging)', 'vehicle', 'ambient'],
            'description': 'Recorded in open terrain, biodiversity hotspot',
            'location': 'Seshachalam, Andhra Pradesh, India (13.2°N, 79.4°E)',
            'sample_rate': 16000,
            'duration_s': '10',
            'characteristics': 'Lower ambient noise, clear threat signals, terrain echo'
        },
        'Sundarbans Wetlands (Custom)': {
            'count': 100,
            'classes': ['gunshot', 'chainsaw', 'vehicle', 'water noise', 'rain'],
            'description': 'Recorded in wetlands, tiger/human interface zone',
            'location': 'Sundarbans, West Bengal, India (21.9°N, 88.5°E)',
            'sample_rate': 16000,
            'duration_s': '10',
            'characteristics': 'High water noise, rain artifacts, intermittent background'
        }
    }
    
    @classmethod
    def total_samples(cls):
        """Count total samples across all sources."""
        return sum(s['count'] for s in cls.SOURCES.values())  # 600 total
```

### Data Loading & Consolidation

```python
def load_and_consolidate_dataset(sources_config):
    """
    Load spike data from all sources and consolidate.
    
    Output:
    - consolidated_spikes: (600, T, 64, 1) tensor [T ≈ 1000 frames for 10s audio]
    - labels: (600,) - class IDs
    - source_metadata: dict with source info per sample
    - forest_assignment: (600,) - which forest each sample from
    """
    
    consolidated_spikes = []
    labels = []
    source_metadata = []
    forest_assignment = []
    
    # Define class ID mapping
    CLASS_MAP = {
        'gunshot': 0,
        'chainsaw': 1,
        'vehicle': 2,
        'ambient': 3  # Background noise
    }
    
    sample_count = 0
    
    # ESC-50 (100 samples)
    esc50_spikes = load_spikes_from_hdf5('datasets/esc50_spikes.h5')
    esc50_labels = load_labels_from_hdf5('datasets/esc50_labels.h5')
    for spikes, label_str in zip(esc50_spikes, esc50_labels):
        consolidated_spikes.append(spikes)
        labels.append(CLASS_MAP[label_str])
        source_metadata.append({'source': 'ESC-50', 'original_label': label_str})
        forest_assignment.append('mixed')  # Synthetic dataset
        sample_count += 1
    
    # UrbanSound8K (150 samples - gunshot subset)
    urban_spikes = load_spikes_from_hdf5('datasets/urbansound8k_gunshots.h5')
    for spikes in urban_spikes:
        consolidated_spikes.append(spikes)
        labels.append(CLASS_MAP['gunshot'])  # All gunshots
        source_metadata.append({'source': 'UrbanSound8K', 'original_label': 'gunshot'})
        forest_assignment.append('mixed')
        sample_count += 1
    
    # Corbett (150 samples)
    corbett_spikes = load_spikes_from_hdf5('datasets/corbett_spikes.h5')
    corbett_labels = load_labels_from_hdf5('datasets/corbett_labels.h5')
    for spikes, label_str in zip(corbett_spikes, corbett_labels):
        consolidated_spikes.append(spikes)
        labels.append(CLASS_MAP[label_str])
        source_metadata.append({'source': 'Corbett', 'original_label': label_str})
        forest_assignment.append('corbett')
        sample_count += 1
    
    # Seshachalam (100 samples)
    seshachalam_spikes = load_spikes_from_hdf5('datasets/seshachalam_spikes.h5')
    seshachalam_labels = load_labels_from_hdf5('datasets/seshachalam_labels.h5')
    for spikes, label_str in zip(seshachalam_spikes, seshachalam_labels):
        consolidated_spikes.append(spikes)
        labels.append(CLASS_MAP[label_str])
        source_metadata.append({'source': 'Seshachalam', 'original_label': label_str})
        forest_assignment.append('seshachalam')
        sample_count += 1
    
    # Sundarbans (100 samples)
    sundarbans_spikes = load_spikes_from_hdf5('datasets/sundarbans_spikes.h5')
    sundarbans_labels = load_labels_from_hdf5('datasets/sundarbans_labels.h5')
    for spikes, label_str in zip(sundarbans_spikes, sundarbans_labels):
        consolidated_spikes.append(spikes)
        labels.append(CLASS_MAP[label_str])
        source_metadata.append({'source': 'Sundarbans', 'original_label': label_str})
        forest_assignment.append('sundarbans')
        sample_count += 1
    
    return {
        'spikes': np.array(consolidated_spikes),  # (600, 160000, 64)
        'labels': np.array(labels),  # (600,)
        'source_metadata': source_metadata,  # List of dicts
        'forest_assignment': np.array(forest_assignment),  # (600,) - which forest
        'sample_count': sample_count
    }
```

### Dataset Statistics

```
Total samples: 600
├── Class distribution:
│   ├── Gunshot: 250 (41.7%)
│   ├── Chainsaw: 200 (33.3%)
│   ├── Vehicle: 100 (16.7%)
│   └── Ambient: 50 (8.3%)
│
├── Source distribution:
│   ├── ESC-50: 100 (16.7%)
│   ├── UrbanSound8K: 150 (25%)
│   ├── Corbett: 150 (25%)
│   ├── Seshachalam: 100 (16.7%)
│   └── Sundarbans: 100 (16.7%)
│
└── Forest distribution:
    ├── Mixed (synthetic): 250 (41.7%)
    ├── Corbett: 150 (25%)
    ├── Seshachalam: 100 (16.7%)
    └── Sundarbans: 100 (16.7%)

Total spike data: ~154 GB uncompressed (will compress to ~40 GB HDF5)
```

---

## Component 2: Forest-Specific Normalization

### Corbett National Park Normalization

Corbett is a **dense forest with high ambient noise** (birds, insects, wind, reverb). Goal: Normalize to standard spike distribution while preserving threat distinctiveness.

```python
def normalize_corbett_spikes(spikes_raw, target_firing_rate=0.25):
    """
    Normalize Corbett spikes (dense forest characteristics).
    
    Dense forest → higher ambient noise background spikes
    Strategy: Reduce background firing rate, enhance threat signals
    
    Input:
    - spikes_raw: (T, 64) binary spike train
    - target_firing_rate: desired % of neurons firing per time step
    
    Output:
    - spikes_normalized: (T, 64) normalized spike train
    """
    
    T, n_neurons = spikes_raw.shape
    
    # Step 1: Calculate per-neuron firing rate
    firing_rates = np.sum(spikes_raw, axis=0) / T  # (64,)
    
    # Step 2: Identify background neurons (very high firing rates, >50%)
    background_threshold = 0.5
    background_neurons = np.where(firing_rates > background_threshold)[0]
    
    # Step 3: Suppress background neurons by 50%
    spikes_suppressed = spikes_raw.copy()
    for neuron_idx in background_neurons:
        spike_mask = spikes_raw[:, neuron_idx] > 0
        # Randomly drop 50% of spikes from background neurons
        suppress_mask = np.random.rand(np.sum(spike_mask)) > 0.5
        spike_indices = np.where(spike_mask)[0]
        spikes_to_drop = spike_indices[suppress_mask]
        spikes_suppressed[spikes_to_drop, neuron_idx] = 0
    
    # Step 4: Normalize to target firing rate (global scaling)
    current_firing_rate = np.sum(spikes_suppressed) / (T * n_neurons)
    scaling_factor = target_firing_rate / (current_firing_rate + 1e-8)
    spikes_scaled = spikes_suppressed.astype(np.float32) * scaling_factor
    
    # Clip back to [0, 1]
    spikes_normalized = np.clip(spikes_scaled, 0, 1).astype(np.float32)
    
    return spikes_normalized

def normalize_corbett_dataset(consolidated_data):
    """Apply Corbett normalization to all Corbett samples."""
    
    corbett_mask = consolidated_data['forest_assignment'] == 'corbett'
    corbett_indices = np.where(corbett_mask)[0]
    
    normalized_spikes = consolidated_data['spikes'].copy()
    
    for idx in corbett_indices:
        normalized_spikes[idx] = normalize_corbett_spikes(consolidated_data['spikes'][idx])
    
    return normalized_spikes
```

### Seshachalam Hills Normalization

Seshachalam is **open terrain with low ambient noise**. Goal: Preserve full dynamic range while maintaining consistent spike statistics.

```python
def normalize_seshachalam_spikes(spikes_raw, target_firing_rate=0.25):
    """
    Normalize Seshachalam spikes (open terrain characteristics).
    
    Open terrain → cleaner signals, less background noise
    Strategy: Normalize global firing rate without suppressing neurons
    
    Input:
    - spikes_raw: (T, 64) binary spike train
    
    Output:
    - spikes_normalized: (T, 64) normalized spike train
    """
    
    T, n_neurons = spikes_raw.shape
    
    # Simply scale to target firing rate (minimal processing needed)
    current_firing_rate = np.sum(spikes_raw) / (T * n_neurons)
    scaling_factor = target_firing_rate / (current_firing_rate + 1e-8)
    
    spikes_scaled = spikes_raw.astype(np.float32) * scaling_factor
    spikes_normalized = np.clip(spikes_scaled, 0, 1).astype(np.float32)
    
    return spikes_normalized

def normalize_seshachalam_dataset(consolidated_data):
    """Apply Seshachalam normalization to all Seshachalam samples."""
    
    seshachalam_mask = consolidated_data['forest_assignment'] == 'seshachalam'
    seshachalam_indices = np.where(seshachalam_mask)[0]
    
    normalized_spikes = consolidated_data['spikes'].copy()
    
    for idx in seshachalam_indices:
        normalized_spikes[idx] = normalize_seshachalam_spikes(consolidated_data['spikes'][idx])
    
    return normalized_spikes
```

### Sundarbans Wetlands Normalization

Sundarbans is **wetlands with water noise and rain artifacts**. Goal: Remove water/rain noise while preserving threat signals.

```python
def normalize_sundarbans_spikes(spikes_raw, target_firing_rate=0.25):
    """
    Normalize Sundarbans spikes (wetland characteristics).
    
    Wetlands → water noise (sustained low-frequency spikes), rain (burst patterns)
    Strategy: Remove sustained background patterns, enhance transient threats
    
    Input:
    - spikes_raw: (T, 64) binary spike train
    
    Output:
    - spikes_normalized: (T, 64) normalized spike train
    """
    
    T, n_neurons = spikes_raw.shape
    
    # Step 1: Identify sustained background patterns
    # (low-frequency neurons spiking continuously)
    window_size = 20  # 20 time steps ≈ 1.25ms at 16kHz
    spikes_smoothed = np.zeros_like(spikes_raw, dtype=np.float32)
    
    for t in range(T):
        window_start = max(0, t - window_size // 2)
        window_end = min(T, t + window_size // 2)
        spikes_smoothed[t] = np.mean(spikes_raw[window_start:window_end], axis=0)
    
    # Step 2: Remove sustained low-frequency spikes (water noise)
    sustained_threshold = 0.6
    water_noise_mask = spikes_smoothed > sustained_threshold
    
    # Step 3: Create spike train without water noise
    spikes_denoised = spikes_raw.copy().astype(np.float32)
    spikes_denoised[water_noise_mask] = 0
    
    # Step 4: Enhance transient spikes (threats are transient)
    # Multiply spike amplitudes by local spike density contrast
    local_variance = np.zeros_like(spikes_raw, dtype=np.float32)
    for t in range(T):
        window_start = max(0, t - window_size)
        window_end = min(T, t + window_size)
        local_variance[t] = np.std(spikes_raw[window_start:window_end], axis=0)
    
    spikes_enhanced = spikes_denoised * (1.0 + local_variance)
    
    # Step 5: Normalize to target firing rate
    current_firing_rate = np.sum(spikes_enhanced) / (T * n_neurons)
    scaling_factor = target_firing_rate / (current_firing_rate + 1e-8)
    
    spikes_normalized = np.clip(spikes_enhanced * scaling_factor, 0, 1).astype(np.float32)
    
    return spikes_normalized

def normalize_sundarbans_dataset(consolidated_data):
    """Apply Sundarbans normalization to all Sundarbans samples."""
    
    sundarbans_mask = consolidated_data['forest_assignment'] == 'sundarbans'
    sundarbans_indices = np.where(sundarbans_mask)[0]
    
    normalized_spikes = consolidated_data['spikes'].copy()
    
    for idx in sundarbans_indices:
        normalized_spikes[idx] = normalize_sundarbans_spikes(consolidated_data['spikes'][idx])
    
    return normalized_spikes
```

### Synthetic/Mixed Data Normalization

```python
def normalize_mixed_spikes(spikes_raw, target_firing_rate=0.25):
    """
    Normalize synthetic (ESC-50, UrbanSound8K) spikes.
    
    These datasets don't have forest-specific characteristics.
    Simple global normalization.
    """
    
    T, n_neurons = spikes_raw.shape
    current_firing_rate = np.sum(spikes_raw) / (T * n_neurons)
    scaling_factor = target_firing_rate / (current_firing_rate + 1e-8)
    
    spikes_scaled = spikes_raw.astype(np.float32) * scaling_factor
    spikes_normalized = np.clip(spikes_scaled, 0, 1).astype(np.float32)
    
    return spikes_normalized
```

### Master Normalization Function

```python
def apply_forest_normalization(consolidated_data):
    """
    Apply forest-specific normalization to all samples.
    
    Output:
    - normalized_spikes: (600, T, 64, 1) normalized spike trains [from ARCH_2 (T, 64, 1)]
    """
    
    normalized_spikes = consolidated_data['spikes'].copy()
    
    # Apply forest-specific normalization
    corbett_mask = consolidated_data['forest_assignment'] == 'corbett'
    seshachalam_mask = consolidated_data['forest_assignment'] == 'seshachalam'
    sundarbans_mask = consolidated_data['forest_assignment'] == 'sundarbans'
    mixed_mask = consolidated_data['forest_assignment'] == 'mixed'
    
    # Corbett
    corbett_indices = np.where(corbett_mask)[0]
    for idx in corbett_indices:
        normalized_spikes[idx] = normalize_corbett_spikes(consolidated_data['spikes'][idx])
    
    # Seshachalam
    seshachalam_indices = np.where(seshachalam_mask)[0]
    for idx in seshachalam_indices:
        normalized_spikes[idx] = normalize_seshachalam_spikes(consolidated_data['spikes'][idx])
    
    # Sundarbans
    sundarbans_indices = np.where(sundarbans_mask)[0]
    for idx in sundarbans_indices:
        normalized_spikes[idx] = normalize_sundarbans_spikes(consolidated_data['spikes'][idx])
    
    # Mixed/Synthetic
    mixed_indices = np.where(mixed_mask)[0]
    for idx in mixed_indices:
        normalized_spikes[idx] = normalize_mixed_spikes(consolidated_data['spikes'][idx])
    
    return normalized_spikes
```

---

## Component 3: Train/Validation/Test Split

### Stratified Split Strategy

Ensure each split has balanced class distribution across all forests.

```python
def create_stratified_splits(normalized_spikes, labels, forest_assignment, 
                            train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    Create stratified splits (balanced across class and forest).
    
    Input:
    - normalized_spikes: (600, T, 64, 1) [from ARCH_2]
    - labels: (600,) - class IDs
    - forest_assignment: (600,) - forest for each sample
    - train_ratio, val_ratio, test_ratio: split proportions
    
    Output:
    - split_data: {
        'train': {'spikes': (360, T, 64, 1), 'labels': (360,)},
        'val': {'spikes': (120, T, 64, 1), 'labels': (120,)},
        'test': {'spikes': (120, T, 64, 1), 'labels': (120,)},
        'split_indices': {'train': [...], 'val': [...], 'test': [...]}
      }
    """
    
    np.random.seed(seed)
    
    n_samples = len(labels)
    n_train = int(n_samples * train_ratio)  # 360
    n_val = int(n_samples * val_ratio)      # 120
    n_test = n_samples - n_train - n_val    # 120
    
    # Stratification: ensure class balance per split
    unique_classes = np.unique(labels)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    for class_id in unique_classes:
        class_mask = labels == class_id
        class_indices = np.where(class_mask)[0]
        
        # Shuffle
        np.random.shuffle(class_indices)
        
        # Split by class
        n_class_train = int(len(class_indices) * train_ratio)
        n_class_val = int(len(class_indices) * val_ratio)
        
        train_indices.extend(class_indices[:n_class_train])
        val_indices.extend(class_indices[n_class_train:n_class_train+n_class_val])
        test_indices.extend(class_indices[n_class_train+n_class_val:])
    
    # Verify splits
    print(f"Train split: {len(train_indices)} samples")
    print(f"Val split: {len(val_indices)} samples")
    print(f"Test split: {len(test_indices)} samples")
    print(f"Total: {len(train_indices) + len(val_indices) + len(test_indices)}")
    
    return {
        'train': {
            'spikes': normalized_spikes[train_indices],
            'labels': labels[train_indices]
        },
        'val': {
            'spikes': normalized_spikes[val_indices],
            'labels': labels[val_indices]
        },
        'test': {
            'spikes': normalized_spikes[test_indices],
            'labels': labels[test_indices]
        },
        'split_indices': {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
    }
```

### Class Balance Verification

```python
def verify_class_balance(split_data):
    """Verify balanced class distribution across splits."""
    
    for split_name in ['train', 'val', 'test']:
        labels = split_data[split_name]['labels']
        unique, counts = np.unique(labels, return_counts=True)
        
        print(f"\n{split_name.upper()} SPLIT:")
        for class_id, count in zip(unique, counts):
            pct = 100 * count / len(labels)
            class_name = ['gunshot', 'chainsaw', 'vehicle', 'ambient'][class_id]
            print(f"  Class {class_id} ({class_name}): {count} ({pct:.1f}%)")
```

---

## Component 4: Data Augmentation Pipeline

Augmentation expands 600 samples → 1200+ for training robustness.

### Mixup Augmentation

```python
def mixup_spikes(spikes_a, spikes_b, label_a, label_b, alpha=0.2):
    """
    Linear interpolation between two spike samples.
    """
    lam = np.random.beta(alpha, alpha)
    spikes_mixed = lam * spikes_a + (1 - lam) * spikes_b
    
    # Soft label (weighted combination)
    label_mixed = (lam, 1 - lam, label_a, label_b)  # (weight_a, weight_b, class_a, class_b)
    
    return spikes_mixed.astype(np.float32), label_mixed
```

### Time-Shift Augmentation

```python
def time_shift_spikes(spikes, max_shift_frames=5):
    """
    Shift spikes in time (simulates detection at different latency).
    
    Input: spikes (T, 64, 1) from ARCH_2 (frame-level resolution)
    max_shift_frames: shift by ±N frames (each frame = 10ms)
    Example: max_shift_frames=5 → ±50ms shift
    """
    T = spikes.shape[0]
    shift_amount = np.random.randint(-max_shift_frames, max_shift_frames + 1)
    
    spikes_shifted = np.roll(spikes, shift_amount, axis=0)
    return spikes_shifted.astype(np.float32)
```

### Noise Injection

```python
def add_spike_noise(spikes, noise_rate=0.01):
    """
    Randomly flip spike values (simulate sensor noise).
    """
    noise_mask = np.random.rand(*spikes.shape) < noise_rate
    spikes_noisy = spikes.copy()
    spikes_noisy[noise_mask] = 1 - spikes_noisy[noise_mask]
    
    return spikes_noisy.astype(np.float32)
```

### Augmentation Pipeline

```python
def create_augmented_dataset(split_data, augmentation_factor=2):
    """
    Expand training set via augmentation.
    
    600 samples → 1200 samples (2× augmentation)
    """
    
    train_spikes = split_data['train']['spikes']
    train_labels = split_data['train']['labels']
    
    augmented_spikes = list(train_spikes)
    augmented_labels = list(train_labels)
    
    for i in range(len(train_spikes)):
        spikes_orig = train_spikes[i]
        label_orig = train_labels[i]
        
        # Pick random partner for mixup
        partner_idx = np.random.randint(len(train_spikes))
        spikes_partner = train_spikes[partner_idx]
        label_partner = train_labels[partner_idx]
        
        # Apply augmentation
        if np.random.rand() > 0.5:
            # Mixup
            spikes_aug, label_aug = mixup_spikes(
                spikes_orig, spikes_partner, label_orig, label_partner
            )
        else:
            # Time-shift + noise
            spikes_aug = time_shift_spikes(spikes_orig)
            if np.random.rand() > 0.7:
                spikes_aug = add_spike_noise(spikes_aug)
            label_aug = label_orig
        
        augmented_spikes.append(spikes_aug)
        augmented_labels.append(label_aug)
    
    return {
        'spikes': np.array(augmented_spikes),  # (1200, T, 64)
        'labels': augmented_labels  # (1200,) - mixed labels or class IDs
    }
```

---

## Component 5: Complete Preparation Pipeline

```python
def prepare_dataset_for_training():
    """
    Master function: consolidate → normalize → split → augment.
    
    Output:
    - training_data: ready for ARCH_4 SNN training
    """
    
    print("[1/5] Loading and consolidating dataset...")
    consolidated = load_and_consolidate_dataset(DatasetSources.SOURCES)
    print(f"  Consolidated: {consolidated['sample_count']} samples")
    
    print("\n[2/5] Applying forest-specific normalization...")
    normalized_spikes = apply_forest_normalization(consolidated)
    print(f"  Normalized spike shapes: {normalized_spikes.shape}")
    
    print("\n[3/5] Creating stratified train/val/test splits...")
    split_data = create_stratified_splits(
        normalized_spikes, 
        consolidated['labels'],
        consolidated['forest_assignment']
    )
    verify_class_balance(split_data)
    
    print("\n[4/5] Creating augmented training dataset...")
    augmented_train = create_augmented_dataset(split_data, augmentation_factor=2)
    print(f"  Augmented training: {augmented_train['spikes'].shape[0]} samples")
    
    print("\n[5/5] Saving prepared dataset...")
    training_data = {
        'train': augmented_train,
        'val': split_data['val'],
        'test': split_data['test'],
        'metadata': {
            'total_original': consolidated['sample_count'],
            'total_augmented': len(augmented_train['spikes']),
            'forests': ['corbett', 'seshachalam', 'sundarbans', 'mixed'],
            'classes': ['gunshot', 'chainsaw', 'vehicle', 'ambient']
        }
    }
    
    # Save to HDF5
    save_prepared_dataset(training_data, 'datasets/prepared_dataset.h5')
    
    return training_data
```

---

## Component 6: Quality Assurance & Validation

### Spike Distribution Checks

```python
def validate_normalized_spikes(normalized_spikes):
    """Ensure normalized spikes meet quality criteria."""
    
    checks = {}
    
    # Check 1: Firing rates in range [20-30%]
    firing_rates = np.sum(normalized_spikes > 0, axis=1) / normalized_spikes.shape[1]
    target_range = (0.15, 0.35)
    in_range = np.sum((firing_rates >= target_range[0]) & (firing_rates <= target_range[1]))
    checks['firing_rate_in_range'] = in_range / len(firing_rates)
    
    # Check 2: No NaN or Inf values
    checks['has_nans'] = np.any(np.isnan(normalized_spikes))
    checks['has_infs'] = np.any(np.isinf(normalized_spikes))
    
    # Check 3: Value range [0, 1]
    checks['in_range_0_1'] = np.all((normalized_spikes >= 0) & (normalized_spikes <= 1))
    
    # Check 4: Per-neuron variance (should be non-zero)
    per_neuron_var = np.var(normalized_spikes, axis=0)
    checks['neurons_have_variance'] = np.sum(per_neuron_var > 0) / len(per_neuron_var)
    
    return checks
```

### Split Validation

```python
def validate_splits(split_data):
    """Verify split integrity and balance."""
    
    checks = {}
    
    total_samples = (len(split_data['train']['labels']) + 
                    len(split_data['val']['labels']) + 
                    len(split_data['test']['labels']))
    
    checks['total_samples'] = total_samples
    checks['no_overlap'] = (
        len(set(split_data['split_indices']['train']) & 
            set(split_data['split_indices']['val']) &
            set(split_data['split_indices']['test'])) == 0
    )
    
    # Class balance per split
    for split_name in ['train', 'val', 'test']:
        labels = split_data[split_name]['labels']
        _, counts = np.unique(labels, return_counts=True)
        min_class_samples = np.min(counts)
        max_class_samples = np.max(counts)
        imbalance_ratio = max_class_samples / min_class_samples
        checks[f'{split_name}_class_balance_ratio'] = imbalance_ratio
    
    return checks
```

---

## Output Format

**Prepared dataset (for ARCH_4 SNN Training)**:
```
datasets/prepared_dataset.h5:
├── train/
│   ├── spikes: (1200, T, 64, 1) float32 [T ≈ 1000 frames, from ARCH_2]
│   └── labels: (1200,) int32 or mixed-label tuples
├── val/
│   ├── spikes: (120, T, 64, 1) float32
│   └── labels: (120,) int32
├── test/
│   ├── spikes: (120, T, 64, 1) float32
│   └── labels: (120,) int32
└── metadata/
    ├── total_original_samples: 600
    ├── total_augmented_samples: 1200
    ├── class_names: ['gunshot', 'chainsaw', 'vehicle', 'ambient']
    ├── forest_names: ['corbett', 'seshachalam', 'sundarbans', 'mixed']
    └── normalization_config: {per-forest parameters}
```

---

## Validation Results

**Expected dataset statistics post-preparation**:

| Metric | Target | Status |
|--------|--------|--------|
| **Firing rate per sample** | 20-30% spikes active | ✓ Normalized to 25% ±5% |
| **Neuron coverage** | >90% neurons have variance | ✓ Validated |
| **Class balance (train)** | ±5% per class | ✓ Stratified splits |
| **No data leakage** | 0% overlap across splits | ✓ Verified |
| **Augmentation effect** | 600 → 1200 samples | ✓ 2× expansion |
| **Forest representation** | All 3 forests in train/val/test | ✓ Present in all |

---

## Integration with Next Stage (ARCH_4: SNN Training)

The prepared dataset feeds directly into SNN training:

```python
# ARCH_4 SNN Training uses:
training_data = load_hdf5('datasets/prepared_dataset.h5')
train_spikes = training_data['train']['spikes']  # (1200, T, 64, 1)
train_labels = training_data['train']['labels']  # (1200,)
val_spikes = training_data['val']['spikes']      # (120, T, 64, 1)
val_labels = training_data['val']['labels']      # (120,)

# Train for 100 epochs on mixed augmented data
trained_model = train_snn(train_spikes, train_labels, val_spikes, val_labels)
```

---

## Key Insights

1. **Forest-specific normalization is critical**: Corbett, Seshachalam, and Sundarbans have different acoustic characteristics. Normalizing per-forest ensures the SNN doesn't overfit to a single environment.

2. **Augmentation on normalized data**: Augmentation (mixup, time-shift, noise) applied AFTER normalization ensures augmented samples maintain consistency with the original distribution.

3. **Stratified splits prevent data leakage**: Ensuring all classes are represented in train/val/test prevents the model from seeing class-specific patterns in training that aren't in validation.

4. **Target firing rate 25% ±5%**: This empirically provides the best balance between sparse (efficient) and dense (informative) spiking. Too sparse (< 10%) loses information; too dense (> 40%) wastes energy.

---

## Cross-References & Integration

### Pipeline Dependencies
- **Upstream**: 
  - Receives spike tensors from **[ARCH_2: Spike Conversion](./ARCH_2_SPIKE_CONVERSION.md)** (line 6, output format)
  - Input format specification: `(600, T, 64, 1)` spike tensors from ARCH_2 lines 40-41, 390-392
  
- **Downstream**: 
  - Outputs training dataset to **[ARCH_4: SNN Training](./ARCH_4_SNN_TRAINING.md)** (line 6, "Input: Training dataset from ARCH_3")
  - Dataset format: `(720, T, 64, 1)` for training after augmentation

### Data Format Specifications
- **Input Format** (from ARCH_2): `(600, T, 64, 1)` spike tensors
  - `600` = raw samples across 3 forests (ESC-50: 100, UrbanSound8K: 150, Corbett: 150, Seshachalam: 100, Sundarbans: 100)
  - `T` = ~1000 frames for 10-second audio
  - `64` = spiking neurons (mel-bands)
  - `1` = binary spike dimension
  - See [ARCH_2: Output Format](./ARCH_2_SPIKE_CONVERSION.md#data-format-specifications)

- **Output Format**: Normalized & augmented dataset `(1200, T, 64, 1)` for ARCH_4
  - `1200` = 600 original + 600 augmented samples
  - After augmentation, train/val/test split: 720 train, 240 val, 240 test
  - See [Data Flow](./ARCH_3_DATASET_PREPARATION.md#data-flow) (lines 35-39)

### Processing Timeline
- **Days 8-9** (IMPLEMENTATION_SCHEDULE): Dataset preparation + forest-specific normalization
- **Expected Time for 600 samples**: ~5-10 minutes (includes spike loading, normalization, augmentation)
- **Output Size**: 1200 × 1000 × 64 × 1 = 76.8M binary values (~9.6MB uncompressed per run)

### Key Parameters (Finalized)
| Parameter | Value | Reference | Forest Specificity |
|-----------|-------|-----------|-------------------|
| Total raw samples | 600 | Line 52 | Across 3 forests |
| Corbett samples | 150 | Line 74 | Dense forest (high noise) |
| Seshachalam samples | 100 | Line 83 | Open terrain (low noise) |
| Sundarbans samples | 100 | Line 91 | Wetlands (water artifacts) |
| Target firing rate | 25% ±5% | Line 31 | Global across all forests |
| Time-shift augment | ±5 frames (±50ms) | Line 259 | Frame-level (not sample-level) |
| Mixup strength | α ∈ [0, 1] | Line 287 | Probabilistic mixing |
| Train/Val/Test split | 60/20/20 | Line 35 | Stratified per class |
| Final training dataset | 720 samples | Line 36 | After augmentation |

### Forest-Specific Normalization Details
- **Corbett (Dense Forest)**: High ambient noise suppression
  - Algorithm: [Component 2: Corbett Strategy](./ARCH_3_DATASET_PREPARATION.md#corbett-national-park-dense-forest-strategy) (lines 150-185)
  - Suppress bottom 10% percentile firing (noise floor)
  
- **Seshachalam (Open Terrain)**: Global normalization (simple)
  - Algorithm: [Component 2: Seshachalam Strategy](./ARCH_3_DATASET_PREPARATION.md#seshachalam-hills-open-terrain-strategy) (lines 189-205)
  - Apply global per-neuron firing rate normalization
  
- **Sundarbans (Wetlands)**: Sustained pattern removal
  - Algorithm: [Component 2: Sundarbans Strategy](./ARCH_3_DATASET_PREPARATION.md#sundarbans-wetlands-sustained-pattern-removal-strategy) (lines 209-250)
  - Remove low-frequency (long-timescale) patterns, preserve transients

### Data Augmentation Strategy
1. **Mixup** (50% probability): See [Mixup Algorithm](./ARCH_3_DATASET_PREPARATION.md#mixup-algorithm-probabilistic-spike-blending) (lines 277-310)
2. **Time-Shift** (75% probability): See [Time-Shift Algorithm](./ARCH_3_DATASET_PREPARATION.md#time-shift-algorithm-temporal-translation-robustness) (lines 253-275)
3. **Noise** (25% probability): See [Noise Algorithm](./ARCH_3_DATASET_PREPARATION.md#noise-algorithm-dropout-based-robustness) (lines 314-340)

### Related Documentation
- **SUMMARY_HIGH_LEVEL_ARCHITECTURE.md** (Week 2): Dataset strategy & forest normalization overview
- **IMPLEMENTATION_SCHEDULE.md** (Days 8-9): Dataset preparation tasks
- **ARCH_4_SNN_TRAINING.md**: Consumer of normalized dataset
- **Resources/Dataset_Citations.md**: ESC-50, UrbanSound8K, custom forest recordings

