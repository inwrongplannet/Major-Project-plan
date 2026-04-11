# Eco-Sentry Implementation Schedule (20 Days)

**Project Duration**: 20 calendar days  
**Team**: Abhishek M (SNN Pipeline), Kavya (Simulations), Abhishek S (Support)  
**Target**: Gunshot/chainsaw detection with <1KB JSON alerts, 5-12mW power consumption

---

## WEEK 1: Audio Processing & Dataset Preparation (Days 1-9)

### Day 1: Audio Processing Setup & Gammatone Filterbank (ARCH_1)
| Abhishek M | Kavya | Abhishek S |
|---|---|---|
| Setup Python environment (librosa, numpy, scipy) | Setup simulation environment (energy profiler skeleton) | Create project documentation structure |
| Implement Gammatone filterbank (64 channels, 50-8000 Hz) | Design energy profiler architecture | Setup monitoring dashboards |
| Load ESC-50 dataset (gunshot, chainsaw, vehicle, silence) | Create data logging framework | Prepare test harness |
| Generate mel-spectrogram outputs: (T, 64) shape | Document profiler assumptions | Create git workflow guide |

### Day 2: Mel-Spectrogram Computation & Validation (ARCH_1)
| Abhishek M | Kavya | Abhishek S |
|---|---|---|
| Compute mel-spectrograms from raw audio @ 16kHz | Profile Gammatone computation time (target <100ms per 10s) | Validate output dimensions |
| Normalize to [-80, 0] dB range per ARCH_1 spec | Test memory footprint (frame buffer size) | Generate test audio samples |
| Validate output shape consistency: (1000, 64) for 10s audio | Create thermal model for continuous operation | Document data format specs |
| Create audio preprocessing pipeline (batch processing) | Test CPU utilization during processing | Verify numerical accuracy |

### Day 3: Spike Conversion - LIF Neuron Model Setup (ARCH_2)
| Abhishek M | Kavya | Abhishek S |
|---|---|---|
| Implement Leaky Integrate-and-Fire (LIF) neuron model | Test LIF latency per frame (target ~15ms per 10s audio) | Validate spike output format |
| Configure LIF parameters: τ_mem=5ms, V_th=1.0, decay=0.9 | Profile energy per spike generation | Create spike visualization tools |
| Convert mel-spectrograms → spike trains (frame-level, NOT sample-level) | Create spike rate analyzer | Generate example spike patterns |
| Output format: (T, 64, 1) binary spikes per frame | Test memory efficiency | Verify temporal dynamics |

### Day 4: Spike Output Validation & Ground-Truth Labels (ARCH_2)
| Abhishek M | Kavya | Abhishek S |
|---|---|---|
| Validate spike trains match LIF dynamics (firing rates 10-50%) | Simulate spike transmission latency | Validate label consistency |
| Create ground-truth labels for 25 ESC-50 samples (gunshot=1, other=0) | Test spike encoding efficiency | Generate label statistics |
| Test spike-to-firing-rate conversion (Hz measurement) | Profile conversion overhead | Document spike format specs |
| Implement spike batch processing (parallel LIF neurons) | Create performance profiler | Verify output reproducibility |

### Day 5: Dataset Consolidation & Normalization (ARCH_3)
| Abhishek M | Kavya | Abhishek S |
|---|---|---|
| Consolidate 25 ESC-50 samples into master dataset | Test normalization accuracy (target ±5% deviation) | Validate consolidation process |
| Create forest-specific normalization: Corbett, Seshachalam, Sundarbans | Profile normalization latency (<1 min per 100 samples) | Document forest characteristics |
| Normalize spike statistics per forest (mean=0, std=1 per forest) | Test normalization stability | Generate forest comparison report |
| Validate normalized output shape: (25, 1000, 64, 1) | Create normalization lookup tables | Verify data integrity |

### Day 6: Data Augmentation Pipeline (ARCH_3)
| Abhishek M | Kavya | Abhishek S |
|---|---|---|
| Implement augmentation: time-shift (±10%), pitch-shift (±2 semitones), noise injection | Test augmentation impact on spike patterns | Validate augmented data quality |
| Generate 600 augmented samples from 25 originals (12x per sample) | Profile augmentation time (target <10 min for 600 samples) | Monitor memory during augmentation |
| Create stratified split: 80% train (480), 20% test (120) | Test class balance preservation | Generate augmentation statistics |
| Validate augmented dataset shape: (600, 1000, 64, 1) with labels | Create augmentation config templates | Verify train/test separation |

### Day 7: Dataset Export & Format Validation (ARCH_3)
| Abhishek M | Kavya | Abhishek S |
|---|---|---|
| Export normalized + augmented dataset to .npz format | Validate export file size & compression efficiency | Verify all samples exportable |
| Create dataset metadata file (forest params, augmentation configs) | Test loading speed from .npz (target <100ms for full dataset) | Generate dataset summary |
| Implement dataset loader for training pipeline | Profile I/O performance (latency, throughput) | Document file formats |
| Validate dataset integrity: shape, dtype, range checks | Create data validation checklist | Backup dataset to git-safe location |

### Day 8: Training Readiness & Pipeline Integration (ARCH_3 → ARCH_4 prep)
| Abhishek M | Kavya | Abhishek S |
|---|---|---|
| Create train/validation/test split with stratification (480/120 train) | Test end-to-end pipeline latency (ARCH_1→2→3) | Validate pipeline correctness |
| Implement data loading batches (batch_size=32, shuffle=True) | Profile memory usage across pipeline | Generate pipeline timing breakdown |
| Create ground-truth label file (600 labels: 300 gunshot, 300 other) | Test label consistency throughout pipeline | Monitor data quality metrics |
| Validate input ready for SNN training (ARCH_4) | Simulate training data flow | Prepare training environment |

### Day 9: Week 1 Integration & Checkpoint
| Abhishek M | Kavya | Abhishek S |
|---|---|---|
| End-to-end test: raw audio → normalized spike dataset (ARCH_1→2→3) | Validate energy profiler baseline measurements | Generate Week 1 summary report |
| Save intermediate checkpoints (mel-specs, spikes, normalized data) | Create baseline power consumption model | Verify all deliverables ready |
| Document data format specs for ARCH_4 (spike input requirements) | Test profiler stability over 8 days | Create deployment readiness checklist |
| Prepare final dataset (600 samples) for SNN training on Day 10 | Validate assumptions with simulated data | Confirm team readiness for Week 2 |

---

## WEEK 2: SNN Training & Optimization (Days 10-16)

### Day 10: SNN Architecture Design & Setup (ARCH_4)
| Abhishek M | Kavya | Abhishek S |
|---|---|---|
| Define SNN architecture: 2 spiking layers (128, 64 neurons) | Extend energy profiler with CPU/GPU modeling | Setup training monitoring dashboard |
| Select surrogate gradient: ArcTan with α=2.0 | Validate thermal modeling | Create hyperparameter tracker |
| Setup brian2/snnTorch framework for training | Design power-per-operation model | Monitor GPU memory during training |
| Load normalized spike dataset from ARCH_3 | | |

### Day 11: Transfer Learning - Feature Extraction Layer (ARCH_4)
| Abhishek M | Kavya | Abhishek S |
|---|---|---|
| Load pre-trained Gammatone weights (from ESC-50) | Test energy profiler on full training pipeline | Track training metrics (loss, accuracy) |
| Freeze first layer (cochlea), train only classifier | Validate power draw accuracy vs. hardware | Generate training curves |
| Reduce SNN to 2-layer architecture (128→64→2) | Test memory footprint calculations | Backup training checkpoints daily |
| Implement batch normalization for stability | | |

### Day 12: SNN Training - Cross-Entropy Loss & Backprop (ARCH_4)
| Abhishek M | Kavya | Abhishek S |
|---|---|---|
| Implement surrogate gradient descent (BPTT + tricks) | Simulate 3-forest ecosystem network topology | Validate loss convergence |
| Loss function: Cross-entropy on final spike count | Test packet routing algorithm | Monitor GPU utilization |
| Learning rate schedule: 1e-3, decay every epoch | Test message latency (target <1.5s) | Generate accuracy vs. loss plots |
| Train for 100 epochs on augmented training set (720 samples) | Vary network density: sparse, medium, dense | |

### Day 13: SNN Training - Data Augmentation & Generalization (ARCH_4)
| Abhishek M | Kavya | Abhishek S |
|---|---|---|
| Validate augmentation effectiveness from ARCH_3 | Create forest noise background model | Validate augmentation in training |
| Monitor training on all 3 forest normalization variants | Add environmental sounds to energy profiles | Verify class balance maintained |
| Re-train 100 epochs with ARCH_3 augmented data | Test simulator stability under load | Plot training vs. validation curves |
| Validate generalization across forests | | |

### Day 14: SNN Inference Optimization & Quantization (ARCH_5)
| Abhishek M | Kavya | Abhishek S |
|---|---|---|
| Implement inference-only model (no training ops) | Test energy profiler accuracy on inference | Validate inference speed |
| Implement confidence scoring: spike count ratio | Simulate real-time message batching | Monitor latency throughout pipeline |
| Quantize weights to int8 (8-bit) | Test network congestion (peak loads) | Generate inference performance metrics |
| Create model checkpoint for deployment | Validate LoRa range assumptions | Create deployment checklist |

### Day 15: Payload Encryption & JSON Structure (ARCH_6)
| Abhishek M | Kavya | Abhishek S |
|---|---|---|
| Design JSON payload schema (<1KB target) | Test energy efficiency of encrypted payloads | Validate payload size limits |
| Fields: timestamp, class_id, confidence, location_hash | Measure encryption time in energy profiler | Generate example payloads |
| Implement AES-256 encryption (64-byte keys) | Test network latency with encryption overhead | Verify encryption correctness |
| Verify payload size <1000 bytes | Create security assumptions document | Monitor payload transmission times |

### Day 16: Week 2 Integration & Model Validation
| Abhishek M | Kavya | Abhishek S |
|---|---|---|
| End-to-end inference: raw audio → JSON alert (ARCH_1→2→3→4→5→6) | Run full simulation with encrypted payloads | Validate inference across all samples |
| Validate model accuracy on test set (target >85%) | Test energy calculations for full pipeline | Generate Week 2 summary report |
| Create inference script for edge deployment | Validate latency <3 seconds end-to-end | Create model performance sheet |
| Save final model weights + metadata | | |

---

## WEEK 3: Simulation Environments & Field Validation (Days 17-20)

### Day 17: Virtual Energy Profiler - Battery Model & Validation (ARCH_7)
| Abhishek M | Kavya | Abhishek S |
|---|---|---|
| Profile actual power consumption per module | Implement battery depletion simulation | Validate profiling methodology |
| Measure: Gammatone (2mW), LIF (1mW), SNN inference (3mW) | Create solar harvesting model (0.5W@noon, 0.1W@dusk) | Run profiler on 100 random samples |
| Test quiescent power (idle listening = 0.5mW) | Model battery discharge curve (non-linear) | Generate power consumption graphs |
| Create power budget table (mW by component) | Simulate 24-hour operation cycle | Validate 5-12mW target achievable |

### Day 18: Virtual Energy Profiler - Forest Ecosystem Scenarios (ARCH_7)
| Abhishek M | Kavya | Abhishek S |
|---|---|---|
| Profile inference for Corbett (dense forest, more false alarms) | Simulate Corbett scenario (high false positive rate) | Validate scenario assumptions |
| Profile inference for Seshachalam (difficult terrain) | Simulate Seshachalam (20% packet loss, 2s latency) | Test alert delivery reliability |
| Profile inference for Sundarbans (intermittent connectivity) | Simulate Sundarbans (queue-based message buffering) | Verify message ordering |
| Generate power consumption per forest (24h estimate) | Model different network topologies | Create scenario comparison report |

### Day 19: Network Topology Simulator - LoRa Mesh & Message Routing (ARCH_8)
| Abhishek M | Kavya | Abhishek S |
|---|---|---|
| Validate JSON payloads transmit in <50ms per hop | Implement LoRa PHY model (SF7-SF12, 20dBm) | Test routing algorithm correctness |
| Verify end-to-end latency <1.5s on 5-node mesh | Create node placement model for 3 forests | Validate mesh connectivity |
| Test alert batching (max 10 alerts/message) | Simulate rain/weather interference | Generate network statistics |
| Trace real alert paths through mesh | Test message acknowledgments & retries | Create routing table examples |

### Day 20: Network Topology Simulator - Reliability & Congestion (ARCH_8)
| Abhishek M | Kavya | Abhishek S |
|---|---|---|
| Validate alert delivery rate >95% (1 miss per 20 alerts) | Test network under peak load (10 alerts/sec) | Verify congestion handling |
| Test alert queuing for intermittent connectivity | Simulate base station failure modes | Validate queue persistence |
| Measure effective throughput (bits/second through mesh) | Test message priority (urgent vs. routine alerts) | Generate reliability metrics |
| Document alert loss scenarios & impact | Create fallback routing paths | Build alert delivery SLA table + compile final deliverables |

---

## Key Metrics Tracking (Per Day)

| Metric | Abhishek M (SNN) | Kavya (Simulations) | Abhishek S (Support) |
|---|---|---|---|
| **Daily Deliverable** | Code + logs | Simulator updates + metrics | Reports + checklists |
| **Testing Focus** | Inference accuracy, latency | Energy & network accuracy | End-to-end validation |
| **Success Criteria** | >85% accuracy, <100ms per sample | ±10% energy accuracy, <3s latency | 100% checklist completion |

---

## Risks & Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| SNN training instability | Inference accuracy <85% | Day 11: implement gradient clipping + learning rate decay |
| Energy profiling inaccurate | Wrong power estimates | Day 15: validate against hardware power meter |
| Network simulator doesn't match reality | Unreliable latency predictions | Day 19-20: real-world field tests validate model |
| Payload encryption overhead too large | >1KB payloads | Day 13: implement compression before encryption |
| Inference latency >3s | Alerts too slow for deployment | Day 20: optimize inference code if needed |

---

## Dependencies Between Tasks

```
Days 1-7: Audio processing + Spike conversion (ARCH_1 + ARCH_2)
    ↓
Days 8-9: Dataset preparation (ARCH_3 - consolidation, normalization, augmentation)
    ↓
Days 10-13: SNN training (ARCH_4 - uses ARCH_3 normalized & augmented data)
    ↓
Days 14-16: Inference + payload (ARCH_5 + ARCH_6 - uses ARCH_4 trained model)
    ↓
Days 17-20: Simulation + field validation (ARCH_7 + ARCH_8 + real deployment - depends on Days 1-16)
```

**Parallel Work Streams**:
- **Abhishek M**: Audio processing (ARCH_1) → Spike generation (ARCH_2) → Dataset prep (ARCH_3) → SNN training (ARCH_4) → Inference (ARCH_5) → Payload (ARCH_6) [linear, sequential]
- **Kavya**: Setup (Day 1) → Energy profiler (ARCH_7, Days 17-18) + Network simulator (ARCH_8, Days 19-20) [can run in parallel from Day 8 onward]
- **Abhishek S**: Support tasks every day (data validation, reporting, checklists)
