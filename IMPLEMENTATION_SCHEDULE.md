# Eco-Sentry Implementation Schedule (20 Days)

**Project Duration**: 20 calendar days  
**Team**: Abhishek M (SNN Pipeline), Kavya (Simulations), Abhishek S (Support)  
**Target**: Gunshot/chainsaw detection with <1KB JSON alerts, 5-12mW power consumption

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
| Document alert loss scenarios & impact | Create fallback routing paths | Build alert delivery SLA table |

### Day 21: Field Validation - Corbett National Park (Tier 1)
| Abhishek M | Kavya | Abhishek S |
|---|---|---|
| Deploy on 2 edge devices in Corbett | Log energy consumption in real conditions | Document deployment procedures |
| Test inference on live audio (birds, vehicles, silence) | Monitor network topology stability | Record all metrics to database |
| Measure actual latency (target <3s end-to-end) | Track power consumption vs. predictions | Compare simulations to reality |
| Validate alert generation (confidence >0.85 for threats) | | Validate data collection scripts |

### Day 22: Field Validation - Seshachalam + Sundarbans (Tier 2-3), Summary Docs
| Abhishek M | Kavya | Abhishek S |
|---|---|---|
| Deploy on edge devices in Seshachalam & Sundarbans | Summarize energy profiler accuracy (target ±10%) | Generate final report |
| Validate inference accuracy across 3 forests | Summarize network simulator results | Create deployment guide |
| Compare real vs. predicted power consumption | Validate 50-75x power reduction achieved | Create operational handbook |
| Finalize model weights & inference code | Document all assumptions & limitations | Archive all test data |
| | | Create high-level summary architecture doc |

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
Days 17-22: Simulation + field validation (ARCH_7 + ARCH_8 + real deployment - depends on Days 1-16)
```

**Parallel Work Streams**:
- **Abhishek M**: Audio processing (ARCH_1) → Spike generation (ARCH_2) → Dataset prep (ARCH_3) → SNN training (ARCH_4) → Inference (ARCH_5) → Payload (ARCH_6) [linear, sequential]
- **Kavya**: Setup (Day 1) → Energy profiler (ARCH_7, Days 17-18) + Network simulator (ARCH_8, Days 19-20) [can run in parallel from Day 8 onward]
- **Abhishek S**: Support tasks every day (data validation, reporting, checklists)
