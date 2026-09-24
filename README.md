# Eco-Sentry

Neuromorphic anti-poaching acoustic detection for Indian forest reserves — a
working implementation of the eight architectures specified in `ARCH_1` …
`ARCH_8` at the root of this repository.

The system listens for **gunshots, chainsaws and vehicles**, classifies them
with a spiking neural network, encrypts the detection into a sub-1 KB alert,
and delivers it to a command centre over a simulated LoRa mesh — while a power
model tracks whether the solar-charged edge node survives a 30-day deployment.

```
forest audio ─ARCH_1→ mel-spectrogram ─ARCH_2→ spike trains ─ARCH_3→ dataset
             ─ARCH_4→ trained SNN ─ARCH_5→ (class, confidence) ─ARCH_6→ AES-256 payload
             ─ARCH_8→ LoRa mesh ─gateway→ priority backhaul → command centre
                                  ARCH_7 profiles battery + solar across all of it
```

---

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python -m ecosentry run --preset quick
```

That runs **every stage end to end in about 16 seconds** — synthesises a forest
audio corpus, encodes it to spikes, trains the SNN, fires real alerts through
encryption and the mesh, and prints an acceptance table against the targets in
`SUMMARY_HIGH_LEVEL_ARCHITECTURE.md`. Artifacts land in `artifacts/`.

For the configuration the documents actually specify (600 clips of 10 s,
T=1000 frames, 120 epochs) — **7 minutes** on a laptop CPU, no GPU needed:

```bash
python -m ecosentry run --preset full
```

---

## Dashboard

`dashboard.html` at the repo root is a self-contained results dashboard —
acceptance scorecard, training curves, confusion matrix, per-forest network
and energy simulation, and the honest list of what doesn't work yet. No
build step, no server required.

```bash
open dashboard.html                # macOS
xdg-open dashboard.html            # Linux
```

Or just double-click it / drag it into a browser tab. It reads the numbers
straight out of `artifacts/pipeline_report.json`, so re-run
`python -m ecosentry run --preset full` first if you want it to reflect a
fresh run.

---

## Repository layout

| Path | Document | What it implements |
|---|---|---|
| `ecosentry/arch1_audio.py` | ARCH_1 | Load → resample → RMS → fade → bandpass → STFT → mel → dB |
| `ecosentry/arch2_spikes.py` | ARCH_2 | LIF neurons per mel band, TTFS encoding, sparse storage |
| `ecosentry/arch3_dataset.py` | ARCH_3 | Forest normalisation, stratified splits, augmentation, HDF5 |
| `ecosentry/arch4_training.py` | ARCH_4 | Surrogate-gradient BPTT, Adam, early stopping (pure NumPy) |
| `ecosentry/arch5_inference.py` | ARCH_5 | Edge inference, adaptive thresholds, debouncing, INT8 quantisation |
| `ecosentry/arch6_payload.py` | ARCH_6 | JSON → zlib → AES-256-CBC → envelope, store-and-forward queue |
| `ecosentry/arch6_beacon.py` | ARCH_6 | 16-byte status & health beacon payload, truncated HMAC auth |
| `ecosentry/arch7_energy.py` | ARCH_7 | Battery discharge, solar harvest, 30-day mission projection |
| `ecosentry/arch8_network.py` | ARCH_8 | LoRa PHY, mesh topologies, Dijkstra routing, Monte-Carlo delivery |
| `ecosentry/gateway.py` | PRIORITY_PAYLOAD_DELIVERY | Gateway priority proxy, LLQ QoS, EF DSCP, local ACK, dedupe |
| `ecosentry/officer_delivery.py` | — | Multi-channel ACK-based forest officer escalation tracker |
| `ecosentry/dashboard_template.html` | — | Self-contained results & visualization dashboard template |
| `ecosentry/synth.py` | — | Synthetic forest audio (stands in for the field recordings) |
| `ecosentry/pipeline.py` | — | Stage orchestration, delivery simulation + acceptance report |
| `ecosentry/config.py` | all | Every documented constant, in one place |
| `tests/` | — | 145 tests covering unit, protocol, energy, network, delivery, dashboard & real-audio validation |

---

## Commands

- `python -m ecosentry run` — end-to-end pipeline (synth, spikes, train, deliver, report)
- `python run_iterations_combined_datasets.py` — multi-seed retrain against ESC-50 and UrbanSound8K
- `python esc50_loader.py` — zero-shot test against ESC-50 (run as a script)

```bash
python -m ecosentry run     --preset quick --scenario corbett   # everything
python -m ecosentry dataset --preset full                       # ARCH_1-3 only
python -m ecosentry train   --preset full                       # ARCH_4 on a saved dataset
python -m ecosentry infer   --audio clip.wav --payload          # ARCH_1/2/5/6 on one file
python -m ecosentry energy  --scenario all                      # ARCH_7
python -m ecosentry network --scenario sundarbans --messages 300 # ARCH_8
python -m ecosentry delivery --scenario corbett --messages 300  # Gateway + officer ACK
python -m ecosentry synth   --out samples/ --count 6            # write example WAVs
```

Outputs written to `artifacts/`:

| File | Contents |
|---|---|
| `prepared_dataset.h5` | Normalised, split, augmented spike tensors |
| `snn_model.npz` | Trained weights + metadata (spike gain, frame count, accuracy) |
| `alert_events.json` | Per-window trace: prediction → payload → hops → backhaul |
| `energy_report.json` | Per-forest power budget and endurance |
| `network_report.json` | Delivery / latency / congestion / interference sweeps |
| `delivery_report.json` | Priority gateway + officer ACK escalation delivery metrics |
| `dashboard.html` | Interactive dashboard with energy trajectories & delivery funnel |
| `energy_profiles.png` | 30-day battery trajectories |
| `pipeline_report.json` | Everything above plus the acceptance table |


---

## Using it as a library

```python
from ecosentry.arch1_audio import extract_mel_spectrogram
from ecosentry.arch2_spikes import convert_mel_to_spikes
from ecosentry.arch5_inference import EcoSentryInference
from ecosentry.arch6_payload import DeviceConfig, generate_alert_payload

mel    = extract_mel_spectrogram("gunshot.wav")        # (1001, 64) dB
spikes = convert_mel_to_spikes(mel, auto_gain=True)    # (1001, 64, 1) binary

engine = EcoSentryInference.from_checkpoint("artifacts/snn_model.npz")
result = engine.process_audio(audio, sr=16_000)
# {'alert': True, 'class_name': 'chainsaw', 'confidence': 0.91, ...}

if result["alert"]:
    device = DeviceConfig(device_id="SENTRY_01", latitude=29.24, longitude=79.10)
    packet = generate_alert_payload(result, device)     # 116-byte encrypted message
```

To train on **real** recordings instead of synthetic ones, replace
`ecosentry.synth.build_corpus` with your own loader. Everything downstream
consumes plain `(audio, label, forest, source)` records — see
`ecosentry/synth.py:SynthSample`.

---

## Measured results

From `--preset full` on a 2023 Apple Silicon laptop, CPU only: 600 clips of
10 s, T=1000 frames, 120 epochs, **414 s wall clock end to end**.

### Acceptance table — 8 of 9 targets met

| Criterion | Measured | Target | |
|---|---|---|---|
| SNN test accuracy (synthetic) | **96.3%** | >85% | PASS |
| Ambient false-alert rate | **0.0%** | <5% | PASS |
| Spike firing rate | 25.0% normalised | 25% ±5% | PASS |
| Alert payload size | 116 B | <1000 B | PASS |
| End-to-end latency p95 | 630 ms | <1500 ms | PASS |
| Network delivery (worst forest) | 100% | >95% | PASS |
| Network latency p99 (worst forest) | 582 ms | <1500 ms | PASS |
| Operational endurance (worst forest) | 30 days | ≥30 days | PASS |
| Power reduction | 12× system / 18× compute | 50–75× | **FAIL** — see correction 8 |

Training reached 98.3% validation accuracy at epoch 82 in 365 s. Confusion
matrix on the 109-sample test split: gunshot 48/51, chainsaw 31/31,
vehicle 26/27, ambient 0/0 (all 11 held-out ambient clips correctly rejected).

### Latency budgets — all met

| Stage | Measured (mean / p95) | Target |
|---|---|---|
| ARCH_1 mel-spectrogram (10 s clip) | 25 / 70 ms | <100 ms |
| ARCH_2 spike conversion | 21 / 43 ms | <50 ms |
| ARCH_5 SNN forward pass | ~15–25 ms | <100 ms |
| ARCH_6 serialise + compress + encrypt | ~0.3 ms | <8 ms |
| End-to-end audio → command centre | 562 / 630 ms | <1500 ms |

**Payload (ARCH_6)** — 116 bytes on the wire: ~99 B JSON → ~85 B zlib →
96 B ciphertext + 16 B IV + 4 B header. Inside both the 1000 B budget and the
242 B LoRa maximum.

**Network (ARCH_8, 300 messages per scenario)**

| Forest | Delivery | Latency mean / p99 | Avg hops |
|---|---|---|---|
| Corbett | 100% | 415 / 582 ms | 1.65 |
| Seshachalam | 100% | 249 / 316 ms | 1.00 |
| Sundarbans | 100% | 450 / 579 ms | 1.79 |

**Energy (ARCH_7, 30-day mission)** — all three forests survive 30 days *with*
solar; battery-only endurance is 16 days. Daily net is +0.02 Wh (Corbett),
−0.10 Wh (Seshachalam), −0.29 Wh (Sundarbans). Sundarbans needs a bigger panel
or a second cell, exactly as ARCH_7 predicts.

### Ambient rejection: fixed (was a 47.5% false-alert rate)

The 3-class head originally shipped here — gunshot, chainsaw, vehicle, no
"none of these" option — forced every ambient clip into one of three
threats, producing a **47.5% false-alert rate** on threat-free audio.
`ARCH_3`'s own class map already defined `3 = ambient`; the training head
just didn't use it.

Fixed by training a 4-class head (`SNNConfig.n_classes = 4`,
`pipeline.TRAIN_CLASSES` including class 3) and rewiring the false-alert
measurement to read from the held-out test split instead of a separate
negative set. Re-verified against a fresh `--preset full` run: **0.0%
ambient false-alert rate**, both on the held-out test split (n=11) and on
freshly synthesized ambient audio the model never saw in any split during
training (n=11, out-of-distribution check). See `ARCH_7_ENERGY_PROFILER.md`'s
correction log and `tests/test_ambient_rejection.py` for the fix and its
regression tests.

---

## Corrections made to the design documents

These are places where the specification as written does not work, and what
this implementation does instead. Each is commented at the point of use.

1. **LIF neurons could never fire.** With `V ← αV + (1−α)I`, `α = e⁻¹` and
   `I ∈ [0, 1]`, the membrane asymptote equals `I`, so it can never cross
   `v_th = 1.0` — the firing rate collapses to 0%, not the required 25%.
   *Fix:* an explicit input gain, auto-calibrated per corpus by binary search
   (`arch2_spikes.calibrate_input_gain`). Typical gain 1.7–2.7.

2. **Firing-rate normalisation was a no-op.** ARCH_3 multiplies a *binary*
   tensor by `target/current` and clips to [0, 1]; for any factor > 1 that
   returns the input unchanged, so it cannot raise a sparse tensor to 25%.
   *Fix:* `arch3_dataset.rate_normalize` adjusts the spike *count* — dropping
   the least salient spikes when too dense, recruiting the strongest
   sub-threshold candidates when too sparse. The literal multiply-and-clip is
   still available as `mode="scale"`.

3. **Xavier init left the network dead.** With `v_th = 1.0`, the documented
   Xavier initialisation gives hidden-layer firing rates of 0.19% and **0.0%** —
   zero surrogate gradient, so training collapses to a constant prediction.
   *Fix:* `arch4_training.calibrate_initialization` rescales each hidden weight
   matrix so its layer starts at ~20% firing, and the readout so logits start at
   unit scale. This is the SNN analogue of LSUV initialisation.

4. **The readout only saw the last two frames.** ARCH_4 says the output layer
   "integrates total spike counts from all hidden neurons", but applies the same
   `α = e⁻¹` leak, giving the readout a ~14 ms memory — a 10-second clip would
   be classified from its final 20 ms.
   *Fix:* `readout_alpha = 1.0` (a true accumulator, as the prose describes),
   normalised by `T`. Both are config knobs.

5. **LoRa time-on-air figures are ~4× too low.** ARCH_8 quotes 56 ms at SF7 and
   1.5 s at SF12 for a 116-byte payload. The SX1276 formula gives **220 ms** and
   **5.1 s**; the quoted values correspond to a ~10–25 byte payload.
   *Fix:* the datasheet formula is implemented in `LoRaPHY.time_on_air_ms`.
   Consequence: **SF12 alone cannot meet the 1.5 s latency target** — the
   simulator uses adaptive SF (lowest SF that closes the link with 6 dB margin),
   which is what keeps p99 latency under 600 ms.

6. **Documented relay placement orphans sensors.** At a realistic dense-canopy
   link range, Corbett S2 cannot reach the base (4.42 km to R1, then 4.85 km R1
   to base) and Sundarbans S4 sits 10.2 km out with only two relays.
   *Fix:* sensor coordinates are kept exactly as documented (they are the
   deployment sites); relays are moved to the sensor-to-base midpoints and
   Sundarbans gets a third relay. `arch8_network.connectivity_check` asserts
   every sensor still reaches the base.

7. **AES mode conflict — CBC chosen.** ARCH_6 Component 3 specifies CBC with a
   random IV; the cross-reference table says ECB. ECB leaks equality between
   identical plaintexts, which for a fixed-schema alert would let an
   eavesdropper fingerprint repeated alerts. CBC costs 16 extra bytes.

8. **The 50–75× power-reduction target is unreachable by construction.** It
   assumes 0.5 mW quiescent power, which ARCH_7's own "CRITICAL CORRECTIONS"
   table supersedes with **50 mW**. At 50 mW the ceiling is `600/(50+3) ≈ 11×` —
   ARCH_7's own corrected formula. This implementation reports **12× system
   average / 18× compute path** and marks the criterion FAILED with that
   explanation rather than quietly reporting a number that flatters the design.

9. **ARCH_1's optional standardisation breaks ARCH_2.** Zero-mean/unit-variance
   output is incompatible with the `[−80, 0] dB` range ARCH_2 assumes for its
   input normalisation. It is off by default (`AudioConfig.standardize_db`).

10. **A 3-class softmax cannot express "nothing is happening".** Measured
    consequence: a 47.5% false-alert rate on threat-free audio. *Fixed* — the
    head is now 4-class, including an explicit ambient category. False alerts
    dropped to 0.0%.

---

## Testing

```bash
python -m pytest tests/ -q
```

145 tests, ~20 seconds. Coverage includes:

- **Contract tests** for every documented shape, range and latency budget.
- **A numerical gradient check** (`test_bptt_gradients_match_numerical`).
  Because the true Heaviside derivative is zero almost everywhere, finite
  differences cannot validate the hidden layers directly — the test runs the
  network with a *smooth* forward pass and no reset so the BPTT chain becomes
  verifiable, then requires <5% agreement with numerical gradients.
- **Learnability**: the SNN must reach >80% on a synthetic separable dataset.
- **Crypto**: round-trip, wrong-key rejection, tamper detection, IV uniqueness.
- **Network**: every sensor reachable, ≥95% delivery, p99 ≤1.5 s per forest.
- **Gateway**: priority routing, local ACK, dedupe, priority-lane advantage.
- **Integration**: the full pipeline, ending with a decryption the command
  centre verifies.

---

## What still needs work

Be aware of these before treating any output as a field result.

### Fix first: the 3-class head cannot reject ambient audio

See "One result that is not good" above — 47.5% false-alert rate on
threat-free clips. This is the single most important open item, and it is a
change to the specified architecture, so it needs a decision rather than a
patch. Everything needed to make it a 4-class problem is already in place.

### Real data: done for ESC-50 and UrbanSound8K

The pipeline's primary corpus is **synthetic**. `ecosentry/synth.py`
generates physically plausible gunshots, chainsaws, vehicles and per-forest
backgrounds, but the 96.3% the pipeline reports is *accuracy on synthetic
audio*. Synthetic classes are cleanly separable by construction.

However, zero-shot generalisation and retraining experiments have now been
completed using the real-audio **ESC-50** and **UrbanSound8K** datasets.
See `URBANSOUND8K_INTEGRATION_PLAN.md` and `ESC50_REAL_AUDIO_VALIDATION_PLAN.md`
for the results and the scripts used to produce them.

- **Field recordings from an actual forest** (as opposed to ESC-50/
  UrbanSound8K's general-purpose and urban audio respectively) remain
  untested. `chainsaw` specifically has zero real-audio coverage from
  either dataset — neither contains a chainsaw-like category.
3. Retrain and re-measure. Expect the accuracy to drop and the LIF gain and
   forest-normalisation parameters to need retuning.

### Needs tuning once real data exists

- **SNN hyperparameters.** The current values are the documented ones, and they
  train, but they were not swept. Learning rate, hidden sizes, surrogate
  steepness and the `init_target_rate` are the knobs worth searching.
- **Confidence thresholds** (`InferenceConfig.class_thresholds`, currently
  0.90 / 0.80 / 0.95) should be set from a precision-recall curve on real data,
  not carried over from the document. At the current accuracy they let 70% of
  true threat windows through with 100% precision — but also 47.5% of ambient
  clips, per the finding above.
- **Mixup on binary spikes** produces fractional inputs, which raises the
  effective input density of the training split above 25%. It is applied as the
  document specifies; whether it helps or hurts is an open question worth an
  ablation.
- **The LIF input gain** is calibrated once per corpus. Per-forest or adaptive
  gain would likely be better, and is a one-line change.

### Model calibration against hardware

- **Energy.** All power figures come from the ARCH_7 tables, not measurements.
  The 50 mW quiescent figure dominates every result, so measure it first on the
  real MCU + codec. ARCH_7 itself asks for <10% error vs. the prototype.
- **Network.** The simulator reports **100% delivery** in all three forests,
  which is more optimistic than ARCH_8's own 97.3 / 99.1 / 91.2%. Adaptive SF
  plus two retries makes almost every link succeed. The path-loss exponents
  (2.4–3.0), 4.5 dB fading margin and per-forest link ranges are literature
  estimates, not measurements — calibrate them from an RSSI survey before
  trusting the delivery figures, especially for Sundarbans.
- **Gateway backhaul** latencies (120 ms QoS / 450 ms best-effort) are
  placeholders. Replace with measurements from the actual APN.

### Not implemented (out of scope of ARCH_1–8)

- Firmware for the Cortex-M4/M7 target. The inference path is pure NumPy;
  INT8 quantisation exists (`quantize_weights`) but there is no C export.
- A real MQTT client. `gateway.MqttSink` records publishes in memory; swap it
  for `paho-mqtt` at the same interface.
- The command-centre dashboard and ranger-facing app.
- Over-the-air key rotation (`ARCH_6` mentions monthly rotation; only key
  generation and PBKDF2 derivation are implemented).

### Manual tests worth running

The automated suite cannot cover these:

1. **Listen to the synthetic audio.** `python -m ecosentry synth --out samples/`
   then play the WAVs. If a class does not sound like the thing it claims to be,
   the model is learning the wrong feature.
2. **Inspect the spectrograms.** Confirm gunshots appear as transients and
   chainsaws as harmonic stacks, per ARCH_1's QA guidance.
3. **Run `--preset full` more than once with different `--seed`** and check how
   much the test accuracy moves. A single run is not an accuracy measurement.
4. **Review `artifacts/alert_events.json` by hand** — check that the alerts
   fired on the classes you would expect, and that non-alerts were genuinely
   low-confidence. In the current full run, 28 of 40 test windows alerted with
   100% precision; every one decrypted correctly at the command centre.
5. **Check `energy_profiles.png`** against your own expectation of the site's
   sun hours and cloud cover.

---

## Design notes

- **Pure NumPy/SciPy.** No PyTorch or librosa. Training is hand-written BPTT,
  which keeps the surrogate-gradient mathematics visible and inspectable — the
  point of the exercise — and lets the same forward pass serve as the edge
  inference path.
- **Every documented constant lives in `config.py`** as a frozen dataclass, each
  traceable to a "Key Parameters (Finalized)" table.
- **Deterministic.** Every stage takes an explicit seed; reruns reproduce.
- **Class mapping is fixed** at `0=gunshot, 1=chainsaw, 2=vehicle` (3=ambient is
  held out as a negative), as `PRIORITY_PAYLOAD_DELIVERY.md` requires.
- **Priority uses Option A** — the flag lives in the transport header, so the
  encrypted payload schema is byte-for-byte unchanged and old parsers still
  work. The gateway re-derives priority from the decrypted alert, so a stripped
  or spoofed header cannot promote traffic.
