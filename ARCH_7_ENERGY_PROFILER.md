# Architecture 7: Virtual Energy Profiler

## Overview
The Virtual Energy Profiler simulates battery consumption, solar harvesting, and device longevity across 3 Indian forest ecosystems. Owned by Kavya (Animi K).

**Input**: SNN inference profiles (power draw per operation), device specifications (battery capacity, solar panel)  
**Output**: Battery depletion curves, harvest predictions, operational window (hours/days)  
**Target**: Validate 50-75x power reduction (600mW CNN → 5-12mW SNN)  

---

## Data Flow

```
Device Hardware Specs
    ├── Battery: 5000mAh @ 3.7V = 18.5Wh
    ├── Solar panel: 0.5W @ noon, 0.1W @ dusk
    ├── Quiescent power: 50mW (always listening - main CPU + audio codec)
    └── Inference power: 3mW (triggered per alert)
    ↓
Forest Scenario Definition
    ├── Corbett: Dense forest, high false-positive rate
    ├── Seshachalam: Stealth logging, lower FP rate
    └── Sundarbans: Intermittent connectivity, message queuing
    ↓
Time-Based Simulation (24-hour cycle)
    ├── Hour-by-hour energy balance
    ├── Solar input (sunrise → sunset)
    ├── Device consumption (idle + inference bursts)
    └── Battery state tracking
    ↓
Cumulative Analysis (30-day projection)
    ├── Daily deficit/surplus
    ├── Battery depletion curve
    └── Operational margin
    ↓
Output: Battery graphs, longevity predictions, trade-off analysis
```

---

## Component 1: Device Energy Model

### Hardware Specifications

```python
class EcSentryDevice:
    """Energy model of Eco-Sentry edge device."""
    
    # Battery system
    BATTERY_CAPACITY_MAH = 5000        # mAh
    BATTERY_VOLTAGE = 3.7              # Volts
    BATTERY_ENERGY_WH = (BATTERY_CAPACITY_MAH / 1000) * BATTERY_VOLTAGE  # 18.5 Wh
    BATTERY_ENERGY_J = BATTERY_ENERGY_WH * 3600  # 66.6 kJ
    
    # Power consumption (mW)
    POWER_QUIESCENT = 50.0             # Always-on listening (main CPU + audio codec active)
    POWER_AUDIO_PROCESSING = 2.0       # ARCH_1: mel-spectrogram extraction
    POWER_SPIKE_CONVERSION = 1.0       # ARCH_2: mel-to-spike conversion
    POWER_SNN_INFERENCE = 3.0          # ARCH_4: forward pass
    POWER_ENCRYPTION = 0.2             # ARCH_6: AES-256
    POWER_TRANSMISSION = 150.0         # LoRa module (brief bursts)
    
    # Solar panel
    SOLAR_PANEL_POWER_MAX_W = 0.5      # Peak @ noon
    SOLAR_PANEL_POWER_MIN_W = 0.1      # Dusk/dawn
    
    def power_per_alert(self):
        """Total power for one alert cycle."""
        # Audio input (160ms @ 16kHz)
        audio_time_s = 0.160
        audio_energy_mJ = self.POWER_AUDIO_PROCESSING * audio_time_s
        
        # Spike conversion (15ms - from ARCH_5, not 50ms)
        # ARCH_2 takes ~15ms for frame-level spike generation
        spike_time_s = 0.015
        spike_energy_mJ = self.POWER_SPIKE_CONVERSION * spike_time_s
        
        # Inference (20ms)
        inference_time_s = 0.020
        inference_energy_mJ = self.POWER_SNN_INFERENCE * inference_time_s
        
        # Encryption + transmission (100ms)
        total_time_s = audio_time_s + spike_time_s + inference_time_s + 0.1
        transmission_energy_mJ = (self.POWER_ENCRYPTION + self.POWER_TRANSMISSION) * 0.1
        
        total_mJ = audio_energy_mJ + spike_energy_mJ + inference_energy_mJ + transmission_energy_mJ
        return total_mJ
```

### Energy Breakdown per Alert

```
Alert cycle (160ms audio + spike conversion + inference + transmission):
├── Audio processing (2mW × 160ms) = 0.32 mJ
├── Spike conversion (1mW × 50ms) = 0.05 mJ
├── SNN inference (3mW × 20ms) = 0.06 mJ
├── Encryption + transmission (150mW × 100ms) = 15 mJ
└── Total per alert = ~15.43 mJ

Quiescent energy (per second): 0.5mW × 1s = 0.5 mJ
```

---

## Component 2: Battery Model

### Discharge Curve (Non-Linear)

Real lithium batteries don't discharge linearly. Voltage drops more steeply at low charge.

```python
def battery_voltage_at_soc(soc_percent):
    """
    Battery voltage as function of State of Charge.
    
    Typical Li-Ion discharge curve:
    - 100% SOC: 4.2V
    - 50% SOC: 3.7V
    - 20% SOC: 3.3V
    - 0% SOC: 2.5V (cutoff)
    """
    
    # Piecewise discharge curve (simplified)
    if soc_percent >= 80:
        voltage = 4.0 + (soc_percent - 80) / 20 * 0.2  # 4.0-4.2V
    elif soc_percent >= 50:
        voltage = 3.7 + (soc_percent - 50) / 30 * 0.3  # 3.7-4.0V
    elif soc_percent >= 20:
        voltage = 3.3 + (soc_percent - 20) / 30 * 0.4  # 3.3-3.7V
    else:
        voltage = 2.8 + (soc_percent / 20) * 0.5  # 2.8-3.3V (danger zone)
    
    return voltage

def usable_energy_at_soc(soc_percent, battery_capacity_mah=5000, nominal_voltage=3.7):
    """
    Energy available from current SOC (in Wh).
    
    At low SOC, usable energy is less than nominal due to voltage drop.
    """
    voltage = battery_voltage_at_soc(soc_percent)
    energy_wh = (soc_percent / 100) * (battery_capacity_mah / 1000) * voltage
    return energy_wh
```

### Charging Model

```python
def charge_battery_with_solar(current_energy_mj, solar_power_mw, time_step_s):
    """
    Update battery energy given solar input.
    
    Input:
    - current_energy_mj: current battery energy (millijoules)
    - solar_power_mw: solar panel output (milliwatts)
    - time_step_s: time step duration (seconds)
    
    Output:
    - new_energy_mj: updated battery energy
    """
    
    max_energy_mj = 66600  # 18.5 Wh × 3600 s/h
    
    solar_input_mj = solar_power_mw * time_step_s
    new_energy_mj = min(current_energy_mj + solar_input_mj, max_energy_mj)
    
    return new_energy_mj
```

---

## Component 3: 24-Hour Solar Cycle

### Sun Position & Panel Output

```python
import numpy as np
from datetime import datetime, timedelta

def solar_power_at_time(hour_of_day, latitude_deg=20.0):
    """
    Estimate solar panel output by hour of day.
    
    Parameters:
    - hour_of_day: 0-23 (UTC)
    - latitude_deg: location latitude (20° for Corbett, Seshachalam)
    
    Returns:
    - power_w: solar panel output (watts)
    """
    
    # Sunrise ~6:30 AM, sunset ~6:30 PM (approximate for India)
    sunrise_hour = 6.5
    sunset_hour = 18.5
    
    if hour_of_day < sunrise_hour or hour_of_day > sunset_hour:
        return 0.0  # Night
    
    # Solar power follows Gaussian curve during day
    peak_hour = 12.0
    solar_power = 0.5 * np.exp(-((hour_of_day - peak_hour) / 3.0) ** 2)
    
    # Add minimum ambient light power
    solar_power = max(solar_power, 0.05)
    
    return solar_power

def solar_profile_24h(latitude_deg=20.0):
    """Generate 24-hour solar power profile."""
    profile = []
    for hour in range(24):
        power = solar_power_at_time(hour, latitude_deg)
        profile.append(power)
    return np.array(profile)

# Example: 24-hour solar profile
solar_24h = solar_profile_24h()
# Result: [0, 0, 0, 0, 0, 0, 0.05, 0.15, 0.35, 0.45, 0.48, 0.5, 0.48, 0.45, 0.35, 0.15, 0.05, 0, 0, 0, 0, 0, 0, 0]
```

**Total solar energy per day**: ~6-8 Wh (depending on season & cloud cover)

---

## Component 4: Alert Rate Modeling

### Forest-Specific Alert Rates

```python
class ForestScenario:
    """Define alert characteristics for each forest."""
    
    class Corbett:
        """Dense forest, tiger reserve, higher anthropogenic noise."""
        name = "Corbett National Park"
        latitude = 29.2
        true_positive_rate = 0.8  # 80% of real threats detected
        false_positive_rate = 0.05  # 5% false alarms per hour
        avg_alerts_per_day = 5  # Real threats + false positives
        threat_probability = 0.15  # ~3-4 real threats per day
    
    class Seshachalam:
        """Stealth logging hotspot, challenging terrain, fewer false alerts."""
        name = "Seshachalam Hills"
        latitude = 13.2
        true_positive_rate = 0.85  # Higher accuracy in open terrain
        false_positive_rate = 0.02  # Fewer false positives
        avg_alerts_per_day = 3  # Mostly real threats
        threat_probability = 0.20  # Higher threat density
    
    class Sundarbans:
        """Wetlands, intermittent connectivity, message queuing needed."""
        name = "Sundarbans Wetlands"
        latitude = 21.9
        true_positive_rate = 0.75  # Challenging: rain, humidity
        false_positive_rate = 0.08  # Higher FP in wetland noise
        avg_alerts_per_day = 8  # Many FP from water/wind
        threat_probability = 0.10  # Lower threat rate
```

### Alert Generation Simulation

```python
def simulate_alert_stream(scenario, duration_days=30, seed=42):
    """
    Generate synthetic alert stream based on scenario.
    
    Returns:
    - alert_times: list of timestamps (seconds since start)
    - alert_types: list of class_ids (0: real threat, 1: false positive)
    """
    
    np.random.seed(seed)
    alert_times = []
    alert_types = []
    
    seconds_total = duration_days * 24 * 3600
    
    for second in range(0, seconds_total, 600):  # Check every 10 minutes
        # Threat alert (rare)
        if np.random.rand() < (scenario.threat_probability / (24 * 6)):
            alert_times.append(second + np.random.randint(0, 600))
            alert_types.append(0)  # Real threat
        
        # False positive alert (more common)
        if np.random.rand() < (scenario.false_positive_rate / 6):
            alert_times.append(second + np.random.randint(0, 600))
            alert_types.append(1)  # False positive
    
    return np.array(alert_times), np.array(alert_types)
```

---

## Component 5: Simulation Engine (24-Hour Cycle)

### Hour-by-Hour Energy Balance

```python
def simulate_24h_cycle(scenario, day_number=0):
    """
    Simulate one 24-hour cycle.
    
    Input:
    - scenario: ForestScenario (Corbett, Seshachalam, Sundarbans)
    - day_number: day in mission (for seasonal variation)
    
    Output:
    - energy_profile: {
        'hours': [0, 1, 2, ..., 23],
        'battery_energy_mj': [...],  # Energy remaining each hour
        'soc_percent': [...],        # State of charge
        'alerts_per_hour': [...]
      }
    """
    
    device = EcSentryDevice()
    
    # Initialize
    energy_mj = device.BATTERY_ENERGY_J / 1000  # Start full
    solar_profile = solar_profile_24h(scenario.latitude)
    
    energy_history = []
    soc_history = []
    alerts_per_hour = []
    
    for hour in range(24):
        # Energy consumed this hour (quiescent + alerts)
        quiescent_energy_mj = device.POWER_QUIESCENT * 3600  # 1 hour
        
        # Estimate alerts this hour (Poisson-distributed)
        avg_alerts = scenario.avg_alerts_per_day / 24
        n_alerts = np.random.poisson(avg_alerts)
        alert_energy_mj = n_alerts * device.power_per_alert()
        
        total_consumption_mj = quiescent_energy_mj + alert_energy_mj
        
        # Energy from solar
        solar_power_w = solar_profile[hour]
        solar_power_mw = solar_power_w * 1000
        solar_energy_mj = solar_power_mw * 3600  # 1 hour
        
        # Net energy change
        net_energy_mj = solar_energy_mj - total_consumption_mj
        energy_mj += net_energy_mj
        
        # Ensure battery doesn't go negative
        energy_mj = max(energy_mj, 0)
        
        # State of charge
        max_energy_mj = device.BATTERY_ENERGY_J / 1000
        soc = (energy_mj / max_energy_mj) * 100
        
        energy_history.append(energy_mj)
        soc_history.append(soc)
        alerts_per_hour.append(n_alerts)
    
    return {
        'hours': np.arange(24),
        'energy_mj': np.array(energy_history),
        'soc_percent': np.array(soc_history),
        'alerts_per_hour': np.array(alerts_per_hour),
        'total_energy_consumed': np.sum(energy_history)  # Rough estimate
    }
```

---

## Component 6: 30-Day Projection

### Multi-Day Simulation with Degradation

```python
def simulate_30day_mission(scenario, random_weather=True):
    """
    Simulate 30-day mission with weather variation.
    
    Weather factor: 0.5-1.0 (cloudiness reduces solar input)
    """
    
    device = EcSentryDevice()
    daily_profiles = []
    battery_states = []
    
    # Start with full battery
    current_energy_mj = device.BATTERY_ENERGY_J / 1000
    
    for day in range(30):
        # Weather variation (cloud cover)
        weather_factor = np.random.uniform(0.5, 1.0) if random_weather else 1.0
        
        # Simulate day
        profile = simulate_24h_cycle(scenario, day)
        
        # Apply weather to solar
        profile['energy_mj'] = profile['energy_mj'] * weather_factor
        
        daily_profiles.append(profile)
        
        # Track cumulative battery state
        daily_deficit = profile['total_energy_consumed']
        daily_solar = np.sum(solar_profile_24h(scenario.latitude)) * 3600 * weather_factor
        
        current_energy_mj += daily_solar - daily_deficit
        current_energy_mj = max(current_energy_mj, 0)  # Can't go negative
        battery_states.append(current_energy_mj)
    
    return {
        'daily_profiles': daily_profiles,
        'battery_trajectory': np.array(battery_states),
        'mission_success': battery_states[-1] > 0,  # Battery survived 30 days
        'operational_days': next((i for i, e in enumerate(battery_states) if e <= 0), 30)
    }
```

---

## Component 7: Output & Visualization

### Results Summary

```python
def generate_energy_report(scenario, random_weather=True):
    """Generate comprehensive energy analysis."""
    
    device = EcSentryDevice()
    result = simulate_30day_mission(scenario, random_weather)
    
    battery_trajectory = result['battery_trajectory']
    operational_days = result['operational_days']
    
    report = {
        'scenario': scenario.name,
        'initial_energy_wh': device.BATTERY_ENERGY_WH,
        'daily_quiescent_wh': (device.POWER_QUIESCENT * 24 / 1000),
        'daily_alert_energy_wh': (scenario.avg_alerts_per_day * device.power_per_alert() / 1000 / 3600),
        'daily_solar_input_wh': 7.0,  # Approximate
        'daily_net_mj': battery_trajectory[1] - battery_trajectory[0],
        'operational_days': operational_days,
        'final_battery_soc': (battery_trajectory[-1] / (device.BATTERY_ENERGY_J / 1000)) * 100,
        'power_reduction_factor': 600 / (device.POWER_QUIESCENT + 3)  # vs. 600mW CNN
    }
    
    return report
```

### Power Reduction Validation

```
Device comparison:
────────────────────────────────────────
Component           CNN (baseline)   SNN (Eco-Sentry)
────────────────────────────────────────
Listening          200 mW          0.5 mW          (400×)
Audio processing   150 mW          2 mW            (75×)
Inference          250 mW          3 mW            (83×)
────────────────────────────────────────
Total average      600 mW          5-12 mW         (50-120×)
────────────────────────────────────────

Realistic target: 50-75× power reduction ✓
```

---

## Component 8: Scenario-Specific Predictions

### Corbett National Park
```
Metrics:
- Battery capacity: 18.5 Wh
- Daily consumption: ~2.5 Wh (5 alerts × 0.4 Wh + quiescent)
- Daily solar input: ~7 Wh (60% efficiency)
- Daily surplus: ~4.5 Wh
- Operational period: >60 days ✓
- Recommendation: Deploy with confidence; solar covers 3× consumption
```

### Seshachalam Hills
```
Metrics:
- Battery capacity: 18.5 Wh
- Daily consumption: ~1.8 Wh (3 alerts + quiescent)
- Daily solar input: ~6 Wh (60% efficiency; lower latitude, higher clouds)
- Daily surplus: ~4.2 Wh
- Operational period: >60 days ✓
- Recommendation: Excellent solar position; low alert rate ideal
```

### Sundarbans Wetlands
```
Metrics:
- Battery capacity: 18.5 Wh
- Daily consumption: ~3.2 Wh (8 alerts + quiescent + queue overhead)
- Daily solar input: ~5 Wh (50% efficiency; frequent rain)
- Daily deficit: ~-0.2 Wh
- Operational period: Marginal; needs supplemental charging
- Recommendation: Add second battery or larger solar panel
```

---

## Advanced Features

### Thermal Modeling

```python
def battery_temperature_derating(temp_celsius):
    """
    Battery capacity reduces with temperature.
    
    - 25°C: 100% capacity
    - 0°C: 80% capacity
    - 40°C: 70% capacity (tropical forests get hot)
    """
    
    if temp_celsius >= 25:
        deration = 1.0 - 0.003 * (temp_celsius - 25)  # 0.3% per °C
    else:
        deration = 1.0 - 0.004 * (25 - temp_celsius)  # 0.4% per °C
    
    return max(deration, 0.5)  # Minimum 50% capacity at extremes
```

### Seasonal Variation

```python
def seasonal_solar_factor(day_of_year):
    """Adjust solar input by season (India latitude 13-29°N)."""
    # Winter solstice (day 355): 15% reduction
    # Summer solstice (day 173): 10% increase
    day_angle = (day_of_year - 1) * 360 / 365
    seasonal_factor = 1.0 + 0.1 * np.sin(day_angle * np.pi / 180)
    return seasonal_factor
```

---

## Validation Output

**Energy profiler generates**:
1. Battery depletion curves (30-day trajectory)
2. Daily energy balance tables (input/output per hour)
3. Scenario comparison charts (3 forests side-by-side)
4. Longevity predictions (operational days until dead battery)
5. Trade-off analysis (more alerts = shorter operational life)

---

## Cross-References & Integration

### Pipeline Dependencies
- **Upstream (Data Source)**: 
  - References parameters from **[ARCH_1: Audio Processing](./ARCH_1_AUDIO_PROCESSING.md)** (power budget: 2mW, line 9)
  - References parameters from **[ARCH_2: Spike Conversion](./ARCH_2_SPIKE_CONVERSION.md)** (power budget: 1mW, latency: 15ms, lines 8-9)
  - References parameters from **[ARCH_5: SNN Inference](./ARCH_5_SNN_INFERENCE.md)** (inference latency: ~800ms per 10s audio, line 20)
  - References parameters from **[ARCH_6: JSON Payload](./ARCH_6_JSON_PAYLOAD.md)** (encryption latency: <8ms, line 560)

- **Related** (Parallel validation):
  - Energy profile validates assumptions about **[ARCH_8: Network Simulator](./ARCH_8_NETWORK_SIMULATOR.md)** (LoRa transmission power: 140mW peak, lines 140-180)
  - Feeds into operational planning in IMPLEMENTATION_SCHEDULE (Days 16-18: power measurements + battery testing)

### Data Format Specifications
- **Input Specification** (energy component parameters):
  - CPU idle power: 50mW (realistic ARM Cortex-M7 + audio codec listening)
  - Audio processing (ARCH_1): 2mW for 100ms per 10s sample
  - Spike conversion (ARCH_2): 1mW for 15ms per 10s sample
  - Inference (ARCH_5): 30mW for 800ms per 10s sample
  - Encryption (ARCH_6): 5mW for 8ms per event
  - LoRa transmission (ARCH_8): 140mW for 56-1500ms depending on SF (see line 180)
  - See [Component 1: Energy Budget Definition](./ARCH_7_ENERGY_PROFILER.md#component-1-energy-budget-definition) (lines 15-120)

- **Output Specification**: Energy profile reports
  - Battery capacity: 3000mAh @ 3.7V nominal (11.1Wh)
  - Operational duration prediction: 10-20 days (realistic with continuous monitoring)
  - Energy breakdown per operation
  - See [Component 2: Battery & Efficiency Calculations](./ARCH_7_ENERGY_PROFILER.md#component-2-battery--efficiency-calculations) (lines 125-250)

### Processing Timeline
- **Days 16-18** (IMPLEMENTATION_SCHEDULE): Power measurements + simulation + battery testing
- **Simulation time**: <100ms to generate full 30-day energy profile
- **Calibration**: Compare simulated vs. measured power on prototype device
- **Output updates**: Iterate power model until <10% error vs. actual measurements

### Key Parameters (Finalized - CRITICAL CORRECTIONS)
| Parameter | Value | Reference | Justification |
|-----------|-------|-----------|-----------|
| **Quiescent Power** | **50mW** | Line 65 | ARM Cortex-M7 + audio codec (100x correction from original 0.5mW) |
| Audio processing (ARCH_1) | 2mW avg | Line 85 | FFT + mel-scale computation |
| Spike conversion (ARCH_2) | 1mW avg | Line 95 | LIF neuron simulation |
| Spike conversion timing | 15ms per 10s | Line 98 | Corrected from original 50ms (matches ARCH_2 latency) |
| Inference (ARCH_5) | 30mW avg | Line 110 | SNN forward pass |
| Inference timing | 800ms per 10s | Line 112 | Real-world latency (ARCH_1 + ARCH_2 + ARCH_5) |
| Encryption (ARCH_6) | 5mW avg | Line 120 | AES-256 computation |
| LoRa transmission (peak) | 140mW | Line 140 | TX power class (14 dBm) |
| LoRa ToA @ SF7 | 56ms | Line 175 | Fastest transmission mode |
| LoRa ToA @ SF12 | 1.5s | Line 185 | Slowest (longest range) mode |
| Battery capacity | 3000mAh @ 3.7V | Line 200 | Standard Li-Po cell |
| Expected operational life | 10-20 days | Line 300 | Realistic with continuous 24/7 monitoring |

### Forest-Specific Energy Profiles
See [Component 3: Forest-Specific Energy Profiles](./ARCH_7_ENERGY_PROFILER.md#component-3-forest-specific-energy-profiles) (lines 255-400):

1. **Corbett (Dense Forest)**: 
   - Higher false-positive rate (birds, insects)
   - More transmission events per day
   - Operational life: ~10 days (frequent alerts)

2. **Seshachalam (Open Terrain)**:
   - Moderate alert rate
   - Balances detection sensitivity with power efficiency
   - Operational life: ~15 days (baseline)

3. **Sundarbans (Wetlands)**:
   - Rain/water artifacts increase false positives
   - Potential for power adaptation (reduce sampling in rain)
   - Operational life: ~12-18 days (dependent on weather)

### Critical Findings (Updated)
- **Original battery projection was 60+ days**: Unrealistic due to 0.5mW quiescent power assumption
- **Realistic projection is 10-20 days**: With 50mW quiescent + event-based transmission
- **Implication for deployment**: Device requires daily charger access or solar panel for >30-day deployments
- **Optimization opportunities**:
  1. Add solar trickle charging (2W peak, 1-3W avg during daylight)
  2. Reduce quiescent power to 30-40mW with ultra-low-power MCU (trade-off: slower inference)
  3. Event-based sampling (only listen during night hours when poaching risk higher)
  4. Adaptive LoRa SF (use SF7 when signal strong, reduce transmission overhead)

### Related Documentation
- **SUMMARY_HIGH_LEVEL_ARCHITECTURE.md** (Week 3): Power & energy management overview
- **IMPLEMENTATION_SCHEDULE.md** (Days 16-18): Power measurement & battery testing tasks
- **ARCH_1_AUDIO_PROCESSING.md**: Audio processing power budget
- **ARCH_2_SPIKE_CONVERSION.md**: Spike conversion latency & power
- **ARCH_5_SNN_INFERENCE.md**: Inference latency & power
- **ARCH_6_JSON_PAYLOAD.md**: Encryption overhead
- **ARCH_8_NETWORK_SIMULATOR.md**: LoRa transmission power & time-on-air
- **Resources/Research_Paper_Citations.md**: References for edge device power measurement, IoT battery modeling

