"""Central configuration for the Eco-Sentry pipeline.

Every value here is traceable to a "Key Parameters (Finalized)" table in the
ARCH_* design documents.  Nothing else in the package hard-codes a constant that
appears in the docs -- import it from here instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Class mapping (SUMMARY, ARCH_3, ARCH_5, PRIORITY_PAYLOAD_DELIVERY)
# ---------------------------------------------------------------------------

CLASS_NAMES: List[str] = ["gunshot", "chainsaw", "vehicle", "ambient"]
CLASS_MAP: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}

#: Classes the trained SNN discriminates.  ARCH_4/ARCH_5 specify a 3-class head;
#: ``ambient`` (class 3) exists in the raw catalogue but is folded into training
#: only when ``AudioConfig.include_ambient_class`` is enabled.
THREAT_CLASSES: Tuple[int, ...] = (0, 1)  # gunshot, chainsaw -- priority lane

FORESTS: List[str] = ["corbett", "seshachalam", "sundarbans", "mixed"]


# ---------------------------------------------------------------------------
# ARCH_1: Audio processing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioConfig:
    """ARCH_1_AUDIO_PROCESSING.md -- Key Parameters (Finalized)."""

    sample_rate: int = 16_000
    max_duration_s: float = 10.0
    target_rms: float = 0.1
    fade_ms: float = 50.0
    f_min: float = 50.0
    f_max: float = 8_000.0
    bandpass_low_hz: float = 100.0
    bandpass_high_hz: float = 8_000.0
    bandpass_order: int = 4
    frame_ms: float = 32.0  # 512 samples @ 16 kHz
    hop_ms: float = 10.0  # 160 samples @ 16 kHz
    n_fft: int = 512
    n_mels: int = 64
    db_floor: float = -80.0
    db_ceiling: float = 0.0
    #: ARCH_1 offers an optional zero-mean/unit-variance step.  It is OFF by
    #: default because ARCH_2 requires the documented [-80, 0] dB input range.
    standardize_db: bool = False

    @property
    def frame_length(self) -> int:
        return int(round(self.frame_ms * self.sample_rate / 1000.0))

    @property
    def hop_length(self) -> int:
        return int(round(self.hop_ms * self.sample_rate / 1000.0))

    @property
    def frames_per_second(self) -> float:
        return self.sample_rate / self.hop_length


# ---------------------------------------------------------------------------
# ARCH_2: Spike conversion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpikeConfig:
    """ARCH_2_SPIKE_CONVERSION.md -- Key Parameters (Finalized)."""

    tau_m_s: float = 10e-3
    v_threshold: float = 1.0
    v_reset: float = 0.0
    refractory_ms: float = 2.0
    hop_ms: float = 10.0
    n_neurons: int = 64
    #: Gain applied to the [0, 1] normalised mel input before integration.
    #:
    #: With the documented LIF update ``V <- aV + (1-a)I`` and ``I <= 1`` the
    #: membrane asymptote equals ``I``, so a unit-gain neuron can never cross
    #: ``v_threshold = 1.0`` and the firing rate collapses to 0%.  The docs
    #: nonetheless require a 25% +/- 5% firing rate, so an explicit input gain
    #: is the free parameter that reconciles the two.  See
    #: :func:`ecosentry.arch2_spikes.calibrate_input_gain`.
    input_gain: float = 2.0

    @property
    def alpha(self) -> float:
        """Leak factor ``exp(-dt / tau_m)`` at frame resolution (~0.3679)."""
        import math

        return math.exp(-(self.hop_ms / 1000.0) / self.tau_m_s)

    @property
    def refractory_frames(self) -> int:
        return max(int(self.refractory_ms / self.hop_ms), 0)


# ---------------------------------------------------------------------------
# ARCH_3: Dataset preparation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetConfig:
    """ARCH_3_DATASET_PREPARATION.md -- Key Parameters (Finalized)."""

    n_frames: int = 1_000  # T for a 10 s sample
    target_firing_rate: float = 0.25
    firing_rate_tolerance: float = 0.05
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    augmentation_factor: int = 2  # 600 -> 1200
    p_mixup: float = 0.5
    p_time_shift: float = 0.75
    p_noise: float = 0.25
    max_shift_frames: int = 5  # +/- 50 ms
    mixup_alpha: float = 0.2
    noise_rate: float = 0.01
    seed: int = 42
    #: Sample budget per source (ARCH_3 Component 1).  Totals 600.
    source_counts: Dict[str, int] = field(
        default_factory=lambda: {
            "ESC-50": 100,
            "UrbanSound8K": 150,
            "Corbett": 150,
            "Seshachalam": 100,
            "Sundarbans": 100,
        }
    )
    source_forest: Dict[str, str] = field(
        default_factory=lambda: {
            "ESC-50": "mixed",
            "UrbanSound8K": "mixed",
            "Corbett": "corbett",
            "Seshachalam": "seshachalam",
            "Sundarbans": "sundarbans",
        }
    )


# ---------------------------------------------------------------------------
# ARCH_4 / ARCH_5: SNN model, training, inference
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SNNConfig:
    """ARCH_4_SNN_TRAINING.md -- architecture + hyperparameters."""

    n_input: int = 64
    n_hidden1: int = 128
    n_hidden2: int = 64
    n_classes: int = 4
    v_threshold: float = 1.0
    tau_m_s: float = 10e-3
    dt_ms: float = 10.0
    surrogate_alpha: float = 2.0
    #: Leak of the non-spiking readout layer.
    #:
    #: ARCH_4 describes the output layer as "membrane potential integrated over
    #: time ... NO reset ... forming a rate-based classifier".  Reusing the
    #: hidden-layer leak (exp(-1) ~ 0.368) would give the readout a ~14 ms
    #: memory, so a 10 s clip would be classified from its final two frames.
    #: 1.0 makes it the pure accumulator the text describes.
    readout_alpha: float = 1.0
    #: Divide the accumulated readout by T so logit scale is sequence-length
    #: independent (an average firing-rate readout).
    readout_normalize: bool = True

    #: Rescale the Xavier-initialised hidden weights so each LIF layer starts
    #: at ``init_target_rate``.
    #:
    #: Plain Xavier init with ``v_th = 1.0`` leaves both hidden layers silent
    #: (measured: 0.2% then 0.0% firing), which zeroes the surrogate gradient
    #: and collapses training to a constant prediction.  The calibration is the
    #: SNN analogue of LSUV / data-dependent init.
    calibrate_init: bool = True
    init_target_rate: float = 0.20

    learning_rate: float = 1e-3
    lr_decay: float = 0.95
    lr_decay_every: int = 10
    batch_size: int = 32
    epochs: int = 120
    grad_clip: float = 5.0
    weight_decay: float = 1e-4
    early_stopping_patience: int = 20
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    seed: int = 42

    @property
    def alpha(self) -> float:
        import math

        return math.exp(-(self.dt_ms / 1000.0) / self.tau_m_s)


@dataclass(frozen=True)
class InferenceConfig:
    """ARCH_5_SNN_INFERENCE.md -- thresholding and debouncing."""

    default_threshold: float = 0.85
    class_thresholds: Dict[int, float] = field(
        default_factory=lambda: {0: 0.90, 1: 0.80, 2: 0.95}
    )
    temporal_window: int = 5
    history_size: int = 10
    min_alert_interval_s: float = 1.0
    softmax_temperature: float = 1.0


# ---------------------------------------------------------------------------
# ARCH_6: Payload
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PayloadConfig:
    """ARCH_6_JSON_PAYLOAD.md -- wire format."""

    magic: bytes = b"\xec\xea"
    protocol_version: int = 0x01
    zlib_level: int = 9
    aes_key_bytes: int = 32  # AES-256
    iv_bytes: int = 16
    max_message_bytes: int = 1_000
    firmware_version: str = "v1.0.0"
    #: PRIORITY_PAYLOAD_DELIVERY.md Option A -- priority is carried in the
    #: transport header, so the encrypted payload schema is untouched.
    priority_header_flag: int = 0x80


@dataclass(frozen=True)
class BeaconConfig:
    """arch6_beacon.py -- compact critical-event packet (new architecture,
    not from the original ARCH_* documents)."""

    size_bytes: int = 16
    body_bytes: int = 13
    mac_bytes: int = 3
    version: int = 1
    #: Only these class_ids are eligible to ride the beacon fast path.
    #: Vehicle (class 2) and ambient (class 3, if enabled) always use the
    #: full payload only -- matches THREAT_CLASSES.
    beacon_eligible_classes: Tuple[int, ...] = (0, 1)


# ---------------------------------------------------------------------------
# ARCH_7: Energy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnergyConfig:
    """ARCH_7_ENERGY_PROFILER.md -- Key Parameters (Finalized, corrected)."""

    battery_capacity_mah: float = 5_000.0
    battery_nominal_v: float = 3.7
    #: 50 mW, not 0.5 mW -- the "CRITICAL CORRECTIONS" table in ARCH_7.
    power_quiescent_mw: float = 50.0
    power_audio_mw: float = 2.0
    power_spike_mw: float = 1.0
    power_inference_mw: float = 30.0
    power_encryption_mw: float = 5.0
    power_transmission_mw: float = 140.0
    cnn_baseline_mw: float = 600.0

    t_audio_s: float = 0.100
    t_spike_s: float = 0.015
    t_inference_s: float = 0.800
    t_encryption_s: float = 0.008
    t_transmission_s: float = 0.352  # SF9 time-on-air for a 116 B payload

    solar_peak_w: float = 0.5
    solar_min_w: float = 0.05
    sunrise_hour: float = 6.5
    sunset_hour: float = 18.5
    solar_sigma_h: float = 3.0
    solar_system_efficiency: float = 0.6
    weather_factor_range: Tuple[float, float] = (0.5, 1.0)
    mission_days: int = 30

    @property
    def battery_energy_wh(self) -> float:
        return (self.battery_capacity_mah / 1000.0) * self.battery_nominal_v

    @property
    def battery_energy_j(self) -> float:
        return self.battery_energy_wh * 3600.0


# ---------------------------------------------------------------------------
# ARCH_8: Network
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NetworkConfig:
    """ARCH_8_NETWORK_SIMULATOR.md -- LoRa PHY + mesh."""

    center_frequency_mhz: float = 865.0
    bandwidth_hz: float = 125_000.0
    coding_rate: int = 1  # 4/5
    preamble_symbols: int = 8
    tx_power_dbm: float = 20.0
    explicit_header: bool = True
    crc_on: bool = True
    payload_bytes: int = 116
    fading_sigma_db: float = 4.5
    #: Excess path-loss exponent beyond free space for forest canopy.
    path_loss_exponent: float = 2.7
    #: Nodes further apart than this are not radio neighbours.
    max_link_km: float = 12.0
    #: Per-forest radio reach, from ARCH_8 "Network Topology (Forest
    #: Scenarios)": dense canopy forces 500-1000 m node spacing and 2-3 hops,
    #: open terrain supports 2-5 km line-of-sight links.  Without these the
    #: mesh degenerates to a single direct hop everywhere.
    scenario_max_link_km: Dict[str, float] = field(
        default_factory=lambda: {"corbett": 4.5, "seshachalam": 9.0, "sundarbans": 6.0}
    )
    scenario_path_loss_exponent: Dict[str, float] = field(
        default_factory=lambda: {"corbett": 3.0, "seshachalam": 2.4, "sundarbans": 2.8}
    )
    processing_delay_ms: float = 10.0
    queue_delay_mean_ms: float = 20.0
    max_retries: int = 2
    latency_target_ms: float = 1_500.0
    delivery_target: float = 0.95
    rx_sensitivity_dbm: Dict[int, float] = field(
        default_factory=lambda: {7: -123, 8: -126, 9: -129, 10: -132, 11: -134, 12: -137}
    )
    scenario_sf: Dict[str, int] = field(
        default_factory=lambda: {"corbett": 12, "seshachalam": 9, "sundarbans": 10}
    )
    #: Beacons are small enough to default to the fastest, shortest-range
    #: spreading factor rather than each forest's payload SF (Phase 5, new
    #: architecture).
    beacon_spreading_factor: int = 7



@dataclass(frozen=True)
class GatewayConfig:
    """PRIORITY_PAYLOAD_DELIVERY.md -- gateway priority proxy."""

    priority_topic: str = "priority/alerts"
    normal_topic: str = "alerts"
    priority_qos: int = 1
    normal_qos: int = 0
    #: EF (Expedited Forwarding, RFC 3246) is the correct marking to pair
    #: with a strict-priority LLQ -- AF41 is meant for bandwidth-guaranteed,
    #: delay-tolerant traffic like video, not low-latency alert traffic.
    priority_dscp: str = "EF"
    confidence_threshold: float = 0.85
    backoff_schedule_s: Tuple[float, ...] = (1, 2, 4, 8, 16, 32, 60)
    retention_hours: int = 24
    #: Backhaul latency models (mean_ms, loss probability).
    qos_backhaul_ms: float = 120.0
    qos_backhaul_loss: float = 0.005
    best_effort_backhaul_ms: float = 450.0
    best_effort_backhaul_loss: float = 0.03
    #: Beacon fast-path topic/QoS (Phase 1/3, new architecture).
    beacon_topic: str = "priority/beacons"
    beacon_qos: int = 0
    #: If a beacon arrives with no matching full payload within this many
    #: seconds, raise a degraded alert instead of silently dropping it.
    beacon_orphan_timeout_s: float = 30.0
    #: Adaptive LLQ token bucket (Phase 3, new architecture).
    llq_max_credits_floor: int = 2
    llq_max_credits_ceiling: int = 20
    llq_overflow_latency_multiplier: float = 2.5


# ---------------------------------------------------------------------------
# Forest scenarios (ARCH_7 Component 4 + ARCH_8 Component 7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForestScenario:
    key: str
    name: str
    latitude: float
    longitude: float
    true_positive_rate: float
    false_positive_rate: float
    avg_alerts_per_day: float
    threat_probability: float
    solar_derate: float
    spreading_factor: int


SCENARIOS: Dict[str, ForestScenario] = {
    "corbett": ForestScenario(
        key="corbett",
        name="Corbett National Park",
        latitude=29.2,
        longitude=79.1,
        true_positive_rate=0.80,
        false_positive_rate=0.05,
        avg_alerts_per_day=5.0,
        threat_probability=0.15,
        solar_derate=1.00,
        spreading_factor=12,
    ),
    "seshachalam": ForestScenario(
        key="seshachalam",
        name="Seshachalam Hills",
        latitude=13.2,
        longitude=79.4,
        true_positive_rate=0.85,
        false_positive_rate=0.02,
        avg_alerts_per_day=3.0,
        threat_probability=0.20,
        solar_derate=0.90,
        spreading_factor=9,
    ),
    "sundarbans": ForestScenario(
        key="sundarbans",
        name="Sundarbans Wetlands",
        latitude=21.9,
        longitude=88.5,
        true_positive_rate=0.75,
        false_positive_rate=0.08,
        avg_alerts_per_day=8.0,
        threat_probability=0.10,
        solar_derate=0.75,
        spreading_factor=10,
    ),
}


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EcoSentryConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    spikes: SpikeConfig = field(default_factory=SpikeConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    snn: SNNConfig = field(default_factory=SNNConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    payload: PayloadConfig = field(default_factory=PayloadConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    gateway: GatewayConfig = field(default_factory=GatewayConfig)

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT = EcoSentryConfig()
