"""ARCH_8 -- LoRa mesh network topology simulator.

Covers the PHY (time-on-air, path loss, link budget), the mesh layer (topology,
Dijkstra routing over radio-reachable links, per-node queues, dedup, retries)
and a Monte-Carlo delivery campaign producing the delivery-rate / latency
statistics the design document validates against.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import NetworkConfig, SCENARIOS

__all__ = [
    "LoRaPHY",
    "MeshNetwork",
    "MessageQueue",
    "create_corbett_topology",
    "create_seshachalam_topology",
    "create_sundarbans_topology",
    "create_topology",
    "find_route_to_base",
    "calculate_link_quality",
    "simulate_message_delivery",
    "simulate_alert_delivery_campaign",
    "generate_network_report",
]


# ---------------------------------------------------------------------------
# Physical layer (ARCH_8 Component 1)
# ---------------------------------------------------------------------------


class LoRaPHY:
    """LoRa SX127x PHY model: time-on-air, path loss and link budget."""

    def __init__(self, cfg: Optional[NetworkConfig] = None):
        self.cfg = cfg or NetworkConfig()

    def time_on_air_ms(self, payload_bytes: int, spreading_factor: int) -> float:
        """Semtech SX1276 time-on-air, in milliseconds.

        ``T_sym = 2^SF / BW``;
        ``n_payload = 8 + max(ceil((8*PL - 4*SF + 28 + 16*CRC - 20*IH)
                                    / (4*(SF - 2*DE))) * (CR + 4), 0)``.
        """
        cfg = self.cfg
        sf = int(spreading_factor)
        t_sym_ms = (2**sf) / cfg.bandwidth_hz * 1000.0

        de = 1 if (sf >= 11 and cfg.bandwidth_hz <= 125_000) else 0
        ih = 0 if cfg.explicit_header else 1
        crc = 1 if cfg.crc_on else 0

        numerator = 8 * payload_bytes - 4 * sf + 28 + 16 * crc - 20 * ih
        denominator = 4 * (sf - 2 * de)
        n_payload = 8 + max(math.ceil(numerator / denominator) * (cfg.coding_rate + 4), 0)

        t_preamble_ms = (cfg.preamble_symbols + 4.25) * t_sym_ms
        return t_preamble_ms + n_payload * t_sym_ms

    def path_loss_db(
        self,
        distance_km: float,
        rng: Optional[np.random.Generator] = None,
        include_fading: bool = True,
    ) -> float:
        """Log-distance path loss with a forest exponent plus Rayleigh fading."""
        cfg = self.cfg
        d_m = max(distance_km * 1000.0, 1.0)

        # Free-space loss at the 1 m reference, then a forest-canopy exponent.
        pl_1m = 20 * math.log10(cfg.center_frequency_mhz) - 27.55
        pl = pl_1m + 10 * cfg.path_loss_exponent * math.log10(d_m)

        if include_fading:
            rng = rng or np.random.default_rng()
            pl += float(rng.normal(0.0, cfg.fading_sigma_db))
        return pl

    def rssi_dbm(self, distance_km: float, rng: Optional[np.random.Generator] = None) -> float:
        return self.cfg.tx_power_dbm - self.path_loss_db(distance_km, rng)

    def can_receive(self, rssi: float, spreading_factor: int) -> bool:
        return rssi >= self.cfg.rx_sensitivity_dbm[int(spreading_factor)]

    def min_spreading_factor(self, distance_km: float) -> int:
        """Lowest SF whose sensitivity closes the link with 6 dB of margin."""
        median_rssi = self.cfg.tx_power_dbm - self.path_loss_db(
            distance_km, include_fading=False
        )
        for sf in sorted(self.cfg.rx_sensitivity_dbm):
            if median_rssi - self.cfg.rx_sensitivity_dbm[sf] >= 6.0:
                return sf
        return max(self.cfg.rx_sensitivity_dbm)


# ---------------------------------------------------------------------------
# Topology (ARCH_8 Component 2)
# ---------------------------------------------------------------------------


@dataclass
class Node:
    node_id: str
    lat: float
    lon: float
    node_type: str = "sensor"  # sensor | relay | base_station
    battery_j: float = 66_600.0


class MeshNetwork:
    """Geographic mesh with radio-reachability-limited edges."""

    def __init__(self, scenario: str, cfg: Optional[NetworkConfig] = None):
        self.scenario = scenario
        self.cfg = cfg or NetworkConfig()
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[Tuple[str, str], float] = {}
        self.base_stations: List[str] = []
        self.phy = LoRaPHY(self.cfg)

    def add_device(self, node_id: str, lat: float, lon: float, node_type: str = "sensor") -> None:
        self.nodes[node_id] = Node(node_id, lat, lon, node_type)
        if node_type == "base_station":
            self.base_stations.append(node_id)

    @staticmethod
    def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        )
        return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def compute_distances(self, max_link_km: Optional[float] = None) -> None:
        """Populate the edge table for every pair within radio range."""
        max_link_km = self.cfg.max_link_km if max_link_km is None else max_link_km
        ids = list(self.nodes)
        self.edges.clear()
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                na, nb = self.nodes[a], self.nodes[b]
                d = self.haversine_km(na.lat, na.lon, nb.lat, nb.lon)
                if d <= max_link_km:
                    self.edges[(a, b)] = d
                    self.edges[(b, a)] = d

    def neighbors(self, node_id: str) -> List[str]:
        return [b for (a, b) in self.edges if a == node_id]

    def sensors(self) -> List[str]:
        return [n for n, node in self.nodes.items() if node.node_type == "sensor"]

    def summary(self) -> Dict:
        distances = [d for (a, b), d in self.edges.items() if a < b]
        return {
            "scenario": self.scenario,
            "n_nodes": len(self.nodes),
            "n_sensors": len(self.sensors()),
            "n_relays": sum(1 for n in self.nodes.values() if n.node_type == "relay"),
            "n_links": len(distances),
            "mean_link_km": round(float(np.mean(distances)), 3) if distances else 0.0,
            "max_link_km": round(float(np.max(distances)), 3) if distances else 0.0,
        }


def create_corbett_topology(cfg: Optional[NetworkConfig] = None) -> MeshNetwork:
    """Dense forest, hilly: 3 sensors + 2 hilltop relays + base.

    Sensor coordinates are the documented deployment sites.  Relay hilltops are
    placed on the sensor-to-base midpoints -- at the 4.5 km dense-canopy link
    budget the ARCH_8 relay coordinates leave S2 orphaned (4.42 km to R1, then
    4.85 km R1 to base).  See :func:`connectivity_check`.
    """
    net = MeshNetwork("corbett", cfg)
    net.add_device("S1", 29.25, 79.10, "sensor")
    net.add_device("S2", 29.30, 79.15, "sensor")
    net.add_device("S3", 29.20, 79.05, "sensor")
    net.add_device("R1", 29.270, 79.125, "relay")
    net.add_device("R2", 29.220, 79.075, "relay")
    net.add_device("BASE", 29.24, 79.10, "base_station")
    net.compute_distances()
    return net


def create_seshachalam_topology(cfg: Optional[NetworkConfig] = None) -> MeshNetwork:
    """Open terrain: 2 sensors + 1 relay + base."""
    net = MeshNetwork("seshachalam", cfg)
    net.add_device("S1", 13.15, 79.35, "sensor")
    net.add_device("S2", 13.20, 79.40, "sensor")
    net.add_device("R1", 13.17, 79.37, "relay")
    net.add_device("BASE", 13.18, 79.38, "base_station")
    net.compute_distances()
    return net


def create_sundarbans_topology(cfg: Optional[NetworkConfig] = None) -> MeshNetwork:
    """Flat wetlands, scattered: 4 sensors + 3 relays + base.

    S4 sits 10.2 km from the base -- beyond a single 6 km wetland hop -- so the
    documented "4 sensors + 2 relays" needs a third relay to close coverage.
    This is the concrete form of the ARCH_8 recommendation to add relay nodes
    for the Sundarbans scenario.
    """
    net = MeshNetwork("sundarbans", cfg)
    net.add_device("S1", 21.90, 88.50, "sensor")
    net.add_device("S2", 21.95, 88.55, "sensor")
    net.add_device("S3", 21.85, 88.45, "sensor")
    net.add_device("S4", 21.88, 88.60, "sensor")
    net.add_device("R1", 21.9275, 88.5275, "relay")
    net.add_device("R2", 21.8775, 88.4775, "relay")
    net.add_device("R3", 21.890, 88.550, "relay")
    net.add_device("BASE", 21.905, 88.505, "base_station")
    net.compute_distances()
    return net


_TOPOLOGIES = {
    "corbett": create_corbett_topology,
    "seshachalam": create_seshachalam_topology,
    "sundarbans": create_sundarbans_topology,
}


def create_topology(scenario: str, cfg: Optional[NetworkConfig] = None) -> MeshNetwork:
    """Build a forest topology with its scenario-specific propagation settings."""
    from dataclasses import replace

    cfg = cfg or NetworkConfig()
    if scenario not in _TOPOLOGIES:
        raise ValueError(f"unknown scenario '{scenario}'")

    cfg = replace(
        cfg,
        max_link_km=cfg.scenario_max_link_km.get(scenario, cfg.max_link_km),
        path_loss_exponent=cfg.scenario_path_loss_exponent.get(
            scenario, cfg.path_loss_exponent
        ),
    )
    return _TOPOLOGIES[scenario](cfg)


# ---------------------------------------------------------------------------
# Routing (ARCH_8 Component 3)
# ---------------------------------------------------------------------------


def connectivity_check(network: "MeshNetwork") -> Dict:
    """Verify every sensor can reach the base station at the modelled range."""
    routes = {}
    orphans = []
    for sensor in network.sensors():
        path, hops, km = find_route_to_base(network, sensor)
        routes[sensor] = {"path": path, "hops": hops, "distance_km": round(km, 2) if path else None}
        if hops < 0:
            orphans.append(sensor)
    return {
        "connected": not orphans,
        "orphaned_sensors": orphans,
        "routes": routes,
        "max_hops": max((r["hops"] for r in routes.values()), default=0),
    }


def find_route_to_base(
    network: MeshNetwork, source: str, cost: str = "hops"
) -> Tuple[List[str], int, float]:
    """Dijkstra from ``source`` to the base station.

    ``cost="hops"`` minimises hop count (each hop costs one time-on-air, which
    dominates latency); ``cost="distance"`` minimises total path length.
    Returns ``(path, hop_count, total_distance_km)``; ``([], -1, inf)`` when the
    base is unreachable.
    """
    if not network.base_stations:
        return [], -1, math.inf
    base = network.base_stations[0]
    if source == base:
        return [base], 0, 0.0

    dist: Dict[str, float] = {n: math.inf for n in network.nodes}
    dist[source] = 0.0
    previous: Dict[str, Optional[str]] = {n: None for n in network.nodes}
    visited: set = set()
    pq: List[Tuple[float, str]] = [(0.0, source)]

    while pq:
        d, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        if node == base:
            break
        for neighbor in network.neighbors(node):
            if neighbor in visited:
                continue
            edge_km = network.edges[(node, neighbor)]
            step = 1.0 if cost == "hops" else edge_km
            if d + step < dist[neighbor]:
                dist[neighbor] = d + step
                previous[neighbor] = node
                heapq.heappush(pq, (dist[neighbor], neighbor))

    if math.isinf(dist[base]):
        return [], -1, math.inf

    path: List[str] = []
    current: Optional[str] = base
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()

    total_km = sum(
        network.edges[(path[i], path[i + 1])] for i in range(len(path) - 1)
    )
    return path, len(path) - 1, total_km


def calculate_link_quality(
    distance_km: float,
    spreading_factor: int = 7,
    cfg: Optional[NetworkConfig] = None,
    interference: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Packet delivery ratio for one link, from its SNR margin."""
    cfg = cfg or NetworkConfig()
    phy = LoRaPHY(cfg)
    rssi = cfg.tx_power_dbm - phy.path_loss_db(distance_km, rng, include_fading=False)
    margin = rssi - cfg.rx_sensitivity_dbm[int(spreading_factor)]

    if margin < 0:
        per = 1.0
    elif margin < 3:
        per = 0.30
    elif margin < 6:
        per = 0.10
    elif margin < 10:
        per = 0.02
    else:
        per = 0.001

    if interference:
        per = min(per * 2.0, 1.0)
    return 1.0 - per


# ---------------------------------------------------------------------------
# Queues (ARCH_8 Component 4)
# ---------------------------------------------------------------------------


@dataclass
class MessageQueue:
    """Bounded per-node FIFO with priority retention."""

    max_size_bytes: int = 50_000
    queue: List[Dict] = field(default_factory=list)
    dropped: int = 0

    @property
    def current_size(self) -> int:
        return sum(len(m["data"]) for m in self.queue)

    def enqueue(
        self,
        message: bytes,
        timestamp: float = 0.0,
        priority: bool = False,
        kind: str = "payload",
    ) -> bool:
        """kind is one of "beacon" or "payload" (Phase 2). Existing callers
        that don't pass kind get "payload", matching pre-Phase-2 behavior
        exactly -- this parameter is additive and does not change eviction
        or dequeue ordering."""
        ok = True
        while self.current_size + len(message) > self.max_size_bytes and self.queue:
            victim = next((i for i, m in enumerate(self.queue) if not m["priority"]), 0)
            self.queue.pop(victim)
            self.dropped += 1
            ok = False
        self.queue.append(
            {"data": message, "timestamp": timestamp, "priority": priority, "kind": kind}
        )
        return ok

    def dequeue_batch(self, max_messages: int = 5) -> List[Dict]:
        order = sorted(
            range(len(self.queue)),
            key=lambda i: (not self.queue[i]["priority"], self.queue[i]["timestamp"]),
        )[:max_messages]
        batch = [self.queue[i] for i in order]
        for i in sorted(order, reverse=True):
            self.queue.pop(i)
        return batch


# ---------------------------------------------------------------------------
# Delivery simulation (ARCH_8 Component 5-6)
# ---------------------------------------------------------------------------


def simulate_message_delivery(
    network: MeshNetwork,
    source: str,
    message_bytes: int = 116,
    spreading_factor: Optional[int] = None,
    cfg: Optional[NetworkConfig] = None,
    rng: Optional[np.random.Generator] = None,
    interference: bool = False,
    max_retries: Optional[int] = None,
    congestion_factor: float = 1.0,
) -> Dict:
    """Push one alert from ``source`` to the base station across the mesh."""
    cfg = cfg or network.cfg
    rng = rng or np.random.default_rng()
    phy = LoRaPHY(cfg)
    max_retries = cfg.max_retries if max_retries is None else max_retries

    path, hops, total_km = find_route_to_base(network, source)
    if hops < 0:
        return {
            "delivered": False,
            "latency_ms": None,
            "path": [],
            "hops": 0,
            "retries": 0,
            "reason": "unreachable",
        }
    if hops == 0:
        return {
            "delivered": True,
            "latency_ms": 0.0,
            "path": path,
            "hops": 0,
            "retries": 0,
            "reason": "source_is_base",
        }

    latency_ms = 0.0
    retries = 0
    traversed = [source]

    for i in range(hops):
        a, b = path[i], path[i + 1]
        distance = network.edges[(a, b)]
        sf = spreading_factor or phy.min_spreading_factor(distance)
        pdr = calculate_link_quality(distance, sf, cfg, interference, rng)
        toa = phy.time_on_air_ms(message_bytes, sf)

        attempt = 0
        while True:
            latency_ms += toa
            latency_ms += cfg.processing_delay_ms
            latency_ms += float(rng.exponential(cfg.queue_delay_mean_ms * congestion_factor))
            latency_ms += distance / 300_000.0  # propagation, effectively nil

            if rng.random() <= pdr:
                break
            attempt += 1
            retries += 1
            if attempt > max_retries:
                return {
                    "delivered": False,
                    "latency_ms": None,
                    "path": traversed,
                    "hops": i,
                    "retries": retries,
                    "reason": f"link_failed:{a}->{b}",
                    "failed_link": (a, b),
                }
            # Exponential backoff before re-transmitting.
            latency_ms += toa * (2**attempt)

        traversed.append(b)

    return {
        "delivered": True,
        "latency_ms": latency_ms,
        "path": traversed,
        "hops": hops,
        "retries": retries,
        "distance_km": total_km,
        "reason": "ok",
    }


def simulate_alert_delivery_campaign(
    network: MeshNetwork,
    num_messages: int = 100,
    message_bytes: int = 116,
    spreading_factor: Optional[int] = None,
    cfg: Optional[NetworkConfig] = None,
    seed: int = 42,
    interference: bool = False,
    congestion_factor: float = 1.0,
) -> Dict:
    """Monte-Carlo campaign; returns delivery rate and latency percentiles."""
    cfg = cfg or network.cfg
    rng = np.random.default_rng(seed)
    sensors = network.sensors() or list(network.nodes)

    delivered = 0
    latencies: List[float] = []
    hop_counts: List[int] = []
    retries_total = 0
    failures: List[Dict] = []

    for _ in range(num_messages):
        source = str(rng.choice(sensors))
        result = simulate_message_delivery(
            network,
            source,
            message_bytes,
            spreading_factor,
            cfg,
            rng,
            interference,
            congestion_factor=congestion_factor,
        )
        retries_total += result["retries"]
        if result["delivered"]:
            delivered += 1
            latencies.append(result["latency_ms"])
            hop_counts.append(result["hops"])
        else:
            failures.append({"source": source, "reason": result["reason"]})

    lat = np.array(latencies) if latencies else np.array([np.nan])
    stats = {
        "scenario": network.scenario,
        "messages": num_messages,
        "delivered": delivered,
        "delivery_rate": delivered / max(num_messages, 1),
        "latency_mean_ms": float(np.nanmean(lat)),
        "latency_std_ms": float(np.nanstd(lat)),
        "latency_p50_ms": float(np.nanpercentile(lat, 50)),
        "latency_p95_ms": float(np.nanpercentile(lat, 95)),
        "latency_p99_ms": float(np.nanpercentile(lat, 99)),
        "latency_max_ms": float(np.nanmax(lat)),
        "avg_hop_count": float(np.mean(hop_counts)) if hop_counts else 0.0,
        "total_retries": retries_total,
        "packet_loss": 1.0 - delivered / max(num_messages, 1),
        "failures": failures[:20],
    }
    stats["meets_delivery_target"] = stats["delivery_rate"] >= cfg.delivery_target
    stats["meets_latency_target"] = (
        bool(stats["latency_p99_ms"] <= cfg.latency_target_ms)
        if latencies
        else False
    )
    return stats


def time_on_air_table(
    payload_bytes: int = 116, cfg: Optional[NetworkConfig] = None
) -> Dict[int, float]:
    """Time-on-air (ms) for SF7..SF12 at the given payload size."""
    phy = LoRaPHY(cfg)
    return {sf: round(phy.time_on_air_ms(payload_bytes, sf), 1) for sf in range(7, 13)}


def generate_network_report(
    scenario: str,
    num_messages: int = 200,
    message_bytes: int = 116,
    cfg: Optional[NetworkConfig] = None,
    seed: int = 42,
) -> Dict:
    """Baseline + congestion + interference sweep for one forest scenario."""
    cfg = cfg or NetworkConfig()
    network = create_topology(scenario, cfg)
    sf = cfg.scenario_sf.get(scenario)

    baseline = simulate_alert_delivery_campaign(
        network, num_messages, message_bytes, None, cfg, seed
    )
    fixed_sf = simulate_alert_delivery_campaign(
        network, num_messages, message_bytes, sf, cfg, seed + 1
    )
    congestion = simulate_alert_delivery_campaign(
        network, num_messages, message_bytes, None, cfg, seed + 2, congestion_factor=5.0
    )
    interference = simulate_alert_delivery_campaign(
        network, num_messages, message_bytes, None, cfg, seed + 3, interference=True
    )

    recommendations: List[str] = []
    if baseline["delivery_rate"] < cfg.delivery_target:
        recommendations.append(
            "Delivery below 95%: add relay nodes or shorten inter-node spacing."
        )
    if baseline["latency_p99_ms"] > cfg.latency_target_ms:
        recommendations.append(
            "p99 latency above 1.5 s: reduce hop count or drop to a lower spreading factor."
        )
    if congestion["delivery_rate"] < baseline["delivery_rate"] - 0.05:
        recommendations.append(
            "Congestion-sensitive: enable per-node queuing with priority retention."
        )
    if interference["delivery_rate"] < cfg.delivery_target:
        recommendations.append(
            "Interference-sensitive: enable channel hopping / adaptive spreading factor."
        )
    if not recommendations:
        recommendations.append("All targets met; deploy the documented topology as-is.")

    connectivity = connectivity_check(network)
    if not connectivity["connected"]:
        recommendations.append(
            "Sensors unreachable at the modelled link range: "
            f"{connectivity['orphaned_sensors']} -- reposition or add relays."
        )

    return {
        "scenario": scenario,
        "topology": network.summary(),
        "connectivity": connectivity,
        "adaptive_sf_baseline": baseline,
        "fixed_sf_baseline": fixed_sf,
        "fixed_spreading_factor": sf,
        "congestion": congestion,
        "interference": interference,
        "time_on_air_ms": time_on_air_table(message_bytes, cfg),
        "recommendations": recommendations,
    }


def compare_network_scenarios(
    num_messages: int = 200, message_bytes: int = 116, cfg: Optional[NetworkConfig] = None
) -> Dict[str, Dict]:
    return {
        key: generate_network_report(key, num_messages, message_bytes, cfg)
        for key in SCENARIOS
    }
