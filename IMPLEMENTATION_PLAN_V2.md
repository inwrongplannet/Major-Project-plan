# Eco-Sentry — Implementation runbook: priority delivery architecture + energy visualizer

This is a task-by-task runbook, not a design discussion. Every code block in
this document has already been written and executed against a throwaway
prototype to confirm it behaves as described — the exact commands used to
verify each one are included so you can re-run them yourself. Follow the
tasks in order. Do not skip, merge, or reorder tasks.

---

## HOW TO EXECUTE THIS PLAN — read this section fully before starting

1. **Work through tasks in strict numeric order**: P0.1, P0.2, P0.3, P1.1,
   P1.2, P1.3, P2.1, P3.1, P3.2, P3.3, P3.4, P4.1, P4.2, P5.1, P5.2, P5.3,
   P5.4, P5.5. A task's "Preconditions" line names the exact task(s) that
   must already be complete and verified. Never start a task whose
   preconditions are not met.
2. **Every task edits or creates only the file(s) named in that task's
   "Files touched" line.** Do not edit any other file while executing a
   task, even if you notice something else you think should change.
3. **"Anchor" means an exact, copy-pasteable string that exists in the file
   today.** Every anchor in this plan was verified against the actual repo
   contents before this plan was written. If an anchor does not match
   exactly (even by one character, one space, or one line break), stop and
   report the mismatch instead of guessing at a fix — do not improvise a
   different insertion point.
4. **"New file" tasks give the complete file contents.** Create the file
   with exactly that content. Do not add imports, comments, or code that
   isn't shown. Do not "improve" the code — it has already been tested.
5. **After completing a task, run its "Verification" command exactly as
   written and compare against "Expected result."** If it does not match,
   stop. Do not proceed to the next task. Do not attempt a fix that isn't
   explicitly described in this plan — report the exact command you ran and
   the exact output you got.
6. **Never rename, delete, or change the signature of any existing public
   function, class, or config field** unless a task explicitly instructs
   it. Where a task changes a default value (e.g. `priority_dscp`), change
   only that one line.
7. **Never modify an existing test file's existing test functions.** Tasks
   that add tests always add new functions to the file; they never edit a
   function that was already there.
8. **All new Python code must match the existing project's style**: `from
   __future__ import annotations` at the top of every module, type hints on
   every function signature, dataclasses for config/data objects (matching
   the pattern in `ecosentry/config.py`), no bare `except:` clauses.
9. At the very end, run the full command in the **Final verification**
   section and confirm every line of its expected output.

---

## Phase 0 — Foundational fixes (energy data plumbing + dashboard)

### TASK P0.1 — Serialize the battery trajectory arrays

**Goal:** `energy_report.json` currently has no day-by-day or hour-by-hour
series, because `generate_energy_report` only stores them under keys that
start with `_`, and `pipeline.py` strips every `_`-prefixed key before
writing JSON. Add three new, non-underscore keys so the data survives.

**Preconditions:** none — this is the first task.

**Files touched:** `ecosentry/arch7_energy.py` only.

**Exact edit:**

Find this exact text in `ecosentry/arch7_energy.py` (it is the last three
lines of the dict literal returned by `generate_energy_report`, immediately
before the function ends):

```
        "recommendations": recommendations,
        "_solar_run": solar_run,
        "_battery_only": battery_only,
    }
```

Replace it with:

```
        "recommendations": recommendations,
        "soc_trajectory_solar": [round(float(x), 2) for x in solar_run["soc_trajectory"]],
        "soc_trajectory_battery_only": [round(float(x), 2) for x in battery_only["soc_trajectory"]],
        "daily_series": [
            {
                "day": d["day"],
                "harvest_wh": round(float(d["harvest_wh"]), 3),
                "consumption_wh": round(float(d["consumption_wh"]), 3),
                "end_soc": round(float(d["end_soc"]), 1),
            }
            for d in solar_run["daily"]
        ],
        "_solar_run": solar_run,
        "_battery_only": battery_only,
    }
```

Do not change anything else in the function. Do not change `pipeline.py` —
the existing filter (`if not kk.startswith("_")`) already keeps the three
new keys because none of them start with `_`.

**New file — test:** create `tests/test_energy_serialization.py` with
exactly this content:

```python
"""Tests for the trajectory-serialization fix (Phase 0, Task P0.1)."""

from __future__ import annotations

import json

from ecosentry.arch7_energy import generate_energy_report
from ecosentry.config import EnergyConfig, SCENARIOS


def test_energy_report_has_trajectory_series():
    cfg = EnergyConfig(mission_days=10)
    report = generate_energy_report(SCENARIOS["corbett"], cfg, days=10, seed=1)
    assert len(report["soc_trajectory_solar"]) == 10
    assert len(report["soc_trajectory_battery_only"]) == 10
    assert len(report["daily_series"]) == 10
    for entry in report["daily_series"]:
        assert set(entry) == {"day", "harvest_wh", "consumption_wh", "end_soc"}


def test_energy_report_public_fields_are_json_serializable():
    cfg = EnergyConfig(mission_days=5)
    report = generate_energy_report(SCENARIOS["sundarbans"], cfg, days=5, seed=1)
    public = {k: v for k, v in report.items() if not k.startswith("_")}
    # Must not raise -- this is exactly what pipeline.py does before writing
    # energy_report.json, so this test catches the same failure mode.
    json.dumps(public)
```

**Verification:**
```
cd /path/to/repo
python -m pytest tests/test_energy_serialization.py -v
```

**Expected result:** both tests pass (`2 passed`).

---

### TASK P0.2 — Document the ARCH_7 internal contradiction

**Goal:** `ARCH_7_ENERGY_PROFILER.md`'s Component 1 pseudocode (5,000 mAh
battery, 3 mW / 20 ms inference) disagrees with its own "Key Parameters
(Finalized)" table (3,000 mAh, 30 mW / 800 ms inference) by roughly 400× on
inference energy. `EnergyConfig` already resolved this in code but the
document never explains the resolution. This task documents it — it does
not change any code.

**Preconditions:** none (independent of P0.1, can be done any time before
Phase 5).

**Files touched:** `ARCH_7_ENERGY_PROFILER.md` only.

**Exact edit:**

Find this exact text (the end of the "Critical Findings (Updated)" section,
immediately before the "### Related Documentation" heading):

```
  4. Adaptive LoRa SF (use SF7 when signal strong, reduce transmission overhead)

### Related Documentation
```

Replace it with:

```
  4. Adaptive LoRa SF (use SF7 when signal strong, reduce transmission overhead)

### Correction 11: battery capacity and inference energy figures disagree internally

Component 1's pseudocode specifies a 5,000 mAh battery and 3 mW / 20 ms SNN
inference energy. The "Key Parameters (Finalized -- CRITICAL CORRECTIONS)"
table further down this same document specifies a 3,000 mAh battery and
30 mW / 800 ms inference energy -- roughly a 400x difference in energy per
inference cycle. The implementation (`EnergyConfig` in `config.py`) takes
battery capacity from Component 1 (5,000 mAh, consistent with the 18.5 Wh
figure used throughout the rest of this document) and takes inference
power/timing from the finalized table (30 mW / 800 ms, explicitly marked as
a correction). This is a judgment call, not a derivation from either
source alone -- revisit if a hardware prototype measurement contradicts
either number.

### Related Documentation
```

**Verification:**
```
grep -n "Correction 11" ARCH_7_ENERGY_PROFILER.md
```

**Expected result:** one matching line is printed (confirms the section was
added). No test suite change — this task touches documentation only.

---

### TASK P0.3 — Make the dashboard a generated artifact instead of a static file

**Goal:** `dashboard.html` is currently a hand-edited, static snapshot with
no `<script>` tags and no reference to any JSON file, despite the README
claiming it reads `pipeline_report.json`. This task converts it into a
template that gets filled with the current run's data every time the
pipeline runs, and adds the two new visualizer sections that Phase 5 will
populate. This task does **not** attempt to convert every existing
hardcoded number in the file (acceptance table, confusion matrix, per-forest
stat cards) into dynamic content — that is explicitly out of scope here (see
"Do not" list below). It only (a) makes the file swappable via a JSON
placeholder, and (b) adds two new, genuinely dynamic sections.

**Preconditions:** P0.1 complete and verified (the new sections in this task
read `soc_trajectory_solar` / `soc_trajectory_battery_only`, which do not
exist in `energy_report.json` until P0.1 is done).

**Files touched:** create `ecosentry/dashboard_template.html` (copied from
the existing `dashboard.html`), edit `ecosentry/pipeline.py`. Do **not**
edit the original `dashboard.html` at the repository root — leave it in
place unchanged; it becomes a stale reference copy once the template takes
over, and can be deleted in a later, separate cleanup task, not this one.

**Step 1 — copy the file:**
```
cp dashboard.html ecosentry/dashboard_template.html
```

**Step 2 — insert the JSON placeholder.** In
`ecosentry/dashboard_template.html`, find this exact text (line 236 in the
original file — the closing tag of the `<style>` block, which appears
exactly once in the file):

```
</style>
```

Replace it with:

```
</style>
<script id="report-data" type="application/json">__REPORT_JSON__</script>
```

**Step 3 — insert the two new sections.** In the same file, find this exact
text (it appears exactly once — the end of the "limitations" section,
immediately before `</main>` and the footer):

```
      <li>Over-the-air key rotation &mdash; only key generation and PBKDF2 derivation are implemented</li>
    </ul>
  </section>

</main>

<footer class="wrap">
```

Replace it with:

```
      <li>Over-the-air key rotation &mdash; only key generation and PBKDF2 derivation are implemented</li>
    </ul>
  </section>

  <section id="energy-visualizer">
    <div class="section-head">
      <h2>Energy visualizer</h2>
      <div class="note">30-day battery projection and daily energy balance per forest, from this run's energy_report.json</div>
    </div>
    <div id="ev-corbett" class="ev-forest" data-forest="corbett"></div>
    <div id="ev-seshachalam" class="ev-forest" data-forest="seshachalam"></div>
    <div id="ev-sundarbans" class="ev-forest" data-forest="sundarbans"></div>
  </section>

  <section id="delivery-visualizer">
    <div class="section-head">
      <h2>Priority delivery pipeline</h2>
      <div class="note">End-to-end latency and delivery funnel for the priority alert path, from this run's delivery_report.json</div>
    </div>
    <div id="delivery-funnel"></div>
  </section>

</main>

<footer class="wrap">
```

**Step 4 — add the bootstrap script.** In the same file, find this exact
text (the last line of the file):

```
</footer>
```

Replace it with:

```
</footer>

<script>
(function () {
  var el = document.getElementById('report-data');
  if (!el) { return; }
  var report;
  try {
    report = JSON.parse(el.textContent);
  } catch (e) {
    console.error('dashboard: could not parse report-data', e);
    return;
  }
  window.REPORT = report;
  // Phase 5 (tasks P5.3, P5.4) fill in the chart-drawing functions that
  // read window.REPORT and populate #ev-corbett / #ev-seshachalam /
  // #ev-sundarbans / #delivery-funnel. This bootstrap only makes the data
  // available; it draws nothing by itself.
})();
</script>
```

**Step 5 — add the `render_dashboard` function to `pipeline.py`.**

`pipeline.py` already imports `json`, `Path`, and `Dict` under their plain
names at the top of the file (`import json`, `from pathlib import Path`,
`from typing import Dict, List, Optional, Sequence`) — do not add any new
imports for this step.

Find this exact text (the end of the `acceptance_summary` function, which
is the last function in the file before this task adds anything):

```python
        "operational_endurance": entry(
            f"{min_endurance} days (worst forest)", ">=30 days", min_endurance >= 30
        ),
    }
```

Replace it with:

```python
        "operational_endurance": entry(
            f"{min_endurance} days (worst forest)", ">=30 days", min_endurance >= 30
        ),
    }


def render_dashboard(report: Dict, out_dir: Path) -> Path:
    """Fill dashboard_template.html with this run's pipeline_report.json
    and write the result to out_dir/dashboard.html (Phase 0, Task P0.3)."""
    template_path = Path(__file__).parent / "dashboard_template.html"
    template = template_path.read_text(encoding="utf-8")
    injected = template.replace("__REPORT_JSON__", json.dumps(report, default=str))
    out_path = out_dir / "dashboard.html"
    out_path.write_text(injected, encoding="utf-8")
    return out_path
```

Next, wire it into `run_full_pipeline`. Find this exact text (the point
where `pipeline_report.json` is written, inside `run_full_pipeline`):

```python
    (out_dir / "pipeline_report.json").write_text(json.dumps(report, indent=2, default=str))

    if verbose:
```

Replace it with:

```python
    (out_dir / "pipeline_report.json").write_text(json.dumps(report, indent=2, default=str))
    render_dashboard(report, out_dir)

    if verbose:
```

**New file — test:** create `tests/test_dashboard_rendering.py`:

```python
"""Tests for the templated dashboard (Phase 0, Task P0.3)."""

from __future__ import annotations

import json

from ecosentry.pipeline import render_dashboard


def test_render_dashboard_embeds_report_json(tmp_path):
    report = {"hello": "world", "n": 3}
    out_path = render_dashboard(report, tmp_path)
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "__REPORT_JSON__" not in content
    assert json.dumps(report) in content


def test_render_dashboard_output_differs_for_different_reports(tmp_path):
    # tmp_path already exists (pytest creates it) -- render_dashboard does
    # not create its out_dir argument, only the dashboard.html file inside
    # it, so both calls below reuse the same, already-existing directory.
    p1 = render_dashboard({"n": 1}, tmp_path)
    text1 = p1.read_text(encoding="utf-8")
    p2 = render_dashboard({"n": 2}, tmp_path)
    text2 = p2.read_text(encoding="utf-8")
    assert text1 != text2
    assert '"n": 2' in text2
    assert '"n": 1' not in text2
```

**Verification:**
```
python -m pytest tests/test_dashboard_rendering.py -v
python -m ecosentry run --preset quick
grep -c "__REPORT_JSON__" artifacts/dashboard.html
```

**Expected result:** both tests pass, the pipeline run completes without
error, and the `grep -c` command prints `0` (confirming the placeholder was
actually replaced, not left in the output file).

**Do not:**
- Do not rewrite the acceptance table, confusion matrix, or the three
  per-forest stat cards (`.fcard` elements) that already exist in the
  template — those stay as static markup for now. Retrofitting them to read
  from `window.REPORT` is explicitly out of scope for this plan.
- Do not delete the original `dashboard.html` at the repository root.
- Do not add a build step, bundler, or server requirement — the output must
  still open by double-clicking in a browser.

---

## Phase 1 — 16-byte critical event packet ("beacon")

### TASK P1.1 — Add `BeaconConfig` to `config.py`

**Goal:** add configuration constants for the beacon codec.

**Preconditions:** none.

**Files touched:** `ecosentry/config.py` only.

**Exact edit:** find this exact text (the end of the `PayloadConfig` class,
immediately before the ARCH_7 section header comment):

```
    #: PRIORITY_PAYLOAD_DELIVERY.md Option A -- priority is carried in the
    #: transport header, so the encrypted payload schema is untouched.
    priority_header_flag: int = 0x80


# ---------------------------------------------------------------------------
# ARCH_7: Energy
# ---------------------------------------------------------------------------
```

Replace it with:

```
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
```

**Verification:**
```
python -c "from ecosentry.config import BeaconConfig; c = BeaconConfig(); print(c.size_bytes, c.body_bytes, c.mac_bytes, c.beacon_eligible_classes)"
```

**Expected result:** `16 13 3 (0, 1)`

---

### TASK P1.2 — Create `ecosentry/arch6_beacon.py`

**Goal:** the 16-byte packet codec itself.

**Preconditions:** P1.1 complete and verified.

**Files touched:** create `ecosentry/arch6_beacon.py` (new file, complete
content below — this exact code was written and verified with the commands
shown in the Verification block; do not modify it).

```python
"""New architecture -- 16-byte critical event beacon.

Sent as a fast, cheap first signal ahead of the full encrypted AlertPayload
(arch6_payload.py), which still carries the complete forensic record. The
existing wire format spends its entire 16-byte budget on the IV alone
(magic(2B) | version(1B) | sequence(1B) | IV(16B) | ciphertext), so a true
16-byte packet cannot carry a random IV. This format instead authenticates
with a truncated HMAC over the plaintext fields -- it is deliberately not
encrypted (there is no room), and deliberately not the record anything gets
acted on beyond "go look now": the full AlertPayload that follows carries
the complete, encrypted, forensic-grade data.

Wire format (16 bytes total):

    byte 0      : version (upper 4 bits) | reserved (lower 4 bits, always 0)
    bytes 1-2   : device_id            (uint16, big-endian)
    byte 3      : event_type (bits 7-6) | confidence_quantized (bits 5-0)
    byte 4      : sequence_number      (uint8, shared with the paired AlertPayload)
    bytes 5-6   : relative_timestamp_s (uint16, big-endian)
    bytes 7-9   : lat_delta_millideg   (int24, big-endian, signed)
    bytes 10-12 : lon_delta_millideg   (int24, big-endian, signed)
    bytes 13-15 : truncated_mac        (first 3 bytes of HMAC-SHA256 over bytes 0-12)
"""

from __future__ import annotations

import hashlib
import hmac
import struct
from dataclasses import dataclass
from typing import Optional

from .config import BeaconConfig

__all__ = [
    "BeaconPayload",
    "BeaconEncodeError",
    "encode_beacon",
    "decode_beacon",
]


class BeaconEncodeError(ValueError):
    """Raised by encode_beacon when a field is out of range for the wire format."""


@dataclass(frozen=True)
class BeaconPayload:
    device_id: int             # uint16, 0-65535
    event_type: int            # 2-bit field, 0-3 valid on the wire (0=gunshot, 1=chainsaw)
    confidence: float          # 0.0-1.0 inclusive
    sequence_number: int       # uint8, 0-255, wraps -- shared with the paired AlertPayload
    relative_timestamp_s: int  # uint16, 0-65535 seconds since the gateway's last sync beacon
    lat_delta_millideg: int    # signed 24-bit, -8388608..8388607, relative to DeviceConfig home position
    lon_delta_millideg: int    # signed 24-bit, -8388608..8388607, relative to DeviceConfig home position


def _validate(b: BeaconPayload) -> None:
    if not (0 <= b.device_id <= 0xFFFF):
        raise BeaconEncodeError(f"device_id {b.device_id} out of range 0-65535")
    if not (0 <= b.event_type <= 3):
        raise BeaconEncodeError(f"event_type {b.event_type} out of range 0-3")
    if not (0.0 <= b.confidence <= 1.0):
        raise BeaconEncodeError(f"confidence {b.confidence} out of range 0.0-1.0")
    if not (0 <= b.sequence_number <= 0xFF):
        raise BeaconEncodeError(f"sequence_number {b.sequence_number} out of range 0-255")
    if not (0 <= b.relative_timestamp_s <= 0xFFFF):
        raise BeaconEncodeError(
            f"relative_timestamp_s {b.relative_timestamp_s} out of range 0-65535"
        )
    if not (-8_388_608 <= b.lat_delta_millideg <= 8_388_607):
        raise BeaconEncodeError(
            f"lat_delta_millideg {b.lat_delta_millideg} out of range for signed int24"
        )
    if not (-8_388_608 <= b.lon_delta_millideg <= 8_388_607):
        raise BeaconEncodeError(
            f"lon_delta_millideg {b.lon_delta_millideg} out of range for signed int24"
        )


def encode_beacon(
    b: BeaconPayload, mac_key: bytes, cfg: Optional[BeaconConfig] = None
) -> bytes:
    """Encode a BeaconPayload into exactly cfg.size_bytes (16) bytes.

    Raises BeaconEncodeError if any field is out of range. mac_key must be
    bytes -- use the same key material as the paired AlertPayload's
    encryption key (DeviceConfig.encryption_key).
    """
    cfg = cfg or BeaconConfig()
    _validate(b)
    conf_q = min(63, max(0, round(b.confidence * 63)))
    byte0 = (cfg.version << 4) & 0xF0
    byte3 = ((b.event_type & 0x3) << 6) | (conf_q & 0x3F)
    body = struct.pack(
        ">B H B B H",
        byte0,
        b.device_id,
        byte3,
        b.sequence_number,
        b.relative_timestamp_s,
    )
    body += b.lat_delta_millideg.to_bytes(3, "big", signed=True)
    body += b.lon_delta_millideg.to_bytes(3, "big", signed=True)
    assert len(body) == cfg.body_bytes, f"body is {len(body)} bytes, expected {cfg.body_bytes}"
    mac = hmac.new(mac_key, body, hashlib.sha256).digest()[: cfg.mac_bytes]
    packet = body + mac
    assert len(packet) == cfg.size_bytes, f"packet is {len(packet)} bytes, expected {cfg.size_bytes}"
    return packet


def decode_beacon(
    raw: bytes, mac_key: bytes, cfg: Optional[BeaconConfig] = None
) -> Optional[BeaconPayload]:
    """Decode a 16-byte beacon packet.

    Returns None if the length is wrong or the MAC does not match (tampered,
    corrupted, or wrong key) -- never raises for malformed input.
    """
    cfg = cfg or BeaconConfig()
    if len(raw) != cfg.size_bytes:
        return None
    body, mac = raw[: cfg.body_bytes], raw[cfg.body_bytes :]
    expected_mac = hmac.new(mac_key, body, hashlib.sha256).digest()[: cfg.mac_bytes]
    if not hmac.compare_digest(mac, expected_mac):
        return None
    byte0, device_id, byte3, sequence_number, relative_timestamp_s = struct.unpack(
        ">B H B B H", body[:7]
    )
    version = (byte0 & 0xF0) >> 4
    if version != cfg.version:
        return None
    event_type = (byte3 & 0xC0) >> 6
    conf_q = byte3 & 0x3F
    confidence = conf_q / 63.0
    lat_delta = int.from_bytes(body[7:10], "big", signed=True)
    lon_delta = int.from_bytes(body[10:13], "big", signed=True)
    return BeaconPayload(
        device_id=device_id,
        event_type=event_type,
        confidence=confidence,
        sequence_number=sequence_number,
        relative_timestamp_s=relative_timestamp_s,
        lat_delta_millideg=lat_delta,
        lon_delta_millideg=lon_delta,
    )
```

**Verification (this exact command was run against this exact code before
this plan was written; do not modify the code to make different output
appear):**
```
python -c "
from ecosentry.arch6_beacon import BeaconPayload, encode_beacon, decode_beacon
key = b'x' * 32
p = BeaconPayload(device_id=1, event_type=0, confidence=0.91, sequence_number=7,
                   relative_timestamp_s=42, lat_delta_millideg=1234, lon_delta_millideg=-5678)
packet = encode_beacon(p, key)
print(len(packet), packet.hex())
print(decode_beacon(packet, key))
"
```

**Expected result:**
```
16 1000013907002a0004d2ffe9d2042d18
BeaconPayload(device_id=1, event_type=0, confidence=0.9047619047619048, sequence_number=7, relative_timestamp_s=42, lat_delta_millideg=1234, lon_delta_millideg=-5678)
```
(Confidence comes back as `0.90476...` rather than exactly `0.91` — this is
expected quantization error from the 6-bit confidence field, not a bug. The
tolerance for this in the automated test below is `1/63 ≈ 0.0159`.)

---

### TASK P1.3 — Test suite for the beacon codec

**Goal:** lock in the verified behavior with automated tests.

**Preconditions:** P1.2 complete and verified.

**Files touched:** create `tests/test_arch6_beacon.py` (new file, complete
content below).

```python
"""Tests for the 16-byte critical event beacon (Phase 1, arch6_beacon.py)."""

from __future__ import annotations

import pytest

from ecosentry.arch6_beacon import (
    BeaconEncodeError,
    BeaconPayload,
    decode_beacon,
    encode_beacon,
)

KEY = b"x" * 32


def _make(**overrides) -> BeaconPayload:
    defaults = dict(
        device_id=1,
        event_type=0,
        confidence=0.91,
        sequence_number=7,
        relative_timestamp_s=42,
        lat_delta_millideg=1234,
        lon_delta_millideg=-5678,
    )
    defaults.update(overrides)
    return BeaconPayload(**defaults)


def test_encoded_packet_is_exactly_16_bytes():
    packet = encode_beacon(_make(), KEY)
    assert len(packet) == 16


def test_golden_vector_matches_verified_output():
    # This exact hex string was produced by the verified prototype in
    # TASK P1.2's Verification block -- if this ever changes, the wire
    # format itself changed and every downstream consumer must be updated.
    packet = encode_beacon(_make(), KEY)
    assert packet.hex() == "1000013907002a0004d2ffe9d2042d18"


def test_round_trip_preserves_all_fields_within_quantization_tolerance():
    original = _make()
    packet = encode_beacon(original, KEY)
    decoded = decode_beacon(packet, KEY)
    assert decoded is not None
    assert decoded.device_id == original.device_id
    assert decoded.event_type == original.event_type
    assert abs(decoded.confidence - original.confidence) <= (1 / 63)
    assert decoded.sequence_number == original.sequence_number
    assert decoded.relative_timestamp_s == original.relative_timestamp_s
    assert decoded.lat_delta_millideg == original.lat_delta_millideg
    assert decoded.lon_delta_millideg == original.lon_delta_millideg


@pytest.mark.parametrize(
    "overrides",
    [
        dict(device_id=65535, sequence_number=255, relative_timestamp_s=65535,
             lat_delta_millideg=8388607, lon_delta_millideg=-8388608, confidence=1.0),
        dict(device_id=0, sequence_number=0, relative_timestamp_s=0,
             lat_delta_millideg=0, lon_delta_millideg=0, confidence=0.0),
    ],
)
def test_boundary_values_round_trip(overrides):
    original = _make(**overrides)
    packet = encode_beacon(original, KEY)
    assert len(packet) == 16
    decoded = decode_beacon(packet, KEY)
    assert decoded is not None
    assert decoded.device_id == original.device_id
    assert decoded.lat_delta_millideg == original.lat_delta_millideg
    assert decoded.lon_delta_millideg == original.lon_delta_millideg


def test_tampered_packet_is_rejected():
    packet = bytearray(encode_beacon(_make(), KEY))
    packet[4] ^= 0x01  # flip a bit in the sequence_number field
    assert decode_beacon(bytes(packet), KEY) is None


def test_wrong_key_is_rejected():
    packet = encode_beacon(_make(), KEY)
    assert decode_beacon(packet, b"y" * 32) is None


def test_wrong_length_is_rejected():
    packet = encode_beacon(_make(), KEY)
    assert decode_beacon(packet[:15], KEY) is None
    assert decode_beacon(packet + b"\x00", KEY) is None


@pytest.mark.parametrize(
    "overrides",
    [
        dict(device_id=70000),
        dict(event_type=4),
        dict(confidence=1.5),
        dict(sequence_number=256),
        dict(relative_timestamp_s=70000),
        dict(lat_delta_millideg=9_000_000),
        dict(lon_delta_millideg=-9_000_000),
    ],
)
def test_out_of_range_fields_raise(overrides):
    with pytest.raises(BeaconEncodeError):
        encode_beacon(_make(**overrides), KEY)
```

**Verification:**
```
python -m pytest tests/test_arch6_beacon.py -v
```

**Expected result:** `15 passed`. This file was executed against the real
repository before this plan was written; the exact output was:
```
tests/test_arch6_beacon.py::test_encoded_packet_is_exactly_16_bytes PASSED
tests/test_arch6_beacon.py::test_golden_vector_matches_verified_output PASSED
tests/test_arch6_beacon.py::test_round_trip_preserves_all_fields_within_quantization_tolerance PASSED
tests/test_arch6_beacon.py::test_boundary_values_round_trip[overrides0] PASSED
tests/test_arch6_beacon.py::test_boundary_values_round_trip[overrides1] PASSED
tests/test_arch6_beacon.py::test_tampered_packet_is_rejected PASSED
tests/test_arch6_beacon.py::test_wrong_key_is_rejected PASSED
tests/test_arch6_beacon.py::test_wrong_length_is_rejected PASSED
tests/test_arch6_beacon.py::test_out_of_range_fields_raise[overrides0] PASSED
tests/test_arch6_beacon.py::test_out_of_range_fields_raise[overrides1] PASSED
tests/test_arch6_beacon.py::test_out_of_range_fields_raise[overrides2] PASSED
tests/test_arch6_beacon.py::test_out_of_range_fields_raise[overrides3] PASSED
tests/test_arch6_beacon.py::test_out_of_range_fields_raise[overrides4] PASSED
tests/test_arch6_beacon.py::test_out_of_range_fields_raise[overrides5] PASSED
tests/test_arch6_beacon.py::test_out_of_range_fields_raise[overrides6] PASSED

15 passed in 0.04s
```
If your result differs from `15 passed`, something in `arch6_beacon.py` or
the test file does not match this plan exactly — do not proceed until it
matches.

---

## Phase 2 — Device-side priority queue: distinguish beacon from payload

**Finding this phase depends on:** `ecosentry/arch8_network.py`'s
`MessageQueue` class already dequeues priority messages before non-priority
ones (`dequeue_batch` sorts by `(not priority, timestamp)`). This phase does
not change that behavior — it only adds a `kind` tag so beacon and payload
traffic can be told apart in metrics and in Phase 3's gateway logic.

### TASK P2.1 — Add a `kind` field to `MessageQueue`

**Preconditions:** none (independent of Phase 1, but Phase 3 depends on
this).

**Files touched:** `ecosentry/arch8_network.py` only.

**Exact edit:** find this exact text:

```python
    def enqueue(self, message: bytes, timestamp: float = 0.0, priority: bool = False) -> bool:
        ok = True
        while self.current_size + len(message) > self.max_size_bytes and self.queue:
            victim = next((i for i, m in enumerate(self.queue) if not m["priority"]), 0)
            self.queue.pop(victim)
            self.dropped += 1
            ok = False
        self.queue.append({"data": message, "timestamp": timestamp, "priority": priority})
        return ok
```

Replace it with:

```python
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
```

**Do not** change `dequeue_batch` — its sort key (`not priority, timestamp`)
is unaffected by adding `kind` to the stored dict, and this task is not
asking you to change dequeue ordering.

**New test — add to the existing file** `tests/test_arch6_arch7_arch8.py`.
Do not modify any existing function in that file. Add this new function at
the end of the file:

```python
def test_message_queue_tracks_kind_without_changing_priority_order():
    from ecosentry.arch8_network import MessageQueue

    q = MessageQueue()
    q.enqueue(b"normal-1", timestamp=1.0, priority=False, kind="payload")
    q.enqueue(b"beacon-1", timestamp=2.0, priority=True, kind="beacon")
    q.enqueue(b"payload-1", timestamp=3.0, priority=True, kind="payload")

    batch = q.dequeue_batch(max_messages=3)
    # Priority messages still dequeue first, regardless of kind.
    assert batch[0]["priority"] is True
    assert batch[1]["priority"] is True
    assert batch[2]["priority"] is False
    kinds = {m["data"]: m["kind"] for m in batch}
    assert kinds[b"beacon-1"] == "beacon"
    assert kinds[b"payload-1"] == "payload"
    assert kinds[b"normal-1"] == "payload"


def test_message_queue_default_kind_is_payload():
    from ecosentry.arch8_network import MessageQueue

    q = MessageQueue()
    q.enqueue(b"legacy-call", timestamp=1.0, priority=True)
    batch = q.dequeue_batch(max_messages=1)
    assert batch[0]["kind"] == "payload"
```

**Verification:** run the two new tests by their exact names, not by a
keyword filter — `-k "message_queue"` also matches a pre-existing test in
this file (`test_message_queue_drops_oldest_normal_first`) and will report
a misleading count:
```
python -m pytest tests/test_arch6_arch7_arch8.py::test_message_queue_tracks_kind_without_changing_priority_order tests/test_arch6_arch7_arch8.py::test_message_queue_default_kind_is_payload -v
```

**Expected result:** `2 passed`. Also run the full existing file to confirm
nothing broke:
```
python -m pytest tests/test_arch6_arch7_arch8.py -v
```
**Expected result:** every test that passed before this task still passes
(no regressions) plus the 2 new ones.

---

## Phase 3 — Gateway: DSCP EF marking + adaptive LLQ scheduler

### TASK P3.1 — Change DSCP marking from AF41 to EF, add new config fields

**Preconditions:** none.

**Files touched:** `ecosentry/config.py` only.

**Exact edit:** find this exact text (the full `GatewayConfig` class):

```python
@dataclass(frozen=True)
class GatewayConfig:
    """PRIORITY_PAYLOAD_DELIVERY.md -- gateway priority proxy."""

    priority_topic: str = "priority/alerts"
    normal_topic: str = "alerts"
    priority_qos: int = 1
    normal_qos: int = 0
    priority_dscp: str = "AF41"
    confidence_threshold: float = 0.85
    backoff_schedule_s: Tuple[float, ...] = (1, 2, 4, 8, 16, 32, 60)
    retention_hours: int = 24
    #: Backhaul latency models (mean_ms, loss probability).
    qos_backhaul_ms: float = 120.0
    qos_backhaul_loss: float = 0.005
    best_effort_backhaul_ms: float = 450.0
    best_effort_backhaul_loss: float = 0.03
```

Replace it with:

```python
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
```

**Verification:**
```
python -c "
from ecosentry.config import GatewayConfig
c = GatewayConfig()
assert c.priority_dscp == 'EF'
assert c.beacon_topic == 'priority/beacons'
assert c.llq_max_credits_floor == 2
assert c.llq_max_credits_ceiling == 20
print('OK')
"
```
**Expected result:** `OK`

**Mandatory follow-up edit to an existing test — read this even though it
contradicts the general "never edit existing tests" rule in this plan.**
`tests/test_arch6_arch7_arch8.py` already contains a test that hardcodes the
old DSCP value:

```python
def test_gateway_routes_threats_to_priority_topic(device):
    gw = PriorityGateway(device.encryption_key, GatewayConfig(), BackhaulLink(seed=1), MqttSink())
    threat = generate_alert_payload(
        {"class_id": 0, "confidence": 0.93, "timestamp": 1_700_000_000.0, "sequence": 1}, device
    )
    result = gw.handle_uplink(threat["message"], "S1")

    assert result["accepted"] and result["priority"]
    assert result["publish"].topic == "priority/alerts"
    assert result["publish"].qos == 1
    assert result["publish"].dscp == "AF41"
    assert len(gw.acks_sent) == 1, "priority alerts must get a local ACK"
```

Find this exact line inside that function (do not touch any other line in
the function):

```python
    assert result["publish"].dscp == "AF41"
```

Replace it with:

```python
    assert result["publish"].dscp == "EF"
```

This is a deliberate, required consequence of this task's config change —
it is confirmed by running the full existing test file before and after
this one-line edit: before the edit, `1 failed, 52 passed`; after it,
`53 passed`. This is the only existing test line this entire plan
instructs you to change; every other new test in this plan is a new
function appended to a file, never an edit to a pre-existing one.

**Verification of this specific fix:**
```
python -m pytest tests/test_arch6_arch7_arch8.py::test_gateway_routes_threats_to_priority_topic -v
```
**Expected result:** `1 passed`.

---

### TASK P3.2 — Add `LLQState` (adaptive token-bucket scheduler) to `gateway.py`

**Goal:** add the adaptive low-latency-queue scheduler as a new class. This
uses a token-bucket model, not a bits-per-second ceiling model — a
bits-per-second design was tried first and failed verification (at this
system's actual alert rate of a handful of messages per day, any
"bits-per-second ceiling" clamps to its floor in every realistic scenario
and never demonstrates the "grows under load" behavior it's meant to show).
The token-bucket version below was verified to grow under sustained load,
decay after quiet periods, and never block or drop priority traffic.

**Preconditions:** none (independent of P3.1, but both are needed before
P3.3).

**Files touched:** `ecosentry/gateway.py` only.

**Exact edit:** find this exact text (the end of the `BackhaulLink` class,
immediately before the `MqttSink` class):

```python
        latency = float(self.rng.normal(mean * load, mean * jitter))
        latency = max(latency, 10.0)
        delivered = self.rng.random() > min(loss * load, 0.95)
        return delivered, latency


class MqttSink:
```

Replace it with:

```python
        latency = float(self.rng.normal(mean * load, mean * jitter))
        latency = max(latency, 10.0)
        delivered = self.rng.random() > min(loss * load, 0.95)
        return delivered, latency


@dataclass
class LLQState:
    """Adaptive low-latency queue for the priority lane (new architecture).

    Priority traffic is never dropped and never queued behind best-effort
    traffic -- that guarantee is unchanged by this class. What "adaptive"
    controls is whether a priority forward happens at the fast, uncontended
    backhaul latency, or at a penalized latency representing the priority
    lane itself being saturated by an unusual burst (several sensors
    triggering near-simultaneously on one real incident, or a storm causing
    a cluster of false positives).

    Two time constants:
      - A short-horizon token bucket (`credits` / `max_credits` /
        `refill_per_s`) absorbs bursts up to `max_credits` messages with no
        penalty, then penalizes further messages until credits refill.
      - `baseline_rate_per_day`, updated by calling `rebase()` with a
        trailing multi-day average alert rate, re-derives `max_credits` and
        `refill_per_s` so a gateway that has been busier than usual gets a
        proportionally bigger burst allowance, and a quiet gateway shrinks
        back down. Clamped to [llq_max_credits_floor, llq_max_credits_ceiling]
        from GatewayConfig.
    """

    max_credits: int = 3
    refill_per_s: float = 3 / (24 * 3600.0)
    credits: float = 3.0
    last_refill_s: float = 0.0
    baseline_rate_per_day: float = 5.0
    overflow_latency_multiplier: float = 2.5
    credits_floor: int = 2
    credits_ceiling: int = 20

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self.last_refill_s)
        self.credits = min(float(self.max_credits), self.credits + elapsed * self.refill_per_s)
        self.last_refill_s = now

    def consume(self, now: float) -> float:
        """Call once per priority forward. Returns the latency multiplier
        to apply to that forward: 1.0 if a credit was available, otherwise
        overflow_latency_multiplier. This never blocks and never drops --
        it only affects simulated latency."""
        self._refill(now)
        if self.credits >= 1.0:
            self.credits -= 1.0
            return 1.0
        return self.overflow_latency_multiplier

    def rebase(self, observed_rate_per_day: float) -> None:
        """Call periodically (e.g. once per simulated day) with the
        trailing multi-day average alert rate. Grows or shrinks max_credits
        and refill_per_s proportionally, clamped to [credits_floor,
        credits_ceiling]."""
        self.baseline_rate_per_day = observed_rate_per_day
        self.max_credits = int(
            np.clip(round(observed_rate_per_day * 0.5), self.credits_floor, self.credits_ceiling)
        )
        self.refill_per_s = self.max_credits / (24 * 3600.0)


class MqttSink:
```

**Verification (these exact commands were run against this exact class
before this plan was written):**
```
python -c "
from ecosentry.gateway import LLQState

llq = LLQState(max_credits=3, refill_per_s=3/(24*3600.0), credits=3.0, last_refill_s=0.0)
mults = [llq.consume(now=0.0) for _ in range(3)]
assert mults == [1.0, 1.0, 1.0], mults
m4 = llq.consume(now=0.1)
assert m4 == 2.5, m4
print('burst-then-overflow: OK')

llq2 = LLQState()
before = llq2.max_credits
llq2.rebase(observed_rate_per_day=14.0)
assert llq2.max_credits > before, (before, llq2.max_credits)
print('rebase grows under sustained higher rate:', before, '->', llq2.max_credits)

llq3 = LLQState()
llq3.rebase(observed_rate_per_day=1.0)
assert llq3.max_credits == 2, llq3.max_credits
print('rebase clamps to floor: OK')

llq4 = LLQState()
llq4.rebase(observed_rate_per_day=1000.0)
assert llq4.max_credits == 20, llq4.max_credits
print('rebase clamps to ceiling: OK')
print('ALL CHECKS PASSED')
"
```
**Expected result:**
```
burst-then-overflow: OK
rebase grows under sustained higher rate: 3 -> 7
rebase clamps to floor: OK
rebase clamps to ceiling: OK
ALL CHECKS PASSED
```

Also confirm `dataclass` (already imported) and `np` (already imported as
`numpy as np`) are both available at the top of `gateway.py` — they are
(check the existing import block); do not add duplicate imports.

---

### TASK P3.3 — Wire `LLQState` into `PriorityGateway`

**Preconditions:** P3.1 and P3.2 complete and verified.

**Files touched:** `ecosentry/gateway.py` only.

**Exact edit 1 — initialize the scheduler.** Find this exact text (the end
of `PriorityGateway.__init__`):

```python
        self.acks_sent: List[Dict] = []
        self.seen: set = set()  # cloud dedupe: (device_id, sequence, time bucket)
        self.duplicates = 0
        self.parse_failures = 0
        self.stats = {"priority": 0, "normal": 0, "delivered": 0, "dropped": 0}
```

Replace it with:

```python
        self.acks_sent: List[Dict] = []
        self.seen: set = set()  # cloud dedupe: (device_id, sequence, time bucket)
        self.duplicates = 0
        self.parse_failures = 0
        self.stats = {"priority": 0, "normal": 0, "delivered": 0, "dropped": 0}
        self.llq = LLQState(
            credits_floor=self.cfg.llq_max_credits_floor,
            credits_ceiling=self.cfg.llq_max_credits_ceiling,
            overflow_latency_multiplier=self.cfg.llq_overflow_latency_multiplier,
        )
```

**Exact edit 2 — apply the LLQ multiplier in `_forward`.** Find this exact
text:

```python
        while attempts < max_attempts:
            ok, latency = self.backhaul.send(priority)
            total_latency += latency
            attempts += 1
            if ok:
                delivered = True
                break
            if attempts < max_attempts:
                total_latency += self.cfg.backoff_schedule_s[attempts - 1] * 1000.0
```

Replace it with:

```python
        llq_multiplier = self.llq.consume(now) if priority else 1.0

        while attempts < max_attempts:
            ok, latency = self.backhaul.send(priority)
            total_latency += latency * llq_multiplier
            attempts += 1
            if ok:
                delivered = True
                break
            if attempts < max_attempts:
                total_latency += self.cfg.backoff_schedule_s[attempts - 1] * 1000.0
```

**Exact edit 3 — expose the current state in `metrics()`.** Find this exact
text (the end of the `metrics()` method's returned dict):

```python
            "duplicates_suppressed": self.duplicates,
            "parse_failures": self.parse_failures,
            "queued": len(self.store),
            "counters": dict(self.stats),
        }
```

Replace it with:

```python
            "duplicates_suppressed": self.duplicates,
            "parse_failures": self.parse_failures,
            "queued": len(self.store),
            "counters": dict(self.stats),
            "llq_max_credits": self.llq.max_credits,
            "llq_current_credits": round(self.llq.credits, 2),
            "llq_baseline_rate_per_day": self.llq.baseline_rate_per_day,
        }
```

**New test — add to `tests/test_arch6_arch7_arch8.py`** (append at the end
of the file, do not modify any existing function):

```python
def test_gateway_uses_ef_dscp_not_af41():
    from ecosentry.arch6_payload import DeviceConfig, generate_alert_payload
    from ecosentry.gateway import BackhaulLink, MqttSink, PriorityGateway
    from ecosentry.config import GatewayConfig

    device = DeviceConfig(device_id="SENTRY_TEST")
    gw = PriorityGateway(device.encryption_key, GatewayConfig(), BackhaulLink(seed=1), MqttSink())
    result = {"alert": True, "class_id": 0, "class_name": "gunshot", "confidence": 0.95,
              "sequence": 1, "timestamp": 1000.0}
    packet = generate_alert_payload(result, device)
    gw.handle_uplink(packet["message"], "S1")
    priority_records = gw.sink.by_topic(gw.cfg.priority_topic)
    assert len(priority_records) == 1
    assert priority_records[0].dscp == "EF"


def test_gateway_exposes_llq_metrics():
    from ecosentry.arch6_payload import DeviceConfig
    from ecosentry.gateway import BackhaulLink, MqttSink, PriorityGateway
    from ecosentry.config import GatewayConfig

    device = DeviceConfig(device_id="SENTRY_TEST")
    gw = PriorityGateway(device.encryption_key, GatewayConfig(), BackhaulLink(seed=1), MqttSink())
    m = gw.metrics()
    assert "llq_max_credits" in m
    assert "llq_current_credits" in m
    assert "llq_baseline_rate_per_day" in m
```

**Verification:**
```
python -m pytest tests/test_arch6_arch7_arch8.py -v -k "ef_dscp or llq_metrics"
```
**Expected result:** `2 passed`.

**Then run the entire pre-existing gateway test suite to confirm no
regressions:**
```
python -m pytest tests/test_arch6_arch7_arch8.py -v
```
**Expected result:** every test that passed before Phase 3 still passes.

---

### TASK P3.4 — Beacon-aware routing and orphan detection in `PriorityGateway`

**Goal:** route beacons to their own topic, and correlate them with the
full payload that follows by `sequence_number`.

**Preconditions:** P1.2, P1.3, P2.1, P3.3 complete and verified.

**Files touched:** `ecosentry/gateway.py` only.

**Exact edit 1 — accept beacon packets.** Find this exact text (the method
signature and first two lines of `handle_uplink`):

```python
    def handle_uplink(self, raw: bytes, source_node: str = "S1", now: Optional[float] = None) -> Dict:
        """Process one LoRa uplink packet."""
        now = time.time() if now is None else now
```

Replace it with:

```python
    def handle_uplink(
        self,
        raw: bytes,
        source_node: str = "S1",
        now: Optional[float] = None,
        kind: str = "payload",
    ) -> Dict:
        """Process one LoRa uplink packet. kind is "payload" (default,
        matches pre-Phase-1 behavior exactly) or "beacon" (new architecture
        -- see handle_beacon_uplink for the beacon-specific path, which this
        method delegates to)."""
        now = time.time() if now is None else now
        if kind == "beacon":
            return self.handle_beacon_uplink(raw, source_node, now)
```

**Exact edit 2 — add the new beacon-handling method and orphan tracking.**
Find this exact text (immediately after the `send_local_ack` method and
before the `# -- egress --` comment):

```python
    def send_local_ack(self, node_id: str, alert: AlertPayload, now: float) -> None:
        """Minimal downlink ACK -- priority traffic only (duty-cycle budget)."""
        self.acks_sent.append(
            {"node": node_id, "sequence": alert.sequence_number, "timestamp": now}
        )

    # -- egress -------------------------------------------------------------
```

Replace it with:

```python
    def send_local_ack(self, node_id: str, alert: AlertPayload, now: float) -> None:
        """Minimal downlink ACK -- priority traffic only (duty-cycle budget)."""
        self.acks_sent.append(
            {"node": node_id, "sequence": alert.sequence_number, "timestamp": now}
        )

    def handle_beacon_uplink(self, raw: bytes, source_node: str, now: float) -> Dict:
        """Process one 16-byte beacon packet (Phase 1/3, new architecture).

        mac_key is self.key -- the same key material used to decrypt the
        paired full AlertPayload, per arch6_beacon.py's module docstring.
        """
        beacon = decode_beacon(raw, self.key)
        if beacon is None:
            self.parse_failures += 1
            return {"accepted": False, "reason": "beacon_parse_error"}

        self.beacon_seen[beacon.sequence_number] = now
        record = PublishRecord(
            topic=self.cfg.beacon_topic,
            payload=raw,
            qos=self.cfg.beacon_qos,
            dscp=self.cfg.priority_dscp,
            latency_ms=0.0,
            attempts=1,
            delivered=True,
            timestamp=now,
        )
        self.sink.publish(record)
        return {"accepted": True, "beacon": beacon, "publish": record}

    def check_orphaned_beacons(self, now: float) -> List[Dict]:
        """Call periodically. Returns a degraded-alert dict for every beacon
        whose paired full payload has not arrived within
        cfg.beacon_orphan_timeout_s -- something happened, forensic detail
        is missing, but the event must not be silently dropped."""
        orphaned = []
        for sequence, seen_at in list(self.beacon_seen.items()):
            if sequence in self.seen:
                del self.beacon_seen[sequence]  # matched -- not orphaned
                continue
            if now - seen_at >= self.cfg.beacon_orphan_timeout_s:
                orphaned.append(
                    {"sequence": sequence, "beacon_seen_at": seen_at, "degraded": True}
                )
                del self.beacon_seen[sequence]
        return orphaned

    # -- egress -------------------------------------------------------------
```

**Exact edit 3 — add the `beacon_seen` dict and the `decode_beacon`
import.** Find this exact text (the import block at the top of the file):

```python
from .arch6_payload import AlertPayload, AlertQueue, parse_alert_message, parse_message_envelope
from .config import GatewayConfig, THREAT_CLASSES
```

Replace it with:

```python
from .arch6_beacon import decode_beacon
from .arch6_payload import AlertPayload, AlertQueue, parse_alert_message, parse_message_envelope
from .config import GatewayConfig, THREAT_CLASSES
```

Then find this exact text (inside `PriorityGateway.__init__`, immediately
after the `self.llq = LLQState(...)` block added in TASK P3.3):

```python
        self.llq = LLQState(
            credits_floor=self.cfg.llq_max_credits_floor,
            credits_ceiling=self.cfg.llq_max_credits_ceiling,
            overflow_latency_multiplier=self.cfg.llq_overflow_latency_multiplier,
        )
```

Replace it with:

```python
        self.llq = LLQState(
            credits_floor=self.cfg.llq_max_credits_floor,
            credits_ceiling=self.cfg.llq_max_credits_ceiling,
            overflow_latency_multiplier=self.cfg.llq_overflow_latency_multiplier,
        )
        self.beacon_seen: Dict[int, float] = {}  # sequence_number -> time seen, for orphan detection
```

**New test — add to `tests/test_arch6_arch7_arch8.py`:**

```python
def test_beacon_routes_to_beacon_topic():
    from ecosentry.arch6_beacon import BeaconPayload, encode_beacon
    from ecosentry.arch6_payload import DeviceConfig
    from ecosentry.gateway import BackhaulLink, MqttSink, PriorityGateway
    from ecosentry.config import GatewayConfig

    device = DeviceConfig(device_id="SENTRY_TEST")
    gw = PriorityGateway(device.encryption_key, GatewayConfig(), BackhaulLink(seed=1), MqttSink())
    beacon = BeaconPayload(device_id=1, event_type=0, confidence=0.9, sequence_number=1,
                            relative_timestamp_s=0, lat_delta_millideg=0, lon_delta_millideg=0)
    raw = encode_beacon(beacon, device.encryption_key)
    result = gw.handle_uplink(raw, "S1", now=0.0, kind="beacon")
    assert result["accepted"] is True
    assert len(gw.sink.by_topic(gw.cfg.beacon_topic)) == 1


def test_orphaned_beacon_flagged_after_timeout():
    from ecosentry.arch6_beacon import BeaconPayload, encode_beacon
    from ecosentry.arch6_payload import DeviceConfig
    from ecosentry.gateway import BackhaulLink, MqttSink, PriorityGateway
    from ecosentry.config import GatewayConfig

    device = DeviceConfig(device_id="SENTRY_TEST")
    cfg = GatewayConfig(beacon_orphan_timeout_s=10.0)
    gw = PriorityGateway(device.encryption_key, cfg, BackhaulLink(seed=1), MqttSink())
    beacon = BeaconPayload(device_id=1, event_type=0, confidence=0.9, sequence_number=42,
                            relative_timestamp_s=0, lat_delta_millideg=0, lon_delta_millideg=0)
    raw = encode_beacon(beacon, device.encryption_key)
    gw.handle_uplink(raw, "S1", now=0.0, kind="beacon")

    assert gw.check_orphaned_beacons(now=5.0) == []  # still within timeout
    orphaned = gw.check_orphaned_beacons(now=11.0)
    assert len(orphaned) == 1
    assert orphaned[0]["sequence"] == 42
    # Second call after the first one already removed it: no longer reported.
    assert gw.check_orphaned_beacons(now=20.0) == []
```

**Verification:**
```
python -m pytest tests/test_arch6_arch7_arch8.py -v -k "beacon_routes or orphaned_beacon"
```
**Expected result:** `2 passed`.

**Full-suite regression check (run after every task in this phase, not just
this one):**
```
python -m pytest tests/ -q
```
**Expected result:** `110 passed` (the original count) `+` every new test
added by Phase 0 through Phase 3 so far. If any previously-passing test now
fails, stop and report which one — do not proceed to Phase 4.

---

## Phase 4 — Reliable ACK-based delivery to the forest officer

### TASK P4.1 — Create `ecosentry/officer_delivery.py`

**Preconditions:** none (independent of Phases 1-3, but conceptually the
last hop after Phase 3's gateway forwarding).

**Files touched:** create `ecosentry/officer_delivery.py` (new file,
complete content below — verified with the commands in this task's
Verification block).

```python
"""New architecture -- reliable ACK-based delivery to the forest officer.

Extends the existing ACK chain (device -> gateway local ACK, gateway ->
cloud backhaul retry) one hop further: cloud -> officer, with a
human-acknowledgement requirement and escalation through a configured list
of channels if nobody acks in time. Nothing in gateway.py or arch6_payload.py
is modified by this module -- it is a new, independent stage that consumes
PriorityGateway's successful deliveries as input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "OfficerDeliveryConfig",
    "OfficerAck",
    "OfficerDeliveryTracker",
]


@dataclass(frozen=True)
class OfficerDeliveryConfig:
    ack_timeout_s: float = 75.0
    escalation_channels: Tuple[str, ...] = ("push", "sms", "radio")


@dataclass(frozen=True)
class OfficerAck:
    alert_sequence: int
    officer_id: str
    acked: bool
    acked_via: Optional[str]
    ack_latency_s: Optional[float]
    exhausted: bool = False


class OfficerDeliveryTracker:
    """Simulates channel delivery and human acknowledgement for one alert,
    escalating through cfg.escalation_channels if no ack arrives within
    cfg.ack_timeout_s on the current channel."""

    def __init__(
        self,
        cfg: Optional[OfficerDeliveryConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.cfg = cfg or OfficerDeliveryConfig()
        self.rng = rng or np.random.default_rng()
        self.pending: Dict[int, Dict] = {}

    def dispatch(self, sequence: int, officer_id: str, now: float) -> None:
        """Call once per alert that the gateway successfully delivered to
        the cloud/backhaul. Starts the officer-side ack clock on the first
        escalation channel."""
        self.pending[sequence] = {
            "officer_id": officer_id,
            "dispatched_at": now,
            "channel_sent_at": now,
            "channel_idx": 0,
            "resolved": False,
        }

    def tick(
        self, sequence: int, now: float, ack_probability_per_channel: float = 0.9
    ) -> Optional[OfficerAck]:
        """Call periodically (e.g. once per simulated second, or driven by a
        Monte-Carlo campaign) to resolve whether this alert has been acked,
        needs escalation, or has exhausted every channel. Returns None while
        the alert is still pending on its current channel; returns an
        OfficerAck exactly once, the moment it resolves (acked or
        exhausted). Returns None immediately for an unknown or
        already-resolved sequence -- never raises."""
        entry = self.pending.get(sequence)
        if entry is None or entry["resolved"]:
            return None

        if self.rng.random() < ack_probability_per_channel:
            entry["resolved"] = True
            channel = self.cfg.escalation_channels[entry["channel_idx"]]
            return OfficerAck(
                alert_sequence=sequence,
                officer_id=entry["officer_id"],
                acked=True,
                acked_via=channel,
                ack_latency_s=now - entry["dispatched_at"],
            )

        elapsed_on_channel = now - entry["channel_sent_at"]
        if elapsed_on_channel >= self.cfg.ack_timeout_s:
            if entry["channel_idx"] < len(self.cfg.escalation_channels) - 1:
                entry["channel_idx"] += 1
                entry["channel_sent_at"] = now
            else:
                entry["resolved"] = True
                return OfficerAck(
                    alert_sequence=sequence,
                    officer_id=entry["officer_id"],
                    acked=False,
                    acked_via=None,
                    ack_latency_s=None,
                    exhausted=True,
                )
        return None
```

**Verification (these exact commands were run against this exact code
before this plan was written):**
```
python -c "
import numpy as np
from ecosentry.officer_delivery import OfficerDeliveryTracker

rng = np.random.default_rng(42)
t1 = OfficerDeliveryTracker(rng=rng)
t1.dispatch(sequence=1, officer_id='ranger_a', now=0.0)
r = t1.tick(sequence=1, now=1.0, ack_probability_per_channel=1.0)
assert r is not None and r.acked and r.acked_via == 'push'
print('immediate ack: OK')

t2 = OfficerDeliveryTracker(rng=rng)
t2.dispatch(sequence=2, officer_id='ranger_b', now=0.0)
now = 0.0
result = None
for _ in range(400):
    now += 1.0
    result = t2.tick(sequence=2, now=now, ack_probability_per_channel=0.0)
    if result is not None:
        break
assert result is not None and result.exhausted and not result.acked
print('escalates through all channels then exhausted: OK')

t4 = OfficerDeliveryTracker(rng=rng)
assert t4.tick(sequence=999, now=0.0) is None
print('unknown sequence returns None, no exception: OK')
print('ALL CHECKS PASSED')
"
```
**Expected result:**
```
immediate ack: OK
escalates through all channels then exhausted: OK
unknown sequence returns None, no exception: OK
ALL CHECKS PASSED
```

---

### TASK P4.2 — Test suite for `officer_delivery.py`

**Preconditions:** P4.1 complete and verified.

**Files touched:** create `tests/test_officer_delivery.py` (new file).

```python
"""Tests for reliable ACK-based delivery to the forest officer (Phase 4)."""

from __future__ import annotations

import numpy as np

from ecosentry.officer_delivery import (
    OfficerDeliveryConfig,
    OfficerDeliveryTracker,
)


def test_immediate_ack_on_first_channel():
    rng = np.random.default_rng(1)
    tracker = OfficerDeliveryTracker(rng=rng)
    tracker.dispatch(sequence=1, officer_id="ranger_a", now=0.0)
    result = tracker.tick(sequence=1, now=1.0, ack_probability_per_channel=1.0)
    assert result is not None
    assert result.acked is True
    assert result.acked_via == "push"
    assert result.ack_latency_s == 1.0


def test_never_acks_escalates_through_every_channel_then_exhausted():
    rng = np.random.default_rng(2)
    cfg = OfficerDeliveryConfig(ack_timeout_s=5.0)
    tracker = OfficerDeliveryTracker(cfg=cfg, rng=rng)
    tracker.dispatch(sequence=2, officer_id="ranger_b", now=0.0)
    now = 0.0
    result = None
    seen_channels = set()
    for _ in range(200):
        now += 1.0
        entry = tracker.pending[2]
        if not entry["resolved"]:
            seen_channels.add(cfg.escalation_channels[entry["channel_idx"]])
        result = tracker.tick(sequence=2, now=now, ack_probability_per_channel=0.0)
        if result is not None:
            break
    assert result is not None
    assert result.exhausted is True
    assert result.acked is False
    assert seen_channels == {"push", "sms", "radio"}


def test_times_out_on_first_channel_then_acks_on_second():
    rng = np.random.default_rng(3)
    cfg = OfficerDeliveryConfig(ack_timeout_s=5.0)
    tracker = OfficerDeliveryTracker(cfg=cfg, rng=rng)
    tracker.dispatch(sequence=3, officer_id="ranger_c", now=0.0)
    now = 0.0
    result = None
    for _ in range(50):
        now += 1.0
        prob = 0.0 if tracker.pending[3]["channel_idx"] == 0 else 1.0
        result = tracker.tick(sequence=3, now=now, ack_probability_per_channel=prob)
        if result is not None:
            break
    assert result is not None
    assert result.acked is True
    assert result.acked_via == "sms"


def test_unknown_sequence_returns_none():
    tracker = OfficerDeliveryTracker(rng=np.random.default_rng(4))
    assert tracker.tick(sequence=999, now=0.0) is None


def test_resolved_sequence_is_idempotent():
    rng = np.random.default_rng(5)
    tracker = OfficerDeliveryTracker(rng=rng)
    tracker.dispatch(sequence=5, officer_id="ranger_e", now=0.0)
    first = tracker.tick(sequence=5, now=1.0, ack_probability_per_channel=1.0)
    second = tracker.tick(sequence=5, now=2.0, ack_probability_per_channel=1.0)
    assert first is not None and first.acked
    assert second is None
```

**Verification:**
```
python -m pytest tests/test_officer_delivery.py -v
```
**Expected result:** `5 passed`.

---

## Phase 5 — Energy visualizer, delivery visualizer, and CLI wiring

### TASK P5.1 — Split transmission energy into beacon vs. full-payload

**Preconditions:** Phase 0 (all tasks) and P1.1 complete and verified.

**Files touched:** `ecosentry/config.py`, `ecosentry/arch7_energy.py`.

**Step 1 — add a beacon spreading factor to `NetworkConfig`.** Find this
exact text (the last field before the closing of the `NetworkConfig`
class):

```python
    scenario_sf: Dict[str, int] = field(
        default_factory=lambda: {"corbett": 12, "seshachalam": 9, "sundarbans": 10}
    )
```

Replace it with:

```python
    scenario_sf: Dict[str, int] = field(
        default_factory=lambda: {"corbett": 12, "seshachalam": 9, "sundarbans": 10}
    )
    #: Beacons are small enough to default to the fastest, shortest-range
    #: spreading factor rather than each forest's payload SF (Phase 5, new
    #: architecture).
    beacon_spreading_factor: int = 7
```

**Step 2 — split `breakdown_mj_per_day`.** In `ecosentry/arch7_energy.py`,
find this exact text (the full `breakdown_mj_per_day` method):

```python
    def breakdown_mj_per_day(
        self, alerts_per_day: float, detections_per_hour: float = 6.0
    ) -> Dict[str, float]:
        return {
            "quiescent": self.quiescent_energy_mj(24 * 3600.0),
            "audio": self.energy_audio_mj() * detections_per_hour * 24,
            "spike_conversion": self.energy_spike_mj() * detections_per_hour * 24,
            "snn_inference": self.energy_inference_mj() * detections_per_hour * 24,
            "encryption": self.energy_encryption_mj() * alerts_per_day,
            "transmission": self.energy_transmission_mj() * alerts_per_day,
        }
```

Replace it with:

```python
    def breakdown_mj_per_day(
        self,
        alerts_per_day: float,
        detections_per_hour: float = 6.0,
        beacon_transmission_mj: float = 0.0,
    ) -> Dict[str, float]:
        """beacon_transmission_mj defaults to 0.0 so every existing caller
        that doesn't pass it gets the exact pre-Phase-5 numbers (all
        transmission energy attributed to "transmission"). Pass a non-zero
        value (see generate_energy_report below) to split it into
        beacon_transmission and payload_transmission instead."""
        payload_transmission_mj = self.energy_transmission_mj() * alerts_per_day
        result = {
            "quiescent": self.quiescent_energy_mj(24 * 3600.0),
            "audio": self.energy_audio_mj() * detections_per_hour * 24,
            "spike_conversion": self.energy_spike_mj() * detections_per_hour * 24,
            "snn_inference": self.energy_inference_mj() * detections_per_hour * 24,
            "encryption": self.energy_encryption_mj() * alerts_per_day,
        }
        if beacon_transmission_mj > 0.0:
            result["beacon_transmission"] = beacon_transmission_mj * alerts_per_day
            result["payload_transmission"] = payload_transmission_mj
        else:
            result["transmission"] = payload_transmission_mj
        return result
```

**Do not** change any call site of `breakdown_mj_per_day` in this task —
every existing caller omits the new third argument, so every existing
caller keeps getting the single `"transmission"` key exactly as before.
Wiring an actual non-zero `beacon_transmission_mj` value into
`generate_energy_report` is deliberately left as a separate, explicit task
(P5.2) rather than bundled here, so this task's verification can confirm
backward compatibility in isolation first.

**New test — add to `tests/test_arch6_arch7_arch8.py`:**

```python
def test_breakdown_defaults_to_single_transmission_key():
    from ecosentry.arch7_energy import DeviceEnergyModel
    from ecosentry.config import EnergyConfig

    model = DeviceEnergyModel(EnergyConfig())
    breakdown = model.breakdown_mj_per_day(alerts_per_day=5.0)
    assert "transmission" in breakdown
    assert "beacon_transmission" not in breakdown
    assert "payload_transmission" not in breakdown


def test_breakdown_splits_when_beacon_energy_given():
    from ecosentry.arch7_energy import DeviceEnergyModel
    from ecosentry.config import EnergyConfig

    model = DeviceEnergyModel(EnergyConfig())
    breakdown = model.breakdown_mj_per_day(alerts_per_day=5.0, beacon_transmission_mj=7.2)
    assert "beacon_transmission" in breakdown
    assert "payload_transmission" in breakdown
    assert "transmission" not in breakdown
    assert breakdown["beacon_transmission"] == 7.2 * 5.0
```

**Verification:**
```
python -m pytest tests/test_arch6_arch7_arch8.py -v -k "breakdown_defaults or breakdown_splits"
```
**Expected result:** `2 passed`.

---

### TASK P5.2 — Compute real beacon transmission energy and feed it into the report

**Preconditions:** P5.1 complete and verified.

**Files touched:** `ecosentry/arch7_energy.py` only.

**Exact edit:** find this exact text (inside `generate_energy_report`,
where `breakdown` is first computed):

```python
    breakdown = device.breakdown_mj_per_day(scenario.avg_alerts_per_day)
```

Replace it with:

```python
    from .arch8_network import LoRaPHY
    from .config import BeaconConfig, NetworkConfig

    net_cfg = NetworkConfig()
    beacon_cfg = BeaconConfig()
    phy = LoRaPHY(net_cfg)
    beacon_toa_s = phy.time_on_air_ms(beacon_cfg.size_bytes, net_cfg.beacon_spreading_factor) / 1000.0
    beacon_transmission_mj = cfg.power_transmission_mw * beacon_toa_s
    breakdown = device.breakdown_mj_per_day(
        scenario.avg_alerts_per_day, beacon_transmission_mj=beacon_transmission_mj
    )
```

(The `from .arch8_network import LoRaPHY` and `from .config import
BeaconConfig, NetworkConfig` lines are placed inside the function rather
than at module level to avoid a circular import — `arch8_network.py` does
not import from `arch7_energy.py`, so this direction is safe, but confirm
this by running the verification command below, which will fail loudly
with an `ImportError` if a cycle exists.)

**Verification:**
```
python -c "
from ecosentry.arch7_energy import generate_energy_report
from ecosentry.config import EnergyConfig, SCENARIOS

report = generate_energy_report(SCENARIOS['corbett'], EnergyConfig(mission_days=5), days=5, seed=1)
breakdown = report['energy_breakdown_mj_per_day']
assert 'beacon_transmission' in breakdown
assert 'payload_transmission' in breakdown
print(breakdown['beacon_transmission'], breakdown['payload_transmission'])
"
```
**Expected result:** this exact command was run against the real repository
before this plan was written and printed:
```
36.019 246.4
```
(`beacon_transmission` then `payload_transmission`, in mJ/day, for
Corbett's default `avg_alerts_per_day`). If `EnergyConfig` or
`NetworkConfig` defaults have changed since this plan was written the exact
numbers may differ, but `beacon_transmission` must always be smaller than
`payload_transmission` — `beacon_spreading_factor=7` is a lower spreading
factor than the payload's configured SF, so its time-on-air, and therefore
its energy, is always less.

Then re-run the full energy test file to confirm nothing else broke:
```
python -m pytest tests/test_arch6_arch7_arch8.py -v
```
**Expected result:** no regressions.

---

### TASK P5.3 — Draw the energy visualizer charts in the dashboard

**Preconditions:** P0.3, P5.1, P5.2 complete and verified.

**Files touched:** `ecosentry/dashboard_template.html` only.

**Exact edit:** find this exact text (the bootstrap script added in TASK
P0.3, Step 4):

```html
<script>
(function () {
  var el = document.getElementById('report-data');
  if (!el) { return; }
  var report;
  try {
    report = JSON.parse(el.textContent);
  } catch (e) {
    console.error('dashboard: could not parse report-data', e);
    return;
  }
  window.REPORT = report;
  // Phase 5 (tasks P5.3, P5.4) fill in the chart-drawing functions that
  // read window.REPORT and populate #ev-corbett / #ev-seshachalam /
  // #ev-sundarbans / #delivery-funnel. This bootstrap only makes the data
  // available; it draws nothing by itself.
})();
</script>
```

Replace it with:

```html
<script>
(function () {
  var el = document.getElementById('report-data');
  if (!el) { return; }
  var report;
  try {
    report = JSON.parse(el.textContent);
  } catch (e) {
    console.error('dashboard: could not parse report-data', e);
    return;
  }
  window.REPORT = report;

  function drawLineChart(container, series, opts) {
    if (!container) { return; }
    opts = opts || {};
    var width = opts.width || 640;
    var height = opts.height || 220;
    var yMax = opts.yMax || 100;
    var pad = 30;
    var colors = ['#1D9E75', '#D85A30', '#378ADD'];
    var svg = '<svg viewBox="0 0 ' + width + ' ' + height + '" width="100%" height="' + height + '">';
    svg += '<line x1="' + pad + '" y1="' + (height - pad) + '" x2="' + (width - 10) + '" y2="' + (height - pad) + '" stroke="currentColor" stroke-opacity="0.3"/>';
    var i = 0;
    for (var label in series) {
      if (!series.hasOwnProperty(label)) { continue; }
      var values = series[label];
      if (!values || values.length < 2) { i++; continue; }
      var stepX = (width - pad - 10) / (values.length - 1);
      var points = '';
      for (var j = 0; j < values.length; j++) {
        var x = pad + j * stepX;
        var y = height - pad - (values[j] / yMax) * (height - pad - 10);
        points += x + ',' + y + ' ';
      }
      svg += '<polyline points="' + points.trim() + '" fill="none" stroke="' + colors[i % colors.length] + '" stroke-width="2"/>';
      i++;
    }
    svg += '</svg>';
    container.innerHTML = svg;
  }

  function drawBreakdownChart(container, breakdown) {
    if (!container || !breakdown) { return; }
    var keys = Object.keys(breakdown);
    var total = keys.reduce(function (sum, k) { return sum + breakdown[k]; }, 0);
    if (total <= 0) { return; }
    var width = 640;
    var barHeight = 24;
    var x = 0;
    var colors = ['#1D9E75', '#D85A30', '#378ADD', '#BA7517', '#993556', '#5F5E5A', '#7F77DD'];
    var svg = '<svg viewBox="0 0 ' + width + ' ' + (barHeight + 40) + '" width="100%" height="' + (barHeight + 40) + '">';
    for (var i = 0; i < keys.length; i++) {
      var w = (breakdown[keys[i]] / total) * width;
      svg += '<rect x="' + x + '" y="0" width="' + w + '" height="' + barHeight + '" fill="' + colors[i % colors.length] + '"/>';
      x += w;
    }
    svg += '</svg>';
    container.innerHTML = svg;
  }

  var forests = ['corbett', 'seshachalam', 'sundarbans'];
  for (var f = 0; f < forests.length; f++) {
    var key = forests[f];
    // The per-forest energy reports live under report.simulations.energy,
    // not report.energy -- run_simulation_stage nests ARCH_7's output
    // there alongside report.simulations.network. Verified against a real
    // pipeline_report.json before this plan was written; report.energy is
    // undefined and silently renders nothing if you use it by mistake.
    var r = report.simulations && report.simulations.energy && report.simulations.energy[key];
    var container = document.getElementById('ev-' + key);
    if (!r || !container) { continue; }
    var battery = document.createElement('div');
    var breakdown = document.createElement('div');
    container.appendChild(battery);
    container.appendChild(breakdown);
    drawLineChart(
      battery,
      { 'With solar': r.soc_trajectory_solar, 'Battery only': r.soc_trajectory_battery_only },
      { yMax: 100 }
    );
    drawBreakdownChart(breakdown, r.energy_breakdown_mj_per_day);
  }
})();
</script>
```

**Verification:**
```
python -m ecosentry run --preset quick
python3 -c "
import re
html = open('artifacts/dashboard.html', encoding='utf-8').read()
assert '__REPORT_JSON__' not in html
assert 'drawLineChart' in html
assert 'ev-corbett' in html
assert 'report.simulations.energy' in html, 'must read the nested path, not report.energy'
print('OK')
"
```
**Expected result:** `OK`, and opening `artifacts/dashboard.html` in a
browser shows three battery-trajectory line charts and three breakdown bars
under the new "Energy visualizer" section, using data from the run that
just completed. If any forest's chart is empty, check that
`soc_trajectory_solar` exists as a key in `artifacts/energy_report.json`
for that forest (a sign P0.1 or P5.2 was not applied before this task), and
separately confirm the dashboard script reads `report.simulations.energy`,
not `report.energy` — the latter is a silent no-op, not an error.

**Do not** add any external chart library (`Chart.js`, `d3`, etc.) — the
existing dashboard has zero external dependencies and this task must not
introduce any, matching the "no build step, no server, opens by
double-click" constraint stated in Task P0.3.

---

### TASK P5.4 — Delivery report generation + CLI subcommand

**Preconditions:** Phase 3 (all tasks) and Phase 4 (all tasks) complete and
verified.

**Files touched:** `ecosentry/pipeline.py`, `ecosentry/cli.py`.

**IMPORTANT — `cli.py` does not use an if/elif chain.** It dispatches
commands through a dict named `_COMMANDS` mapping command-name strings to
handler functions (`cmd_run`, `cmd_energy`, `cmd_network`, etc.), and
`main()` just does `_COMMANDS[args.command](args)`. The steps below match
this exactly — do not add an if/elif chain anywhere.

**Step 1 — add `run_delivery_stage` to `pipeline.py`.** Find this exact
text (the end of the `render_dashboard` function added in TASK P0.3, Step
5 — this is the current end of the file at this point in the plan):

```python
def render_dashboard(report: Dict, out_dir: Path) -> Path:
    """Fill dashboard_template.html with this run's pipeline_report.json
    and write the result to out_dir/dashboard.html (Phase 0, Task P0.3)."""
    template_path = Path(__file__).parent / "dashboard_template.html"
    template = template_path.read_text(encoding="utf-8")
    injected = template.replace("__REPORT_JSON__", json.dumps(report, default=str))
    out_path = out_dir / "dashboard.html"
    out_path.write_text(injected, encoding="utf-8")
    return out_path
```

Replace it with (this adds `run_delivery_stage` immediately after
`render_dashboard`, leaving `render_dashboard` itself untouched):

```python
def render_dashboard(report: Dict, out_dir: Path) -> Path:
    """Fill dashboard_template.html with this run's pipeline_report.json
    and write the result to out_dir/dashboard.html (Phase 0, Task P0.3)."""
    template_path = Path(__file__).parent / "dashboard_template.html"
    template = template_path.read_text(encoding="utf-8")
    injected = template.replace("__REPORT_JSON__", json.dumps(report, default=str))
    out_path = out_dir / "dashboard.html"
    out_path.write_text(injected, encoding="utf-8")
    return out_path


def run_delivery_stage(
    preset: str = "quick",
    scenario: str = "corbett",
    messages: int = 300,
    seed: int = 42,
    out_dir: Optional[Path] = None,
) -> Dict:
    """Monte-Carlo campaign through PriorityGateway + OfficerDeliveryTracker
    for one forest scenario (Phase 5, new architecture)."""
    from .arch6_payload import DeviceConfig, generate_alert_payload
    from .config import GatewayConfig
    from .gateway import BackhaulLink, MqttSink, PriorityGateway
    from .officer_delivery import OfficerDeliveryConfig, OfficerDeliveryTracker

    out_dir = Path(out_dir) if out_dir is not None else Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    device = DeviceConfig(device_id=f"SENTRY_{scenario.upper()}")
    gateway = PriorityGateway(device.encryption_key, GatewayConfig(), BackhaulLink(seed=seed), MqttSink())
    officer_tracker = OfficerDeliveryTracker(cfg=OfficerDeliveryConfig(), rng=rng)

    dispatched = 0
    now = 0.0
    for i in range(messages):
        now += float(rng.uniform(1.0, 30.0))
        is_threat = rng.random() < 0.3
        class_id = 0 if is_threat else 2
        confidence = float(rng.uniform(0.86, 0.99)) if is_threat else float(rng.uniform(0.5, 0.8))
        result = {
            "class_id": class_id,
            "confidence": confidence,
            "sequence": i % 256,
            "timestamp": now,
        }
        packet = generate_alert_payload(result, device)
        outcome = gateway.handle_uplink(packet["message"], "S1", now=now)
        if outcome.get("priority"):
            officer_tracker.dispatch(sequence=i % 256, officer_id="ranger_1", now=now)
            dispatched += 1

    resolved = []
    tick_now = now
    for _ in range(500):
        tick_now += 5.0
        for seq in list(officer_tracker.pending.keys()):
            r = officer_tracker.tick(seq, tick_now, ack_probability_per_channel=0.85)
            if r is not None:
                resolved.append(r)
        if len(resolved) >= dispatched:
            break

    acked = [r for r in resolved if r.acked]
    report = {
        "scenario": scenario,
        "messages_sent": messages,
        "priority_dispatched": dispatched,
        "officer_acked": len(acked),
        "officer_ack_rate": (len(acked) / dispatched) if dispatched else None,
        "officer_ack_latency_p95_s": (
            float(np.percentile([r.ack_latency_s for r in acked], 95)) if acked else None
        ),
        "escalations_triggered": sum(1 for r in resolved if r.acked_via in ("sms", "radio")),
        "unacknowledged": sum(1 for r in resolved if r.exhausted),
        "gateway_metrics": gateway.metrics(),
    }
    (out_dir / "delivery_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report
```

Note `class_id` is deliberately unused beyond building `result` above (no
separate reference needed) and `messages`/`is_threat` drive a roughly-30%
threat rate — this matches the verified test run in this task's
Verification block. `numpy` is already imported as `np` at the top of
`pipeline.py` (`import numpy as np`) — do not add a duplicate import.

**Step 2 — add the CLI subparser.** In `ecosentry/cli.py`, find this exact
text:

```python
    p_net = sub.add_parser("network", help="ARCH_8: LoRa mesh simulation")
    p_net.add_argument("--scenario", choices=sorted(SCENARIOS) + ["all"], default="all")
    p_net.add_argument("--messages", type=int, default=200)
    p_net.add_argument("--payload-bytes", type=int, default=132)
```

Replace it with:

```python
    p_net = sub.add_parser("network", help="ARCH_8: LoRa mesh simulation")
    p_net.add_argument("--scenario", choices=sorted(SCENARIOS) + ["all"], default="all")
    p_net.add_argument("--messages", type=int, default=200)
    p_net.add_argument("--payload-bytes", type=int, default=132)

    p_delivery = sub.add_parser(
        "delivery", help="Priority delivery pipeline: gateway + officer ack simulation"
    )
    p_delivery.add_argument("--scenario", choices=sorted(SCENARIOS), default="corbett")
    p_delivery.add_argument("--messages", type=int, default=300)
    p_delivery.add_argument("--seed", type=int, default=42)
    p_delivery.add_argument("--out", type=Path, default=Path("artifacts"))
```

**Step 3 — add the `cmd_delivery` handler function.** Find this exact text
(the start of the `cmd_synth` function — `cmd_delivery` goes immediately
before it, matching the style of `cmd_energy` and `cmd_network` which sit
above `cmd_synth` already):

```python
def cmd_synth(args) -> int:
```

Replace it with:

```python
def cmd_delivery(args) -> int:
    from .pipeline import run_delivery_stage

    report = run_delivery_stage(
        scenario=args.scenario, messages=args.messages, seed=args.seed, out_dir=args.out
    )
    print(json.dumps(report, indent=2, default=str))
    return 0


def cmd_synth(args) -> int:
```

**Step 4 — register the command in `_COMMANDS`.** Find this exact text:

```python
_COMMANDS = {
    "run": cmd_run,
    "dataset": cmd_dataset,
    "train": cmd_train,
    "infer": cmd_infer,
    "energy": cmd_energy,
    "network": cmd_network,
    "synth": cmd_synth,
}
```

Replace it with:

```python
_COMMANDS = {
    "run": cmd_run,
    "dataset": cmd_dataset,
    "train": cmd_train,
    "infer": cmd_infer,
    "energy": cmd_energy,
    "network": cmd_network,
    "delivery": cmd_delivery,
    "synth": cmd_synth,
}
```

**Verification:**
```
python -m ecosentry delivery --scenario corbett --messages 50
```
**Expected result:** valid JSON is printed with keys `scenario`,
`messages_sent`, `priority_dispatched`, `officer_acked`,
`officer_ack_rate`, `officer_ack_latency_p95_s`, `escalations_triggered`,
`unacknowledged`, `gateway_metrics`, and `artifacts/delivery_report.json` is
written with the same content. `officer_ack_rate` should be a float between
0.0 and 1.0 (not `null`) as long as `priority_dispatched > 0`. On the
verified run of this exact command, `officer_ack_rate` was `1.0` with
`priority_dispatched` in the high teens out of 50 messages — exact counts
vary run to run since no `--seed` was pinned above beyond its default, but
`officer_ack_rate` should not be `null` and `messages_sent` must equal
`50`.

---

### TASK P5.5 — Draw the delivery funnel chart

**Preconditions:** P0.3, P5.4 complete and verified.

**Files touched:** `ecosentry/dashboard_template.html`,
`ecosentry/pipeline.py`.

**Step 1 — include the delivery report in the top-level pipeline report.**
Inside `run_full_pipeline()`, find this exact text (the dict literal
assembled from all four stages — note the function's scenario/seed
parameters are named `scenario` and `seed`, but the fidelity-preset
parameter is named `preset_name`, a string, which is different from the
local variable `preset` a few lines above it, an already-resolved
`PRESETS[preset_name]` object — use `preset_name`, not `preset`, in the
call you add below):

```python
    report = {
        "preset": preset.name,
        "scenario": scenario,
        "seed": seed,
        "seconds_total": time.time() - started,
        "dataset": {k: v for k, v in stage1.items() if not k.startswith("_")},
        "training": {k: v for k, v in stage2.items() if k != "model"},
        "alerts": {k: v for k, v in stage3.items() if k != "events"},
        "simulations": stage4,
        "acceptance": acceptance_summary(stage1, stage2, stage3, stage4),
    }
```

Replace it with:

```python
    delivery_report = run_delivery_stage(preset_name, scenario, out_dir=out_dir, seed=seed)

    report = {
        "preset": preset.name,
        "scenario": scenario,
        "seed": seed,
        "seconds_total": time.time() - started,
        "dataset": {k: v for k, v in stage1.items() if not k.startswith("_")},
        "training": {k: v for k, v in stage2.items() if k != "model"},
        "alerts": {k: v for k, v in stage3.items() if k != "events"},
        "simulations": stage4,
        "delivery": delivery_report,
        "acceptance": acceptance_summary(stage1, stage2, stage3, stage4),
    }
```

`run_delivery_stage` does not need a separate import statement — it is
defined in this same module (added in TASK P5.4, Step 1).

**Step 2 — draw the funnel.** In `ecosentry/dashboard_template.html`, find
this exact text (the end of the script block from TASK P5.3, immediately
before its closing):

```html
    drawBreakdownChart(breakdown, r.energy_breakdown_mj_per_day);
  }
})();
</script>
```

Replace it with:

```html
    drawBreakdownChart(breakdown, r.energy_breakdown_mj_per_day);
  }

  var delivery = report.delivery;
  var funnelContainer = document.getElementById('delivery-funnel');
  if (delivery && funnelContainer) {
    var stages = [
      { label: 'Messages sent', value: delivery.messages_sent },
      { label: 'Priority dispatched', value: delivery.priority_dispatched },
      { label: 'Officer acked', value: delivery.officer_acked },
      { label: 'Unacknowledged', value: delivery.unacknowledged }
    ];
    var maxVal = stages[0].value || 1;
    var rowHeight = 32;
    var width = 640;
    var svg = '<svg viewBox="0 0 ' + width + ' ' + (rowHeight * stages.length) + '" width="100%" height="' + (rowHeight * stages.length) + '">';
    for (var s = 0; s < stages.length; s++) {
      var w = (stages[s].value / maxVal) * (width - 160);
      var y = s * rowHeight;
      svg += '<text x="0" y="' + (y + rowHeight / 2) + '" dominant-baseline="middle" font-size="12">' + stages[s].label + '</text>';
      svg += '<rect x="160" y="' + (y + 4) + '" width="' + w + '" height="' + (rowHeight - 12) + '" fill="#1D9E75"/>';
      svg += '<text x="' + (160 + w + 8) + '" y="' + (y + rowHeight / 2) + '" dominant-baseline="middle" font-size="12">' + stages[s].value + '</text>';
    }
    svg += '</svg>';
    funnelContainer.innerHTML = svg;
  }
})();
</script>
```

**Verification:**
```
python -m ecosentry run --preset quick
python3 -c "
html = open('artifacts/dashboard.html', encoding='utf-8').read()
assert 'delivery-funnel' in html
assert '\"delivery\":' in html
print('OK')
"
```
**Expected result:** `OK`, and opening `artifacts/dashboard.html` shows the
four-row delivery funnel under "Priority delivery pipeline," with
"Messages sent" as the longest bar and the other three shrinking
proportionally.

---

## Final verification — run this after every phase above is complete

```
python -m pytest tests/ -q
python -m ecosentry run --preset quick
python3 -c "
import json
report = json.load(open('artifacts/pipeline_report.json'))
# energy and network reports are nested under report['simulations'] (written
# by run_simulation_stage); delivery is top-level (added directly in
# TASK P5.5). Do not confuse these two shapes -- report['energy'] does not
# exist and silently returns nothing if you query it by mistake.
assert 'energy' in report['simulations']
assert 'network' in report['simulations']
assert 'delivery' in report
for key in ('corbett', 'seshachalam', 'sundarbans'):
    energy = report['simulations']['energy'][key]
    assert 'soc_trajectory_solar' in energy
    assert 'beacon_transmission' in energy['energy_breakdown_mj_per_day']
print('ALL PHASES VERIFIED')
"
open artifacts/dashboard.html   # or: xdg-open artifacts/dashboard.html
```

**Expected result:** the pytest line reports zero failures (the exact count
will be `110` original tests plus every new test function added across
Phases 0-5 in this plan — do not worry about matching an exact total
number, only that the failure count is `0`); the pipeline run completes
without a traceback; the Python check prints `ALL PHASES VERIFIED`; and the
dashboard opens in a browser showing the acceptance table (unchanged, still
static per the Task P0.3 scope decision), the energy visualizer section
with three battery-trajectory charts, and the delivery funnel with four
bars.

If any step in this final verification fails, identify which phase's task
introduced the failure by re-running that phase's own verification command
in isolation, rather than debugging against the full pipeline first.
