"""Eco-Sentry -- neuromorphic anti-poaching acoustic detection.

Reference implementation of the eight architectures described in the design
documents at the repository root:

===========  ==========================  ===============================
Document     Module                      Responsibility
===========  ==========================  ===============================
ARCH_1       :mod:`~ecosentry.arch1_audio`     audio -> mel-spectrogram
ARCH_2       :mod:`~ecosentry.arch2_spikes`    mel -> LIF spike trains
ARCH_3       :mod:`~ecosentry.arch3_dataset`   normalise / split / augment
ARCH_4       :mod:`~ecosentry.arch4_training`  surrogate-gradient SNN training
ARCH_5       :mod:`~ecosentry.arch5_inference` edge inference + alerting
ARCH_6       :mod:`~ecosentry.arch6_payload`   JSON -> zlib -> AES-256 -> wire
ARCH_7       :mod:`~ecosentry.arch7_energy`    battery + solar profiler
ARCH_8       :mod:`~ecosentry.arch8_network`   LoRa mesh simulator
PRIORITY     :mod:`~ecosentry.gateway`         gateway priority proxy
===========  ==========================  ===============================

Quick start::

    from ecosentry.pipeline import run_full_pipeline
    report = run_full_pipeline("quick")
"""

from __future__ import annotations

__version__ = "1.0.0"

from .config import (  # noqa: F401
    CLASS_MAP,
    CLASS_NAMES,
    DEFAULT,
    SCENARIOS,
    THREAT_CLASSES,
    EcoSentryConfig,
)

__all__ = [
    "__version__",
    "CLASS_NAMES",
    "CLASS_MAP",
    "THREAT_CLASSES",
    "SCENARIOS",
    "DEFAULT",
    "EcoSentryConfig",
]
