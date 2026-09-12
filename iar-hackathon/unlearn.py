"""Targeted JANUS unlearning utilities.

This is deliberately a diagnostic intervention, not a claim of certified
machine unlearning: it removes a requested class from replay and reports the
before/after retention trade-off for the proof gate.
"""
from __future__ import annotations
import numpy as np


def remove_target_from_buffer(buffer, target_class: int) -> None:
    """Honor a deletion request by removing all target exemplars from memory."""
    buffer.samples = buffer.samples[buffer.labels[buffer.samples] != target_class]


def unlearning_report(target_before: float, target_after: float,
                     retained_before: float, retained_after: float) -> dict:
    return {"target_before": target_before, "target_after": target_after,
            "retained_before": retained_before, "retained_after": retained_after,
            "target_drop": target_before - target_after,
            "retained_drop": retained_before - retained_after,
            "proof_gate_passed": bool(target_after <= 0.03 and retained_before - retained_after <= 0.02)}
