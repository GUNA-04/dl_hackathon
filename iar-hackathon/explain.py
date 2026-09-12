"""Human-readable explanations for JANUS memory decisions."""
from __future__ import annotations


def explain_class(class_id: int, risk: float, quota: int, forgetting: float | None = None) -> str:
    outcome = (f" Its measured validation forgetting was {forgetting:.1%}."
               if forgetting is not None else "")
    return (f"Class {class_id} received {quota} replay slots because its predicted "
            f"interference risk was {risk:.3f}; similar incoming classes are likely "
            f"to overwrite it.{outcome}")
