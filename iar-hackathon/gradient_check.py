"""Validate JANUS's prototype proxy against an expensive gradient proxy."""
from __future__ import annotations
import numpy as np


def gradient_interference_proxy(old_embeddings: np.ndarray, new_embeddings: np.ndarray) -> float:
    """Cosine of mean feature gradients for a linear head at initialization.

    This controlled proxy is intentionally small and reproducible; it is not
    presented as a substitute for full per-example gradient analysis.
    """
    a, b = old_embeddings.mean(0), new_embeddings.mean(0)
    return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))


def correlation(prototype_scores: np.ndarray, gradient_scores: np.ndarray) -> float:
    if len(prototype_scores) < 2 or np.std(prototype_scores) == 0 or np.std(gradient_scores) == 0:
        return float("nan")
    return float(np.corrcoef(prototype_scores, gradient_scores)[0, 1])
