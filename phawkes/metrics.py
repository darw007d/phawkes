"""Stability and systemic-risk metrics for Hawkes adjacency matrices."""

from __future__ import annotations

import numpy as np


def branching_ratio(adjacency: np.ndarray) -> float:
    """Spectral radius of the branching-coefficient matrix.

    For a multivariate Hawkes process with exponential kernels and shared
    decay, the branching ratio equals max(|eig(alpha)|). The process is
    stationary (event count per unit time is finite) iff this is < 1.

    Values close to 1 indicate a near-critical system where a single shock
    can trigger a long avalanche of downstream events — the financial-
    contagion analogue of a near-critical epidemic.
    """
    a = np.asarray(adjacency, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"adjacency must be square 2D, got shape {a.shape}")
    return float(np.max(np.abs(np.linalg.eigvals(a))))
