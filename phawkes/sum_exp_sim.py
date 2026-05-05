"""Ogata thinning simulator for sum-of-exponential-kernel multivariate Hawkes.

Generalises HawkesExpSim from K=1 (single decay) to K kernel components.
Intensity for dim i at time t:

    lambda_i(t) = mu_i + sum_j sum_{k=1..K} alpha_ij_k * beta_k * sum_{t_j^m < t} exp(-beta_k * (t - t_j^m))

where alpha is shape (K, d, d) and decays is shape (K,).

Used primarily by tests/recovery experiments to generate synthetic
ground-truth data for HawkesSumExpMLE recovery verification. Parallels
HawkesExpSim conventions:
- Maintains r[k, j] = beta_k * sum_{m: t_j^m < t} exp(-beta_k * (t - t_j^m))
- Between events: r[k, j] *= exp(-beta_k * dt)
- Event on source dim j adds beta_k to r[k, j] for all k

Stability requires spectral radius of integrated kernel matrix
sum_k alpha[k] < 1 (sum of integrated kernel mass per pair).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class HawkesSumExpSim:
    """Simulator for a multivariate Hawkes process with sum-of-exp kernels.

    Args:
        baseline: (d,) array of background intensities mu_i >= 0.
        adjacency: (K, d, d) array of per-component branching coefficients
            alpha_ij_k >= 0. Sum-of-kernel-mass per pair is sum_k alpha_ij_k.
        decays: (K,) array of decay rates beta_k > 0.
        seed: PRNG seed.
    """

    baseline: np.ndarray
    adjacency: np.ndarray
    decays: Sequence[float]
    seed: int | None = None

    def __post_init__(self) -> None:
        self.baseline = np.asarray(self.baseline, dtype=np.float64)
        self.adjacency = np.asarray(self.adjacency, dtype=np.float64)
        decays = np.asarray(self.decays, dtype=np.float64)

        if self.baseline.ndim != 1:
            raise ValueError(f"baseline must be 1D, got shape {self.baseline.shape}")
        d = self.baseline.shape[0]
        if decays.ndim != 1 or decays.size < 1:
            raise ValueError(f"decays must be 1D non-empty, got shape {decays.shape}")
        K = decays.size
        if self.adjacency.shape != (K, d, d):
            raise ValueError(
                f"adjacency must be ({K}, {d}, {d}) to match decays + baseline, "
                f"got {self.adjacency.shape}"
            )
        if np.any(self.baseline < 0):
            raise ValueError("baseline must be non-negative")
        if np.any(self.adjacency < 0):
            raise ValueError("adjacency must be non-negative")
        if np.any(decays <= 0):
            raise ValueError("all decays must be positive")
        self.decays = decays  # type: ignore[assignment]
        self._d = d
        self._K = K
        self._rng = np.random.default_rng(self.seed)

    @property
    def dim(self) -> int:
        return self._d

    @property
    def n_kernels(self) -> int:
        return self._K

    def branching_ratio(self) -> float:
        """Spectral radius of integrated kernel matrix sum_k adjacency[k] (stable iff < 1)."""
        integrated = self.adjacency.sum(axis=0)  # (d, d)
        return float(np.max(np.abs(np.linalg.eigvals(integrated))))

    def simulate(self, end_time: float) -> list[np.ndarray]:
        """Simulate on [0, end_time]. Returns list of length d of sorted event-time arrays.

        Same Ogata thinning structure as HawkesExpSim, with K running kernel
        sums per source dim instead of 1.
        """
        if end_time <= 0:
            raise ValueError(f"end_time must be positive, got {end_time}")
        if self.branching_ratio() >= 1.0:
            import warnings

            warnings.warn(
                f"branching ratio {self.branching_ratio():.3f} >= 1: "
                "process is non-stationary, event counts may explode",
                RuntimeWarning,
                stacklevel=2,
            )

        d = self._d
        K = self._K
        decays = np.asarray(self.decays, dtype=np.float64)
        mu = self.baseline
        a = self.adjacency  # (K, d, d)

        # r[k, j] = beta_k * sum exp(-beta_k * (t - t_j^m)) over past events on source j
        # lambda_i(t) = mu_i + sum_k sum_j a[k, i, j] * r[k, j]
        r = np.zeros((K, d), dtype=np.float64)
        t = 0.0
        out: list[list[float]] = [[] for _ in range(d)]

        while True:
            # intensities[i] = mu[i] + sum_k sum_j a[k, i, j] * r[k, j]
            # which is mu + sum_k (a[k] @ r[k]). Vectorise:
            #   contrib = a * r[:, None, :]  shape (K, d, d) — but we want sum over k and j.
            # Simpler: einsum.
            intensities = mu + np.einsum("kij,kj->i", a, r)
            total = float(intensities.sum())
            if total <= 0.0:
                break

            dt = self._rng.exponential(scale=1.0 / total)
            t_candidate = t + dt
            if t_candidate > end_time:
                break

            # decay r to candidate time (each component at its own rate)
            r *= np.exp(-decays[:, None] * dt)
            intensities_cand = mu + np.einsum("kij,kj->i", a, r)
            total_cand = float(intensities_cand.sum())

            u = self._rng.uniform()
            if u * total <= total_cand:
                probs = intensities_cand / total_cand
                which = int(self._rng.choice(d, p=probs))
                out[which].append(t_candidate)
                # self-excitation: event on source `which` adds beta_k to r[k, which] for all k
                # (the integrated kernel alpha_ij_k * beta_k * exp(...) at impulse moment
                # contributes beta_k to the running sum at source index `which`)
                r[:, which] += decays
            t = t_candidate

        return [np.array(ts, dtype=np.float64) for ts in out]
