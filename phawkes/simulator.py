"""Ogata thinning simulator for multivariate exponential-kernel Hawkes processes.

Intensity for dim i at time t:
    lambda_i(t) = mu_i + sum_j sum_{t_j^k < t} alpha_ij * beta * exp(-beta * (t - t_j^k))

Parameters mu (baseline, d,), alpha (branching, d x d, non-negative),
and beta (decay, scalar or d x d). Stability requires max eigenvalue of
alpha (with shared beta) or of the integrated kernel matrix to be < 1.

Reference: Ogata (1981), "On Lewis' simulation method for point processes",
IEEE Trans. Inform. Theory 27(1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HawkesExpSim:
    """Simulator for a multivariate Hawkes process with exponential kernels.

    Args:
        baseline: (d,) array of background intensities mu_i >= 0.
        adjacency: (d, d) matrix of branching coefficients alpha_ij >= 0.
        decay: scalar beta > 0 (shared across all pairs for v0.1).
        seed: PRNG seed.
    """

    baseline: np.ndarray
    adjacency: np.ndarray
    decay: float
    seed: int | None = None

    def __post_init__(self) -> None:
        self.baseline = np.asarray(self.baseline, dtype=np.float64)
        self.adjacency = np.asarray(self.adjacency, dtype=np.float64)
        if self.baseline.ndim != 1:
            raise ValueError(f"baseline must be 1D, got shape {self.baseline.shape}")
        d = self.baseline.shape[0]
        if self.adjacency.shape != (d, d):
            raise ValueError(
                f"adjacency must be ({d}, {d}) to match baseline, got {self.adjacency.shape}"
            )
        if np.any(self.baseline < 0):
            raise ValueError("baseline must be non-negative")
        if np.any(self.adjacency < 0):
            raise ValueError("adjacency must be non-negative")
        if self.decay <= 0:
            raise ValueError(f"decay must be positive, got {self.decay}")
        self._d = d
        self._rng = np.random.default_rng(self.seed)

    @property
    def dim(self) -> int:
        return self._d

    def branching_ratio(self) -> float:
        """Spectral radius of the adjacency (stability iff < 1)."""
        return float(np.max(np.abs(np.linalg.eigvals(self.adjacency))))

    def simulate(self, end_time: float) -> list[np.ndarray]:
        """Simulate the process on [0, end_time].

        Returns a list of length d; entry i is the sorted array of event
        times for dim i.

        Uses Ogata's thinning: at each step compute a global upper bound on
        the total intensity, draw a candidate time from an exponential, then
        accept into dim i with probability lambda_i / upper_bound.
        """
        if end_time <= 0:
            raise ValueError(f"end_time must be positive, got {end_time}")
        if self.branching_ratio() >= 1.0:
            # not fatal — the simulator may still terminate if end_time is
            # small — but warn the caller that intensities may blow up
            import warnings

            warnings.warn(
                f"branching ratio {self.branching_ratio():.3f} >= 1: "
                "process is non-stationary, event counts may explode",
                RuntimeWarning,
                stacklevel=2,
            )

        d = self._d
        beta = self.decay
        mu = self.baseline
        a = self.adjacency

        # per-dim running kernel sums r_i = sum over past events of
        # beta * exp(-beta * (t - t_k)); then lambda_i(t) = mu_i + sum_j a_ij * r_j(t)
        # We maintain r per source dim (size d) and decay it between events.
        r = np.zeros(d, dtype=np.float64)
        t = 0.0
        out: list[list[float]] = [[] for _ in range(d)]

        while True:
            # lambda_i(t) right after decaying r to current time
            intensities = mu + a @ r
            total = float(intensities.sum())
            if total <= 0.0:
                # only possible if mu == 0 and r == 0 — then no events ever occur
                break

            # draw next candidate time from homogeneous Poisson with rate `total`
            dt = self._rng.exponential(scale=1.0 / total)
            t_candidate = t + dt
            if t_candidate > end_time:
                break

            # decay r to candidate time
            r *= np.exp(-beta * dt)
            # recompute intensities at the candidate time (after decay; r has shrunk)
            intensities_cand = mu + a @ r
            total_cand = float(intensities_cand.sum())

            # accept with prob total_cand / total (thinning)
            u = self._rng.uniform()
            if u * total <= total_cand:
                # decide which dim fired
                probs = intensities_cand / total_cand
                which = int(self._rng.choice(d, p=probs))
                out[which].append(t_candidate)
                # self-excitation: add an impulse of size beta to r[which]
                # (the integrated kernel alpha_ij * beta * exp(-beta * dt)
                # contributes `beta` to r at the source dim's index)
                r[which] += beta
            t = t_candidate

        return [np.array(ts, dtype=np.float64) for ts in out]
