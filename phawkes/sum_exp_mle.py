"""Maximum-likelihood estimation for sum-of-exponential-kernel multivariate Hawkes.

A K-component sum-of-exp kernel between source dim j and target dim i is:

    phi_ij(t) = sum_{k=1..K} alpha_ij_k * beta_k * exp(-beta_k * t)

Each kernel component k has its own decay beta_k (shared across all (i,j)
pairs in v0.2; per-pair decays would inflate parameter count). The weights
alpha_ij_k are per-pair-per-component, giving K * d * d adjacency parameters
plus d baselines.

This generalizes v0.1's HawkesExpMLE (K=1, single beta) to multi-scale
temporal patterns: fast self-excitation + slow cross-excitation in one
fit, for example. Set decays = [b1, b2] and the optimiser apportions the
kernel mass between fast (b1) and slow (b2) components per pair.

The Ogata recursion extends naturally: maintain K running kernel sums
R_jk(t) per source dim, each decaying at its own rate beta_k. Intensity
at time t for dim i is:

    lambda_i(t) = mu_i + sum_j sum_k alpha_ij_k * beta_k * R_jk(t)

Between events, R_jk(t+dt) = R_jk(t) * exp(-beta_k * dt). When an event
fires on source j at time s, R_jk increments by 1 for ALL k (the unit
impulse adds to every component's running sum).

Reference parallels v0.1's exp_mle.py — same Laub-Taimre-Pollett framework,
just with K kernel components instead of 1.

For K=1, this reduces exactly to HawkesExpMLE — verified by recovery test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import minimize


@dataclass
class HawkesSumExpMLE:
    """Sum-of-exponential-kernel multivariate Hawkes MLE.

    Args:
        decays: array of K positive decay rates beta_k > 0. Sorted ascending
            is recommended (slow→fast) for interpretability but not required.
        penalty_l2: ridge on log_alpha (not on log_mu). 0.0 disables. Same
            semantic as v0.1 HawkesExpMLE.
        max_iter: L-BFGS-B iterations.
        tol: L-BFGS-B gradient tolerance.
        verbose: print scipy's optimiser log.
    """

    decays: Sequence[float]
    penalty_l2: float = 0.0
    max_iter: int = 500
    tol: float = 1e-6
    verbose: bool = False

    # fitted state
    baseline_: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    adjacency_: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    # adjacency_ shape: (K, d, d) — adjacency_[k, i, j] = alpha_ij_k
    n_dim_: int = 0
    n_kernels_: int = 0
    end_time_: float = 0.0
    converged_: bool = False
    n_iter_: int = 0
    final_loglik_: float = float("nan")

    def __post_init__(self) -> None:
        decays = np.asarray(self.decays, dtype=np.float64)
        if decays.ndim != 1 or decays.size < 1:
            raise ValueError(f"decays must be 1D non-empty, got shape {decays.shape}")
        if np.any(decays <= 0):
            raise ValueError(f"all decays must be positive, got {decays}")
        self.decays = decays  # type: ignore[assignment]

    def fit(self, events: Sequence[np.ndarray], end_time: float) -> "HawkesSumExpMLE":
        """Fit to a single realization.

        Args:
            events: length-d list where events[i] is the sorted 1D array of
                event times for dim i.
            end_time: observation window length T.

        Returns self with .baseline_ (shape d) and .adjacency_ (shape K, d, d).
        """
        if end_time <= 0:
            raise ValueError(f"end_time must be positive, got {end_time}")

        events = [np.asarray(ts, dtype=np.float64) for ts in events]
        d = len(events)
        if d < 1:
            raise ValueError("need at least 1 dim of events")
        for i, ts in enumerate(events):
            if ts.ndim != 1:
                raise ValueError(f"events[{i}] must be 1D, got shape {ts.shape}")
            if ts.size and (ts[0] < 0 or ts[-1] > end_time):
                raise ValueError(f"events[{i}] has times outside [0, {end_time}]")
            if ts.size > 1 and np.any(np.diff(ts) < 0):
                raise ValueError(f"events[{i}] must be sorted ascending")

        decays = np.asarray(self.decays, dtype=np.float64)
        K = decays.size
        self.n_dim_ = d
        self.n_kernels_ = K
        self.end_time_ = float(end_time)

        # initial guess: mu = N_i / T (empirical rate),
        # alpha_ij_k = 0.05 / (K * d) uniform across components + diagonal nudge
        counts = np.array([ts.size for ts in events], dtype=np.float64)
        mu0 = np.clip(counts / end_time, 1e-6, None)
        alpha0 = np.full((K, d, d), 0.05 / max(K * d, 1))
        for k in range(K):
            np.fill_diagonal(alpha0[k], 0.05)
        alpha0 = np.clip(alpha0, 1e-6, None)

        x0 = np.concatenate([np.log(mu0), np.log(alpha0).ravel()])

        def neg_loglik_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
            log_mu = x[:d]
            log_alpha = x[d:].reshape(K, d, d)
            mu = np.exp(log_mu)
            alpha = np.exp(log_alpha)
            ll, dmu, dalpha = _loglik_and_grad_sum_exp(events, end_time, mu, alpha, decays)
            # chain rule: d(ll)/d(log_mu) = mu * d(ll)/d(mu); same for alpha
            g_log_mu = mu * dmu
            g_log_alpha = alpha * dalpha
            if self.penalty_l2 > 0:
                # L2 on log_alpha (not log_mu)
                ll = ll - 0.5 * self.penalty_l2 * float(np.sum(log_alpha * log_alpha))
                g_log_alpha = g_log_alpha - self.penalty_l2 * log_alpha
            grad = -np.concatenate([g_log_mu, g_log_alpha.ravel()])
            return -ll, grad

        options = {"maxiter": self.max_iter, "gtol": self.tol}
        if self.verbose:
            options["disp"] = True
        result = minimize(
            neg_loglik_and_grad,
            x0,
            jac=True,
            method="L-BFGS-B",
            options=options,
        )

        log_mu = result.x[:d]
        log_alpha = result.x[d:].reshape(K, d, d)
        self.baseline_ = np.exp(log_mu)
        self.adjacency_ = np.exp(log_alpha)
        self.converged_ = bool(result.success)
        self.n_iter_ = int(result.nit)
        self.final_loglik_ = float(-result.fun)
        return self

    def branching_ratio(self) -> float:
        """Spectral radius of the integrated kernel matrix sum_k adjacency_[k].

        For exp kernels with shared decay, integrated kernel mass per pair is
        alpha_ij. With sum-of-exp it is sum_k alpha_ij_k. Stability requires
        spectral radius of that sum < 1 (Hawkes self-exciting stability).
        """
        if self.adjacency_ is None:
            raise RuntimeError("fit() must be called before branching_ratio()")
        integrated = self.adjacency_.sum(axis=0)  # (d, d)
        return float(np.max(np.abs(np.linalg.eigvals(integrated))))


def _loglik_and_grad_sum_exp(
    events: list[np.ndarray],
    end_time: float,
    mu: np.ndarray,
    alpha: np.ndarray,  # shape (K, d, d)
    decays: np.ndarray,  # shape (K,)
) -> tuple[float, np.ndarray, np.ndarray]:
    """Log-likelihood and gradients for sum-of-exp Hawkes.

    Returns (ll, dmu, dalpha) where dalpha has same shape (K, d, d) as alpha.
    """
    K = decays.size
    d = mu.shape[0]

    # ---- integral part: -sum_i int_0^T lambda_i(t) dt
    # Per-component-per-source integrated kernel sum:
    # S_jk = sum_m (1 - exp(-beta_k * (T - t_j^m)))
    S = np.zeros((K, d), dtype=np.float64)
    for j in range(d):
        ts = events[j]
        if ts.size:
            for k in range(K):
                S[k, j] = float(np.sum(1.0 - np.exp(-decays[k] * (end_time - ts))))
    # integral over dim i = mu_i * T + sum_j sum_k alpha[k, i, j] * S[k, j]
    # gradients:
    # d(integral_i)/d(mu_i) = T
    # d(integral_i)/d(alpha[k, i, j]) = S[k, j]
    dmu_from_integral = np.full(d, end_time)
    # dalpha_from_integral[k, i, j] = S[k, j] — broadcast over i
    dalpha_from_integral = np.broadcast_to(S[:, None, :], (K, d, d)).copy()

    # ---- log-intensity part: sum_i sum_k log(lambda_i(t_i^n))
    # Maintain R[k, j] running kernel sums per (component, source dim).
    # Between events, R[k, j] *= exp(-beta_k * dt).
    # When event fires on source dim j at time s, R[:, j] += 1 (unit impulse
    # adds to ALL components' running sums for that source dim).
    merged = []
    for j, ts in enumerate(events):
        merged.extend((float(t), j) for t in ts)
    merged.sort(key=lambda x: x[0])

    log_sum = 0.0
    dmu_from_log = np.zeros(d, dtype=np.float64)
    dalpha_from_log = np.zeros((K, d, d), dtype=np.float64)

    R = np.zeros((K, d), dtype=np.float64)
    t_prev = 0.0
    for (t_n, dim_n) in merged:
        dt = t_n - t_prev
        if dt > 0:
            # vectorised decay: each component decays at its own rate
            R *= np.exp(-decays[:, None] * dt)
        # intensity at firing time using R as of just before the firing
        # kernel_contrib[k, j] = beta_k * R[k, j]
        kernel_contrib = decays[:, None] * R  # (K, d)
        # lambda_i(t_n) = mu_i + sum_k sum_j alpha[k, i, j] * kernel_contrib[k, j]
        # For dim_n specifically: lam_n = mu[dim_n] + sum_{k,j} alpha[k, dim_n, j] * kernel_contrib[k, j]
        lam_n = mu[dim_n] + float(np.sum(alpha[:, dim_n, :] * kernel_contrib))
        if lam_n <= 0:
            lam_n = 1e-300
        log_sum += float(np.log(lam_n))
        # d(log lam_n) / d(mu_i) = 1[i == dim_n] / lam_n
        dmu_from_log[dim_n] += 1.0 / lam_n
        # d(log lam_n) / d(alpha[k, i, j]) = 1[i == dim_n] * kernel_contrib[k, j] / lam_n
        dalpha_from_log[:, dim_n, :] += kernel_contrib / lam_n
        # update R: event on source dim_n adds 1 to R[:, dim_n] for all components
        R[:, dim_n] += 1.0
        t_prev = t_n

    integral = float(np.sum(mu * end_time) + np.sum(alpha * S[:, None, :]))
    ll = log_sum - integral

    dmu = dmu_from_log - dmu_from_integral
    dalpha = dalpha_from_log - dalpha_from_integral
    return ll, dmu, dalpha
