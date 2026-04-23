"""Maximum-likelihood estimation for exponential-kernel multivariate Hawkes.

For a d-dimensional Hawkes process with shared decay beta, the log-likelihood
on an observation window [0, T] with event times {t_i^k} is

    ell = sum_i [ sum_k log(lambda_i(t_i^k)) - integral_0^T lambda_i(t) dt ]

With exponential kernels the integral has a closed form and the intensities at
event times can be computed in O(N) total using the Ogata (1981) recursion
exploited by tick / HawkesExpKern.

We parameterise mu_i = exp(log_mu_i) and alpha_ij = exp(log_alpha_ij) to keep
them strictly positive; decay beta is held fixed (passed in by the caller, or
selected via cross-validation).

Reference:
- Ogata, "On Lewis' simulation method for point processes", 1981.
- Laub, Taimre & Pollett, "Hawkes Processes", arXiv:1507.02822 (tutorial).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import minimize


@dataclass
class HawkesExpMLE:
    """Exponential-kernel multivariate Hawkes MLE.

    Args:
        decay: fixed scalar beta > 0. Choose by information criterion sweep
            if unknown; v0.1 does not optimise it jointly (see roadmap).
        penalty_l2: ridge on log_alpha (not on log_mu). Stabilises fits when
            some dims are near-empty. 0.0 disables.
        max_iter: L-BFGS-B iterations.
        tol: L-BFGS-B gradient tolerance.
        verbose: print scipy's optimiser log.
    """

    decay: float
    penalty_l2: float = 0.0
    max_iter: int = 500
    tol: float = 1e-6
    verbose: bool = False

    # fitted state
    baseline_: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    adjacency_: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    n_dim_: int = 0
    end_time_: float = 0.0
    converged_: bool = False
    n_iter_: int = 0
    final_loglik_: float = float("nan")

    def fit(self, events: Sequence[np.ndarray], end_time: float) -> "HawkesExpMLE":
        """Fit to a single realization.

        Args:
            events: length-d list where events[i] is the sorted 1D array of
                event times for dim i.
            end_time: observation window length T (the right edge; we assume
                the left edge is 0).

        Returns self with .baseline_ and .adjacency_ populated.
        """
        if self.decay <= 0:
            raise ValueError(f"decay must be positive, got {self.decay}")
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

        self.n_dim_ = d
        self.end_time_ = float(end_time)

        # initial guess: mu = N_i / T (empirical rate),
        # alpha = 0.1 * I / d (weak self-excitation, no cross for starters)
        counts = np.array([ts.size for ts in events], dtype=np.float64)
        mu0 = np.clip(counts / end_time, 1e-6, None)
        alpha0 = np.full((d, d), 0.1 / max(d, 1))
        np.fill_diagonal(alpha0, 0.1)
        alpha0 = np.clip(alpha0, 1e-6, None)

        x0 = np.concatenate([np.log(mu0), np.log(alpha0).ravel()])

        def neg_loglik_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
            log_mu = x[:d]
            log_alpha = x[d:].reshape(d, d)
            mu = np.exp(log_mu)
            alpha = np.exp(log_alpha)
            ll, dmu, dalpha = _loglik_and_grad(events, end_time, mu, alpha, self.decay)
            # chain rule: d(ll)/d(log_mu) = mu * d(ll)/d(mu); same for alpha
            g_log_mu = mu * dmu
            g_log_alpha = alpha * dalpha
            if self.penalty_l2 > 0:
                # L2 on log_alpha (not log_mu)
                ll = ll - 0.5 * self.penalty_l2 * float(np.sum(log_alpha * log_alpha))
                g_log_alpha = g_log_alpha - self.penalty_l2 * log_alpha
            # minimise negative log-lik
            grad = -np.concatenate([g_log_mu, g_log_alpha.ravel()])
            return -ll, grad

        options = {"maxiter": self.max_iter, "gtol": self.tol}
        if self.verbose:
            # scipy 1.18 removes `disp`; the callback-free path is silent by default
            options["disp"] = True
        result = minimize(
            neg_loglik_and_grad,
            x0,
            jac=True,
            method="L-BFGS-B",
            options=options,
        )

        log_mu = result.x[:d]
        log_alpha = result.x[d:].reshape(d, d)
        self.baseline_ = np.exp(log_mu)
        self.adjacency_ = np.exp(log_alpha)
        self.converged_ = bool(result.success)
        self.n_iter_ = int(result.nit)
        self.final_loglik_ = float(-result.fun)
        return self

    def branching_ratio(self) -> float:
        """Spectral radius of fitted adjacency. < 1 ⇒ stable."""
        if self.adjacency_ is None:
            raise RuntimeError("fit() must be called before branching_ratio()")
        return float(np.max(np.abs(np.linalg.eigvals(self.adjacency_))))


def _loglik_and_grad(
    events: list[np.ndarray],
    end_time: float,
    mu: np.ndarray,
    alpha: np.ndarray,
    beta: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute log-likelihood and gradients wrt mu and alpha.

    Uses the Ogata recursion for O(N_total) evaluation. Gradients are exact.
    """
    d = mu.shape[0]

    # ---- integral part: -sum_i int_0^T lambda_i(t) dt
    # int_0^T mu_i dt = mu_i * T
    # int_0^T alpha_ij * beta * exp(-beta * (t - t_j^k)) dt
    #       = alpha_ij * [1 - exp(-beta * (T - t_j^k))]   (summed over t_j^k < T)
    # For each source dim j define
    #   S_j = sum_k (1 - exp(-beta * (T - t_j^k)))
    # Then integral over dim i = mu_i * T + sum_j alpha_ij * S_j.
    S = np.zeros(d, dtype=np.float64)
    for j in range(d):
        ts = events[j]
        if ts.size:
            S[j] = float(np.sum(1.0 - np.exp(-beta * (end_time - ts))))
    # derivative: d(integral_i)/d(mu_i) = T; d(integral_i)/d(alpha_ij) = S_j
    dmu_from_integral = np.full(d, end_time)
    dalpha_from_integral = np.tile(S, (d, 1))  # shape (d, d); row i all equal to S

    # ---- log-intensity part: sum_i sum_k log(lambda_i(t_i^k))
    # Compute lambda_i at every event time. Ogata recursion:
    # For each ordered *merged* event stream, maintain per-source-dim running
    # kernel sums r_j(t) = beta * sum_{t_j^m < t} exp(-beta * (t - t_j^m)).
    # Then lambda_i(t) = mu_i + sum_j alpha_ij * r_j(t) / beta?
    # Actually let's define R_j(t) = sum_{t_j^m < t} exp(-beta * (t - t_j^m)).
    # Then the kernel contribution from source j to intensity of dim i is
    #   alpha_ij * beta * R_j(t).
    # Between events R_j decays as R_j(t+dt) = R_j(t) * exp(-beta * dt).
    # When an event occurs on source j at time s, R_j increments by 1 (adding
    # the unit impulse that then decays).
    #
    # We want log lambda_i at each event in dim i's stream. Process events in
    # a merged chronological order, updating R per source dim between times,
    # and record lambda_i at each time that belongs to dim i.

    # Build merged stream: list of (t, dim_of_firing). Events from dim i
    # contribute to the likelihood at row i. Events from dim j contribute to
    # R_j (incrementing it by 1 at time t_j^k+).
    # Note: at an event time t_i^k we need lambda_i(t_i^k) using R_j values
    # just BEFORE the event (since lambda is left-continuous at the firing).
    merged = []
    for j, ts in enumerate(events):
        merged.extend((float(t), j) for t in ts)
    merged.sort(key=lambda x: x[0])

    log_sum = 0.0
    dmu_from_log = np.zeros(d, dtype=np.float64)  # accumulator
    dalpha_from_log = np.zeros((d, d), dtype=np.float64)

    R = np.zeros(d, dtype=np.float64)
    t_prev = 0.0
    for (t_k, dim_k) in merged:
        dt = t_k - t_prev
        if dt > 0:
            R *= np.exp(-beta * dt)
        # intensity at firing time, using R as of just before the firing
        kernel_vec = beta * R  # shape (d,)
        # lambda_i(t_k) = mu_i + sum_j alpha_ij * kernel_vec[j]
        lam_k = mu[dim_k] + float(alpha[dim_k] @ kernel_vec)
        # guard: lam_k should be > 0 since mu > 0; clamp for numerical safety
        if lam_k <= 0:
            lam_k = 1e-300
        log_sum += float(np.log(lam_k))
        # d(log lam_k) / d(mu_i) = 1[i == dim_k] / lam_k
        dmu_from_log[dim_k] += 1.0 / lam_k
        # d(log lam_k) / d(alpha_ij) = 1[i == dim_k] * kernel_vec[j] / lam_k
        dalpha_from_log[dim_k] += kernel_vec / lam_k
        # update R: the event on source dim_k adds 1 to R[dim_k]
        R[dim_k] += 1.0
        t_prev = t_k

    # total log-likelihood = log_sum - sum_i integral_i
    integral = float(np.sum(mu * end_time) + np.sum(alpha * S[None, :]))
    ll = log_sum - integral

    # gradients of -integral part: negate the "from_integral" arrays
    dmu = dmu_from_log - dmu_from_integral
    dalpha = dalpha_from_log - dalpha_from_integral
    return ll, dmu, dalpha
