"""phawkes quickstart — simulate, fit, read off the branching ratio."""

import numpy as np

from phawkes import HawkesExpMLE, HawkesExpSim, branching_ratio

# --- ground-truth bivariate Hawkes process ---
true_mu = np.array([0.3, 0.2])
true_alpha = np.array([[0.2, 0.1], [0.3, 0.15]])
beta = 1.0

print(f"true branching ratio: {branching_ratio(true_alpha):.3f}")

# --- simulate on [0, T] ---
sim = HawkesExpSim(baseline=true_mu, adjacency=true_alpha, decay=beta, seed=7)
events = sim.simulate(end_time=4000.0)
print(f"event counts per dim: {[e.size for e in events]}")

# --- fit by MLE ---
learner = HawkesExpMLE(decay=beta).fit(events, end_time=4000.0)

print()
print("fitted baseline:", np.round(learner.baseline_, 3))
print("fitted adjacency:")
print(np.round(learner.adjacency_, 3))
print(f"fitted branching ratio: {learner.branching_ratio():.3f}")
print(f"converged in {learner.n_iter_} iters, log-lik = {learner.final_loglik_:.1f}")
