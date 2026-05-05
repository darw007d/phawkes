"""phawkes — pure-Python multivariate Hawkes processes with exponential kernels.

Public API (v0.2):

    from phawkes import HawkesExpSim, HawkesExpMLE, branching_ratio

    # Single-exp kernel (v0.1 API, unchanged)
    sim = HawkesExpSim(baseline=[0.3, 0.2], adjacency=[[0.2, 0.1], [0.3, 0.15]], decay=1.0)
    events = sim.simulate(end_time=2000)

    learner = HawkesExpMLE(decay=1.0).fit(events, end_time=2000)
    learner.baseline_
    learner.adjacency_
    learner.branching_ratio()

    # Sum-of-exp kernel (v0.2 — multi-scale temporal patterns)
    from phawkes import HawkesSumExpSim, HawkesSumExpMLE
    import numpy as np

    decays = [0.1, 1.0]  # slow + fast components
    alpha_K = np.array([
        [[0.05, 0.02], [0.01, 0.04]],   # slow component (k=0)
        [[0.10, 0.05], [0.03, 0.08]],   # fast component (k=1)
    ])  # shape (K=2, d=2, d=2)
    sim = HawkesSumExpSim(baseline=[0.3, 0.2], adjacency=alpha_K, decays=decays)
    events = sim.simulate(end_time=2000)

    learner = HawkesSumExpMLE(decays=decays).fit(events, end_time=2000)
    learner.adjacency_  # shape (K, d, d)
    learner.branching_ratio()  # spectral radius of integrated sum_k adjacency_[k]
"""

from phawkes.exp_mle import HawkesExpMLE
from phawkes.metrics import branching_ratio
from phawkes.regimes import RegimeSegment, fit_per_regime
from phawkes.simulator import HawkesExpSim
from phawkes.sum_exp_mle import HawkesSumExpMLE
from phawkes.sum_exp_sim import HawkesSumExpSim

__all__ = [
    "HawkesExpSim",
    "HawkesExpMLE",
    "HawkesSumExpSim",
    "HawkesSumExpMLE",
    "branching_ratio",
    "RegimeSegment",
    "fit_per_regime",
]
__version__ = "0.2.0"
