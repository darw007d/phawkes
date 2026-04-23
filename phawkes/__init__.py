"""phawkes — pure-Python multivariate Hawkes processes with exponential kernels.

Public API:

    from phawkes import HawkesExpSim, HawkesExpMLE, branching_ratio

    sim = HawkesExpSim(baseline=[0.3, 0.2], adjacency=[[0.2, 0.1], [0.3, 0.15]], decay=1.0)
    events = sim.simulate(end_time=2000)

    learner = HawkesExpMLE(decay=1.0).fit(events, end_time=2000)
    learner.baseline_
    learner.adjacency_
    learner.branching_ratio()
"""

from phawkes.exp_mle import HawkesExpMLE
from phawkes.metrics import branching_ratio
from phawkes.regimes import RegimeSegment, fit_per_regime
from phawkes.simulator import HawkesExpSim

__all__ = [
    "HawkesExpSim",
    "HawkesExpMLE",
    "branching_ratio",
    "RegimeSegment",
    "fit_per_regime",
]
__version__ = "0.1.0"
