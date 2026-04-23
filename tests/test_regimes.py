"""Per-regime fit: simulate two regimes with different adjacencies, fit
separately, check each recovers its own ground truth."""

import numpy as np

from phawkes import HawkesExpSim, RegimeSegment, fit_per_regime


def test_two_regime_fit_separates():
    beta = 1.0
    # Regime A: weak self-excitation
    mu_a = np.array([0.3, 0.2])
    alpha_a = np.array([[0.15, 0.05], [0.05, 0.1]])
    # Regime B: strong, critical-ish
    mu_b = np.array([0.4, 0.3])
    alpha_b = np.array([[0.4, 0.25], [0.3, 0.35]])

    sim_a = HawkesExpSim(baseline=mu_a, adjacency=alpha_a, decay=beta, seed=11)
    ev_a = sim_a.simulate(end_time=2000.0)
    sim_b = HawkesExpSim(baseline=mu_b, adjacency=alpha_b, decay=beta, seed=13)
    ev_b = sim_b.simulate(end_time=2000.0)

    # stitch: regime A on [0, 2000), regime B on [2000, 4000)
    d = 2
    stitched = []
    for i in range(d):
        a_times = ev_a[i]
        b_times = ev_b[i] + 2000.0
        stitched.append(np.sort(np.concatenate([a_times, b_times])))

    segments = [
        RegimeSegment(0.0, 2000.0, "A"),
        RegimeSegment(2000.0, 4000.0, "B"),
    ]
    fits = fit_per_regime(stitched, segments, decay=beta)

    assert set(fits.keys()) == {"A", "B"}

    fa = fits["A"]
    fb = fits["B"]
    assert fa.converged_ and fb.converged_

    # A should recover alpha_a with small self-excitation; B recovers larger alpha
    assert np.mean(np.abs(fa.baseline_ - mu_a)) < 0.08
    assert np.mean(np.abs(fb.baseline_ - mu_b)) < 0.1
    assert np.mean(np.abs(fa.adjacency_ - alpha_a)) < 0.1
    assert np.mean(np.abs(fb.adjacency_ - alpha_b)) < 0.15

    # regime B's branching ratio should be clearly higher than A's
    assert fb.branching_ratio() > fa.branching_ratio() + 0.1
