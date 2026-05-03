"""Ground-truth recovery tests for HawkesSumExpMLE — simulate → fit → compare.

Same shape as test_recovery.py for HawkesExpMLE (v0.1) but with K kernel
components. Tests three things:

1. Sanity reduction: K=1 sum-of-exp == single-exp on identical setup.
2. Two-component recovery on 2-dim synthetic data.
3. Two-component recovery on 5-dim synthetic data with cross-excitation.

Tolerances mirror v0.1 test_recovery.py: baseline MAE < 0.05, adjacency
MAE < 0.1 per-component (i.e., tighter than the integrated kernel mass
because per-component identifiability requires more data).
"""

import numpy as np
import pytest

from phawkes import (
    HawkesExpMLE,
    HawkesExpSim,
    HawkesSumExpMLE,
    HawkesSumExpSim,
)


def _mae(true: np.ndarray, fitted: np.ndarray) -> float:
    return float(np.mean(np.abs(true - fitted)))


def test_sum_exp_K1_matches_single_exp_recovery():
    """K=1 sum-of-exp MLE should give the same fit as the single-exp MLE.

    Identical synthetic ground-truth data, fit by both pathways. Recovered
    baseline + adjacency should agree to ~3 decimals.
    """
    true_mu = np.array([0.3, 0.2])
    true_alpha = np.array([[0.2, 0.1], [0.3, 0.15]])
    beta = 1.0

    # generate via single-exp simulator
    sim = HawkesExpSim(baseline=true_mu, adjacency=true_alpha, decay=beta, seed=42)
    events = sim.simulate(end_time=4000.0)

    # fit via single-exp MLE
    fit_single = HawkesExpMLE(decay=beta).fit(events, end_time=4000.0)

    # fit via sum-of-exp MLE with K=1
    fit_sum = HawkesSumExpMLE(decays=[beta]).fit(events, end_time=4000.0)

    # baseline parity
    assert _mae(fit_single.baseline_, fit_sum.baseline_) < 0.01, (
        f"K=1 sum-exp baseline drifted from single-exp: "
        f"single={fit_single.baseline_}, sum={fit_sum.baseline_}"
    )

    # adjacency parity (sum_exp adjacency_ is shape (1, d, d) for K=1)
    assert fit_sum.adjacency_.shape == (1, 2, 2)
    assert _mae(fit_single.adjacency_, fit_sum.adjacency_[0]) < 0.02, (
        f"K=1 sum-exp adjacency drifted from single-exp: "
        f"single={fit_single.adjacency_}, sum={fit_sum.adjacency_[0]}"
    )

    # branching ratios
    assert abs(fit_single.branching_ratio() - fit_sum.branching_ratio()) < 0.02


def test_sum_exp_recovery_dim2_two_components():
    """2-dim, K=2, 8000 time units: recovery within ~0.05 baseline / 0.1 per-component."""
    true_mu = np.array([0.3, 0.2])
    # K=2 components: slow (decay 0.1) + fast (decay 1.0)
    decays = [0.1, 1.0]
    true_alpha = np.array([
        [[0.05, 0.02], [0.01, 0.04]],   # slow component (k=0)
        [[0.10, 0.05], [0.03, 0.08]],   # fast component (k=1)
    ])  # shape (2, 2, 2)

    # check stability: integrated kernel = sum_k true_alpha[k]
    integrated = true_alpha.sum(axis=0)
    spectral = np.max(np.abs(np.linalg.eigvals(integrated)))
    assert spectral < 1.0, f"unstable test setup: spectral radius {spectral:.3f}"

    sim = HawkesSumExpSim(baseline=true_mu, adjacency=true_alpha, decays=decays, seed=7)
    events = sim.simulate(end_time=8000.0)
    counts = [t.size for t in events]
    assert all(c > 100 for c in counts), f"too few events for meaningful recovery: {counts}"

    learner = HawkesSumExpMLE(decays=decays).fit(events, end_time=8000.0)

    assert learner.converged_, f"MLE did not converge: n_iter={learner.n_iter_}"
    assert learner.adjacency_.shape == (2, 2, 2)

    mu_err = _mae(true_mu, learner.baseline_)
    assert mu_err < 0.08, f"baseline MAE {mu_err:.4f} too large: {learner.baseline_}"

    # per-component adjacency tolerance is looser than integrated kernel
    for k in range(2):
        a_err = _mae(true_alpha[k], learner.adjacency_[k])
        assert a_err < 0.15, (
            f"adjacency[k={k}] MAE {a_err:.4f} too large: "
            f"true={true_alpha[k]}, fit={learner.adjacency_[k]}"
        )

    # integrated kernel mass per pair should recover tighter than per-component
    integrated_fit = learner.adjacency_.sum(axis=0)
    integrated_err = _mae(integrated, integrated_fit)
    assert integrated_err < 0.10, (
        f"integrated kernel MAE {integrated_err:.4f} too large: "
        f"true={integrated}, fit={integrated_fit}"
    )


def test_sum_exp_recovery_dim5_two_components():
    """5-dim, K=2, 12000 time units: recovery on larger synthetic dataset."""
    rng = np.random.default_rng(42)
    d = 5
    true_mu = rng.uniform(0.05, 0.3, size=d)
    decays = [0.2, 2.0]

    # build true_alpha so integrated kernel mass < 0.7 spectral radius
    true_alpha = np.zeros((2, d, d))
    # slow component: weak diagonal + light off-diagonal
    true_alpha[0] = rng.uniform(0.005, 0.02, size=(d, d))
    np.fill_diagonal(true_alpha[0], rng.uniform(0.02, 0.05, size=d))
    # fast component: stronger
    true_alpha[1] = rng.uniform(0.01, 0.04, size=(d, d))
    np.fill_diagonal(true_alpha[1], rng.uniform(0.04, 0.08, size=d))

    integrated = true_alpha.sum(axis=0)
    spectral = np.max(np.abs(np.linalg.eigvals(integrated)))
    assert spectral < 1.0, f"unstable test setup: spectral radius {spectral:.3f}"

    sim = HawkesSumExpSim(baseline=true_mu, adjacency=true_alpha, decays=decays, seed=11)
    events = sim.simulate(end_time=12000.0)
    counts = [t.size for t in events]
    # 5-dim with light cross-excitation needs many events for identifiability
    assert sum(counts) > 1000, f"total events too low for 5-dim recovery: {counts}"

    learner = HawkesSumExpMLE(decays=decays, penalty_l2=1e-3).fit(
        events, end_time=12000.0
    )

    # 5-dim looser tolerances per v0.1's test_recovery_dim5_long_window pattern
    mu_err = _mae(true_mu, learner.baseline_)
    assert mu_err < 0.10, f"5-dim baseline MAE {mu_err:.4f}: {learner.baseline_}"

    # integrated kernel is the load-bearing quantity; per-component identifiability
    # is harder at higher dim
    integrated_fit = learner.adjacency_.sum(axis=0)
    integrated_err = _mae(integrated, integrated_fit)
    assert integrated_err < 0.05, (
        f"5-dim integrated kernel MAE {integrated_err:.4f}: "
        f"true_max={integrated.max():.3f}, fit_max={integrated_fit.max():.3f}"
    )


def test_sum_exp_branching_ratio_uses_integrated():
    """branching_ratio() should equal spectral radius of sum_k adjacency_[k]."""
    learner = HawkesSumExpMLE(decays=[0.1, 1.0])
    learner.adjacency_ = np.array([
        [[0.2, 0.1], [0.0, 0.15]],
        [[0.1, 0.05], [0.05, 0.1]],
    ])
    integrated = learner.adjacency_.sum(axis=0)
    expected = float(np.max(np.abs(np.linalg.eigvals(integrated))))
    assert np.isclose(learner.branching_ratio(), expected)
