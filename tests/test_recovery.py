"""Ground-truth recovery tests: simulate → fit → compare."""

import numpy as np
import pytest

from phawkes import HawkesExpMLE, HawkesExpSim, branching_ratio


def _recovery_error(true: np.ndarray, fitted: np.ndarray) -> float:
    """Mean absolute error between a ground-truth and a fitted parameter."""
    return float(np.mean(np.abs(true - fitted)))


def test_branching_ratio_matches_eigvals():
    a = np.array([[0.2, 0.1], [0.3, 0.15]])
    assert np.isclose(branching_ratio(a), np.max(np.abs(np.linalg.eigvals(a))))


def test_recovery_dim2_long_window():
    """2-dim, 4000 time units: MLE should recover mu and alpha within ~0.05."""
    true_mu = np.array([0.3, 0.2])
    true_alpha = np.array([[0.2, 0.1], [0.3, 0.15]])
    beta = 1.0

    sim = HawkesExpSim(baseline=true_mu, adjacency=true_alpha, decay=beta, seed=7)
    events = sim.simulate(end_time=4000.0)
    counts = [t.size for t in events]
    assert all(c > 100 for c in counts), f"too few events for meaningful recovery: {counts}"

    learner = HawkesExpMLE(decay=beta).fit(events, end_time=4000.0)

    assert learner.converged_, "MLE failed to converge"
    assert _recovery_error(true_mu, learner.baseline_) < 0.05, (
        f"baseline: true {true_mu}, fitted {learner.baseline_}"
    )
    assert _recovery_error(true_alpha, learner.adjacency_) < 0.1, (
        f"adjacency: true\n{true_alpha}\nfitted\n{learner.adjacency_}"
    )
    true_br = branching_ratio(true_alpha)
    assert abs(learner.branching_ratio() - true_br) < 0.1


def test_recovery_dim5():
    """5-dim, 3000 time units, sparse branching. Larger tolerance."""
    rng = np.random.default_rng(42)
    d = 5
    true_mu = rng.uniform(0.1, 0.3, size=d)
    true_alpha = rng.uniform(0.0, 0.08, size=(d, d))
    np.fill_diagonal(true_alpha, rng.uniform(0.1, 0.2, size=d))
    beta = 1.0

    sim = HawkesExpSim(baseline=true_mu, adjacency=true_alpha, decay=beta, seed=11)
    events = sim.simulate(end_time=3000.0)

    learner = HawkesExpMLE(decay=beta).fit(events, end_time=3000.0)
    assert learner.converged_

    # Per-entry tolerance grows with d; check averages
    mu_err = _recovery_error(true_mu, learner.baseline_)
    alpha_err = _recovery_error(true_alpha, learner.adjacency_)
    assert mu_err < 0.1, f"mu err {mu_err:.3f}"
    assert alpha_err < 0.08, f"alpha err {alpha_err:.3f}"


def test_fit_rejects_bad_input():
    learner = HawkesExpMLE(decay=1.0)
    with pytest.raises(ValueError, match="end_time"):
        learner.fit([np.array([0.5])], end_time=0.0)
    with pytest.raises(ValueError, match="outside"):
        learner.fit([np.array([2.0])], end_time=1.0)
    with pytest.raises(ValueError, match="sorted"):
        learner.fit([np.array([1.0, 0.5])], end_time=2.0)


def test_branching_ratio_before_fit_raises():
    learner = HawkesExpMLE(decay=1.0)
    with pytest.raises(RuntimeError):
        learner.branching_ratio()
