# phawkes

Pure-Python multivariate Hawkes processes with exponential kernels — simulation, MLE, and systemic-risk metrics. No C++ dependencies.

## Why this exists

There is already an excellent Hawkes library: [**tick**](https://github.com/X-DataInitiative/tick) from École Polytechnique. We recommend it whenever it works for you.

However, as of Python 3.13, `tick 0.8.0.1` builds but its `Base` metaclass breaks on instantiation of any learner class (`'HawkesExpKern' object has no settable attribute 'events'`). This is a known incompatibility with CPython 3.13's stricter descriptor protocol. The simulator side of `tick` still works; only the fitters fail.

`phawkes` fills the gap with a minimal, tested, pure-`numpy` + `scipy` implementation of the exponential-kernel multivariate Hawkes MLE. It is deliberately small — if you need sum-of-exponentials kernels, L1 / trace-norm regularization, EM for semi-parametric kernels, or GPU Riemannian fits, `tick` on Python ≤ 3.12 is still the better tool.

## Install

```bash
pip install phawkes
```

Requires Python ≥ 3.9, `numpy ≥ 1.23`, `scipy ≥ 1.10`.

## Quickstart

```python
import numpy as np
from phawkes import HawkesExpSim, HawkesExpMLE, branching_ratio

true_mu = np.array([0.3, 0.2])
true_alpha = np.array([[0.2, 0.1], [0.3, 0.15]])
beta = 1.0

# simulate a realization on [0, 4000]
sim = HawkesExpSim(baseline=true_mu, adjacency=true_alpha, decay=beta, seed=7)
events = sim.simulate(end_time=4000.0)

# fit by MLE
learner = HawkesExpMLE(decay=beta).fit(events, end_time=4000.0)
print(learner.baseline_)        # ~ true_mu
print(learner.adjacency_)       # ~ true_alpha
print(learner.branching_ratio())  # spectral radius of fitted alpha
```

## The math (1-minute version)

A multivariate exponential-kernel Hawkes process with shared decay `β` has conditional intensity

$$
\lambda_i(t) = \mu_i + \sum_j \sum_{t_j^k < t} \alpha_{ij}\, \beta\, e^{-\beta (t - t_j^k)}
$$

- `μᵢ ≥ 0` is the baseline (background) rate on dim `i`
- `αᵢⱼ ≥ 0` is the branching coefficient from dim `j` to dim `i`
- `β > 0` controls how fast past events' excitation decays

Stability (finite expected event count per unit time) requires the spectral radius of `α` — the **branching ratio** `λ_max(α)` — to be strictly less than 1. As it approaches 1, a single shock can cascade through many downstream events; this is the quantitative signature of *near-critical contagion* in finance and epidemics.

Log-likelihood on the window `[0, T]`:

$$
\ell = \sum_i \left[\sum_k \log \lambda_i(t_i^k)\right] - \sum_i \int_0^T \lambda_i(t)\, dt
$$

For exponential kernels the integral is closed-form, and the sum of log-intensities admits [Ogata's 1981 recursion](https://doi.org/10.1109/TIT.1981.1056305) for O(N) total evaluation. `phawkes.HawkesExpMLE` implements this, parameterises `μ = exp(log_μ)` and `α = exp(log_α)` for strict positivity, optimises by L-BFGS-B with analytic gradients, and optionally ridges `log_α`.

## Per-regime fitting

Pooling Hawkes events across different market regimes creates spurious criticality — the MLE interprets regime shifts as endogenous self-excitation and inflates the branching ratio. `fit_per_regime` slices event streams by labelled time intervals and fits a separate MLE per label:

```python
from phawkes import RegimeSegment, fit_per_regime

segments = [
    RegimeSegment(0.0,    500.0, "calm"),
    RegimeSegment(500.0,  700.0, "crisis"),
    RegimeSegment(700.0, 1000.0, "calm"),
]
fits = fit_per_regime(events, segments, decay=1.0)
fits["calm"].branching_ratio()     # e.g. 0.3
fits["crisis"].branching_ratio()   # e.g. 0.85 — near-critical
```

## Roadmap

**v0.1 (this release):**
- Exponential-kernel multivariate Hawkes MLE with shared decay
- Ogata-thinning simulator
- Per-regime convenience wrapper
- Branching-ratio utility
- Unit-tested against simulator ground truth on 2-dim and 5-dim problems

**v0.2+ (planned):**
- Sum-of-exponentials kernels (richer temporal structure)
- L1 / trace-norm regularization (Bacry et al. 2020, *JMLR*)
- Per-edge decay `βᵢⱼ` (currently shared)
- Cross-validation for choosing `β`
- Benchmark parity with `tick` on Python 3.12

## Authors

- **Pierre Samson** ([@darw007d](https://github.com/darw007d)) — idea, use-case, design decisions
- **Claude Opus** (Anthropic) — implementation and tests

Originally motivated by the [OMEGA Swarm](https://github.com/darw007d/hedge-fund-mcp) hedge-fund project, where regime-stratified Hawkes calibration replaces a heuristic `5% per cycle` edge-decay in a live contagion network. The library is deliberately agnostic of that use-case.

## Contributing

Issues and PRs welcome. We care about:
- Numerical correctness (every new fitter needs a simulator-recovery test)
- Zero heavy dependencies (no C++, no `tensorflow` / `torch`)
- Honest documentation — if something is slow or unstable, say so

## Citations

If `phawkes` contributes to a published result, please cite the foundational references alongside this library:

- Hawkes, A.G. (1971). *Spectra of some self-exciting and mutually exciting point processes*. Biometrika.
- Ogata, Y. (1981). *On Lewis' simulation method for point processes*. IEEE Trans. Inform. Theory.
- Laub, P.J., Taimre, T. & Pollett, P.K. (2015). *Hawkes Processes*. arXiv:1507.02822.
- Bacry, E., Bompaire, M., Gaïffas, S. & Muzy, J.-F. (2020). *Sparse and low-rank multivariate Hawkes processes*. JMLR.

## License

MIT — see [LICENSE](LICENSE).
