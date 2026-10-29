# QUBO portfolio optimization

Cardinality-constrained portfolio selection as a QUBO, mapped to an Ising model and sampled with THRML.

`portfolio_optimization.py` is a standalone script (it needs only `thrml`, `jax` and `networkx`; `numpy` is installed
with `jax`). It builds a synthetic 20-asset market, selects exactly `K = 5` assets, and compares the sampled result
with exhaustive enumeration of all `C(20, 5) = 15,504` portfolios.

```bash
python examples/qubo_portfolio_optimization/portfolio_optimization.py
```

## The problem

Choose a binary vector `x in {0,1}^N` (`x_i = 1` means asset `i` is held), with `N = 20` assets and a target of
`K = 5` holdings:

$$
\min_{x \in \{0,1\}^N} \; q\, x^\top \Sigma x \;-\; \mu^\top x \;+\; \lambda \Big(\sum_{i=1}^{N} x_i - K\Big)^2 .
$$

- `x^T Sigma x` is the portfolio variance (the quadratic risk term), weighted by the risk aversion `q`.
- `mu^T x` is the expected return.
- `lambda (sum_i x_i - K)^2` is a quadratic budget penalty. It is zero exactly when `K` assets are held and grows
  quadratically otherwise. The constraint is soft: whether samples satisfy it depends on `lambda` and on the
  temperature, which is why the script reports a feasibility rate.

The market is synthetic (`make_market`): returns drawn uniformly from 5-20%, a factor-model covariance, and five
"star" assets (2, 5, 9, 14, 18) with boosted returns and low loadings.

## Hamiltonian derivation

Use `x_i^2 = x_i` to write the objective, up to a constant, as

$$
\sum_i Q_{ii} x_i + 2 \sum_{i<j} Q_{ij} x_i x_j,
\qquad
Q_{ii} = q\,\Sigma_{ii} - \mu_i + \lambda (1 - 2K),
\qquad
Q_{ij} = q\,\Sigma_{ij} + \lambda \;\; (i \ne j).
$$

Substitute `x_i = (s_i + 1) / 2` with `s_i in {-1, +1}`. The pair term becomes
`2 Q_ij x_i x_j = (Q_ij / 2)(s_i s_j + s_i + s_j + 1)`, so every pair contributes `Q_ij / 2` to the field on both of its
spins. Collecting terms gives `E(s) = sum_{i<j} (Q_ij / 2) s_i s_j + sum_i (Q_ii / 2 + (1/2) sum_{j != i} Q_ij) s_i + const`.
THRML's `IsingEBM` uses `E(s) = -beta (sum_i b_i s_i + sum_{i<j} J_ij s_i s_j)`, hence

$$
J_{ij} = -\tfrac{1}{2} Q_{ij},
\qquad
b_i = -\tfrac{1}{2} Q_{ii} - \tfrac{1}{2} \sum_{j \ne i} Q_{ij}.
$$

The Boltzmann distribution at inverse temperature `beta` is proportional to `exp(-beta * E_qubo(x))`, so energy
*differences* between portfolios are preserved exactly by the mapping (the dropped constant does not matter for
sampling). THRML samples booleans, and `True` corresponds to `s_i = +1`, i.e. `x_i = 1`.

## Sampling

- **Block partitioning.** `build_program` colours the interaction graph with `networkx` and uses each colour class as a
  free block, which is the correct conditionally independent partition. **Note:** the budget penalty couples every
  pair of assets, so this graph is complete and the colouring has 20 classes of one spin each. For this problem the
  sampler therefore degenerates to sequential Gibbs; there is no parallelism across spins to exploit. The example is
  about the QUBO-to-Ising mapping, not about parallel speedup.
- **Schedule.** Six stages, `beta in (0.4, 0.8, 1.5, 2.5, 5.0, 12.0)`, each with 60 warmup sweeps and 350 samples.
  After each stage the next one is warm-started from the lowest-energy *feasible* sample found so far (the first stage
  starts from a random 5-asset portfolio). This is a heuristic that makes the schedule behave like annealing with
  restarts; it is not a plain single-chain anneal.

## Results

Run on CPU with `thrml` 0.1.4 (fixed seeds, so the output is reproducible):

| `beta` | Budget feasibility (`sum x = 5`) | Mean QUBO energy | Min energy |
| ---: | ---: | ---: | ---: |
| 0.4 | 18.3% | +1.4588 | -0.9539 |
| 0.8 | 25.4% | +0.3275 | -0.9522 |
| 1.5 | 41.1% | -0.1302 | -1.0673 |
| 2.5 | 56.6% | -0.3679 | -1.0654 |
| 5.0 | 81.1% | -0.5710 | -1.0561 |
| 12.0 | **99.1%** | -0.8591 | -1.0762 |

- **Budget feasibility: 99.1%** of the samples at the coldest stage (`beta = 12`) hold exactly 5 assets.
- **Pareto / objective match: 94.1%.** This is the ratio of the best feasible sampled energy to the brute-force
  optimum, `-1.0762 / -1.1443`. It is a ratio of objective values (both negative), not the fraction of runs that find
  the optimum.
- The sampled portfolio is `[0, 2, 5, 9, 18]`; the exact optimum is `[2, 5, 9, 14, 18]`. The sampler found 4 of the
  5 optimal assets and substituted asset 0 for asset 14, so it did **not** reach the global optimum on this instance.
- Wall time is about 15 s on CPU, dominated by JIT compilation of one program per temperature.

These numbers come from a single synthetic instance and a single seed; they illustrate the mapping and are not a
statistical benchmark against classical solvers.

## Using your own data

Replace `make_market` with real `mu` (length `N`) and `sigma` (`N x N`, symmetric positive semi-definite). Choose
`lambda` large enough that infeasible portfolios are never lower in energy than feasible ones (here `lambda = 0.65`
against per-asset terms of order 0.1-0.3), and increase `beta` until feasibility saturates.
