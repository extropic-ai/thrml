"""Cardinality-constrained portfolio selection as a QUBO, sampled with THRML.

Select exactly ``K`` of ``N`` assets so as to trade off expected return against covariance risk:

    min_{x in {0,1}^N}   q * x^T Sigma x  -  mu^T x  +  lambda * (sum_i x_i - K)^2

The binary variables are mapped to spins ``s_i = 2 x_i - 1`` and the resulting Ising model is sampled with
chromatic block Gibbs sampling (greedy graph colouring -> conditionally independent free blocks) under a
temperature schedule. The result is compared against exhaustive enumeration of all C(N, K) portfolios.

Dependencies: ``thrml``, ``jax`` and ``networkx`` (``numpy`` is installed as a dependency of ``jax``).

Run with::

    python portfolio_optimization.py
"""

import time
from itertools import combinations

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np

from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram

N_ASSETS = 20
K_ASSETS = 5
RISK_AVERSION = 1.0  # q
PENALTY = 0.65  # lambda
BETAS = (0.4, 0.8, 1.5, 2.5, 5.0, 12.0)  # inverse-temperature schedule, hot -> cold
N_WARMUP = 60
N_SAMPLES = 350


def make_market(n_assets: int = N_ASSETS, seed: int = 42):
    """Synthetic expected returns ``mu`` and a positive-definite covariance ``sigma``.

    Five "star" assets get a return boost and low factor loadings, so a good (and, for this seed, the optimal)
    portfolio is known to exist.
    """
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.05, 0.20, size=n_assets)
    stars = [2, 5, 9, 14, 18]
    mu[stars] += 0.12

    loadings = rng.normal(0.0, 0.08, size=(n_assets, 4))
    for k, idx in enumerate(stars):
        loadings[idx] = np.eye(4)[k % 4] * 0.04
    idiosyncratic = np.diag(rng.uniform(0.03, 0.06, size=n_assets))
    sigma = loadings @ loadings.T + idiosyncratic
    return mu, sigma


def qubo_to_ising(mu: np.ndarray, sigma: np.ndarray, k: int, q: float, lam: float):
    """Build the QUBO matrix ``Q`` and convert it to Ising biases and couplings.

    Writing the objective as ``sum_i Q_ii x_i + 2 sum_{i<j} Q_ij x_i x_j`` (up to a constant) gives

        Q_ii = q * Sigma_ii - mu_i + lam * (1 - 2K)
        Q_ij = q * Sigma_ij + lam                      (i != j)

    and substituting ``x_i = (s_i + 1) / 2`` yields, in THRML's convention ``E = -(sum b_i s_i + sum J_ij s_i s_j)``,

        J_ij = -Q_ij / 2
        b_i  = -Q_ii / 2 - (1/2) * sum_{j != i} Q_ij
    """
    n = len(mu)
    big_q = q * sigma + lam * np.ones((n, n))
    np.fill_diagonal(big_q, q * np.diag(sigma) - mu + lam * (1.0 - 2.0 * k))

    off_diag_row_sums = big_q.sum(axis=1) - np.diag(big_q)
    biases = -0.5 * np.diag(big_q) - 0.5 * off_diag_row_sums
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    weights = np.array([-0.5 * big_q[i, j] for i, j in edges])
    return biases.astype(np.float32), edges, weights.astype(np.float32)


def objective(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray, k: int, q: float, lam: float):
    """Return ``(expected_return, volatility, qubo_energy)`` of a binary allocation ``x``."""
    ret = float(mu @ x)
    var = float(x @ sigma @ x)
    energy = q * var - ret + lam * (int(x.sum()) - k) ** 2
    return ret, float(np.sqrt(var)), energy


def brute_force(mu: np.ndarray, sigma: np.ndarray, k: int, q: float, lam: float):
    """Exhaustively enumerate all C(N, K) feasible portfolios and return the best one."""
    n = len(mu)
    best_x, best_e = None, np.inf
    for combo in combinations(range(n), k):
        x = np.zeros(n, dtype=np.int32)
        x[list(combo)] = 1
        energy = objective(x, mu, sigma, k, q, lam)[2]
        if energy < best_e:
            best_x, best_e = x, energy
    return best_x, best_e


def build_program(biases, edges, weights, beta: float):
    """Colour the interaction graph and build the THRML sampling program.

    Spins of the same colour share no edge, so each colour class is a conditionally independent free block.
    """
    n = len(biases)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edges)
    coloring = nx.coloring.greedy_color(graph, strategy="largest_first")

    nodes = [SpinNode() for _ in range(n)]
    n_colors = max(coloring.values()) + 1
    free_blocks = [Block([nodes[i] for i in range(n) if coloring[i] == c]) for c in range(n_colors)]

    ebm = IsingEBM(
        nodes,
        [(nodes[i], nodes[j]) for i, j in edges],
        jnp.asarray(biases),
        jnp.asarray(weights),
        jnp.array(beta, dtype=jnp.float32),
    )
    program = IsingSamplingProgram(ebm, free_blocks, clamped_blocks=[])
    return program, free_blocks, nodes


def main():
    mu, sigma = make_market()
    biases, edges, weights = qubo_to_ising(mu, sigma, K_ASSETS, RISK_AVERSION, PENALTY)

    exact_x, exact_e = brute_force(mu, sigma, K_ASSETS, RISK_AVERSION, PENALTY)
    print(f"Brute force over C({N_ASSETS},{K_ASSETS}) portfolios: assets {np.flatnonzero(exact_x).tolist()}")
    print(f"Brute-force optimum E = {exact_e:.4f}")

    # Warm start from a random portfolio with the target cardinality.
    rng = np.random.default_rng(42)
    current = np.zeros(N_ASSETS, dtype=bool)
    current[rng.choice(N_ASSETS, size=K_ASSETS, replace=False)] = True

    key = jax.random.key(777)
    best_x, best_e = None, np.inf  # best *feasible* sample seen over the whole schedule
    feasibility = []

    start = time.perf_counter()
    for beta in BETAS:
        key, subkey = jax.random.split(key)
        program, free_blocks, nodes = build_program(biases, edges, weights, beta)

        position = {node: i for i, node in enumerate(nodes)}
        init_state = [
            jnp.array([bool(current[position[node]]) for node in block], dtype=jnp.bool_) for block in free_blocks
        ]
        schedule = SamplingSchedule(n_warmup=N_WARMUP, n_samples=N_SAMPLES, steps_per_sample=1)
        samples = np.array(sample_states(subkey, program, schedule, init_state, [], [Block(nodes)])[0], dtype=np.int32)

        energies = np.array([objective(x, mu, sigma, K_ASSETS, RISK_AVERSION, PENALTY)[2] for x in samples])
        feasible = samples.sum(axis=1) == K_ASSETS
        feasibility.append(100.0 * feasible.mean())
        print(
            f"beta = {beta:5.2f} | feasible (sum x = {K_ASSETS}): {feasibility[-1]:5.1f}% "
            f"| mean E = {energies.mean():+.4f} | min E = {energies.min():+.4f}"
        )

        for x, e, ok in zip(samples, energies, feasible):
            if ok and e < best_e:
                best_x, best_e = x.copy(), e

        # Next (colder) stage starts from the best feasible state found so far.
        if best_x is not None:
            current = best_x.astype(bool)
        else:
            current = samples[int(np.argmin(energies))].astype(bool)
    elapsed = time.perf_counter() - start

    # Match = sampled_E / exact_E, a ratio of objective values (both negative here), not a success rate.
    match = 100.0 * best_e / exact_e
    print(f"\nSampling time: {elapsed:.2f}s (includes JIT compilation)")
    print(
        f"Sampled portfolio: assets {np.flatnonzero(best_x).tolist()} (sum x = {int(best_x.sum())}), E = {best_e:.4f}"
    )
    print(f"Final-stage budget feasibility: {feasibility[-1]:.1f}%")
    print(f"Objective match vs. brute force: {match:.1f}% ({best_e:.4f} / {exact_e:.4f})")


if __name__ == "__main__":
    main()
