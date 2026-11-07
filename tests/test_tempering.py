import jax
import jax.numpy as jnp

from thrml import Block, SpinNode, make_empty_block_state
from thrml.models import IsingEBM, IsingSamplingProgram
from thrml.tempering import parallel_tempering


def _tiny_ising():
    # 2x2 torus
    grid = [[SpinNode() for _ in range(2)] for _ in range(2)]
    nodes = [n for row in grid for n in row]
    edges = []
    for i in range(2):
        for j in range(2):
            n = grid[i][j]
            edges.append((n, grid[i][(j + 1) % 2]))  # right
            edges.append((n, grid[(i + 1) % 2][j]))  # down
    # Two-coloring
    even_nodes = [grid[i][j] for i in range(2) for j in range(2) if (i + j) % 2 == 0]
    odd_nodes = [grid[i][j] for i in range(2) for j in range(2) if (i + j) % 2 == 1]
    free_blocks = [Block(even_nodes), Block(odd_nodes)]
    return nodes, edges, free_blocks


def test_parallel_tempering_smoke():
    nodes, edges, free_blocks = _tiny_ising()
    biases = jnp.zeros((len(nodes),))
    weights = jnp.zeros((len(edges),))

    # Two temperatures; energies are zero so swaps should always accept
    ebm_cold = IsingEBM(nodes, edges, biases, weights, jnp.array(1.0))
    ebm_hot = IsingEBM(nodes, edges, biases, weights, jnp.array(0.5))
    programs = [
        IsingSamplingProgram(ebm_cold, free_blocks, clamped_blocks=[]),
        IsingSamplingProgram(ebm_hot, free_blocks, clamped_blocks=[]),
    ]

    init_state = make_empty_block_state(free_blocks, ebm_cold.node_shape_dtypes)
    init_states = [init_state, init_state]

    @jax.jit
    def run(key, init_states):
        return parallel_tempering(
            key,
            [ebm_cold, ebm_hot],
            programs,
            init_states,
            clamp_state=[],
            n_rounds=2,
            gibbs_steps_per_round=1,
        )

    key = jax.random.key(0)
    final_states, sampler_states, stats = run(key, init_states)

    assert len(final_states) == 2
    assert len(sampler_states) == 2

    # One adjacent pair
    assert stats["accepted"].shape == (1,)
    assert stats["attempted"].shape == (1,)
    assert stats["acceptance_rate"].shape == (1,)
    assert stats["accepted"][0] == 1
    assert stats["attempted"][0] == 1
    assert stats["acceptance_rate"][0] == 1.0
