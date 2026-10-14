"""Construction-time guard: BlockSamplingProgram raises when an interaction
instance couples nodes updated in the same sampling group (issue #75).

Run with the thrml venv:
    venv-thrml/bin/python -m pytest tests/test_f1_block_check.py -q

Covers: the headline repro (must now RAISE), the legitimate neighbor
(properly blocked pair must still sample the exact correlation), and the
boundary variants: three-node block with one bad pair, superblock-grouped
pair, high-order factor clique (bad partition raises / singleton partition
constructs), free-clamped edge (must stay legal), the custom-sampler
exception exercised both ways, and the batched-bias-factor canary
(per-instance semantics must not false-positive on Ising programs).
"""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec, SamplingSchedule, sample_states
from thrml.conditional_samplers import AbstractConditionalSampler
from thrml.factor import FactorSamplingProgram
from thrml.models.discrete_ebm import (
    CategoricalEBMFactor,
    CategoricalGibbsConditional,
    DiscreteEBMFactor,
    SpinEBMFactor,
    SpinGibbsConditional,
)
from thrml.models.ising import IsingEBM, IsingSamplingProgram
from thrml.pgm import CategoricalNode, SpinNode

EXACT_CORR = float(np.tanh(1.5))  # 0.9051


def _pair_factors(n1, n2):
    fac_b = SpinEBMFactor([Block([n1, n2])], jnp.array([0.0, 0.0], dtype=jnp.float32))
    fac_e = SpinEBMFactor([Block([n1]), Block([n2])], jnp.array([1.5], dtype=jnp.float32))
    return fac_b, fac_e


def _pair_corr(blocks, n=100000):
    n1, n2 = SpinNode(), SpinNode()
    fac_b, fac_e = _pair_factors(n1, n2)
    spec = BlockGibbsSpec(blocks(n1, n2), [])
    prog = FactorSamplingProgram(spec, [SpinGibbsConditional() for _ in spec.free_blocks], [fac_b, fac_e], [])
    init = [jnp.zeros((len(b.nodes),), dtype=jnp.bool_) for b in spec.free_blocks]
    out = sample_states(
        key=jax.random.key(3), program=prog, schedule=SamplingSchedule(20, n, 1),
        init_state_free=init, state_clamp=[], nodes_to_sample=[Block([n1]), Block([n2])],
    )
    a = 2 * np.asarray(out[0]).reshape(-1).astype(float) - 1
    b = 2 * np.asarray(out[1]).reshape(-1).astype(float) - 1
    return float((a * b).mean())


def test_one_block_pair_raises():
    n1, n2 = SpinNode(), SpinNode()
    fac_b, fac_e = _pair_factors(n1, n2)
    spec = BlockGibbsSpec([Block([n1, n2])], [])
    with pytest.raises(RuntimeError, match="sampling group"):
        FactorSamplingProgram(spec, [SpinGibbsConditional()], [fac_b, fac_e], [])


def test_two_block_pair_constructs_and_correlation():
    corr = _pair_corr(lambda n1, n2: [Block([n1]), Block([n2])])
    assert abs(corr - EXACT_CORR) < 0.03, f"two-block corr {corr} vs exact {EXACT_CORR}"


def test_superblock_grouped_pair_raises():
    # Two blocks, ONE superblock: sample_blocks updates both from one snapshot.
    n1, n2 = SpinNode(), SpinNode()
    fac_b, fac_e = _pair_factors(n1, n2)
    spec = BlockGibbsSpec([(Block([n1]), Block([n2]))], [])
    with pytest.raises(RuntimeError, match="sampling group"):
        FactorSamplingProgram(
            spec, [SpinGibbsConditional(), SpinGibbsConditional()], [fac_b, fac_e], []
        )


def test_three_node_block_one_bad_pair_raises():
    m1, m2, m3 = SpinNode(), SpinNode(), SpinNode()
    fac_b = SpinEBMFactor([Block([m1, m2, m3])], jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32))
    fac_e = SpinEBMFactor([Block([m1]), Block([m2])], jnp.array([1.5], dtype=jnp.float32))
    spec = BlockGibbsSpec([Block([m1, m2, m3])], [])
    with pytest.raises(RuntimeError, match="sampling group"):
        FactorSamplingProgram(spec, [SpinGibbsConditional()], [fac_b, fac_e], [])


def _high_order_program(blocks_spec):
    """One DiscreteEBMFactor coupling 5 spins + 1 categorical."""
    K, nspin = 8, 5
    rng = np.random.default_rng(99)
    W = rng.normal(0, 0.8, size=K).astype(np.float32)
    sbias = rng.normal(0, 0.5, size=nspin).astype(np.float32)
    cbias = rng.normal(0, 0.5, size=K).astype(np.float32)
    spins = [SpinNode() for _ in range(nspin)]
    c = CategoricalNode()
    fac_main = DiscreteEBMFactor([Block([s]) for s in spins], [Block([c])], jnp.asarray(W).reshape(1, K))
    fac_sb = SpinEBMFactor([Block(spins)], jnp.asarray(sbias))
    fac_cb = CategoricalEBMFactor([Block([c])], jnp.asarray(cbias).reshape(1, K))
    free = [Block([spins[i] for i in grp]) if grp != "cat" else Block([c]) for grp in blocks_spec]
    spec = BlockGibbsSpec(free, [])
    samplers = [SpinGibbsConditional() if b.node_type is SpinNode else CategoricalGibbsConditional(K) for b in free]
    return FactorSamplingProgram(spec, samplers, [fac_main, fac_sb, fac_cb], [])


def test_high_order_factor_bad_partition_raises():
    with pytest.raises(RuntimeError, match="sampling group"):
        _high_order_program([[0, 2, 4], [1, 3], "cat"])


def test_high_order_factor_singleton_blocks_constructs():
    _high_order_program([[0], [1], [2], [3], [4], "cat"])  # must not raise


def test_free_clamped_edge_constructs():
    k1, k2 = SpinNode(), SpinNode()
    fac_b, fac_e = _pair_factors(k1, k2)
    spec = BlockGibbsSpec([Block([k1])], [Block([k2])])
    FactorSamplingProgram(spec, [SpinGibbsConditional()], [fac_b, fac_e], [])  # must not raise


def test_ising_checkerboard_constructs():
    # Canary for per-instance semantics: the bias factor batches one
    # singleton instance per node and must NOT be treated as a clique.
    nodes = [SpinNode() for _ in range(9)]
    edges = []
    for r in range(3):
        for c in range(3):
            if r < 2:
                edges.append((nodes[3 * r + c], nodes[3 * (r + 1) + c]))
            if c < 2:
                edges.append((nodes[3 * r + c], nodes[3 * r + c + 1]))
    ebm = IsingEBM(nodes, edges, jnp.zeros(9), jnp.ones(len(edges)), jnp.array(1.0))
    even = Block([nodes[i] for i in range(9) if i % 2 == 0])
    odd = Block([nodes[i] for i in range(9) if i % 2 == 1])
    IsingSamplingProgram(ebm, [even, odd], [])  # must not raise


class ExactPairSampler(AbstractConditionalSampler):
    """Test double: draws the block's two spins from their exact joint
    conditional (the block is the whole model in this test). A direct
    AbstractConditionalSampler subclass, i.e. NOT a per-node parametric
    sampler, so the F1 gate must exempt it."""

    bias: jax.Array
    weight: jax.Array

    def sample(self, key, interactions, active_flags, states, sampler_state, output_sd):
        s = jnp.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
        logits = self.bias[0] * s[:, 0] + self.bias[1] * s[:, 1] + self.weight * s[:, 0] * s[:, 1]
        idx = jax.random.categorical(key, logits)
        return s[idx] > 0, sampler_state


def test_custom_joint_sampler_exempt_and_samples_correctly():
    n1, n2 = SpinNode(), SpinNode()
    fac_b, fac_e = _pair_factors(n1, n2)
    spec = BlockGibbsSpec([Block([n1, n2])], [])
    sampler = ExactPairSampler(jnp.array([0.0, 0.0]), jnp.array(1.5))
    prog = FactorSamplingProgram(spec, [sampler], [fac_b, fac_e], [])  # must not raise
    out = sample_states(
        key=jax.random.key(3), program=prog, schedule=SamplingSchedule(20, 60000, 1),
        init_state_free=[jnp.zeros((2,), dtype=jnp.bool_)], state_clamp=[],
        nodes_to_sample=[Block([n1]), Block([n2])],
    )
    a = 2 * np.asarray(out[0]).reshape(-1).astype(float) - 1
    b = 2 * np.asarray(out[1]).reshape(-1).astype(float) - 1
    corr = float((a * b).mean())
    assert abs(corr - EXACT_CORR) < 0.03, f"custom-sampler corr {corr} vs exact {EXACT_CORR}"
    # ... and the other way: the same spec with the built-in sampler raises.
    with pytest.raises(RuntimeError, match="sampling group"):
        FactorSamplingProgram(spec, [SpinGibbsConditional()], [fac_b, fac_e], [])
