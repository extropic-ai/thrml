import itertools
import random
import time
import unittest

import jax
import networkx as nx
import numpy as np
from jax import numpy as jnp

from thrml.block_management import Block
from thrml.block_sampling import (
    BlockGibbsSpec,
    SamplingSchedule,
    sample_single_block,
    sample_states,
)
from thrml.factor import FactorSamplingProgram
from thrml.models.discrete_ebm import (
    CategoricalEBMFactor,
    CategoricalGibbsConditional,
    DiscreteEBMFactor,
    SpinEBMFactor,
    SpinGibbsConditional,
    SquareCategoricalEBMFactor,
    SquareDiscreteEBMFactor,
)
from thrml.models.ebm import FactorizedEBM
from thrml.pgm import AbstractNode, CategoricalNode, SpinNode

from .utils import (
    count_samples,
    generate_all_states_binary,
    sample_and_compare_distribution,
)


class TestFactor(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fac_size = 4

        self.bin_nodes_good = [Block([SpinNode() for _ in range(self.fac_size)])]
        self.cat_nodes_good = [Block([CategoricalNode() for _ in range(self.fac_size)])]

        self.weights_good = jnp.zeros((self.fac_size, 1))

    def test_good(self):
        _ = DiscreteEBMFactor(self.bin_nodes_good, self.cat_nodes_good, self.weights_good)

    def test_wrong_n_cat(self):
        weights_bad = jnp.zeros((self.fac_size, 1, 1))
        with self.assertRaises(RuntimeError) as error:
            _ = DiscreteEBMFactor(self.bin_nodes_good, self.cat_nodes_good, weights_bad)

        self.assertIn("weight tensor", str(error.exception))

    def test_duplicated_type(self):
        with self.assertRaises(RuntimeError) as error:
            _ = DiscreteEBMFactor(self.bin_nodes_good, self.bin_nodes_good, self.weights_good)

        self.assertIn("categorical and spin", str(error.exception))


class TestSamplerType(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        nodes = [SpinNode(), SpinNode(), CategoricalNode()]

        weights = jnp.zeros((1, 3), dtype=jnp.float32)

        self.fac = DiscreteEBMFactor([Block([x]) for x in nodes[:-1]], [Block([nodes[-1]])], weights)

        self.free_blocks = [Block([nodes[0]])]
        self.clamped_blocks = [Block([nodes[1]]), Block([nodes[2]])]
        self.sampler = SpinGibbsConditional()

        self.key = jax.random.key(3434)

        self.good_bin_type = jax.ShapeDtypeStruct(shape=(), dtype=jnp.bool)
        self.good_cat_type = jax.ShapeDtypeStruct(shape=(), dtype=jnp.uint8)

    def test_good(self):
        node_sd = {SpinNode: self.good_bin_type, CategoricalNode: self.good_cat_type}

        spec = BlockGibbsSpec(self.free_blocks, self.clamped_blocks, node_sd)

        prog = FactorSamplingProgram(spec, [self.sampler], [self.fac], [])

        state_free = [jax.random.bernoulli(self.key, 0.5, (1,)).astype(jnp.bool)]
        state_clamped = [
            jax.random.bernoulli(self.key, 0.5, (1,)).astype(jnp.bool),
            jax.random.randint(self.key, (1,), minval=0, maxval=3, dtype=jnp.uint8),
        ]

        sample_single_block(self.key, state_free, state_clamped, prog, 0, None)

    def test_bad_bin(self):
        node_sd = {SpinNode: jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32), CategoricalNode: self.good_cat_type}

        spec = BlockGibbsSpec(self.free_blocks, self.clamped_blocks, node_sd)

        prog = FactorSamplingProgram(spec, [self.sampler], [self.fac], [])

        state_free = [jax.random.bernoulli(self.key, 0.5, (1,)).astype(jnp.float32)]
        state_clamped = [
            jax.random.bernoulli(self.key, 0.5, (1,)).astype(jnp.float32),
            jax.random.randint(self.key, (1,), minval=0, maxval=3, dtype=jnp.uint8),
        ]

        with self.assertRaises(RuntimeError) as error:
            _ = sample_single_block(self.key, state_free, state_clamped, prog, 0, None)

        self.assertIn("bool", str(error.exception))

    def test_bad_cat(self):
        node_sd = {SpinNode: self.good_bin_type, CategoricalNode: jax.ShapeDtypeStruct(shape=(), dtype=jnp.int8)}

        spec = BlockGibbsSpec(self.free_blocks, self.clamped_blocks, node_sd)

        prog = FactorSamplingProgram(spec, [self.sampler], [self.fac], [])

        state_free = [jax.random.bernoulli(self.key, 0.5, (1,)).astype(jnp.bool)]
        state_clamped = [
            jax.random.bernoulli(self.key, 0.5, (1,)).astype(jnp.bool),
            jax.random.randint(self.key, (1,), minval=0, maxval=3, dtype=jnp.int8),
        ]

        with self.assertRaises(RuntimeError) as error:
            _ = sample_single_block(self.key, state_free, state_clamped, prog, 0, None)

        self.assertIn("unsigned", str(error.exception))


class TestSquare(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        block_len = 4
        n_cat = 3

        self.blocks = [Block([CategoricalNode() for _ in range(block_len)]) for _ in range(n_cat)]

        self.good_weights = jnp.zeros((block_len, 5, 5, 5))

    def test_good(self):
        _ = SquareDiscreteEBMFactor([], self.blocks, self.good_weights)

    def test_bad(self):
        bad_weights = jnp.zeros((self.good_weights.shape[0], 5, 3, 1))

        with self.assertRaises(RuntimeError) as error:
            _ = SquareDiscreteEBMFactor([], self.blocks, bad_weights)

        self.assertIn("square", str(error.exception))


class TestSampling(unittest.TestCase):
    def test_binary(self):
        """Test that a simple sampling program involving only binary variables produces samples that follow the
        Boltzmann distribution."""

        key = jax.random.key(4243)
        nodes = [SpinNode() for _ in range(5)]
        extra_node = SpinNode()

        key, subkey = jax.random.split(key, 2)
        biases = jax.random.normal(subkey, (len(nodes),))
        b_fac = SpinEBMFactor([Block(nodes)], biases)

        key, subkey = jax.random.split(key, 2)
        weights = jax.random.normal(subkey, (len(nodes) - 1,))
        w_fac = SpinEBMFactor([Block(nodes[:-1]), Block(nodes[1:])], weights)

        key, subkey = jax.random.split(key, 2)
        triplet_weight = jax.random.normal(subkey, (1,))
        t_fac = SpinEBMFactor([Block([nodes[2]]), Block([nodes[3]]), Block([extra_node])], triplet_weight)

        ebm = FactorizedEBM([b_fac, w_fac, t_fac])

        free_blocks = [Block(nodes[1::2]), Block(nodes[2::2])]

        clamped_blocks = [Block([nodes[0], extra_node])]

        gibbs_spec = BlockGibbsSpec(free_blocks, clamped_blocks)

        samp = SpinGibbsConditional()

        prog = FactorSamplingProgram(gibbs_spec, [samp, samp], [b_fac, w_fac, t_fac], [])

        clamp_vals = [jnp.array([False, True], dtype=bool)]

        sched = SamplingSchedule(5, 10000, 5)

        empirical, exact = sample_and_compare_distribution(key, ebm, prog, clamp_vals, sched, 0)

        error = jnp.max(jnp.abs((empirical - exact))) / jnp.max(jnp.abs(exact))

        self.assertLess(error, 0.02)

    def test_categorical(self):
        """Test that a simple sampling program involving only categorical variables produces samples that follow the
        Boltzmann distribution."""

        n_cats = 3

        key = jax.random.key(443)
        nodes = [CategoricalNode() for _ in range(5)]
        extra_node = CategoricalNode()

        key, subkey = jax.random.split(key, 2)
        biases = jax.random.normal(subkey, (len(nodes), n_cats))
        b_fac = CategoricalEBMFactor([Block(nodes)], biases)

        key, subkey = jax.random.split(key, 2)
        weights = jax.random.normal(subkey, (len(nodes) - 1, n_cats, n_cats))
        w_fac = CategoricalEBMFactor([Block(nodes[:-1]), Block(nodes[1:])], weights)

        key, subkey = jax.random.split(key, 2)
        triplet_weight = jax.random.normal(subkey, (1, n_cats, n_cats, n_cats))
        t_fac = SquareCategoricalEBMFactor([Block([nodes[2]]), Block([nodes[3]]), Block([extra_node])], triplet_weight)

        ebm = FactorizedEBM(
            [
                b_fac,
                w_fac,
                t_fac,
            ]
        )  # t_fac])

        free_blocks = [Block(nodes[1::2]), Block(nodes[2::2])]

        clamped_blocks = [Block([nodes[0], extra_node])]

        gibbs_spec = BlockGibbsSpec(free_blocks, clamped_blocks)

        samp = CategoricalGibbsConditional(n_cats)

        prog = FactorSamplingProgram(gibbs_spec, [samp, samp], [b_fac, w_fac, t_fac], [])

        clamp_vals = [jnp.array([1, 1], dtype=jnp.uint8)]

        sched = SamplingSchedule(5, 10000, 5)

        empirical, exact = sample_and_compare_distribution(key, ebm, prog, clamp_vals, sched, n_cats)

        error = jnp.max(jnp.abs((empirical - exact))) / jnp.max(jnp.abs(exact))

        self.assertLess(error, 0.02)

    def test_mixed(self):
        """Test that a simple sampling program involving both binary and categorical variables produces samples that
        follow the Boltzmann distribution."""

        n_cats = 3

        key = jax.random.key(443)
        bin_nodes = [SpinNode() for _ in range(3)]
        cat_nodes = [CategoricalNode() for _ in range(4)]

        key, subkey = jax.random.split(key, 2)
        bin_biases = jax.random.normal(subkey, (len(bin_nodes),))
        bin_bias_fac = SpinEBMFactor([Block(bin_nodes)], bin_biases)

        key, subkey = jax.random.split(key, 2)
        cat_biases = jax.random.normal(subkey, (len(cat_nodes), n_cats))
        cat_bias_fac = CategoricalEBMFactor([Block(cat_nodes)], cat_biases)

        key, subkey = jax.random.split(key, 2)
        weights = jax.random.normal(subkey, (len(bin_nodes), n_cats))
        weight_fac = DiscreteEBMFactor([Block(bin_nodes)], [Block(cat_nodes[:-1])], weights)

        key, subkey = jax.random.split(key, 2)
        triple_weights = jax.random.normal(subkey, (1, n_cats))
        triple_weight_fac = SquareDiscreteEBMFactor(
            [Block([bin_nodes[-1]]), Block([bin_nodes[-2]])], [Block([cat_nodes[-1]])], triple_weights
        )

        free_blocks = [
            Block(bin_nodes[1::2]),
            Block(cat_nodes[0:-1:2]),
            Block(bin_nodes[::2]),
            Block(cat_nodes[1:-1:2]),
            Block([cat_nodes[-1]]),
        ]

        sched = SamplingSchedule(5, 10000, 5)

        factors = [bin_bias_fac, cat_bias_fac, weight_fac, triple_weight_fac]

        ebm = FactorizedEBM(factors)

        gibbs_spec = BlockGibbsSpec(free_blocks, [])

        samp_bin = SpinGibbsConditional()
        samp_cat = CategoricalGibbsConditional(n_cats)

        prog = FactorSamplingProgram(gibbs_spec, [samp_bin, samp_cat, samp_bin, samp_cat, samp_cat], factors, [])

        empirical, exact = sample_and_compare_distribution(key, ebm, prog, [], sched, n_cats)

        error = jnp.max(jnp.abs((empirical - exact))) / jnp.max(jnp.abs(exact))

        # larger graph so harder to converge. Error goes down if you increase number of samples
        self.assertLess(error, 0.05)


class TestInteractions(unittest.TestCase):
    def test_to_interactions(self):
        key = jax.random.key(342)

        n_cats = 3
        chain_len = 4

        block_1 = Block([SpinNode() for _ in range(chain_len)])
        block_2 = Block([CategoricalNode() for _ in range(chain_len)])
        block_3 = Block([SpinNode() for _ in range(chain_len)])
        block_4 = Block([CategoricalNode() for _ in range(chain_len)])

        weights = jax.random.normal(key, (chain_len, n_cats, n_cats))

        factor = DiscreteEBMFactor([block_1, block_3], [block_2, block_4], weights)

        groups = factor.to_interaction_groups()

        self.assertEqual(len(groups), 3)

    def test_to_interactions_binary(self):
        # make sure that all possible combinations are being covered

        key = jax.random.key(342)
        block_1 = Block([SpinNode() for _ in range(4)])
        block_2 = Block([SpinNode() for _ in range(4)])
        block_3 = Block([SpinNode() for _ in range(4)])
        weights = jax.random.normal(key, (4,))

        factor = SpinEBMFactor([block_1, block_2, block_3], weights)

        interaction = factor.to_interaction_groups()[0]

        self.assertEqual(len(interaction.head_nodes.nodes), 12)
        self.assertEqual(len(interaction.tail_nodes), 2)
        for i in range(len(interaction.tail_nodes)):
            self.assertEqual(len(interaction.tail_nodes[i].nodes), 12)
        self.assertEqual(interaction.interaction.weights.shape[0], 12)

        def validate(node, other_nodes, w, head_block: Block, tail_blocks: list[Block], weights):
            if node in head_block.nodes:
                idx = head_block.nodes.index(node)
                if weights[idx] == w:
                    return set([b.nodes[idx] for b in tail_blocks]) == set(other_nodes)
            return False

        for i, head in enumerate(interaction.head_nodes):
            other = (interaction.tail_nodes[0][i], interaction.tail_nodes[1][i])
            a = [
                validate(head, other, interaction.interaction.weights[i], block_1, (block_2, block_3), weights),
                validate(head, other, interaction.interaction.weights[i], block_2, (block_1, block_3), weights),
                validate(head, other, interaction.interaction.weights[i], block_3, (block_1, block_2), weights),
            ]
            self.assertEqual(sum(a), 1)


class TestBlockSample(unittest.TestCase):
    """Make sure that sampling is working correctly for a few simple cases"""

    def test_binary_bias(self):
        key = jax.random.key(342)

        block_len = 20

        block_1 = Block([SpinNode() for _ in range(block_len)])

        weights = jax.random.normal(key, (block_len,))

        factor = SpinEBMFactor([block_1], weights)

        gibbs_spec = BlockGibbsSpec([block_1], [])

        samp = SpinGibbsConditional()

        prog = FactorSamplingProgram(gibbs_spec, [samp], [factor], [])

        state = []
        for i in range(3):
            key, subkey = jax.random.split(key, 2)
            state.append(jax.random.bernoulli(key, 0.5, block_len))

        k, _ = jax.random.split(key, 2)

        samps_true = jax.random.bernoulli(k, jax.nn.sigmoid(2 * weights))

        samps = sample_single_block(key, state[:-1], [state[-1]], prog, 0, None)[0]

        self.assertTrue(np.allclose(samps_true, samps))

    def test_categorical_bias(self):
        key = jax.random.key(342)

        n_cats = 3

        block_len = 20

        block_1 = Block([CategoricalNode() for _ in range(block_len)])

        weights = jax.random.normal(key, (block_len, n_cats))

        factor = CategoricalEBMFactor([block_1], weights)

        gibbs_spec = BlockGibbsSpec([block_1], [])

        samp = CategoricalGibbsConditional(n_cats)

        prog = FactorSamplingProgram(gibbs_spec, [samp], [factor], [])

        state = []
        for i in range(3):
            key, subkey = jax.random.split(key, 2)
            state.append(jax.random.randint(key, (block_len,), minval=0, maxval=n_cats))

        k, _ = jax.random.split(key, 2)

        samps_true = jax.random.categorical(k, weights)

        samps = sample_single_block(key, state[:-1], [state[-1]], prog, 0, None)[0]

        self.assertTrue(np.allclose(samps_true, samps))

    def test_categorical_triplet(self):
        """Test sampling for the case of a three-body categorical interaction where two of the nodes in each
        interaction are clamped. This makes sure all the slicing code is working properly."""

        key = jax.random.key(342)

        n_cats = 3

        block_len = 20

        blocks = [Block([CategoricalNode() for _ in range(block_len)]) for i in range(3)]

        weights = jax.random.normal(key, (block_len, 3, 3, 3))

        factor = CategoricalEBMFactor(blocks, weights)

        samp = CategoricalGibbsConditional(n_cats)

        free_state = jax.random.randint(key, (block_len,), minval=0, maxval=n_cats, dtype=jnp.uint8)

        clamped_state = []

        for i in range(2):
            key, subkey = jax.random.split(key, 2)

            clamped_state.append(jax.random.randint(subkey, (block_len,), minval=0, maxval=n_cats, dtype=jnp.uint8))

        key, subkey = jax.random.split(key, 2)

        k, _ = jax.random.split(subkey, 2)

        sl_1 = jnp.expand_dims(clamped_state[0], (-1, -2, -3))
        sl_2 = jnp.expand_dims(clamped_state[1], (-1, -2, -3))

        # just manually enumerate each possible permutation and test that we get the right answer

        weight_sl_1 = jnp.squeeze(jnp.take_along_axis(jnp.take_along_axis(weights, sl_1, -2), sl_2, -1), (-1, -2))
        spec_1 = BlockGibbsSpec([blocks[0]], blocks[1:])
        prog_1 = FactorSamplingProgram(spec_1, [samp], [factor], [])
        samps_1 = sample_single_block(subkey, [free_state], clamped_state, prog_1, 0, None)
        true_samps_1 = jax.random.categorical(k, weight_sl_1, axis=-1)

        self.assertTrue(np.all(np.equal(samps_1[0], true_samps_1)))

        weight_sl_2 = jnp.squeeze(jnp.take_along_axis(jnp.take_along_axis(weights, sl_1, 1), sl_2, -1), (-1, 1))
        spec_2 = BlockGibbsSpec([blocks[1]], [blocks[0], blocks[-1]])
        prog_2 = FactorSamplingProgram(spec_2, [samp], [factor], [])
        samps_2 = sample_single_block(subkey, [free_state], clamped_state, prog_2, 0, None)
        true_samps_2 = jax.random.categorical(k, weight_sl_2, axis=-1)
        self.assertTrue(np.all(np.equal(samps_2[0], true_samps_2)))

        weight_sl_3 = jnp.squeeze(jnp.take_along_axis(jnp.take_along_axis(weights, sl_1, 1), sl_2, 2), (1, 2))
        spec_3 = BlockGibbsSpec([blocks[2]], [blocks[0], blocks[1]])
        prog_3 = FactorSamplingProgram(spec_3, [samp], [factor], [])
        samps_3 = sample_single_block(subkey, [free_state], clamped_state, prog_3, 0, None)
        true_samps_3 = jax.random.categorical(k, weight_sl_3, axis=-1)
        self.assertTrue(np.all(np.equal(samps_3[0], true_samps_3)))

    def test_ragged_mixed(self):
        """Test that a mixed sampling problem with ragged interactions produces the right output."""

        n_cats = 7

        key = jax.random.key(34233434)

        binary_free_nodes = [SpinNode() for _ in range(2)]

        cat_free_nodes = [CategoricalNode() for _ in range(2)]

        binary_clamped_nodes = [SpinNode() for _ in range(2)]
        cat_clamped_nodes = [CategoricalNode() for _ in range(2)]

        # this time the graph is kind of small so repeat it a bunch of times to make sure passing isn't a fluke

        for i in range(50):
            key, subkey = jax.random.split(key, 2)
            b_fac_1 = SpinEBMFactor(
                [Block([binary_free_nodes[0]]), Block([binary_clamped_nodes[0]])], jax.random.normal(subkey, (1,))
            )

            key, subkey = jax.random.split(key, 2)
            b_fac_2 = DiscreteEBMFactor(
                [Block([binary_free_nodes[1]])],
                [Block([cat_clamped_nodes[0]]), Block([cat_clamped_nodes[1]])],
                jax.random.normal(subkey, (1, n_cats, n_cats)),
            )

            key, subkey = jax.random.split(key, 2)
            cat_fac_1 = CategoricalEBMFactor(
                [Block([cat_clamped_nodes[0]]), Block([cat_clamped_nodes[1]]), Block([cat_free_nodes[0]])],
                jax.random.normal(subkey, (1, n_cats, n_cats, n_cats)),
            )

            key, subkey = jax.random.split(key, 2)
            cat_fac_2 = DiscreteEBMFactor(
                [Block([binary_clamped_nodes[0]]), Block([binary_clamped_nodes[1]])],
                [Block([cat_free_nodes[1]])],
                jax.random.normal(
                    subkey,
                    (
                        1,
                        n_cats,
                    ),
                ),
            )

            factors = [b_fac_1, b_fac_2, cat_fac_1, cat_fac_2]

            free_blocks = [Block(binary_free_nodes), Block(cat_free_nodes)]
            clamped_blocks = [Block(binary_clamped_nodes), Block(cat_clamped_nodes)]

            spec = BlockGibbsSpec(free_blocks, clamped_blocks)

            samplers = [SpinGibbsConditional(), CategoricalGibbsConditional(n_cats)]

            prog = FactorSamplingProgram(spec, samplers, factors, [])

            free_state = [
                jax.random.bernoulli(key, 0.5, (2,)).astype(jnp.bool),
                jax.random.randint(key, (2,), minval=0, maxval=n_cats, dtype=jnp.uint8),
            ]

            key, subkey = jax.random.split(key, 2)

            clamp_state = [
                jax.random.bernoulli(key, 0.5, (2,)).astype(jnp.bool),
                jax.random.randint(key, (2,), minval=0, maxval=n_cats, dtype=jnp.uint8),
            ]

            k, _ = jax.random.split(subkey, 2)

            weights_bin = jnp.array(
                [
                    (2 * clamp_state[0][0].astype(jnp.int8) - 1).astype(b_fac_1.weights[0].dtype) * b_fac_1.weights[0],
                    b_fac_2.weights[0, clamp_state[1][0], clamp_state[1][1]],
                ]
            )

            actual_samps_bin = jax.random.bernoulli(k, jax.nn.sigmoid(2 * weights_bin))

            weights_cat = jnp.array(
                [
                    cat_fac_1.weights[0, clamp_state[1][0], clamp_state[1][1], :],
                    (2 * clamp_state[0][0].astype(jnp.int8) - 1).astype(cat_fac_2.weights[0].dtype)
                    * (2 * clamp_state[0][1].astype(jnp.int8) - 1).astype(cat_fac_2.weights[0].dtype)
                    * cat_fac_2.weights[0],
                ]
            )

            actual_samps_cat = jax.random.categorical(k, weights_cat)

            samples_bin = sample_single_block(subkey, free_state, clamp_state, prog, 0, None)[0]
            samples_cat = sample_single_block(subkey, free_state, clamp_state, prog, 1, None)[0]

            self.assertTrue(np.all(np.equal(samples_bin, actual_samps_bin)))
            self.assertTrue(np.all(np.equal(samples_cat, actual_samps_cat)))


class TestEnergy(unittest.TestCase):
    def test_bin(self):
        """Test that energy function evaluation produces the right value for a simple binary model."""

        chain_len = 20

        key = jax.random.key(443)

        blocks = [Block([SpinNode() for _ in range(chain_len)]) for _ in range(3)]

        key, subkey = jax.random.split(key, 2)

        weights = jax.random.normal(subkey, (chain_len,))

        factor = SpinEBMFactor([blocks[-1], blocks[0], blocks[1]], weights)

        ebm = FactorizedEBM([factor])

        state = []

        for _ in range(3):
            key, subkey = jax.random.split(key, 2)
            state.append(jax.random.bernoulli(subkey, 0.5, (chain_len,)))

        e = ebm.energy(state, blocks)

        true_energy = -jnp.sum(
            weights * jnp.prod(2 * jnp.stack(state, axis=-1).astype(jnp.int8) - 1, axis=-1).astype(weights.dtype)
        )

        self.assertTrue(np.allclose(e, true_energy, rtol=1e-6))

    def test_cat(self):
        """Test that energy function evaluation produces the right value for a simple categorical model"""

        n_cats = 3
        chain_len = 20

        key = jax.random.key(443)

        blocks = [Block([CategoricalNode() for _ in range(chain_len)]) for _ in range(3)]

        key, subkey = jax.random.split(key, 2)

        weights = jax.random.normal(subkey, (chain_len, n_cats, n_cats, n_cats))

        factor = CategoricalEBMFactor([blocks[-1], blocks[0], blocks[1]], weights)

        ebm = FactorizedEBM([factor])

        state = []

        for _ in range(3):
            key, subkey = jax.random.split(key, 2)
            state.append(jax.random.randint(subkey, (chain_len,), minval=0, maxval=n_cats, dtype=jnp.uint8))

        e = ebm.energy(state, blocks)

        sl_states = [jnp.expand_dims(x, (-1, -2, -3)) for x in state]

        true_energy = -jnp.sum(
            jnp.take_along_axis(
                jnp.take_along_axis(jnp.take_along_axis(weights, sl_states[-1], axis=1), sl_states[0], axis=2),
                sl_states[1],
                axis=3,
            )
        )

        self.assertTrue(np.allclose(e, true_energy, rtol=1e-6))

    def test_mixed(self):
        """Test that energy function evaluation produces the right  value for a mixed discrete model."""

        n_cats = 3
        chain_len = 20

        key = jax.random.key(443)

        cat_blocks = [Block([CategoricalNode() for _ in range(chain_len)]) for _ in range(2)]
        bin_blocks = [Block([SpinNode() for _ in range(chain_len)])]

        key, subkey = jax.random.split(key, 2)

        weights = jax.random.normal(subkey, (chain_len, n_cats, n_cats))

        factor = DiscreteEBMFactor(bin_blocks, cat_blocks[::-1], weights)

        ebm = FactorizedEBM([factor])

        state = []

        for _ in range(2):
            key, subkey = jax.random.split(key, 2)
            state.append(jax.random.randint(subkey, (chain_len,), minval=0, maxval=n_cats, dtype=jnp.uint8))

        state.append(jax.random.bernoulli(subkey, 0.5, (chain_len,)))

        e = ebm.energy(state, cat_blocks + bin_blocks)

        sl_states = [
            jnp.expand_dims(
                x,
                (
                    -1,
                    -2,
                ),
            )
            for x in state[:-1]
        ]

        true_energy = -jnp.sum(
            (2 * state[-1].astype(jnp.int8) - 1).astype(jnp.float32)
            * jnp.squeeze(
                jnp.take_along_axis(jnp.take_along_axis(weights, sl_states[-1], axis=1), sl_states[0], axis=2), (-2, -1)
            )
        )

        self.assertTrue(np.allclose(e, true_energy, rtol=1e-6))


def _generate_grid_graph(side_lengths, node_type):
    if len(side_lengths) == 0:
        raise ValueError("At least one side length must be provided.")
    if any(d <= 0 for d in side_lengths):
        raise ValueError("All side lengths must be positive integers.")

    bases = [1]
    for d in side_lengths[::-1][:-1]:
        bases.append(bases[-1] * d)
    bases = bases[::-1]

    def flat_index(coords):
        return sum(c * b for c, b in zip(coords, bases))

    total_nodes = 1
    for d in side_lengths:
        total_nodes *= d
    nodes = [node_type() for _ in range(total_nodes)]

    color0, color1 = [], []
    u, v = [], []

    for coords in itertools.product(*(range(d) for d in side_lengths)):
        idx = flat_index(coords)
        node = nodes[idx]

        (color0 if sum(coords) & 1 == 0 else color1).append(node)

        for axis, dim in enumerate(side_lengths):
            if coords[axis] + 1 < dim:
                nbr_coords = list(coords)
                nbr_coords[axis] += 1
                nbr_idx = flat_index(nbr_coords)
                u.append(node)
                v.append(nodes[nbr_idx])

    return (color0, color1), (u, v)


class TestEquivalence(unittest.TestCase):
    """Mathematically equivalent binary sampling programs that use the spin representation vs the categorical
    representation should produce the same results."""

    def test_equivalence(self):
        n_cats = 2

        class Node(AbstractNode):
            pass

        color_groups, edge_groups = _generate_grid_graph((3, 3), Node)

        key = jax.random.key(2232)

        categorical_weight_matrix = jax.random.normal(key, (len(edge_groups[0]), n_cats, n_cats))

        vec_a = jnp.array([1, 1], dtype=categorical_weight_matrix.dtype)
        vec_b = jnp.array([-1, 1], dtype=categorical_weight_matrix.dtype)

        first_spin_bias = 1 / 4 * jnp.einsum("...ij, i, j -> ...", categorical_weight_matrix, vec_b, vec_a)
        second_spin_bias = 1 / 4 * jnp.einsum("...ij, i, j -> ...", categorical_weight_matrix, vec_a, vec_b)
        spin_weight = 1 / 4 * jnp.einsum("...ij, i, j -> ...", categorical_weight_matrix, vec_b, vec_b)

        cat_factor = CategoricalEBMFactor([Block(x) for x in edge_groups], categorical_weight_matrix)

        spin_bias_factor = SpinEBMFactor(
            [Block(edge_groups[0] + edge_groups[1])], jnp.concatenate([first_spin_bias, second_spin_bias])
        )

        spin_weight_factor = SpinEBMFactor([Block(x) for x in edge_groups], spin_weight)

        free_blocks = [Block(x) for x in color_groups]

        spec_binary = BlockGibbsSpec(free_blocks, [], {Node: jax.ShapeDtypeStruct((), jnp.bool)})
        spec_cat = BlockGibbsSpec(free_blocks, [], {Node: jax.ShapeDtypeStruct((), jnp.uint8)})

        samp_binary = SpinGibbsConditional()
        samp_cat = CategoricalGibbsConditional(2)

        init_states = []

        for block in free_blocks:
            key, subkey = jax.random.split(key, 2)
            init_states.append(jax.random.bernoulli(subkey, 0.5, (len(block.nodes),)))

        prog_binary = FactorSamplingProgram(
            spec_binary, [samp_binary, samp_binary], [spin_bias_factor, spin_weight_factor], []
        )

        prog_cat = FactorSamplingProgram(spec_cat, [samp_cat, samp_cat], [cat_factor], [])

        schedule = SamplingSchedule(5, 25000, 5)

        all_nodes = Block(color_groups[0] + color_groups[1])

        samples_bin = sample_states(
            key=key,
            program=prog_binary,
            schedule=schedule,
            init_state_free=init_states,
            state_clamp=[],
            nodes_to_sample=[all_nodes],
        )

        samples_cat = sample_states(
            key=key,
            program=prog_cat,
            schedule=schedule,
            init_state_free=[x.astype(jnp.uint8) for x in init_states],
            state_clamp=[],
            nodes_to_sample=[all_nodes],
        )

        all_states = generate_all_states_binary(len(all_nodes.nodes))
        cat_states = jnp.empty((all_states.shape[0], 0), dtype=jnp.uint8)

        fake_cat_samples = jnp.empty((samples_cat[0].shape[0], 0), dtype=jnp.uint8)

        counts_bin = count_samples(all_states, cat_states, samples_bin[0], fake_cat_samples)

        counts_cat = count_samples(all_states, cat_states, samples_cat[0].astype(jnp.bool), fake_cat_samples)

        error = jnp.max(jnp.abs(counts_bin - counts_cat)) / jnp.max(jnp.abs(counts_bin))

        # little slow to converge because of large graph. Error goes down as number of samples is increased
        self.assertLess(error, 0.05)


class TestHeteroGrid(unittest.TestCase):
    """Test sampling on a very heterogeneous grid to triple check that padding is working correctly"""

    def test_grid(self):
        n_cats = 3

        p_bin = 0.5
        seed = 42424

        rng = random.Random(seed)

        G = nx.grid_graph(dim=(3, 3), periodic=False)

        coord_to_node = {}
        for coord in G.nodes():
            node = SpinNode() if rng.random() < p_bin else CategoricalNode()
            coord_to_node[coord] = node

        nx.relabel_nodes(G, coord_to_node, copy=False)

        colors = nx.bipartite.color(G)

        color_groups = [([], []), ([], [])]

        for node, color in colors.items():
            if isinstance(node, SpinNode):
                color_groups[color][0].append(node)
            else:
                color_groups[color][1].append(node)

        free_blocks = [[Block(x) for x in y] for y in color_groups]

        bb_edges = [[], []]
        cc_edges = [[], []]
        bc_edges = [[], []]

        for edge in G.edges:
            if isinstance(edge[0], SpinNode) and isinstance(edge[1], SpinNode):
                bb_edges[0].append(edge[0])
                bb_edges[1].append(edge[1])
            elif isinstance(edge[0], CategoricalNode) and isinstance(edge[1], CategoricalNode):
                cc_edges[0].append(edge[0])
                cc_edges[1].append(edge[1])
            elif isinstance(edge[0], SpinNode):
                bc_edges[0].append(edge[0])
                bc_edges[1].append(edge[1])
            else:
                bc_edges[1].append(edge[0])
                bc_edges[0].append(edge[1])

        key = jax.random.key(seed)

        key, subkey = jax.random.split(key, 2)
        bb_fac = SpinEBMFactor([Block(x) for x in bb_edges], jax.random.normal(subkey, (len(bb_edges[0]),)))
        key, subkey = jax.random.split(key, 2)
        cc_fac = CategoricalEBMFactor(
            [Block(x) for x in cc_edges], jax.random.normal(subkey, (len(cc_edges[0]), n_cats, n_cats))
        )
        key, subkey = jax.random.split(key, 2)
        bc_fac = DiscreteEBMFactor(
            [Block(bc_edges[0])], [Block(bc_edges[1])], jax.random.normal(subkey, (len(cc_edges[0]), n_cats))
        )

        ebm = FactorizedEBM([bb_fac, cc_fac, bc_fac])

        spec = BlockGibbsSpec(free_blocks, [])

        bin_sampler = SpinGibbsConditional()
        cat_sampler = CategoricalGibbsConditional(n_cats)

        samplers = []
        for block in spec.free_blocks:
            if isinstance(block.nodes[0], SpinNode):
                samplers.append(bin_sampler)
            else:
                samplers.append(cat_sampler)

        prog = FactorSamplingProgram(spec, samplers, ebm.factors, [])

        sched = SamplingSchedule(0, 50000, 5)

        empirical, exact = sample_and_compare_distribution(key, ebm, prog, [], sched, n_cats)

        error = jnp.max(jnp.abs((empirical - exact))) / jnp.max(jnp.abs(exact))

        # large grid so takes a while to converge
        # error goes down if you use more samples
        self.assertLess(error, 0.04)


class TestBigGrid(unittest.TestCase):
    """
    Make sure that we can compile a sampling program on a big grid.
    this test ensures that there is no part of our core pipeline that scales worse than linearly with the number
    of degrees of freedom.
    """

    def test_big(self):
        side_lens = [50, 100, 200, 400, 500]
        times = []
        for side_len in side_lens:
            g = nx.grid_graph((side_len, side_len))

            nx.relabel_nodes(g, lambda x: SpinNode(), copy=False)

            edge_groups = [[], []]

            for edge in g.edges:
                edge_groups[0].append(edge[0])
                edge_groups[1].append(edge[1])

            key = jax.random.key(424)
            key, subkey = jax.random.split(key, 2)
            fac = SpinEBMFactor([Block(x) for x in edge_groups], jax.random.normal(subkey, (len(edge_groups[0]),)))

            ebm = FactorizedEBM([fac])

            bicol = nx.bipartite.color(g)
            color0 = [n for n, c in bicol.items() if c == 0]
            color1 = [n for n, c in bicol.items() if c == 1]

            free_blocks = [Block(color0), Block(color1)]

            spec = BlockGibbsSpec(free_blocks, [])

            samp = SpinGibbsConditional()

            start_time = time.time()
            _ = FactorSamplingProgram(spec, [samp for _ in spec.free_blocks], ebm.factors, [])
            end_time = time.time()

            times.append(end_time - start_time)

        side_lens = np.array(side_lens)
        times = np.array(times)

        delta_side = (side_lens[1:] / side_lens[:-1]) ** 2
        delta_time = times[1:] / times[:-1]

        scaling_correct = delta_time < 1.1 * delta_side

        # we should try to improve the constant factors here
        self.assertTrue(np.all(scaling_correct))
