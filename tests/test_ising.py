import unittest

import equinox as eqx
import jax
from jax import numpy as jnp

from thrml.block_management import Block
from thrml.block_sampling import SamplingSchedule, sample_with_observation
from thrml.models.ising import (
    IsingEBM,
    IsingSamplingProgram,
    IsingTrainingSpec,
    estimate_kl_grad,
    estimate_moments,
    hinton_init,
)
from thrml.observers import StateObserver
from thrml.pgm import SpinNode

from .utils import sample_and_compare_distribution


class TestLine(unittest.TestCase):
    def test_sample(self):
        """
        Quick check that sampling produces samples from the Boltzmann distribution. There is basically no way
        for this to not work assuming the discrete EBM tests are passing.
        """

        dim = 5
        key = jax.random.key(43524)
        nodes = [SpinNode() for _ in range(dim)]
        edges = [(x, y) for x, y in zip(nodes[:-1], nodes[1:])]
        key, subkey = jax.random.split(key)
        biases = jax.random.uniform(subkey, (dim,), minval=-1.0, maxval=1.0)
        key, subkey = jax.random.split(key)
        weights = jax.random.uniform(subkey, (len(edges),), minval=-1.0, maxval=1.0)

        ebm = IsingEBM(nodes, edges, biases, weights, jnp.array(1.0))

        rng_key = key
        schedule = SamplingSchedule(n_warmup=1000, n_samples=10_000, steps_per_sample=5)

        free_blocks = [Block(nodes[1::2]), Block(nodes[2::2])]  # indices 0,2
        clamped_blocks = [Block([nodes[0]])]

        # single clamped block of size 2 => we pass a single array of shape(2,)
        clamp_vals = [
            jnp.array([True], dtype=jnp.bool_),
        ]

        program = IsingSamplingProgram(ebm, free_blocks, clamped_blocks)

        emp_dist, exact_dist = sample_and_compare_distribution(rng_key, ebm, program, clamp_vals, schedule, 0)

        max_err = jnp.max(jnp.abs(emp_dist - exact_dist)) / jnp.max(exact_dist)
        self.assertLess(max_err, 0.02, f"Distribution mismatch (clamped): {max_err}")


def _random_complete_graph(key, dim):
    key, subkey = jax.random.split(key, 2)
    weight_matrix = jax.random.uniform(subkey, shape=(dim, dim), minval=-0.6, maxval=0.6)
    key, subkey = jax.random.split(key, 2)
    biases = jax.random.uniform(subkey, shape=(dim,), minval=-1, maxval=1)
    nodes = [SpinNode() for _ in range(dim)]
    edges = []
    edge_weights = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            key, subkey = jax.random.split(key, 2)
            make_edge = jax.random.bernoulli(subkey, p=0.7)
            make_edge = make_edge or (i == 0 and j == 1)  # at least one edge
            if make_edge:
                edges.append((nodes[i], nodes[j]))
                edge_weights.append(weight_matrix[i, j])
    return nodes, edges, biases, jnp.array(edge_weights)


class TestMomentAccumulator(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dim = 6
        key = jax.random.key(42)
        key, subkey = jax.random.split(key, 2)
        nodes, edges, biases, weights = _random_complete_graph(
            subkey,
            dim,
        )

        self.ebm = IsingEBM(nodes, edges, biases, weights, jnp.array(1.0))

        schedule = SamplingSchedule(
            n_warmup=10,
            n_samples=1000,
            steps_per_sample=1,
        )

        rng_key = key

        free_block = tuple(Block([n]) for n in self.ebm.nodes)

        program = IsingSamplingProgram(self.ebm, free_block, [])

        free_data = hinton_init(rng_key, self.ebm, free_block, ())

        self.first_moments, self.second_moments = estimate_moments(
            rng_key, self.ebm.nodes, self.ebm.edges, program, schedule, free_data, []
        )

        state_observe = StateObserver([Block(self.ebm.nodes)])
        carry_init = state_observe.init()
        _, samples = sample_with_observation(rng_key, program, schedule, free_data, [], carry_init, state_observe)

        self.samples = jnp.array(2) * samples[0].astype(jnp.int8) - jnp.array(1)

    def test_first_moments(self):
        avg_node_vals = jnp.mean(self.samples, axis=0)
        self.assertTrue(jnp.allclose(self.first_moments, avg_node_vals, atol=1e-6))

    def test_second_moments(self):
        avg_edge_vals = []
        node_map = {node: idx for idx, node in enumerate(self.ebm.nodes)}
        for edge in self.ebm.edges:
            avg_edge_vals.append(jnp.mean(self.samples[:, node_map[edge[0]]] * self.samples[:, node_map[edge[1]]]))
        avg_edge_vals = jnp.array(avg_edge_vals)
        self.assertTrue(jnp.allclose(self.second_moments, avg_edge_vals, atol=1e-6))


def _all_bitstrings(N):
    num_bitstrings = 1 << N  # Calculate 2^N
    indices = jnp.arange(num_bitstrings, dtype=jnp.int64)  # Generate numbers from 0 to 2^N - 1
    bits = ((indices[:, None] >> jnp.arange(N)) & 1).astype(jnp.int8).astype(jnp.bool)
    bits = bits[:, ::-1]  # Reverse to maintain correct bit order
    return bits


class TestEstimateKLGrad(unittest.TestCase):
    def test_estimate_kl_grad(self):
        key = jax.random.key(44)

        beta = jnp.array(1.0)
        nodes = [SpinNode() for _ in range(4)]
        data_nodes = [nodes[0], nodes[2]]
        latent_nodes = [nodes[1], nodes[3]]
        edges = [(a, b) for a, b in zip(nodes[:-1], nodes[1:])]

        key, subkey = jax.random.split(key, 2)
        biases = jax.random.uniform(subkey, (4,))

        key, subkey = jax.random.split(key, 2)
        weights = jax.random.uniform(subkey, (3,))

        model = IsingEBM(nodes, edges, biases, weights, beta)

        positive_sampling_blocks = [Block(latent_nodes)]
        negative_sampling_blocks = [Block(latent_nodes), Block(data_nodes)]

        data = [_all_bitstrings(2)[:1, :]]

        batch_size = 1000

        schedule_positive = SamplingSchedule(n_warmup=1000, n_samples=1000, steps_per_sample=10)
        schedule_negative = SamplingSchedule(n_warmup=1000, n_samples=1000, steps_per_sample=10)

        training_spec = IsingTrainingSpec(
            model,
            [Block(data_nodes)],
            [],
            positive_sampling_blocks,
            negative_sampling_blocks,
            schedule_positive,
            schedule_negative,
        )

        key, subkey = jax.random.split(key, 2)
        init_state_positive = hinton_init(subkey, model, positive_sampling_blocks, (batch_size, data[0].shape[0]))
        key, subkey = jax.random.split(key, 2)
        init_state_negative = hinton_init(subkey, model, negative_sampling_blocks, (batch_size,))

        grad_w, grad_b, _, _ = estimate_kl_grad(
            key, training_spec, model.nodes, model.edges, data, [], init_state_positive, init_state_negative
        )

        all_bs = _all_bitstrings(4)

        def compute_kl(m):
            energies = jax.vmap(lambda x: m.energy([x], [Block(data_nodes + latent_nodes)]))(all_bs)
            unnorm_prob = jnp.exp(-beta * energies)
            fold_un_prob = jnp.reshape(unnorm_prob, (2, 2, 2, 2))
            marginal = jnp.sum(fold_un_prob, axis=(2, 3))

            norm_prob = marginal / jnp.sum(marginal)

            return jnp.log(1 / norm_prob[0, 0])

        # this should match our MC grad
        val, grad = eqx.filter_value_and_grad(compute_kl)(model)

        error_w = jnp.max(jnp.abs(grad_w - grad.weights)) / jnp.max(jnp.abs(grad_w))
        error_b = jnp.max(jnp.abs(grad_b - grad.biases)) / jnp.max(jnp.abs(grad_b))

        self.assertLess(error_w, 0.01)
        self.assertLess(error_b, 0.01)
