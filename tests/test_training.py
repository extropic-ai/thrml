import unittest

import equinox as eqx
import jax
import jax.numpy as jnp

from thrml.block_management import Block
from thrml.models import (
    CategoricalEBMFactor,
    EBMFactor,
    FactorizedEBM,
    IsingEBM,
    contrastive_divergence_loss,
)
from thrml.pgm import AbstractNode, CategoricalNode, SpinNode


class TestContrastiveDivergenceLoss(unittest.TestCase):
    def test_ising_gradient_matches_exact_nll_at_uniform_model(self):
        nodes = [SpinNode(), SpinNode()]
        edges = [(nodes[0], nodes[1])]
        blocks = [Block([nodes[1]]), Block([nodes[0]])]
        positive_samples = [
            jnp.array([[False], [False], [True], [True], [True]], dtype=jnp.bool_),
            jnp.array([[False], [True], [False], [False], [True]], dtype=jnp.bool_),
        ]
        negative_samples = [
            jnp.array([[False], [False], [True], [True]], dtype=jnp.bool_),
            jnp.array([[False], [True], [False], [True]], dtype=jnp.bool_),
        ]
        beta = jnp.array(1.7)

        def make_model(parameters):
            return IsingEBM(nodes, edges, parameters[:2], parameters[2:], beta)

        def loss(parameters):
            return contrastive_divergence_loss(make_model(parameters), positive_samples, negative_samples, blocks)

        def exact_nll(parameters):
            model = make_model(parameters)
            positive_energy = jax.vmap(lambda state: model.energy(state, blocks))(positive_samples).mean()
            all_state_energy = jax.vmap(lambda state: model.energy(state, blocks))(negative_samples)
            return positive_energy + jax.nn.logsumexp(-all_state_energy)

        parameters = jnp.zeros((3,))
        actual_gradient = jax.grad(loss)(parameters)
        exact_gradient = jax.grad(exact_nll)(parameters)

        self.assertTrue(jnp.allclose(actual_gradient, exact_gradient))
        self.assertTrue(jnp.allclose(jax.jit(loss)(parameters), loss(parameters)))

    def test_categorical_factor_with_reordered_blocks_and_unequal_phase_sizes(self):
        first, second = CategoricalNode(), CategoricalNode()
        weights = jnp.array([[[0.2, -0.4, 0.7], [0.1, 0.8, -0.3]]])
        model = FactorizedEBM([CategoricalEBMFactor([Block([first]), Block([second])], weights)])
        blocks = [Block([second]), Block([first])]
        positive_samples = [
            jnp.array([[1], [2], [0]], dtype=jnp.uint8),
            jnp.array([[0], [1], [1]], dtype=jnp.uint8),
        ]
        negative_samples = [
            jnp.array([[2], [0]], dtype=jnp.uint8),
            jnp.array([[1], [0]], dtype=jnp.uint8),
        ]

        actual = contrastive_divergence_loss(model, positive_samples, negative_samples, blocks)
        positive_energy = -jnp.mean(weights[0, positive_samples[1][:, 0], positive_samples[0][:, 0]])
        negative_energy = -jnp.mean(weights[0, negative_samples[1][:, 0], negative_samples[0][:, 0]])

        self.assertTrue(jnp.allclose(actual, positive_energy - negative_energy))

    def test_sample_pytrees_are_constants(self):
        class ContinuousNode(AbstractNode):
            pass

        class LinearFactor(EBMFactor):
            weights: jax.Array

            def __init__(self, node, weights):
                super().__init__([Block([node])])
                self.weights = weights

            def energy(self, global_state, block_spec):
                return -jnp.sum(global_state[0] * self.weights)

            def to_interaction_groups(self):
                return []

        node = ContinuousNode()
        factor = LinearFactor(node, jnp.array([0.5]))
        model = FactorizedEBM([factor], {ContinuousNode: jax.ShapeDtypeStruct((), jnp.float32)})
        blocks = [Block([node])]
        positive = [jnp.array([[0.1], [0.3]])]
        negative = [jnp.array([[0.5]])]

        _, model_gradient = eqx.filter_value_and_grad(
            lambda current_model: contrastive_divergence_loss(current_model, positive, negative, blocks)
        )(model)
        positive_gradient, negative_gradient = jax.grad(
            lambda pos, neg: contrastive_divergence_loss(model, pos, neg, blocks), argnums=(0, 1)
        )(positive, negative)

        expected_weight_gradient = -jnp.mean(positive[0]) + jnp.mean(negative[0])
        self.assertTrue(jnp.allclose(model_gradient._factors[0].weights, expected_weight_gradient))
        self.assertTrue(jnp.all(positive_gradient[0] == 0))
        self.assertTrue(jnp.all(negative_gradient[0] == 0))

    def test_nested_state_and_nonzero_compiled_value(self):
        class FeatureNode(AbstractNode):
            pass

        class FeatureFactor(EBMFactor):
            weights: jax.Array

            def __init__(self, node, weights):
                super().__init__([Block([node])])
                self.weights = weights

            def energy(self, global_state, block_spec):
                state = global_state[0]
                return -jnp.sum(state["features"] * self.weights[:2]) - self.weights[2] * jnp.sum(state["offset"])

            def to_interaction_groups(self):
                return []

        node = FeatureNode()
        blocks = [Block([node])]
        state_shapes = {
            FeatureNode: {
                "features": jax.ShapeDtypeStruct((2,), jnp.float32),
                "offset": jax.ShapeDtypeStruct((), jnp.float32),
            }
        }
        positive = [
            {
                "features": jnp.array([[[1.0, 0.0]], [[0.0, 2.0]]]),
                "offset": jnp.array([[0.5], [1.0]]),
            }
        ]
        negative = [
            {
                "features": jnp.array([[[0.0, 1.0]], [[1.0, 1.0]], [[2.0, 0.0]]]),
                "offset": jnp.array([[0.0], [0.5], [0.0]]),
            }
        ]

        def loss(weights, pos=positive, neg=negative):
            model = FactorizedEBM([FeatureFactor(node, weights)], state_shapes)
            return contrastive_divergence_loss(model, pos, neg, blocks)

        weights = jnp.array([0.5, -0.25, 0.75])
        eager_value, actual_gradient = jax.value_and_grad(loss)(weights)
        compiled_value = jax.jit(loss)(weights)
        positive_mean = jnp.array([0.5, 1.0, 0.75])
        negative_mean = jnp.array([1.0, 2.0 / 3.0, 1.0 / 6.0])
        expected_gradient = negative_mean - positive_mean
        positive_gradient, negative_gradient = jax.grad(lambda pos, neg: loss(weights, pos, neg), argnums=(0, 1))(
            positive, negative
        )

        self.assertTrue(jnp.allclose(actual_gradient, expected_gradient))
        self.assertGreater(abs(float(eager_value)), 0.1)
        self.assertTrue(jnp.allclose(compiled_value, eager_value))
        self.assertTrue(all(jnp.all(leaf == 0) for leaf in jax.tree.leaves(positive_gradient)))
        self.assertTrue(all(jnp.all(leaf == 0) for leaf in jax.tree.leaves(negative_gradient)))

    def test_empty_sample_phases_fail(self):
        nodes = [SpinNode(), SpinNode()]
        model = IsingEBM(nodes, [(nodes[0], nodes[1])], jnp.zeros((2,)), jnp.zeros((1,)), jnp.array(1.0))
        blocks = [Block(model.nodes)]
        valid_samples = [jnp.ones((1, 2), dtype=jnp.bool_)]

        with self.assertRaisesRegex(ValueError, "at least one sample"):
            contrastive_divergence_loss(model, [], valid_samples, blocks)

        with self.assertRaisesRegex(ValueError, "at least one sample"):
            contrastive_divergence_loss(model, valid_samples, [jnp.empty((0, 2), dtype=jnp.bool_)], blocks)

        with self.assertRaisesRegex(ValueError, "one state per block"):
            contrastive_divergence_loss(model, valid_samples * 2, valid_samples, blocks)
