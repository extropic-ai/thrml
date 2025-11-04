import unittest

import equinox as eqx
import jax
from jax import numpy as jnp

from thrml.block_management import Block
from thrml.block_sampling import SamplingSchedule
from thrml.models.rbm import (
    RBMEBM,
    RBMSamplingProgram,
    RBMTrainingSpec,
    estimate_rbm_grad,
    estimate_rbm_moments,
    rbm_init,
)
from thrml.pgm import SpinNode

from .utils import sample_and_compare_distribution


class TestRBMBasics(unittest.TestCase):
    """Test basic RBM functionality."""

    def test_rbm_creation(self):
        """Test that we can create an RBM with valid parameters."""
        n_visible = 4
        n_hidden = 3

        visible_nodes = [SpinNode() for _ in range(n_visible)]
        hidden_nodes = [SpinNode() for _ in range(n_hidden)]

        visible_biases = jnp.zeros((n_visible,))
        hidden_biases = jnp.zeros((n_hidden,))
        weights = jnp.zeros((n_visible, n_hidden))
        beta = jnp.array(1.0)

        rbm = RBMEBM(visible_nodes, hidden_nodes, visible_biases, hidden_biases, weights, beta)

        self.assertEqual(len(rbm.visible_nodes), n_visible)
        self.assertEqual(len(rbm.hidden_nodes), n_hidden)
        self.assertEqual(rbm.weights.shape, (n_visible, n_hidden))

    def test_rbm_invalid_weights(self):
        """Test that creating an RBM with mismatched weight dimensions raises an error."""
        n_visible = 4
        n_hidden = 3

        visible_nodes = [SpinNode() for _ in range(n_visible)]
        hidden_nodes = [SpinNode() for _ in range(n_hidden)]

        visible_biases = jnp.zeros((n_visible,))
        hidden_biases = jnp.zeros((n_hidden,))
        weights = jnp.zeros((n_visible + 1, n_hidden))  # Wrong shape!
        beta = jnp.array(1.0)

        with self.assertRaises(ValueError):
            RBMEBM(visible_nodes, hidden_nodes, visible_biases, hidden_biases, weights, beta)

    def test_rbm_factors(self):
        """Test that RBM produces the correct factors."""
        n_visible = 2
        n_hidden = 2

        visible_nodes = [SpinNode() for _ in range(n_visible)]
        hidden_nodes = [SpinNode() for _ in range(n_hidden)]

        visible_biases = jnp.array([0.5, -0.3])
        hidden_biases = jnp.array([0.2, 0.4])
        weights = jnp.array([[0.1, 0.2], [0.3, 0.4]])
        beta = jnp.array(2.0)

        rbm = RBMEBM(visible_nodes, hidden_nodes, visible_biases, hidden_biases, weights, beta)

        factors = rbm.factors
        # Should have 3 factors: visible biases, hidden biases, and interactions
        self.assertEqual(len(factors), 3)


class TestRBMSampling(unittest.TestCase):
    """Test RBM sampling."""

    def test_rbm_init(self):
        """Test RBM initialization function."""
        n_visible = 4
        n_hidden = 3
        batch_size = 10

        visible_nodes = [SpinNode() for _ in range(n_visible)]
        hidden_nodes = [SpinNode() for _ in range(n_hidden)]

        visible_biases = jnp.ones((n_visible,))
        hidden_biases = jnp.ones((n_hidden,))
        weights = jnp.zeros((n_visible, n_hidden))
        beta = jnp.array(1.0)

        rbm = RBMEBM(visible_nodes, hidden_nodes, visible_biases, hidden_biases, weights, beta)

        key = jax.random.key(42)
        init_state = rbm_init(key, rbm, [Block(visible_nodes), Block(hidden_nodes)], (batch_size,))

        # Should have 2 blocks (visible and hidden)
        self.assertEqual(len(init_state), 2)
        # Check shapes
        self.assertEqual(init_state[0].shape, (batch_size, n_visible))
        self.assertEqual(init_state[1].shape, (batch_size, n_hidden))
        # Check dtypes
        self.assertTrue(jnp.isdtype(init_state[0].dtype, "bool"))
        self.assertTrue(jnp.isdtype(init_state[1].dtype, "bool"))

    def test_rbm_sampling_program(self):
        """Test that we can create an RBM sampling program."""
        n_visible = 3
        n_hidden = 2

        visible_nodes = [SpinNode() for _ in range(n_visible)]
        hidden_nodes = [SpinNode() for _ in range(n_hidden)]

        visible_biases = jnp.zeros((n_visible,))
        hidden_biases = jnp.zeros((n_hidden,))
        weights = jnp.zeros((n_visible, n_hidden))
        beta = jnp.array(1.0)

        rbm = RBMEBM(visible_nodes, hidden_nodes, visible_biases, hidden_biases, weights, beta)

        # Test creating a sampling program with visible clamped
        program = RBMSamplingProgram(rbm, [Block(hidden_nodes)], [Block(visible_nodes)])

        self.assertEqual(len(program.gibbs_spec.free_blocks), 1)
        self.assertEqual(len(program.gibbs_spec.clamped_blocks), 1)

    def test_small_rbm_distribution(self):
        """Test that sampling from a small RBM produces the correct Boltzmann distribution."""
        n_visible = 2
        n_hidden = 2

        key = jax.random.key(12345)

        visible_nodes = [SpinNode() for _ in range(n_visible)]
        hidden_nodes = [SpinNode() for _ in range(n_hidden)]

        # Use random but small values to make the distribution non-trivial
        key, subkey = jax.random.split(key)
        visible_biases = jax.random.uniform(subkey, (n_visible,), minval=-0.5, maxval=0.5)
        key, subkey = jax.random.split(key)
        hidden_biases = jax.random.uniform(subkey, (n_hidden,), minval=-0.5, maxval=0.5)
        key, subkey = jax.random.split(key)
        weights = jax.random.uniform(subkey, (n_visible, n_hidden), minval=-0.5, maxval=0.5)

        beta = jnp.array(1.0)

        rbm = RBMEBM(visible_nodes, hidden_nodes, visible_biases, hidden_biases, weights, beta)

        # Sample from the joint distribution (all nodes free)
        program = RBMSamplingProgram(rbm, [Block(visible_nodes), Block(hidden_nodes)], [])

        schedule = SamplingSchedule(n_warmup=1000, n_samples=10000, steps_per_sample=5)

        emp_dist, exact_dist = sample_and_compare_distribution(key, rbm, program, [], schedule, 0)

        max_err = jnp.max(jnp.abs(emp_dist - exact_dist)) / jnp.max(exact_dist)
        self.assertLess(max_err, 0.03, f"Distribution mismatch: {max_err}")


class TestRBMMoments(unittest.TestCase):
    """Test RBM moment estimation."""

    def test_estimate_rbm_moments(self):
        """Test that moment estimation runs without errors."""
        n_visible = 3
        n_hidden = 2

        key = jax.random.key(99)

        visible_nodes = [SpinNode() for _ in range(n_visible)]
        hidden_nodes = [SpinNode() for _ in range(n_hidden)]

        visible_biases = jnp.zeros((n_visible,))
        hidden_biases = jnp.zeros((n_hidden,))
        weights = jnp.ones((n_visible, n_hidden)) * 0.1
        beta = jnp.array(1.0)

        rbm = RBMEBM(visible_nodes, hidden_nodes, visible_biases, hidden_biases, weights, beta)

        # Sample with visible clamped
        program = RBMSamplingProgram(rbm, [Block(hidden_nodes)], [Block(visible_nodes)])

        schedule = SamplingSchedule(n_warmup=10, n_samples=100, steps_per_sample=1)

        key, subkey = jax.random.split(key)
        init_hidden = rbm_init(subkey, rbm, [Block(hidden_nodes)], ())
        clamped_visible = [jnp.ones((n_visible,), dtype=jnp.bool_)]

        vis_moments, hid_moments, int_moments = estimate_rbm_moments(
            key, visible_nodes, hidden_nodes, program, schedule, init_hidden, clamped_visible
        )

        # Check shapes
        self.assertEqual(vis_moments.shape, (n_visible,))
        self.assertEqual(hid_moments.shape, (n_hidden,))
        self.assertEqual(int_moments.shape, (n_visible, n_hidden))

        # Moments should be in range [-1, 1] (spin values)
        self.assertTrue(jnp.all(vis_moments >= -1.0) and jnp.all(vis_moments <= 1.0))
        self.assertTrue(jnp.all(hid_moments >= -1.0) and jnp.all(hid_moments <= 1.0))


class TestRBMTraining(unittest.TestCase):
    """Test RBM training utilities."""

    def test_training_spec_creation(self):
        """Test that we can create an RBM training spec."""
        n_visible = 4
        n_hidden = 3

        visible_nodes = [SpinNode() for _ in range(n_visible)]
        hidden_nodes = [SpinNode() for _ in range(n_hidden)]

        visible_biases = jnp.zeros((n_visible,))
        hidden_biases = jnp.zeros((n_hidden,))
        weights = jnp.zeros((n_visible, n_hidden))
        beta = jnp.array(1.0)

        rbm = RBMEBM(visible_nodes, hidden_nodes, visible_biases, hidden_biases, weights, beta)

        schedule_pos = SamplingSchedule(n_warmup=10, n_samples=10, steps_per_sample=1)
        schedule_neg = SamplingSchedule(n_warmup=10, n_samples=10, steps_per_sample=1)

        training_spec = RBMTrainingSpec(rbm, schedule_pos, schedule_neg)

        self.assertIsNotNone(training_spec.program_positive)
        self.assertIsNotNone(training_spec.program_negative)

    def test_estimate_rbm_grad(self):
        """Test gradient estimation for RBM training."""
        n_visible = 3
        n_hidden = 2

        key = jax.random.key(777)

        visible_nodes = [SpinNode() for _ in range(n_visible)]
        hidden_nodes = [SpinNode() for _ in range(n_hidden)]

        key, subkey = jax.random.split(key)
        visible_biases = jax.random.uniform(subkey, (n_visible,), minval=-0.5, maxval=0.5)
        key, subkey = jax.random.split(key)
        hidden_biases = jax.random.uniform(subkey, (n_hidden,), minval=-0.5, maxval=0.5)
        key, subkey = jax.random.split(key)
        weights = jax.random.uniform(subkey, (n_visible, n_hidden), minval=-0.5, maxval=0.5)
        beta = jnp.array(1.0)

        rbm = RBMEBM(visible_nodes, hidden_nodes, visible_biases, hidden_biases, weights, beta)

        schedule_pos = SamplingSchedule(n_warmup=50, n_samples=100, steps_per_sample=2)
        schedule_neg = SamplingSchedule(n_warmup=50, n_samples=100, steps_per_sample=2)

        training_spec = RBMTrainingSpec(rbm, schedule_pos, schedule_neg)

        # Create some dummy visible data
        batch_size = 4
        key, subkey = jax.random.split(key)
        visible_data = [jax.random.bernoulli(subkey, shape=(batch_size, n_visible)).astype(jnp.bool_)]

        # Initialize hidden states for positive phase
        n_chains_pos = 2
        key, subkey = jax.random.split(key)
        init_hidden_pos = rbm_init(subkey, rbm, [Block(hidden_nodes)], (n_chains_pos, batch_size))

        # Initialize both visible and hidden for negative phase
        n_chains_neg = 2
        key, subkey = jax.random.split(key)
        init_neg = rbm_init(subkey, rbm, [Block(visible_nodes), Block(hidden_nodes)], (n_chains_neg,))

        grad_w, grad_vb, grad_hb = estimate_rbm_grad(key, training_spec, visible_data, init_hidden_pos, init_neg)

        # Check shapes
        self.assertEqual(grad_w.shape, (n_visible, n_hidden))
        self.assertEqual(grad_vb.shape, (n_visible,))
        self.assertEqual(grad_hb.shape, (n_hidden,))

        # Gradients should be finite
        self.assertTrue(jnp.all(jnp.isfinite(grad_w)))
        self.assertTrue(jnp.all(jnp.isfinite(grad_vb)))
        self.assertTrue(jnp.all(jnp.isfinite(grad_hb)))


class TestRBMEnergy(unittest.TestCase):
    """Test RBM energy function."""

    def test_energy_computation(self):
        """Test that energy computation works correctly."""
        n_visible = 2
        n_hidden = 2

        visible_nodes = [SpinNode() for _ in range(n_visible)]
        hidden_nodes = [SpinNode() for _ in range(n_hidden)]

        # Simple case: all zeros
        visible_biases = jnp.zeros((n_visible,))
        hidden_biases = jnp.zeros((n_hidden,))
        weights = jnp.zeros((n_visible, n_hidden))
        beta = jnp.array(1.0)

        rbm = RBMEBM(visible_nodes, hidden_nodes, visible_biases, hidden_biases, weights, beta)

        # All spins down (False = -1 in spin representation)
        state = [jnp.array([False, False]), jnp.array([False, False])]
        blocks = [Block(visible_nodes), Block(hidden_nodes)]

        energy = rbm.energy(state, blocks)

        # With all zeros, energy should be 0
        self.assertAlmostEqual(float(energy), 0.0, places=5)

    def test_energy_with_biases(self):
        """Test energy with non-zero biases."""
        n_visible = 2
        n_hidden = 1

        visible_nodes = [SpinNode() for _ in range(n_visible)]
        hidden_nodes = [SpinNode() for _ in range(n_hidden)]

        visible_biases = jnp.array([1.0, 0.0])
        hidden_biases = jnp.array([1.0])
        weights = jnp.zeros((n_visible, n_hidden))
        beta = jnp.array(1.0)

        rbm = RBMEBM(visible_nodes, hidden_nodes, visible_biases, hidden_biases, weights, beta)

        # All spins up (True = +1)
        state_up = [jnp.array([True, True]), jnp.array([True])]
        # All spins down (False = -1)
        state_down = [jnp.array([False, False]), jnp.array([False])]

        blocks = [Block(visible_nodes), Block(hidden_nodes)]

        energy_up = rbm.energy(state_up, blocks)
        energy_down = rbm.energy(state_down, blocks)

        # Energy with spins up should be lower (more favorable) due to positive biases
        self.assertLess(float(energy_up), float(energy_down))


if __name__ == "__main__":
    unittest.main()
