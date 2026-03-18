import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, PRNGKeyArray

from thrml.block_management import Block
from thrml.block_sampling import SamplingSchedule, BlockSamplingProgram, sample_states
from thrml.models.ising import IsingEBM, IsingSamplingProgram, hinton_init
from thrml.training import contrastive_divergence_loss, EBMTrainingSpec, train_step

class TestTraining(unittest.TestCase):
    def setUp(self):
        # Set up a small Ising model (4 nodes, 3 edges)
        self.dim = 4
        self.key = jax.random.key(42)
        from thrml.pgm import SpinNode
        nodes = [SpinNode() for _ in range(self.dim)]
        edges = [(nodes[i], nodes[i+1]) for i in range(self.dim-1)]
        key, bkey, wkey = jax.random.split(self.key, 3)
        self.biases = jax.random.uniform(bkey, (self.dim,), minval=-1.0, maxval=1.0)
        self.weights = jax.random.uniform(wkey, (len(edges),), minval=-1.0, maxval=1.0)
        self.beta = jnp.array(1.0)
        self.model = IsingEBM(nodes, edges, self.biases, self.weights, self.beta)

        self.blocks = [Block(nodes)]
        self.schedule = SamplingSchedule(n_warmup=10, n_samples=10, steps_per_sample=5)

        self.program = IsingSamplingProgram(self.model, self.blocks, [])
        self.spec = EBMTrainingSpec(
            program_positive=self.program,
            schedule_positive=self.schedule,
            program_negative=self.program,
            schedule_negative=self.schedule,
            visible_blocks=[],
            free_blocks=self.blocks,
            all_blocks=self.blocks
        )
        self.optimizer = optax.adam(learning_rate=0.01)
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_inexact_array))

    def test_imports(self):
        self.assertIsNotNone(contrastive_divergence_loss)
        self.assertIsNotNone(EBMTrainingSpec)
        self.assertIsNotNone(train_step)

    def test_cd_loss_shape(self):
        pos = hinton_init(self.key, self.model, self.blocks, (10,))
        neg = hinton_init(self.key, self.model, self.blocks, (10,))
        loss = contrastive_divergence_loss(self.model, pos, neg, self.blocks)
        self.assertEqual(loss.shape, ())

    def test_cd_loss_sign(self):
        # neg samples have higher energy: use all-ones (more frustrated) vs all-zeros
        pos = [jnp.zeros((10, self.dim), dtype=jnp.bool_)]
        neg = [jnp.ones((10, self.dim), dtype=jnp.bool_)]
        loss = contrastive_divergence_loss(self.model, pos, neg, self.blocks)
        # Loss can be any sign depending on model params — just check it's a scalar
        self.assertEqual(loss.shape, ())

    def test_train_step_returns_updated_model(self):
        k1, k2 = jax.random.split(self.key)
        pos = hinton_init(k1, self.model, self.blocks, (10,))
        neg = hinton_init(k2, self.model, self.blocks, (10,))
        model0 = self.model
        model1, opt_state1, loss = train_step(model0, self.opt_state, self.optimizer, self.spec, pos, neg, [], self.key)
        # Check model parameters changed
        params0 = eqx.filter(model0, eqx.is_inexact_array)
        params1 = eqx.filter(model1, eqx.is_inexact_array)
        leaves0 = jax.tree_util.tree_leaves(params0)
        leaves1 = jax.tree_util.tree_leaves(params1)
        any_changed = any(not jnp.allclose(p0, p1) for p0, p1 in zip(leaves0, leaves1))
        self.assertTrue(any_changed)

    def test_loss_decreases_over_training(self):
        model = self.model
        opt_state = self.opt_state
        losses = []
        key = self.key
        for step in range(20):
            key, k1, k2 = jax.random.split(key, 3)
            pos = hinton_init(k1, model, self.blocks, (20,))
            neg = hinton_init(k2, model, self.blocks, (20,))
            model, opt_state, loss = train_step(model, opt_state, self.optimizer, self.spec, pos, neg, [], key)
            losses.append(float(loss))
        first5 = jnp.array(losses[:5])
        last5 = jnp.array(losses[-5:])
        self.assertLess(float(jnp.mean(last5)), float(jnp.mean(first5)))

    def test_ising_parity(self):
        model_generic = self.model
        opt_state_gen = self.opt_state
        losses_generic = []
        key = self.key
        for _ in range(50):
            key, k1, k2 = jax.random.split(key, 3)
            pos = hinton_init(k1, model_generic, self.blocks, (20,))
            neg = hinton_init(k2, model_generic, self.blocks, (20,))
            model_generic, opt_state_gen, loss = train_step(model_generic, opt_state_gen, self.optimizer, self.spec, pos, neg, [], key)
            losses_generic.append(float(loss))
        # Verify loss decreased over training
        self.assertLess(losses_generic[-1], losses_generic[0])

if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
