import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Key, PyTree

from thrml.block_management import Block
from thrml.block_sampling import (
    BlockGibbsSpec,
    BlockSamplingProgram,
    SamplingSchedule,
    sample_blocks,
    sample_single_block,
    sample_states,
)
from thrml.conditional_samplers import (
    AbstractConditionalSampler,
    _SamplerState,
    _State,
)
from thrml.interaction import InteractionGroup
from thrml.pgm import AbstractNode


class ContinousScalarNode(AbstractNode):
    pass


class PlusInteraction(eqx.Module):
    multiplier: Array


class MinusInteraction(eqx.Module):
    multiplier: Array


class MemoryInteraction(eqx.Module):
    multiplier: Array


class PlusMinusSampler(AbstractConditionalSampler):
    def sample(
        self,
        key,
        interactions: list[PyTree],
        active_flags: list[Array],
        states: list[list[_State]],
        sampler_state: _SamplerState,
        output_sd: jax.ShapeDtypeStruct,
    ):
        output = jnp.zeros(output_sd.shape, dtype=output_sd.dtype)
        for interaction, active, state in zip(interactions, active_flags, states):
            active = active.astype(interaction.multiplier.dtype)
            s = state[0].astype(interaction.multiplier.dtype)
            if isinstance(interaction, (PlusInteraction, MemoryInteraction)):
                output += jnp.sum(interaction.multiplier * active * s, axis=-1)
            elif isinstance(interaction, MinusInteraction):
                output -= jnp.sum(interaction.multiplier * active * s, axis=-1)
            else:
                raise RuntimeError("Invalid interaction passed to PlusMinusSampler")

        return output, sampler_state

    def init(self) -> _SamplerState:
        return None


class TestPlusMinus(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        key = jax.random.key(424)

        free_nodes = [ContinousScalarNode() for _ in range(3)]

        minus_nodes = [ContinousScalarNode() for _ in range(2)]
        plus_nodes = [ContinousScalarNode() for _ in range(2)]

        key, subkey = jax.random.split(key, 2)
        self.minus_weights = jax.random.uniform(subkey, (3,), minval=0, maxval=1)
        key, subkey = jax.random.split(key, 2)
        self.plus_weights = jax.random.uniform(subkey, (3,), minval=0, maxval=1)

        minus_interaction_group = InteractionGroup(
            MinusInteraction(self.minus_weights),
            Block([free_nodes[0], free_nodes[0], free_nodes[1]]),
            [Block([minus_nodes[0], minus_nodes[1], minus_nodes[1]])],
        )

        plus_interaction_group = InteractionGroup(
            PlusInteraction(self.plus_weights),
            Block([free_nodes[1], free_nodes[2], free_nodes[2]]),
            [Block([plus_nodes[0], plus_nodes[0], plus_nodes[1]])],
        )

        memory_interaction_group = InteractionGroup(
            MemoryInteraction(jnp.ones(len(free_nodes))), Block(free_nodes), [Block(free_nodes)]
        )

        block_spec = BlockGibbsSpec(
            [Block([free_nodes[0]]), Block(free_nodes[1:])],
            [Block(plus_nodes + minus_nodes)],
            {ContinousScalarNode: jax.ShapeDtypeStruct((), jnp.float32)},
        )

        self.program = BlockSamplingProgram(
            block_spec,
            [PlusMinusSampler(), PlusMinusSampler()],
            [minus_interaction_group, plus_interaction_group, memory_interaction_group],
        )

        keys = jax.random.split(key, 4)

        self.state_free = [
            jax.random.uniform(keys[0], (1,), minval=1.0, maxval=5.0),
            jax.random.uniform(keys[1], (2,), minval=1.0, maxval=5.0),
        ]
        self.state_clamped = [jax.random.uniform(keys[2], (4,), minval=1.0, maxval=5.0)]

        self.key = keys[-1]

    def test_sample_block(self):
        outputs = []
        for block in [0, 1]:
            outputs.append(
                sample_single_block(self.key, self.state_free, self.state_clamped, self.program, block, None)[0]
            )

        first_output = self.state_free[0][0] - jnp.sum(self.minus_weights[:2] * self.state_clamped[0][2:])
        second_output = (
            self.state_free[1][0]
            - self.minus_weights[-1] * self.state_clamped[0][-1]
            + self.plus_weights[0] * self.state_clamped[0][0]
        )
        third_output = self.state_free[1][1] + jnp.sum(self.plus_weights[1:] * self.state_clamped[0][:2])

        self.assertTrue(np.allclose(outputs[0], [first_output], rtol=1e-6))
        self.assertTrue(np.allclose(outputs[1], [second_output, third_output], rtol=1e-6))

    def test_sample_blocks(self):
        # just make sure this runs, nothing new to learn
        sample_blocks(self.key, self.state_free, self.state_clamped, self.program, [None, None])

    def test_sample_states(self):
        # just make sure this runs
        schedule = SamplingSchedule(5, 5, 5)
        sample_states(
            self.key, self.program, schedule, self.state_free, self.state_clamped, self.program.gibbs_spec.free_blocks
        )

    def test_state_gaurdrailing(self):
        wrong_state_free = [self.state_free[0], jnp.zeros((2,), dtype=jnp.bool)]

        wrong_state_clamped = [jnp.zeros((4,), dtype=jnp.bool)]

        with self.assertRaises(RuntimeError) as error:
            _ = sample_blocks(self.key, wrong_state_free, self.state_clamped, self.program, [None, None])

        self.assertIn("type", str(error.exception))

        with self.assertRaises(RuntimeError) as error:
            _ = sample_blocks(self.key, self.state_free, wrong_state_clamped, self.program, [None, None])

        self.assertIn("type", str(error.exception))


class TestSamplerValidation(unittest.TestCase):
    def test_mismatched_sampler_list_raises(self):
        block_a = Block([ContinousScalarNode()])
        block_b = Block([ContinousScalarNode()])
        node_shape_dtypes = {ContinousScalarNode: jax.ShapeDtypeStruct((), jnp.float32)}
        spec = BlockGibbsSpec([block_a, block_b], [], node_shape_dtypes)

        with self.assertRaisesRegex(ValueError, "Expected 2 samplers"):
            BlockSamplingProgram(spec, [PlusMinusSampler()], [])


class MultiNode(AbstractNode):
    pass


class MultiNodeState(eqx.Module):
    float_counter: Array
    cat_counter: Array


class IncrementSampler(AbstractConditionalSampler):
    def sample(
        self,
        key: Key,
        interactions: list[PyTree],
        active_flags: list[Array],
        states: list[list[_State]],
        sampler_state: _SamplerState,
        output_sd: PyTree[jax.ShapeDtypeStruct],
    ):
        assert isinstance(output_sd, MultiNodeState)

        for interaction, active, state in zip(interactions, active_flags, states):
            if isinstance(interaction, PlusInteraction):
                return (
                    MultiNodeState(state[0].float_counter[:, 0, :] + 1, state[0].cat_counter[:, 0, :] + 1),
                    sampler_state,
                )

    def init(self) -> _SamplerState:
        return None


class TestPyTreeState(unittest.TestCase):
    """Test that a sampling program involving nodes with more general pytree shapedtypes works"""

    def test_pytree_state(self):
        n_float = 2
        n_cat = 4

        sd_map = {
            MultiNode: MultiNodeState(
                jax.ShapeDtypeStruct((n_float,), jnp.float32), jax.ShapeDtypeStruct((n_cat,), jnp.int8)
            )
        }

        nodes = [MultiNode() for _ in range(10)]

        key = jax.random.key(424)

        interaction_group = InteractionGroup(PlusInteraction(jnp.ones((len(nodes),))), Block(nodes), [Block(nodes)])

        spec = BlockGibbsSpec([Block(nodes)], [], sd_map)

        key, subkey = jax.random.split(key, 2)
        init_float = jax.random.normal(subkey, (len(nodes), n_float))
        key, subkey = jax.random.split(key, 2)
        init_cat = jax.random.randint(subkey, (len(nodes), n_cat), minval=-4, maxval=4)

        init_state = [MultiNodeState(init_float, init_cat)]

        prog = BlockSamplingProgram(spec, [IncrementSampler()], [interaction_group])

        res, _ = sample_single_block(key, init_state, [], prog, 0, None)

        self.assertTrue(jnp.allclose(init_state[0].cat_counter + 1, res.cat_counter))
        self.assertTrue(jnp.allclose(init_state[0].float_counter + 1, res.float_counter))
