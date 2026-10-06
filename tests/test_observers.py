import types
import unittest

import jax
import jax.numpy as jnp

from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec
from thrml.observers import MomentAccumulatorObserver
from thrml.pgm import CategoricalNode, SpinNode


class TestMomentObserver(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spin = SpinNode()
        self.cat = CategoricalNode()

        self.blocks = [Block([self.spin]), Block([self.cat])]
        self.node_shape_dtypes = {
            SpinNode: jax.ShapeDtypeStruct((), jnp.bool_),
            CategoricalNode: jax.ShapeDtypeStruct((), jnp.uint8),
        }
        self.gibbs_spec = BlockGibbsSpec(self.blocks, [], self.node_shape_dtypes)
        self.program = types.SimpleNamespace(gibbs_spec=self.gibbs_spec)

    def test_preserves_mixed_node_values(self):
        """Test that moment observer correctly preserves mixed node data types."""
        observer = MomentAccumulatorObserver([[(self.spin, self.cat)]])
        carry = observer.init()

        state_free = [
            jnp.array([True], dtype=jnp.bool_),
            jnp.array([2], dtype=jnp.uint8),
        ]

        with jax.numpy_dtype_promotion("standard"):
            carry_out, _ = observer(self.program, state_free, [], carry, jnp.array(0, dtype=jnp.int32))

        self.assertEqual(carry_out[0][0], 2)

    def test_deduplicates_nodes_across_moments(self):
        nodes = [SpinNode() for _ in range(4)]
        first_moments = [(node,) for node in nodes]
        second_moments = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes))]

        observer = MomentAccumulatorObserver([first_moments, second_moments])

        self.assertEqual(len(observer.blocks_to_sample), 1)
        self.assertEqual(observer.blocks_to_sample[0].nodes, nodes)
        self.assertEqual(len(observer.flat_nodes_list), len(nodes))
        self.assertEqual(len(observer.flat_to_type_slices_list[0]), len(nodes))
