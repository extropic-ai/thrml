import types
import unittest

import equinox as eqx
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


def test_first_record_can_initialize_carry_structure():
    """Both JAX and ordered windows initialize carry before scanning later records."""
    import numpy as np

    from thrml import BlockSamplingProgram, SamplingSchedule, sample_with_observation

    from .test_backends import ReplayBackend, assert_tree_equal

    node = SpinNode()
    program = BlockSamplingProgram(BlockGibbsSpec([], [Block([node])]), [], [])
    clamps = [jnp.array([True])]
    records = ([], [jnp.ones((6, 1), dtype=jnp.bool_)])

    def observe(program, state_free, state_clamped, carry, iteration):
        if carry is None:
            carry = {"sum": jnp.array(0, dtype=jnp.int32)}
        total = carry["sum"] + state_clamped[0][0].astype(jnp.int32) + iteration
        return {"sum": total}, (total, iteration)

    args = (jax.random.key(4), program, SamplingSchedule(2, 6, 3), [], clamps, None, observe)
    expected = sample_with_observation(*args)
    np.testing.assert_array_equal(expected[0]["sum"], 21)
    for partitions in ((1, 3, 2), (2, 1, 3)):
        actual = sample_with_observation(*args, backend=ReplayBackend(records, partitions))
        assert_tree_equal(actual, expected)


def test_moment_observers_with_the_same_statistics_are_equal():
    # A backend compiles the observer fold once for equal observers, so two calls of estimate_moments with the same
    # nodes and edges must give equal observers.
    from thrml.models.ising import _ising_moment_observer

    nodes = [SpinNode() for _ in range(3)]
    edges = [(nodes[0], nodes[1]), (nodes[1], nodes[2])]
    first, second = (_ising_moment_observer(nodes, edges) for _ in range(2))
    assert eqx.tree_equal(first, second)
    assert not eqx.tree_equal(first, _ising_moment_observer(nodes, edges[:1]))
