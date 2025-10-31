import types

import jax
import jax.numpy as jnp

from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec
from thrml.observers import MomentAccumulatorObserver
from thrml.pgm import CategoricalNode, SpinNode


def test_moment_observer_preserves_mixed_node_values():
    spin = SpinNode()
    cat = CategoricalNode()

    blocks = [Block([spin]), Block([cat])]
    node_shape_dtypes = {
        SpinNode: jax.ShapeDtypeStruct((), jnp.bool_),
        CategoricalNode: jax.ShapeDtypeStruct((), jnp.uint8),
    }
    gibbs_spec = BlockGibbsSpec(blocks, [], node_shape_dtypes)
    program = types.SimpleNamespace(gibbs_spec=gibbs_spec)

    observer = MomentAccumulatorObserver([[(spin, cat)]])
    carry = observer.init()

    state_free = [
        jnp.array([True], dtype=jnp.bool_),
        jnp.array([2], dtype=jnp.uint8),
    ]

    carry_out, _ = observer(program, state_free, [], carry, jnp.array(0, dtype=jnp.int32))

    assert carry_out[0][0] == 2
