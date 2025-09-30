import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

import thrml.pgm
from thrml import block_management


class Node1(thrml.pgm.AbstractNode):
    pass


class Node2(thrml.pgm.AbstractNode):
    pass


class Node3(thrml.pgm.AbstractNode):
    pass


class TestBlocks(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng_key = jax.random.key(424)
        self.blocks = [
            block_management.Block([Node1() for _ in range(5)]),
            block_management.Block([Node2() for _ in range(9)]),
            block_management.Block([Node2() for _ in range(7)]),
            block_management.Block([Node3() for _ in range(3)]),
        ]

        # ShapeDtypeStruct definitions
        bool_type = jax.ShapeDtypeStruct((2,), dtype=jnp.bool_)
        float_type = jax.ShapeDtypeStruct((5,), dtype=jnp.float32)
        int_type = jax.ShapeDtypeStruct((2,), dtype=jnp.int16)

        self.node_types = {
            Node1: [bool_type],
            Node2: [float_type, bool_type],
            Node3: [int_type, bool_type],
        }

        self.node_types1 = {
            Node1: [bool_type],
            Node2: [float_type, bool_type],
            Node3: [int_type, float_type],
        }

        self.node_types2 = {
            Node1: [int_type],
            Node2: [int_type],
            Node3: [int_type],
        }

        class CustomObj(eqx.Module):
            a: jax.ShapeDtypeStruct = float_type

        self.node_types3 = {
            Node1: int_type,
            Node2: {"a": bool_type, "b": int_type},
            Node3: CustomObj(),
        }

        # Collect them for testing
        self.node_type_dicts = {
            "variation0": self.node_types,
            "variation1": self.node_types1,
            "variation2": self.node_types2,
            "variation3": self.node_types3,
        }

        self.configs = {}
        for label, node_dict in self.node_type_dicts.items():
            spec = block_management.BlockSpec(self.blocks, node_dict)
            all_types = [type_list for type_list in node_dict.values()]
            block_state = block_management.make_empty_block_state(self.blocks, node_dict)
            self.configs[label] = (spec, block_state, all_types)

    def test_shape_transforms(self):
        for label, (spec, block_state, _) in self.configs.items():
            with self.subTest(msg=f"Testing shape_transforms with {label}"):
                global_state = block_management.block_state_to_global(block_state, spec)
                re_block = block_management.from_global_state(global_state, spec, spec.blocks)
                self.assertTrue(eqx.tree_equal(block_state, re_block))

    def test_node_lookup(self):
        for label, (spec, block_state, _) in self.configs.items():
            with self.subTest(msg=f"Testing node_lookup with {label}"):
                global_state = block_management.block_state_to_global(block_state, spec)
                for block, state in zip(spec.blocks, block_state):
                    type_inds, arr_inds = block_management.get_node_locations(block, spec)
                    vals = jax.tree.map(lambda x: x[arr_inds], global_state[type_inds])
                    self.assertTrue(eqx.tree_equal(vals, state))

    def test_empty_state(self):
        for label, (spec, block_state, _) in self.configs.items():
            with self.subTest(msg=f"Testing empty_state with {label}"):
                batch_shape = (10, 2)
                empty_state = block_management.make_empty_block_state(spec.blocks, spec.node_shape_struct, batch_shape)
                empty_state = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), empty_state)
                b_state = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), block_state)
                eqx.tree_equal(empty_state, b_state)


class Template2(eqx.Module):
    scalar: int
    data: Array


class Template1(eqx.Module):
    temp_2: Template2
    data: Array
    scalar: float


class TestBlockCompat(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.batch_shape = (4, 2, 10)

        temp_2_sd = Template2(1, jax.ShapeDtypeStruct(shape=(4,), dtype=jnp.float32))
        self.temp_1_sd = Template1(temp_2_sd, jax.ShapeDtypeStruct(shape=(), dtype=jnp.int8), 4.3)
        self.temp_2_good = Template2(3, jnp.zeros((*self.batch_shape, 4), dtype=jnp.float32))

        self.block_1 = block_management.Block([Node1() for _ in range(5)])
        self.block_2 = block_management.Block([Node2() for _ in range(3)])
        self.block_3 = block_management.Block([Node3() for _ in range(9)])

        self.blocks = [self.block_1, self.block_2, self.block_3]

        self.node_sd_map = {
            Node1: (
                jax.ShapeDtypeStruct(shape=(1, 2), dtype=jnp.bool),
                jax.ShapeDtypeStruct(shape=(), dtype=jnp.uint8),
            ),
            Node2: self.temp_1_sd,
            Node3: jax.ShapeDtypeStruct(shape=(7,), dtype=jnp.float32),
        }

        self.good_state_1 = (
            jnp.zeros((*self.batch_shape, 5, 1, 2), dtype=jnp.bool),
            jnp.zeros((*self.batch_shape, 5), dtype=jnp.uint8),
        )

        t2 = Template2(5, jnp.zeros((*self.batch_shape, 3, 4), dtype=jnp.float32))
        self.good_state_2 = Template1(
            t2,
            jnp.zeros(
                (
                    *self.batch_shape,
                    3,
                ),
                dtype=jnp.int8,
            ),
            19.9,
        )

        self.good_state_3 = jnp.zeros((*self.batch_shape, 9, 7), dtype=jnp.float32)

    def test_good(self):
        temp_1_good = Template1(self.temp_2_good, jnp.zeros(self.batch_shape, dtype=jnp.int8), 7.1)
        batch_shape = block_management._check_pytree_compat(self.temp_1_sd, temp_1_good)

        self.assertEqual(batch_shape, self.batch_shape)

    def test_bad_dtype(self):
        temp_1_bad = Template1(self.temp_2_good, jnp.zeros(self.batch_shape, dtype=jnp.float32), 10.2)
        with self.assertRaises(RuntimeError) as error:
            _ = block_management._check_pytree_compat(self.temp_1_sd, temp_1_bad)

        self.assertIn("type", str(error.exception))

    def test_bad_shape(self):
        temp_1_bad = Template1(self.temp_2_good, jnp.zeros((*self.batch_shape, 1), dtype=jnp.int8), 11.9)
        with self.assertRaises(RuntimeError) as error:
            _ = block_management._check_pytree_compat(self.temp_1_sd, temp_1_bad)

        self.assertIn("shape", str(error.exception))

    def test_missing_array(self):
        temp_1_bad = Template1(self.temp_2_good, 1.0, 11.9)
        with self.assertRaises(RuntimeError) as error:
            _ = block_management._check_pytree_compat(self.temp_1_sd, temp_1_bad)

        self.assertIn("missing", str(error.exception))

    def test_bad_structure(self):
        temp_1_bad = jnp.array(1.0)
        with self.assertRaises(RuntimeError) as error:
            _ = block_management._check_pytree_compat(self.temp_1_sd, temp_1_bad)

        self.assertIn("structure", str(error.exception))

    def test_good_state(self):
        # should pass
        block_management.verify_block_state(
            self.blocks, [self.good_state_1, self.good_state_2, self.good_state_3], self.node_sd_map, block_axis=-1
        )

    def test_wrong_state_len(self):
        with self.assertRaises(RuntimeError) as error:
            block_management.verify_block_state(
                self.blocks, [self.good_state_1, self.good_state_2], self.node_sd_map, block_axis=-1
            )

        self.assertIn("of states not equal", str(error.exception))

    def test_bad_block(self):
        bad_state = self.good_state_3.astype(jnp.bool)

        with self.assertRaises(RuntimeError) as error:
            block_management.verify_block_state(
                self.blocks, [self.good_state_1, self.good_state_2, bad_state], self.node_sd_map, block_axis=-1
            )

        self.assertIn("type", str(error.exception))

    def test_length_mismatch(self):
        bad_state = jnp.zeros((*self.batch_shape, 4, 7), dtype=jnp.float32)

        with self.assertRaises(RuntimeError) as error:
            block_management.verify_block_state(
                self.blocks, [self.good_state_1, self.good_state_2, bad_state], self.node_sd_map, block_axis=-1
            )

        self.assertIn("block length", str(error.exception))


class TestDuplicate(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.good_blocks = [block_management.Block([Node1() for _ in range(3)]) for _ in range(2)]

        self.node_sd = {Node1: jax.ShapeDtypeStruct}

    def test_good(self):
        _ = block_management.BlockSpec(self.good_blocks, self.node_sd)

    def test_duplicate(self):
        with self.assertRaises(RuntimeError) as error:
            _ = block_management.BlockSpec(self.good_blocks + self.good_blocks, self.node_sd)

        self.assertIn("show up twice", str(error.exception))
