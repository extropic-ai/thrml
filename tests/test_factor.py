import unittest

import jax.numpy as jnp

from thrml.block_management import Block
from thrml.factor import AbstractFactor, WeightedFactor
from thrml.interaction import InteractionGroup
from thrml.pgm import AbstractNode


class Node(AbstractNode):
    pass


class PointlessFactor(AbstractFactor):
    def to_interaction_groups(self) -> list[InteractionGroup]:
        return []


class TestFactorCreate(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_nodes = 5
        self.good_node_groups = [Block([Node() for _ in range(self.n_nodes)]) for _ in range(3)]

    def test_good(self):
        _ = PointlessFactor(self.good_node_groups)

    def test_empty(self):
        with self.assertRaises(RuntimeError) as error:
            _ = PointlessFactor([])

        self.assertIn("empty", str(error.exception))

    def test_ragged(self):
        bad_block = Block([Node() for _ in range(self.n_nodes + 1)])
        with self.assertRaises(RuntimeError) as error:
            _ = PointlessFactor(self.good_node_groups + [bad_block])

        self.assertIn("same number", str(error.exception))


class SimpleWeighted(WeightedFactor):
    def to_interaction_groups(self) -> list[InteractionGroup]:
        return []


class TestWeighted(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_nodes = 5
        self.good_node_groups = [Block([Node() for _ in range(self.n_nodes)]) for _ in range(3)]
        self.good_weights = jnp.zeros((self.n_nodes, 1, 3))

    def test_good(self):
        _ = SimpleWeighted(self.good_weights, self.good_node_groups)

    def test_bad(self):
        bad_weights = jnp.zeros((self.n_nodes + 1, 1, 3))
        with self.assertRaises(RuntimeError) as error:
            _ = SimpleWeighted(bad_weights, self.good_node_groups)
        self.assertIn("weights", str(error.exception))
