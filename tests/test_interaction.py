import unittest

import jax.numpy as jnp

from thrml.block_management import Block
from thrml.interaction import InteractionGroup
from thrml.pgm import AbstractNode


class Node(AbstractNode):
    pass


class TestInteractionInputs(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        block_size = 4
        self.good_head = Block([Node() for _ in range(block_size)])
        self.good_tails = [Block([Node() for _ in range(block_size)]) for _ in range(2)]
        self.good_interaction = jnp.zeros((block_size,))

    def test_good(self):
        _ = InteractionGroup(self.good_interaction, self.good_head, self.good_tails)

    def test_bad_tail(self):
        bad_tail = [Block([Node() for _ in range(len(self.good_head) + 1) for _ in range(2)]) for _ in range(2)]

        with self.assertRaises(RuntimeError) as error:
            _ = InteractionGroup(self.good_interaction, self.good_head, bad_tail)

        self.assertIn("tail node blocks", str(error.exception))

    def test_bad_interaction(self):
        bad_interaction = (self.good_interaction, jnp.array(1.0))
        with self.assertRaises(RuntimeError) as error:
            _ = InteractionGroup(bad_interaction, self.good_head, self.good_tails)

        self.assertIn("leading dimension", str(error.exception))

        bad_interaction = (self.good_interaction, jnp.zeros((len(self.good_head) + 1,)))
        with self.assertRaises(RuntimeError) as error:
            _ = InteractionGroup(bad_interaction, self.good_head, self.good_tails)
        self.assertIn("leading dimension", str(error.exception))
