import unittest
from typing import Sequence, Type

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest
from jaxtyping import Array, Key

from thrml.block_management import Block
from thrml.block_sampling import SamplingSchedule, sample_states
from thrml.models.ising import (
    Edge,
    IsingEBM,
    IsingSamplingProgram,
    IsingTrainingSpec,
    estimate_kl_grad,
    hinton_init,
)
from thrml.pgm import AbstractNode, SpinNode


def get_double_grid(
    side_len: int,
    jumps: Sequence[int],
    n_visible: int,
    node: Type[AbstractNode],
    key: Key[Array, ""],
) -> tuple[Block, Block, Block, Block, list[AbstractNode], list[Edge]]:
    # n_groups = 2
    size = side_len**2
    assert n_visible <= size

    def get_idx(i, j):
        i = (i + side_len) % side_len
        j = (j + side_len) % side_len
        return i * side_len + j

    def get_coords(idx):
        return idx // side_len, idx % side_len

    def _make_edge(idx, di, dj):
        i, j = get_coords(idx)
        return jnp.array([idx, get_idx(i + di, j + dj)])

    make_edge = jax.jit(jax.vmap(_make_edge, in_axes=(0, None, None), out_axes=0))

    indices = jnp.arange(size)
    edges_arr = jnp.stack([indices, indices], axis=1)
    for d in jumps:
        left_edges = make_edge(indices, -d, 0)
        right_edges = make_edge(indices, d, 0)
        upper_edges = make_edge(indices, 0, -d)
        lower_edges = make_edge(indices, 0, d)
        edges_arr = jnp.concatenate([edges_arr, left_edges, right_edges, upper_edges, lower_edges], axis=0)

    deg = 4 * len(jumps) + 1
    total_edges = size * deg
    assert edges_arr.shape == (total_edges, 2)

    nodes_upper = [node() for _ in range(size)]
    nodes_lower = [node() for _ in range(size)]
    all_nodes = nodes_upper + nodes_lower
    all_edges = [(nodes_upper[i], nodes_lower[j]) for i, j in edges_arr]

    visible_indices = jax.random.permutation(key, jnp.arange(size))[:n_visible]
    visible_nodes = [nodes_upper[i] for i in visible_indices]
    upper_without_visible = [node for node in nodes_upper if node not in visible_nodes]

    return (
        Block(nodes_upper),
        Block(nodes_lower),
        Block(visible_nodes),
        Block(upper_without_visible),
        all_nodes,
        all_edges,
    )


@pytest.mark.slow
class TestTrainMnist(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_classes = [0, 3, 4]
        self.num_label_spots = 10
        label_size = len(self.target_classes) * self.num_label_spots
        data_dim = 28 * 28 + label_size

        self.train_data_filtered = jnp.load("tests/mnist_test_data/train_data_filtered.npy")
        self.sep_images_test = {}
        for digit in self.target_classes:
            self.sep_images_test[digit] = jnp.load(f"tests/mnist_test_data/sep_images_test_{digit}.npy")

        (upper_grid, lower_grid, visible_nodes, upper_without_visible, all_nodes, all_edges) = get_double_grid(
            40, [1, 4, 15], data_dim, SpinNode, jax.random.key(0)
        )

        self.init_model = IsingEBM(
            all_nodes,
            all_edges,
            jnp.zeros((len(all_nodes),), dtype=float),
            jnp.zeros((len(all_edges),), dtype=float),
            jnp.array(1.0),
        )

        self.positive_sampling_blocks = [upper_without_visible, lower_grid]
        self.negative_sampling_blocks = [upper_grid, lower_grid]
        self.training_data_blocks = [visible_nodes]

        image_block = Block(visible_nodes.nodes[: 28 * 28])
        upper_without_image = Block([node for node in upper_grid if node not in image_block.nodes])
        self.classification_sampling_blocks = [upper_without_image, lower_grid]
        self.classification_data_blocks = [image_block]
        self.classification_label_block = Block(visible_nodes.nodes[28 * 28 :])

        self.schedule_negative = SamplingSchedule(200, 40, 5)
        self.schedule_positive = SamplingSchedule(200, 20, 10)
        self.accuracy_schedule = SamplingSchedule(400, 40, 10)

        self.optim = optax.adam(learning_rate=0.01)
        self.n_epochs = 1

    def test_train_mnist(self):
        def do_epoch_simplified(
            key,
            model,
            bsz_positive,
            bsz_negative,
            data_positive,
            opt_state=None,
        ):
            def batch_data(key, data, _bsz, clamped_blocks):
                clamped_nodes = [node for block in clamped_blocks for node in block]
                data_size = data.shape[0]
                assert data.shape == (data_size, len(clamped_nodes))
                key, key_shuffle = jax.random.split(key)
                idxs = jax.random.permutation(key_shuffle, jnp.arange(data_size))
                data = data[idxs]
                _n_batches = data_size // _bsz
                tot_len = _n_batches * _bsz
                batched_data = jnp.reshape(data[:tot_len], (_n_batches, _bsz, len(clamped_nodes))).astype(jnp.bool)
                return batched_data, _n_batches

            key, key_pos = jax.random.split(key, 2)
            batched_data_pos, n_batches = batch_data(key_pos, data_positive, bsz_positive, self.training_data_blocks)

            def body_fun(carry, key_and_data):
                _key, _data_pos = key_and_data

                _opt_state, _params = carry
                _model = eqx.tree_at(lambda m: (m.weights, m.biases), model, _params)
                key_train, key_init_pos, key_init_neg = jax.random.split(_key, 3)
                vals_free_pos = hinton_init(key_init_pos, _model, self.positive_sampling_blocks, (1, bsz_positive))
                vals_free_neg = hinton_init(
                    key_init_neg,
                    _model,
                    self.negative_sampling_blocks,
                    (bsz_negative,),
                )

                ebm = IsingTrainingSpec(
                    _model,
                    self.training_data_blocks,
                    [],
                    self.positive_sampling_blocks,
                    self.negative_sampling_blocks,
                    self.schedule_positive,
                    self.schedule_negative,
                )

                grad_w, grad_b, _, _ = estimate_kl_grad(
                    key_train, ebm, _model.nodes, model.edges, [_data_pos], [], vals_free_pos, vals_free_neg
                )

                grads = (grad_w, grad_b)
                # optax does not obey this
                with jax.numpy_dtype_promotion("standard"):
                    updates, _opt_state = self.optim.update(grads, _opt_state, _params)
                _weights, _biases = _params
                # todo: grads are negative vs. positive
                _weights += updates[0]
                _biases += updates[1]

                new_carry = _opt_state, (_weights, _biases)

                return new_carry, None

            params = model.weights, model.biases

            init_carry = opt_state, params

            keys = jax.random.split(key, n_batches)
            out_carry, _ = jax.lax.scan(body_fun, init_carry, (keys, batched_data_pos))

            opt_state, params = out_carry
            new_model = eqx.tree_at(lambda m: (m.weights, m.biases), model, params)

            return new_model, opt_state

        def compute_accuracy(
            key,
            model,
            bsz_per_digit,
        ):
            """Takes images separated into classes based on which digit they are and
            for each class computes the probability that the model assigns a 1 to the label
            of each digit. Based on this it computes the accuracy (the fraction of
            digits where the argmax of the output labels gives the correct digit)
            and records the average label probabilities for each digit.
            """
            accuracy = 0.0
            for i, digit in enumerate(self.target_classes):
                images = self.sep_images_test[digit][:bsz_per_digit].astype(jnp.bool)

                program = IsingSamplingProgram(
                    model, self.classification_sampling_blocks, self.classification_data_blocks
                )

                key, subkey1, subkey2 = jax.random.split(key, 3)

                init_free_states = hinton_init(subkey2, model, self.classification_sampling_blocks, (bsz_per_digit,))

                keys_samp = jax.random.split(subkey1, bsz_per_digit)

                samples = jax.vmap(
                    lambda k, i, d: sample_states(
                        k, program, self.accuracy_schedule, i, d, [self.classification_label_block]
                    )
                )(keys_samp, init_free_states, [images])[0]

                labels = samples.reshape(
                    samples.shape[0],
                    samples.shape[1],
                    self.num_label_spots,
                    len(self.target_classes),
                )
                labels = jnp.mean(labels, axis=(1, 2))
                generated_digit = jnp.argmax(labels, axis=1)
                # note that the generated label will not be the digit itself,
                # but the index of the digit in the list of target digits
                accuracy += jnp.mean(i == generated_digit)

            accuracy /= len(self.target_classes)
            return accuracy

        best_accuracy = 0.0
        opt_state = self.optim.init((self.init_model.weights, self.init_model.biases))

        model = self.init_model

        for i in range(self.n_epochs):
            model, opt_state = do_epoch_simplified(
                jax.random.key(0),
                model,
                50,
                25,
                self.train_data_filtered,
                opt_state,
            )

            accuracy = compute_accuracy(
                jax.random.key(2),
                model,
                500,
            )
            best_accuracy = max(best_accuracy, accuracy)

        self.assertGreater(best_accuracy, 0.9)
