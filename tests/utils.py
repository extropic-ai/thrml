import itertools

import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Key

from thrml.block_management import Block
from thrml.block_sampling import BlockSamplingProgram, SamplingSchedule, sample_states
from thrml.models.ebm import AbstractEBM
from thrml.pgm import DEFAULT_NODE_SHAPE_DTYPES, CategoricalNode, SpinNode


def generate_all_states_binary(num_binary: int) -> Array:
    """All 0/1 states for num_binary variables -> shape (2^num_binary, num_binary)."""
    if num_binary == 0:
        return jnp.zeros((1, 0), dtype=jnp.bool_)
    combos = itertools.product([0, 1], repeat=num_binary)
    return jnp.array(list(combos), dtype=jnp.bool_)


def generate_all_states_categorical(num_categorical: int, n_categories: int) -> Array:
    """
    All 0..(n_categories-1) states -> shape (n_categories^num_categorical, num_categorical).
    """
    if num_categorical == 0:
        return jnp.zeros((1, 0), dtype=jnp.int32)
    combos = itertools.product(range(n_categories), repeat=num_categorical)
    return jnp.array(list(combos), dtype=jnp.int32)


def generate_all_states_bin_cat(num_binary: int, num_categorical: int, n_categories: int) -> tuple[Array, Array]:
    bin_states = generate_all_states_binary(num_binary)
    cat_states = generate_all_states_categorical(num_categorical, n_categories)

    final_batch_shape = (bin_states.shape[0], cat_states.shape[0])

    bin_states_expand = jnp.broadcast_to(jnp.expand_dims(bin_states, 1), (*final_batch_shape, bin_states.shape[1]))
    cat_states_expand = jnp.broadcast_to(jnp.expand_dims(cat_states, 0), (*final_batch_shape, cat_states.shape[1]))

    batch_size = final_batch_shape[0] * final_batch_shape[1]

    return (
        bin_states_expand.reshape((batch_size, bin_states.shape[-1])),
        cat_states_expand.reshape((batch_size, cat_states.shape[-1])),
    )


def count_samples(all_states_bin: Array, all_states_cat: Array, samples_bin: Array, samples_cat: Array) -> Array:
    count_dict = {}

    for i, (bin_state, cat_state) in enumerate(zip(all_states_bin, all_states_cat)):
        count_dict[(tuple(bin_state.tolist()), tuple(cat_state.tolist()))] = i
    counts = np.zeros(all_states_bin.shape[0], dtype=int)

    for bin_samp, cat_samp in zip(samples_bin, samples_cat):
        key = (tuple(bin_samp.tolist()), tuple(cat_samp.tolist()))
        if key in count_dict:
            counts[count_dict[key]] += 1

    return counts / samples_bin.shape[0]


def sample_and_compare_distribution(
    key: Key[Array, ""],
    ebm: AbstractEBM,
    program: BlockSamplingProgram,
    clamp_vals: list[Array],  # each block gets an array
    schedule: SamplingSchedule,
    n_cats: int,
) -> tuple[Array, Array]:
    init_free = []

    all_binary_nodes = []
    all_cat_nodes = []

    for block in program.gibbs_spec.free_blocks:
        dtype = program.gibbs_spec.node_shape_struct[block[0].__class__].dtype

        if isinstance(block.nodes[0], SpinNode):
            all_binary_nodes += block.nodes
        else:
            all_cat_nodes += block.nodes

        arr = jnp.zeros((len(block),), dtype=dtype)
        init_free.append(arr)

    all_node_sets = [all_binary_nodes, all_cat_nodes]
    all_node_types = [SpinNode, CategoricalNode]
    observe_blocks = []
    used = [False, False]

    for i, nodes in enumerate(all_node_sets):
        if len(nodes) > 0:
            used[i] = True
            observe_blocks.append(Block(nodes))

    all_block_samples = sample_states(
        key=key,
        program=program,
        schedule=schedule,
        init_state_free=init_free,
        state_clamp=clamp_vals,
        nodes_to_sample=observe_blocks,
    )

    counter = 0

    all_samples = []
    for use, typ in zip(used, all_node_types):
        sd = DEFAULT_NODE_SHAPE_DTYPES[typ]
        if use:
            all_samples.append(all_block_samples[counter])
            counter += 1
        else:
            all_samples.append(jnp.empty((schedule.n_samples, 0, *sd.shape), sd.dtype))

    all_bin_states, all_cat_states = generate_all_states_bin_cat(len(all_binary_nodes), len(all_cat_nodes), n_cats)

    empirical_dist = count_samples(all_bin_states, all_cat_states, *all_samples)

    energy_states = []
    all_states = [all_bin_states, all_cat_states]
    for state, use in zip(all_states, used):
        if use:
            energy_states.append(state)

    clamp_vals = jax.tree.map(lambda x: x.astype(jnp.int32), clamp_vals)
    energy_states = jax.tree.map(lambda x: x.astype(jnp.int32), energy_states)

    energies = jax.vmap(lambda x: ebm.energy([*x] + clamp_vals, observe_blocks + program.gibbs_spec.clamped_blocks))(
        energy_states
    )

    un = jnp.exp(-energies)

    return empirical_dist, un / jnp.sum(un)
