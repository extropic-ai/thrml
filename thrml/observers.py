import abc
from collections import defaultdict
from typing import Callable, Sequence, TypeVar

import jax
import numpy as np
from ihoop.eqx import AbstractStrictModule
from jax import numpy as jnp
from jaxtyping import Array, Int, Key, PyTree, Shaped

from thrml.block_management import Block, block_state_to_global, from_global_state
from thrml.block_sampling import AbstractBlockSamplingProgram, SamplingSchedule, _run_blocks, _State
from thrml.conditional_samplers import AbstractConditionalSampler
from thrml.pgm import AbstractNode

ObserveCarry = TypeVar("ObserveCarry", bound=PyTree)


class AbstractObserver(AbstractStrictModule):
    """
    Interface for objects that inspect the sampling program while it is running.

    A concrete Observer is called once per block-sampling iteration and can maintain an
    arbitrary "carry" state across calls (e.g. running averages, histogram
    buffers, log-probs, etc.).
    """

    @abc.abstractmethod
    def __call__(
        self,
        program: AbstractBlockSamplingProgram,
        state_free: list[PyTree[Array]],
        state_clamped: list[PyTree[Array]],
        carry: ObserveCarry,
        iteration: Int[Array, ""],
    ) -> tuple[ObserveCarry, PyTree]:
        """Make an observation.

        This function is called at the end of a block-sampling iteration and can record information about the
        current state of the sampling program that might be useful for something later.

        **Arguments:**

        - `program`: The sampling program that is running when this function is called.
        - `state_free`: The current state of the free nodes involved in the sampling program.
        - `state_clamped`: The state of the clamped nodes involved in the sampling program.
        - `carry`: The "memory" available to this observer. This function should modify this PyTree to record
            information about the sampling program.
        - `iteration`: How many iterations of block sampling have happened before this function was called.

        **Returns:**

        A tuple, where the first element is the updated carry, and the second is a PyTree that will be
        recorded by the sampler.

        """
        return NotImplemented

    @abc.abstractmethod
    def init(self) -> PyTree:
        """Initialize the memory for the observer."""
        raise NotImplementedError


class StateObserver(AbstractObserver):
    """
    Observer which logs the raw state of some set of nodes.

    **Attributes:**

    - `blocks_to_sample`: the list of `Block`s which the states are logged for
    """

    blocks_to_sample: list[Block]

    def __call__(
        self,
        program: AbstractBlockSamplingProgram,
        state_free: list[_State],
        state_clamped: list[_State],
        carry: None,
        iteration: Int[Array, ""],
    ) -> tuple[None, PyTree]:
        """Simply returns the state of the blocks that are being logged to be recorded by the sampler."""
        global_state = block_state_to_global(state_free + state_clamped, program.gibbs_spec)
        sampled_state = from_global_state(global_state, program.gibbs_spec, self.blocks_to_sample)
        return None, sampled_state

    def init(self) -> None:
        return None


def _f_identity(*x):
    return x[0]


class MomentAccumulatorObserver(AbstractObserver):
    r"""
    Observer that accumulates and updates the provided moments.

    It doesn't log any samples, and will only accumulate moments. Note that this observer does not
    scale the accumulated values by the number of times it was called. It simply records a running sum of a product
    of some state variables,

    $$\sum_i f(x_1^i) f(x_2^i) \dots f(x_N^i)$$


    **Attributes:**

    - `blocks_to_sample`: the blocks to accumulate the moments over. These
        are for constructing the final state, and aren't truly "blocks"
        in the algorithmic sense (they can be connected to each other).
        There is one block per node type.
    - `flat_nodes_list`: a list of all of the nodes in the moments (each
        occurring only once, so len(set(x)) = len(x)).
    - `flat_to_type_slices_list`: a list over node types in which each element
        is an array of indices of the `flat_node_list` which that type
        corresponds to
    - `flat_to_full_moment_slices`: a list over moment types in which each
        element is a 2D array, which matches the shape of the `moment_spec[i]`
        and of which each element is the index in the `flat_node_list`.
    - `f_transform`: the element-wise transformation $f$ to apply to sample values before
        accumulation.

    """

    blocks_to_sample: list[Block]
    flat_nodes_list: list[AbstractNode]
    flat_to_type_slices_list: list[Int[Array, " nodes_in_slice"]]
    flat_to_full_moment_slices: list[Int[Array, "num_groups nodes_in_moment"]]
    f_transform: Callable

    def __init__(self, moment_spec: Sequence[Sequence[Sequence[AbstractNode]]], f_transform: Callable = _f_identity):
        r"""
        Create a MomentAccumulatorObserver.

        **Arguments:**

        - `moment_spec`: A 3 depth sequence. The first is a sequence
            over different moment types. A given moment type should have the same
            number of nodes in each moment. Then for each moment type, there is a
            sequence over moments. Each given moment is defined by a certain set
            of nodes.

            For example, to get the first and second moments on a simple o-o graph,
            it would be

            [
                [(node1,), (node2,)],
                [(node1, node2)]
            ]
        - `f_transform`: A function that takes in (state, blocks) and returns something with the same structure as
            state. This is used to apply functions to the samples before moments are computed. i.e this function
            defines a transformation of the state variable $y=f(x)$, such that the accumulated moments
            are of the form $\langle f(x_1) f(x_2) \rangle$.
        """

        self.f_transform = f_transform

        flat_nodes_list = []
        node_to_flat_idx = {}
        flat_to_full_moment_slices = []
        nodes_by_type = defaultdict(list)
        flat_to_type_slices = defaultdict(list)

        for i, moment in enumerate(moment_spec):
            # moment = tuple of “rows” => each row is a tuple of nodes
            shape = (len(moment), len(moment[0]))
            moment_slice = np.zeros(shape, dtype=int)

            for j, nodes in enumerate(moment):
                for k, node in enumerate(nodes):
                    # node_to_flat_idx[node] is the integer index assigned
                    idx = node_to_flat_idx.get(node, -1)
                    if idx == -1:
                        idx = len(flat_nodes_list)
                        node_to_flat_idx[node] = idx
                        flat_nodes_list.append(node)
                    moment_slice[j, k] = idx
                    nodes_by_type[node.__class__].append(node)
                    flat_to_type_slices[node.__class__].append(node_to_flat_idx[node])

            flat_to_full_moment_slices.append(jnp.array(moment_slice, dtype=int))

        blocks_to_sample = []
        flat_to_type_slices_list = []

        for node_type, nodes in nodes_by_type.items():
            blocks_to_sample.append(Block(nodes))
            type_slice = jnp.array(flat_to_type_slices[node_type], dtype=int)
            flat_to_type_slices_list.append(type_slice)

        self.flat_nodes_list = flat_nodes_list
        self.flat_to_full_moment_slices = flat_to_full_moment_slices
        self.blocks_to_sample = blocks_to_sample
        self.flat_to_type_slices_list = flat_to_type_slices_list

    def __call__(
        self,
        program: AbstractBlockSamplingProgram,
        state_free: list[PyTree[Array]],
        state_clamped: list[PyTree[Array]],
        carry: list[Array],
        iteration: Int[Array, ""],
    ) -> tuple[list[Array], PyTree]:
        """Accumulate the moments via `carry`. Does not return anything for the sampler to write down."""
        global_state = block_state_to_global(state_free + state_clamped, program.gibbs_spec)

        sampled_state = from_global_state(global_state, program.gibbs_spec, self.blocks_to_sample)

        sampled_state = self.f_transform(sampled_state, self.blocks_to_sample)
        sampled_state = list(sampled_state)

        flat_state = jnp.zeros(len(self.flat_nodes_list))
        result_type = jnp.result_type(*jax.tree.leaves(sampled_state))
        for i, type_slice in enumerate(self.flat_to_type_slices_list):
            if i == 0:
                flat_state = flat_state.astype(result_type)
            state = sampled_state[i]
            flat_state = flat_state.at[type_slice].set(state)

        def accumulate_moment(mem_entry, sl):
            update = jnp.prod(flat_state[sl], axis=1)
            return mem_entry.astype(update.dtype) + update

        mem = jax.tree.map(accumulate_moment, carry, self.flat_to_full_moment_slices)

        return mem, None

    def init(self) -> list[Array]:
        """Initialize the memory that will store the accumulated values."""
        return jax.tree.map(
            lambda x: jnp.zeros(x.shape[:1], dtype=float),
            self.flat_to_full_moment_slices,
        )


def sample_with_observation(
    key: Key[Array, ""],
    program: AbstractBlockSamplingProgram,
    schedule: SamplingSchedule,
    init_chain_state: list[PyTree[Shaped[Array, "nodes ?*state"]]],
    state_clamp: list[_State],
    observation_carry_init: ObserveCarry,
    f_observe: AbstractObserver,
) -> tuple[ObserveCarry, list[PyTree[Shaped[Array, "n_samples nodes ?*state"]]]]:
    """Run the full chain and call an Observer after every recorded sample.

    **Arguments:**

    - `key`: RNG key.
    - `program`: The sampling program.
    - `schedule`: Warm-up length, number of samples, number of steps between samples.
    - `init_chain_state`: Initial free-block state.
    - `state_clamp`: Clamped-block state.
    - `observation_carry_init`: Initial carry handed to `f_observe`.
    - `f_observe`: Observer instance.

    **Returns:**

    - Tuple `(final_observer_carry, samples)` where `samples` is a PyTree whose
        leading axis has size `schedule.n_samples`.
    """
    # run warmup
    sampler_states = jax.tree.map(
        lambda x: x.init(),
        program.samplers,
        is_leaf=lambda a: isinstance(a, AbstractConditionalSampler),
    )
    key, subkey = jax.random.split(key, 2)
    warmup_state, warmup_sampler_states = _run_blocks(
        subkey,
        program,
        init_chain_state,
        state_clamp,
        schedule.n_warmup,
        sampler_states,
    )
    mem, warmup_observation = f_observe(program, warmup_state, state_clamp, observation_carry_init, jnp.array(0))

    if schedule.n_samples <= 1:
        warmup_observation = jax.tree.map(lambda x: x[None], warmup_observation)
        return mem, warmup_observation

    # collect samples

    def body_fn(carry, input):
        (prev_state, prev_sampler_state), _mem = carry

        _key, i = input

        new_state, new_sampler_state = _run_blocks(
            _key,
            program,
            prev_state,
            state_clamp,
            schedule.steps_per_sample,
            prev_sampler_state,
        )
        _mem, observe_out = f_observe(program, new_state, state_clamp, _mem, i)
        new_carry = ((new_state, new_sampler_state), _mem)
        return new_carry, observe_out

    keys = jax.random.split(key, schedule.n_samples - 1)
    outer_iters = jnp.arange(1, schedule.n_samples)

    inputs = (keys, outer_iters)

    (_, mem_out), observed_results = jax.lax.scan(body_fn, ((warmup_state, warmup_sampler_states), mem), inputs)

    # need to prepend the first observation from the warmup
    def prepend_warmup_observation(_warmup, _rest):
        return jnp.concatenate([_warmup[None], _rest], axis=0)

    observed_results = jax.tree.map(prepend_warmup_observation, warmup_observation, observed_results)

    return mem_out, observed_results


def sample_states(
    key: Key[Array, ""],
    program: AbstractBlockSamplingProgram,
    schedule: SamplingSchedule,
    init_state_free: list[PyTree[Shaped[Array, "nodes ?*state"]]],
    state_clamp: list[_State],
    nodes_to_sample: list[Block],
) -> list[PyTree[Shaped[Array, "n_samples nodes ?*state"]]]:
    """Convenience wrapper to collect state information for nodes_to_sample only.

    Internally builds a [`thrml.StateObserver`][], runs
    [`thrml.sample_with_observation`][], and returns a stacked tensor of shape
    `(schedule.n_samples, ...)`.
    """
    f_observe = StateObserver(nodes_to_sample)
    carry_init = f_observe.init()

    mem_out, results_out = sample_with_observation(
        key,
        program,
        schedule,
        init_state_free,
        state_clamp,
        carry_init,
        f_observe,
    )

    return results_out
