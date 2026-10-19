import abc
from collections import defaultdict
from typing import TYPE_CHECKING, Callable, Protocol, Sequence, TypeVar

import equinox as eqx
import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Int, PyTree

from thrml.block_management import Block, block_state_to_global, from_global_state

if TYPE_CHECKING:
    from thrml.block_sampling import _State, BlockGibbsSpec

from thrml.pgm import AbstractNode

ObserveCarry = TypeVar("ObserveCarry", bound=PyTree)


class SamplingContext(Protocol):
    """The block layout available to portable observers.

    Both logical and executable programs expose this interface. JAX passes its
    compiled `BlockSamplingProgram` to observers, so JAX-specific observers may
    additionally inspect `.samplers` and `.per_block_*` fields.
    """

    @property
    def gibbs_spec(self) -> "BlockGibbsSpec":
        """Layout of every free and clamped block supplied to the observer."""
        ...


class AbstractObserver(eqx.Module):
    """
    Interface for objects that inspect the sampling program while it is running.

    An observer is called once per requested record and can maintain an explicit
    carry across calls, such as running sums or histogram buffers.

    A portable observer is a pure JAX function of `program.gibbs_spec`, the
    complete free and clamped states, its carry, and the logical record index.
    Immutable configuration is allowed; Python side effects and hidden mutable
    state are not. Any state transformations must also be pure.

    A backend may evaluate records individually or use an equivalent ordered
    fold over windows. Purity does not allow independent `vmap` calls when an
    observer depends on carry. Windows share carry and request-global indices,
    and each requested record is emitted exactly once. A new request restarts
    indices at zero, even when the physical chain continues.

    Emitted PyTrees must have a consistent structure, shape, and dtype across
    records. The first observation may initialize carry with a new dtype or
    structure; subsequent scanned carry must keep that structure, shape, and
    dtype. Carry and emitted outputs need not have the same structure. JAX
    observes the first record before its scan to preserve this initialization.
    """

    @abc.abstractmethod
    def __call__(
        self,
        program: SamplingContext,
        state_free: list[PyTree[Array]],
        state_clamped: list[PyTree[Array]],
        carry: ObserveCarry,
        iteration: Int[Array, ""],
    ) -> tuple[ObserveCarry, PyTree]:
        """Make an observation.

        The supplied state includes every free and clamped program block in
        layout order. Portable observers use only the layout and explicit inputs.
        Observers that inspect compiled samplers or interactions remain supported
        on JAX, but are outside the portable subset.

        **Arguments:**

        - `program`: The sampling program that is running when this function is called.
        - `state_free`: The current state of the free nodes involved in the sampling program.
        - `state_clamped`: The state of the clamped nodes involved in the sampling program.
        - `carry`: The "memory" available to this observer. This function should modify this PyTree to record
            information about the sampling program.
        - `iteration`: The record index, from zero to `n_samples - 1` within the
            request. This is not the number of sampling sweeps.

        **Returns:**

        A tuple, where the first element is the updated carry, and the second is a PyTree that will be
        recorded by the sampler, or `None` when nothing should be recorded.

        """
        return NotImplemented

    def init(self) -> PyTree:
        """Initialize the memory for the observer. Defaults to None."""
        return None


class StateObserver(AbstractObserver):
    """
    Portable observer which logs the raw state of some set of nodes.

    **Attributes:**

    - `blocks_to_sample`: the list of `Block`s which the states are logged for
    """

    blocks_to_sample: list[Block]

    def __call__(
        self,
        program: SamplingContext,
        state_free: list["_State"],
        state_clamped: list["_State"],
        carry: None,
        iteration: Int[Array, ""],
    ) -> tuple[None, PyTree]:
        """Simply returns the state of the blocks that are being logged to be recorded by the sampler."""
        global_state = block_state_to_global(state_free + state_clamped, program.gibbs_spec)
        sampled_state = from_global_state(global_state, program.gibbs_spec, self.blocks_to_sample)
        return None, sampled_state


def _f_identity(*x):
    return x[0]


class MomentAccumulatorObserver(AbstractObserver):
    r"""
    Portable observer that accumulates and updates the provided moments.

    It doesn't log any samples, and will only accumulate moments. Note that this observer does not
    scale the accumulated values by the number of times it was called. It simply records a running sum of a product
    of some state variables,

    $$\sum_i f(x_1^i) f(x_2^i) \dots f(x_N^i)$$

    The transformation must be pure for portability. `init()` creates floating
    zeros; the first observation casts carry to the product dtype. Backends must
    observe that first record before scanning subsequent records, including
    when records are partitioned into windows. Normalize the final sums once
    by the total record count, rather than averaging per-window means.

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
        program: SamplingContext,
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
