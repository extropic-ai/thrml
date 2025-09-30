from typing import (
    Generic,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeAlias,
    TypeVar,
)

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, PyTree, Shaped

from .pgm import AbstractNode

_Node = TypeVar("_Node", bound=AbstractNode)
# hashable description of general pytrees
_PyTreeStruct: TypeAlias = tuple[
    PyTree,
    tuple[jax.ShapeDtypeStruct, ...],
]
_GlobalState: TypeAlias = PyTree[Shaped[Array, "nodes_global ?*state"], "_GlobalState"]
_State = PyTree[Shaped[Array, "nodes ?*state"], "State"]
_Node_SD = Mapping[Type[AbstractNode], PyTree[jax.ShapeDtypeStruct]]


class Block(Generic[_Node]):
    """
    A Block is the basic unit through which Gibbs sampling can operate.

    Each block represents a collection of nodes that can efficiently be sampled
    simultaneously in a JAX-friendly SIMD manner. In THRML, this means that the nodes must all be of the same type.

    **Attributes:**

    - `nodes`: the tuple of nodes that this block contains
    """

    nodes: tuple[_Node, ...]

    def __init__(self, nodes: Sequence[_Node]) -> None:
        nodes_tuple = tuple(nodes)
        if nodes_tuple:
            first_type = type(nodes_tuple[0])
            if {type(node) for node in nodes_tuple} != {first_type}:
                raise ValueError("All nodes in a block must be of the same type")
        self.nodes = nodes_tuple

    @property
    def node_type(self) -> Type[_Node]:
        if not self.nodes:
            raise ValueError(
                "Block is empty and doesn't have a node type. Most methods in thrml do not support empty blocks."
            )
        return type(self.nodes[0])

    def __getitem__(self, index: int) -> _Node:
        return self.nodes[index]

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterator[_Node]:
        return iter(self.nodes)

    def __contains__(self, item) -> bool:
        return item in self.nodes

    def __add__(self, other):
        if isinstance(other, Block):
            if self.nodes and other.nodes:
                if type(self.nodes[0]) is not type(other.nodes[0]):
                    raise ValueError("Cannot add Blocks of different node types")
            return Block(self.nodes + other.nodes)
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nodes={self.nodes!r})"


def _hash_pytree(x: PyTree[jax.ShapeDtypeStruct]) -> _PyTreeStruct:
    return (jax.tree.structure(x), tuple(jax.tree.leaves(x)))


class BlockSpec:
    """
    This contains the necessary mappings for logging indices of states and node types.

    This helps convert between block states and global states. A block state is a list
    of pytrees, where each pytree leaf has shape[0] = number of nodes in the block.
    The length of the block state is the number of blocks. The global state is a
    flattened version of this. Each pytree type is combined (regardless of which block
    they are in), to make a list of pytrees where each leaf shape[0] is the total
    number of nodes of that pytree shape. As an example, imagine an Ising model,
    every node is the same pytree (just a scalar array), as such the block state is
    a list of arrays where each array is the state of the block and the global state
    would be a length-1 list that contains an array of shape (total_nodes,).

    Why is this global/block representation necessary? The answer is that the global
    representation is preferred for operating over in many JAX cases, but requires
    careful indexing (to know where in this long array each block resides) and thus
    the block representation is more natural/easy to use for many users. Why is the
    global state easier to work with? Well consider sampling, in order to sample a
    block (or even just a node) we need to collect all the states of the neighboring
    nodes. If we only had the block state we would have to loop over the block state
    and collect from each block the neighbors, we would then pass this to the
    sampler. The sampler would then have to know the type of each block (to know
    what to do with the states) then for loop over the blocks in order to collect
    each. This (programmatically) is fine, but results in additional for loops that
    slow down JAX, compared to gathering indexes from a single array.


    **Attributes:**

    - `blocks`: the list of blocks this spec contains
    - `all_block_sds`: a SD is a single `_PyTreeStruct`. Each node/block has only
        one SD associated with it, but each node can have neighbors of many types.
        This is the SD of each block (in the same order as blocks, this internal
        ordering is quite important for bookkeeping). This list is just the list
        of SDs for each block (and thus has length = len(blocks)).
    - `global_sd_order`: the list of SDs, providing a SoT for the global ordering
    - `sd_index_map`: a dictionary mapping the SD to an integer in the
        `global_sd_order`. This is like calling `.index` on it.
    - `node_global_location_map`: a dictionary mapping a given node to a tuple.
        That tuple contains the global index (i.e. which element in the global
        list it is in) and the relative position in that pytree. That is to say,
        you can get the state of the node via
        `map(x[tuple[1]], global_repr[tuple[0]])`
    - `block_to_global_slice_spec`: a list over unique SDs (so length
        global_sd_order), where each list inside this is the list over blocks
        which contain that pytree. E.g. [[0, 1], [2]] indicates that blocks[0]
        and blocks[1] are both of pytree SD 0.
    - `node_shape_dtypes`: a dictionary mapping node types to hashable `_PyTreeStruct`
    - `node_shape_struct`: a dictionary mapping node types to pytrees of JAX-shaped
        dtype structs (just for user access, since the keys aren't hashable that
        creates issues for JAX in other areas.)

    """

    blocks: list[Block]
    all_block_sds: list[_PyTreeStruct]
    global_sd_order: list[_PyTreeStruct]
    sd_index_map: dict[_PyTreeStruct, int]
    node_global_location_map: dict[AbstractNode, tuple[int, int]]
    block_to_global_slice_spec: list[list[int]]
    node_shape_dtypes: dict[Type[AbstractNode], _PyTreeStruct]
    node_shape_struct: dict[Type[AbstractNode], PyTree[jax.ShapeDtypeStruct]]

    def __init__(
        self,
        blocks: list[Block],
        node_shape_dtypes: _Node_SD,
    ) -> None:
        """
        Create a BlockSpec from blocks.

        Based on the information passed in via node_shape_dtypes, determine the minimal global state that can be used
        to represent the blocks.

        **Arguments:**

        - `blocks`: the list of `Block`s that this specification operates on
        - `node_shape_dtypes`: the mapping of node types to their structures. This
                should be a pytree of `jax.ShapeDtypeStruct`s.
        """
        # variable types are assigned to nodes based on their class
        self.node_shape_struct = dict(node_shape_dtypes)
        self.node_shape_dtypes = {i: _hash_pytree(j) for i, j in node_shape_dtypes.items()}

        self.blocks = blocks

        # come up with an ordering of SDs for the global representation of the blocks
        all_sds = list({sd for sd in self.node_shape_dtypes.values()})
        self.global_sd_order = all_sds

        # map from SD to location in global representation
        self.sd_index_map = {sd: i for i, sd in enumerate(self.global_sd_order)}

        for block in blocks:
            if len(block) == 0:
                raise ValueError("Encountered an empty block in BlockSpec.")

            if block.node_type not in node_shape_dtypes:
                raise ValueError(f"Block with node type {block.node_type} not found in node_shape_dtypes.")

        self.all_block_sds = [self.node_shape_dtypes[block.node_type] for block in blocks]

        block_to_global_slice_spec = [[] for _ in self.global_sd_order]

        node_global_location_map = {}
        arr_ind_tracker = [0 for _ in self.global_sd_order]
        for block_idx, (block, sds) in enumerate(zip(blocks, self.all_block_sds)):
            block_len = len(block)

            sd_ind = self.sd_index_map[sds]
            start_ind = arr_ind_tracker[sd_ind]
            arr_ind_tracker[sd_ind] += block_len
            block_to_global_slice_spec[sd_ind].append(block_idx)
            for k, node in enumerate(block.nodes):
                if node in node_global_location_map:
                    raise RuntimeError("A node should not show up twice in the blocks input to BlockSpec.")
                node_global_location_map[node] = (sd_ind, start_ind + k)
        self.block_to_global_slice_spec = block_to_global_slice_spec
        self.node_global_location_map = node_global_location_map


def _stack(*args):
    if eqx.is_array(args[0]):
        if args[0].shape == ():
            return jnp.stack(args)
        # concate across node dim
        return jnp.concatenate(args, axis=0)
    else:
        assert all(args[0] == arg for arg in args[1:])
        return args[0]


def block_state_to_global(block_state: list[_State], spec: BlockSpec) -> list[_GlobalState]:
    """
    Convert block-local state to the global stacked representation.

    The block representation is a list where ``block_state[i]`` contains the
    state of ``spec.blocks[i]`` and every node occupies index 0 of its leaf.

    The global representation is a shorter list (one entry per distinct
    PyTree structure) in which all blocks with the same structure are
    concatenated along their node axis.

    **Arguments:**

    - `block_state`: State organised per block, same length as
        ``spec.blocks``.
    - `spec`: The [`thrml.BlockSpec`][] that defines the mapping.

    **Returns:**

    A list whose length equals
    ``len(spec.global_sd_order)``—the stacked global state.
    """
    # list is len(blocks), global_sd_order = len(unique pytrees) <= len(blocks)
    global_state = []
    for sd_indexes in spec.block_to_global_slice_spec:
        if not sd_indexes:
            global_state.append(None)
            continue

        collected = [block_state[i] for i in sd_indexes]

        # todo: should probably expand dims to be 1 to be consistent?
        if len(collected) == 1:
            global_state.append(collected[0])
        else:
            global_state.append(jax.tree.map(_stack, *collected))

    return global_state


def get_node_locations(nodes: Block, spec: BlockSpec) -> tuple[int, Int[Array, " nodes"]]:
    """
    Locate a contiguous set of nodes inside the global state.

    **Arguments:**

    - `nodes`: A [`thrml.Block`][] whose nodes you want locations for.
    - `spec`: The [`thrml.BlockSpec`][] generated from the same graph.

    **Returns:**

    Tuple ``(sd_index, positions)`` where

    * *sd_index* is the position inside the global list returned by
      [`thrml.block_state_to_global`][], and
    * *positions* is a 1D array with the indices each node
      occupies inside that particular PyTree.
    """

    # Make sure all nodes are of the same type
    # if len(set([type(node) for node in nodes])) > 1:
    #     raise ValueError("All nodes must be of the same type")

    node_sds = spec.node_shape_dtypes[nodes.node_type]

    sd_inds = spec.sd_index_map[node_sds]
    global_locs = [spec.node_global_location_map[node][1] for node in nodes]
    slices = jnp.array(global_locs)

    return sd_inds, slices


def from_global_state(
    global_state: list[_GlobalState],
    spec_from: BlockSpec,
    blocks_to_extract: list[Block],
) -> list[_State]:
    """
    Extract the states for a subset of blocks from a global state.

    **Arguments:**

    - `global_state`: A state produced by
        [`thrml.block_state_to_global(spec_from)`][].
    - `spec_from`: The [`thrml.BlockSpec`][] associated with *global_state*.
    - `blocks_to_extract`: The blocks whose node states should be returned.

    **Returns:**

    A list with one element per *blocks_to_extract*—each element is a PyTree
    with exactly ``len(block)`` nodes in its leading dimension.
    """

    all_sd_inds = []
    all_sd_slices = []
    for block in blocks_to_extract:
        sd_inds, slices = get_node_locations(block, spec_from)
        all_sd_inds.append(sd_inds)
        all_sd_slices.append(slices)

    return [
        jax.tree.map(lambda x: jnp.take(x, sls, axis=0), global_state[_sd_ind])
        for _sd_ind, sls in zip(all_sd_inds, all_sd_slices)
    ]


def make_empty_block_state(
    blocks: list[Block],
    node_shape_dtypes: _Node_SD,
    batch_shape: Optional[tuple] = None,
) -> list[_State]:
    """
    Allocate a zero-initialised block state.

    **Arguments:**

    - `blocks`: All blocks in the graph (order is preserved).
    - `node_shape_dtypes`: Maps every node class to its
        `jax.ShapeDtypeStruct` PyTree template.
    - `batch_shape`: Optional batch dimension(s) to prepend to every leaf.

    **Returns:**

    A list of PyTrees—one per *block*—whose leaves are
    ``zeros(batch_shape + (len(block),) + leaf.shape)``.
    """
    state = []
    for block in blocks:
        types = node_shape_dtypes[block.node_type]
        if batch_shape is None:
            this_state = jax.tree.map(
                lambda x: jnp.zeros(shape=(len(block), *x.shape), dtype=x.dtype),
                types,
            )
        else:
            this_state = jax.tree.map(
                lambda x: jnp.zeros(shape=(*batch_shape, len(block), *x.shape), dtype=x.dtype),
                types,
            )
        state.append(this_state)
    return state


def _check_pytree_compat(
    spec_tree,
    data_tree,
) -> tuple[int, ...] | None:
    """
    Verify that a PyTree of arrays matches up with a PyTree of ShapeDtypeStructs, up to a uniform batch shape.

    **Arguments:**

    - `spec_tree`: Pytree with `jax.ShapeDtypeStruct` leaves (at positions you want checked).
    - `data_tree`: Pytree with arrays at matching positions.

    **Returns:**

    The extracted batch shape if the two pytrees are compatible
    """

    if not jax.tree.structure(spec_tree) == jax.tree.structure(data_tree):
        raise RuntimeError("Tree structure mismatch between shape/dtype spec and data")

    spec_leaves, _ = jax.tree.flatten_with_path(spec_tree)
    val_leaves, _ = jax.tree.flatten_with_path(data_tree)

    batch_shape = None

    for (path, spec_leaf), (_, val_leaf) in zip(spec_leaves, val_leaves):
        if isinstance(spec_leaf, jax.ShapeDtypeStruct):
            if not eqx.is_array(val_leaf):
                raise RuntimeError("Array missing from data")

            vshape, vdtype = val_leaf.shape, val_leaf.dtype
            sshape, sdtype = spec_leaf.shape, spec_leaf.dtype

            val_shape_without_batch = () if not len(sshape) else vshape[-(len(sshape)) :]

            if val_shape_without_batch != sshape:
                raise RuntimeError("Shape of data mismatched with spec")

            cur_batch = vshape[: len(vshape) - len(sshape)]
            if batch_shape is None:
                batch_shape = cur_batch
            elif cur_batch != batch_shape:
                raise RuntimeError("Inconsistent batch shape in data")

            if vdtype != sdtype:
                raise RuntimeError(f"Data has incorrect type {vdtype} vs {sdtype}")

    return batch_shape


def verify_block_state(
    blocks: list[Block], states: list[_State], node_shape_dtypes: _Node_SD, block_axis: Optional[int] = None
) -> None:
    """
    Check that a state is what it should be given some blocks and node shape/dtypes.

    Passing incompatible state information into THRML functions can lead to unintended casting/other weird silent
    errors, so we should always check this.

    **Arguments:**

    - `blocks`: A list of Blocks.
    - `states`: A list of states to verify against blocks.
    - `node_shape_dtypes`: Maps every node class to its
        `jax.ShapeDtypeStruct` PyTree template.
    - `block_axis`: Index in the state batch shape at which to expect the block length.

    **Returns:**

    None. Raises RuntimeError if blocks and states are incompatible.
    """

    if not len(blocks) == len(states):
        raise RuntimeError("Number of states not equal to number of blocks")

    for block, state in zip(blocks, states):
        expected_sd = node_shape_dtypes[type(block.nodes[0])]
        batch_shape = _check_pytree_compat(expected_sd, state)
        assert batch_shape is not None
        if block_axis is not None:
            if not batch_shape[block_axis] == len(block.nodes):
                raise RuntimeError("State shape did not match detected block length")
