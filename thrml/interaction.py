import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from thrml.block_management import Block


class InteractionGroup(eqx.Module):
    """Defines computational dependencies for conditional sampling updates.

    An `InteractionGroup` specifies information that is required to update the state of some subset
    of the nodes of a PGM during a block sampling routine.

    More concretely, when the state of the node at head_nodes[i] is being updated, the sampler will receive the current
    state of the nodes at tail_nodes[k][i] for all k, and the ith element of each array in the Interaction PyTree
    (sliced along the first dimension).

    **Attributes:**

    - `head_nodes`: these are the nodes whose conditional updates should be affected by this InteractionGroup.
    - `tail_nodes`: these are the nodes whose state information is required to update `head_nodes`.
    - `interaction`: this specifies the parametric (independent of the state of the sampling program)
        required to update 'head_nodes'.
    """

    head_nodes: Block
    tail_nodes: list[Block]
    interaction: PyTree

    def __init__(self, interaction: PyTree, head_nodes: Block, tail_nodes: list[Block]):
        """Create an `InteractionGroup`.

        An `InteractionGroup` implements a group of directed interactions between nodes
        in a PGM sampling program.

        **Arguments:**

        - `interaction`: A PyTree specifying the static information associated with the
            interaction. The first dimension of every Array in interaction must be equal
            to the length of `head_nodes`.
        - `head_nodes`: The nodes whose update is affected by the interaction.
        - `tail_nodes`: The groups of nodes whose state is required to update
            `head_nodes`. Each block in this list of blocks is intended to be parallel
            to `head_nodes`. i.e, to update the state of head_nodes[i] during sampling
            we need state info about tail_nodes[k][i] for all values of k.
        """

        interaction_size = len(head_nodes.nodes)

        for block in tail_nodes:
            if not len(block.nodes) == interaction_size:
                raise RuntimeError("All tail node blocks must have the same length as head_nodes")

        def _get_dim(x):
            return (-1 if not len(x.shape) else x.shape[0]) if isinstance(x, jnp.ndarray) else interaction_size

        dims = jax.tree.leaves(jax.tree.map(_get_dim, interaction))
        if not all(dim == interaction_size for dim in dims):
            raise RuntimeError(
                "All arrays in interaction must have leading dimension equal to the length of head_nodes"
            )

        self.interaction = interaction
        self.head_nodes = head_nodes
        self.tail_nodes = tail_nodes
