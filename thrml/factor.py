import abc
from typing import Sequence

import equinox as eqx
from ihoop.eqx import AbstractStrictModule
from jaxtyping import Array, PyTree

from thrml.block_management import Block
from thrml.block_sampling import AbstractBlockSamplingProgram, BlockGibbsSpec, _compile
from thrml.conditional_samplers import AbstractConditionalSampler
from thrml.interaction import InteractionGroup


class AbstractFactor(AbstractStrictModule):
    """A factor represents a batch of undirected interactions between sets of random variables.

    Concretely, this class implements a batch of factors defined over a bunch of parallel node groups. A single
    factor is defined over the nodes given by node_groups[k][i] for all k and a particular i. The defining trait of a
    factor is to produce InteractionGroups that affect each member of the factor in some way during the conditional
    updates of a block sampling program. As a user, you specify how this is done by implementing a
    concrete to_interaction_groups method for your child class.

    **Attributes:**

    - `node_groups`: the list of blocks that makes up this batch of factors.
    """

    node_groups: eqx.AbstractVar[list[Block]]

    def __check_init__(self):
        if not len(self.node_groups) > 0:
            raise RuntimeError("A factor should not be empty.")

        n_nodes = len(self.node_groups[0].nodes)

        for group in self.node_groups:
            if not len(group.nodes) == n_nodes:
                raise RuntimeError("Every block in node_groups must contain the same number of nodes.")

    @abc.abstractmethod
    def to_interaction_groups(self) -> list[InteractionGroup]:
        """Compile a factor to a set of directed interactions."""
        pass


class AbstractWeightedFactor(AbstractFactor):
    """A factor that is parameterized by a weight tensor.

    The leading dimension of the weights tensor must be the same length as the batch dimension of the factor (i.e
    the number of nodes in each of the node_groups).

    **Attributes:**

    - `weights`: the weight tensor.
    """

    weights: eqx.AbstractVar[Array]


def _compile_from_factors(
    gibbs_spec: BlockGibbsSpec,
    samplers: list[AbstractConditionalSampler],
    factors: Sequence[AbstractFactor],
    other_interaction_groups: list[InteractionGroup],
) -> tuple[list[list[PyTree]], list[list[Array]], list[list[list[int]]], list[list[list[Array]]]]:
    """Build interaction groups from `factors` (plus `other_interaction_groups`)."""
    interaction_groups = list(other_interaction_groups)
    for factor in factors:
        interaction_groups += factor.to_interaction_groups()
    return _compile(gibbs_spec, samplers, interaction_groups)


class FactorSamplingProgram(AbstractBlockSamplingProgram):
    """A sampling program built out of factors.

    This breaks each factor passed to it down into interaction groups and uses them to build the same compiled
    representation held by a `BlockSamplingProgram`.
    """

    gibbs_spec: BlockGibbsSpec
    samplers: list[AbstractConditionalSampler]
    per_block_interactions: list[list[PyTree]]
    per_block_interaction_active: list[list[Array]]
    per_block_interaction_global_inds: list[list[list[int]]]
    per_block_interaction_global_slices: list[list[list[Array]]]

    def __init__(
        self,
        gibbs_spec: BlockGibbsSpec,
        samplers: list[AbstractConditionalSampler],
        factors: Sequence[AbstractFactor],
        other_interaction_groups: list[InteractionGroup],
    ):
        """Create a FactorSamplingProgram.

        **Arguments:**

        - `gibbs_spec`: A division of some PGM into free and clamped blocks.
        - `samplers`: The update rule to use for each free block in gibbs_spec.
        - `factors`: The factors to use to build this sampling program.
        - `other_interaction_groups`: Other interaction groups to include in your program alongside what the
            factors produce.
        """
        self.gibbs_spec = gibbs_spec
        self.samplers = samplers
        (
            self.per_block_interactions,
            self.per_block_interaction_active,
            self.per_block_interaction_global_inds,
            self.per_block_interaction_global_slices,
        ) = _compile_from_factors(gibbs_spec, samplers, factors, other_interaction_groups)
