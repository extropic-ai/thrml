from collections import defaultdict
from typing import Type

import equinox as eqx
import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Key, PyTree

from thrml.block_management import Block, BlockSpec, from_global_state
from thrml.block_sampling import _State
from thrml.conditional_samplers import (
    BernoulliConditional,
    SoftmaxConditional,
)
from thrml.factor import WeightedFactor
from thrml.interaction import InteractionGroup
from thrml.models.ebm import EBMFactor
from thrml.pgm import AbstractNode


class DiscreteEBMInteraction(eqx.Module):
    """An interaction that shows up when sampling from discrete-variable EBMs.

    **Attributes:**

    - `n_spin`: the number of spin states involved in the interaction.
    - `weights`: the weight tensor associated with this interaction.
    """

    n_spin: int
    weights: Array


def _spin_product(spin_vals: list[Array]):
    """Take the product of a bunch of spin states. If the input has length 0 return 1."""

    if len(spin_vals) == 0:
        return jnp.array(1)

    return jnp.prod(2 * jnp.stack(spin_vals, -1).astype(jnp.int8) - 1, axis=-1)


def _batch_gather(x, *idx):
    n = len(idx)

    if n == 0:
        return x

    batch_shape = x.shape[:-n]

    x_flat = x.reshape((-1, *x.shape[-n:]))
    idx_flat = tuple(y.flatten() for y in idx)

    batch_idx = jnp.arange(idx_flat[0].shape[0])

    return x_flat[(batch_idx,) + idx_flat].reshape(batch_shape)


class DiscreteEBMFactor(EBMFactor, WeightedFactor):
    """Implements batches of energy function terms of the form s_1 * ... * s_M * W[c_1, ..., c_N],
    where the s_i are spin variables and the c_i are categorical variables.

    No variable should show up twice in any given interaction. If this happens, the result of sampling from a model
    that includes the bad factor might not agree with the Boltzmann distribution. For example, the interaction
    w * s_1 * s_1 * s_2 would violate this rule because s_1 shows up twice. To allow you to do something weird if
    you want to, this condition has not been enforced in the code.


    **Attributes:**

    - `spin_node_groups`: the node groups involved in the batch of factors that represent spin-valued random variables.
    - `categorical_node_groups`: the node groups involved in the batch of factors that represent categorical-valued
        random variables.
    - `weights`: the batch of weight tensors W associated with the factors we are implementing. `weights` should have
        leading dimension b, where b is number of nodes in each element of `spin_node_groups` and
        `categorical_node_groups`. This tensor has shape [b, x_1, ..., x_N] where b is the number of nodes
        in each block and N is the length of `categorical_node_groups`.
    - `is_spin`: a map that indicates if a given node type represents a spin-valued random variable or not.
    """

    spin_node_groups: list[Block]
    categorical_node_groups: list[Block]
    weights: Array
    is_spin: dict[Type[AbstractNode], bool]

    def __init__(self, spin_node_groups: list[Block], categorical_node_groups: list[Block], weights: Array):
        """
        Create a `DiscreteEBMFactor`.

        **Arguments:**

        - `spin_node_groups`: The spin node groups
        - `categorical_node_groups`: The categorical node groups
        - `weights`: The interaction weight tensor
        """

        WeightedFactor.__init__(self, weights, spin_node_groups + categorical_node_groups)

        is_spin = defaultdict(lambda: False)

        # remember which node types are spin
        for group in spin_node_groups:
            is_spin[type(group.nodes[0])] = True

        for group in categorical_node_groups:
            curr_type = type(group.nodes[0])
            if is_spin[curr_type]:
                raise RuntimeError("A node cannot be both categorical and spin.")
            is_spin[curr_type] = False

        # don't want this to recognize nodes that haven't been seen here
        # not sure how that would happen but better to be careful
        self.is_spin = dict(is_spin)
        self.spin_node_groups = spin_node_groups
        self.categorical_node_groups = categorical_node_groups

        if not len(weights.shape) == 1 + len(categorical_node_groups):
            raise RuntimeError(
                "The shape of the weight tensor must be [b, x_1, ..., x_k], where"
                "k is the length of categorical_node_groups."
            )

        self.weights = weights

    def to_interaction_groups(self) -> list[InteractionGroup]:
        """Produce interaction groups that implement this factor.

        In this case, we have to treat the spin and categorical node groups slightly differently.
        """
        interaction_groups = []

        n_spin = len(self.spin_node_groups)
        n_cat = len(self.categorical_node_groups)
        n_total = n_spin + n_cat

        # handle the interaction groups with spin head nodes
        if n_spin > 0:
            spin_inds = list(range(len(self.spin_node_groups)))
            spin_combos = [(x, spin_inds[:i] + spin_inds[i + 1 :]) for i, x in enumerate(spin_inds)]

            all_head_nodes = []
            all_tail_nodes = [[] for _ in range(n_total - 1)]
            for combo in spin_combos:
                all_head_nodes += self.spin_node_groups[combo[0]].nodes
                for i, tail_ind in enumerate(combo[1]):
                    all_tail_nodes[i] += self.spin_node_groups[tail_ind].nodes
                for j, cat_group in enumerate(self.categorical_node_groups):
                    all_tail_nodes[n_spin - 1 + j] += cat_group.nodes

            tiler = [1] * len(self.weights.shape)
            tiler[0] = n_spin

            rep_weights = jnp.tile(self.weights, tiler)
            interaction_groups.append(
                InteractionGroup(
                    DiscreteEBMInteraction(n_spin - 1, rep_weights),
                    Block(all_head_nodes),
                    [Block(x) for x in all_tail_nodes],
                )
            )

        # handle the interaction groups with categorical head nodes
        if n_cat > 0:
            cat_inds = list(range(len(self.categorical_node_groups)))
            cat_combos = [(x, cat_inds[:i] + cat_inds[i + 1 :]) for i, x in enumerate(cat_inds)]

            for combo in cat_combos:
                head_nodes = self.categorical_node_groups[combo[0]]

                cat_blocks = [self.categorical_node_groups[i] for i in combo[1]]

                reind = (0, combo[0] + 1, *[x + 1 for x in combo[1]])

                weights_reind = jnp.moveaxis(self.weights, reind, list(range(len(reind))))

                interaction_groups.append(
                    InteractionGroup(
                        DiscreteEBMInteraction(n_spin, weights_reind), head_nodes, self.spin_node_groups + cat_blocks
                    )
                )

        return interaction_groups

    def energy(self, global_state: list[Array], block_spec: BlockSpec):
        """Compute the energy associated with this factor.

        In this case, that is the sum of terms like s_1 * ... * s_M * W[c_1, ..., c_N].
        """
        spin_vals = from_global_state(global_state, block_spec, self.spin_node_groups)
        cat_vals = from_global_state(global_state, block_spec, self.categorical_node_groups)
        spin_prod = _spin_product(spin_vals)
        weights = _batch_gather(self.weights, *cat_vals)
        return -jnp.sum(weights * spin_prod.astype(weights.dtype))


def _merge_groups(groups, n_tail_groups):
    if len(groups) == 0:
        return groups

    all_head = []
    all_tail = [[] for _ in range(n_tail_groups)]
    all_weights = []
    for group in groups:
        all_head += group.head_nodes.nodes
        for i, block in enumerate(group.tail_nodes):
            all_tail[i] += block.nodes
        all_weights.append(group.interaction.weights)

    return [
        InteractionGroup(
            DiscreteEBMInteraction(groups[0].interaction.n_spin, jnp.concatenate(all_weights, axis=0)),
            Block(all_head),
            [Block(x) for x in all_tail],
        )
    ]


class SquareDiscreteEBMFactor(DiscreteEBMFactor):
    """A discrete factor with a square interaction weight tensor (shape [b, x, x, ..., x]).

    If a discrete factor is square, the interaction groups corresponding to different choices of the head
    node blocks can be merged. This could yield smaller XLA programs and improved runtime performance via
    more efficient use of accelerators.
    """

    def __init__(self, spin_node_groups: list[Block], categorical_node_groups: list[Block], weights: Array):
        """Enforce that the weights are actually square."""
        super().__init__(spin_node_groups, categorical_node_groups, weights)

        if len(weights.shape) > 2:
            target_shape = weights.shape[1]
            for shape in weights.shape[1:]:
                if not shape == target_shape:
                    raise RuntimeError("Interaction tensor is not square.")

    def to_interaction_groups(self) -> list[InteractionGroup]:
        """Call the parent class to_interaction_groups, and merge the results."""
        groups = super().to_interaction_groups()

        spin_groups = []
        cat_groups = []

        for group in groups:
            if self.is_spin[type(group.head_nodes[0])]:
                spin_groups.append(group)
            else:
                cat_groups.append(group)

        n_tail = len(self.node_groups) - 1

        return _merge_groups(spin_groups, n_tail) + _merge_groups(cat_groups, n_tail)


class SpinEBMFactor(SquareDiscreteEBMFactor):
    """A `DiscreteEBMFactor` that involves only spin variables."""

    def __init__(self, node_groups: list[Block], weights: Array):
        super().__init__(node_groups, [], weights)


class CategoricalEBMFactor(DiscreteEBMFactor):
    """A `DiscreteEBMFactor` that involves only categorical variables."""

    def __init__(self, node_groups: list[Block], weights: Array):
        super().__init__([], node_groups, weights)


class SquareCategoricalEBMFactor(SquareDiscreteEBMFactor):
    """A `DiscreteEBMFactor` that involves only categorical variables that also has a square weight tensor."""

    def __init__(self, node_groups: list[Block], weights: Array):
        super().__init__([], node_groups, weights)


def _batch_gather_with_k(x, *idx):
    n = len(idx)
    batch_shape = x.shape[:-n]

    if len(batch_shape) == 0:
        return x

    k = batch_shape[-1]

    new_idx = [jnp.broadcast_to(jnp.expand_dims(y, -1), (*y.shape, k)).flatten() for y in idx]

    return _batch_gather(x.reshape((np.prod(batch_shape), *x.shape[-n:])), *new_idx).reshape(batch_shape)


def _split_states(states, n_spin):
    states_spin, states_cat = states[:n_spin], states[n_spin:]

    # make sure spin values are actually bool, cat values are actually unsigned
    # violating this could lead to some weird behaviour

    def _validate(check_states, name, ex_type):
        for state in check_states:
            if not (len(state.shape) == 2):
                raise RuntimeError("{} states must be scalar.".format(name))
            if not jnp.isdtype(state.dtype, ex_type):
                raise RuntimeError("{} states must be {}.".format(name, ex_type))

    _validate(states_spin, "Spin", "bool")
    _validate(states_cat, "Categorical", "unsigned integer")

    return states_spin, states_cat


class SpinGibbsConditional(BernoulliConditional):
    r"""A conditional update for spin-valued random variables that will perform a Gibbs sampling update given one or
    more `DiscreteEBMInteractions`.

    This function can be extended to handle a broader class of interactions via inheritance. Specifically, a
    child class can override the `compute_parameters` method defined here, compute contributions to $\gamma$
    from other types of interactions, and then call this method to take into account the contributions from
    `DiscreteEBMInteractions`."""

    def compute_parameters(
        self,
        key: Key,
        interactions: list[PyTree],
        active_flags: list[Array],
        states: list[list[_State]],
        sampler_state: None,
        output_sd: PyTree[jax.ShapeDtypeStruct],
    ) -> PyTree:
        r"""Compute the parameter $\gamma$ of a spin-valued Bernoulli distribution given DiscreteEBMInteractions:

        $$\gamma = \sum_i s_1^i \dots s_K^i \: W^i[x_1^i, \dots, x_M^i]$$

        where the sum over $i$ is over all the `DiscreteEBMInteractions` seen by this function.
        """

        gamma = jnp.zeros(output_sd.shape)
        for i, (interaction, active, state) in enumerate(zip(interactions, active_flags, states)):
            if isinstance(interaction, DiscreteEBMInteraction):
                state_bin, state_cat = _split_states(state, interaction.n_spin)

                weights = _batch_gather(interaction.weights, *state_cat)
                spin_prod = _spin_product(state_bin).astype(weights.dtype)
                active = active.astype(weights.dtype)
                if i == 0:
                    gamma = gamma.astype(weights.dtype)

                gamma += jnp.sum(weights * active * spin_prod, axis=-1)
            else:
                raise RuntimeError("Unsupported interaction found")
        return gamma, sampler_state


class CategoricalGibbsConditional(SoftmaxConditional):
    """A conditional update for categorical random variables that will perform a Gibbs sampling update given one or
        more `DiscreteEBMInteractions`.

    This function can be extended to handle other interactions in the same way as [`thrml.models.SpinGibbsConditional`][].

    **Attributes:**

    - `n_categories`: how many categories are involved in the softmax distribution this sampler will sample from.
    """

    n_categories: int

    def compute_parameters(
        self,
        key: Key,
        interactions: list[PyTree],
        active_flags: list[Array],
        states: list[list[_State]],
        sampler_state: None,
        output_sd: PyTree[jax.ShapeDtypeStruct],
    ) -> PyTree:
        r"""Compute the parameter $\theta$ of a softmax distribution given DiscreteEBMInteractions:

        $$\theta = \sum_i s_1^i \dots s_K^i \: W^i[:, x_1^i, \dots, x_M^i]$$

        where the sum over $i$ is over all the `DiscreteEBMInteractions` seen by this function.
        """

        theta = jnp.zeros((*output_sd.shape, self.n_categories))
        for i, (interaction, active, state) in enumerate(zip(interactions, active_flags, states)):
            if isinstance(interaction, DiscreteEBMInteraction):
                state_bin, state_cat = _split_states(state, interaction.n_spin)

                weights = _batch_gather_with_k(interaction.weights, *state_cat)
                spin_prod = jnp.expand_dims(_spin_product(state_bin), -1).astype(weights.dtype)
                if i == 0:
                    theta = theta.astype(weights.dtype)

                theta += jnp.sum(spin_prod * weights * jnp.expand_dims(active, -1).astype(weights.dtype), axis=-2)

            else:
                raise RuntimeError("Unsupported interaction found")

        return theta, sampler_state
