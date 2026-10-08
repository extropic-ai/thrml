import abc
from typing import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree, Shaped

from thrml.block_management import Block, BlockSpec, block_state_to_global
from thrml.block_sampling import _SD, _State
from thrml.factor import AbstractFactor
from thrml.pgm import DEFAULT_NODE_SHAPE_DTYPES


class AbstractEBM(eqx.Module):
    """
    Something that has a well-defined energy function (map from a state to a scalar).
    """

    @abc.abstractmethod
    def energy(self, state: list[_State], blocks: list[Block]) -> Float[Array, ""]:
        """Evaluate the energy function of the EBM given some state information.

        **Arguments:**

        - `state`: The state for which to evaluate the energy function. Must be compatible with `blocks`.
        - `blocks`: Specifies how the information in `state` is organized.

        **Returns:**

        A scalar representing the energy value associated with `state`.
        """
        raise NotImplementedError


class EBMFactor(AbstractFactor):
    """A factor that defines an energy function."""

    @abc.abstractmethod
    def energy(self, global_state: list[Array], block_spec: BlockSpec) -> Float[Array, ""]:
        """Evaluate the energy function of the factor.

        **Arguments:**

        - `global_state`: The state information to use to evaluate the energy function.
            Is a global state of `block_spec`.
        - `block_spec`: The `BlockSpec` used to generate `global_state`.
        """
        raise NotImplementedError


class AbstractFactorizedEBM(AbstractEBM):
    r"""An EBM that is made up of Factors, i.e., an EBM with an energy function like,

    $$\mathcal{E}(x) = \sum_i \mathcal{E}^i(x)$$

    where the sum over $i$ is taken over factors.

    Child classes must define a property which returns a list of
    factors that substantiate the EBM.

    **Attributes:**

    - `node_shape_dtypes`: the shape/dtypes of the nodes involved in this EBM. Used to generate the BlockSpec that
        defines the global state that factors receive to compute energy.
    """

    node_shape_dtypes: _SD

    def __init__(self, node_shape_dtypes: _SD = DEFAULT_NODE_SHAPE_DTYPES):
        self.node_shape_dtypes = node_shape_dtypes

    def energy(self, state: list[_State], blocks: list[Block]) -> Float[Array, ""]:
        block_spec = BlockSpec(blocks, self.node_shape_dtypes)
        global_state = block_state_to_global(state, block_spec)
        energy = jnp.array(0.0)
        for factor in self.factors:
            energy += factor.energy(global_state, block_spec)
        return energy

    @property
    @abc.abstractmethod
    def factors(self) -> list[EBMFactor]:
        """A concrete implementation of this class must define this method that returns a list of factors that
        substantiate the EBM."""
        raise NotImplementedError


class FactorizedEBM(AbstractFactorizedEBM):
    """An EBM that is defined by a concrete list of factors.

    **Attributes:**

    - `_factors`: the list of factors that defines this EBM.
    """

    _factors: list[EBMFactor]

    def __init__(self, factors: list[EBMFactor], node_shape_dtypes: _SD = DEFAULT_NODE_SHAPE_DTYPES):
        super().__init__(node_shape_dtypes)
        self._factors = factors

    @property
    def factors(self):
        return self._factors


_BatchedState: TypeAlias = PyTree[Shaped[Array, "samples nodes ?*state"], "_BatchedState"]


def contrastive_divergence_loss(
    model: AbstractFactorizedEBM,
    positive_samples: list[_BatchedState],
    negative_samples: list[_BatchedState],
    blocks: list[Block],
) -> Float[Array, ""]:
    r"""Compute the contrastive-divergence energy-difference surrogate.

    The model defines energy for a distribution proportional to exp(-energy).
    The loss is the mean positive-phase energy minus the mean negative-phase energy.
    It is an energy-difference surrogate, not a normalized likelihood or a convergence
    metric. Gradients pass through the model energy but not through sampling. The model
    remains responsible for temperature scaling, including beta for Ising models.
    The phases may have different sample counts. Both phases must use the supplied
    blocks in the same order and provide one state per block. Each state leaf must have
    one leading sample axis. Flatten extra chain or batch axes before calling this function.

    **Arguments:**

    - `model`: The factorized energy-based model.
    - `positive_samples`: Positive-phase states with one leading sample axis.
    - `negative_samples`: Negative-phase states with one leading sample axis.
    - `blocks`: The shared block ordering for both phases.

    **Returns:**

    The mean positive energy minus the mean negative energy.

    **Raises:**

    `ValueError` if either phase has no samples, has an empty sample axis, or has a
    different number of block states than `blocks`.
    """

    def mean_energy(samples: list[_BatchedState]) -> Float[Array, ""]:
        leaves = jax.tree.leaves(samples)
        if not leaves or any(leaf.ndim == 0 or leaf.shape[0] == 0 for leaf in leaves):
            raise ValueError("Each sample phase must contain at least one sample.")
        if len(samples) != len(blocks):
            raise ValueError("Each sample phase must contain one state per block.")
        constant_samples = jax.tree.map(jax.lax.stop_gradient, samples)
        energies = jax.vmap(lambda state: model.energy(state, blocks))(constant_samples)
        return jnp.mean(energies)

    return mean_energy(positive_samples) - mean_energy(negative_samples)
