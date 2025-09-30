import abc
from typing import TypeVar

import equinox as eqx
import jax
from jaxtyping import Array, Key, PyTree, Shaped

_State = PyTree[Shaped[Array, "nodes ?*state"], "State"]
_SamplerState = TypeVar("_SamplerState", bound=PyTree)


class AbstractConditionalSampler(eqx.Module):
    """
    Base class for all conditional samplers.

    A conditional sampler is used to update the state of a block of nodes during each iteration of a sampling algorithm.
    It takes in the states of all the neighbors and produces a sample for the current block of nodes.
    This can often be done exactly, but need not be. One could embed MCMC methods within this sampler
    (to do Metropolis within Gibbs, for example).
    """

    @abc.abstractmethod
    def sample(
        self,
        key: Key,
        interactions: list[PyTree],
        active_flags: list[Array],
        states: list[list[_State]],
        sampler_state: _SamplerState,
        output_sd: PyTree[jax.ShapeDtypeStruct],
    ) -> tuple[_State, _SamplerState]:
        """Draw a sample from this conditional.

        If this sampler is involved in a block sampling program, this function is called every iteration to update the
        state of a block of nodes.

        **Arguments:**

        - `key`: A RNG key that the sampler can use to sample from distributions using `jax.random`.
        - `interactions`: A list of interactions that influence the result of this block update. Each interaction
            is a PyTree. Each array in the PyTree will have shape [n, k, ...], where n is the number of nodes in
             the block that is being updated and k is the maximum number of times any node in this block was
             detected as a head node for this interaction.
        - `active_flags`: A list of arrays of flags that is parallel to interactions. Each array indicates which
            instances of a given interaction are active for each node in the block. This array has shape [n, k],
            and is False if a given instance is inactive (which means that it should be ignored during the
             computation that happens in this function).
        - `states`: A list of PyTrees that is parallel to interactions, representing the sampling state information
            that is relevant to computing the influence of each interaction. Every array in each PyTree will have
            shape [n, k, ...].
        - `sampler_state`: The current state of this sampler. Will be replaced by the second return from this function
            the next time it is called.
        - `output_sd`: A PyTree indicating the expected shape/dtype of the output of this function.

        **Returns:**

        A new state for the block of nodes, matching the template given by `output_sd`.
        """
        pass

    def init(self) -> None:
        """Initialize the sampler state before sampling begins.

        This is called before the first iteration of block sampling, after which the return of this method is
        superseded by the return from `sample`.

        Returns:
            the initial sampler state to use for the first iteration of block sampling.
        """
        return None


class AbstractParametricConditionalSampler(AbstractConditionalSampler):
    """A conditional sampler that leverages a parameterized distribution.

    When `sample` is called, this sampler will first compute a set of parameters, and then use those parameters
    to draw a sample from some distribution. This workflow is frequently useful in practical cases; for example, to
    sample from a Gaussian, we can first compute a mean vector and covariance matrix using any procedure, and then
    draw a sample from the corresponding Gaussian distribution by appropriately transforming a vector of standard
    normal random variables."""

    @abc.abstractmethod
    def compute_parameters(
        self,
        key: Key,
        interactions: list[PyTree],
        active_flags: list[Array],
        states: list[list[_State]],
        sampler_state: PyTree,
        output_sd: PyTree[jax.ShapeDtypeStruct],
    ) -> PyTree:
        """Compute the parameters of the distribution. For a description of the arguments, see
        [`thrml.AbstractConditionalSampler.sample`][]"""
        pass

    @abc.abstractmethod
    def sample_given_parameters(
        self, key: Key, parameters: PyTree, sampler_state: _SamplerState, output_sd: PyTree[jax.ShapeDtypeStruct]
    ) -> tuple[_State, _SamplerState]:
        """Produce a sample given the parameters of the distribution, passed in as the `parameters` argument."""
        pass

    def sample(
        self,
        key: Key,
        interactions: list[PyTree],
        active_flags: list[Array],
        states: list[list[_State]],
        sampler_state: _SamplerState,
        output_sd: PyTree[jax.ShapeDtypeStruct],
    ) -> tuple[_State, _SamplerState]:
        """Sample from the distribution by first computing the parameters and then generating
        a sample based off of them."""
        key, subkey = jax.random.split(key, 2)
        parameters, state = self.compute_parameters(
            subkey, interactions, active_flags, states, sampler_state, output_sd
        )
        return self.sample_given_parameters(key, parameters, state, output_sd)


class BernoulliConditional(AbstractParametricConditionalSampler):
    r"""Sample from a bernoulli distribution.

    This sampler is designed to sample from a spin-valued bernoulli distribution:

    $$\mathbb{P}(S=s) \propto e^{\gamma s}$$

    where $S$ is a spin-valued random variable, $s \in \{-1, 1\}$. The parameter $\gamma$ must be
    computed by `compute_parameters`.
    """

    @abc.abstractmethod
    def compute_parameters(
        self,
        key: Key,
        interactions: list[PyTree],
        active_flags: list[Array],
        states: list[list[_State]],
        sampler_state: None,
        output_sd: PyTree[jax.ShapeDtypeStruct],
    ) -> PyTree:
        r"""A concrete implementation of this function has to return a value of $\gamma$ for every node
        in the block that is being updated. This array should have shape [b]."""
        pass

    def sample_given_parameters(
        self, key: Key, parameters: PyTree, sampler_state: None, output_sd: PyTree[jax.ShapeDtypeStruct]
    ) -> tuple[_State, None]:
        r"""Sample from a spin-valued bernoulli distribution given the parameter $\gamma$. In THRML,
        1 is represented by the boolean value `True` and -1 is represented by `False`."""
        return jax.random.bernoulli(key, jax.nn.sigmoid(2 * parameters)), sampler_state


class SoftmaxConditional(AbstractParametricConditionalSampler):
    r"""Sample from a softmax distribution.

    This sampler samples from the standard softmax distribution:

    $$\mathbb{P}(X=k) \propto e^{\theta_k}$$

    where $X$ is a categorical random variable and $\theta$ is a vector that parameterizes the relative
    probabilities of each of the categories.
    """

    @abc.abstractmethod
    def compute_parameters(
        self,
        key: Key,
        interactions: list[PyTree],
        active_flags: list[Array],
        states: list[list[_State]],
        sampler_state: None,
        output_sd: PyTree[jax.ShapeDtypeStruct],
    ) -> PyTree:
        """A concrete implementation of this function has to return $\theta$ vector for every node
        in the block that is being updated. This array should have shape [b, M], where $M$ is the
        number of possible values that $X$ may take on."""
        pass

    def sample_given_parameters(
        self, key: Key, parameters: PyTree, sampler_state: None, output_sd: PyTree[jax.ShapeDtypeStruct]
    ) -> tuple[_State, None]:
        """Sample from a softmax distribution given the parameter vector $\theta$."""
        return jax.random.categorical(key, parameters, axis=-1).astype(output_sd.dtype), sampler_state
