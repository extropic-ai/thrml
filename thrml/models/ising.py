from typing import TYPE_CHECKING

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Bool, Key

from thrml.block_sampling import (
    Block,
    BlockGibbsSpec,
    BlockSamplingProgram,
    SamplingSchedule,
    SuperBlock,
    _validate_sampling_state,
    sample_with_observation,
)
from thrml.models.discrete_ebm import SpinEBMFactor
from thrml.models.ebm import AbstractFactorizedEBM, EBMFactor
from thrml.observers import MomentAccumulatorObserver
from thrml.pgm import AbstractNode

if TYPE_CHECKING:
    from thrml.backends import SamplingBackend

Edge = tuple[AbstractNode, AbstractNode]


class IsingEBM(AbstractFactorizedEBM):
    r"""An EBM with the energy function,

    $$\mathcal{E}(s) = -\beta \left( \sum_{i \in S_1} b_i s_i + \sum_{(i, j) \in S_2} J_{ij} s_i s_j \right)$$

    where $S_1$ and $S_2$ are the sets of biases and weights that make up the model, respectively.
    $b_i$ represents the bias associated with the spin $s_i$ and $J_{ij}$ is a weight that couples
    $s_i$ and $s_j$. $\beta$ is the usual temperature parameter.

    **Attributes:**

    - `nodes`: the nodes that have an associated bias (i.e $S_1$)
    - `biases`: the bias associated with each node in `nodes`.
    - `edges`: the edges that have an associated weight (i.e $S_2$)
    - `weights`: the weight associated with each pair of nodes in `edges`.
    - `beta`: the scalar temperature parameter for the model.

    """

    nodes: list[AbstractNode]
    biases: Array
    edges: list[Edge]
    weights: Array
    beta: Array

    def __init__(self, nodes: list[AbstractNode], edges: list[Edge], biases: Array, weights: Array, beta: Array):
        """Initialize an Ising EBM.

        **Arguments:**

        - `nodes`: List of nodes with associated biases
        - `edges`: List of edge pairs with associated weights
        - `biases`: Bias values for each node
        - `weights`: Weight values for each edge
        - `beta`: Temperature parameter
        """
        # nodes should be same type, should be at least one node passed in
        sd_map = {nodes[0].__class__: jax.ShapeDtypeStruct((), jnp.bool_)}

        super().__init__(sd_map)

        self.nodes = nodes
        self.edges = edges
        self.beta = beta
        self.weights = weights
        self.biases = biases

    @property
    def factors(self) -> list[EBMFactor]:
        return [
            SpinEBMFactor([Block(self.nodes)], self.beta * self.biases),
            SpinEBMFactor(
                [Block([x[0] for x in self.edges]), Block([x[1] for x in self.edges])], self.beta * self.weights
            ),
        ]


class IsingSamplingProgram(eqx.Module):
    """A logical Ising model and its free/clamped block layout.

    The current model remains available to every backend. Biases, weights, and beta
    are dynamic JAX leaves. For low-level JAX updates or access to ``samplers`` and
    ``per_block_*``, first use ``JaxBackend().compile(program)``.
    """

    model: IsingEBM
    gibbs_spec: BlockGibbsSpec

    def __init__(self, ebm: IsingEBM, free_blocks: list[SuperBlock], clamped_blocks: list[Block]):
        """Initialize an Ising sampling program.

        **Arguments:**

        - `ebm`: The Ising EBM to sample from
        - `free_blocks`: List of super blocks that are free to vary
        - `clamped_blocks`: List of blocks that are held fixed
        """
        self.model = ebm
        self.gibbs_spec = BlockGibbsSpec(free_blocks, clamped_blocks, ebm.node_shape_dtypes)


class IsingTrainingSpec(eqx.Module):
    """Contains a complete specification of an Ising EBM that can be trained using sampling-based gradients.

    Defines sampling programs and schedules that allow for collection of the positive and negative phase samples
    required for Monte Carlo estimation of the gradient of the KL-divergence between the model and a data distribution.

    Rebuild this specification after each model update so both phase programs retain
    the updated model. Updating only ``ebm`` leaves their models stale. A backend can
    be reused independently across rebuilt specifications.
    """

    ebm: IsingEBM
    program_positive: IsingSamplingProgram
    program_negative: IsingSamplingProgram
    schedule_positive: SamplingSchedule
    schedule_negative: SamplingSchedule

    def __init__(
        self,
        ebm: IsingEBM,
        data_blocks: list[Block],
        conditioning_blocks: list[Block],
        positive_sampling_blocks: list[SuperBlock],
        negative_sampling_blocks: list[SuperBlock],
        schedule_positive: SamplingSchedule,
        schedule_negative: SamplingSchedule,
    ):
        self.ebm = ebm

        self.program_positive = IsingSamplingProgram(ebm, positive_sampling_blocks, data_blocks + conditioning_blocks)
        self.program_negative = IsingSamplingProgram(ebm, negative_sampling_blocks, conditioning_blocks)

        _validate_training_layout(self)
        self.schedule_positive = schedule_positive
        self.schedule_negative = schedule_negative


def _validate_training_layout(training_spec):
    positive = training_spec.program_positive.gibbs_spec
    negative = training_spec.program_negative.gibbs_spec
    n_data_blocks = len(positive.clamped_blocks) - len(negative.clamped_blocks)
    if n_data_blocks < 1:
        raise ValueError("Training requires at least one data block.")
    if positive.clamped_blocks[n_data_blocks:] != negative.clamped_blocks:
        raise ValueError("Positive conditioning blocks must match the negative clamped blocks.")

    data_nodes = {node for block in positive.clamped_blocks[:n_data_blocks] for node in block}
    positive_free = {node for block in positive.free_blocks for node in block}
    negative_free = {node for block in negative.free_blocks for node in block}
    if negative_free != positive_free | data_nodes:
        raise ValueError("Negative free nodes must equal positive free nodes plus data nodes.")

    model = training_spec.ebm
    model_nodes = set(model.nodes) | {node for edge in model.edges for node in edge}
    for name, spec in (("positive", positive), ("negative", negative)):
        if not model_nodes.issubset(spec.node_global_location_map):
            raise ValueError(f"The {name} layout must cover every model bias node and edge endpoint.")
    return n_data_blocks


@eqx.filter_jit
def hinton_init(
    key: Key[Array, ""], model: IsingEBM, blocks: list[Block[AbstractNode]], batch_shape: tuple[int]
) -> list[Bool[Array, "batch_size block_size"]]:
    r"""
    Initialize the blocks according to the marginal bias.

    Each binary unit $i$ in a block is sampled independently as

    $$\mathbb{P}(S_i = 1) = \sigma(\beta h_i) = \frac{1}{1 + e^{-\beta h_i}}$$

    where $h_i$ is the bias of unit *i* and $\beta$ is the
    inverse-temperature scaling factor. See Hinton (2012) for a discussion of this initialization heuristic.

    Arguments:
        key: the JAX PRNG key to use
        model: the Ising model to initialize for
        blocks: the blocks that are to be initialized
        batch_shape: the pre-pended dimension

    Returns:
        the initialized blocks
    """
    node_map = {node: i for i, node in enumerate(model.nodes)}

    data = []
    keys = jax.random.split(key, len(blocks))
    for i, block in enumerate(blocks):
        if len(block) == 0:
            data.append(jnp.zeros((*batch_shape, 0), dtype=jnp.bool_))
            continue

        block_indices = jnp.array([node_map[node] for node in block], dtype=jnp.int32)
        block_biases = model.biases[block_indices]
        probs = jax.nn.sigmoid(model.beta * block_biases)

        block_data = jax.random.bernoulli(keys[i], p=probs, shape=(*batch_shape, len(block))).astype(jnp.bool_)

        data.append(block_data)

    return data


def _spin_transform(state, _):
    """Map Boolean states to spins in {-1, +1}. A module-level function, so that two moment observers with the same
    nodes and edges compare equal and a backend can use one compiled fold for both."""
    return [2 * x.astype(jnp.int8) - 1 for x in state]


def _ising_moment_observer(first_moment_nodes, second_moment_edges):
    first_moments = ((),)
    if len(first_moment_nodes) > 0:
        first_moments = [(node,) for node in first_moment_nodes]
    return MomentAccumulatorObserver((first_moments, second_moment_edges), _spin_transform)


def _validate_moment_nodes(program, first_moment_nodes, second_moment_edges, name="program"):
    nodes = set(first_moment_nodes) | {node for edge in second_moment_edges for node in edge}
    if not nodes.issubset(program.gibbs_spec.node_global_location_map):
        raise ValueError(f"Requested statistic nodes must exist in the {name} layout.")


def estimate_moments(
    key: Key[Array, ""],
    first_moment_nodes: list[AbstractNode],
    second_moment_edges: list[Edge],
    program: IsingSamplingProgram | BlockSamplingProgram,
    schedule: SamplingSchedule,
    init_state: list[Array] | None,
    clamped_data: list[Array],
    *,
    backend: "SamplingBackend | None" = None,
):
    """Estimate first and second Ising moments through a sampling backend.

    Arguments:
        key: the JAX PRNG key
        first_moment_nodes: nodes whose first moments are requested
        second_moment_edges: edges whose second moments are requested
        program: the logical Ising program or an executable BlockSamplingProgram
        schedule: the sampling schedule
        init_state: initial free-block values, [] for no free blocks, or None for
         backend-owned continuation. JAX requires an explicit state when sampling.
        clamped_data: fixed values in the program's clamped-block order
        backend: execution backend; None selects the default JaxBackend

    Returns:
        The first and second moments, normalized once by the total record count.
    """
    _validate_moment_nodes(program, first_moment_nodes, second_moment_edges)
    observer = _ising_moment_observer(first_moment_nodes, second_moment_edges)
    moments, _ = sample_with_observation(
        key, program, schedule, init_state, clamped_data, observer.init(), observer, backend=backend
    )
    node_sums, edge_sums = moments
    return node_sums / schedule.n_samples, edge_sums / schedule.n_samples


def _map_with_optional_initial_state(backend, fn, keys, initial_states, *batched_args):
    if initial_states is None:
        return backend.map_chains(lambda key, *args: fn(key, None, *args), keys, *batched_args)
    return backend.map_chains(fn, keys, initial_states, *batched_args)


def _fully_clamped_moments(program, observer, clamped_state):
    sums, _ = observer(program, [], clamped_state, observer.init(), jnp.array(0))
    return tuple(x / 1 for x in sums)


def _validate_training_state(
    training_spec, bias_nodes, weight_edges, data, conditioning_values, init_state_positive, init_state_negative
):
    """Check both phases before any backend operation, using only static array metadata."""
    n_data_blocks = _validate_training_layout(training_spec)
    positive, negative = training_spec.program_positive, training_spec.program_negative
    training_spec.schedule_positive.validate()
    training_spec.schedule_negative.validate()
    if len(data) != n_data_blocks:
        raise ValueError("data must have one array per data block.")
    if len(conditioning_values) != len(negative.gibbs_spec.clamped_blocks):
        raise ValueError("conditioning_values must have one array per conditioning block.")

    for name, arrays, rank in (
        ("data", data, 2),
        ("conditioning_values", conditioning_values, 1),
        ("init_state_positive", init_state_positive, 3),
        ("init_state_negative", init_state_negative, 2),
    ):
        if arrays is not None and any(getattr(array, "ndim", None) != rank for array in arrays):
            raise ValueError(f"{name} arrays must have rank {rank}.")
    batch_size = data[0].shape[0]
    if batch_size < 1 or any(array.shape[0] != batch_size for array in data):
        raise ValueError("data arrays must share a nonempty batch dimension.")

    replicas = []
    for name, program, initial in (
        ("positive", positive, init_state_positive),
        ("negative", negative, init_state_negative),
    ):
        if initial is not None and len(initial) != len(program.gibbs_spec.free_blocks):
            raise ValueError(f"init_state_{name} must have one array per free block.")
        n_chains = initial[0].shape[0] if initial else 1
        if n_chains < 1 or (initial is not None and any(array.shape[0] != n_chains for array in initial)):
            raise ValueError(f"init_state_{name} arrays must share a positive replica count.")
        replicas.append(n_chains)
    if init_state_positive is not None and any(array.shape[1] != batch_size for array in init_state_positive):
        raise ValueError("init_state_positive arrays must match the data batch dimension.")

    _validate_moment_nodes(positive, bias_nodes, weight_edges, "positive")
    _validate_moment_nodes(negative, bias_nodes, weight_edges, "negative")
    _validate_sampling_state(positive, init_state_positive, data + conditioning_values)
    _validate_sampling_state(negative, init_state_negative, conditioning_values)
    return batch_size, *replicas, not positive.gibbs_spec.free_blocks


def estimate_kl_grad(
    key: Key[Array, ""],
    training_spec: IsingTrainingSpec,
    bias_nodes: list[AbstractNode],
    weight_edges: list[Edge],
    data: list[Array],
    conditioning_values: list[Array],
    init_state_positive: list[Array] | None,
    init_state_negative: list[Array] | None,
    *,
    backend: "SamplingBackend | None" = None,
) -> tuple:
    r"""
    Estimate the KL-gradients of an Ising model with respect to its weights and biases.

    Uses the standard two-term Monte Carlo estimator of the gradient of the KL-divergence between an Ising model and
    a data distribution

    The gradients are:

    $$\Delta W = -\beta (\langle s_i s_j \rangle_{+} - \langle s_i s_j \rangle_{-})$$

    $$\Delta b = -\beta (\langle s_i \rangle_{+} - \langle s_i \rangle_{-})$$

    Here, $\langle\cdot\rangle_{+}$ denotes an expectation under the
    *positive* phase (data-clamped Boltzmann distribution) and
    $\langle\cdot\rangle_{-}$ under the *negative* phase (model
    distribution).

    Arguments:
        key: the JAX PRNG key
        training_spec: the Ising EBM for which to estimate the gradients
        bias_nodes: the nodes for which to estimate the bias gradients
        weight_edges: the edges for which to estimate the weight gradients
        data: The data values to use for the positive phase of the gradient estimate. Each array has shape [batch nodes]
        conditioning_values: values to assign to the nodes that the model is conditioned on.
         Each array has shape [nodes]
        init_state_positive: initial state for the positive sampling chain. Each array has
         shape [n_chains_pos batch nodes]. None requests one backend-owned continuation replica.
         Pass [] only when the positive program has no free blocks; [] and None then read positive moments
         directly from data and conditioning values without backend execution.
        init_state_negative: initial state for the negative sampling chain. Each array has
         shape [n_chains_neg nodes]. None requests one continuation replica; [] is invalid.
        backend: execution backend; None selects JaxBackend, which requires explicit initial states for
         sampled phases. The backend maps requests and owns continuation across them. External execution
         runs on the host; do not wrap it in JAX transformations.
    Returns:
        Weight gradients, bias gradients, positive moments, and negative moments, in that order.
        Positive moment arrays have shape [n_chains_pos batch statistics]; negative arrays have
        shape [n_chains_neg statistics]. Rebuild training_spec after updating its model.
    """

    from thrml.backends import _resolve_backend

    batch_size, n_positive, n_negative, fully_clamped = _validate_training_state(
        training_spec, bias_nodes, weight_edges, data, conditioning_values, init_state_positive, init_state_negative
    )
    backend = _resolve_backend(backend)
    float_type = training_spec.ebm.beta.dtype
    key_pos, key_neg = jax.random.split(key, 2)
    cond_batched_pos = jax.tree.map(lambda x: jnp.broadcast_to(x, (batch_size, *x.shape)), conditioning_values)

    if fully_clamped:
        observer = _ising_moment_observer(bias_nodes, weight_edges)
        moms_b_pos, moms_w_pos = jax.vmap(
            lambda clamps: _fully_clamped_moments(training_spec.program_positive, observer, clamps)
        )(data + cond_batched_pos)
        moms_b_pos = moms_b_pos[None]
        moms_w_pos = moms_w_pos[None]
    else:
        keys_pos = jax.random.split(key_pos, (n_positive, batch_size))

        def positive_one(key_i, initial_i, clamps_i):
            return estimate_moments(
                key_i,
                bias_nodes,
                weight_edges,
                training_spec.program_positive,
                training_spec.schedule_positive,
                initial_i,
                clamps_i,
                backend=backend,
            )

        def positive_replica(keys_for_data, initials_for_data):
            return _map_with_optional_initial_state(
                backend, positive_one, keys_for_data, initials_for_data, data + cond_batched_pos
            )

        moms_b_pos, moms_w_pos = _map_with_optional_initial_state(
            backend, positive_replica, keys_pos, init_state_positive
        )

    def negative_one(key_i, initial_i):
        return estimate_moments(
            key_i,
            bias_nodes,
            weight_edges,
            training_spec.program_negative,
            training_spec.schedule_negative,
            initial_i,
            conditioning_values,
            backend=backend,
        )

    keys_neg = jax.random.split(key_neg, n_negative)
    moms_b_neg, moms_w_neg = _map_with_optional_initial_state(backend, negative_one, keys_neg, init_state_negative)

    grad_b = -training_spec.ebm.beta * (
        jnp.mean(moms_b_pos, axis=(0, 1), dtype=float_type) - jnp.mean(moms_b_neg, axis=0, dtype=float_type)
    )
    grad_w = -training_spec.ebm.beta * (
        jnp.mean(moms_w_pos, axis=(0, 1), dtype=float_type) - jnp.mean(moms_w_neg, axis=0, dtype=float_type)
    )
    return grad_w, grad_b, (moms_b_pos, moms_w_pos), (moms_b_neg, moms_w_neg)
