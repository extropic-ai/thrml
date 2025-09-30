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
    sample_with_observation,
)
from thrml.factor import FactorSamplingProgram
from thrml.models.discrete_ebm import SpinEBMFactor, SpinGibbsConditional
from thrml.models.ebm import AbstractFactorizedEBM, EBMFactor
from thrml.observers import MomentAccumulatorObserver
from thrml.pgm import AbstractNode

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


class IsingSamplingProgram(FactorSamplingProgram):
    """A very thin wrapper on FactorSamplingProgram that specializes it to the case of an Ising Model."""

    def __init__(self, ebm: IsingEBM, free_blocks: list[SuperBlock], clamped_blocks: list[Block]):
        """Initialize an Ising sampling program.

        **Arguments:**

        - `ebm`: The Ising EBM to sample from
        - `free_blocks`: List of super blocks that are free to vary
        - `clamped_blocks`: List of blocks that are held fixed
        """
        samp = SpinGibbsConditional()

        spec = BlockGibbsSpec(free_blocks, clamped_blocks, ebm.node_shape_dtypes)

        super().__init__(spec, [samp for _ in spec.free_blocks], ebm.factors, [])


class IsingTrainingSpec(eqx.Module):
    """Contains a complete specification of an Ising EBM that can be trained using sampling-based gradients.

    Defines sampling programs and schedules that allow for collection of the positive and negative phase samples
    required for Monte Carlo estimation of the gradient of the KL-divergence between the model and a data distribution.
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

        self.schedule_positive = schedule_positive
        self.schedule_negative = schedule_negative


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


def estimate_moments(
    key: Key[Array, ""],
    first_moment_nodes: list[AbstractNode],
    second_moment_edges: list[Edge],
    program: BlockSamplingProgram,
    schedule: SamplingSchedule,
    init_state: list[Array],
    clamped_data: list[Array],
):
    """
    Estimates the first and second moments of an Ising model Boltzmann distribution via sampling.

    Arguments:
        key: the jax PRNG key
        first_moment_nodes: the nodes that represent the variables we want to estimate the first moments of
        second_moment_edges: the edges that connect the variables we want to estimate the second moments of
        program: the `BlockSamplingProgram` to be used for sampling
        schedule: the schedule to use for sampling
        init_state: the variable values to use to initialize the sampling
        clamped_data: the variable values to assign to the clamped nodes
    Returns:
        the first and second moment data
    """

    # add a layer of tuple
    first_moments = ((),)
    if len(first_moment_nodes) > 0:
        first_moments = [(node,) for node in first_moment_nodes]

    def _spin_transform(state, _):
        return [2 * x.astype(jnp.int8) - 1 for x in state]

    observer = MomentAccumulatorObserver((first_moments, second_moment_edges), _spin_transform)
    init_mem = observer.init()

    moments, _ = sample_with_observation(key, program, schedule, init_state, clamped_data, init_mem, observer)

    node_sums, edge_sums = moments
    node_moments = node_sums / schedule.n_samples
    edge_moments = edge_sums / schedule.n_samples

    return node_moments, edge_moments


def estimate_kl_grad(
    key: Key[Array, ""],
    training_spec: IsingTrainingSpec,
    bias_nodes: list[AbstractNode],
    weight_edges: list[Edge],
    data: list[Array],
    conditioning_values: list[Array],
    init_state_positive: list[Array],
    init_state_negative: list[Array],
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
         shape [n_chains_pos batch nodes]
        init_state_negative: initial state for the negative sampling chain. Each array has
         shape [n_chains_neg nodes]
    Returns:
        the weight gradients and the bias gradients
    """

    key_pos, key_neg = jax.random.split(key, 2)

    cond_batched_pos = jax.tree.map(lambda x: jnp.broadcast_to(x, (data[0].shape[0], *x.shape)), conditioning_values)

    keys_pos = jax.random.split(key_pos, init_state_positive[0].shape[:2])

    moms_b_pos, moms_w_pos = jax.vmap(
        lambda k_out, i_out: jax.vmap(
            lambda k, i, c: estimate_moments(
                k, bias_nodes, weight_edges, training_spec.program_positive, training_spec.schedule_positive, i, c
            )
        )(k_out, i_out, data + cond_batched_pos)
    )(keys_pos, init_state_positive)

    keys_neg = jax.random.split(key_neg, init_state_negative[0].shape[0])

    moms_b_neg, moms_w_neg = jax.vmap(
        lambda k, i: estimate_moments(
            k,
            bias_nodes,
            weight_edges,
            training_spec.program_negative,
            training_spec.schedule_negative,
            i,
            conditioning_values,
        )
    )(keys_neg, init_state_negative)

    float_type = training_spec.ebm.beta.dtype
    grad_b = -training_spec.ebm.beta * (
        jnp.mean(moms_b_pos, axis=(0, 1), dtype=float_type) - jnp.mean(moms_b_neg, axis=0, dtype=float_type)
    )
    grad_w = -training_spec.ebm.beta * (
        jnp.mean(moms_w_pos, axis=(0, 1), dtype=float_type) - jnp.mean(moms_w_neg, axis=0, dtype=float_type)
    )
    return grad_w, grad_b, (moms_b_pos, moms_w_pos), (moms_b_neg, moms_w_neg)
