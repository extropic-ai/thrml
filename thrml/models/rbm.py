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
from thrml.models.discrete_ebm import DiscreteEBMFactor, SpinGibbsConditional
from thrml.models.ebm import AbstractFactorizedEBM, EBMFactor
from thrml.observers import MomentAccumulatorObserver
from thrml.pgm import AbstractNode


class RBMEBM(AbstractFactorizedEBM):
    r"""A Restricted Boltzmann Machine (RBM) Energy-Based Model.

    An RBM is a bipartite graphical model with visible and hidden units.
    The energy function is:

    $$\mathcal{E}(v, h) = -\beta \left( \sum_{i} a_i v_i + \sum_{j} b_j h_j + \sum_{i,j} W_{ij} v_i h_j \right)$$

    where:
    - $v_i$ are the visible units (observed data)
    - $h_j$ are the hidden units (latent features)
    - $a_i$ are visible biases
    - $b_j$ are hidden biases
    - $W_{ij}$ is the weight matrix connecting visible to hidden units
    - $\beta$ is the inverse temperature parameter

    The bipartite structure means visible units only connect to hidden units,
    and vice versa - there are no visible-visible or hidden-hidden connections.

    **Attributes:**

    - `visible_nodes`: List of visible (observed) nodes
    - `hidden_nodes`: List of hidden (latent) nodes
    - `visible_biases`: Bias values for visible units
    - `hidden_biases`: Bias values for hidden units
    - `weights`: Weight matrix of shape [n_visible, n_hidden]
    - `beta`: Temperature parameter
    """

    visible_nodes: list[AbstractNode]
    hidden_nodes: list[AbstractNode]
    visible_biases: Array
    hidden_biases: Array
    weights: Array
    beta: Array

    def __init__(
        self,
        visible_nodes: list[AbstractNode],
        hidden_nodes: list[AbstractNode],
        visible_biases: Array,
        hidden_biases: Array,
        weights: Array,
        beta: Array,
    ):
        """Initialize an RBM Energy-Based Model.

        **Arguments:**

        - `visible_nodes`: List of visible (observed) nodes
        - `hidden_nodes`: List of hidden (latent) nodes
        - `visible_biases`: Bias values for each visible node (shape: [n_visible])
        - `hidden_biases`: Bias values for each hidden node (shape: [n_hidden])
        - `weights`: Weight matrix connecting visible to hidden (shape: [n_visible, n_hidden])
        - `beta`: Temperature parameter (scalar)
        """
        # Nodes should be SpinNode type
        sd_map = {visible_nodes[0].__class__: jax.ShapeDtypeStruct((), jnp.bool_)}

        super().__init__(sd_map)

        if weights.shape != (len(visible_nodes), len(hidden_nodes)):
            raise ValueError(
                f"Weight matrix shape {weights.shape} does not match "
                f"expected shape ({len(visible_nodes)}, {len(hidden_nodes)})"
            )

        self.visible_nodes = visible_nodes
        self.hidden_nodes = hidden_nodes
        self.visible_biases = visible_biases
        self.hidden_biases = hidden_biases
        self.weights = weights
        self.beta = beta

    @property
    def factors(self) -> list[EBMFactor]:
        """Return the factors that define the RBM energy function.

        Returns three factors:
        1. Visible bias terms: -beta * sum(a_i * v_i)
        2. Hidden bias terms: -beta * sum(b_j * h_j)
        3. Interaction terms: -beta * sum(W_ij * v_i * h_j)
        """
        # Create edges as pairs (visible_i, hidden_j) for all i, j
        # For an RBM with n_visible=2, n_hidden=3, we have edges:
        # (v0,h0), (v0,h1), (v0,h2), (v1,h0), (v1,h1), (v1,h2)
        visible_indices = []
        hidden_indices = []
        interaction_weights = []

        for i, v_node in enumerate(self.visible_nodes):
            for j, h_node in enumerate(self.hidden_nodes):
                visible_indices.append(v_node)
                hidden_indices.append(h_node)
                interaction_weights.append(self.weights[i, j])

        return [
            # Visible bias factor
            DiscreteEBMFactor([Block(self.visible_nodes)], [], self.beta * self.visible_biases),
            # Hidden bias factor
            DiscreteEBMFactor([Block(self.hidden_nodes)], [], self.beta * self.hidden_biases),
            # Interaction factor (bipartite connections between visible and hidden)
            DiscreteEBMFactor(
                [Block(visible_indices), Block(hidden_indices)], [], self.beta * jnp.array(interaction_weights)
            ),
        ]


class RBMSamplingProgram(FactorSamplingProgram):
    """A sampling program specialized for Restricted Boltzmann Machines.

    This is a thin wrapper around FactorSamplingProgram that sets up
    the appropriate Gibbs sampling for RBMs using SpinGibbsConditional samplers.
    """

    def __init__(self, ebm: RBMEBM, free_blocks: list[SuperBlock], clamped_blocks: list[Block]):
        """Initialize an RBM sampling program.

        **Arguments:**

        - `ebm`: The RBM EBM to sample from
        - `free_blocks`: List of super blocks that are free to vary during sampling
        - `clamped_blocks`: List of blocks that are held fixed (e.g., visible units during training)
        """
        samp = SpinGibbsConditional()

        spec = BlockGibbsSpec(free_blocks, clamped_blocks, ebm.node_shape_dtypes)

        super().__init__(spec, [samp for _ in spec.free_blocks], ebm.factors, [])


class RBMTrainingSpec(eqx.Module):
    """Complete specification for training an RBM using contrastive divergence.

    Defines sampling programs and schedules for collecting positive phase
    (data-driven) and negative phase (model-driven) samples needed for
    gradient estimation in RBM training.

    **Attributes:**

    - `ebm`: The RBM model being trained
    - `program_positive`: Sampling program for positive phase (visible clamped)
    - `program_negative`: Sampling program for negative phase (free sampling)
    - `schedule_positive`: Sampling schedule for positive phase
    - `schedule_negative`: Sampling schedule for negative phase
    """

    ebm: RBMEBM
    program_positive: RBMSamplingProgram
    program_negative: RBMSamplingProgram
    schedule_positive: SamplingSchedule
    schedule_negative: SamplingSchedule

    def __init__(
        self,
        ebm: RBMEBM,
        schedule_positive: SamplingSchedule,
        schedule_negative: SamplingSchedule,
    ):
        """Initialize an RBM training specification.

        **Arguments:**

        - `ebm`: The RBM model to train
        - `schedule_positive`: Sampling schedule for positive phase (hidden units given visible)
        - `schedule_negative`: Sampling schedule for negative phase (free sampling)
        """
        self.ebm = ebm

        # Positive phase: clamp visible units, sample hidden units
        self.program_positive = RBMSamplingProgram(ebm, [Block(ebm.hidden_nodes)], [Block(ebm.visible_nodes)])

        # Negative phase: sample both visible and hidden units freely
        self.program_negative = RBMSamplingProgram(ebm, [Block(ebm.visible_nodes), Block(ebm.hidden_nodes)], [])

        self.schedule_positive = schedule_positive
        self.schedule_negative = schedule_negative


@eqx.filter_jit
def rbm_init(
    key: Key[Array, ""], model: RBMEBM, blocks: list[Block[AbstractNode]], batch_shape: tuple[int, ...]
) -> list[Bool[Array, "..."]]:
    r"""Initialize RBM blocks according to marginal biases.

    Each binary unit $i$ in a block is sampled independently as:

    $$\mathbb{P}(S_i = 1) = \sigma(\beta h_i) = \frac{1}{1 + e^{-\beta h_i}}$$

    where $h_i$ is the bias of unit $i$ and $\beta$ is the inverse temperature.

    This is the same initialization strategy used for Ising models (Hinton, 2012).

    **Arguments:**

    - `key`: JAX PRNG key
    - `model`: The RBM model to initialize for
    - `blocks`: The blocks to initialize (visible and/or hidden nodes)
    - `batch_shape`: Batch dimensions to prepend

    **Returns:**

    List of initialized block states
    """
    # Create a mapping from nodes to their biases
    node_to_bias = {}
    for node, bias in zip(model.visible_nodes, model.visible_biases):
        node_to_bias[node] = bias
    for node, bias in zip(model.hidden_nodes, model.hidden_biases):
        node_to_bias[node] = bias

    data = []
    keys = jax.random.split(key, len(blocks))

    for i, block in enumerate(blocks):
        if len(block) == 0:
            data.append(jnp.zeros((*batch_shape, 0), dtype=jnp.bool_))
            continue

        # Get biases for nodes in this block
        block_biases = jnp.array([node_to_bias[node] for node in block])
        probs = jax.nn.sigmoid(model.beta * block_biases)

        block_data = jax.random.bernoulli(keys[i], p=probs, shape=(*batch_shape, len(block))).astype(jnp.bool_)

        data.append(block_data)

    return data


def estimate_rbm_moments(
    key: Key[Array, ""],
    visible_nodes: list[AbstractNode],
    hidden_nodes: list[AbstractNode],
    program: BlockSamplingProgram,
    schedule: SamplingSchedule,
    init_state: list[Array],
    clamped_data: list[Array],
) -> tuple[Array, Array, Array]:
    """Estimate first and second moments for RBM training.

    Computes:
    - Mean visible activations: <v_i>
    - Mean hidden activations: <h_j>
    - Mean visible-hidden products: <v_i h_j>

    **Arguments:**

    - `key`: JAX PRNG key
    - `visible_nodes`: List of visible nodes
    - `hidden_nodes`: List of hidden nodes
    - `program`: BlockSamplingProgram for sampling
    - `schedule`: Sampling schedule
    - `init_state`: Initial state for free blocks
    - `clamped_data`: Values for clamped blocks

    **Returns:**

    Tuple of (visible_moments, hidden_moments, interaction_moments)
    """
    # Determine which nodes to track based on what's in the sampling program
    # We need to track ALL nodes (both visible and hidden) for moment estimation
    all_nodes = visible_nodes + hidden_nodes

    # Define first and second moments
    first_moments = [(node,) for node in all_nodes]
    second_moments = [(v, h) for v in visible_nodes for h in hidden_nodes]

    def _spin_transform(state, _):
        """Convert bool states to {-1, +1} spin values."""
        return [2 * x.astype(jnp.int8) - 1 for x in state]

    # Create observer for all nodes we want to track
    observer = MomentAccumulatorObserver((first_moments, second_moments), _spin_transform)
    init_mem = observer.init()

    # Sample and observe - this will track moments from both free and clamped blocks
    moments, _ = sample_with_observation(
        key, program, schedule, init_state, clamped_data, init_mem, observer
    )

    node_sums, edge_sums = moments
    node_moments = node_sums / schedule.n_samples
    edge_moments = edge_sums / schedule.n_samples

    # Split node moments into visible and hidden
    n_visible = len(visible_nodes)
    visible_moments = node_moments[:n_visible]
    hidden_moments = node_moments[n_visible:]

    # Reshape edge moments to [n_visible, n_hidden]
    interaction_moments = edge_moments.reshape(len(visible_nodes), len(hidden_nodes))

    return visible_moments, hidden_moments, interaction_moments


def estimate_rbm_grad(
    key: Key[Array, ""],
    training_spec: RBMTrainingSpec,
    visible_data: list[Array],
    init_state_positive: list[Array],
    init_state_negative: list[Array],
) -> tuple[Array, Array, Array]:
    r"""Estimate gradients for RBM training using contrastive divergence.

    Computes the gradient of the log-likelihood using the standard two-phase approach:

    $$\Delta W_{ij} = -\beta (\langle v_i h_j \rangle_{data} - \langle v_i h_j \rangle_{model})$$
    $$\Delta a_i = -\beta (\langle v_i \rangle_{data} - \langle v_i \rangle_{model})$$
    $$\Delta b_j = -\beta (\langle h_j \rangle_{data} - \langle h_j \rangle_{model})$$

    where $\langle \cdot \rangle_{data}$ denotes expectations under the data distribution
    (visible units clamped) and $\langle \cdot \rangle_{model}$ denotes expectations under
    the model distribution (free sampling).

    **Arguments:**

    - `key`: JAX PRNG key
    - `training_spec`: RBM training specification
    - `visible_data`: Visible data values [batch, n_visible]
    - `init_state_positive`: Initial hidden states for positive phase [n_chains, batch, n_hidden]
    - `init_state_negative`: Initial states for negative phase [n_chains, n_visible+n_hidden]

    **Returns:**

    Tuple of (weight_grad, visible_bias_grad, hidden_bias_grad)
    """
    key_pos, key_neg = jax.random.split(key, 2)

    model = training_spec.ebm

    # Positive phase: clamp visible data, sample hidden
    batch_size = visible_data[0].shape[0]

    # Split keys for positive phase: one for each chain and batch item
    keys_pos = jax.random.split(key_pos, init_state_positive[0].shape[:2])

    # Run positive phase sampling for each chain and each data point
    vis_pos, hid_pos, int_pos = jax.vmap(
        lambda k_out, h_out: jax.vmap(
            lambda k, h, v: estimate_rbm_moments(
                k,
                model.visible_nodes,
                model.hidden_nodes,
                training_spec.program_positive,
                training_spec.schedule_positive,
                h,  # init hidden state
                [v],  # clamped visible data
            )
        )(k_out, h_out, visible_data[0])
    )(keys_pos, init_state_positive)

    # Negative phase: free sampling
    keys_neg = jax.random.split(key_neg, init_state_negative[0].shape[0])

    vis_neg, hid_neg, int_neg = jax.vmap(
        lambda k, init: estimate_rbm_moments(
            k,
            model.visible_nodes,
            model.hidden_nodes,
            training_spec.program_negative,
            training_spec.schedule_negative,
            init,  # both visible and hidden
            [],  # nothing clamped
        )
    )(keys_neg, init_state_negative)

    # Compute gradients
    float_type = model.beta.dtype

    grad_visible_bias = -model.beta * (
        jnp.mean(vis_pos, axis=(0, 1), dtype=float_type) - jnp.mean(vis_neg, axis=0, dtype=float_type)
    )

    grad_hidden_bias = -model.beta * (
        jnp.mean(hid_pos, axis=(0, 1), dtype=float_type) - jnp.mean(hid_neg, axis=0, dtype=float_type)
    )

    grad_weights = -model.beta * (
        jnp.mean(int_pos, axis=(0, 1), dtype=float_type) - jnp.mean(int_neg, axis=0, dtype=float_type)
    )

    return grad_weights, grad_visible_bias, grad_hidden_bias
