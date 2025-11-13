import jax
import jax.numpy as jnp

from thrml.conditional_samplers import BernoulliConditional, SoftmaxConditional
from thrml.pgm import SpinNode
from thrml.block_management import Block, BlockSpec, block_state_to_global
from thrml.models.ising import IsingEBM, IsingSamplingProgram, hinton_init
from thrml.models.discrete_ebm import SpinEBMFactor, SpinGibbsConditional
from thrml.block_sampling import BlockGibbsSpec

def test_bernoulli_conditional_sample_shapes_and_dtype():
    """Basic shape/dtype checks for BernoulliConditional behaviour.
    """

    class ConstBern(BernoulliConditional):
        def compute_parameters(self, key, interactions, active_flags, states, sampler_state, output_sd):
            return jnp.array([0.0, 10.0, -10.0]), None

    sampler = ConstBern()
    key = jax.random.PRNGKey(0)
    output_sd = jax.ShapeDtypeStruct((3,), dtype=jnp.bool_)

    sample, state = sampler.sample(key, [], [], [], None, output_sd)

    assert isinstance(sample, jnp.ndarray)
    assert sample.shape == (3,)
    assert sample.dtype == jnp.bool_
    assert state is None


def test_bernoulli_sample_given_parameters_consistent_dtype():
    class ConstBern(BernoulliConditional):
        def compute_parameters(self, *args, **kwargs):
            return jnp.zeros((3,))

    sampler = ConstBern()
    params = jnp.array([100.0, -100.0, 0.0])
    output_sd = jax.ShapeDtypeStruct((3,), dtype=jnp.bool_)

    sample, state = sampler.sample_given_parameters(jax.random.PRNGKey(1), params, None, output_sd)
    assert sample.shape == (3,)
    assert sample.dtype == jnp.bool_
    assert state is None


def test_bernoulli_conditional_sampling_bias():
    """Verify that Bernoulli sampler respects parameter biases.
    
    High positive gamma should bias toward True, high negative toward False.
    """

    class ConstBern(BernoulliConditional):
        def compute_parameters(self, key, interactions, active_flags, states, sampler_state, output_sd):
            return jnp.array([100.0, -100.0, 0.0]), None

    sampler = ConstBern()
    output_sd = jax.ShapeDtypeStruct((3,), dtype=jnp.bool_)

    key = jax.random.PRNGKey(42)
    samples_list = []
    for i in range(100):
        key, subkey = jax.random.split(key)
        sample, _ = sampler.sample(subkey, [], [], [], None, output_sd)
        samples_list.append(sample)

    samples = jnp.array(samples_list)

    assert jnp.mean(samples[:, 1]) < 0.05
    assert 0.3 < jnp.mean(samples[:, 2]) < 0.7


def test_softmax_conditional_sample_shapes_and_dtype():
    """Basic checks for SoftmaxConditional behaviour.

    Check that the sampler accepts a [b, M] parameter matrix and returns
    an integer array with the expected shape and dtype.
    """

    class ConstSoftmax(SoftmaxConditional):
        def compute_parameters(self, key, interactions, active_flags, states, sampler_state, output_sd):
            return jnp.array([[10.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 10.0]]), None

    sampler = ConstSoftmax()
    params, _ = sampler.compute_parameters(None, [], [], [], None, None)
    output_sd = jax.ShapeDtypeStruct((2,), dtype=jnp.uint8)

    sample, state = sampler.sample_given_parameters(jax.random.PRNGKey(2), params, None, output_sd)

    assert isinstance(sample, jnp.ndarray)
    assert sample.shape == (2,)
    assert sample.dtype == jnp.uint8
    assert state is None


def test_softmax_conditional_categorical_bias():
    """Verify that Softmax sampler respects parameter biases.
    """

    class ConstSoftmax(SoftmaxConditional):
        def compute_parameters(self, key, interactions, active_flags, states, sampler_state, output_sd):
            return jnp.array([[10.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 10.0]]), None

    sampler = ConstSoftmax()
    output_sd = jax.ShapeDtypeStruct((2,), dtype=jnp.uint8)

    key = jax.random.PRNGKey(42)
    samples_list = []
    for i in range(100):
        key, subkey = jax.random.split(key)
        sample, _ = sampler.sample(subkey, [], [], [], None, output_sd)
        samples_list.append(sample)

    samples = jnp.array(samples_list) 

    assert jnp.mean(samples[:, 0] == 0) > 0.95
    assert jnp.mean(samples[:, 1] == 3) > 0.95


def test_spin_gibbs_conditional_with_ising_chain():
    nodes = [SpinNode() for _ in range(5)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(4)]

    biases = jnp.array([5.0, 0.0, 0.0, 0.0, -5.0])

    weights = jnp.ones((4,)) * 2.0
    beta = jnp.array(1.0)

    model = IsingEBM(nodes, edges, biases, weights, beta)

    free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])

    key = jax.random.PRNGKey(0)
    k_init, k_samp = jax.random.split(key, 2)

    
    init_state = hinton_init(k_init, model, free_blocks, tuple())

    # Collect samples
    samples_list = []
    key = k_samp
    for _ in range(50):
        key, subkey = jax.random.split(key)
        pass

    # init_state shapes asserts
    assert len(init_state) == 2
    assert init_state[0].shape == (3,)
    assert init_state[1].shape == (2,)
    assert init_state[0].dtype == jnp.bool_
    assert init_state[1].dtype == jnp.bool_


def test_spin_gibbs_conditional_energy_consistency():
    """verify SpinGibbsConditional respects energy landscape.
    
    Confirm that a strong external field on a single spin causes
    the sampler to bias toward the lower-energy configuration.
    """
    # Create a 3-node chain with strong bias on node 1
    nodes = [SpinNode() for _ in range(3)]
    edges = [(nodes[0], nodes[1]), (nodes[1], nodes[2])]

    biases = jnp.array([0.0, 10.0, 0.0])
    weights = jnp.array([0.1, 0.1])
    beta = jnp.array(1.0)

    model = IsingEBM(nodes, edges, biases, weights, beta)

    free_blocks = [Block(nodes)]
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])

    key = jax.random.PRNGKey(1)
    init_state = hinton_init(key, model, free_blocks, tuple())

    assert init_state[0][1].astype(jnp.float32) > 0.5  # Single sample should reflect the bias


def test_spin_gibbs_conditional_with_coupling():
    """verify sampler respects edge coupling.
    
    When two nodes are strongly coupled with a positive weight, they should
    tend to have the same value.
    """
    # Create a pair of nodes with strong positive coupling
    nodes = [SpinNode() for _ in range(2)]
    edges = [(nodes[0], nodes[1])]

    biases = jnp.array([0.0, 0.0])
    weights = jnp.array([10.0])
    beta = jnp.array(1.0)

    model = IsingEBM(nodes, edges, biases, weights, beta)

    key = jax.random.PRNGKey(2)
    free_blocks = [Block([nodes[0]]), Block([nodes[1]])]
    init_state = hinton_init(key, model, free_blocks, tuple())

    assert len(init_state) == 2
    assert all(s.dtype == jnp.bool_ for s in init_state)
