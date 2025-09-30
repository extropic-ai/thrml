def test_readme_quick_example():
    import jax
    import jax.numpy as jnp

    from thrml import Block, SamplingSchedule, SpinNode, sample_states
    from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

    nodes = [SpinNode() for _ in range(5)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(4)]
    biases = jnp.zeros((5,))
    weights = jnp.ones((4,)) * 0.5
    beta = jnp.array(1.0)
    model = IsingEBM(nodes, edges, biases, weights, beta)

    free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])

    key = jax.random.key(0)
    k_init, k_samp = jax.random.split(key, 2)
    init_state = hinton_init(k_init, model, free_blocks, ())
    schedule = SamplingSchedule(n_warmup=100, n_samples=1000, steps_per_sample=2)

    samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])

    assert samples[0].shape == (1000, 5)
