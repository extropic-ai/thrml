"""Backend dispatch, parity of the default JAX backend, ordered observer windows, and training through a backend."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from thrml import (
    Block,
    JaxBackend,
    SamplingSchedule,
    SpinNode,
    StateObserver,
    sample_blocks,
    sample_states,
    sample_with_observation,
)
from thrml.models import IsingEBM, IsingSamplingProgram, IsingTrainingSpec, estimate_kl_grad, estimate_moments
from thrml.models.ising import _ising_moment_observer


def assert_tree_equal(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert x.dtype == y.dtype
        np.testing.assert_array_equal(x, y)


def stack(outputs):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *outputs)


class SerialBackend:
    """A host-side backend that runs requests one at a time. It observes the same state for every record; `None`
    continues the state it retained for the program's free blocks."""

    def __init__(self):
        self.runs = []
        self.states = {}

    def run(self, key, program, schedule, init_state_free, state_clamp, observation_carry_init, f_observe):
        self.runs.append((key, program, init_state_free, state_clamp))
        layout = tuple(program.gibbs_spec.free_blocks)
        if init_state_free is not None:
            self.states[layout] = init_state_free
        state = self.states.setdefault(layout, [jnp.ones(len(block), dtype=jnp.bool_) for block in layout])
        carry, outputs = observation_carry_init, []
        for i in range(schedule.n_samples):
            carry, output = f_observe(program, state, state_clamp, carry, jnp.array(i))
            outputs.append(output)
        return carry, stack(outputs)

    def map_chains(self, fn, *batched_args):
        n = jax.tree.leaves(batched_args)[0].shape[0]
        return stack([fn(*jax.tree.map(lambda x: x[i], batched_args)) for i in range(n)])


@pytest.fixture
def ising():
    nodes = [SpinNode() for _ in range(3)]
    edges = [(nodes[0], nodes[1]), (nodes[1], nodes[2])]
    model = IsingEBM(nodes, edges, jnp.array([0.3, -0.2, 0.4]), jnp.array([0.6, -0.7]), jnp.array(1.2))
    program = IsingSamplingProgram(model, [Block(nodes[:1]), Block(nodes[1:2])], [Block(nodes[2:])])
    return program, [jnp.array([False]), jnp.array([True])], [jnp.array([True])]


def test_external_backend_receives_the_logical_request(ising):
    program, state, clamps = ising
    key, schedule, readout = jax.random.key(0), SamplingSchedule(0, 2, 1), [Block(program.model.nodes[::-1])]
    nodes, edges = program.model.nodes, program.model.edges
    backend = SerialBackend()
    sample_states(key, program, schedule, state, clamps, readout, backend=backend)
    samples = sample_states(key, program, schedule, None, clamps, readout, backend=backend)
    moments = estimate_moments(key, nodes, edges, program, schedule, None, clamps, backend=backend)

    assert all(run[1] is program and run[3] is clamps for run in backend.runs)
    assert [run[2] for run in backend.runs] == [state, None, None]
    # The None requests continue the state of the first request.
    spins = jnp.concatenate(state + clamps)
    np.testing.assert_array_equal(samples[0], jnp.stack([spins[::-1]] * 2))
    np.testing.assert_array_equal(moments[0], 2 * spins.astype(jnp.float32) - 1)


def test_logical_program_matches_the_compiled_program(ising):
    program, state, clamps = ising
    key, schedule, readout = jax.random.key(13), SamplingSchedule(2, 4, 1), [Block(program.model.nodes[::-1])]
    run = eqx.filter_jit(lambda p: sample_states(key, p, schedule, state, clamps, readout))
    for weights in (program.model.weights, -program.model.weights):
        logical = eqx.tree_at(lambda p: p.model.weights, program, weights)
        compiled = JaxBackend().compile(logical)
        expected = sample_states(key, compiled, schedule, state, clamps, readout)
        assert_tree_equal(sample_states(key, logical, schedule, state, clamps, readout), expected)
        assert_tree_equal(run(logical), run(compiled))
    with pytest.raises(TypeError, match="compile"):
        sample_blocks(key, state, clamps, program, [None, None])


def test_none_and_empty_initial_states(ising):
    program, state, clamps = ising
    key, schedule = jax.random.key(4), SamplingSchedule(0, 1, 0)
    with pytest.raises(ValueError, match="explicit initial state"):
        sample_states(key, program, schedule, None, clamps, [])

    full = IsingSamplingProgram(program.model, [], [Block(program.model.nodes)])
    clamps = [jnp.array([True, False, True])]
    np.testing.assert_array_equal(
        sample_states(key, full, schedule, [], clamps, [Block(full.model.nodes)])[0][0], clamps[0]
    )
    # Without first-moment nodes, the first moment is the unit statistic.
    first, second = estimate_moments(key, [], full.model.edges, full, schedule, [], clamps)
    np.testing.assert_array_equal(first, [1.0])
    np.testing.assert_array_equal(second, [-1.0, -1.0])


def test_estimate_moments_rejects_unknown_nodes_before_running(ising):
    program, state, clamps = ising
    backend = SerialBackend()
    with pytest.raises(ValueError, match="statistic nodes"):
        estimate_moments(
            jax.random.key(0), [SpinNode()], [], program, SamplingSchedule(0, 1, 0), state, clamps, backend=backend
        )
    assert not backend.runs


def test_schedule_rejects_a_bad_range():
    with pytest.raises(ValueError, match="n_samples"):
        SamplingSchedule(0, 0, 1)


class ReplayBackend:
    """Observes prescribed records in windows, as a device that reads records in batches: each window observes its
    first record, then scans the rest, with the carry and the record indices of the whole request."""

    def __init__(self, records, partitions):
        self.records = records
        self.partitions = partitions

    def run(self, key, program, schedule, init_state_free, state_clamp, observation_carry_init, f_observe):
        carry, outputs, start = observation_carry_init, [], 0
        for size in self.partitions:
            free, clamped = jax.tree.map(lambda x: x[start : start + size], self.records)
            carry, first = f_observe(program, [x[0] for x in free], [x[0] for x in clamped], carry, jnp.array(start))

            def step(carry, inputs):
                free, clamped, index = inputs
                return f_observe(program, free, clamped, carry, index)

            rest = ([x[1:] for x in free], [x[1:] for x in clamped], start + jnp.arange(1, size))
            carry, rest = jax.lax.scan(step, carry, rest)
            outputs.append(jax.tree.map(lambda x, xs: jnp.concatenate([x[None], xs]), first, rest))
            start += size
        return carry, jax.tree.map(lambda *xs: jnp.concatenate(xs), *outputs)


def test_windows_match_one_request(ising):
    program, _, clamps = ising
    free = [jnp.array([[True], [False], [True], [True], [False], [False]]), jnp.array([[False], [True], [True]] * 2)]
    records = (free, [jnp.ones((6, 1), dtype=jnp.bool_)])

    def indexed(program, state_free, state_clamped, carry, iteration):
        carry = 3 * carry + state_free[0][0].astype(jnp.int32) + 2 * iteration
        return carry, (iteration, carry)

    moments = _ising_moment_observer(program.model.nodes, program.model.edges)
    observers = [
        (indexed, jnp.array(0)),
        (moments, moments.init()),
        (StateObserver([Block(program.model.nodes)]), None),
    ]
    schedule = SamplingSchedule(0, 6, 0)
    windowed = []
    for observer, carry in observers:
        windows, single = (
            sample_with_observation(
                jax.random.key(1), program, schedule, None, clamps, carry, observer, backend=ReplayBackend(records, p)
            )
            for p in ((1, 3, 2), (6,))
        )
        assert_tree_equal(windows, single)
        windowed.append(windows)
    np.testing.assert_array_equal(windowed[0][1][0], np.arange(6))
    # The first record casts the floating zeros of the moment observer to the integer dtype of the spin products.
    assert jnp.issubdtype(windowed[1][0][0].dtype, jnp.integer)


def test_map_chains_keeps_structure_order_and_empty_trees():
    backend = JaxBackend()
    values = {"x": jnp.arange(6).reshape(2, 3)}
    result = backend.map_chains(
        lambda row, empty: {"twice": backend.map_chains(lambda x: (2 * x, None), row["x"])}, values, []
    )
    assert_tree_equal(result, {"twice": (2 * values["x"], None)})


def training_case(fully_clamped=False):
    nodes = [SpinNode() for _ in range(5)]
    edges = [(nodes[0], nodes[2]), (nodes[1], nodes[3]), (nodes[2], nodes[4])]
    model = IsingEBM(nodes, edges, jnp.array([0.2, -0.4, 0.3, 0.1, 0.6]), jnp.array([0.7, -0.2, 0.4]), jnp.array(1.3))
    data_block = Block(nodes[:4]) if fully_clamped else Block(nodes[2:4])
    latent_blocks = [] if fully_clamped else [Block(nodes[:2])]
    schedule = SamplingSchedule(1, 3, 1)
    spec = IsingTrainingSpec(
        model, [data_block], [Block(nodes[4:])], latent_blocks, [Block(nodes[:4])], schedule, schedule
    )
    keys = jax.random.split(jax.random.key(2), 3)
    return dict(
        key=jax.random.key(923),
        training_spec=spec,
        bias_nodes=nodes,
        weight_edges=edges,
        data=[jax.random.bernoulli(keys[0], 0.5, (3, len(data_block)))],
        conditioning_values=[jnp.array([True])],
        init_state_positive=[jax.random.bernoulli(keys[1], 0.5, (2, 3, 2)) for _ in latent_blocks],
        init_state_negative=[jax.random.bernoulli(keys[2], 0.5, (4, 4))],
    )


@pytest.mark.parametrize("explicit", [False, True])
def test_training_through_an_external_backend(explicit):
    case = training_case()
    if not explicit:
        case["init_state_positive"] = case["init_state_negative"] = None
    backend = SerialBackend()
    _, _, positive, negative = estimate_kl_grad(**case, backend=backend)

    n_positive, n_negative = (2, 4) if explicit else (1, 1)
    assert [x.shape for x in positive] == [(n_positive, 3, 5), (n_positive, 3, 3)]
    assert [x.shape for x in negative] == [(n_negative, 5), (n_negative, 3)]
    spec = case["training_spec"]
    programs = [spec.program_positive] * (3 * n_positive) + [spec.program_negative] * n_negative
    assert len(backend.runs) == len(programs)
    for (_, program, initial, _), expected in zip(backend.runs, programs):
        assert program is expected and program.model is spec.ebm
        assert (initial is None) != explicit
    assert_tree_equal(backend.runs[0][3], [case["data"][0][0], *case["conditioning_values"]])
    assert backend.runs[-1][3] is case["conditioning_values"]


def test_fully_clamped_positive_phase_bypasses_the_backend():
    case = training_case(fully_clamped=True)
    backend = SerialBackend()
    _, _, (biases, weights), _ = estimate_kl_grad(**case, backend=backend)

    assert all(run[1] is case["training_spec"].program_negative for run in backend.runs)
    spins = 2 * jnp.concatenate([case["data"][0], jnp.ones((3, 1), dtype=jnp.bool_)], axis=1).astype(jnp.float32) - 1
    np.testing.assert_array_equal(biases, spins[None])
    np.testing.assert_array_equal(weights, (spins[:, [0, 1, 2]] * spins[:, [2, 3, 4]])[None])
    assert biases.dtype == weights.dtype == case["training_spec"].ebm.beta.dtype


def test_jax_training_matches_the_default():
    case = training_case()
    assert_tree_equal(estimate_kl_grad(**case, backend=JaxBackend()), estimate_kl_grad(**case))


def test_training_spec_rejects_inconsistent_free_nodes():
    spec = training_case()["training_spec"]
    nodes, schedule = spec.ebm.nodes, spec.schedule_positive
    with pytest.raises(ValueError, match="Negative free nodes"):
        IsingTrainingSpec(
            spec.ebm,
            [Block(nodes[2:4])],
            [Block(nodes[4:])],
            [Block(nodes[:2])],
            [Block(nodes[:3])],
            schedule,
            schedule,
        )


def test_training_rejects_a_bad_negative_state_before_running():
    case = training_case()
    case["init_state_negative"] = [jnp.ones((4, 3), dtype=jnp.bool_)]  # the negative free block has 4 nodes
    backend = SerialBackend()
    with pytest.raises(RuntimeError, match="block length"):
        estimate_kl_grad(**case, backend=backend)
    assert not backend.runs
