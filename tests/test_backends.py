"""Backend dispatch, exact JAX compatibility, and portable observer contracts."""

import importlib
import inspect
import subprocess
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from thrml import (
    Block,
    BlockGibbsSpec,
    BlockSamplingProgram,
    FactorSamplingProgram,
    JaxBackend,
    MomentAccumulatorObserver,
    SamplingBackend,
    SamplingContext,
    SamplingSchedule,
    SpinNode,
    StateObserver,
    sample_blocks,
    sample_single_block,
    sample_states,
    sample_with_observation,
)
from thrml.conditional_samplers import AbstractConditionalSampler
from thrml.models import IsingEBM, IsingSamplingProgram, SpinGibbsConditional, estimate_kl_grad, estimate_moments


def assert_tree_equal(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert x.shape == y.shape
        assert x.dtype == y.dtype
        np.testing.assert_array_equal(x, y)


@pytest.fixture
def ising():
    nodes = [SpinNode() for _ in range(3)]
    edges = [(nodes[0], nodes[1]), (nodes[1], nodes[2])]
    model = IsingEBM(nodes, edges, jnp.array([0.3, -0.2, 0.4]), jnp.array([0.6, -0.7]), jnp.array(1.2))
    program = IsingSamplingProgram(model, [Block([n]) for n in nodes[:2]], [Block(nodes[2:])])
    return program, [jnp.array([True]), jnp.array([False])], [jnp.array([True])]


class RecordingBackend:
    def __init__(self):
        self.calls = []
        self.result = (object(), object())

    def __bool__(self):
        return False

    def run(self, *args):
        self.calls.append(args)
        return self.result

    def map_chains(self, fn, *args):
        raise AssertionError("Unexpected mapping")


def test_dispatch_preserves_original_request(ising, monkeypatch):
    program, _, clamps = ising
    backend = RecordingBackend()
    assert SamplingBackend.__dict__.get("_is_protocol")
    assert SamplingContext.__dict__.get("_is_protocol")
    key, schedule, carry, observer = jax.random.key(17), SamplingSchedule(0, 1, 0), object(), object()

    def forbidden(*args, **kwargs):
        raise AssertionError("External dispatch must not execute JAX")

    monkeypatch.setattr(JaxBackend, "compile", forbidden)
    monkeypatch.setattr(importlib.import_module("thrml.block_sampling"), "_sample_with_observation_jax", forbidden)
    result = sample_with_observation(key, program, schedule, None, clamps, carry, observer, backend=backend)
    assert result is backend.result
    assert len(backend.calls) == 1
    for actual, expected in zip(backend.calls[0], (key, program, schedule, None, clamps, carry, observer)):
        assert actual is expected
    assert backend.calls[0][1].model is program.model


@pytest.mark.parametrize("field", ["n_warmup", "n_samples", "steps_per_sample"])
@pytest.mark.parametrize("invalid", [True, 1.0, "1", jnp.array(1)])
def test_schedule_types(ising, field, invalid):
    args = dict(n_warmup=0, n_samples=1, steps_per_sample=0)
    args[field] = invalid
    with pytest.raises(TypeError, match=field):
        SamplingSchedule(**args)
    schedule = SamplingSchedule(0, 1, 0)
    setattr(schedule, field, invalid)
    program, state, clamps = ising
    backend = RecordingBackend()
    with pytest.raises(TypeError, match=field):
        sample_with_observation(
            jax.random.key(1), program, schedule, state, clamps, None, StateObserver([]), backend=backend
        )
    assert not backend.calls


@pytest.mark.parametrize("field, invalid", [("n_warmup", -1), ("n_samples", 0), ("steps_per_sample", -1)])
def test_schedule_ranges_and_mutation(ising, field, invalid):
    args = dict(n_warmup=0, n_samples=1, steps_per_sample=0)
    args[field] = invalid
    with pytest.raises(ValueError, match=field):
        SamplingSchedule(**args)
    schedule = SamplingSchedule(0, 1, 0)
    setattr(schedule, field, invalid)
    program, state, clamps = ising
    backend = RecordingBackend()
    with pytest.raises(ValueError, match=field):
        sample_states(jax.random.key(1), program, schedule, state, clamps, [], backend=backend)
    with pytest.raises(ValueError, match=field):
        JaxBackend().run(jax.random.key(1), program, schedule, state, clamps, None, StateObserver([]))
    assert not backend.calls


def test_integral_schedule_and_hash():
    schedule = SamplingSchedule(np.int64(0), np.int64(1), np.int64(0))
    assert hash(schedule) == hash((0, 1, 0))


@pytest.mark.parametrize("which", ["free_count", "clamp_count", "shape", "dtype", "readout"])
def test_common_validation_precedes_execution(ising, which):
    program, state, clamps = ising
    readout = program.gibbs_spec.free_blocks
    if which == "free_count":
        state = []
    elif which == "clamp_count":
        clamps = []
    elif which == "shape":
        state = [jnp.ones(2, dtype=bool), state[1]]
    elif which == "dtype":
        clamps = [jnp.ones(1, dtype=jnp.int8)]
    else:
        readout = [Block([SpinNode()])]
    backend = RecordingBackend()
    with pytest.raises((RuntimeError, ValueError)):
        sample_states(jax.random.key(1), program, SamplingSchedule(0, 1, 0), state, clamps, readout, backend=backend)
    assert not backend.calls
    if which != "readout":
        with pytest.raises(RuntimeError):
            JaxBackend().run(
                jax.random.key(1), program, SamplingSchedule(0, 1, 0), state, clamps, None, StateObserver([])
            )


def test_none_and_empty_initialization(ising):
    program, _, clamps = ising
    key, schedule = jax.random.key(4), SamplingSchedule(0, 1, 0)
    with pytest.raises(ValueError, match="explicit initial state"):
        sample_states(key, program, schedule, None, clamps, [])
    full = IsingSamplingProgram(program.model, [], [Block(program.model.nodes)])
    clamps = [jnp.array([True, False, True])]
    backend = RecordingBackend()
    sample_with_observation(key, full, schedule, [], clamps, None, StateObserver([]), backend=backend)
    assert backend.calls[0][3] == []
    np.testing.assert_array_equal(
        sample_states(key, full, schedule, [], clamps, full.gibbs_spec.clamped_blocks)[0], clamps[0][None]
    )
    moments = estimate_moments(key, program.model.nodes, program.model.edges, full, schedule, [], clamps)
    np.testing.assert_array_equal(moments[0], [1.0, -1.0, 1.0])
    np.testing.assert_array_equal(moments[1], [-1.0, -1.0])


@pytest.mark.parametrize("n_warmup", [0, 2])
@pytest.mark.parametrize("n_samples", [1, 4])
@pytest.mark.parametrize("steps", [0, 2])
@pytest.mark.parametrize("superblock", [False, True])
def test_logical_and_original_executable_parity(ising, n_warmup, n_samples, steps, superblock):
    program, state, clamps = ising
    if superblock:
        program = IsingSamplingProgram(
            program.model, [tuple(program.gibbs_spec.free_blocks)], program.gibbs_spec.clamped_blocks
        )
    spec = program.gibbs_spec
    compiled = FactorSamplingProgram(
        spec, [SpinGibbsConditional() for _ in spec.free_blocks], program.model.factors, []
    )
    backend = JaxBackend()
    assert backend.compile(compiled) is compiled
    key, schedule, readout = (
        jax.random.key(13),
        SamplingSchedule(n_warmup, n_samples, steps),
        [Block(program.model.nodes[::-1])],
    )
    expected = sample_states(key, compiled, schedule, state, clamps, readout)
    actual = sample_states(
        key=key,
        program=program,
        schedule=schedule,
        init_state_free=state,
        state_clamp=clamps,
        nodes_to_sample=readout,
        backend=backend,
    )
    assert_tree_equal(actual, expected)
    assert_tree_equal(sample_states(key, program, schedule, state, clamps, readout), expected)
    assert actual[0].shape == (n_samples, 3)
    assert actual[0].dtype == jnp.bool_
    assert_tree_equal(
        estimate_moments(
            key, program.model.nodes, program.model.edges, program, schedule, state, clamps, backend=backend
        ),
        estimate_moments(key, program.model.nodes, program.model.edges, compiled, schedule, state, clamps),
    )


def test_jit_and_batched_sampling(ising):
    program, state, clamps = ising
    schedule = SamplingSchedule(1, 3, 1)
    keys = jax.random.split(jax.random.key(7), 3)
    run = eqx.filter_jit(lambda key, p: sample_states(key, p, schedule, state, clamps, [Block(p.model.nodes)]))
    actual = jax.vmap(lambda key: run(key, program))(keys)
    expected = jax.tree.map(lambda *xs: jnp.stack(xs), *[run(key, program) for key in keys])
    assert_tree_equal(actual, expected)


def test_low_level_migration_and_compile_identity(ising):
    program, state, clamps = ising
    with pytest.raises(TypeError, match=r"JaxBackend\(\).compile\(program\)"):
        sample_blocks(jax.random.key(1), state, clamps, program, [None, None])
    with pytest.raises(TypeError, match=r"JaxBackend\(\).compile\(program\)"):
        sample_single_block(jax.random.key(1), state, clamps, program, 0, None)
    plain = BlockSamplingProgram(program.gibbs_spec, [SpinGibbsConditional(), SpinGibbsConditional()], [])
    assert JaxBackend().compile(plain) is plain
    with pytest.raises(TypeError, match="program type"):
        JaxBackend().compile(object())


def compiled_fields(program, state_free, state_clamped, carry, iteration):
    assert len(program.samplers) == 2
    fields = jnp.stack(
        [
            sum(
                jnp.sum(interaction.weights * active.astype(interaction.weights.dtype))
                for interaction, active in zip(interactions, flags)
            )
            for interactions, flags in zip(program.per_block_interactions, program.per_block_interaction_active)
        ]
    )
    return carry, fields


def test_fresh_targets_under_jit():
    nodes = [SpinNode(), SpinNode()]
    model = IsingEBM(nodes, [(nodes[0], nodes[1])], jnp.array([0.2, 0.4]), jnp.array([0.7]), jnp.array(1.3))
    program = IsingSamplingProgram(model, [Block([n]) for n in nodes], [])
    backend = JaxBackend()
    backend.compile(program)
    schedule, state, key = SamplingSchedule(0, 1, 0), [jnp.array([True]), jnp.array([True])], jax.random.key(11)
    run = eqx.filter_jit(
        lambda p: sample_with_observation(key, p, schedule, state, [], None, compiled_fields, backend=backend)[1]
    )
    targets = [
        model,
        eqx.tree_at(lambda m: m.biases, model, jnp.array([0.9, 0.6])),
        eqx.tree_at(lambda m: m.weights, model, jnp.array([1.1])),
        eqx.tree_at(lambda m: m.beta, model, jnp.array(2.1)),
        model,
    ]
    for target in targets:
        updated = eqx.tree_at(lambda p: p.model, program, target)
        assert updated.gibbs_spec is program.gibbs_spec
        expected = (target.beta * (target.biases + target.weights[0]))[None]
        np.testing.assert_allclose(run(updated), expected, rtol=1e-6)
        for executable in (updated, backend.compile(updated)):
            for selected in (None, backend):
                result = sample_with_observation(
                    key, executable, schedule, state, [], None, compiled_fields, backend=selected
                )[1]
                np.testing.assert_allclose(result, expected, rtol=1e-6)


class CountingSampler(AbstractConditionalSampler):
    def init(self):
        return jnp.array(0, dtype=jnp.int32)

    def sample(self, key, interactions, active_flags, states, sampler_state, output_sd):
        count = sampler_state + 1
        return jnp.full(output_sd.shape, count, dtype=output_sd.dtype), count


@pytest.mark.parametrize("warmup, records, steps", [(0, 1, 0), (0, 4, 2), (3, 4, 2), (2, 3, 0)])
def test_record_timing_and_sampler_memory(warmup, records, steps):
    from .test_block_sampling import ContinousScalarNode

    block = Block([ContinousScalarNode()])
    spec = BlockGibbsSpec([block], [], {ContinousScalarNode: jax.ShapeDtypeStruct((), jnp.float32)})
    program = FactorSamplingProgram(spec, [CountingSampler()], [], [])
    result = sample_states(
        jax.random.key(0), program, SamplingSchedule(warmup, records, steps), [jnp.zeros(1)], [], [block]
    )
    np.testing.assert_array_equal(result[0][:, 0], warmup + np.arange(records) * steps)


def test_map_chains_preserves_structure_order_and_empty_trees():
    backend = JaxBackend()
    values = {"x": jnp.arange(6).reshape(2, 3)}
    result = backend.map_chains(
        lambda row, empty: {"twice": backend.map_chains(lambda x: (x * 2, None), row["x"])}, values, []
    )
    assert_tree_equal(result, {"twice": (values["x"] * 2, None)})
    with pytest.raises(ValueError):
        backend.map_chains(lambda x, y: x + y, jnp.arange(2), jnp.arange(3))


def test_signatures_and_independent_imports():
    assert list(inspect.signature(sample_with_observation).parameters)[3] == "init_chain_state"
    assert list(inspect.signature(sample_states).parameters)[3] == "init_state_free"
    assert list(inspect.signature(estimate_moments).parameters)[5] == "init_state"
    for fn in (sample_with_observation, sample_states, estimate_moments, estimate_kl_grad):
        assert inspect.signature(fn).parameters["backend"].kind == inspect.Parameter.KEYWORD_ONLY
    for module in ("thrml", "thrml.models", "thrml.backends"):
        subprocess.run([sys.executable, "-c", f"import {module}"], check=True)


class ReplayBackend:
    """Test-only host execution with ordered, jitted observation windows."""

    def __init__(self, records, partitions):
        self.records = records
        self.partitions = partitions

    def run(self, key, program, schedule, init_state_free, state_clamp, observation_carry_init, f_observe):
        assert sum(self.partitions) == schedule.n_samples

        @eqx.filter_jit
        def observe_window(records, carry, start):
            free, clamps = jax.tree.map(lambda x: x[0], records)
            carry, first = f_observe(program, free, clamps, carry, start)

            def step(carry, inputs):
                free, clamps, index = inputs
                return f_observe(program, free, clamps, carry, index)

            n_records = jax.tree.leaves(records)[0].shape[0]
            remaining = jax.tree.map(lambda x: x[1:], records)
            carry, rest = jax.lax.scan(step, carry, (*remaining, start + jnp.arange(1, n_records, dtype=jnp.int32)))
            return carry, jax.tree.map(lambda x, xs: jnp.concatenate((x[None], xs)), first, rest)

        carry, outputs, offset = observation_carry_init, [], 0
        for size in self.partitions:
            records = jax.tree.map(lambda x: x[offset : offset + size], self.records)
            carry, output = observe_window(records, carry, jnp.array(offset, dtype=jnp.int32))
            outputs.append(output)
            offset += size
        return carry, jax.tree.map(lambda *xs: jnp.concatenate(xs), *outputs)

    def map_chains(self, fn, *args):
        raise AssertionError("This replay fixture contains a single request")


def replay_records():
    free = [
        jnp.array([[True], [False], [True], [True], [False], [False]]),
        jnp.array([[False], [True], [True], [False], [True], [False]]),
    ]
    return free, [jnp.ones((6, 1), dtype=jnp.bool_)]


def sequential_observations(program, observer, records, carry):
    outputs = []
    for i in range(jax.tree.leaves(records)[0].shape[0]):
        free, clamps = jax.tree.map(lambda x: x[i], records)
        carry, output = observer(program, free, clamps, carry, jnp.array(i, dtype=jnp.int32))
        outputs.append(output)
    return carry, jax.tree.map(lambda *xs: jnp.stack(xs), *outputs)


@pytest.mark.parametrize("partitions", [(1, 3, 2), (2, 1, 3)])
def test_ordered_window_carry_and_global_indices(ising, partitions):
    program, _, clamps = ising
    records = replay_records()

    def observe(program, state_free, state_clamped, carry, iteration):
        value = state_free[0][0].astype(jnp.int32)
        carry = 3 * carry + value + 2 * iteration + 1
        return carry, {"index": iteration, "carry": carry}

    initial = jnp.array(0, dtype=jnp.int32)
    backend = ReplayBackend(records, partitions)
    actual = sample_with_observation(
        jax.random.key(1), program, SamplingSchedule(0, 6, 0), None, clamps, initial, observe, backend=backend
    )
    assert_tree_equal(actual, sequential_observations(program, observe, records, initial))
    np.testing.assert_array_equal(actual[1]["index"], np.arange(6))
    # Each continuation request restarts record indices and explicit carry.
    assert_tree_equal(
        sample_with_observation(
            jax.random.key(2), program, SamplingSchedule(0, 6, 0), None, clamps, initial, observe, backend=backend
        ),
        actual,
    )


@pytest.mark.parametrize("partitions", [(1, 3, 2), (2, 1, 3), (1,)])
def test_state_observer_through_windows(ising, partitions):
    program, _, clamps = ising
    records = jax.tree.map(lambda x: x[: sum(partitions)], replay_records())
    observer = StateObserver([Block(program.model.nodes[::-1])])
    actual = sample_with_observation(
        jax.random.key(1),
        program,
        SamplingSchedule(0, sum(partitions), 0),
        None,
        clamps,
        None,
        observer,
        backend=ReplayBackend(records, partitions),
    )
    assert_tree_equal(actual, sequential_observations(program, observer, records, None))
    assert actual[0] is None
    assert actual[1][0].dtype == jnp.bool_
    assert actual[1][0].shape == (sum(partitions), 3)


@pytest.mark.parametrize("partitions", [(1, 3, 2), (2, 1, 3)])
@pytest.mark.parametrize("floating", [False, True])
def test_moment_observer_through_windows(ising, partitions, floating):
    program, _, clamps = ising

    def transform(state, _):
        return [2 * x.astype(jnp.float32 if floating else jnp.int8) - 1 for x in state]

    observer = MomentAccumulatorObserver(([(n,) for n in program.model.nodes], program.model.edges), transform)
    records = replay_records()
    expected = sequential_observations(program, observer, records, observer.init())
    actual = sample_with_observation(
        jax.random.key(1),
        program,
        SamplingSchedule(0, 6, 0),
        None,
        clamps,
        observer.init(),
        observer,
        backend=ReplayBackend(records, partitions),
    )
    assert_tree_equal(actual, expected)
    assert actual[1] is None
    first, _ = observer(program, [x[0] for x in records[0]], clamps, observer.init(), jnp.array(0))
    assert actual[0][0].dtype == first[0].dtype


@pytest.mark.parametrize("partitions", [(1, 3, 2), (2, 1, 3)])
def test_estimate_moments_normalizes_all_records_once(ising, partitions):
    program, _, clamps = ising
    records = replay_records()
    actual = estimate_moments(
        jax.random.key(1),
        program.model.nodes,
        program.model.edges,
        program,
        SamplingSchedule(0, 6, 0),
        None,
        clamps,
        backend=ReplayBackend(records, partitions),
    )
    signed = 2 * jnp.concatenate(records[0] + records[1], axis=1).astype(jnp.float32) - 1
    np.testing.assert_allclose(actual[0], jnp.mean(signed, axis=0))
    np.testing.assert_allclose(actual[1], jnp.mean(signed[:, :-1] * signed[:, 1:], axis=0))


def test_mixed_boolean_uint8_observer_windows():
    from thrml import CategoricalNode

    spin, cat = SpinNode(), CategoricalNode()
    spec = BlockGibbsSpec(
        [Block([spin]), Block([cat])],
        [],
        {SpinNode: jax.ShapeDtypeStruct((), jnp.bool_), CategoricalNode: jax.ShapeDtypeStruct((), jnp.uint8)},
    )
    program = BlockSamplingProgram(spec, [CountingSampler(), CountingSampler()], [])
    records = (
        [
            jnp.array([[True], [False], [True], [True], [False], [True]]),
            jnp.array([[2], [3], [4], [5], [6], [7]], dtype=jnp.uint8),
        ],
        [],
    )
    observer = MomentAccumulatorObserver([[(spin, cat)]])
    with jax.numpy_dtype_promotion("standard"):
        expected = sequential_observations(program, observer, records, observer.init())
        actual = sample_with_observation(
            jax.random.key(1),
            program,
            SamplingSchedule(0, 6, 0),
            None,
            [],
            observer.init(),
            observer,
            backend=ReplayBackend(records, (1, 3, 2)),
        )
    assert_tree_equal(actual, expected)
