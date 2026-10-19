"""Training orchestration tests with an intentionally host-only backend."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from thrml import Block, JaxBackend, SamplingSchedule, SpinNode
from thrml.models import IsingEBM, IsingSamplingProgram, IsingTrainingSpec, estimate_kl_grad


class SerialBackend:
    """Map requests serially and observe prescribed states without tracing execution."""

    def __init__(self):
        self.map_calls = []
        self.run_calls = []

    def map_chains(self, fn, *batched_args):
        leaves = jax.tree.leaves(batched_args)
        assert not any(isinstance(leaf, jax.core.Tracer) for leaf in leaves)
        count = leaves[0].shape[0]
        assert all(leaf.shape[0] == count for leaf in leaves)
        self.map_calls.append(count)
        outputs = [fn(*jax.tree.map(lambda value: value[i], batched_args)) for i in range(count)]
        return jax.tree.map(lambda *values: jnp.stack(values), *outputs)

    def run(self, key, program, schedule, initial, clamps, carry, observer):
        leaves = jax.tree.leaves((key, program.model, initial, clamps, carry))
        assert not any(isinstance(leaf, jax.core.Tracer) for leaf in leaves)
        self.run_calls.append((key, program, initial, clamps))
        state = initial
        if state is None:
            state = [jnp.ones((len(block),), dtype=jnp.bool_) for block in program.gibbs_spec.free_blocks]
        emitted = []
        for index in range(schedule.n_samples):
            carry, output = observer(program, state, clamps, carry, jnp.array(index))
            emitted.append(output)
        return carry, jax.tree.map(lambda *values: jnp.stack(values), *emitted)


def training_case(fully_clamped=False):
    nodes = [SpinNode() for _ in range(5)]
    edges = [(nodes[0], nodes[2]), (nodes[1], nodes[3]), (nodes[2], nodes[4])]
    model = IsingEBM(
        nodes,
        edges,
        jnp.array([0.2, -0.4, 0.3, 0.1, 0.6]),
        jnp.array([0.7, -0.2, 0.4]),
        jnp.array(1.3),
    )
    latent_blocks = [] if fully_clamped else [Block([nodes[0]]), Block([nodes[1]])]
    data_nodes = nodes[:4] if fully_clamped else nodes[2:4]
    data_blocks = [Block([node]) for node in data_nodes]
    negative_blocks = [Block([node]) for node in nodes[:4]]
    spec = IsingTrainingSpec(
        model,
        data_blocks,
        [Block([nodes[4]])],
        latent_blocks,
        negative_blocks,
        SamplingSchedule(1, 3, 1),
        SamplingSchedule(2, 4, 1),
    )
    patterns = [jnp.array([[True], [False], [True]]), jnp.array([[False], [True], [True]])]
    return dict(
        key=jax.random.key(923),
        training_spec=spec,
        bias_nodes=nodes,
        weight_edges=edges,
        data=[patterns[i % 2] for i in range(len(data_blocks))],
        conditioning_values=[jnp.array([True])],
        init_state_positive=[jnp.broadcast_to(patterns[i], (2, 3, 1)) for i in range(len(latent_blocks))],
        init_state_negative=[jnp.array([[True], [False], [False], [True]]) for _ in negative_blocks],
    )


def assert_tree_equal(left, right, *, atol=0):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for actual, expected in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        if atol:
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=atol)
        else:
            np.testing.assert_array_equal(actual, expected)
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype


@pytest.mark.parametrize("positive_none,negative_none", [(True, True), (True, False), (False, True), (False, False)])
def test_serial_training_optional_states_and_key_tree(positive_none, negative_none):
    case = training_case()
    if positive_none:
        case["init_state_positive"] = None
    if negative_none:
        case["init_state_negative"] = None
    backend = SerialBackend()
    result = estimate_kl_grad(**case, backend=backend)
    c_pos, c_neg = (1 if positive_none else 2), (1 if negative_none else 4)
    grad_w, grad_b, positive, negative = result
    assert [x.shape for x in positive] == [(c_pos, 3, 5), (c_pos, 3, 3)]
    assert [x.shape for x in negative] == [(c_neg, 5), (c_neg, 3)]
    assert backend.map_calls == [c_pos, *([3] * c_pos), c_neg]
    assert len(backend.run_calls) == c_pos * 3 + c_neg
    key_pos, key_neg = jax.random.split(case["key"], 2)
    keys = [*jax.random.split(key_pos, (c_pos, 3)).reshape(-1), *jax.random.split(key_neg, c_neg)]
    for index, ((key, program, initial, clamps), expected_key) in enumerate(zip(backend.run_calls, keys)):
        np.testing.assert_array_equal(jax.random.key_data(key), jax.random.key_data(expected_key))
        is_positive = index < c_pos * 3
        phase = "positive" if is_positive else "negative"
        assert program is getattr(case["training_spec"], f"program_{phase}")
        assert program.model is case["training_spec"].ebm
        continuation = positive_none if is_positive else negative_none
        if continuation:
            assert initial is None
        else:
            source = case[f"init_state_{phase}"]
            expected = (
                [x[index // 3, index % 3] for x in source] if is_positive else [x[index - c_pos * 3] for x in source]
            )
            assert_tree_equal(initial, expected)
        if is_positive:
            assert_tree_equal(clamps, [x[index % 3] for x in case["data"]] + case["conditioning_values"])
        else:
            assert_tree_equal(clamps, case["conditioning_values"])
    beta = case["training_spec"].ebm.beta
    for gradient, pos, neg in ((grad_b, positive[0], negative[0]), (grad_w, positive[1], negative[1])):
        expected = -beta * (jnp.mean(pos, axis=(0, 1), dtype=beta.dtype) - jnp.mean(neg, axis=0, dtype=beta.dtype))
        np.testing.assert_array_equal(gradient, expected)
        assert gradient.dtype == beta.dtype


@pytest.mark.parametrize("initial", [[], None])
def test_fully_clamped_positive_bypasses_backend_and_includes_conditioning(initial):
    case = training_case(fully_clamped=True)
    case["init_state_positive"] = initial
    backend = SerialBackend()
    _, _, (biases, weights), _ = estimate_kl_grad(**case, backend=backend)
    assert backend.map_calls == [4]
    assert len(backend.run_calls) == 4
    assert all(program is case["training_spec"].program_negative for _, program, _, _ in backend.run_calls)
    spins = (
        2
        * jnp.concatenate(case["data"] + [jnp.broadcast_to(case["conditioning_values"][0], (3, 1))], axis=1).astype(
            jnp.int8
        )
        - 1
    )
    np.testing.assert_array_equal(biases, spins[None])
    expected_edges = jnp.stack(
        [spins[:, 0] * spins[:, 2], spins[:, 1] * spins[:, 3], spins[:, 2] * spins[:, 4]], axis=-1
    )
    np.testing.assert_array_equal(weights, expected_edges[None])
    assert biases.shape == (1, 3, 5)
    assert weights.shape == (1, 3, 3)
    assert jnp.issubdtype(biases.dtype, jnp.floating)
    assert jnp.issubdtype(weights.dtype, jnp.floating)


@pytest.mark.parametrize("initial", [[], None])
def test_fully_clamped_positive_with_jax(initial):
    case = training_case(fully_clamped=True)
    case["init_state_positive"] = initial
    result = estimate_kl_grad(**case)
    assert result[2][0].shape == (1, 3, 5)


@pytest.mark.parametrize("phase", ["positive", "negative"])
def test_actual_jax_sampling_rejects_continuation(phase):
    case = training_case()
    case[f"init_state_{phase}"] = None
    with pytest.raises(ValueError, match="JaxBackend requires an explicit initial state"):
        estimate_kl_grad(**case)


def test_default_and_explicit_jax_training_and_enclosing_jit():
    case = training_case()
    default = estimate_kl_grad(**case)
    explicit = estimate_kl_grad(**case, backend=JaxBackend())
    compiled = eqx.filter_jit(estimate_kl_grad)(**case, backend=JaxBackend())
    assert_tree_equal(default, explicit)
    assert_tree_equal(default, compiled, atol=1e-7)


@pytest.mark.parametrize("argument", ["data", "conditioning_values", "init_state_positive", "init_state_negative"])
@pytest.mark.parametrize("defect", ["count", "rank", "size", "dtype"])
def test_training_preflight_state_structure(argument, defect):
    case = training_case()
    values = case[argument]
    if defect == "count":
        case[argument] = values[:-1]
    elif defect == "rank":
        case[argument] = [values[0][None], *values[1:]]
    elif defect == "size":
        case[argument] = [jnp.zeros((*values[0].shape[:-1], 2), dtype=jnp.bool_), *values[1:]]
    else:
        case[argument] = [values[0].astype(jnp.int8), *values[1:]]
    backend = SerialBackend()
    with pytest.raises(ValueError, match=argument):
        estimate_kl_grad(**case, backend=backend)
    assert backend.map_calls == []
    assert backend.run_calls == []


@pytest.mark.parametrize(
    "argument,shape,message",
    [
        ("data", (2, 1), "data"),
        ("init_state_positive", (1, 3, 1), "init_state_positive"),
        ("init_state_negative", (2, 1), "init_state_negative"),
        ("init_state_positive", (2, 2, 1), "init_state_positive"),
    ],
)
def test_training_preflight_inconsistent_batches(argument, shape, message):
    case = training_case()
    case[argument] = [jnp.zeros(shape, dtype=jnp.bool_), *case[argument][1:]]
    backend = SerialBackend()
    with pytest.raises(ValueError, match=message):
        estimate_kl_grad(**case, backend=backend)
    assert backend.map_calls == backend.run_calls == []


@pytest.mark.parametrize("argument", ["data", "init_state_positive", "init_state_negative"])
def test_training_preflight_empty_batch_or_replicas(argument):
    case = training_case()
    case[argument] = [x[:0] for x in case[argument]]
    backend = SerialBackend()
    with pytest.raises(ValueError, match=argument):
        estimate_kl_grad(**case, backend=backend)
    assert backend.map_calls == backend.run_calls == []


@pytest.mark.parametrize("phase", ["positive", "negative"])
def test_training_preflight_empty_initializers_with_free_blocks(phase):
    case = training_case()
    case[f"init_state_{phase}"] = []
    backend = SerialBackend()
    with pytest.raises(ValueError, match=f"init_state_{phase}"):
        estimate_kl_grad(**case, backend=backend)
    assert backend.map_calls == backend.run_calls == []


def test_training_preflight_nonempty_fully_clamped_initializer():
    case = training_case(fully_clamped=True)
    case["init_state_positive"] = [jnp.ones((2, 3, 1), dtype=jnp.bool_)]
    backend = SerialBackend()
    with pytest.raises(ValueError, match="init_state_positive"):
        estimate_kl_grad(**case, backend=backend)
    assert backend.map_calls == backend.run_calls == []


@pytest.mark.parametrize("phase", ["positive", "negative"])
@pytest.mark.parametrize("fully_clamped", [False, True])
def test_training_preflight_revalidates_both_schedules(phase, fully_clamped):
    case = training_case(fully_clamped)
    getattr(case["training_spec"], f"schedule_{phase}").n_samples = 0
    backend = SerialBackend()
    with pytest.raises(ValueError, match="n_samples"):
        estimate_kl_grad(**case, backend=backend)
    assert backend.map_calls == backend.run_calls == []


@pytest.mark.parametrize("argument", ["bias_nodes", "weight_edges"])
def test_training_preflight_unknown_statistics(argument):
    case = training_case()
    unknown = SpinNode()
    case[argument] = [unknown] if argument == "bias_nodes" else [(case["bias_nodes"][0], unknown)]
    backend = SerialBackend()
    with pytest.raises(ValueError, match="statistic nodes"):
        estimate_kl_grad(**case, backend=backend)
    assert backend.map_calls == backend.run_calls == []


def replace_phase(case, phase, free, clamped):
    spec = case["training_spec"]
    program = IsingSamplingProgram(spec.ebm, free, clamped)
    case["training_spec"] = eqx.tree_at(lambda value: getattr(value, f"program_{phase}"), spec, program)


@pytest.mark.parametrize(
    "defect", ["conditioning", "negative_free", "missing_bias", "missing_edge", "missing_data", "no_data"]
)
def test_training_preflight_partition_checks_even_for_unrequested_nodes(defect):
    case = training_case()
    spec = case["training_spec"]
    positive, negative = spec.program_positive.gibbs_spec, spec.program_negative.gibbs_spec
    nodes = spec.ebm.nodes
    # These statistics intentionally do not include nodes removed below.
    case["bias_nodes"] = [nodes[0]]
    case["weight_edges"] = [(nodes[0], nodes[2])]
    if defect == "conditioning":
        replace_phase(case, "negative", negative.free_blocks, [Block([SpinNode()])])
    elif defect == "negative_free":
        replace_phase(case, "negative", negative.free_blocks[:-1], negative.clamped_blocks)
    elif defect == "missing_data":
        replace_phase(case, "positive", positive.free_blocks, [positive.clamped_blocks[0], positive.clamped_blocks[-1]])
    elif defect == "no_data":
        replace_phase(case, "positive", negative.free_blocks, negative.clamped_blocks)
    elif defect == "missing_bias":
        model = IsingEBM([*nodes, SpinNode()], spec.ebm.edges, jnp.ones(6), spec.ebm.weights, spec.ebm.beta)
        case["training_spec"] = eqx.tree_at(lambda value: value.ebm, spec, model)
    else:
        model = IsingEBM(nodes, [*spec.ebm.edges, (nodes[0], SpinNode())], spec.ebm.biases, jnp.ones(4), spec.ebm.beta)
        case["training_spec"] = eqx.tree_at(lambda value: value.ebm, spec, model)
    backend = SerialBackend()
    with pytest.raises(ValueError, match="conditioning|free nodes|partition|data block"):
        estimate_kl_grad(**case, backend=backend)
    assert backend.map_calls == backend.run_calls == []


def test_training_partition_accepts_different_negative_order_and_grouping():
    case = training_case()
    spec = case["training_spec"]
    nodes = spec.ebm.nodes
    replace_phase(
        case,
        "negative",
        [Block([nodes[3], nodes[1]]), Block([nodes[2], nodes[0]])],
        spec.program_negative.gibbs_spec.clamped_blocks,
    )
    case["init_state_negative"] = [jnp.ones((4, 2), dtype=jnp.bool_), jnp.zeros((4, 2), dtype=jnp.bool_)]
    backend = SerialBackend()
    estimate_kl_grad(**case, backend=backend)
    assert len(backend.run_calls) == 10


def test_rebuilt_specs_expose_fresh_models_with_reused_external_backend():
    case = training_case()
    original = case["training_spec"]
    backend = SerialBackend()
    for offset in (0.0, 0.5, -0.5):
        model = eqx.tree_at(lambda value: value.biases, original.ebm, original.ebm.biases + offset)
        positive, negative = original.program_positive.gibbs_spec, original.program_negative.gibbs_spec
        spec = IsingTrainingSpec(
            model,
            positive.clamped_blocks[:-1],
            negative.clamped_blocks,
            positive.free_blocks,
            negative.free_blocks,
            original.schedule_positive,
            original.schedule_negative,
        )
        case["training_spec"] = spec
        estimate_kl_grad(**case, backend=backend)
        assert spec.program_positive.model is model
        assert spec.program_negative.model is model
        assert all(program.model is model for _, program, _, _ in backend.run_calls[-10:])
    assert len(backend.run_calls) == 30


def test_training_scan_rebuilds_spec_with_dynamic_model_parameters():
    case = training_case()
    original = case["training_spec"]
    positive, negative = original.program_positive.gibbs_spec, original.program_negative.gibbs_spec

    def step(parameters, key):
        biases, weights = parameters
        model = IsingEBM(original.ebm.nodes, original.ebm.edges, biases, weights, original.ebm.beta)
        spec = IsingTrainingSpec(
            model,
            positive.clamped_blocks[:-1],
            negative.clamped_blocks,
            positive.free_blocks,
            negative.free_blocks,
            original.schedule_positive,
            original.schedule_negative,
        )
        grad_w, grad_b, _, _ = estimate_kl_grad(
            key,
            spec,
            case["bias_nodes"],
            case["weight_edges"],
            case["data"],
            case["conditioning_values"],
            case["init_state_positive"],
            case["init_state_negative"],
        )
        return (biases - 0.01 * grad_b, weights - 0.01 * grad_w), (grad_b, grad_w)

    initial = (original.ebm.biases, original.ebm.weights)
    keys = jax.random.split(case["key"], 2)
    final, gradients = eqx.filter_jit(lambda values: jax.lax.scan(step, values, keys))(initial)
    expected = initial
    history = []
    for key in keys:
        expected, gradient = step(expected, key)
        history.append(gradient)
    assert_tree_equal(final, expected, atol=1e-7)
    assert_tree_equal(gradients, jax.tree.map(lambda *values: jnp.stack(values), *history), atol=1e-7)


def test_training_preflight_positive_batch_must_match_data():
    case = training_case()
    case["init_state_positive"] = [x[:, :2] for x in case["init_state_positive"]]
    backend = SerialBackend()
    with pytest.raises(ValueError, match="init_state_positive must match the data batch size"):
        estimate_kl_grad(**case, backend=backend)
    assert backend.map_calls == backend.run_calls == []


def test_training_preflight_fully_clamped_negative_is_not_valid():
    case = training_case()
    replace_phase(case, "negative", [], [Block(case["bias_nodes"])])
    case["init_state_negative"] = []
    backend = SerialBackend()
    with pytest.raises(ValueError, match="conditioning|free nodes|data block"):
        estimate_kl_grad(**case, backend=backend)
    assert backend.map_calls == backend.run_calls == []
