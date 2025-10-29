import jax
import jax.numpy as jnp

from thrml.block_management import Block
from thrml.block_sampling import SamplingSchedule, sample_with_observation
from thrml.models.ising import IsingEBM, IsingSamplingProgram
from thrml.observers import EnergyObserver
from thrml.pgm import SpinNode


def _make_ising():
    nodes = [SpinNode() for _ in range(2)]
    edges = [(nodes[0], nodes[1])]
    biases = jnp.array([0.3, -0.2])
    weights = jnp.array([0.5])
    beta = jnp.array(1.0)
    ebm = IsingEBM(nodes, edges, biases, weights, beta)
    program = IsingSamplingProgram(ebm, [Block(nodes)], [])
    return ebm, program


def test_energy_observer_matches_model_energy():
    ebm, program = _make_ising()
    observer = EnergyObserver(ebm)

    spin_state = jnp.array([True, False])

    initial_carry = observer.init()
    carry_out, observation = observer(
        program,
        [spin_state],
        [],
        initial_carry,
        jnp.array(0, dtype=jnp.int32),
    )

    expected = ebm.energy([spin_state], program.gibbs_spec.blocks)

    assert carry_out is None
    assert jnp.allclose(observation, expected)


def test_energy_observer_with_sampling_loop_updates_carry():
    ebm, program = _make_ising()
    base_state = jnp.array([True, True])
    schedule = SamplingSchedule(n_warmup=0, n_samples=3, steps_per_sample=0)

    observer = EnergyObserver(
        ebm,
        carry_init=jnp.array(0.0),
        update_carry=lambda carry, energy, _: carry + energy,
    )

    carry_out, energies = sample_with_observation(
        key=jax.random.key(0),
        program=program,
        schedule=schedule,
        init_chain_state=[base_state],
        state_clamp=[],
        observation_carry_init=observer.init(),
        f_observe=observer,
    )

    energy_val = ebm.energy([base_state], program.gibbs_spec.blocks)

    assert energies.shape == (schedule.n_samples,)
    assert jnp.allclose(energies, jnp.full((schedule.n_samples,), energy_val))
    assert jnp.allclose(carry_out, schedule.n_samples * energy_val)
