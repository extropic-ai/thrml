"""
Parallel tempering utilities built on THRML's block Gibbs samplers.

This module orchestrates multiple tempered chains (one per beta/model), runs
blocked Gibbs steps on each, and proposes swaps between adjacent temperatures.
It does not alter core sampling code; it simply composes existing
`BlockSamplingProgram`s.
"""

from typing import Sequence

import jax
import jax.numpy as jnp
from jax import lax

from thrml.block_sampling import BlockSamplingProgram, sample_blocks
from thrml.conditional_samplers import AbstractConditionalSampler
from thrml.models.ebm import AbstractEBM


def _init_sampler_states(program: BlockSamplingProgram):
    """Initialize sampler state list for a BlockSamplingProgram."""
    return jax.tree.map(
        lambda x: x.init(),
        program.samplers,
        is_leaf=lambda a: isinstance(a, AbstractConditionalSampler),
    )


def _gibbs_steps(
    key,
    program: BlockSamplingProgram,
    state_free: list,
    state_clamp: list,
    sampler_state: list,
    n_iters: int,
) -> tuple[list, list]:
    """Run n_iters block-Gibbs sweeps for a single chain."""
    if n_iters == 0:
        return state_free, sampler_state

    keys = jax.random.split(key, n_iters)
    for k in keys:
        state_free, sampler_state = sample_blocks(k, state_free, state_clamp, program, sampler_state)
    return state_free, sampler_state


def _attempt_swap_pair(
    key,
    ebm_i: AbstractEBM,
    ebm_j: AbstractEBM,
    program_i: BlockSamplingProgram,
    program_j: BlockSamplingProgram,
    state_i: list,
    state_j: list,
    clamp_state: list,
):
    """
    Propose a swap between two adjacent temperature chains (i, j).

    Acceptance ratio uses energies of both states under both models:
    log r = (E_i(x_i) + E_j(x_j)) - (E_i(x_j) + E_j(x_i))
    """
    blocks_i = program_i.gibbs_spec.blocks
    blocks_j = program_j.gibbs_spec.blocks

    Ei_xi = ebm_i.energy(state_i + clamp_state, blocks_i)
    Ej_xj = ebm_j.energy(state_j + clamp_state, blocks_j)
    Ei_xj = ebm_i.energy(state_j + clamp_state, blocks_i)
    Ej_xi = ebm_j.energy(state_i + clamp_state, blocks_j)

    log_r = (Ei_xi + Ej_xj) - (Ei_xj + Ej_xi)
    accept_prob = jnp.exp(jnp.minimum(0.0, log_r))
    u = jax.random.uniform(key)

    def do_swap(states):
        s_i, s_j = states
        # swap and mark accepted
        return s_j, s_i, jnp.int32(1)

    def no_swap(states):
        s_i, s_j = states
        # keep as-is and mark rejected
        return s_i, s_j, jnp.int32(0)

    return lax.cond(u < accept_prob, do_swap, no_swap, (state_i, state_j))


def _swap_pass(
    key,
    ebms: Sequence[AbstractEBM],
    programs: Sequence[BlockSamplingProgram],
    states: list[list],
    sampler_states: list[list],
    clamp_state: list,
    pair_indices: Sequence[int],
):
    """Perform one swap pass over a fixed set of adjacent pairs."""
    n_pairs = len(ebms) - 1
    accept_counts = [0] * n_pairs
    attempt_counts = [0] * n_pairs

    if len(pair_indices) == 0:
        return states, sampler_states, accept_counts, attempt_counts

    keys = jax.random.split(key, len(pair_indices))
    new_states = list(states)
    new_sampler_states = list(sampler_states)

    for idx, pair in enumerate(pair_indices):
        i, j = pair, pair + 1
        attempt_counts[pair] = 1
        new_i, new_j, accepted = _attempt_swap_pair(
            keys[idx], ebms[i], ebms[j], programs[i], programs[j], new_states[i], new_states[j], clamp_state
        )
        # states already come swapped or not from _attempt_swap_pair
        new_states[i], new_states[j] = new_i, new_j
        # sampler states follow the same swap pattern
        new_sampler_states[i], new_sampler_states[j] = new_sampler_states[j], new_sampler_states[i]
        accept_counts[pair] = accepted

    return new_states, new_sampler_states, accept_counts, attempt_counts


def parallel_tempering(
    key,
    ebms: Sequence[AbstractEBM],
    programs: Sequence[BlockSamplingProgram],
    init_states: Sequence[list],
    clamp_state: list,
    n_rounds: int,
    gibbs_steps_per_round: int,
    sampler_states: Sequence[list] | None = None,
):
    """Run parallel tempering across a sequence of tempered chains.

    Each round performs block Gibbs updates in every chain, then proposes
    swaps between adjacent temperatures. All chains share the same
    block structure and clamped state layout.
    """
    if not (len(ebms) == len(programs) == len(init_states)):
        raise ValueError("ebms, programs, and init_states must have the same length.")
    if sampler_states is not None and len(sampler_states) != len(programs):
        raise ValueError("sampler_states must match length of programs if provided.")

    base_spec = programs[0].gibbs_spec
    base_free = len(base_spec.free_blocks)
    base_clamped = len(base_spec.clamped_blocks)
    for prog in programs[1:]:
        if len(prog.gibbs_spec.free_blocks) != base_free or len(prog.gibbs_spec.clamped_blocks) != base_clamped:
            raise ValueError("All programs must share the same block structure (free + clamped blocks).")

    clamp_state = clamp_state or []
    states = [list(s) for s in init_states]
    sampler_states = (
        [list(s) for s in sampler_states] if sampler_states is not None else [_init_sampler_states(p) for p in programs]
    )

    n_pairs = max(len(ebms) - 1, 0)
    accepted = jnp.zeros((n_pairs,), dtype=jnp.int32)
    attempted = jnp.zeros((n_pairs,), dtype=jnp.int32)

    # Precompute adjacent pair indices for even and odd rounds.
    even_pair_indices = list(range(0, n_pairs, 2))
    odd_pair_indices = list(range(1, n_pairs, 2))

    def one_round(carry, round_idx):
        key, states, sampler_states, accepted, attempted = carry

        # Keys for this round
        key, key_round = jax.random.split(key)
        keys = jax.random.split(key_round, len(ebms) + 1)
        swap_key = keys[-1]

        # Gibbs updates per chain (number of chains is static)
        for i in range(len(ebms)):
            states[i], sampler_states[i] = _gibbs_steps(
                keys[i],
                programs[i],
                states[i],
                clamp_state,
                sampler_states[i],
                gibbs_steps_per_round,
            )

        def do_even(args):
            states, sampler_states, accepted, attempted, swap_key = args
            states, sampler_states, acc_counts, att_counts = _swap_pass(
                swap_key,
                ebms,
                programs,
                states,
                sampler_states,
                clamp_state,
                even_pair_indices,
            )
            accepted = accepted + jnp.array(acc_counts, dtype=jnp.int32)
            attempted = attempted + jnp.array(att_counts, dtype=jnp.int32)
            return states, sampler_states, accepted, attempted

        def do_odd(args):
            states, sampler_states, accepted, attempted, swap_key = args
            states, sampler_states, acc_counts, att_counts = _swap_pass(
                swap_key,
                ebms,
                programs,
                states,
                sampler_states,
                clamp_state,
                odd_pair_indices,
            )
            accepted = accepted + jnp.array(acc_counts, dtype=jnp.int32)
            attempted = attempted + jnp.array(att_counts, dtype=jnp.int32)
            return states, sampler_states, accepted, attempted

        parity = round_idx & 1
        states, sampler_states, accepted, attempted = lax.cond(
            parity == 0,
            do_even,
            do_odd,
            (states, sampler_states, accepted, attempted, swap_key),
        )

        new_carry = (key, states, sampler_states, accepted, attempted)
        return new_carry, None

    if n_rounds > 0:
        init_carry = (key, states, sampler_states, accepted, attempted)
        final_carry, _ = lax.scan(one_round, init_carry, jnp.arange(n_rounds))
        key, states, sampler_states, accepted, attempted = final_carry

    acceptance_rate = jnp.where(attempted > 0, accepted / attempted, 0.0)
    stats = {
        "accepted": accepted,
        "attempted": attempted,
        "acceptance_rate": acceptance_rate,
    }

    return states, sampler_states, stats
