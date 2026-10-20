from typing import TYPE_CHECKING, Callable

import equinox as eqx
import jax
from jaxtyping import Array, Key, PyTree

from thrml.block_sampling import (
    BlockSamplingProgram,
    SamplingSchedule,
    _sample_with_observation_jax,
    _State,
    _validate_sampling_state,
)
from thrml.factor import FactorSamplingProgram
from thrml.models.discrete_ebm import SpinGibbsConditional
from thrml.observers import AbstractObserver, ObserveCarry

if TYPE_CHECKING:
    from thrml.models.ising import IsingSamplingProgram


# Equinox forwards JIT options, but its type overloads omit them.
@eqx.filter_jit(inline=True)  # pyright: ignore[reportCallIssue]
def _compile_ising(program: "IsingSamplingProgram") -> BlockSamplingProgram:
    """Reuse structural lowering while keeping model parameters dynamic."""
    spec = program.gibbs_spec
    sampler = SpinGibbsConditional()
    return FactorSamplingProgram(spec, [sampler for _ in spec.free_blocks], program.model.factors, [])


class JaxBackend:
    """Stateless backend for THRML's existing JAX block-sampling algorithm.

    This backend supports JAX transformations and requires explicit initial
    states. It does not retain chains, compiled programs, or model parameters.
    """

    def compile(self, program: "BlockSamplingProgram | IsingSamplingProgram") -> BlockSamplingProgram:
        """Lower a logical Ising program to the executable JAX representation.

        Existing `BlockSamplingProgram` and `FactorSamplingProgram` instances
        are returned unchanged. A private JIT helper reuses structural lowering
        for the same graph, block layout, and parameter shapes/dtypes. Current
        model factors, including beta, are evaluated on every call; numerical
        targets are not cached. The first call for a new structure compiles the
        conversion. Inside an enclosing JIT, this helper is inlined.

        The result is an executable program, not a JIT-compiled sampling function.
        Use it with `sample_blocks` or `sample_single_block`, or to inspect
        `.samplers` and `.per_block_*` fields. Compile again after updating the model.
        """
        from thrml.models.ising import IsingSamplingProgram

        if isinstance(program, BlockSamplingProgram):
            return program
        if not isinstance(program, IsingSamplingProgram):
            raise TypeError("JaxBackend does not support this program type.")

        return _compile_ising(program)

    def run(
        self,
        key: Key[Array, ""],
        program: "BlockSamplingProgram | IsingSamplingProgram",
        schedule: SamplingSchedule,
        init_state_free: list[_State] | None,
        state_clamp: list[_State],
        observation_carry_init: ObserveCarry,
        f_observe: AbstractObserver,
    ) -> tuple[ObserveCarry, PyTree]:
        """Lower and run one explicitly initialized JAX chain.

        Observers receive the compiled program, preserving JAX-specific access
        to samplers and interactions. The first record follows warmup; later
        records follow `steps_per_sample` further sweeps.
        """
        schedule.validate()
        if init_state_free is None:
            raise ValueError("JaxBackend requires an explicit initial state. Chain continuation is unsupported.")
        _validate_sampling_state(program, init_state_free, state_clamp)
        return _sample_with_observation_jax(
            key,
            self.compile(program),
            schedule,
            init_state_free,
            state_clamp,
            observation_carry_init,
            f_observe,
        )

    def map_chains(self, fn: Callable[..., PyTree], *batched_args: PyTree) -> PyTree:
        """Map request arguments over their leading axis with `jax.vmap`."""
        return jax.vmap(fn)(*batched_args)
