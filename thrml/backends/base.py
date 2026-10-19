from typing import TYPE_CHECKING, Callable, Protocol

from jaxtyping import Array, Key, PyTree

from thrml.observers import AbstractObserver, ObserveCarry

if TYPE_CHECKING:
    from thrml.block_sampling import BlockSamplingProgram, SamplingSchedule, _State
    from thrml.models.ising import IsingSamplingProgram


class SamplingBackend(Protocol):
    """Execute sampling requests and choose how requests are mapped over chains.

    Backends own execution, randomness, and retained-chain continuation. External
    backends may run requests serially on the host. Physical initialization,
    calibration, device control, and execution reports belong to the external
    package, not to the logical program or sampling return values.
    """

    def run(
        self,
        key: Key[Array, ""],
        program: "BlockSamplingProgram | IsingSamplingProgram",
        schedule: "SamplingSchedule",
        init_state_free: "list[_State] | None",
        state_clamp: "list[_State]",
        observation_carry_init: ObserveCarry,
        f_observe: AbstractObserver,
    ) -> tuple[ObserveCarry, PyTree]:
        """Return `(final_observer_carry, stacked_observations)` for one request.

        `None` requests continuation of a backend-owned chain; `[]` is an
        explicit empty free state. Clamps follow the program's block order.
        Observers receive every free and clamped block. Observation leaves have
        one leading record axis of length `schedule.n_samples`; an observer that
        emits `None` produces observations of `None`. A backend may reject
        program types or schedules that it cannot execute.
        """
        ...

    def map_chains(self, fn: Callable[..., PyTree], *batched_args: PyTree) -> PyTree:
        """Map corresponding leading-axis entries and stack results in order.

        Accept one or more argument PyTrees with matching leading sizes across
        mapped leaves. Preserve argument and result structures and input order;
        do not reduce or average results. Constants belong in `fn`'s closure.
        Empty PyTrees are valid when another argument supplies the map size;
        training always supplies keys. Nested mapping must also be supported.

        Implementations may vectorize, serialize, or use native batching.
        Continuation across requests belongs to the backend, and mapped requests
        need not represent statistically independent physical chains.
        """
        ...
