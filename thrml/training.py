"""
thrml/training.py — Generic Contrastive Divergence training API for EBMs.

Public API:
    contrastive_divergence_loss(model, positive_samples, negative_samples, blocks)
        → scalar loss L = E(neg) - E(pos), differentiable via eqx.filter_value_and_grad.

    EBMTrainingSpec
        Static training configuration (programs + schedules + block specs).
        Build once before the training loop; pass every step.

    train_step(model, opt_state, optimizer, spec, pos_init, neg_init, data, key)
        → (updated_model, updated_opt_state, loss_scalar)
        One complete CD step: MCMC positive + negative phases, grad update.

Usage hint:
    spec = EBMTrainingSpec(...); model, opt_state, loss = train_step(model, opt_state, optimizer, spec, ...)
"""

import dataclasses
from typing import TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from thrml.block_management import Block
from thrml.block_sampling import (
    BlockSamplingProgram,
    SamplingSchedule,
)
from thrml.models.ebm import AbstractEBM
from thrml.observers import StateObserver

Model = TypeVar("Model", bound=AbstractEBM)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


def contrastive_divergence_loss(
    model: AbstractEBM,
    positive_samples: list[Array],
    negative_samples: list[Array],
    blocks: list[Block],
) -> Float[Array, ""]:
    """Contrastive divergence loss for an arbitrary EBM.

    L(θ) = mean_neg(E_θ(x_neg)) − mean_pos(E_θ(x_pos))

    Minimising this with gradient descent pushes energy down on positive
    (data) samples and up on negative (model) samples — i.e., standard CD.

    **Arguments:**

    - ``model``: Any ``AbstractEBM`` implemented as an equinox Module.
    - ``positive_samples``: Batched positive-phase states.  Each element is
      a per-block array of shape ``[batch, *block_shape]``.
    - ``negative_samples``: Batched negative-phase states.  Same layout.
    - ``blocks``: Block specification that matches both sample lists.

    **Returns:**

    A scalar float — differentiable w.r.t. all array leaves of *model*.

    **Sign convention (Risk R2 / Q1 from S2):**
    ``L = E(neg) − E(pos)``.  With standard gradient descent (``θ ← θ − lr·∇L``)
    this lowers ``E(pos)`` and raises ``E(neg)``, which is exactly what CD
    training should do.  No sign inversion is needed in the optimizer.
    """
    # vmap energy over batch dimension (axis 0 of each per-block array)
    # positive_samples[i]: shape [batch, *block_shape]
    # We unstack along axis 0 to get per-sample lists and vmap.
    def _energy_single(state: list[Array]) -> Float[Array, ""]:
        return model.energy(state, blocks)

    # vmap expects a function (list[Array],) → scalar where each array has
    # the *block_shape without the batch prefix.  We transpose the list-of-
    # batched-arrays into a batched-list-of-arrays via vmap over axis 0.
    energy_pos = jax.vmap(_energy_single)(positive_samples)
    energy_neg = jax.vmap(_energy_single)(negative_samples)

    return jnp.mean(energy_neg) - jnp.mean(energy_pos)


# ---------------------------------------------------------------------------
# Training spec
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class EBMTrainingSpec:
    """Static training configuration for a generic EBM training loop.

    Build this **once** before the training loop and pass the same instance to
    every ``train_step`` call.  It holds only static configuration (sampling
    programs, schedules, block specs) — NOT the model parameters.  This is the
    key design difference from ``IsingTrainingSpec``, which embeds the model
    and must be rebuilt at each step.

    **Attributes:**

    - ``program_positive``: ``BlockSamplingProgram`` for the positive phase
      (data-clamped free nodes).
    - ``schedule_positive``: ``SamplingSchedule`` (warm-up + sample steps) for
      the positive phase.
    - ``program_negative``: ``BlockSamplingProgram`` for the negative phase
      (all nodes free).
    - ``schedule_negative``: ``SamplingSchedule`` for the negative phase.
    - ``visible_blocks``: Blocks clamped to observed data in the positive phase.
    - ``free_blocks``: Blocks sampled freely in both phases.
    - ``all_blocks``: ``visible_blocks + free_blocks`` — passed to ``energy()``.
    """

    program_positive: BlockSamplingProgram
    schedule_positive: SamplingSchedule
    program_negative: BlockSamplingProgram
    schedule_negative: SamplingSchedule
    visible_blocks: list[Block]
    free_blocks: list[Block]
    all_blocks: list[Block]


# ---------------------------------------------------------------------------
# train_step
# ---------------------------------------------------------------------------


def train_step(
    model: Model,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    spec: EBMTrainingSpec,
    positive_samples: list[Array],
    negative_samples: list[Array],
    data: list[Array],
    key: PRNGKeyArray,
    filter_spec: PyTree = eqx.is_inexact_array,
) -> tuple[Model, optax.OptState, Float[Array, ""]]:
    """One complete contrastive divergence training step.

    Executes:
      1. ``contrastive_divergence_loss`` on pre-sampled positive + negative states.
      2. ``eqx.filter_value_and_grad`` to obtain model-parameter gradients.
      3. ``optimizer.update`` + ``eqx.apply_updates`` to produce the updated model.

    **Sampling is the caller's responsibility.** Use ``sample_states`` with
    ``spec.program_positive`` / ``spec.program_negative`` to generate
    ``positive_samples`` and ``negative_samples`` before calling this function.
    This separation keeps ``train_step`` a pure differentiable update step and
    avoids threading MCMC state through the gradient computation.

    **Arguments:**

    - ``model``: Current model (equinox Module / ``AbstractEBM`` subclass).
    - ``opt_state``: Current optax optimizer state.
    - ``optimizer``: An optax ``GradientTransformation`` (e.g. ``optax.adam(1e-3)``).
    - ``spec``: ``EBMTrainingSpec`` — build once, pass every step.
    - ``positive_samples``: Batched positive-phase samples. Each element has
      shape ``[batch, *block_shape]``. Typically from ``hinton_init`` or
      ``sample_states`` with data clamped.
    - ``negative_samples``: Batched negative-phase samples. Same layout.
    - ``data``: Unused in this signature (reserved for future clamped-phase
      extensions). Pass ``[]``.
    - ``key``: JAX PRNGKey (unused here; reserved for future stochastic
      regularisation). Pass a fresh key for forward compatibility.
    - ``filter_spec``: Filter passed to ``eqx.filter_value_and_grad``.  Defaults
      to ``eqx.is_inexact_array`` (differentiates only float arrays, leaving
      integer/bool leaves — e.g. discrete topology arrays — untouched).

    **Returns:**

    A 3-tuple ``(updated_model, updated_opt_state, loss_scalar)``:
    - ``updated_model``: New model with updated parameters.
    - ``updated_opt_state``: New optax state.
    - ``loss_scalar``: CD loss value (for logging/diagnostics).
    """
    # --- CD loss + gradients ---
    def loss_fn(m: Model) -> Float[Array, ""]:
        return contrastive_divergence_loss(
            m, positive_samples, negative_samples, spec.free_blocks
        )

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)

    # --- Optax update ---
    # jax_numpy_dtype_promotion="standard" needed because optax.adam does
    # float ** int32 internally which fails under strict JAX dtype promotion.
    with jax.numpy_dtype_promotion("standard"):
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, filter_spec)
        )
    new_model = eqx.apply_updates(model, updates)

    return new_model, new_opt_state, loss
