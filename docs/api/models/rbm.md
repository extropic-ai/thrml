# Restricted Boltzmann Machine (RBM)

The RBM module provides a complete implementation of Restricted Boltzmann Machines, including energy-based model definitions, sampling programs, and training utilities.

## Overview

A Restricted Boltzmann Machine is a bipartite energy-based model with:
- **Visible layer**: Observed data units
- **Hidden layer**: Latent feature units
- **Bipartite structure**: Connections only between layers, not within layers

The energy function is:

$$E(v, h) = -\beta \left( \sum_i a_i v_i + \sum_j b_j h_j + \sum_{i,j} W_{ij} v_i h_j \right)$$

where $v_i$ are visible units, $h_j$ are hidden units, $a_i, b_j$ are biases, $W_{ij}$ are connection weights, and $\beta$ is the inverse temperature.

## Core Classes

::: thrml.models.RBMEBM
    options:
      show_root_heading: true
      show_source: false

::: thrml.models.RBMSamplingProgram
    options:
      show_root_heading: true
      show_source: false

::: thrml.models.RBMTrainingSpec
    options:
      show_root_heading: true
      show_source: false

## Training Functions

::: thrml.models.estimate_rbm_grad
    options:
      show_root_heading: true
      show_source: false

::: thrml.models.estimate_rbm_moments
    options:
      show_root_heading: true
      show_source: false

## Initialization

::: thrml.models.rbm_init
    options:
      show_root_heading: true
      show_source: false

## Usage Example

```python
import jax
from jax import numpy as jnp
from thrml import SpinNode, Block, SamplingSchedule
from thrml.models import RBMEBM, RBMSamplingProgram, rbm_init

# Create nodes
visible_nodes = [SpinNode() for _ in range(784)]  # MNIST-sized
hidden_nodes = [SpinNode() for _ in range(128)]

# Initialize parameters
key = jax.random.key(0)
visible_biases = jnp.zeros((784,))
hidden_biases = jnp.zeros((128,))
weights = jax.random.normal(key, (784, 128)) * 0.01
beta = jnp.array(1.0)

# Create RBM
rbm = RBMEBM(
    visible_nodes=visible_nodes,
    hidden_nodes=hidden_nodes,
    visible_biases=visible_biases,
    hidden_biases=hidden_biases,
    weights=weights,
    beta=beta
)

# Sample from the model
program = RBMSamplingProgram(
    ebm=rbm,
    free_blocks=[Block(visible_nodes), Block(hidden_nodes)],
    clamped_blocks=[]
)

schedule = SamplingSchedule(n_warmup=100, n_samples=1000, steps_per_sample=2)
init_state = rbm_init(key, rbm, [Block(visible_nodes), Block(hidden_nodes)], ())

from thrml.block_sampling import sample_states
samples = sample_states(
    key=key,
    program=program,
    schedule=schedule,
    init_state_free=init_state,
    state_clamp=[],
    nodes_to_sample=[Block(visible_nodes), Block(hidden_nodes)]
)
```

## Training with Contrastive Divergence

```python
from thrml.models import RBMTrainingSpec, estimate_rbm_grad

# Create training specification
training_spec = RBMTrainingSpec(
    ebm=rbm,
    schedule_positive=SamplingSchedule(n_warmup=10, n_samples=50, steps_per_sample=1),
    schedule_negative=SamplingSchedule(n_warmup=10, n_samples=50, steps_per_sample=1)
)

# Prepare training data
batch_size = 64
visible_data = [jax.random.bernoulli(key, 0.5, shape=(batch_size, 784)).astype(jnp.bool_)]

# Initialize states
n_chains = 4
init_hidden_pos = rbm_init(key, rbm, [Block(hidden_nodes)], (n_chains, batch_size))
init_neg = rbm_init(key, rbm, [Block(visible_nodes), Block(hidden_nodes)], (n_chains,))

# Compute gradients
grad_weights, grad_visible_bias, grad_hidden_bias = estimate_rbm_grad(
    key=key,
    training_spec=training_spec,
    visible_data=visible_data,
    init_state_positive=init_hidden_pos,
    init_state_negative=init_neg
)

# Update parameters
learning_rate = 0.01
new_weights = rbm.weights - learning_rate * grad_weights
new_visible_biases = rbm.visible_biases - learning_rate * grad_visible_bias
new_hidden_biases = rbm.hidden_biases - learning_rate * grad_hidden_bias
```

## Gradient Update Rules

The contrastive divergence algorithm estimates gradients as:

$$\Delta W_{ij} = -\beta (\langle v_i h_j \rangle_{data} - \langle v_i h_j \rangle_{model})$$

$$\Delta a_i = -\beta (\langle v_i \rangle_{data} - \langle v_i \rangle_{model})$$

$$\Delta b_j = -\beta (\langle h_j \rangle_{data} - \langle h_j \rangle_{model})$$

where $\langle \cdot \rangle_{data}$ are expectations under the data distribution (visible units clamped) and $\langle \cdot \rangle_{model}$ are expectations under the model distribution (free sampling).

## Applications

RBMs are useful for:

- **Dimensionality reduction**: Learn compact representations of high-dimensional data
- **Feature learning**: Automatically discover useful features in unsupervised manner
- **Collaborative filtering**: Recommendation systems (e.g., Netflix prize)
- **Generative modeling**: Sample synthetic data from learned distribution
- **Deep belief networks**: Stack multiple RBMs for deep architectures
- **Pre-training**: Initialize deep neural networks

## Performance

RBMs in THRML are:
- **JIT-compiled**: Fast execution via JAX
- **GPU-accelerated**: Automatic GPU dispatch when available
- **Vectorized**: Efficient batch processing with vmap
- **Memory-efficient**: Uses block-based sampling

For a 784x128 RBM on CPU:
- Gradient estimation: ~6-10 seconds
- Sampling (100 samples): ~1 second

On GPU (Tesla T4), expect 10-50x speedups depending on batch size.

## See Also

- [IsingEBM](ising.md): Related spin-based model
- [Block Sampling](../block_sampling.md): Underlying sampling algorithm
- [Observers](../observers.md): Collecting statistics during sampling
