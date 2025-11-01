<div align="center">
  <img src="_static/logo/logo.svg" alt="THRML Logo" width="200" style="margin-bottom: 20px;">
</div>

# **Thermodynamic HypergRaphical Model Library (THRML)**

---

`THRML` is a JAX library for building and sampling probabilistic graphical models, with a focus on efficient block Gibbs sampling and energy-based models. Extropic is developing hardware to make sampling from certain classes of discrete PGMs massively more energy‑efficient; `THRML` provides GPU‑accelerated tools for block sampling on sparse, heterogeneous graphs, making it a natural place to prototype today and experiment with future Extropic hardware.

Features include:

- Block Gibbs sampling for PGMs
- Arbitrary PyTree node states
- Support for heterogeneous graphical models
- Discrete EBM utilities (Ising/RBM‑like)
- Enables early experimentation with future Extropic hardware

## Installation

Requires >=python 3.10

```bash
pip install thrml
```

or

```bash
uv pip install thrml
```

For installing from the source:

```bash
git clone https://github.com/extropic-ai/thrml
cd thrml
pip install -e .
```

or

```bash
git clone https://github.com/extropic-ai/thrml
cd thrml
uv pip install -e .
```

## Quick example

Sampling a small Ising chain with two‑color block Gibbs:

```python
import jax
import jax.numpy as jnp
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

nodes = [SpinNode() for _ in range(5)]
edges = [(nodes[i], nodes[i+1]) for i in range(4)]
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
```
