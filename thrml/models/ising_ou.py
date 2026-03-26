import jax
from jax import numpy as jnp

from thrml.models.discrete_ebm import SpinGibbsConditional


class SpinOUGibbsConditional(SpinGibbsConditional):
    theta: float
    mu: float
    sigma: float
    n_nodes: int

    def __init__(self, n_nodes, theta=0.5, mu=0.0, sigma=1.0):
        self.n_nodes = n_nodes
        self.theta = theta
        self.mu = mu
        self.sigma = sigma

    def init(self):
        return jnp.zeros(self.n_nodes)

    def sample(self, key, interactions, active_flags, states, sampler_state, output_sd):
        key_ou, key_sample = jax.random.split(key)

        gamma, _ = self.compute_parameters(key, interactions, active_flags, states, None, output_sd)

        eps = jax.random.normal(key_ou, shape=sampler_state.shape, dtype=gamma.dtype)
        noise = sampler_state + self.theta * (self.mu - sampler_state) + self.sigma * eps

        noisy_gamma = gamma + noise
        new_state = jax.random.bernoulli(key_sample, jax.nn.sigmoid(2 * noisy_gamma))

        return new_state, noise
