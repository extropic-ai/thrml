from .base import SamplingBackend as SamplingBackend
from .jax import JaxBackend as JaxBackend


def _resolve_backend(backend: SamplingBackend | None) -> SamplingBackend:
    return JaxBackend() if backend is None else backend
