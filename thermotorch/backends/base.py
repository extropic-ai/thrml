"""
Backend Base Classes

Abstract base classes defining the interface for thermodynamic backends.
All backends must implement the settling operation.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional


class BackendType(Enum):
    """Available backend types."""

    CPU = "cpu"  # NumPy/SciPy fallback
    THRML = "thrml"  # JAX THRML simulation
    TSU = "tsu"  # Physical hardware


class BaseBackend(ABC):
    """
    Abstract base class for thermodynamic backends.

    All backends must implement:
        - settle: Perform thermodynamic settling to attractor
        - is_available: Check if backend is available
    """

    @abstractmethod
    def settle(self, energy_fn, flow_fn, x0, steps: int = 20, **kwargs):
        """
        Settle initial state to attractor.

        Args:
            energy_fn: Function computing energy s(x)
            flow_fn: Function computing flow v(x)
            x0: Initial state tensor
            steps: Number of settling steps
            **kwargs: Backend-specific parameters

        Returns:
            attractor: Settled state
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the system."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name."""
        pass


class ThermoBackend:
    """
    Unified interface for thermodynamic backends.

    Provides hardware-agnostic access to settling operations.
    Users can write code once and run on CPU, THRML, or TSU
    by changing only the backend_type parameter.

    Args:
        device: Backend type ('cpu', 'thrml', 'tsu')
        **kwargs: Backend-specific configuration

    Example:
        >>> backend = ThermoBackend(device='cpu')
        >>> attractor = backend.settle(energy_fn, flow_fn, x0)
    """

    _backends = {}

    def __init__(self, device: str = "cpu", **kwargs):
        self.device_type = BackendType(device)
        self._backend = None
        self._kwargs = kwargs
        self._initialize_backend()

    def _initialize_backend(self):
        """Initialize the appropriate backend."""
        if self.device_type == BackendType.CPU:
            from .cpu_backend import CPUBackend

            self._backend = CPUBackend(**self._kwargs)
        elif self.device_type == BackendType.THRML:
            # Future: THRML backend
            raise NotImplementedError("THRML backend not yet implemented. " "Use 'cpu' for now.")
        elif self.device_type == BackendType.TSU:
            # Future: Physical TSU hardware
            raise NotImplementedError("TSU hardware backend not yet implemented. " "Use 'cpu' for testing.")

    def settle(self, energy_fn, flow_fn, x0, steps: int = 20, **kwargs):
        """
        Settle initial state to attractor.

        Delegates to the underlying backend implementation.
        """
        return self._backend.settle(energy_fn, flow_fn, x0, steps, **kwargs)

    def is_available(self) -> bool:
        """Check if current backend is available."""
        return self._backend.is_available()

    @property
    def name(self) -> str:
        """Get backend name."""
        return self._backend.name

    @classmethod
    def register_backend(cls, name: str, backend_class):
        """Register a new backend type."""
        cls._backends[name] = backend_class
