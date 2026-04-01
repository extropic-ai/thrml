"""
CPU Backend

Pure Python/NumPy fallback backend for testing and debugging.
Uses Probability Flow ODE for settling.

This backend is:
- Slow: Not optimized for production
- Portable: Works on any system with PyTorch
- Useful: For CI/CD testing and debugging autograd graphs
"""

from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn

from .base import BackendType, BaseBackend


class CPUBackend(BaseBackend):
    """
    CPU-based settling backend using PyTorch ODE integration.

    This backend uses standard PyTorch operations for settling,
    making it portable but slower than GPU-accelerated backends.

    Use for:
        - Development and debugging
        - CI/CD testing
        - Verifying correctness before hardware deployment

    For production, use:
        - thrml_backend: JAX GPU simulation
        - tsu_backend: Physical Extropic hardware
    """

    def __init__(self, method: str = "euler", device: Optional[str] = None):
        """
        Initialize CPU backend.

        Args:
            method: Integration method ('euler', 'rk4', 'ode')
            device: Torch device ('cpu', 'cuda', etc.)
        """
        self.method = method
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def name(self) -> str:
        return "cpu"

    def is_available(self) -> bool:
        """CPU backend is always available."""
        return True

    def settle(
        self,
        energy_fn: Callable,
        flow_fn: Callable,
        x0: torch.Tensor,
        steps: int = 20,
        return_trajectory: bool = False,
        dt: Optional[float] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Settle using CPU-based ODE integration.

        Args:
            energy_fn: Function computing energy s(x) (not used in current impl)
            flow_fn: Function computing flow v(x)
            x0: Initial state [batch, dim]
            steps: Number of integration steps
            return_trajectory: Whether to return full trajectory
            dt: Time step (default: 1.0/steps)

        Returns:
            attractor: Settled state
            trajectory: (optional) Full trajectory
        """
        dt = dt or (1.0 / steps)
        x = x0.clone().to(self.device)

        if return_trajectory:
            trajectory = [x.clone()]

        if self.method == "euler":
            result = self._euler_settle(flow_fn, x, steps, dt, return_trajectory)
        elif self.method == "rk4":
            result = self._rk4_settle(flow_fn, x, steps, dt, return_trajectory)
        elif self.method == "ode":
            result = self._ode_settle(flow_fn, x, steps, return_trajectory, **kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return result

    def _euler_settle(
        self, flow_fn: Callable, x: torch.Tensor, steps: int, dt: float, return_trajectory: bool
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Euler integration."""
        if return_trajectory:
            trajectory = [x.clone()]

        for _ in range(steps):
            v = flow_fn(x)
            x = x + dt * v
            if return_trajectory:
                trajectory.append(x.clone())

        if return_trajectory:
            return x, torch.stack(trajectory)
        return x

    def _rk4_settle(
        self, flow_fn: Callable, x: torch.Tensor, steps: int, dt: float, return_trajectory: bool
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """4th-order Runge-Kutta integration."""
        if return_trajectory:
            trajectory = [x.clone()]

        for _ in range(steps):
            k1 = flow_fn(x)
            k2 = flow_fn(x + dt / 2 * k1)
            k3 = flow_fn(x + dt / 2 * k2)
            k4 = flow_fn(x + dt * k3)

            x = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

            if return_trajectory:
                trajectory.append(x.clone())

        if return_trajectory:
            return x, torch.stack(trajectory)
        return x

    def _ode_settle(
        self, flow_fn: Callable, x: torch.Tensor, steps: int, return_trajectory: bool, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Use torchdiffeq ODE solver."""
        try:
            from torchdiffeq import odeint
        except ImportError:
            raise ImportError("torchdiffeq required for 'ode' method. " "Install with: pip install torchdiffeq")

        def ode_func(t, state):
            return flow_fn(state)

        t = torch.linspace(0, 1, steps, device=x.device, dtype=x.dtype)
        trajectory = odeint(ode_func, x, t, **kwargs)

        if return_trajectory:
            return trajectory[-1], trajectory
        return trajectory[-1]
