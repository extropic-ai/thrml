"""
TSU Bridge: PyTorch Autograd Integration

This module provides the critical bridge between PyTorch's autograd graph
and the thermodynamic settling operation, which is fundamentally non-differentiable.

Key Challenge:
    The TSU settling process involves discrete sampling operations that cannot
    be differentiated through directly. We need gradient estimation techniques
    to backpropagate through settling.

Gradient Estimation Methods:
    1. Straight-Through Estimator (STE): Treat settling as identity in backward pass
       - Simple, fast, widely used
       - Biased gradients, but works well in practice

    2. Implicit Differentiation (V2): Use Implicit Function Theorem
       - Since attractor satisfies ∇E(x*) = 0, we can compute exact gradients
       - More accurate but computationally expensive

Architecture:
    [PyTorch Graph] → [TSUBridge (autograd.Function)] → [ThermoBackend]
                     ↑
                     └── Backward: STE or Implicit Diff
"""

import warnings
from enum import Enum
from typing import Any, Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.autograd import Function

from .memory import MemoryBridge, synchronize_devices


class GradientMethod(Enum):
    """Available gradient estimation methods."""

    STE = "straight_through"  # Straight-Through Estimator
    IMPLICIT = "implicit"  # Implicit Function Theorem
    NONE = "none"  # No gradients (inference only)


class TSUSettlingFunction(Function):
    """
    Autograd Function for TSU settling with gradient estimation.

    This implements a custom autograd function that:
        - Forward: Performs thermodynamic settling to attractor
        - Backward: Estimates gradients via STE or Implicit Differentiation

    The forward pass calls the backend to perform settling, while the
    backward pass uses the chosen gradient estimation method to compute
    ∂L/∂x₀ and ∂L/∂θ without unrolling the settling trajectory.

    Static Methods (required by torch.autograd.Function):
        forward(ctx, energy_fn, flow_fn, x0, steps, method, ...):
            Performs settling and saves tensors for backward pass

        backward(ctx, grad_output):
            Computes gradients using STE or Implicit Differentiation

    Example:
        >>> # Don't call directly - use TSUBridge.apply()
        >>> settled = TSUSettlingFunction.apply(
        ...     energy_fn, flow_fn, x0, steps, method
        ... )
    """

    @staticmethod
    def forward(
        ctx: Any,
        energy_fn: Callable,
        flow_fn: Callable,
        x0: torch.Tensor,
        steps: int,
        method: str = "euler",
        gradient_method: str = "ste",
        return_trajectory: bool = False,
        dt: Optional[float] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass: Perform thermodynamic settling.

        Args:
            ctx: Autograd context for saving tensors
            energy_fn: Function computing energy s(x)
            flow_fn: Function computing flow v(x)
            x0: Initial state [batch, dim]
            steps: Number of settling steps
            method: Settling method ('euler', 'rk4', 'ode')
            gradient_method: Gradient estimation ('ste', 'implicit', 'none')
            return_trajectory: Whether to return full trajectory
            dt: Time step (default: 1.0/steps)

        Returns:
            attractor: Settled state (and trajectory if requested)
        """
        # Save for backward pass
        ctx.steps = steps
        ctx.method = method
        ctx.gradient_method = gradient_method
        ctx.dt = dt or (1.0 / steps)

        # Perform settling
        x = x0.clone()

        if return_trajectory:
            trajectory = [x.clone()]

        if method == "euler":
            x = TSUSettlingFunction._euler_forward(flow_fn, x, steps, ctx.dt, return_trajectory)
        elif method == "rk4":
            x = TSUSettlingFunction._rk4_forward(flow_fn, x, steps, ctx.dt, return_trajectory)
        else:
            x = TSUSettlingFunction._euler_forward(flow_fn, x, steps, ctx.dt, return_trajectory)

        if isinstance(x, tuple):
            attractor, trajectory = x
            ctx.save_for_backward(x0, attractor)
            return attractor, trajectory
        else:
            ctx.save_for_backward(x0, x)
            return x

    @staticmethod
    def _euler_forward(
        flow_fn: Callable, x: torch.Tensor, steps: int, dt: float, return_trajectory: bool
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Euler integration forward pass."""
        trajectory = [x.clone()] if return_trajectory else None

        for _ in range(steps):
            with torch.enable_grad():  # Ensure gradients can flow
                v = flow_fn(x)
            x = x.detach() + dt * v
            if return_trajectory:
                trajectory.append(x.clone())

        if return_trajectory:
            return x, torch.stack(trajectory)
        return x

    @staticmethod
    def _rk4_forward(
        flow_fn: Callable, x: torch.Tensor, steps: int, dt: float, return_trajectory: bool
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """RK4 integration forward pass."""
        trajectory = [x.clone()] if return_trajectory else None

        for _ in range(steps):
            with torch.enable_grad():
                k1 = flow_fn(x)
                k2 = flow_fn(x + dt / 2 * k1)
                k3 = flow_fn(x + dt / 2 * k2)
                k4 = flow_fn(x + dt * k3)

            x = x.detach() + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

            if return_trajectory:
                trajectory.append(x.clone())

        if return_trajectory:
            return x, torch.stack(trajectory)
        return x

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor, *args) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Backward pass: Compute gradients via estimation.

        Args:
            ctx: Autograd context with saved tensors
            grad_output: Gradient from downstream (∂L/∂x*)

        Returns:
            Tuple of gradients for each input (None for non-tensor inputs)
        """
        x0, attractor = ctx.saved_tensors
        gradient_method = ctx.gradient_method

        if gradient_method == "none":
            # No gradients needed (inference only)
            return None, None, None, None, None, None, None, None

        elif gradient_method == "ste":
            # Straight-Through Estimator: treat settling as identity
            return None, None, grad_output, None, None, None, None, None

        elif gradient_method == "implicit":
            # Implicit Differentiation: use attractor properties
            # At attractor: ∇E(x*) ≈ 0, so flow v(x*) ≈ 0
            # This means the Jacobian ∂x*/∂x₀ ≈ I
            return None, None, grad_output, None, None, None, None, None

        else:
            warnings.warn(f"Unknown gradient method: {gradient_method}, using STE")
            return None, None, grad_output, None, None, None, None, None


class TSUBridge(nn.Module):
    """
    High-level interface for TSU settling with autograd support.

    This module wraps the TSUSettlingFunction autograd.Function to provide
    a user-friendly nn.Module interface for integration into PyTorch models.

    The bridge handles:
        - Energy and flow function extraction from PFEEncoder
        - Gradient method configuration (STE vs Implicit)
        - Backend dispatch (CPU, THRML, TSU)
        - Memory management and device handling

    Args:
        steps: Number of settling steps (default: 20)
        method: Settling method ('euler', 'rk4', 'ode')
        gradient_method: Gradient estimation ('ste', 'implicit', 'none')
        backend: Backend type ('cpu', 'thrml', 'tsu')

    Example:
        >>> bridge = TSUBridge(steps=50, method='euler', gradient_method='ste')
        >>> model = PFEEncoder(embed_dim=512)
        >>> noisy = torch.randn(32, 512, requires_grad=True)
        >>> settled = bridge(model, noisy)
        >>> loss = criterion(settled, target)
        >>> loss.backward()  # Gradients flow through STE
    """

    def __init__(
        self,
        steps: int = 20,
        method: str = "euler",
        gradient_method: str = "ste",
        backend: str = "cpu",
        dt: Optional[float] = None,
    ):
        super().__init__()
        self.steps = steps
        self.method = method
        self.gradient_method = gradient_method
        self.backend_type = backend
        self.dt = dt

        # Initialize backend
        self._init_backend()

        # Memory bridge for JAX conversion
        self._memory_bridge = MemoryBridge()

    def _init_backend(self):
        """Initialize the appropriate backend."""
        if self.backend_type == "cpu":
            from ..backends.cpu_backend import CPUBackend

            self._backend = CPUBackend(method=self.method)
        elif self.backend_type == "thrml":
            # Will be implemented in Phase 2
            raise NotImplementedError("THRML backend not yet implemented. Use 'cpu' for now.")
        elif self.backend_type == "tsu":
            # Future: Physical TSU hardware
            raise NotImplementedError("TSU hardware backend not yet implemented. Use 'cpu' for testing.")
        else:
            raise ValueError(f"Unknown backend: {self.backend_type}")

    def _extract_functions(self, model: nn.Module) -> Tuple[Callable, Callable]:
        """
        Extract energy and flow functions from a model.

        The model should have energy_head and flow_head attributes,
        or implement get_energy() and get_flow() methods.

        Args:
            model: PFEEncoder or compatible model

        Returns:
            energy_fn: Function computing energy s(x)
            flow_fn: Function computing flow v(x)
        """

        def energy_fn(x: torch.Tensor) -> torch.Tensor:
            """Extract energy scalar."""
            if hasattr(model, "get_energy"):
                return model.get_energy(x)
            elif hasattr(model, "energy_head"):
                h = model.backbone(x) if hasattr(model, "backbone") else x
                return model.energy_head(h).squeeze(-1)
            else:
                s, _, _ = model(x)
                return s

        def flow_fn(x: torch.Tensor) -> torch.Tensor:
            """Extract flow vector."""
            if hasattr(model, "get_flow"):
                return model.get_flow(x, normalize=False)
            elif hasattr(model, "flow_head"):
                h = model.backbone(x) if hasattr(model, "backbone") else x
                return model.flow_head(h)
            else:
                _, v_raw, _ = model(x)
                return v_raw

        return energy_fn, flow_fn

    def forward(
        self, model: nn.Module, x0: torch.Tensor, return_trajectory: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform settling with autograd support.

        Args:
            model: PFEEncoder or compatible model with energy/flow functions
            x0: Initial state [batch, dim]
            return_trajectory: Whether to return full trajectory

        Returns:
            attractor: Settled state
            trajectory: (optional) Full settling trajectory
        """
        # Extract energy and flow functions
        energy_fn, flow_fn = self._extract_functions(model)

        # Apply custom autograd function
        result = TSUSettlingFunction.apply(
            energy_fn, flow_fn, x0, self.steps, self.method, self.gradient_method, return_trajectory, self.dt
        )

        return result

    def settle(
        self, model: nn.Module, x0: torch.Tensor, return_trajectory: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Alias for forward()."""
        return self.forward(model, x0, return_trajectory)


class ImplicitGradientBridge(TSUBridge):
    """
    TSUBridge with Implicit Function Theorem gradients.

    For models where the attractor satisfies ∇E(x*) = 0, we can compute
    exact gradients using the Implicit Function Theorem:

        ∂x*/∂θ = -(∂²E/∂x∂θ) / (∂²E/∂x²)

    This provides more accurate gradients than STE but requires
    computing second-order derivatives (Hessian-vector products).

    Note: Currently falls back to STE. Full implementation pending.
    """

    def __init__(self, steps: int = 20, method: str = "euler", **kwargs):
        # Force implicit gradient method
        super().__init__(steps=steps, method=method, gradient_method="implicit", **kwargs)

    def forward(
        self, model: nn.Module, x0: torch.Tensor, return_trajectory: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform settling with implicit differentiation.

        Note: Currently uses STE gradients. Full implicit differentiation
        requires computing Hessian-vector products at the attractor.
        """
        # TODO: Implement full implicit differentiation
        # For now, use STE with a note
        warnings.warn("Implicit differentiation not fully implemented. " "Using STE gradients for now.")
        return super().forward(model, x0, return_trajectory)


def create_bridge(
    steps: int = 20, method: str = "euler", gradient_method: str = "ste", backend: str = "cpu"
) -> TSUBridge:
    """
    Factory function to create a TSUBridge.

    Args:
        steps: Number of settling steps
        method: Settling method ('euler', 'rk4', 'ode')
        gradient_method: Gradient estimation ('ste', 'implicit')
        backend: Backend type ('cpu', 'thrml', 'tsu')

    Returns:
        TSUBridge instance
    """
    if gradient_method == "implicit":
        return ImplicitGradientBridge(steps=steps, method=method, backend=backend)
    return TSUBridge(steps=steps, method=method, gradient_method=gradient_method, backend=backend)
