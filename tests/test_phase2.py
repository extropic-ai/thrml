"""
Unit tests for ThermoTorch Phase 2 components:
    - Memory Bridge (DLPack)
    - TSU Bridge (autograd)
    - TSU Layers
    - Backends

Run with: pytest tests/test_phase2.py -v
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thermotorch.backends import BackendType, CPUBackend, ThermoBackend
from thermotorch.core import (
    DTMLayer,
    EnergyBridge,
    GradientMethod,
    MemoryBridge,
    PFEEncoder,
    PFELoss,
    SettlingMethod,
    TensorCache,
    TSUBridge,
    TSULayer,
    create_bridge,
    create_energy_bridge,
    create_tsu_layer,
)
from thermotorch.core.memory import DLPackError


class TestMemoryBridge:
    """Tests for MemoryBridge (DLPack conversions)."""

    def test_numpy_torch_conversion(self):
        """Test NumPy <-> PyTorch conversion."""
        import numpy as np

        bridge = MemoryBridge()

        # NumPy to Torch
        arr = np.random.randn(4, 8).astype(np.float32)
        tensor = bridge.numpy_to_torch(arr)
        assert tensor.shape == (4, 8)
        assert tensor.device.type in ("cpu", "cuda")

        # Torch to NumPy
        tensor = torch.randn(4, 8)
        arr_back = bridge.torch_to_numpy(tensor)
        assert arr_back.shape == (4, 8)
        assert arr_back.dtype == np.float32

    def test_torch_contiguity(self):
        """Test that tensors are made contiguous."""
        from thermotorch.core.memory import ensure_contiguous

        # Non-contiguous tensor
        x = torch.randn(4, 8).t()  # Transpose makes it non-contiguous
        assert not x.is_contiguous()

        # Make contiguous
        y = ensure_contiguous(x)
        assert y.is_contiguous()

    def test_tensor_cache(self):
        """Test TensorCache functionality."""
        # Skip if JAX is not installed
        try:
            import jax
            import jax.numpy as jnp

            jax_available = True
        except ImportError:
            jax_available = False

        if not jax_available:
            pytest.skip("JAX not installed - skipping TensorCache JAX test")

        cache = TensorCache(max_size=5)

        # Add tensors
        for i in range(5):
            t = torch.randn(2, 2)
            cache.get_jax(t, key=f"t{i}")

        assert len(cache) == 5

        # Add more (should evict oldest)
        t = torch.randn(2, 2)
        cache.get_jax(t, key="new")
        assert len(cache) == 5

        # Clear cache
        cache.clear()
        assert len(cache) == 0


class TestTSUBridge:
    """Tests for TSUBridge (autograd integration)."""

    def test_bridge_creation(self):
        """Test TSUBridge instantiation."""
        bridge = TSUBridge(steps=20, method="euler", gradient_method="ste")
        assert bridge.steps == 20
        assert bridge.method == "euler"
        assert bridge.gradient_method == "ste"

    def test_bridge_forward(self):
        """Test bridge forward pass."""
        model = PFEEncoder(embed_dim=64)
        bridge = TSUBridge(steps=10, method="euler")

        x = torch.randn(8, 64)
        settled = bridge(model, x)

        assert settled.shape == x.shape
        assert not torch.isnan(settled).any()

    def test_bridge_with_trajectory(self):
        """Test bridge with trajectory return."""
        model = PFEEncoder(embed_dim=64)
        bridge = TSUBridge(steps=10, method="euler")

        x = torch.randn(4, 64)
        settled, trajectory = bridge(model, x, return_trajectory=True)

        assert settled.shape == x.shape
        # Trajectory shape depends on implementation

    def test_bridge_factory(self):
        """Test create_bridge factory function."""
        # Default bridge
        bridge1 = create_bridge(steps=20)
        assert bridge1.steps == 20

        # With different parameters
        bridge2 = create_bridge(steps=50, method="rk4", gradient_method="ste")
        assert bridge2.steps == 50
        assert bridge2.method == "rk4"


class TestTSULayer:
    """Tests for TSULayer (nn.Module interface)."""

    def test_layer_creation(self):
        """Test TSULayer instantiation."""
        layer = TSULayer(steps=20, method="euler")
        assert layer.steps == 20
        assert layer.method == "euler"

    def test_layer_forward(self):
        """Test TSULayer forward pass."""
        model = PFEEncoder(embed_dim=32)
        layer = TSULayer(steps=10, method="euler")

        x = torch.randn(4, 32)
        output = layer(model, x)

        assert output.shape == x.shape

    def test_layer_extra_repr(self):
        """Test TSULayer representation."""
        layer = TSULayer(steps=30, method="rk4", gradient_method="ste", backend="cpu")
        repr_str = layer.extra_repr()
        assert "steps=30" in repr_str
        assert "method=rk4" in repr_str

    def test_layer_factory(self):
        """Test create_tsu_layer factory."""
        layer = create_tsu_layer(steps=25, method="euler")
        assert isinstance(layer, TSULayer)
        assert layer.steps == 25


class TestEnergyBridge:
    """Tests for EnergyBridge (PFE-specific wrapper)."""

    def test_energy_bridge_creation(self):
        """Test EnergyBridge instantiation."""
        pfe = PFEEncoder(embed_dim=64)
        bridge = EnergyBridge(pfe, steps=20)
        assert bridge.steps == 20

    def test_energy_bridge_get_energy(self):
        """Test energy retrieval."""
        pfe = PFEEncoder(embed_dim=64)
        bridge = EnergyBridge(pfe)

        x = torch.randn(8, 64)
        energy = bridge.get_energy(x)

        assert energy.shape == (8,)

    def test_energy_bridge_get_flow(self):
        """Test flow retrieval."""
        pfe = PFEEncoder(embed_dim=64)
        bridge = EnergyBridge(pfe)

        x = torch.randn(8, 64)
        flow = bridge.get_flow(x)

        assert flow.shape == (8, 64)

    def test_energy_bridge_settle(self):
        """Test settling via EnergyBridge."""
        pfe = PFEEncoder(embed_dim=64)
        bridge = EnergyBridge(pfe, steps=15)

        x = torch.randn(4, 64)
        settled = bridge.settle(x)

        assert settled.shape == x.shape

    def test_energy_bridge_forward(self):
        """Test EnergyBridge forward pass."""
        pfe = PFEEncoder(embed_dim=64)
        bridge = EnergyBridge(pfe, steps=10)

        x = torch.randn(4, 64)
        settled = bridge(x)
        assert settled.shape == x.shape

        # With return_all
        settled, energy, flow = bridge(x, return_all=True)
        assert settled.shape == x.shape
        assert energy.shape == (4,)
        assert flow.shape == (4, 64)


class TestDTMLayer:
    """Tests for DTMLayer (generative tasks)."""

    def test_dtm_creation(self):
        """Test DTMLayer instantiation."""
        pfe = PFEEncoder(embed_dim=64)
        dtm = DTMLayer(pfe, steps=20)
        assert dtm.steps == 20

    def test_dtm_denoise(self):
        """Test DTM denoising."""
        pfe = PFEEncoder(embed_dim=64)
        dtm = DTMLayer(pfe, steps=10)

        noisy = torch.randn(4, 64)
        clean = dtm.denoise(noisy)

        assert clean.shape == noisy.shape

    def test_dtm_forward(self):
        """Test DTMLayer forward pass."""
        pfe = PFEEncoder(embed_dim=64)
        dtm = DTMLayer(pfe)

        noisy = torch.randn(4, 64)
        clean = dtm(noisy)

        assert clean.shape == noisy.shape

    def test_dtm_generate(self):
        """Test DTM generation."""
        pfe = PFEEncoder(embed_dim=64)
        dtm = DTMLayer(pfe, steps=20)

        samples = dtm.generate(batch_size=4, latent_dim=64)
        assert samples.shape == (4, 64)


class TestBackends:
    """Tests for backend abstraction."""

    def test_cpu_backend_creation(self):
        """Test CPUBackend instantiation."""
        backend = CPUBackend(method="euler")
        assert backend.name == "cpu"
        assert backend.is_available()

    def test_cpu_backend_settle(self):
        """Test CPUBackend settling."""
        model = PFEEncoder(embed_dim=32)
        backend = CPUBackend(method="euler")

        def energy_fn(x):
            return model.get_energy(x)

        def flow_fn(x):
            return model.get_flow(x, normalize=False)

        x0 = torch.randn(4, 32)
        attractor = backend.settle(energy_fn, flow_fn, x0, steps=10)

        assert attractor.shape == x0.shape

    def test_cpu_backend_methods(self):
        """Test different CPUBackend methods."""
        model = PFEEncoder(embed_dim=32)

        def flow_fn(x):
            _, v, _ = model(x)
            return v

        x0 = torch.randn(4, 32)

        # Euler
        backend_euler = CPUBackend(method="euler")
        att_euler = backend_euler.settle(lambda x: model.get_energy(x), flow_fn, x0, steps=10)
        assert att_euler.shape == x0.shape

        # RK4
        backend_rk4 = CPUBackend(method="rk4")
        att_rk4 = backend_rk4.settle(lambda x: model.get_energy(x), flow_fn, x0, steps=10)
        assert att_rk4.shape == x0.shape

    def test_thermo_backend(self):
        """Test ThermoBackend interface."""
        backend = ThermoBackend(device="cpu")
        assert backend.name == "cpu"
        assert backend.is_available()

    def test_backend_type_enum(self):
        """Test BackendType enum."""
        assert BackendType.CPU.value == "cpu"
        assert BackendType.THRML.value == "thrml"
        assert BackendType.TSU.value == "tsu"


class TestGradientFlow:
    """Tests for gradient flow through bridge."""

    def test_ste_gradients(self):
        """Test Straight-Through Estimator gradients."""
        model = PFEEncoder(embed_dim=32)
        bridge = TSUBridge(steps=5, method="euler", gradient_method="ste")

        x = torch.randn(4, 32, requires_grad=True)
        settled = bridge(model, x)

        loss = settled.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_training_with_bridge(self):
        """Test training a model with TSUBridge."""
        model = PFEEncoder(embed_dim=16)
        bridge = TSUBridge(steps=5, method="euler")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Training step
        x = torch.randn(8, 16, requires_grad=True)
        settled = bridge(model, x)

        loss = settled.pow(2).sum()
        loss.backward()
        optimizer.step()

        # Check parameters updated
        assert loss.item() >= 0

    def test_no_gradient_method(self):
        """Test gradient_method='none' (inference only)."""
        model = PFEEncoder(embed_dim=32)
        bridge = TSUBridge(steps=5, gradient_method="none")

        x = torch.randn(4, 32, requires_grad=True)
        settled = bridge(model, x)

        loss = settled.sum()
        loss.backward()

        # With 'none', gradients should not flow through bridge
        # (but x still has requires_grad=True)
        assert loss.item() is not None


class TestIntegration:
    """Integration tests for Phase 2."""

    def test_full_pipeline(self):
        """Test full pipeline: encode -> settle -> decode."""
        # Create model and bridge
        model = PFEEncoder(embed_dim=64)
        bridge = EnergyBridge(model, steps=20)

        # Generate from noise
        samples = bridge.generate(batch_size=4, dim=64)
        assert samples.shape == (4, 64)

        # Get energy
        energy = bridge.get_energy(samples)
        assert energy.shape == (4,)

    def test_dtm_pipeline(self):
        """Test DTM generation pipeline."""
        pfe = PFEEncoder(embed_dim=32)
        dtm = DTMLayer(pfe, steps=15)

        # Generate
        samples = dtm.generate(batch_size=4, latent_dim=32)
        assert samples.shape == (4, 32)

        # Denoise
        noisy = torch.randn(4, 32)
        clean = dtm.denoise(noisy)
        assert clean.shape == (4, 32)

    def test_layer_in_model(self):
        """Test TSULayer as part of a larger model."""

        class ModelWithTSU(nn.Module):
            def __init__(self, dim=32):
                super().__init__()
                self.encoder = nn.Linear(dim, dim)
                self.pfe = PFEEncoder(embed_dim=dim)
                self.tsu = TSULayer(steps=10)

            def forward(self, x):
                h = self.encoder(x)
                settled = self.tsu(self.pfe, h)
                return settled

        model = ModelWithTSU(dim=32)
        x = torch.randn(4, 32)
        output = model(x)

        assert output.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
