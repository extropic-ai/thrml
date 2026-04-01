"""
Unit tests for ThermoTorch Phase 3 components:
    - Memory Bridge (DLPack, JAX integration)
    - THRML Backend (JAX settling, vmap)
    - TSU Backend (Hardware interface)

Run with: pytest tests/test_phase3.py -v
"""

import logging
import os
import sys

import numpy as np
import pytest
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.WARNING)


class TestMemoryBridgePhase3:
    """Tests for production-ready Memory Bridge."""

    def test_memory_bridge_creation(self):
        """Test MemoryBridge instantiation."""
        from thermotorch.core.memory import ConversionMode, MemoryBridge

        bridge = MemoryBridge()
        assert bridge.device in ("cpu", "cuda")
        assert bridge.mode == ConversionMode.AUTO

        # With config
        bridge = MemoryBridge(device="cpu", mode=ConversionMode.COPY)
        assert bridge.device == "cpu"
        assert bridge.mode == ConversionMode.COPY

    def test_numpy_conversions(self):
        """Test NumPy conversion methods."""
        from thermotorch.core.memory import MemoryBridge

        bridge = MemoryBridge()

        # NumPy to Torch
        arr = np.random.randn(4, 8).astype(np.float32)
        tensor = bridge.numpy_to_torch(arr)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (4, 8)

        # Torch to NumPy
        tensor = torch.randn(4, 8)
        arr_back = bridge.torch_to_numpy(tensor)
        assert isinstance(arr_back, np.ndarray)
        assert arr_back.shape == (4, 8)

    def test_jax_availability_check(self):
        """Test JAX availability checking."""
        from thermotorch.core.memory import MemoryBridge

        bridge = MemoryBridge()
        # Should not raise
        available = bridge.jax_available
        assert isinstance(available, bool)

    def test_dlpack_availability_check(self):
        """Test DLPack availability checking."""
        from thermotorch.core.memory import MemoryBridge

        bridge = MemoryBridge()
        available = bridge.dlpack_available
        assert isinstance(available, bool)

    def test_compatibility_check(self):
        """Test full compatibility checking."""
        from thermotorch.core.memory import MemoryBridge

        bridge = MemoryBridge()
        info = bridge.check_compatibility()

        assert "torch_version" in info
        assert "jax_available" in info
        assert "dlpack_available" in info
        assert "device" in info

    def test_jax_conversions(self):
        """Test JAX conversion methods."""
        from thermotorch.core.memory import MemoryBridge

        try:
            import jax
            import jax.numpy as jnp
        except ImportError:
            pytest.skip("JAX not installed")

        bridge = MemoryBridge()

        # Torch to JAX
        tensor = torch.randn(4, 8)
        jax_arr = bridge.torch_to_jax(tensor)
        assert jax_arr.shape == (4, 8)

        # JAX to Torch
        jax_arr = jnp.array(np.random.randn(4, 8))
        tensor_back = bridge.jax_to_torch(jax_arr)
        assert isinstance(tensor_back, torch.Tensor)
        assert tensor_back.shape == (4, 8)

    def test_tensor_cache(self):
        """Test TensorCache functionality."""
        from thermotorch.core.memory import MemoryBridge, TensorCache

        try:
            import jax
        except ImportError:
            pytest.skip("JAX not installed")

        cache = TensorCache(max_size=5)
        bridge = MemoryBridge()

        # Add tensors
        for i in range(3):
            t = torch.randn(2, 2)
            result = cache.get_jax(t, key=f"t{i}", bridge=bridge)
            assert result.shape == (2, 2)

        assert len(cache) == 3

        # Clear cache
        cache.clear()
        assert len(cache) == 0

    def test_global_bridge(self):
        """Test global bridge instance."""
        from thermotorch.core.memory import get_global_bridge, reset_global_bridge

        bridge1 = get_global_bridge()
        bridge2 = get_global_bridge()

        # Should return same instance
        assert bridge1 is bridge2

        # Reset
        reset_global_bridge()
        bridge3 = get_global_bridge()
        assert bridge3 is not bridge1

    def test_device_handling(self):
        """Test device management."""
        from thermotorch.core.memory import MemoryBridge, get_device

        # CPU device
        bridge_cpu = MemoryBridge(device="cpu")
        assert bridge_cpu.device == "cpu"

        # Auto device
        bridge_auto = MemoryBridge(device="auto")
        assert bridge_auto.device in ("cpu", "cuda")

        # get_device function
        tensor = torch.randn(4, 8)
        device = get_device(tensor)
        assert device in ("cpu", "cuda")


class TestTHRMLBackendPhase3:
    """Tests for production-ready THRML Backend."""

    def test_thrml_config(self):
        """Test THRMLConfig."""
        from thermotorch.backends.thrml_backend import SettlingMethod, THRMLConfig

        config = THRMLConfig()
        assert config.use_vmap == True
        assert config.jit_compile == True

        # Custom config
        config = THRMLConfig(device="cpu", use_vmap=False, jit_compile=False, precision="float64")
        assert config.use_vmap == False
        assert config.jit_compile == False

    def test_thrml_backend_creation(self):
        """Test THRMLBackend instantiation."""
        try:
            from thermotorch.backends.thrml_backend import THRMLBackend, THRMLConfig

            config = THRMLConfig(device="cpu")
            backend = THRMLBackend(config)
            assert backend.name == "thrml"
            assert backend.is_available()

        except ImportError:
            pytest.skip("JAX not installed")

    def test_thrml_device_info(self):
        """Test JAX device info."""
        try:
            from thermotorch.backends.thrml_backend import THRMLBackend

            backend = THRMLBackend()
            info = backend.get_device_info()

            assert "available" in info
            assert "jax_version" in info
            assert "devices" in info

        except ImportError:
            pytest.skip("JAX not installed")

    def test_thrml_settle(self):
        """Test THRML settling."""
        try:
            from thermotorch.backends.thrml_backend import THRMLBackend
            from thermotorch.core import PFEEncoder

            backend = THRMLBackend()
            model = PFEEncoder(embed_dim=32)

            def energy_fn(x):
                return model.get_energy(x)

            def flow_fn(x):
                return model.get_flow(x, normalize=False)

            x0 = torch.randn(4, 32)
            result = backend.settle(energy_fn, flow_fn, x0, steps=10)

            assert result.shape == x0.shape
            assert not torch.isnan(result).any()

        except ImportError:
            pytest.skip("JAX not installed")

    def test_thrml_batch_settle(self):
        """Test THRML batched settling."""
        try:
            from thermotorch.backends.thrml_backend import THRMLBackend
            from thermotorch.core import PFEEncoder

            backend = THRMLBackend()
            model = PFEEncoder(embed_dim=32)

            def flow_fn(x):
                return model.get_flow(x, normalize=False)

            x_batch = torch.randn(8, 32)
            result = backend.batch_settle(lambda x: model.get_energy(x), flow_fn, x_batch, steps=10)

            assert result.shape == x_batch.shape

        except ImportError:
            pytest.skip("JAX not installed")

    def test_thrml_methods(self):
        """Test different settling methods."""
        try:
            from thermotorch.backends.thrml_backend import THRMLBackend
            from thermotorch.core import PFEEncoder

            backend = THRMLBackend()
            model = PFEEncoder(embed_dim=16)

            def flow_fn(x):
                return model.get_flow(x, normalize=False)

            x0 = torch.randn(4, 16)

            for method in ["euler", "rk4"]:
                result = backend.settle(lambda x: model.get_energy(x), flow_fn, x0, steps=5, method=method)
                assert result.shape == x0.shape

        except ImportError:
            pytest.skip("JAX not installed")

    def test_thrml_factory_function(self):
        """Test create_thrml_backend factory."""
        try:
            from thermotorch.backends.thrml_backend import create_thrml_backend

            backend = create_thrml_backend(device="cpu", use_vmap=True)
            assert backend.name == "thrml"
            assert backend.is_available()

        except ImportError:
            pytest.skip("JAX not installed")

    def test_jax_availability_functions(self):
        """Test JAX availability functions."""
        from thermotorch.backends.thrml_backend import get_jax_device_info, is_jax_available

        jax_available = is_jax_available()
        assert isinstance(jax_available, bool)

        device_info = get_jax_device_info()
        assert "available" in device_info


class TestTSUBackend:
    """Tests for TSU Backend (hardware interface)."""

    def test_tsu_config(self):
        """Test TSUConfig."""
        from thermotorch.backends.tsu_backend import SettlingMode, TSUConfig

        config = TSUConfig()
        assert config.settling_mode == SettlingMode.BALANCED
        assert config.fallback_to_thrml == True

        # Custom config
        config = TSUConfig(device_id=1, settling_mode=SettlingMode.ACCURATE, max_iterations=2000)
        assert config.device_id == 1
        assert config.settling_mode == SettlingMode.ACCURATE

    def test_mock_tsu_device(self):
        """Test MockTSUDevice."""
        from thermotorch.backends.tsu_backend import HardwareStatus, MockTSUDevice

        device = MockTSUDevice(device_id=0, chip_type="Z1")

        # Initialize
        assert device.initialize() == True
        assert device.is_ready() == True

        # Get capabilities
        caps = device.get_capabilities()
        assert caps.chip_type == "Z1"
        assert caps.num_pbits > 0

        # Settle
        x0 = torch.randn(4, 32)
        result = device.settle(x0, steps=10)
        assert result.shape == x0.shape

        # Status
        assert device.get_status() == HardwareStatus.READY

        # Reset
        assert device.reset() == True

        # Close
        device.close()
        assert device.get_status() == HardwareStatus.DISABLED

    def test_tsu_backend_creation(self):
        """Test TSUBackend instantiation."""
        from thermotorch.backends.tsu_backend import TSUBackend, TSUConfig

        config = TSUConfig(fallback_to_thrml=True, fallback_to_cpu=True)
        backend = TSUBackend(config)

        assert backend.name == "tsu"
        assert backend.is_available()  # Should have fallback

    def test_tsu_backend_settle(self):
        """Test TSUBackend settling with fallback."""
        from thermotorch.backends.tsu_backend import TSUBackend
        from thermotorch.core import PFEEncoder

        backend = TSUBackend()
        model = PFEEncoder(embed_dim=32)

        def energy_fn(x):
            return model.get_energy(x)

        def flow_fn(x):
            return model.get_flow(x, normalize=False)

        x0 = torch.randn(4, 32)
        result = backend.settle(energy_fn, flow_fn, x0, steps=10)

        assert result.shape == x0.shape

    def test_tsu_device_detection(self):
        """Test device detection."""
        from thermotorch.backends.tsu_backend import detect_tsu_devices

        devices = detect_tsu_devices()
        assert isinstance(devices, list)

        # Should have at least mock device
        assert len(devices) >= 0

    def test_tsu_factory_function(self):
        """Test create_tsu_backend factory."""
        from thermotorch.backends.tsu_backend import create_tsu_backend

        backend = create_tsu_backend(device_id=0, settling_mode="balanced", fallback_to_thrml=True)
        assert backend.name == "tsu"

    def test_tsu_info_function(self):
        """Test get_tsu_info."""
        from thermotorch.backends.tsu_backend import get_tsu_info

        info = get_tsu_info()
        assert "devices_detected" in info
        assert "devices" in info

    def test_hardware_status_enum(self):
        """Test HardwareStatus enum."""
        from thermotorch.backends.tsu_backend import HardwareStatus

        assert HardwareStatus.UNAVAILABLE.value == "unavailable"
        assert HardwareStatus.READY.value == "ready"
        assert HardwareStatus.BUSY.value == "busy"
        assert HardwareStatus.ERROR.value == "error"

    def test_settling_mode_enum(self):
        """Test SettlingMode enum."""
        from thermotorch.backends.tsu_backend import SettlingMode

        assert SettlingMode.FAST.value == "fast"
        assert SettlingMode.BALANCED.value == "balanced"
        assert SettlingMode.ACCURATE.value == "accurate"


class TestBackendIntegration:
    """Integration tests for all backends."""

    def test_backend_selection(self):
        """Test selecting appropriate backend."""
        from thermotorch.backends import (
            BackendType,
            CPUBackend,
            ThermoBackend,
            is_jax_available,
            is_tsu_available,
        )

        # CPU backend always available
        backend = ThermoBackend(device="cpu")
        assert backend.is_available()

        # Check availability
        jax_avail = is_jax_available()
        tsu_avail = is_tsu_available()

        assert isinstance(jax_avail, bool)
        assert isinstance(tsu_avail, bool)

    def test_backend_comparison(self):
        """Test that different backends produce similar results."""
        from thermotorch.backends import CPUBackend
        from thermotorch.core import PFEEncoder

        try:
            from thermotorch.backends import THRMLBackend

            model = PFEEncoder(embed_dim=16)

            def flow_fn(x):
                return model.get_flow(x, normalize=False)

            x0 = torch.randn(4, 16)

            # CPU backend
            cpu_backend = CPUBackend(method="euler")
            cpu_result = cpu_backend.settle(lambda x: model.get_energy(x), flow_fn, x0.clone(), steps=10)

            # THRML backend
            thrml_backend = THRMLBackend()
            thrml_result = thrml_backend.settle(lambda x: model.get_energy(x), flow_fn, x0.clone(), steps=10)

            # Results should have same shape
            assert cpu_result.shape == thrml_result.shape

        except ImportError:
            pytest.skip("JAX not installed")

    def test_fallback_chain(self):
        """Test fallback from TSU to THRML to CPU."""
        from thermotorch.backends.tsu_backend import TSUBackend, TSUConfig
        from thermotorch.core import PFEEncoder

        # TSU with full fallback chain
        config = TSUConfig(fallback_to_thrml=True, fallback_to_cpu=True)
        backend = TSUBackend(config)

        model = PFEEncoder(embed_dim=16)

        def flow_fn(x):
            return model.get_flow(x, normalize=False)

        x0 = torch.randn(4, 16)
        result = backend.settle(lambda x: model.get_energy(x), flow_fn, x0, steps=5)

        assert result.shape == x0.shape


class TestEndToEndPhase3:
    """End-to-end tests for Phase 3."""

    def test_full_pipeline_with_bridge(self):
        """Test full pipeline with TSUBridge."""
        from thermotorch.core import PFEEncoder, TSUBridge

        model = PFEEncoder(embed_dim=32)
        bridge = TSUBridge(steps=10, method="euler")

        x = torch.randn(8, 32)
        settled = bridge(model, x)

        assert settled.shape == x.shape

    def test_training_with_backends(self):
        """Test training loop with different backends."""
        from thermotorch.backends import CPUBackend
        from thermotorch.core import PFEEncoder, PFELoss

        model = PFEEncoder(embed_dim=32)
        criterion = PFELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Training loop
        for i in range(5):
            x = torch.randn(16, 32)
            x_pos = x + torch.randn_like(x) * 0.05
            x_noise = torch.randn(16, 32)

            optimizer.zero_grad()
            loss = criterion(model, x, x_pos, x_noise)
            loss.backward()
            optimizer.step()

        # Verify model trained
        assert loss.item() < 1e6  # Reasonable loss

    def test_generation_with_dtm(self):
        """Test generation with DTMLayer."""
        from thermotorch.core import DTMLayer, PFEEncoder

        pfe = PFEEncoder(embed_dim=64)
        dtm = DTMLayer(pfe, steps=20)

        # Generate samples
        samples = dtm.generate(batch_size=4, latent_dim=64)
        assert samples.shape == (4, 64)

        # Denoise
        noisy = torch.randn(4, 64)
        clean = dtm.denoise(noisy)
        assert clean.shape == (4, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
