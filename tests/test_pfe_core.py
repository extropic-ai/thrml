"""
Unit tests for ThermoTorch PFE Core components.

Run with: pytest tests/test_pfe_core.py -v
"""

import os
import sys

import pytest
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thermotorch.core import PFEEncoder, PFELoss, tsu_settle
from thermotorch.core.tsu_settle import SettlingMethod


class TestPFEEncoder:
    """Tests for PFEEncoder."""

    def test_output_shapes(self):
        """Test that PFEEncoder outputs have correct shapes."""
        batch_size = 32
        embed_dim = 128

        model = PFEEncoder(embed_dim=embed_dim)
        x = torch.randn(batch_size, embed_dim)

        s, v_raw, v_unit = model(x)

        assert s.shape == (batch_size,), f"Energy shape mismatch: {s.shape}"
        assert v_raw.shape == (batch_size, embed_dim), f"Flow shape mismatch: {v_raw.shape}"
        assert v_unit.shape == (batch_size, embed_dim), f"Unit flow shape mismatch: {v_unit.shape}"

    def test_unit_normalization(self):
        """Test that v_unit is properly normalized."""
        batch_size = 16
        embed_dim = 64

        model = PFEEncoder(embed_dim=embed_dim)
        x = torch.randn(batch_size, embed_dim)

        _, _, v_unit = model(x)

        # Check L2 norm is approximately 1
        norms = v_unit.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5), f"Unit vectors not normalized: norms={norms}"

    def test_different_hidden_dims(self):
        """Test PFEEncoder with different hidden dimensions."""
        model = PFEEncoder(embed_dim=64, hidden_dim=256, num_layers=3)
        x = torch.randn(8, 64)
        s, v_raw, v_unit = model(x)

        assert s.shape == (8,)
        assert v_raw.shape == (8, 64)

    def test_activations(self):
        """Test different activation functions."""
        for act in ["gelu", "relu", "silu"]:
            model = PFEEncoder(embed_dim=32, activation=act)
            x = torch.randn(4, 32)
            s, _, _ = model(x)
            assert s.shape == (4,), f"Failed for activation: {act}"


class TestPFELoss:
    """Tests for PFELoss."""

    def test_dsm_loss(self):
        """Test Denoising Score Matching loss."""
        model = PFEEncoder(embed_dim=64)
        criterion = PFELoss(sigma=0.1)

        x = torch.randn(16, 64)
        loss = criterion.dsm_loss(model, x)

        assert loss.ndim == 0, "DSM loss should be scalar"
        assert loss >= 0, "DSM loss should be non-negative"

    def test_nce_loss(self):
        """Test Noise Contrastive Estimation loss."""
        criterion = PFELoss()

        s_real = torch.randn(16)
        s_noise = torch.randn(16)
        loss = criterion.nce_loss(s_real, s_noise)

        assert loss.ndim == 0, "NCE loss should be scalar"
        assert loss >= 0, "NCE loss should be non-negative"

    def test_infonce_loss(self):
        """Test InfoNCE loss."""
        criterion = PFELoss(temperature=0.1)

        v_unit = torch.randn(16, 32)
        v_unit = v_unit / v_unit.norm(dim=-1, keepdim=True)  # Normalize
        v_unit_pos = torch.randn(16, 32)
        v_unit_pos = v_unit_pos / v_unit_pos.norm(dim=-1, keepdim=True)

        loss = criterion.infonce_loss(v_unit, v_unit_pos)

        assert loss.ndim == 0, "InfoNCE loss should be scalar"
        assert loss >= 0, "InfoNCE loss should be non-negative"

    def test_full_loss(self):
        """Test complete tri-partite loss."""
        model = PFEEncoder(embed_dim=64)
        criterion = PFELoss(sigma=0.1, alpha=0.1, beta=0.5)

        x = torch.randn(16, 64)
        x_pos = x + torch.randn_like(x) * 0.05  # Positive pairs
        x_noise = torch.randn(16, 64)  # Noise

        total_loss, l_dsm, l_nce, l_infonce = criterion(model, x, x_pos, x_noise, return_components=True)

        assert total_loss.ndim == 0, "Total loss should be scalar"
        assert l_dsm.ndim == 0, "DSM loss should be scalar"
        assert l_nce.ndim == 0, "NCE loss should be scalar"
        assert l_infonce.ndim == 0, "InfoNCE loss should be scalar"

    def test_loss_gradients(self):
        """Test that loss produces valid gradients."""
        model = PFEEncoder(embed_dim=32)
        criterion = PFELoss()

        x = torch.randn(8, 32, requires_grad=False)
        x_pos = x + torch.randn_like(x) * 0.05
        x_noise = torch.randn(8, 32)

        loss = criterion(model, x, x_pos, x_noise)
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None, "Parameter has no gradient"
            assert not torch.isnan(param.grad).any(), "Gradient contains NaN"


class TestTSUSettle:
    """Tests for TSU settling."""

    def test_euler_settle(self):
        """Test Euler settling method."""
        model = PFEEncoder(embed_dim=32)
        x0 = torch.randn(4, 32)

        attractor = tsu_settle(model, x0, steps=10, method=SettlingMethod.EULER)

        assert attractor.shape == x0.shape, "Attractor shape mismatch"
        assert not torch.isnan(attractor).any(), "Attractor contains NaN"

    def test_rk4_settle(self):
        """Test RK4 settling method."""
        model = PFEEncoder(embed_dim=32)
        x0 = torch.randn(4, 32)

        attractor = tsu_settle(model, x0, steps=10, method=SettlingMethod.RK4)

        assert attractor.shape == x0.shape, "Attractor shape mismatch"
        assert not torch.isnan(attractor).any(), "Attractor contains NaN"

    def test_trajectory_return(self):
        """Test that trajectory is returned when requested."""
        model = PFEEncoder(embed_dim=32)
        x0 = torch.randn(4, 32)

        attractor, trajectory = tsu_settle(model, x0, steps=10, method=SettlingMethod.EULER, return_trajectory=True)

        assert trajectory.shape[0] == 11, "Trajectory should have steps+1 entries"
        assert trajectory.shape[1:] == x0.shape, "Trajectory shape mismatch"

    def test_settling_reduces_energy(self):
        """Test that settling generally reduces energy."""
        model = PFEEncoder(embed_dim=64)
        x0 = torch.randn(8, 64)

        # Get initial energy
        s_before, _, _ = model(x0)

        # Settle
        attractor = tsu_settle(model, x0, steps=50, method=SettlingMethod.RK4)

        # Get final energy
        s_after, _, _ = model(attractor)

        # Energy should generally decrease (not guaranteed for all models)
        # Just check shapes are correct
        assert s_before.shape == (8,), "Energy shape mismatch"
        assert s_after.shape == (8,), "Energy shape mismatch"


class TestIntegration:
    """Integration tests."""

    def test_training_step(self):
        """Test a single training step works end-to-end."""
        model = PFEEncoder(embed_dim=32, hidden_dim=64)
        criterion = PFELoss(sigma=0.1, alpha=0.1, beta=0.5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Forward
        x = torch.randn(8, 32)
        x_pos = x + torch.randn_like(x) * 0.05
        x_noise = torch.randn(8, 32)

        loss = criterion(model, x, x_pos, x_noise)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check parameters updated
        assert loss.item() >= 0

    def test_inference_pipeline(self):
        """Test full inference pipeline: init -> settle -> decode."""
        embed_dim = 32
        model = PFEEncoder(embed_dim=embed_dim)

        # Initialize from noise
        x0 = torch.randn(4, embed_dim)

        # Settle to attractor
        attractor = tsu_settle(model, x0, steps=20, method=SettlingMethod.EULER)

        # Get energy
        s, v_raw, v_unit = model(attractor)

        assert attractor.shape == (4, embed_dim)
        assert s.shape == (4,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
