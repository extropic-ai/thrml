"""
Potential-Flow Embedding (PFE) Loss Functions

Tri-partite loss that trains the PFE landscape without requiring
a partition function:

    L_total = L_DSM + α * L_NCE + β * L_InfoNCE

Components:
    - DSM Loss: Trains flow vectors to point toward clean data
    - NCE Loss: Forces real data into energy valleys, noise onto hills
    - InfoNCE Loss: Forces related concepts to flow in same direction

Mathematical Foundation:
    - Denoising Score Matching (Song et al. 2020)
    - Noise Contrastive Estimation (Gutmann & Hyvärinen 2010)
    - von Mises-Fisher (vMF) similarity on unit sphere
"""

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PFELoss(nn.Module):
    """
    Tri-partite loss for training PFE landscapes.

    This loss trains the energy landscape and flow field simultaneously
    without requiring explicit computation of the partition function.

    Loss Components:
        1. DSM Loss: Flow vector should match score function
           L_DSM = E[||v(x̃) - ∇log p(x̃|x)||²]
           Forces the flow to point toward high-density regions.

        2. NCE Loss: Real data should have lower energy than noise
           L_NCE = E[log(1 + exp(s_noise - s_real))]
           Forces attractors (valleys) at data, repellers (hills) at noise.

        3. InfoNCE Loss: Similar concepts should have aligned flows
           L_InfoNCE = -log(exp(sim(v_i, v_i+)) / Σ_j exp(sim(v_i, v_j)))
           Forces flow alignment on the unit sphere (vMF).

    Args:
        sigma: Noise scale for DSM (default: 0.1)
        alpha: Weight for NCE loss (default: 0.1)
        beta: Weight for InfoNCE loss (default: 0.5)
        temperature: Temperature for InfoNCE similarity (default: 0.07)

    Example:
        >>> criterion = PFELoss(sigma=0.1, alpha=0.1, beta=0.5)
        >>> model = PFEEncoder(embed_dim=512)
        >>> x = torch.randn(32, 512)  # real data
        >>> x_pos = torch.randn(32, 512)  # positive pairs
        >>> x_noise = torch.randn(32, 512)  # noise samples
        >>> loss = criterion(model, x, x_pos, x_noise)
    """

    def __init__(self, sigma: float = 0.1, alpha: float = 0.1, beta: float = 0.5, temperature: float = 0.07):
        super().__init__()
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def dsm_loss(self, model: nn.Module, x: torch.Tensor, noise_scale: Optional[float] = None) -> torch.Tensor:
        """
        Denoising Score Matching Loss.

        Forces the flow vectors to point exactly toward the clean data
        by matching the score function ∇_x log p(x).

        The score function for Gaussian noise is:
            ∇_x̃ log p(x̃|x) = -(x̃ - x) / σ² = -ε / σ

        where x̃ = x + σε and ε ~ N(0, 1).

        Args:
            model: PFEEncoder model with flow_head
            x: Clean data samples [batch, dim]
            noise_scale: Override sigma (default: use self.sigma)

        Returns:
            loss: DSM loss scalar
        """
        sigma = noise_scale or self.sigma

        # Add noise to create corrupted samples
        noise = torch.randn_like(x)
        x_noisy = x + sigma * noise

        # Get flow prediction from noisy samples
        _, v_raw, _ = model(x_noisy)

        # Target score: ∇_x̃ log p(x̃|x) = -ε / σ
        # For DSM, flow should predict the negative noise direction
        target_score = -noise / (sigma**2)

        # MSE loss between predicted flow and target score
        loss = (v_raw - target_score).pow(2).mean()

        return loss

    def nce_loss(self, s_real: torch.Tensor, s_noise: torch.Tensor) -> torch.Tensor:
        """
        Noise Contrastive Estimation Loss.

        Forces true data to have lower energy (deeper valleys) than noise.
        Uses logistic NCE formulation:

            L = E[log(1 + exp(s_noise - s_real))]

        This encourages s(x_real) << s(x_noise), meaning real data
        sits in energy minima while noise sits on hills.

        Args:
            s_real: Energy scalar for real data [batch]
            s_noise: Energy scalar for noise samples [batch]

        Returns:
            loss: NCE loss scalar
        """
        # Logistic NCE: log(1 + exp(s_noise - s_real))
        logits = s_noise - s_real
        loss = torch.log(1 + torch.exp(-logits)).mean()

        return loss

    def infonce_loss(self, v_unit: torch.Tensor, v_unit_pos: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE Loss with von Mises-Fisher (vMF) similarity.

        Forces related concepts (positive pairs) to have aligned flow
        directions on the unit sphere. Uses cosine similarity which
        corresponds to vMF distribution on S^(d-1).

        InfoNCE formulation:
            L = -log(exp(sim(v_i, v_i+)/τ) / Σ_j exp(sim(v_i, v_j)/τ))

        Args:
            v_unit: Unit flow vectors for anchors [batch, dim]
            v_unit_pos: Unit flow vectors for positives [batch, dim]

        Returns:
            loss: InfoNCE loss scalar
        """
        batch_size = v_unit.size(0)

        # Compute similarity matrix (cosine similarity on unit sphere)
        # sim[i, j] = <v_i, v_j> / τ
        sim_matrix = (v_unit @ v_unit_pos.T) / self.temperature

        # Labels: each sample's positive is at the same index
        labels = torch.arange(batch_size, device=v_unit.device)

        # Cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)

        return loss

    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        x_pos: torch.Tensor,
        x_noise: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        Compute the complete tri-partite PFE loss.

        Args:
            model: PFEEncoder model
            x: Clean data samples [batch, dim]
            x_pos: Positive pair samples (augmented/related) [batch, dim]
            x_noise: Noise samples [batch, dim]
            return_components: If True, return individual losses

        Returns:
            loss: Total loss or tuple of (total, dsm, nce, infonce)
        """
        # 1. DSM Loss: Train the compass (flow vectors)
        l_dsm = self.dsm_loss(model, x)

        # 2. NCE Loss: Train the altimeter (energy scalar)
        s_real, _, v_unit = model(x)
        s_noise, _, _ = model(x_noise)
        l_nce = self.nce_loss(s_real, s_noise)

        # 3. InfoNCE Loss: Train flow alignment
        _, _, v_unit_pos = model(x_pos)
        l_infonce = self.infonce_loss(v_unit, v_unit_pos)

        # Combined loss
        total_loss = l_dsm + (self.alpha * l_nce) + (self.beta * l_infonce)

        if return_components:
            return total_loss, l_dsm, l_nce, l_infonce

        return total_loss


class DSMOnlyLoss(nn.Module):
    """
    Pure Denoising Score Matching loss for baseline comparison.

    Useful for ablation studies or when you only want to train
    the flow field without energy or contrastive components.
    """

    def __init__(self, sigma: float = 0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x)
        x_noisy = x + self.sigma * noise
        _, v_raw, _ = model(x_noisy)
        target_score = -noise / (self.sigma**2)
        return (v_raw - target_score).pow(2).mean()


class NCEOnlyLoss(nn.Module):
    """
    Pure Noise Contrastive Estimation loss for baseline comparison.

    Useful when you only want to train the energy landscape
    without flow or contrastive components.
    """

    def __init__(self):
        super().__init__()

    def forward(self, s_real: torch.Tensor, s_noise: torch.Tensor) -> torch.Tensor:
        logits = s_noise - s_real
        return torch.log(1 + torch.exp(-logits)).mean()


class vMFInfoNCELoss(nn.Module):
    """
    Pure von Mises-Fisher InfoNCE loss for baseline comparison.

    Useful when you only want to train flow alignment
    without energy or score matching components.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, v_unit: torch.Tensor, v_unit_pos: torch.Tensor) -> torch.Tensor:
        batch_size = v_unit.size(0)
        sim_matrix = (v_unit @ v_unit_pos.T) / self.temperature
        labels = torch.arange(batch_size, device=v_unit.device)
        return F.cross_entropy(sim_matrix, labels)
