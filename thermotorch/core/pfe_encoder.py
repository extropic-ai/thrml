"""
Potential-Flow Embedding (PFE) Encoders

Core components that define the energy landscape and flow field:
- PFEEncoder: Base encoder producing energy scalar s(x) and flow vector v(x)
- LatentPFEEncoder: Wraps PFE inside a VAE for high-dimensional data

Mathematical Mapping:
    Flow Vector v(x)    → Denoising Score Matching (DSM)
    Energy Scalar s(x)   → Noise Contrastive Estimation (NCE)
    Flow Alignment       → vMF InfoNCE on unit sphere
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class PFEEncoder(nn.Module):
    """
    The core PFE engine that defines the Energy Landscape and Flow Field.

    This module maps input embeddings to:
        - s(x): Energy scalar (the "altimeter" - depth in energy valley)
        - v(x): Flow vector (the "compass" - direction to attractor)

    The flow vector points toward high-density regions (attractors),
    and the energy scalar indicates how "real" or "deep" a sample is.

    Args:
        embed_dim: Dimension of input embeddings
        hidden_dim: Hidden layer dimension (default: 2 * embed_dim)
        num_layers: Number of transformer-style blocks (default: 2)

    Example:
        >>> encoder = PFEEncoder(embed_dim=512)
        >>> x = torch.randn(32, 512)  # batch of embeddings
        >>> s, v_raw, v_unit = encoder(x)
        >>> s.shape  # [32] - energy scalars
        >>> v_raw.shape  # [32, 512] - flow vectors
    """

    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None, num_layers: int = 2, activation: str = "gelu"):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or (embed_dim * 2)

        # Select activation function
        act_fn = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
        }.get(activation, nn.GELU())

        # Backbone network (can be replaced with Transformer blocks)
        layers = []
        current_dim = embed_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(current_dim, self.hidden_dim),
                    act_fn,
                    nn.LayerNorm(self.hidden_dim),
                ]
            )
            current_dim = self.hidden_dim

        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()

        # Energy head: maps to scalar s(x) - the "altimeter"
        self.energy_head = nn.Linear(self.hidden_dim, 1)

        # Flow head: maps to vector v(x) - the "compass"
        self.flow_head = nn.Linear(self.hidden_dim, embed_dim)

        # Small constant for numerical stability in normalization
        self.eps = 1e-8

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute energy and flow for input embeddings.

        Args:
            x: Input tensor of shape [batch_size, embed_dim]

        Returns:
            s: Energy scalar of shape [batch_size]
            v_raw: Raw flow vector of shape [batch_size, embed_dim]
            v_unit: Unit-normalized flow vector of shape [batch_size, embed_dim]
        """
        # Pass through backbone
        h = self.backbone(x)

        # 1. Energy scalar s(x) - measures depth in energy landscape
        s = self.energy_head(h).squeeze(-1)  # [batch_size]

        # 2. Raw flow vector v(x) - used for Denoising Score Matching
        v_raw = self.flow_head(h)  # [batch_size, embed_dim]

        # 3. Unit-normalized flow vector - used for vMF similarity / InfoNCE
        v_norm = v_raw.norm(dim=-1, keepdim=True)
        v_unit = v_raw / (v_norm + self.eps)  # [batch_size, embed_dim]

        return s, v_raw, v_unit

    def get_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Get only the energy scalar s(x)."""
        h = self.backbone(x)
        return self.energy_head(h).squeeze(-1)

    def get_flow(self, x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """Get only the flow vector v(x)."""
        h = self.backbone(x)
        v = self.flow_head(h)
        if normalize:
            v = v / (v.norm(dim=-1, keepdim=True) + self.eps)
        return v


class LatentPFEEncoder(nn.Module):
    """
    Wraps PFE inside a VAE to process high-dimensional images efficiently.

    For high-dimensional data (e.g., 4K images, megapixel tensors), direct
    thermodynamic settling is inefficient. This encoder:
        1. Compresses images to latent space via frozen VAE
        2. Applies PFE in the compact latent space
        3. Returns latents ready for TSU settling

    The VAE is kept frozen - we only train the PFE landscape parameters.

    Args:
        vae: A VAE encoder with `encode()` method returning latent distribution
        pfe_encoder: A PFEEncoder instance for the latent space
        latent_dim: Flattened latent dimension (default: auto-detect)

    Example:
        >>> from diffusers import AutoencoderKL
        >>> vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        >>> pfe = PFEEncoder(embed_dim=4096)  # 4x64x64 latent
        >>> latent_pfe = LatentPFEEncoder(vae, pfe)
        >>> images = torch.randn(8, 3, 512, 512)
        >>> z_flat, s, v_raw, v_unit = latent_pfe(images)
    """

    def __init__(self, vae: nn.Module, pfe_encoder: PFEEncoder, latent_dim: Optional[int] = None):
        super().__init__()
        self.vae = vae
        self.pfe = pfe_encoder

        # Freeze VAE - we only train the PFE landscape
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()

        # Auto-detect latent dimension if not provided
        if latent_dim is None:
            # Try to infer from VAE config or default
            if hasattr(vae, "config") and hasattr(vae.config, "latent_channels"):
                # Typical VAE: 4 channels at 64x64 for 512x512 input
                self.latent_dim = vae.config.latent_channels * 64 * 64
            else:
                self.latent_dim = None
        else:
            self.latent_dim = latent_dim

    def encode_to_latent(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent space.

        Args:
            image: Input images of shape [batch, channels, height, width]

        Returns:
            z: Flattened latent tensor of shape [batch, latent_dim]
        """
        with torch.no_grad():
            # VAE encode - returns distribution
            latent_dist = self.vae.encode(image).latent_dist
            z = latent_dist.sample()  # [batch, latent_channels, h, w]

        # Flatten for PFE backbone
        z_flat = z.view(z.size(0), -1)  # [batch, latent_dim]
        return z_flat

    def decode_from_latent(self, z_flat: torch.Tensor, shape: Optional[tuple] = None) -> torch.Tensor:
        """
        Decode flattened latent back to image space.

        Args:
            z_flat: Flattened latent of shape [batch, latent_dim]
            shape: Target shape for unflatten (default: auto-detect)

        Returns:
            image: Reconstructed image tensor
        """
        # Unflatten if we have shape info
        if shape is not None:
            z = z_flat.view(shape)
        elif hasattr(self.vae, "config"):
            # Typical SD VAE latent shape
            batch = z_flat.size(0)
            channels = getattr(self.vae.config, "latent_channels", 4)
            h = w = int((z_flat.size(1) // channels) ** 0.5)
            z = z_flat.view(batch, channels, h, w)
        else:
            raise ValueError("Cannot determine latent shape for decoding")

        with torch.no_grad():
            image = self.vae.decode(z).sample

        return image

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode image and compute PFE landscape.

        Args:
            image: Input images of shape [batch, channels, height, width]

        Returns:
            z_flat: Flattened latent [batch, latent_dim]
            s: Energy scalar [batch]
            v_raw: Raw flow vector [batch, latent_dim]
            v_unit: Unit flow vector [batch, latent_dim]
        """
        # 1. Encode image to latent (no gradients - VAE frozen)
        z_flat = self.encode_to_latent(image)

        # 2. Get PFE landscape mapping for the latents
        s, v_raw, v_unit = self.pfe(z_flat)

        return z_flat, s, v_raw, v_unit

    def get_latent_shape(self, image: torch.Tensor) -> tuple:
        """Get the shape of latent representation before flattening."""
        with torch.no_grad():
            latent_dist = self.vae.encode(image).latent_dist
            z = latent_dist.sample()
        return z.shape[1:]  # (channels, h, w)
