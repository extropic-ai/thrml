"""
TSU Layers: User-Facing nn.Module Components

This module provides drop-in PyTorch nn.Module layers that integrate
thermodynamic settling into standard neural network architectures.

Available Layers:
    - TSULayer: General-purpose thermodynamic settling layer
    - EnergyBridge: PFE-specific wrapper for energy/flow extraction
    - DTMLayer: Denoising Thermodynamic Model layer for generative tasks

Usage:
    These layers can be used like standard PyTorch nn.Module:

        >>> layer = TSULayer(steps=50, method='euler')
        >>> output = layer(model, x)

    Or integrated into larger models:

        >>> class MyModel(nn.Module):
        ...     def __init__(self):
        ...         self.encoder = nn.TransformerEncoder(...)
        ...         self.tsu = TSULayer(steps=20)
        ...     def forward(self, x):
        ...         h = self.encoder(x)
        ...         settled = self.tsu(self.pfe, h)
        ...         return settled
"""

import warnings
from enum import Enum
from typing import Any, Callable, Optional, Tuple, Union

import torch
import torch.nn as nn

from .bridge import GradientMethod, TSUBridge, create_bridge
from .pfe_encoder import PFEEncoder


class SettlingMode(Enum):
    """Available settling modes."""

    ATTRACTOR = "attractor"  # Settle to energy minimum
    TRAJECTORY = "trajectory"  # Return full settling trajectory
    SAMPLE = "sample"  # Sample from equilibrium distribution


class TSULayer(nn.Module):
    """
    General-purpose thermodynamic settling layer.

    This layer wraps a PFE model and performs settling to attractors.
    It can be used as a drop-in replacement for standard layers in
    architectures that benefit from energy-based reasoning.

    The layer handles:
        - Model-agnostic settling (works with any model having energy/flow)
        - Configurable settling parameters
        - Autograd integration via STE or implicit gradients

    Args:
        steps: Number of settling steps (default: 20)
        method: Integration method ('euler', 'rk4', 'ode')
        gradient_method: Gradient estimation ('ste', 'implicit', 'none')
        backend: Backend type ('cpu', 'thrml', 'tsu')
        return_trajectory: Whether to return full trajectory (default: False)

    Example:
        >>> layer = TSULayer(steps=50, method='rk4')
        >>> model = PFEEncoder(embed_dim=512)
        >>> noisy = torch.randn(32, 512)
        >>> settled = layer(model, noisy)

    Shape:
        - Input: (batch, dim) - any tensor matching model's expected shape
        - Output: (batch, dim) - settled tensor with same shape
    """

    def __init__(
        self,
        steps: int = 20,
        method: str = "euler",
        gradient_method: str = "ste",
        backend: str = "cpu",
        return_trajectory: bool = False,
        dt: Optional[float] = None,
    ):
        super().__init__()
        self.steps = steps
        self.method = method
        self.gradient_method = gradient_method
        self.backend_type = backend
        self.return_trajectory = return_trajectory
        self.dt = dt

        # Create bridge
        self._bridge = create_bridge(steps=steps, method=method, gradient_method=gradient_method, backend=backend)

    def forward(self, model: nn.Module, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform thermodynamic settling.

        Args:
            model: PFE model with energy/flow functions
            x: Input tensor [batch, dim]

        Returns:
            Settled tensor (and trajectory if return_trajectory=True)
        """
        return self._bridge(model, x, self.return_trajectory)

    def extra_repr(self) -> str:
        return (
            f"steps={self.steps}, "
            f"method={self.method}, "
            f"gradient={self.gradient_method}, "
            f"backend={self.backend_type}"
        )


class EnergyBridge(nn.Module):
    """
    PFE-specific wrapper for energy-based computations.

    This layer wraps a PFE encoder and provides convenient methods for:
        - Energy evaluation: s(x)
        - Flow computation: v(x)
        - Attractor settling: x* = settle(x)
        - Energy-guided generation

    Unlike TSULayer, this provides a more structured interface specifically
    designed for PFE models with explicit energy and flow heads.

    Args:
        pfe_encoder: PFEEncoder model
        steps: Number of settling steps
        method: Integration method
        gradient_method: Gradient estimation method
        backend: Backend type

    Example:
        >>> pfe = PFEEncoder(embed_dim=512)
        >>> bridge = EnergyBridge(pfe, steps=50)
        >>> noisy = torch.randn(32, 512)
        >>> settled, energy, flow = bridge(noisy, return_all=True)
    """

    def __init__(
        self,
        pfe_encoder: PFEEncoder,
        steps: int = 20,
        method: str = "euler",
        gradient_method: str = "ste",
        backend: str = "cpu",
    ):
        super().__init__()
        self.pfe = pfe_encoder
        self.steps = steps
        self.method = method
        self.gradient_method = gradient_method

        self._bridge = create_bridge(steps=steps, method=method, gradient_method=gradient_method, backend=backend)

    def get_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Get energy scalar s(x)."""
        return self.pfe.get_energy(x)

    def get_flow(self, x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """Get flow vector v(x)."""
        return self.pfe.get_flow(x, normalize=normalize)

    def settle(
        self, x: torch.Tensor, steps: Optional[int] = None, return_trajectory: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Settle to energy minimum.

        Args:
            x: Initial state
            steps: Override default steps
            return_trajectory: Return full trajectory

        Returns:
            Attractor (and trajectory if requested)
        """
        steps = steps or self.steps

        if steps != self.steps:
            # Create temporary bridge with different steps
            bridge = create_bridge(steps=steps, method=self.method, gradient_method=self.gradient_method)
            return bridge(self.pfe, x, return_trajectory)

        return self._bridge(self.pfe, x, return_trajectory)

    def forward(
        self, x: torch.Tensor, return_all: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Compute PFE representation and optionally settle.

        Args:
            x: Input tensor
            return_all: If True, return (settled, energy, flow)

        Returns:
            If return_all:
                settled: Settled tensor
                energy: Energy scalar
                flow: Flow vector
            Else:
                settled: Settled tensor
        """
        # Get energy and flow
        s, v_raw, v_unit = self.pfe(x)

        # Settle
        settled = self._bridge(self.pfe, x)

        if return_all:
            return settled, s, v_raw
        return settled

    def generate(
        self, batch_size: int, dim: int, device: Optional[str] = None, steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate samples by settling from random noise.

        Args:
            batch_size: Number of samples
            dim: Latent dimension
            device: Target device
            steps: Settling steps

        Returns:
            Generated samples
        """
        device = device or next(self.pfe.parameters()).device
        noise = torch.randn(batch_size, dim, device=device)
        return self.settle(noise, steps=steps)


class DTMLayer(nn.Module):
    """
    Denoising Thermodynamic Model Layer.

    This layer implements the DTM algorithm for generative tasks,
    replacing standard diffusion model denoisers with thermodynamic
    settling. It's designed to work with VAE latent spaces.

    DTM vs Standard Diffusion:
        - Standard: Neural network denoiser on GPU
        - DTM: Physical settling on TSU (or THRML simulation)

    Architecture:
        1. Noisy latent → PFE energy landscape
        2. Thermodynamic settling to attractor
        3. Attractor → clean latent

    Args:
        pfe_encoder: PFEEncoder for energy landscape
        steps: Number of settling steps (default: 20)
        method: Integration method
        temperature: Sampling temperature (default: 1.0)

    Example:
        >>> pfe = PFEEncoder(embed_dim=4*64*64)  # VAE latent
        >>> dtm = DTMLayer(pfe, steps=50)
        >>> noisy_latent = torch.randn(8, 4*64*64)
        >>> clean_latent = dtm(noisy_latent)
    """

    def __init__(
        self,
        pfe_encoder: PFEEncoder,
        steps: int = 20,
        method: str = "euler",
        gradient_method: str = "ste",
        temperature: float = 1.0,
        backend: str = "cpu",
    ):
        super().__init__()
        self.pfe = pfe_encoder
        self.steps = steps
        self.method = method
        self.temperature = temperature

        self._bridge = create_bridge(steps=steps, method=method, gradient_method=gradient_method, backend=backend)

    def denoise(
        self, noisy_latent: torch.Tensor, noise_level: Optional[float] = None, steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Denoise a latent by thermodynamic settling.

        Args:
            noisy_latent: Noisy latent tensor
            noise_level: Optional noise scale (for temperature scaling)
            steps: Override default settling steps

        Returns:
            Denoised latent
        """
        steps = steps or self.steps

        # Scale by temperature if provided
        if noise_level is not None and self.temperature != 1.0:
            # Adjust energy landscape based on noise level
            scaled_latent = noisy_latent / self.temperature
            settled = self._bridge(self.pfe, scaled_latent)
            return settled * self.temperature

        return self._bridge(self.pfe, noisy_latent)

    def forward(self, noisy_latent: torch.Tensor) -> torch.Tensor:
        """Alias for denoise()."""
        return self.denoise(noisy_latent)

    @torch.no_grad()
    def generate(
        self, batch_size: int, latent_dim: int, device: Optional[str] = None, steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate samples from pure noise.

        Args:
            batch_size: Number of samples
            latent_dim: Latent dimension
            device: Target device
            steps: Settling steps

        Returns:
            Generated samples
        """
        device = device or next(self.pfe.parameters()).device
        noise = torch.randn(batch_size, latent_dim, device=device)
        return self.denoise(noise, steps=steps)


class LatentDTM(nn.Module):
    """
    Latent-space DTM for high-dimensional data (e.g., images).

    Combines a VAE encoder/decoder with DTM for generative tasks:
        1. Image → VAE encoder → latent
        2. Noisy latent → DTM settling → clean latent
        3. Clean latent → VAE decoder → image

    Args:
        vae: VAE model with encode/decode methods
        pfe_encoder: PFEEncoder for latent space
        steps: Settling steps
        freeze_vae: Whether to freeze VAE weights (default: True)

    Example:
        >>> from diffusers import AutoencoderKL
        >>> vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        >>> pfe = PFEEncoder(embed_dim=4*64*64)
        >>> latent_dtm = LatentDTM(vae, pfe, steps=50)
        >>> noisy_images = torch.randn(8, 3, 512, 512)
        >>> clean_images = latent_dtm(noisy_images)
    """

    def __init__(
        self,
        vae: nn.Module,
        pfe_encoder: PFEEncoder,
        steps: int = 20,
        method: str = "euler",
        freeze_vae: bool = True,
        backend: str = "cpu",
    ):
        super().__init__()
        self.vae = vae
        self.pfe = pfe_encoder
        self.steps = steps

        # Freeze VAE if requested
        if freeze_vae:
            for param in self.vae.parameters():
                param.requires_grad = False
            self.vae.eval()

        self._dtm = DTMLayer(pfe_encoder=pfe_encoder, steps=steps, method=method, backend=backend)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space."""
        with torch.no_grad():
            latent_dist = self.vae.encode(images).latent_dist
            latent = latent_dist.sample()
        return latent.view(latent.size(0), -1)  # Flatten

    def decode(self, latent: torch.Tensor, latent_shape: Tuple[int, ...]) -> torch.Tensor:
        """Decode latent to images."""
        latent = latent.view(latent_shape)
        with torch.no_grad():
            images = self.vae.decode(latent).sample
        return images

    def forward(
        self, images: torch.Tensor, return_latent: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process images through latent DTM.

        Args:
            images: Input images [batch, C, H, W]
            return_latent: If True, also return latent

        Returns:
            Processed images (and latent if requested)
        """
        # Encode to latent
        latent = self.encode(images)
        latent_shape = (-1,) + latent.shape[1:]

        # Settle in latent space
        settled_latent = self._dtm(latent)

        # Decode to image
        output_images = self.decode(settled_latent, latent_shape)

        if return_latent:
            return output_images, settled_latent
        return output_images

    @torch.no_grad()
    def generate(
        self, batch_size: int, image_shape: Tuple[int, int, int] = (3, 512, 512), device: Optional[str] = None
    ) -> torch.Tensor:
        """
        Generate images from pure noise.

        Args:
            batch_size: Number of images
            image_shape: (channels, height, width)
            device: Target device

        Returns:
            Generated images
        """
        device = device or next(self.pfe.parameters()).device

        # Generate in latent space
        # Assuming standard VAE latent: 4 channels, spatial dims / 8
        C, H, W = image_shape
        latent_C = 4  # Standard SD VAE
        latent_H, latent_W = H // 8, W // 8
        latent_dim = latent_C * latent_H * latent_W

        noise = torch.randn(batch_size, latent_dim, device=device)
        settled_latent = self._dtm(noise)

        # Decode
        latent_shape = (batch_size, latent_C, latent_H, latent_W)
        images = self.decode(settled_latent, latent_shape)

        return images


def create_tsu_layer(
    steps: int = 20, method: str = "euler", gradient_method: str = "ste", backend: str = "cpu"
) -> TSULayer:
    """Factory function to create a TSULayer."""
    return TSULayer(steps=steps, method=method, gradient_method=gradient_method, backend=backend)


def create_energy_bridge(
    pfe_encoder: PFEEncoder, steps: int = 20, method: str = "euler", backend: str = "cpu"
) -> EnergyBridge:
    """Factory function to create an EnergyBridge."""
    return EnergyBridge(pfe_encoder=pfe_encoder, steps=steps, method=method, backend=backend)
