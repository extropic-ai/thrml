"""
ThermoTorch: A PyTorch framework for Potential-Flow Embeddings (PFE)
and thermodynamic computing on Extropic TSU hardware.

Core Philosophy:
- PyTorch owns the computational graph and parameter updates
- TSU owns the energy landscape settling and denoising
- PFE maps physical concepts to proven ML techniques:
    Flow Vector v(x) → Denoising Score Matching (DSM)
    Energy Scalar s(x) → Noise Contrastive Estimation (NCE)
    Similarity → vMF InfoNCE
    Inference → Probability Flow ODE

Architecture:
- PyTorch Frontend (Map Maker): Defines energy landscape and flow vectors
- TSU Bridge (Translator): Zero-copy DLPack, STE/Implicit gradients
- ThermoBackend (Gravity): CPU/THRML/TSU execution
"""

__version__ = "0.2.0"

# Backends
from .backends import (
    BackendType,
    BaseBackend,
    CPUBackend,
    ThermoBackend,
    THRMLBackend,
    is_thrml_available,
)

# Core components
from .core import (  # Encoders; Losses; Settling; Memory; Bridge; Layers
    DSMOnlyLoss,
    DTMLayer,
    EnergyBridge,
    GradientMethod,
    ImplicitGradientBridge,
    LatentDTM,
    LatentPFEEncoder,
    MemoryBridge,
    NCEOnlyLoss,
    PFEEncoder,
    PFELoss,
    SettlingMethod,
    SettlingMode,
    TensorCache,
    TSUBridge,
    TSULayer,
    TSUSettler,
    create_bridge,
    create_energy_bridge,
    create_tsu_layer,
    tsu_settle,
    vMFInfoNCELoss,
)

__all__ = [
    # Encoders
    "PFEEncoder",
    "LatentPFEEncoder",
    # Losses
    "PFELoss",
    "DSMOnlyLoss",
    "NCEOnlyLoss",
    "vMFInfoNCELoss",
    # Settling
    "tsu_settle",
    "TSUSettler",
    "SettlingMethod",
    # Memory
    "MemoryBridge",
    "TensorCache",
    # Bridge
    "TSUBridge",
    "ImplicitGradientBridge",
    "create_bridge",
    "GradientMethod",
    # Layers
    "TSULayer",
    "EnergyBridge",
    "DTMLayer",
    "LatentDTM",
    "SettlingMode",
    "create_tsu_layer",
    "create_energy_bridge",
    # Backends
    "ThermoBackend",
    "BaseBackend",
    "BackendType",
    "CPUBackend",
    "THRMLBackend",
    "is_thrml_available",
]
