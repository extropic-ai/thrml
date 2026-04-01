"""
ThermoTorch Backends

Backend abstraction for running PFE settling on different hardware:
    - cpu_backend: Pure Python/NumPy fallback for testing
    - thrml_backend: JAX THRML simulation (GPU)
    - tsu_backend: Physical Extropic hardware (XTR-0, Z1)

Hardware Roadmap:
    - CPU: Always available, slow but portable
    - THRML: JAX simulation, GPU accelerated
    - TSU: Physical hardware (when available)
"""

from .base import BackendType, BaseBackend, ThermoBackend
from .cpu_backend import CPUBackend

# THRML backend (requires JAX)
try:
    from .thrml_backend import SettlingMethod as THRMLSettlingMethod
    from .thrml_backend import (
        THRMLBackend,
        THRMLConfig,
        THRMLSampler,
        create_thrml_backend,
        get_jax_device_info,
        is_jax_available,
        is_thrml_available,
    )

    _THRML_AVAILABLE = True
except ImportError:
    _THRML_AVAILABLE = False
    THRMLBackend = None
    THRMLSampler = None
    THRMLConfig = None
    THRMLSettlingMethod = None
    is_jax_available = lambda: False
    is_thrml_available = lambda: False
    create_thrml_backend = None
    get_jax_device_info = lambda: {"available": False}

# TSU hardware backend
try:
    from .tsu_backend import (
        HardwareStatus,
        MockTSUDevice,
    )
    from .tsu_backend import SettlingMode as TSUSettlingMode
    from .tsu_backend import (
        TSUBackend,
        TSUConfig,
        TSUDeviceCapabilities,
        TSUDeviceStub,
        TSUHardwareError,
        create_tsu_backend,
        detect_tsu_devices,
        get_tsu_info,
        is_tsu_available,
    )

    _TSU_AVAILABLE = True
except ImportError:
    _TSU_AVAILABLE = False
    TSUBackend = None
    TSUConfig = None
    TSUDeviceCapabilities = None
    TSUDeviceStub = None
    MockTSUDevice = None
    HardwareStatus = None
    TSUSettlingMode = None
    TSUHardwareError = None
    detect_tsu_devices = lambda: []
    create_tsu_backend = None
    is_tsu_available = lambda: False
    get_tsu_info = lambda: {"devices_detected": 0}

__all__ = [
    # Base
    "ThermoBackend",
    "BaseBackend",
    "BackendType",
    # CPU
    "CPUBackend",
    # THRML
    "THRMLBackend",
    "THRMLSampler",
    "THRMLConfig",
    "THRMLSettlingMethod",
    "is_jax_available",
    "is_thrml_available",
    "create_thrml_backend",
    "get_jax_device_info",
    # TSU
    "TSUBackend",
    "TSUConfig",
    "TSUDeviceCapabilities",
    "TSUDeviceStub",
    "MockTSUDevice",
    "HardwareStatus",
    "TSUSettlingMode",
    "TSUHardwareError",
    "detect_tsu_devices",
    "create_tsu_backend",
    "is_tsu_available",
    "get_tsu_info",
]
