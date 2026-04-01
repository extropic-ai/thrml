"""
Memory Bridge: Zero-Copy Tensor Exchange (Production-Ready)

Handles conversion between PyTorch tensors and JAX arrays using DLPack.
This enables efficient data transfer between PyTorch and THRML (JAX) backends
without copying data to CPU.

Key Features:
    - Zero-copy tensor exchange via DLPack (when compatible)
    - Automatic fallback to safe copy when DLPack unavailable
    - Device management (CPU/GPU)
    - Type preservation and validation
    - Thread-safe operations
    - Comprehensive error handling

DLPack Protocol:
    PyTorch Tensor → DLPack Capsule → JAX Array
    JAX Array → DLPack Capsule → PyTorch Tensor

Compatibility:
    - PyTorch >= 1.10 (DLPack support)
    - JAX >= 0.4.0 (DLPack support)
    - Works on CPU and CUDA (when both frameworks configured for same GPU)
"""

import logging
import threading
import warnings
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases
ArrayLike = Union[torch.Tensor, np.ndarray, "jnp.Array"]


class ConversionMode(Enum):
    """Available conversion modes."""

    DLPACK = "dlpack"  # Zero-copy DLPack (preferred)
    COPY = "copy"  # Safe copy via NumPy
    AUTO = "auto"  # Try DLPack, fallback to copy


class DLPackError(Exception):
    """Raised when DLPack conversion fails."""

    pass


class DeviceMismatchError(Exception):
    """Raised when tensor and target device are incompatible."""

    pass


class MemoryBridge:
    """
    Production-ready memory bridge between PyTorch and JAX/THRML.

    This class handles the low-level tensor conversions required
    for passing data between PyTorch's computational graph and
    the THRML backend without expensive CPU copies.

    Features:
        - Zero-copy DLPack transfer (when available)
        - Automatic device management
        - Thread-safe operations
        - Conversion caching
        - Comprehensive error handling

    Example:
        >>> bridge = MemoryBridge(device='cuda')
        >>> torch_tensor = torch.randn(32, 512, device='cuda')
        >>> jax_array = bridge.torch_to_jax(torch_tensor)
        >>> # Process in JAX...
        >>> result_torch = bridge.jax_to_torch(jax_array)
    """

    def __init__(
        self,
        device: Optional[str] = None,
        mode: ConversionMode = ConversionMode.AUTO,
        cache_conversions: bool = False,
        cache_size: int = 100,
    ):
        """
        Initialize memory bridge.

        Args:
            device: Target device ('cuda', 'cpu', 'auto' for auto-detect)
            mode: Conversion mode (DLPACK, COPY, or AUTO)
            cache_conversions: Enable conversion caching (use with caution)
            cache_size: Maximum cache entries
        """
        # Handle 'auto' device
        if device == "auto" or device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.mode = mode
        self.cache_conversions = cache_conversions
        self.cache_size = cache_size

        # Lazy initialization flags
        self._jax_available: Optional[bool] = None
        self._dlpack_available: Optional[bool] = None
        self._jax_version: Optional[str] = None
        self._torch_version: Optional[str] = None

        # Thread lock for safety
        self._lock = threading.Lock()

        # Conversion cache
        self._cache: Dict[int, Any] = {}
        self._cache_order: list = []

        # Compatibility info
        self._compatibility_checked = False
        self._compatibility_info: Dict[str, Any] = {}

    @property
    def jax_available(self) -> bool:
        """Check if JAX is available (lazy initialization)."""
        if self._jax_available is None:
            try:
                import jax
                import jax.numpy as jnp

                self._jax_available = True
                self._jax_version = jax.__version__
                logger.debug(f"JAX {self._jax_version} available")
            except ImportError:
                self._jax_available = False
                logger.debug("JAX not available")
        return self._jax_available

    @property
    def dlpack_available(self) -> bool:
        """Check if DLPack is available (lazy initialization)."""
        if self._dlpack_available is None:
            # Check PyTorch DLPack support
            self._dlpack_available = hasattr(torch.Tensor, "__dlpack__")
            self._torch_version = torch.__version__
            logger.debug(f"PyTorch {self._torch_version} DLPack: {self._dlpack_available}")
        return self._dlpack_available

    def check_compatibility(self) -> Dict[str, Any]:
        """
        Check full compatibility between PyTorch and JAX.

        Returns:
            Dictionary with compatibility information
        """
        if self._compatibility_checked:
            return self._compatibility_info

        info = {
            "torch_version": torch.__version__,
            "jax_available": self.jax_available,
            "jax_version": self._jax_version,
            "dlpack_available": self.dlpack_available,
            "cuda_available": torch.cuda.is_available(),
            "device": self.device,
            "compatible": True,
            "warnings": [],
            "recommendations": [],
        }

        # Check JAX device compatibility
        if self.jax_available:
            try:
                import jax

                jax_devices = jax.devices()
                info["jax_devices"] = [str(d) for d in jax_devices]

                # Check GPU compatibility
                if self.device == "cuda" and torch.cuda.is_available():
                    gpu_devices = [d for d in jax_devices if d.platform == "gpu"]
                    if not gpu_devices:
                        info["warnings"].append(
                            "CUDA requested but JAX has no GPU devices. " "Install jaxlib with CUDA support."
                        )
                        info["recommendations"].append("pip install jax[cuda] jaxlib[cuda]")
            except Exception as e:
                info["warnings"].append(f"JAX device check failed: {e}")

        # Check DLPack compatibility
        if not self.dlpack_available:
            info["warnings"].append("DLPack not available. Zero-copy transfers disabled.")
            info["recommendations"].append("Upgrade PyTorch to >= 1.10 for DLPack support.")

        # Set overall compatibility
        if info["warnings"]:
            info["compatible"] = False

        self._compatibility_info = info
        self._compatibility_checked = True
        return info

    def _try_dlpack_conversion(self, tensor: torch.Tensor, to_jax: bool = True) -> Optional[Any]:
        """
        Attempt DLPack conversion.

        Args:
            tensor: PyTorch tensor to convert
            to_jax: If True, convert to JAX; if False, assume JAX input

        Returns:
            Converted array/tensor or None if conversion failed
        """
        if not self.dlpack_available:
            return None

        if to_jax and not self.jax_available:
            return None

        try:
            if to_jax:
                import jax.dlpack as jax_dlpack

                # Ensure contiguous tensor
                tensor = tensor.contiguous()

                # PyTorch -> DLPack -> JAX
                try:
                    # New PyTorch API (>= 1.13)
                    capsule = tensor.__dlpack__()
                    jax_array = jax_dlpack.from_dlpack(capsule)
                    return jax_array
                except (AttributeError, TypeError):
                    # Old PyTorch API
                    capsule = torch.utils.dlpack.to_dlpack(tensor)
                    jax_array = jax_dlpack.from_dlpack(capsule)
                    return jax_array

            else:
                # JAX -> DLPack -> PyTorch
                import jax
                import jax.dlpack as jax_dlpack
                import jax.numpy as jnp

                if not isinstance(tensor, jnp.ndarray):
                    return None

                try:
                    # JAX -> DLPack
                    capsule = jax_dlpack.to_dlpack(tensor)
                    # DLPack -> PyTorch
                    result = torch.utils.dlpack.from_dlpack(capsule)
                    return result
                except (AttributeError, TypeError):
                    return None

        except Exception as e:
            logger.debug(f"DLPack conversion failed: {e}")
            return None

    def _fallback_copy(
        self, tensor: Union[torch.Tensor, "jnp.Array"], to_jax: bool = True
    ) -> Union["jnp.Array", torch.Tensor]:
        """
        Safe copy conversion via NumPy.

        Args:
            tensor: Input tensor/array
            to_jax: If True, convert to JAX; if False, convert to PyTorch

        Returns:
            Converted array/tensor
        """
        if to_jax:
            # PyTorch -> NumPy -> JAX
            if isinstance(tensor, torch.Tensor):
                np_array = tensor.detach().cpu().numpy()
            else:
                np_array = np.array(tensor)

            import jax.numpy as jnp

            return jnp.array(np_array)
        else:
            # JAX -> NumPy -> PyTorch
            import jax.numpy as jnp

            if isinstance(tensor, jnp.ndarray):
                np_array = np.array(tensor)
            else:
                np_array = np.array(tensor)

            result = torch.from_numpy(np_array)
            return result.to(self.device)

    def torch_to_jax(self, tensor: torch.Tensor, force_copy: bool = False) -> "jnp.Array":
        """
        Convert PyTorch tensor to JAX array.

        This attempts zero-copy DLPack transfer first, falling back
        to a safe copy if DLPack is unavailable or fails.

        Args:
            tensor: PyTorch tensor to convert
            force_copy: If True, skip DLPack and use copy

        Returns:
            JAX array with same data

        Raises:
            ImportError: If JAX is not installed
        """
        if not self.jax_available:
            raise ImportError("JAX is required for torch_to_jax conversion. " "Install with: pip install jax jaxlib")

        import jax.numpy as jnp

        with self._lock:
            # Check cache
            if self.cache_conversions:
                cache_key = id(tensor)
                if cache_key in self._cache:
                    return self._cache[cache_key]

            # Determine conversion mode
            use_dlpack = self.mode in (ConversionMode.DLPACK, ConversionMode.AUTO) and not force_copy

            result = None

            # Try DLPack
            if use_dlpack:
                result = self._try_dlpack_conversion(tensor, to_jax=True)

            # Fallback to copy
            if result is None:
                result = self._fallback_copy(tensor, to_jax=True)

            # Cache result
            if self.cache_conversions and result is not None:
                self._cache_tensor(cache_key, result)

            return result

    def jax_to_torch(self, array: "jnp.Array", device: Optional[str] = None, force_copy: bool = False) -> torch.Tensor:
        """
        Convert JAX array to PyTorch tensor.

        Args:
            array: JAX array to convert
            device: Target device (default: self.device)
            force_copy: If True, skip DLPack and use copy

        Returns:
            PyTorch tensor with same data
        """
        device = device or self.device

        with self._lock:
            # Determine conversion mode
            use_dlpack = self.mode in (ConversionMode.DLPACK, ConversionMode.AUTO) and not force_copy

            result = None

            # Try DLPack
            if use_dlpack:
                result = self._try_dlpack_conversion(array, to_jax=False)

            # Fallback to copy
            if result is None:
                result = self._fallback_copy(array, to_jax=False)

            # Move to target device
            if result is not None and device:
                result = result.to(device)

            return result

    def _cache_tensor(self, key: int, value: Any):
        """Cache a conversion result."""
        if len(self._cache) >= self.cache_size:
            # Evict oldest
            oldest_key = self._cache_order.pop(0)
            del self._cache[oldest_key]

        self._cache[key] = value
        self._cache_order.append(key)

    def clear_cache(self):
        """Clear the conversion cache."""
        with self._lock:
            self._cache.clear()
            self._cache_order.clear()

    def numpy_to_jax(self, arr: np.ndarray) -> "jnp.Array":
        """Convert NumPy array to JAX array."""
        if not self.jax_available:
            raise ImportError("JAX is required. Install with: pip install jax jaxlib")

        import jax.numpy as jnp

        return jnp.array(arr)

    def jax_to_numpy(self, array: "jnp.Array") -> np.ndarray:
        """Convert JAX array to NumPy array."""
        return np.array(array)

    def torch_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor to NumPy array."""
        return tensor.detach().cpu().numpy()

    def numpy_to_torch(self, arr: np.ndarray, device: Optional[str] = None) -> torch.Tensor:
        """Convert NumPy array to PyTorch tensor."""
        device = device or self.device
        return torch.from_numpy(arr).to(device)

    def synchronize(self):
        """
        Synchronize devices for safe transfer.

        Call this before DLPack operations when mixing PyTorch and JAX
        on GPU to ensure all operations are complete.
        """
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.synchronize()

        if self.jax_available:
            try:
                import jax

                # JAX synchronization happens automatically
                jax.devices()
            except Exception:
                pass


class TensorCache:
    """
    Thread-safe cache for frequently converted tensors.

    Useful when the same tensors are repeatedly passed between
    PyTorch and JAX (e.g., during training loops).

    Example:
        >>> cache = TensorCache(max_size=100)
        >>> jax_arr = cache.get_jax(tensor, key='features')
    """

    def __init__(self, max_size: int = 100):
        """
        Initialize tensor cache.

        Args:
            max_size: Maximum number of cached tensors
        """
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def get_jax(
        self, tensor: torch.Tensor, key: Optional[str] = None, bridge: Optional[MemoryBridge] = None
    ) -> "jnp.Array":
        """
        Get JAX array from cache or convert.

        Args:
            tensor: PyTorch tensor
            key: Cache key (default: tensor.data_ptr())
            bridge: MemoryBridge instance (default: create new)

        Returns:
            JAX array
        """
        key = key or str(tensor.data_ptr())

        with self._lock:
            if key in self._cache:
                return self._cache[key]

            # Convert
            if bridge is None:
                bridge = MemoryBridge()

            jax_array = bridge.torch_to_jax(tensor)

            # Evict if needed
            if len(self._cache) >= self.max_size:
                # Remove oldest (first key)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            self._cache[key] = jax_array
            return jax_array

    def get_torch(
        self,
        array: "jnp.Array",
        key: Optional[str] = None,
        device: Optional[str] = None,
        bridge: Optional[MemoryBridge] = None,
    ) -> torch.Tensor:
        """
        Get PyTorch tensor from cache or convert.

        Args:
            array: JAX array
            key: Cache key
            device: Target device
            bridge: MemoryBridge instance

        Returns:
            PyTorch tensor
        """
        import jax.numpy as jnp

        key = key or str(id(array))

        with self._lock:
            if key in self._cache:
                return self._cache[key]

            if bridge is None:
                bridge = MemoryBridge(device=device)

            tensor = bridge.jax_to_torch(array, device=device)

            # Evict if needed
            if len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            self._cache[key] = tensor
            return tensor

    def clear(self):
        """Clear the cache."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


def ensure_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor is contiguous in memory.

    Args:
        tensor: Input tensor

    Returns:
        Contiguous tensor (same tensor or copy)
    """
    if not tensor.is_contiguous():
        return tensor.contiguous()
    return tensor


def get_device(tensor_or_array: ArrayLike) -> str:
    """
    Get device string from tensor or array.

    Args:
        tensor_or_array: Input tensor/array

    Returns:
        Device string ('cpu', 'cuda', etc.)
    """
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.device.type
    return "cpu"  # NumPy/JAX arrays are on CPU by default


def get_dtype_info(tensor_or_array: ArrayLike) -> Dict[str, Any]:
    """
    Get dtype information from tensor or array.

    Args:
        tensor_or_array: Input tensor/array

    Returns:
        Dictionary with dtype information
    """
    if isinstance(tensor_or_array, torch.Tensor):
        return {
            "dtype": str(tensor_or_array.dtype),
            "size": tensor_or_array.element_size(),
            "is_floating_point": tensor_or_array.is_floating_point(),
        }
    elif isinstance(tensor_or_array, np.ndarray):
        return {
            "dtype": str(tensor_or_array.dtype),
            "size": tensor_or_array.itemsize,
            "is_floating_point": np.issubdtype(tensor_or_array.dtype, np.floating),
        }
    else:
        # JAX array
        import jax.numpy as jnp

        if isinstance(tensor_or_array, jnp.ndarray):
            return {
                "dtype": str(tensor_or_array.dtype),
                "size": tensor_or_array.itemsize,
                "is_floating_point": jnp.issubdtype(tensor_or_array.dtype, jnp.floating),
            }
    return {"dtype": "unknown", "size": 0, "is_floating_point": False}


# Global bridge instance for convenience
_global_bridge: Optional[MemoryBridge] = None
_global_bridge_lock = threading.Lock()


def get_global_bridge(device: Optional[str] = None) -> MemoryBridge:
    """
    Get or create global MemoryBridge instance.

    Args:
        device: Target device (default: auto-detect)

    Returns:
        MemoryBridge instance
    """
    global _global_bridge

    with _global_bridge_lock:
        if _global_bridge is None:
            _global_bridge = MemoryBridge(device=device)
        return _global_bridge


def reset_global_bridge():
    """Reset the global bridge instance."""
    global _global_bridge

    with _global_bridge_lock:
        _global_bridge = None


def synchronize_devices():
    """
    Synchronize CUDA devices for safe DLPack transfer.

    Call this before DLPack operations when mixing PyTorch and JAX
    on GPU to ensure all operations are complete.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    try:
        import jax

        # JAX synchronization happens automatically
        jax.devices()
    except ImportError:
        pass
