"""
Optimized Triton/CUDA kernels for Anamnesis.

These kernels fuse multiple operations to minimize HBM round-trips:
- cms_update: forward + gradient + weight update in one kernel
- assoc_scan: parallel associative scan for momentum accumulation
- newton_schulz: fused Newton-Schulz orthogonalization

Requires: pip install anamnesis[kernels] (triton>=2.2.0)
Falls back to pure PyTorch if Triton is not available.
"""

_TRITON_AVAILABLE = False
try:
    import triton
    _TRITON_AVAILABLE = True
except ImportError:
    pass


def is_triton_available() -> bool:
    return _TRITON_AVAILABLE
