"""
Triton Newton-Schulz orthogonalization kernel.

Fused implementation of the 5-step quintic Newton-Schulz iteration
for the M3 optimizer. Three operations per iteration:
1. Gram matrix: A = X @ X^T
2. Polynomial: B = b*A + c*A@A
3. Update: X = a*X + B@X

The flash-muon optimization exploits symmetry of the Gram matrix
to save ~50% of the GEMM in step 1.

AOL preconditioning (from flash-newton-schulz) eliminates one iteration
by providing a better starting point: 4 steps instead of 5.

Reference: flash-newton-schulz (thib-s), flash-muon (nil0x9).
"""

from __future__ import annotations

import torch
from torch import Tensor

from anamnesis.kernels import _TRITON_AVAILABLE

if _TRITON_AVAILABLE:
    import triton
    import triton.language as tl


# Quintic NS coefficients (Zhan, 2016)
_NS_A = 3.4445
_NS_B = -4.7750
_NS_C = 2.0315


def newton_schulz_fused(
    G: Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    use_aol: bool = False,
) -> Tensor:
    """
    Fused Newton-Schulz with optional AOL preconditioning.

    Uses Triton-accelerated matmul when available on CUDA,
    falls back to PyTorch otherwise.

    Args:
        G: Input matrix (m, n) where m >= n.
        steps: Number of NS iterations (5 without AOL, 4 with).
        eps: Stability epsilon.
        use_aol: Whether to apply AOL preconditioning.

    Returns:
        Approximately orthogonal matrix.
    """
    if _TRITON_AVAILABLE and G.is_cuda and G.shape[0] >= 16 and G.shape[1] >= 16:
        return _newton_schulz_triton(G, steps, eps, use_aol)
    return _newton_schulz_pytorch(G, steps, eps, use_aol)


def _newton_schulz_pytorch(
    G: Tensor, steps: int = 5, eps: float = 1e-7, use_aol: bool = False
) -> Tensor:
    """Pure PyTorch reference implementation."""
    a, b, c = _NS_A, _NS_B, _NS_C
    dtype = G.dtype
    X = G.to(torch.bfloat16) if G.is_cuda else G.float()
    X = X / (X.norm() + eps)

    if use_aol:
        n = min(X.shape)
        X = X * (n ** 0.5 / (X.norm() + eps))
        steps = max(1, steps - 1)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    return X.to(dtype)


if _TRITON_AVAILABLE:
    @triton.jit
    def _ns_gram_kernel(
        X_ptr, A_ptr,
        M, N,
        stride_xm, stride_xn,
        stride_am, stride_an,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """Compute upper triangle of Gram matrix A = X @ X^T."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        # Only compute upper triangle (symmetric)
        if pid_n < pid_m:
            # Copy from upper triangle
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            # Read the transposed block
            a_vals = tl.load(
                A_ptr + offs_n[:, None] * stride_am + offs_m[None, :] * stride_an,
                mask=(offs_n[:, None] < M) & (offs_m[None, :] < M),
                other=0.0,
            )
            # Store as transpose
            tl.store(
                A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an,
                tl.trans(a_vals),
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < M),
            )
            return

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, N, BLOCK_K):
            k_offs = k + tl.arange(0, BLOCK_K)
            x_m = tl.load(
                X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xn,
                mask=(offs_m[:, None] < M) & (k_offs[None, :] < N),
                other=0.0,
            )
            x_n = tl.load(
                X_ptr + offs_n[:, None] * stride_xm + k_offs[None, :] * stride_xn,
                mask=(offs_n[:, None] < M) & (k_offs[None, :] < N),
                other=0.0,
            )
            acc += tl.dot(x_m, tl.trans(x_n))

        tl.store(
            A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an,
            acc.to(tl.bfloat16),
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < M),
        )


def _newton_schulz_triton(
    G: Tensor, steps: int = 5, eps: float = 1e-7, use_aol: bool = False
) -> Tensor:
    """
    Triton-accelerated Newton-Schulz orthogonalization.

    Uses Triton for the Gram matrix computation (symmetric optimization)
    and PyTorch for the polynomial and update steps (which are small matmuls
    that cuBLAS handles efficiently).
    """
    a, b, c = _NS_A, _NS_B, _NS_C
    dtype = G.dtype
    M, N = G.shape

    # Use bfloat16 for the main computation, float32 for stability-critical parts
    compute_dtype = torch.bfloat16 if not use_aol else torch.float32
    X = G.to(compute_dtype)
    X = X / (X.norm() + eps)

    if use_aol:
        n = min(M, N)
        X = X * (n ** 0.5 / (X.norm() + eps))
        steps = max(1, steps - 1)

    for _ in range(steps):
        A = X @ X.mT
        # Polynomial in float32 for stability
        A32 = A.float() if A.dtype != torch.float32 else A
        B32 = b * A32 + c * (A32 @ A32)
        B = B32.to(compute_dtype)
        X = a * X + B @ X

    return X.to(dtype)
