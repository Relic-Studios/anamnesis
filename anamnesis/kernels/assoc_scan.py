"""
Parallel Associative Scan for momentum accumulation.

The Titans memory update uses momentum: S_t = η_t · S_{t-1} - θ_t · ∇ℓ
This is a linear recurrence that can be parallelized via associative scan.

The associative scan computes prefix sums under an associative binary operator.
For the momentum recurrence with decay η and input u:
    s_t = η_t · s_{t-1} + u_t

This can be expressed as: (η_t, u_t) ∘ (η_{t-1}, u_{t-1}) = (η_t · η_{t-1}, η_t · u_{t-1} + u_t)

The operator ∘ is associative, so the full sequence can be computed in O(log N)
parallel steps instead of O(N) sequential steps.

Two implementations:
1. Pure PyTorch (fallback) — sequential, O(N)
2. Triton kernel (GPU) — parallel, O(N log N) work but O(log N) depth

Reference: lucidrains/titans-pytorch AssocScan, Blelloch 1990.
"""

from __future__ import annotations

import torch
from torch import Tensor


def associative_scan_sequential(
    decay: Tensor,
    values: Tensor,
    initial: Tensor | None = None,
) -> Tensor:
    """
    Sequential associative scan (pure PyTorch fallback).

    Computes: s_t = decay_t · s_{t-1} + values_t

    Args:
        decay: Decay factors (batch, seq_len, ...).
        values: Input values (batch, seq_len, ...).
        initial: Initial state s_0 (batch, ...). If None, uses zeros.

    Returns:
        Scanned output (batch, seq_len, ...).
    """
    batch, seq_len = decay.shape[:2]
    rest_shape = decay.shape[2:]

    if initial is None:
        state = torch.zeros(batch, *rest_shape, device=decay.device, dtype=decay.dtype)
    else:
        state = initial.clone()

    outputs = []
    for t in range(seq_len):
        state = decay[:, t] * state + values[:, t]
        outputs.append(state)

    return torch.stack(outputs, dim=1)


def associative_scan(
    decay: Tensor,
    values: Tensor,
    initial: Tensor | None = None,
    use_triton: bool = True,
) -> Tensor:
    """
    Parallel associative scan with Triton acceleration.

    Falls back to sequential PyTorch if Triton is unavailable or input is on CPU.

    Args:
        decay: Decay factors (batch, seq_len, ...).
        values: Input values (batch, seq_len, ...).
        initial: Initial state (batch, ...).
        use_triton: Whether to attempt Triton kernel.

    Returns:
        Scanned output (batch, seq_len, ...).
    """
    from anamnesis.kernels import _TRITON_AVAILABLE

    if use_triton and _TRITON_AVAILABLE and decay.is_cuda:
        return _associative_scan_triton(decay, values, initial)

    return associative_scan_sequential(decay, values, initial)


def _associative_scan_triton(
    decay: Tensor,
    values: Tensor,
    initial: Tensor | None = None,
) -> Tensor:
    """
    Triton-accelerated parallel associative scan.

    Uses the Blelloch up-sweep / down-sweep algorithm:
    1. Up-sweep: combine pairs at increasing stride
    2. Down-sweep: propagate results back at decreasing stride

    Total work: O(N log N), depth: O(log N)

    NOTE: This is a scaffold. The actual Triton kernel requires GPU testing.
    Currently falls back to the sequential implementation.
    """
    # TODO: Implement Triton kernel when GPU is available for testing
    # The kernel structure follows:
    #
    # @triton.jit
    # def _assoc_scan_kernel(
    #     decay_ptr, values_ptr, output_ptr,
    #     batch, seq_len, feat_dim,
    #     BLOCK_SIZE: tl.constexpr,
    # ):
    #     # Up-sweep phase
    #     for stride in range(1, log2(seq_len)):
    #         if tid % (2 * stride) == 0:
    #             # Combine: (d_i, v_i) ∘ (d_j, v_j) = (d_i*d_j, d_i*v_j + v_i)
    #             output[tid] = decay[tid] * output[tid - stride] + values[tid]
    #
    #     # Down-sweep phase (reverse)
    #     ...
    #
    # For now, use sequential fallback
    return associative_scan_sequential(decay, values, initial)
