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

import math

import torch
from torch import Tensor

from anamnesis.kernels import _TRITON_AVAILABLE

if _TRITON_AVAILABLE:
    import triton
    import triton.language as tl


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
        decay: Decay factors (batch, seq_len, dim).
        values: Input values (batch, seq_len, dim).
        initial: Initial state (batch, dim).
        use_triton: Whether to attempt Triton kernel.

    Returns:
        Scanned output (batch, seq_len, dim).
    """
    if use_triton and _TRITON_AVAILABLE and decay.is_cuda and decay.dim() == 3:
        return _associative_scan_triton(decay, values, initial)

    return associative_scan_sequential(decay, values, initial)


if _TRITON_AVAILABLE:
    @triton.jit
    def _assoc_scan_up_kernel(
        decay_ptr, values_ptr, out_decay_ptr, out_values_ptr,
        batch, seq_len, dim,
        stride_b, stride_s, stride_d,
        step: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """Up-sweep phase: combine pairs at increasing stride."""
        pid_b = tl.program_id(0)
        pid_s = tl.program_id(1)
        pid_d = tl.program_id(2)

        # Which pair are we combining?
        pair_stride = 1 << step  # 2^step
        right_idx = (pid_s + 1) * (pair_stride * 2) - 1
        left_idx = right_idx - pair_stride

        if right_idx >= seq_len or left_idx < 0:
            return

        d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = d_off < dim

        # Load left and right (decay, value) pairs
        left_base = pid_b * stride_b + left_idx * stride_s
        right_base = pid_b * stride_b + right_idx * stride_s

        d_left = tl.load(out_decay_ptr + left_base + d_off * stride_d, mask=mask)
        v_left = tl.load(out_values_ptr + left_base + d_off * stride_d, mask=mask)
        d_right = tl.load(out_decay_ptr + right_base + d_off * stride_d, mask=mask)
        v_right = tl.load(out_values_ptr + right_base + d_off * stride_d, mask=mask)

        # Combine: (d_r, v_r) ∘ (d_l, v_l) = (d_r * d_l, d_r * v_l + v_r)
        new_decay = d_right * d_left
        new_value = d_right * v_left + v_right

        tl.store(out_decay_ptr + right_base + d_off * stride_d, new_decay, mask=mask)
        tl.store(out_values_ptr + right_base + d_off * stride_d, new_value, mask=mask)

    @triton.jit
    def _assoc_scan_down_kernel(
        out_decay_ptr, out_values_ptr,
        batch, seq_len, dim,
        stride_b, stride_s, stride_d,
        step: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """Down-sweep phase: propagate results at decreasing stride."""
        pid_b = tl.program_id(0)
        pid_s = tl.program_id(1)
        pid_d = tl.program_id(2)

        pair_stride = 1 << step
        # Target: elements at positions that need propagation
        left_idx = (pid_s + 1) * (pair_stride * 2) - 1
        right_idx = left_idx + pair_stride

        if right_idx >= seq_len or left_idx < 0:
            return

        d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = d_off < dim

        left_base = pid_b * stride_b + left_idx * stride_s
        right_base = pid_b * stride_b + right_idx * stride_s

        d_left = tl.load(out_decay_ptr + left_base + d_off * stride_d, mask=mask)
        v_left = tl.load(out_values_ptr + left_base + d_off * stride_d, mask=mask)
        d_right = tl.load(out_decay_ptr + right_base + d_off * stride_d, mask=mask)
        v_right = tl.load(out_values_ptr + right_base + d_off * stride_d, mask=mask)

        new_decay = d_right * d_left
        new_value = d_right * v_left + v_right

        tl.store(out_decay_ptr + right_base + d_off * stride_d, new_decay, mask=mask)
        tl.store(out_values_ptr + right_base + d_off * stride_d, new_value, mask=mask)


def _associative_scan_triton(
    decay: Tensor,
    values: Tensor,
    initial: Tensor | None = None,
) -> Tensor:
    """
    Triton-accelerated parallel associative scan using Blelloch algorithm.

    Args:
        decay: (batch, seq_len, dim)
        values: (batch, seq_len, dim)
        initial: (batch, dim) or None

    Returns:
        output: (batch, seq_len, dim)
    """
    batch, seq_len, dim = decay.shape

    # Incorporate initial state into first position
    if initial is not None:
        values = values.clone()
        values[:, 0] = decay[:, 0] * initial + values[:, 0]

    # For very short sequences, just use sequential
    if seq_len <= 2:
        return associative_scan_sequential(decay, values, initial if initial is not None and seq_len > 0 else None)

    # Work on contiguous copies
    out_decay = decay.contiguous().clone()
    out_values = values.contiguous().clone()

    stride_b = out_decay.stride(0)
    stride_s = out_decay.stride(1)
    stride_d = out_decay.stride(2)

    BLOCK_D = triton.next_power_of_2(dim)
    if BLOCK_D > 1024:
        BLOCK_D = 1024

    num_steps = int(math.ceil(math.log2(seq_len)))
    num_d_blocks = (dim + BLOCK_D - 1) // BLOCK_D

    # Up-sweep
    for step in range(num_steps):
        pair_stride = 1 << step
        num_pairs = seq_len // (pair_stride * 2)
        if num_pairs == 0:
            break
        grid = (batch, num_pairs, num_d_blocks)
        _assoc_scan_up_kernel[grid](
            out_decay, out_values, out_decay, out_values,
            batch, seq_len, dim,
            stride_b, stride_s, stride_d,
            step=step, BLOCK_D=BLOCK_D,
        )

    # Down-sweep
    for step in range(num_steps - 2, -1, -1):
        pair_stride = 1 << step
        num_pairs = seq_len // (pair_stride * 2) - 1
        if num_pairs <= 0:
            continue
        grid = (batch, num_pairs, num_d_blocks)
        _assoc_scan_down_kernel[grid](
            out_decay, out_values,
            batch, seq_len, dim,
            stride_b, stride_s, stride_d,
            step=step, BLOCK_D=BLOCK_D,
        )

    return out_values
