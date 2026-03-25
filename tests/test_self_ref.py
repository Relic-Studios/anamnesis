"""Tests for Self-Referential Projections."""

import pytest
import torch
import torch.nn as nn

from anamnesis.core.self_ref import AdaptiveProjection, SelfReferentialAttention


class TestAdaptiveProjection:
    """Tests for individual adaptive projections."""

    def test_forward_shape(self):
        proj = AdaptiveProjection(64, 128)
        x = torch.randn(2, 16, 64)
        y = proj(x)
        assert y.shape == (2, 16, 128)

    def test_from_linear_preserves_initial_behavior(self):
        """When initialized from a linear layer, output should approximately match."""
        linear = nn.Linear(64, 128, bias=True)
        proj = AdaptiveProjection.from_linear(linear)

        x = torch.randn(2, 16, 64)
        with torch.no_grad():
            y_linear = linear(x)
            y_proj = proj(x)

        # Should be close (memory initialized near zero, gate biased toward 0)
        assert torch.allclose(y_linear, y_proj, atol=0.5), (
            f"Max diff: {(y_linear - y_proj).abs().max().item()}"
        )

    def test_gradient_flow(self):
        proj = AdaptiveProjection(64, 128)
        x = torch.randn(2, 16, 64, requires_grad=True)
        y = proj(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None

        # Memory parameters should have gradients
        for name, p in proj.memory.named_parameters():
            assert p.grad is not None, f"No grad for {name}"

    def test_gated_vs_ungated(self):
        gated = AdaptiveProjection(64, 128, gate_output=True)
        ungated = AdaptiveProjection(64, 128, gate_output=False)
        x = torch.randn(2, 16, 64)
        y_g = gated(x)
        y_u = ungated(x)
        assert y_g.shape == y_u.shape


class TestSelfReferentialAttention:
    """Tests for the full self-referential attention module."""

    def test_forward_shape(self):
        attn = SelfReferentialAttention(dim=64, num_heads=4, num_kv_heads=2, head_dim=16)
        x = torch.randn(2, 16, 64)
        out, lr, decay = attn(x)
        assert out.shape == (2, 16, 64)
        assert lr.shape == (2, 16, 2)   # num_kv_heads
        assert decay.shape == (2, 16, 2)

    def test_lr_and_decay_in_range(self):
        """Learning rate and decay should be in (0, 1) after sigmoid."""
        attn = SelfReferentialAttention(dim=64, num_heads=4, num_kv_heads=2, head_dim=16)
        x = torch.randn(2, 16, 64)
        _, lr, decay = attn(x)
        assert (lr >= 0).all() and (lr <= 1).all()
        assert (decay >= 0).all() and (decay <= 1).all()

    def test_gqa_expansion(self):
        """GQA should work with fewer KV heads than Q heads."""
        attn = SelfReferentialAttention(dim=128, num_heads=8, num_kv_heads=2, head_dim=16)
        x = torch.randn(2, 16, 128)
        out, _, _ = attn(x)
        assert out.shape == (2, 16, 128)

    def test_gradient_flow(self):
        attn = SelfReferentialAttention(dim=64, num_heads=4, num_kv_heads=2, head_dim=16)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out, lr, decay = attn(x)
        loss = out.sum() + lr.sum() + decay.sum()
        loss.backward()
        assert x.grad is not None

        # Adaptive K/V projections should have gradients
        for name, p in attn.k_proj.named_parameters():
            assert p.grad is not None, f"No grad for k_proj.{name}"
        for name, p in attn.v_proj.named_parameters():
            assert p.grad is not None, f"No grad for v_proj.{name}"

    def test_from_standard_attention(self):
        """Converting standard attention should preserve initial behavior."""
        q = nn.Linear(64, 64, bias=True)
        k = nn.Linear(64, 32, bias=True)
        v = nn.Linear(64, 32, bias=True)
        o = nn.Linear(64, 64, bias=False)

        attn = SelfReferentialAttention.from_standard_attention(
            q_proj=q, k_proj=k, v_proj=v, o_proj=o,
            num_heads=4, num_kv_heads=2, head_dim=16,
        )
        x = torch.randn(2, 16, 64)
        out, lr, decay = attn(x)
        assert out.shape == (2, 16, 64)
