"""Tests for Neural Memory with gradient-based learning."""

import pytest
import torch
import torch.nn as nn

from anamnesis.core.memory import NeuralMemory, MemoryMLP, MemoryState, _memory_loss_fn
from torch.func import grad, vmap


class TestMemoryMLP:
    """Tests for the memory MLP module."""

    def test_forward_shape(self):
        mlp = MemoryMLP(dim=32, depth=2, expansion=2.0)
        x = torch.randn(4, 16, 32)
        y = mlp(x)
        assert y.shape == x.shape

    def test_residual(self):
        mlp = MemoryMLP(dim=32, depth=2)
        # Zero init the last layer — should give identity
        with torch.no_grad():
            for p in mlp.net[-1].parameters():
                p.zero_()
        x = torch.randn(4, 16, 32)
        y = mlp(x)
        assert torch.allclose(y, x, atol=1e-6)

    def test_gradient_flow(self):
        mlp = MemoryMLP(dim=32, depth=2)
        x = torch.randn(4, 32, requires_grad=True)
        y = mlp(x)
        y.sum().backward()
        assert x.grad is not None


class TestMemoryLossFn:
    """Tests for the associative memory loss function."""

    def test_basic_loss(self):
        mlp = MemoryMLP(dim=16, depth=1)
        params = dict(mlp.named_parameters())
        keys = torch.randn(16)
        values = torch.randn(16)
        loss = _memory_loss_fn(params, mlp, keys, values)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_grad_computable(self):
        """torch.func.grad should work on the loss function."""
        mlp = MemoryMLP(dim=16, depth=1)
        params = dict(mlp.named_parameters())
        keys = torch.randn(16)
        values = torch.randn(16)

        grad_fn = grad(_memory_loss_fn, argnums=0)
        grads = grad_fn(params, mlp, keys, values)

        assert len(grads) > 0
        for name, g in grads.items():
            assert g.shape == params[name].shape, f"Grad shape mismatch for {name}"

    def test_vmap_grad(self):
        """vmap over grad should produce per-sample gradients."""
        mlp = MemoryMLP(dim=16, depth=1)
        params = dict(mlp.named_parameters())

        batch_size = 4
        keys = torch.randn(batch_size, 16)
        values = torch.randn(batch_size, 16)

        # Expand params for batching
        batched_params = {
            name: p.unsqueeze(0).expand(batch_size, *p.shape)
            for name, p in params.items()
        }

        grad_fn = grad(_memory_loss_fn, argnums=0)
        batched_grad_fn = vmap(grad_fn, in_dims=(0, None, 0, 0))
        per_sample_grads = batched_grad_fn(batched_params, mlp, keys, values)

        for name, g in per_sample_grads.items():
            assert g.shape[0] == batch_size, f"Batch dim wrong for {name}"
            # Each sample should produce different gradients
            assert not torch.allclose(g[0], g[1], atol=1e-4), (
                f"Samples 0 and 1 have identical gradients for {name}"
            )


class TestNeuralMemory:
    """Tests for the full NeuralMemory module."""

    def test_forward_shape(self):
        mem = NeuralMemory(dim=32, num_heads=2, chunk_size=8)
        x = torch.randn(2, 16, 32)
        out, state = mem(x)
        assert out.shape == x.shape

    def test_state_returned(self):
        mem = NeuralMemory(dim=32, num_heads=2, chunk_size=8)
        x = torch.randn(2, 16, 32)
        out, state = mem(x)
        assert isinstance(state, MemoryState)
        assert state.seq_index == 16
        assert len(state.weights) == 2  # 2 heads

    def test_state_persistence(self):
        """Memory state from one call should be usable in the next."""
        mem = NeuralMemory(dim=32, num_heads=1, chunk_size=8)
        x1 = torch.randn(2, 8, 32)
        x2 = torch.randn(2, 8, 32)

        _, state1 = mem(x1)
        assert state1.seq_index == 8

        out2, state2 = mem(x2, state=state1)
        assert state2.seq_index == 16

    def test_memory_updates_weights(self):
        """After processing tokens, memory weights should have changed."""
        mem = NeuralMemory(dim=32, num_heads=1, chunk_size=4)
        x = torch.randn(1, 8, 32)  # 2 chunks

        # Get initial weights
        initial_params = {
            name: p.clone()
            for name, p in mem.memory_heads[0].named_parameters()
        }

        _, state = mem(x)

        # The state should contain updated weights that differ from initial
        head_weights = state.weights.get("head_0", {})
        if head_weights:
            for name, p_new in head_weights.items():
                p_old = initial_params.get(name)
                if p_old is not None:
                    # p_new is (batch, *param_shape), p_old is (*param_shape)
                    assert not torch.allclose(p_new[0], p_old, atol=1e-4), (
                        f"Weight {name} didn't change after memory update"
                    )

    def test_different_inputs_different_updates(self):
        """Different inputs should produce different memory updates."""
        mem = NeuralMemory(dim=32, num_heads=1, chunk_size=8)

        x1 = torch.randn(1, 8, 32)
        x2 = torch.randn(1, 8, 32) * 5  # very different

        _, state1 = mem(x1)
        _, state2 = mem(x2)

        # States should differ
        for key in state1.weights:
            w1 = state1.weights[key]
            w2 = state2.weights[key]
            for name in w1:
                assert not torch.allclose(w1[name], w2[name], atol=1e-3), (
                    f"Different inputs produced same weights for {key}.{name}"
                )

    def test_multi_head(self):
        """Multiple heads should have independent memory states."""
        mem = NeuralMemory(dim=64, num_heads=4, chunk_size=8)
        x = torch.randn(2, 16, 64)
        out, state = mem(x)
        assert out.shape == (2, 16, 64)
        assert len(state.weights) == 4

    def test_chunk_processing(self):
        """Results should be valid with different chunk sizes."""
        x = torch.randn(2, 32, 32)

        mem_small = NeuralMemory(dim=32, num_heads=1, chunk_size=8)
        mem_large = NeuralMemory(dim=32, num_heads=1, chunk_size=32)

        # Both should produce valid output (shapes match)
        out_s, _ = mem_small(x)
        out_l, _ = mem_large(x)
        assert out_s.shape == out_l.shape == x.shape

    def test_gradient_flow_through_memory(self):
        """Outer loop gradients should flow through retrieval path."""
        mem = NeuralMemory(dim=32, num_heads=1, chunk_size=16)
        x = torch.randn(2, 16, 32, requires_grad=True)
        out, _ = mem(x)
        loss = out.sum()
        loss.backward()

        # Input should receive gradients through the retrieval path
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

        # Query projection feeds the retrieval path → should have gradients
        assert mem.to_q.weight.grad is not None
        # Output projection is in the retrieval path → should have gradients
        assert mem.out_proj.weight.grad is not None

        # Note: to_k and to_v feed the INNER loop (vmap+grad scope),
        # which is detached from the outer autograd graph by design.
        # Their gradients come from the outer loop during training
        # (meta-learning the initial state), not from retrieval.
