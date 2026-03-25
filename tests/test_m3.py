"""Tests for M3 optimizer and Newton-Schulz orthogonalization."""

import pytest
import torch
import torch.nn as nn

from hope.optim.m3 import M3
from hope.optim.newton_schulz import newton_schulz


class TestNewtonSchulz:
    """Tests for Newton-Schulz orthogonalization."""

    def test_output_shape(self):
        G = torch.randn(64, 32)
        O = newton_schulz(G, steps=5)
        assert O.shape == G.shape

    def test_approximate_orthogonality(self):
        """Output rows should have similar norms (approximately orthogonal)."""
        G = torch.randn(32, 32)
        O = newton_schulz(G, steps=5)
        gram = O @ O.mT
        # NS produces "something like" orthogonal — diagonal values in ~[0.5, 1.5]
        # Off-diagonal should be smaller than diagonal
        diag = gram.diag()
        off_diag = gram - torch.diag(diag)
        assert diag.mean() > off_diag.abs().mean(), (
            "Diagonal should dominate off-diagonal in Gram matrix"
        )

    def test_preserves_dtype(self):
        G = torch.randn(16, 16, dtype=torch.float32)
        O = newton_schulz(G, steps=3, dtype=torch.float32)
        assert O.dtype == torch.float32

    def test_zero_steps_returns_normalized(self):
        G = torch.randn(16, 16)
        O = newton_schulz(G, steps=0)
        # With 0 steps, just normalization
        expected = G / (G.norm() + 1e-7)
        assert torch.allclose(O, expected, atol=1e-5)

    def test_deterministic(self):
        G = torch.randn(16, 16)
        O1 = newton_schulz(G, steps=5)
        O2 = newton_schulz(G, steps=5)
        assert torch.allclose(O1, O2)


class TestM3Optimizer:
    """Tests for the M3 optimizer."""

    @pytest.fixture
    def simple_model(self):
        """A small model with both 2D and 1D parameters."""
        model = nn.Sequential(
            nn.Linear(16, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 8, bias=True),
        )
        return model

    def test_step_runs(self, simple_model):
        optimizer = M3(simple_model.parameters(), lr=0.01)
        x = torch.randn(4, 16)
        y = simple_model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def test_loss_decreases(self, simple_model):
        """M3 should decrease loss on a simple problem."""
        target = torch.randn(4, 8)
        optimizer = M3(simple_model.parameters(), lr=0.01, adam_lr=0.01)

        losses = []
        for _ in range(50):
            x = torch.randn(4, 16)
            y = simple_model(x)
            loss = nn.functional.mse_loss(y, target)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Loss should generally decrease (allow some noise)
        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_weight_decay(self, simple_model):
        """With weight decay, parameter norms should shrink."""
        optimizer = M3(simple_model.parameters(), lr=0.0, weight_decay=0.1)
        initial_norm = sum(p.norm().item() for p in simple_model.parameters())

        # Run a few steps with zero gradient (only weight decay acts)
        for _ in range(10):
            x = torch.randn(4, 16)
            y = simple_model(x)
            loss = y.sum()
            optimizer.zero_grad()
            loss.backward()
            # Zero out gradients manually to isolate weight decay
            for p in simple_model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            optimizer.step()

        final_norm = sum(p.norm().item() for p in simple_model.parameters())
        assert final_norm < initial_norm

    def test_slow_momentum_updates(self, simple_model):
        """Slow momentum should only update every slow_freq steps."""
        optimizer = M3(simple_model.parameters(), lr=0.01, slow_freq=5)

        for step in range(10):
            x = torch.randn(4, 16)
            y = simple_model(x)
            loss = y.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Check that slow momentum state exists
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.dim() >= 2:
                    state = optimizer.state[p]
                    assert "slow_momentum" in state
                    # After 10 steps with freq=5, slow momentum should have been updated twice
                    assert state["slow_momentum"].abs().sum() > 0

    def test_2d_vs_1d_paths(self, simple_model):
        """2D params should use M3 path, 1D params should use Adam path."""
        optimizer = M3(simple_model.parameters(), lr=0.01)

        x = torch.randn(4, 16)
        y = simple_model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state[p]
                if p.dim() >= 2:
                    assert "fast_momentum" in state, "2D param should use M3 path"
                else:
                    assert "exp_avg" in state, "1D param should use Adam path"
