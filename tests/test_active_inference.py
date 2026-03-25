"""Tests for Active Inference extensions (Cluster A: signal loss + drift)."""

import pytest
import torch
import torch.nn as nn

from hope.active_inference.free_energy import (
    SignalProxy,
    SignalFreeEnergy,
    IdentityDrift,
    CompositeHopeLoss,
)
from hope.active_inference.drift import NeutralDrift
from hope.core.cms import ContinuumMemorySystem


class TestSignalProxy:
    """Tests for the differentiable signal proxy network."""

    def test_forward_shape(self):
        proxy = SignalProxy(dim=64)
        x = torch.randn(4, 16, 64)
        health = proxy(x)
        assert health.shape == (4,)

    def test_output_range(self):
        proxy = SignalProxy(dim=64)
        x = torch.randn(4, 16, 64)
        health = proxy(x)
        assert (health >= 0).all() and (health <= 1).all()

    def test_predict_facets(self):
        proxy = SignalProxy(dim=64, num_facets=5)
        x = torch.randn(4, 16, 64)
        facets = proxy.predict_facets(x)
        assert facets.shape == (4, 5)
        assert (facets >= 0).all() and (facets <= 1).all()

    def test_gradient_flow(self):
        proxy = SignalProxy(dim=64)
        x = torch.randn(4, 16, 64, requires_grad=True)
        health = proxy(x)
        health.sum().backward()
        assert x.grad is not None


class TestSignalFreeEnergy:
    """Tests for the signal free energy loss."""

    def test_precomputed_signal(self):
        sfe = SignalFreeEnergy()
        signal = torch.tensor([0.8, 0.5, 0.3])
        fe = sfe(precomputed_signal=signal)
        expected = 1.0 - signal
        assert torch.allclose(fe, expected)

    def test_proxy_signal(self):
        sfe = SignalFreeEnergy(dim=64, use_proxy=True)
        x = torch.randn(4, 16, 64)
        fe = sfe(hidden_states=x)
        assert fe.shape == (4,)
        assert (fe >= 0).all() and (fe <= 1).all()

    def test_requires_input(self):
        sfe = SignalFreeEnergy()
        with pytest.raises(ValueError):
            sfe()


class TestIdentityDrift:
    """Tests for identity drift penalty."""

    def test_zero_drift_when_identical(self):
        drift = IdentityDrift()
        params = [{"w": torch.randn(32, 32)}]
        soul = [{"w": params[0]["w"].clone()}]
        loss = drift(params, soul)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_positive_drift_when_different(self):
        drift = IdentityDrift()
        params = [{"w": torch.randn(32, 32)}]
        soul = [{"w": torch.randn(32, 32)}]
        loss = drift(params, soul)
        assert loss.item() > 0

    def test_slower_levels_penalized_more(self):
        """Slower levels (tighter prior) should incur more penalty for same deviation."""
        drift = IdentityDrift(sigma_per_level=[1.0, 0.01])  # fast=loose, slow=tight

        deviation = torch.randn(32, 32)
        base = torch.zeros(32, 32)

        # Same deviation at both levels
        params = [{"w": deviation.clone()}, {"w": deviation.clone()}]
        soul = [{"w": base.clone()}, {"w": base.clone()}]

        # Compute per-level
        fast_drift = IdentityDrift(sigma_per_level=[1.0])
        slow_drift = IdentityDrift(sigma_per_level=[0.01])

        fast_loss = fast_drift([params[0]], [soul[0]])
        slow_loss = slow_drift([params[1]], [soul[1]])

        assert slow_loss.item() > fast_loss.item() * 10, (
            "Slow levels should be penalized much more for same deviation"
        )


class TestCompositeHopeLoss:
    """Tests for the full composite loss."""

    def test_reconstruction_only(self):
        loss_fn = CompositeHopeLoss(lambda_recon=1.0, lambda_signal=0.0, lambda_identity=0.0)
        recon = torch.tensor(2.5)
        result = loss_fn(recon)
        assert torch.allclose(result["total"], recon)
        assert "signal" not in result
        assert "identity" not in result

    def test_with_precomputed_signal(self):
        loss_fn = CompositeHopeLoss(lambda_recon=1.0, lambda_signal=0.5)
        recon = torch.tensor(2.0)
        signal = torch.tensor([0.8, 0.6])
        result = loss_fn(recon, precomputed_signal=signal)
        assert "signal" in result
        assert result["total"] > recon  # signal penalty adds to loss

    def test_anneal_signal(self):
        loss_fn = CompositeHopeLoss(lambda_signal=0.0)
        assert loss_fn.lambda_signal == 0.0
        loss_fn.anneal_signal(target=0.5, step=0.1)
        assert loss_fn.lambda_signal == 0.1
        for _ in range(10):
            loss_fn.anneal_signal(target=0.5, step=0.1)
        assert loss_fn.lambda_signal == 0.5  # capped at target

    def test_signal_capping(self):
        """Signal weight should be capped at lambda_signal_max."""
        loss_fn = CompositeHopeLoss(lambda_signal=1.0, lambda_signal_max=0.1)
        recon = torch.tensor(0.0)
        signal = torch.tensor([0.0])  # worst signal → free energy = 1.0
        result = loss_fn(recon, precomputed_signal=signal)
        # With cap of 0.1, signal contribution should be at most 0.1
        assert result["total"].item() <= 0.15  # small tolerance

    def test_gradient_through_proxy(self):
        loss_fn = CompositeHopeLoss(
            lambda_recon=1.0, lambda_signal=0.5,
            dim=64, use_proxy=True,
        )
        recon = torch.tensor(1.0, requires_grad=True)
        hidden = torch.randn(2, 8, 64, requires_grad=True)
        result = loss_fn(recon, hidden_states=hidden)
        result["total"].backward()
        assert hidden.grad is not None


class TestNeutralDrift:
    """Tests for the neutral drift mechanism."""

    def test_disabled_by_default(self):
        drift = NeutralDrift()
        assert drift.enabled is False

    def test_no_effect_when_disabled(self):
        drift = NeutralDrift(enabled=False)
        params = {"w": torch.zeros(32, 32)}
        drift.apply(params)
        assert torch.allclose(params["w"], torch.zeros(32, 32))

    def test_adds_noise_when_enabled(self):
        drift = NeutralDrift(sigma_base=0.1, enabled=True)
        params = {"w": torch.zeros(32, 32)}
        drift.apply(params)
        assert not torch.allclose(params["w"], torch.zeros(32, 32))

    def test_noise_scales_with_frequency(self):
        """Slower levels (higher chunk_size) should get less noise."""
        drift = NeutralDrift(sigma_base=1.0, enabled=True, scale_by_frequency=True)

        torch.manual_seed(42)
        fast_params = {"w": torch.zeros(64, 64)}
        drift.apply(fast_params, chunk_size=1)
        fast_noise = fast_params["w"].abs().mean().item()

        torch.manual_seed(42)
        slow_params = {"w": torch.zeros(64, 64)}
        drift.apply(slow_params, chunk_size=100)
        slow_noise = slow_params["w"].abs().mean().item()

        assert slow_noise < fast_noise, (
            f"Slow noise ({slow_noise}) should be less than fast noise ({fast_noise})"
        )

    def test_plasticity_gate_modulation(self):
        """Plasticity gate of 0 should suppress all drift."""
        drift = NeutralDrift(sigma_base=1.0, enabled=True)
        params = {"w": torch.zeros(32, 32)}
        drift.apply(params, plasticity_gate=0.0)
        assert torch.allclose(params["w"], torch.zeros(32, 32))

    def test_apply_to_cms(self):
        drift = NeutralDrift(sigma_base=0.01, enabled=True)
        cms = ContinuumMemorySystem(dim=32, num_levels=3, chunk_sizes=[1, 8, 32])

        # Record initial state
        initial = {
            f"level_{i}": {name: p.clone() for name, p in level.named_parameters()}
            for i, level in enumerate(cms.levels)
        }

        drift.apply_to_cms(cms)

        # Check that parameters changed
        changed = False
        for i, level in enumerate(cms.levels):
            for name, p in level.named_parameters():
                if not torch.allclose(p, initial[f"level_{i}"][name], atol=1e-8):
                    changed = True
                    break
        assert changed, "CMS parameters should have changed after drift"

    def test_apply_to_module(self):
        drift = NeutralDrift(sigma_base=0.01, enabled=True)
        linear = nn.Linear(32, 32)
        initial = linear.weight.clone()
        drift.apply(linear)
        assert not torch.allclose(linear.weight, initial, atol=1e-8)
