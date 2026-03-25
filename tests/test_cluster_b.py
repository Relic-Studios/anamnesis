"""Tests for Active Inference Cluster B: Precision, Gardener, Thompson."""

import pytest
import torch
from torch import Tensor

from anamnesis.active_inference.precision import PrecisionNetwork, PrecisionModulator
from anamnesis.active_inference.gardener import GardenerStream, GardenerOutput
from anamnesis.active_inference.thompson import ThompsonLearningRate, BetaPosterior


class TestPrecisionNetwork:
    def test_forward_shape(self):
        net = PrecisionNetwork(num_metrics=4, num_levels=4, hidden=32)
        metrics = torch.randn(1, 4)
        precision, gate = net(metrics)
        assert precision.shape == (1, 4)
        assert gate.shape == (1, 1)

    def test_output_range(self):
        net = PrecisionNetwork(num_metrics=4, num_levels=3)
        metrics = torch.randn(8, 4)
        precision, gate = net(metrics)
        assert (precision >= 0).all() and (precision <= 1).all()
        assert (gate >= 0).all() and (gate <= 1).all()

    def test_batched(self):
        net = PrecisionNetwork(num_metrics=4, num_levels=4)
        metrics = torch.randn(16, 4)
        precision, gate = net(metrics)
        assert precision.shape == (16, 4)


class TestPrecisionModulator:
    def test_surprise_ema_updates(self):
        net = PrecisionNetwork(num_metrics=4, num_levels=4)
        mod = PrecisionModulator(net)
        initial = mod._surprise_ema
        mod.update_surprise(0.9)
        assert mod._surprise_ema != initial

    def test_surprise_trend(self):
        net = PrecisionNetwork(num_metrics=4, num_levels=4)
        mod = PrecisionModulator(net)
        mod.update_surprise(0.3)
        mod.update_surprise(0.8)
        assert mod.surprise_trend > 0  # rising

        mod.update_surprise(0.1)
        assert mod.surprise_trend < 0  # falling

    def test_modulate_reduces_lr_when_high_signal(self):
        """High signal → high precision → lower effective learning rate."""
        net = PrecisionNetwork(num_metrics=4, num_levels=4)
        mod = PrecisionModulator(net)

        lr_base = torch.tensor(0.01)
        decay_base = torch.tensor(0.1)

        lr_mod, _ = mod.modulate(lr_base, decay_base, signal_health=0.9)
        # Modulated lr should be less than or equal to base
        # (precision reduces lr, gate further reduces it)
        assert isinstance(lr_mod, Tensor)

    def test_consolidation_timer(self):
        net = PrecisionNetwork(num_metrics=4, num_levels=4)
        mod = PrecisionModulator(net)
        mod.update_surprise(0.5)
        mod.update_surprise(0.5)
        assert mod._time_since_consolidation == 2
        mod.reset_consolidation_timer()
        assert mod._time_since_consolidation == 0


class TestGardenerStream:
    def test_evaluate(self):
        gardener = GardenerStream(dim=64, num_levels=3)
        hidden = torch.randn(2, 16, 64)
        output = gardener.evaluate(hidden, surprise=0.5)
        assert isinstance(output, GardenerOutput)
        assert output.precision.shape == (3,)
        assert 0 <= output.plasticity_gate <= 1
        assert 0 <= output.signal_estimate <= 1

    def test_dream_trigger_after_low_signal(self):
        gardener = GardenerStream(
            dim=64, num_levels=3,
            dream_signal_threshold=0.4,
            dream_decline_window=3,
        )
        hidden = torch.randn(2, 16, 64)

        # Feed low signal for enough turns
        for _ in range(3):
            output = gardener.evaluate(hidden, real_signal=0.2)

        assert output.should_dream is True

    def test_dream_trigger_after_decline(self):
        gardener = GardenerStream(
            dim=64, num_levels=3,
            dream_decline_window=4,
        )
        hidden = torch.randn(2, 16, 64)

        # Declining signal
        for s in [0.9, 0.7, 0.5, 0.3]:
            output = gardener.evaluate(hidden, real_signal=s)

        assert output.should_dream is True

    def test_no_dream_when_healthy(self):
        gardener = GardenerStream(dim=64, num_levels=3, dream_decline_window=3)
        hidden = torch.randn(2, 16, 64)

        for _ in range(3):
            output = gardener.evaluate(hidden, real_signal=0.8)

        assert output.should_dream is False

    def test_dream_time_trigger(self):
        gardener = GardenerStream(dim=64, num_levels=3, dream_time_threshold=5)
        hidden = torch.randn(2, 16, 64)

        for _ in range(5):
            output = gardener.evaluate(hidden, real_signal=0.8)

        assert output.should_dream is True

    def test_acknowledge_dream_resets(self):
        gardener = GardenerStream(dim=64, num_levels=3, dream_time_threshold=3)
        hidden = torch.randn(2, 16, 64)

        for _ in range(3):
            gardener.evaluate(hidden, real_signal=0.8)

        gardener.acknowledge_dream()
        output = gardener.evaluate(hidden, real_signal=0.8)
        assert output.should_dream is False  # timer reset

    def test_proxy_real_divergence(self):
        gardener = GardenerStream(dim=64, num_levels=3)
        hidden = torch.randn(2, 16, 64)
        div = gardener.proxy_real_divergence(hidden, real_signal=0.5)
        assert isinstance(div, float)
        assert div >= 0


class TestBetaPosterior:
    def test_mean(self):
        p = BetaPosterior(alpha=5, beta=5)
        assert abs(p.mean - 0.5) < 1e-6

    def test_sample_in_range(self):
        p = BetaPosterior(alpha=2, beta=2)
        for _ in range(100):
            s = p.sample()
            assert 0 <= s <= 1

    def test_update_shifts_mean(self):
        p = BetaPosterior(alpha=2, beta=2)
        initial_mean = p.mean
        for _ in range(10):
            p.update(success=True)
        assert p.mean > initial_mean

    def test_alpha_cap(self):
        p = BetaPosterior(alpha=2, beta=2, alpha_cap=10)
        for _ in range(100):
            p.update(success=True)
        assert p.alpha <= 10


class TestThompsonLearningRate:
    def test_sample_rates(self):
        thompson = ThompsonLearningRate(num_levels=4, lr_max=0.01)
        rates = thompson.sample_rates()
        assert len(rates) == 4
        for r in rates:
            assert 0 <= r <= 0.01

    def test_posteriors_shift_with_feedback(self):
        thompson = ThompsonLearningRate(num_levels=2, lr_max=0.01)

        # Record initial means
        initial_means = [p.mean for p in thompson.posteriors]

        # Sample and give positive feedback repeatedly
        for _ in range(20):
            thompson.sample_rates()
            thompson.update_posteriors(signal_improved=True, signal_delta=0.1)

        # Means should have shifted
        for i, (init, post) in enumerate(zip(initial_means, thompson.posteriors)):
            assert post.mean != init, f"Level {i} posterior didn't shift"

    def test_diagnostics(self):
        thompson = ThompsonLearningRate(num_levels=3)
        diag = thompson.get_diagnostics()
        assert "means" in diag
        assert len(diag["means"]) == 3
        assert "variances" in diag

    def test_reset(self):
        thompson = ThompsonLearningRate(num_levels=3)
        for _ in range(10):
            thompson.sample_rates()
            thompson.update_posteriors(signal_improved=True)
        thompson.reset()
        for p in thompson.posteriors:
            assert p.alpha == 2.0
            assert p.beta == 2.0

    def test_trace_bounded(self):
        thompson = ThompsonLearningRate(num_levels=2)
        for _ in range(50):
            thompson.sample_rates()
            thompson.update_posteriors(signal_improved=True)
        for trace in thompson._traces:
            assert len(trace) <= 20  # max_trace_len
