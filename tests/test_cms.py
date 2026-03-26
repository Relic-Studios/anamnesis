"""Tests for Continuum Memory System (Titans-style inner-loop learning)."""

import pytest
import torch

from anamnesis.core.cms import CMSLevel, ContinuumMemorySystem, CMSVariant


class TestCMSLevel:
    """Tests for individual CMS levels."""

    def test_forward_shape(self):
        level = CMSLevel(dim=64, hidden_mult=4.0, chunk_size=1)
        x = torch.randn(2, 16, 64)
        y = level(x)
        assert y.shape == x.shape

    def test_swiglu_level_forward(self):
        """Level 0 (SwiGLU) should produce non-trivial output different from input."""
        level = CMSLevel(dim=64, hidden_mult=4.0, chunk_size=1, swiglu=True)
        x = torch.randn(2, 16, 64)
        y = level(x)
        assert y.shape == x.shape
        # SwiGLU output should differ from input (no residual gate)
        assert not torch.allclose(y, x, atol=1e-3)

    def test_residual_level_starts_near_identity(self):
        """Levels 1+ start with residual_gate = -10, so sigmoid ≈ 0, output ≈ input."""
        level = CMSLevel(dim=64, hidden_mult=4.0, chunk_size=32, swiglu=False)
        x = torch.randn(2, 16, 64)
        y = level(x)
        # residual_gate initialized to -10 → sigmoid(-10) ≈ 4.5e-5 → delta ≈ 0
        assert torch.allclose(y, x, atol=1e-2)

    def test_should_update(self):
        level = CMSLevel(dim=64, chunk_size=32)
        assert level.should_update(0) is True
        assert level.should_update(1) is False
        assert level.should_update(31) is False
        assert level.should_update(32) is True
        assert level.should_update(64) is True

    def test_drift_disabled_by_default(self):
        level = CMSLevel(dim=64, chunk_size=32)
        assert level.drift_enabled is False

    def test_drift_adds_noise_in_eval(self):
        level = CMSLevel(dim=64, chunk_size=32)
        level.drift_enabled = True
        level.eval()
        torch.manual_seed(42)
        x = torch.randn(2, 16, 64)
        y1 = level(x)
        torch.manual_seed(43)
        y2 = level(x)
        # With different seeds, drift noise should differ
        assert not torch.allclose(y1, y2, atol=1e-7)

    def test_no_drift_in_training(self):
        level = CMSLevel(dim=64, chunk_size=32)
        level.drift_enabled = True
        level.train()
        torch.manual_seed(42)
        x = torch.randn(2, 16, 64)
        y1 = level(x)
        torch.manual_seed(43)
        y2 = level(x)
        # In training mode, drift is disabled even if flag is set
        assert torch.allclose(y1, y2, atol=1e-6)

    def test_inner_loop_learning_updates_weights(self):
        """Predictive coding should modify MLP weights during inference."""
        level = CMSLevel(dim=32, hidden_mult=2.0, chunk_size=1, swiglu=False, lr=1e-3)
        level.learning_enabled = True
        level.eval()

        # Snapshot weights before
        up_before = level.up_proj.weight.data.clone()

        # Need seq_len > 1 for predictive coding (position t predicts t+1)
        x = torch.randn(2, 16, 32)
        with torch.no_grad():
            _ = level(x)

        # Weights should have been updated by predictive coding
        assert not torch.allclose(level.up_proj.weight.data, up_before, atol=1e-8), \
            "Predictive coding should update MLP weights"

    def test_learning_disabled_preserves_weights(self):
        """Disabling learning should prevent weight updates."""
        level = CMSLevel(dim=32, hidden_mult=2.0, chunk_size=1, swiglu=False)
        level.learning_enabled = False
        level.eval()

        up_before = level.up_proj.weight.data.clone()

        x = torch.randn(2, 16, 32)
        with torch.no_grad():
            _ = level(x)

        assert torch.allclose(level.up_proj.weight.data, up_before), \
            "With learning disabled, weights should not change"

    def test_surprise_property(self):
        """Surprise should reflect gradient accumulation state."""
        level = CMSLevel(dim=32, hidden_mult=2.0, chunk_size=128)
        level.learning_enabled = True
        assert level.surprise == 0.0  # No gradients yet

        level.eval()
        x = torch.randn(2, 16, 32)
        with torch.no_grad():
            _ = level(x)
        # With chunk_size=128 and only 15 prediction tokens, grads accumulate but don't apply
        # So surprise should be > 0
        assert level.surprise > 0.0

    def test_reset_state(self):
        level = CMSLevel(dim=32, hidden_mult=2.0, chunk_size=128)
        level.learning_enabled = True
        level.eval()
        x = torch.randn(2, 16, 32)
        with torch.no_grad():
            _ = level(x)
        assert level.surprise > 0.0

        level.reset_state()
        assert level.surprise == 0.0
        assert level._tokens_in_chunk == 0


class TestContinuumMemorySystem:
    """Tests for the full CMS module."""

    def test_forward_chain_shape(self):
        cms = ContinuumMemorySystem(
            dim=64, num_levels=4, chunk_sizes=[1, 8, 32, 128],
            variant=CMSVariant.NESTED,
        )
        x = torch.randn(2, 32, 64)
        y = cms(x)
        assert y.shape == x.shape

    def test_forward_independent_shape(self):
        cms = ContinuumMemorySystem(
            dim=64, num_levels=3, chunk_sizes=[1, 8, 32],
            variant=CMSVariant.INDEPENDENT,
        )
        x = torch.randn(2, 32, 64)
        y = cms(x)
        assert y.shape == x.shape

    def test_all_variants(self):
        for variant in CMSVariant:
            cms = ContinuumMemorySystem(
                dim=64, num_levels=3, chunk_sizes=[1, 8, 32],
                variant=variant,
            )
            x = torch.randn(2, 16, 64)
            y = cms(x)
            assert y.shape == x.shape, f"Failed for variant {variant}"

    def test_chunk_sizes_must_be_ascending(self):
        with pytest.raises(AssertionError):
            ContinuumMemorySystem(dim=64, num_levels=3, chunk_sizes=[32, 8, 1])

    def test_chunk_sizes_must_match_num_levels(self):
        with pytest.raises(AssertionError):
            ContinuumMemorySystem(dim=64, num_levels=3, chunk_sizes=[1, 8])

    def test_level_0_is_swiglu(self):
        """First level should be SwiGLU, rest should be residual."""
        cms = ContinuumMemorySystem(dim=64, num_levels=3, chunk_sizes=[1, 8, 32])
        assert cms.levels[0].swiglu is True
        assert cms.levels[1].swiglu is False
        assert cms.levels[2].swiglu is False

    def test_enable_learning(self):
        cms = ContinuumMemorySystem(dim=64, num_levels=3, chunk_sizes=[1, 8, 32])
        # Disable all
        cms.enable_learning(False)
        for level in cms.levels:
            assert level.learning_enabled is False
        # Enable specific levels
        cms.enable_learning(True, levels=[0, 2])
        assert cms.levels[0].learning_enabled is True
        assert cms.levels[1].learning_enabled is False
        assert cms.levels[2].learning_enabled is True

    def test_get_surprise(self):
        cms = ContinuumMemorySystem(dim=32, num_levels=3, chunk_sizes=[1, 128, 256])
        surprises = cms.get_surprise()
        assert len(surprises) == 3
        assert all(s == 0.0 for s in surprises)

    def test_gradient_flow(self):
        """Verify outer-loop gradients flow through the MLP computation path."""
        cms = ContinuumMemorySystem(
            dim=64, num_levels=3, chunk_sizes=[1, 8, 32],
            variant=CMSVariant.NESTED,
        )
        x = torch.randn(2, 16, 64, requires_grad=True)
        y = cms(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        # MLP weights should have gradients (they're in the forward computation path)
        # Inner-loop-only params (W_k, W_v, to_lr, to_decay) won't have outer-loop grads
        mlp_params = {"up_proj.weight", "down_proj.weight", "gate_proj.weight", "residual_gate"}
        for i, level in enumerate(cms.levels):
            for name, param in level.named_parameters():
                if name in mlp_params:
                    assert param.grad is not None, f"No gradient for level {i} param {name}"

    def test_enable_drift(self):
        cms = ContinuumMemorySystem(dim=64, num_levels=3, chunk_sizes=[1, 8, 32])
        cms.enable_drift(True)
        for level in cms.levels:
            assert level.drift_enabled is True
        cms.enable_drift(False)
        for level in cms.levels:
            assert level.drift_enabled is False

    def test_reset_learning_state(self):
        cms = ContinuumMemorySystem(dim=32, num_levels=2, chunk_sizes=[1, 128])
        cms.eval()
        x = torch.randn(2, 16, 32)
        with torch.no_grad():
            _ = cms(x)
        # Some levels should have accumulated state
        cms.reset_learning_state()
        for level in cms.levels:
            assert level.surprise == 0.0
            assert level._tokens_in_chunk == 0

    def test_default_chunk_sizes(self):
        cms = ContinuumMemorySystem(dim=64, num_levels=4)
        assert cms.chunk_sizes == [1, 32, 256, 2048]

    def test_default_chunk_sizes_fewer_levels(self):
        cms = ContinuumMemorySystem(dim=64, num_levels=2)
        assert cms.chunk_sizes == [1, 32]

    def test_tapered_hidden_mult(self):
        """Different hidden multipliers per level should work."""
        cms = ContinuumMemorySystem(
            dim=64, num_levels=3, chunk_sizes=[1, 8, 32],
            hidden_mult=[4.0, 2.0, 1.0],
        )
        assert cms.levels[0].hidden_dim == 256  # 64 * 4
        assert cms.levels[1].hidden_dim == 128  # 64 * 2
        assert cms.levels[2].hidden_dim == 64   # 64 * 1
        x = torch.randn(2, 16, 64)
        y = cms(x)
        assert y.shape == x.shape
