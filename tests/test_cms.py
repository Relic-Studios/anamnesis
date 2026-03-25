"""Tests for Continuum Memory System."""

import pytest
import torch

from hope.core.cms import CMSLevel, ContinuumMemorySystem, CMSVariant


class TestCMSLevel:
    """Tests for individual CMS levels."""

    def test_forward_shape(self):
        level = CMSLevel(dim=64, hidden_mult=4.0, chunk_size=1)
        x = torch.randn(2, 16, 64)
        y = level(x)
        assert y.shape == x.shape

    def test_residual_connection(self):
        """Output should be input + transform, so with zero-init transform, output ≈ input."""
        level = CMSLevel(dim=64, hidden_mult=4.0, chunk_size=1)
        # Zero out the down projection — residual should dominate
        with torch.no_grad():
            level.down_proj.weight.zero_()
        x = torch.randn(2, 16, 64)
        y = level(x)
        assert torch.allclose(y, x, atol=1e-6)

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

    def test_update_schedule(self):
        cms = ContinuumMemorySystem(
            dim=64, num_levels=3, chunk_sizes=[1, 4, 8],
        )
        schedule = cms.get_update_schedule(seq_len=8)
        # Position 0: all levels update (0 % 1 == 0, 0 % 4 == 0, 0 % 8 == 0)
        assert 0 in schedule
        assert set(schedule[0]) == {0, 1, 2}
        # Position 1: only level 0
        assert 1 in schedule
        assert schedule[1] == [0]
        # Position 4: levels 0 and 1
        assert 4 in schedule
        assert set(schedule[4]) == {0, 1}

    def test_gradient_flow(self):
        """Verify gradients flow through all CMS levels."""
        cms = ContinuumMemorySystem(
            dim=64, num_levels=3, chunk_sizes=[1, 8, 32],
            variant=CMSVariant.NESTED,
        )
        x = torch.randn(2, 16, 64, requires_grad=True)
        y = cms(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        # All level parameters should have gradients
        for i, level in enumerate(cms.levels):
            for name, param in level.named_parameters():
                assert param.grad is not None, f"No gradient for level {i} param {name}"

    def test_enable_drift(self):
        cms = ContinuumMemorySystem(dim=64, num_levels=3, chunk_sizes=[1, 8, 32])
        cms.enable_drift(True)
        for level in cms.levels:
            assert level.drift_enabled is True
        cms.enable_drift(False)
        for level in cms.levels:
            assert level.drift_enabled is False

    def test_from_pretrained_mlp(self):
        """Test initialization from pre-trained SwiGLU weights."""
        hidden = 64
        intermediate = 256
        gate = torch.randn(intermediate, hidden)
        up = torch.randn(intermediate, hidden)
        down = torch.randn(hidden, intermediate)

        cms = ContinuumMemorySystem.from_pretrained_mlp(
            gate_proj=gate, up_proj=up, down_proj=down,
            num_levels=3, chunk_sizes=[1, 8, 32],
        )
        assert cms.dim == hidden
        assert cms.num_levels == 3
        # Check weights were actually copied (not just zeros)
        for level in cms.levels:
            assert not torch.allclose(
                level.up_proj.weight, torch.zeros_like(level.up_proj.weight)
            )

    def test_default_chunk_sizes(self):
        cms = ContinuumMemorySystem(dim=64, num_levels=4)
        assert cms.chunk_sizes == [1, 32, 256, 2048]

    def test_default_chunk_sizes_fewer_levels(self):
        cms = ContinuumMemorySystem(dim=64, num_levels=2)
        assert cms.chunk_sizes == [1, 32]
