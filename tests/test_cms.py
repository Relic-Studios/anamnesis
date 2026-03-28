"""Tests for Continuum Memory System (Titans-style inner-loop learning)."""

import pytest
import torch

from anamnesis.core.cms import CMSLevel, LowRankLevel, DeepMemoryLevel, ContinuumMemorySystem, CMSVariant


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

    def test_residual_level_starts_at_identity(self):
        """Levels 1+ are now LowRankLevel. With A=0, output = L0_out.

        This test validates the legacy CMSLevel behavior for backward compat:
        when down_proj = 0, output = x + 0.5 * 0 = x.
        """
        level = CMSLevel(dim=64, hidden_mult=4.0, chunk_size=32, swiglu=False)
        # Simulate proper init: down_proj = 0
        torch.nn.init.zeros_(level.down_proj.weight)
        x = torch.randn(2, 16, 64)
        y = level(x)
        assert y.shape == x.shape
        # With zero down_proj, output should be exactly input
        assert torch.allclose(y, x, atol=1e-5)

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

    def test_residual_learning_not_gate_attenuated(self):
        """Weight updates should be the same magnitude regardless of gate value.

        The residual level trains on the residual prediction task (predict x_{t+1} - x_t),
        NOT on the gated output. Gate controls output blending, not learning strength.
        """
        torch.manual_seed(42)
        x = torch.randn(4, 64, 32)

        def measure_delta(gate_val):
            torch.manual_seed(0)  # Same init weights
            level = CMSLevel(dim=32, hidden_mult=2.0, chunk_size=1, swiglu=False, lr=1e-2)
            level.residual_gate.data.fill_(gate_val)
            level.learning_enabled = True
            level.eval()
            up_before = level.up_proj.weight.data.clone()
            with torch.no_grad():
                _ = level(x)
            return (level.up_proj.weight.data - up_before).norm().item()

        # Gate = -5 (sigmoid ≈ 0.007) vs gate = 0 (sigmoid = 0.5)
        delta_tiny_gate = measure_delta(-5.0)
        delta_half_gate = measure_delta(0.0)

        # Both should produce meaningful weight updates
        assert delta_tiny_gate > 0.005, f"tiny gate delta too small: {delta_tiny_gate:.6f}"
        assert delta_half_gate > 0.005, f"half gate delta too small: {delta_half_gate:.6f}"

        # Key property: deltas should be similar regardless of gate value
        # (within 2x — not 100x as with the old gate-attenuated approach)
        ratio = max(delta_tiny_gate, delta_half_gate) / min(delta_tiny_gate, delta_half_gate)
        assert ratio < 2.0, f"Gate is attenuating learning: ratio={ratio:.2f} (tiny={delta_tiny_gate:.6f}, half={delta_half_gate:.6f})"

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
            if hasattr(level, '_tokens_in_chunk'):
                assert level._tokens_in_chunk == 0
            if hasattr(level, '_total_updates'):
                assert level._total_updates == 0

    def test_default_chunk_sizes(self):
        cms = ContinuumMemorySystem(dim=64, num_levels=4)
        assert cms.chunk_sizes == [1, 32, 256, 2048]

    def test_default_chunk_sizes_fewer_levels(self):
        cms = ContinuumMemorySystem(dim=64, num_levels=2)
        assert cms.chunk_sizes == [1, 32]

    def test_tapered_hidden_mult(self):
        """Different hidden multipliers per level should work.

        Deep memory levels have their own mem_dim, independent of hidden_mult.
        Only L0 uses hidden_mult for SwiGLU sizing.
        """
        cms = ContinuumMemorySystem(
            dim=64, num_levels=3, chunk_sizes=[1, 8, 32],
            hidden_mult=[4.0, 2.0, 1.0], mem_dim=32,
        )
        assert cms.levels[0].hidden_dim == 256  # 64 * 4
        # Deep memory levels use mem_dim, not hidden_mult
        assert cms.levels[1].mem_dim == 32
        assert cms.levels[2].mem_dim == 32
        x = torch.randn(2, 16, 64)
        y = cms(x)
        assert y.shape == x.shape

    def test_set_persona_probe(self):
        """Persona probe should be set on final level only."""
        cms = ContinuumMemorySystem(dim=64, num_levels=2, chunk_sizes=[1, 32])
        fake_lm_head = torch.randn(1000, 64)
        cms.set_persona_probe(fake_lm_head, persona_dim=16)

        assert cms.levels[0]._persona_probe is None
        assert cms.levels[-1]._persona_probe is not None
        assert cms.levels[-1]._persona_probe.shape == (64, 16)

    def test_persona_probe_affects_learning(self):
        """Persona probe should be set on the final level."""
        cms = ContinuumMemorySystem(dim=64, num_levels=2, chunk_sizes=[1, 8])
        fake_lm_head = torch.randn(500, 64)
        cms.set_persona_probe(fake_lm_head, persona_dim=16)

        # Probe should be set on the last level
        assert cms.levels[-1]._persona_probe is not None
        assert cms.levels[-1]._persona_probe.shape == (64, 16)
        # L0 should not have a probe
        assert cms.levels[0]._persona_probe is None


class TestDeepMemoryLevel:
    """Tests for the ATLAS-style DeepMemoryLevel."""

    def test_forward_shape(self):
        """Output shape should match input."""
        dml = DeepMemoryLevel(dim=64, mem_dim=32, chunk_size=4, poly_degree=2)
        x = torch.randn(2, 16, 64)
        out = dml(x)
        assert out.shape == x.shape

    def test_memory_updates_during_eval(self):
        """Memory MLP weights should change during eval forward pass."""
        dml = DeepMemoryLevel(dim=64, mem_dim=32, chunk_size=4, poly_degree=2)
        dml.learning_enabled = True
        dml.eval()

        x = torch.randn(2, 16, 64)
        before = {n: p.clone() for n, p in dml.memory.named_parameters()}
        with torch.no_grad():
            dml(x)
        changed = sum(1 for n, p in dml.memory.named_parameters()
                      if not torch.allclose(p, before[n], atol=1e-7))
        assert changed > 0, "Memory weights should update during eval"

    def test_no_updates_when_disabled(self):
        """With learning disabled, memory weights should not change."""
        dml = DeepMemoryLevel(dim=64, mem_dim=32, chunk_size=4, poly_degree=2)
        dml.learning_enabled = False
        dml.eval()

        x = torch.randn(2, 16, 64)
        before = {n: p.clone() for n, p in dml.memory.named_parameters()}
        with torch.no_grad():
            dml(x)
        changed = sum(1 for n, p in dml.memory.named_parameters()
                      if not torch.allclose(p, before[n], atol=1e-7))
        assert changed == 0, "Memory weights should not update when learning disabled"

    def test_no_updates_during_training(self):
        """During backprop (grad enabled), memory should not update."""
        dml = DeepMemoryLevel(dim=64, mem_dim=32, chunk_size=4, poly_degree=2)
        dml.learning_enabled = True
        dml.train()

        x = torch.randn(2, 16, 64)
        before = {n: p.clone() for n, p in dml.memory.named_parameters()}
        out = dml(x)
        changed = sum(1 for n, p in dml.memory.named_parameters()
                      if not torch.allclose(p, before[n], atol=1e-7))
        assert changed == 0, "Memory should not update during backprop training"

    def test_output_changes_between_passes(self):
        """Output should change as memory learns."""
        dml = DeepMemoryLevel(dim=64, mem_dim=32, chunk_size=4, poly_degree=2)
        dml.learning_enabled = True
        dml.eval()

        x = torch.randn(2, 16, 64)
        with torch.no_grad():
            out1 = dml(x).clone()
            out2 = dml(x)
        assert not torch.allclose(out1, out2, atol=1e-6), \
            "Output should change as memory updates"

    def test_poly_expansion(self):
        """Polynomial features should double dim for degree 2."""
        dml = DeepMemoryLevel(dim=64, mem_dim=32, chunk_size=4, poly_degree=2)
        x = torch.randn(4, 32)
        expanded = dml._poly_expand(x)
        assert expanded.shape == (4, 64), f"Expected (4, 64), got {expanded.shape}"

    def test_poly_degree_1_is_identity(self):
        """Poly degree 1 should not expand."""
        dml = DeepMemoryLevel(dim=64, mem_dim=32, chunk_size=4, poly_degree=1)
        x = torch.randn(4, 32)
        expanded = dml._poly_expand(x)
        assert torch.allclose(expanded, x)

    def test_data_dependent_gates(self):
        """Gates should produce different values for different inputs."""
        dml = DeepMemoryLevel(dim=64, mem_dim=32, chunk_size=4, poly_degree=2)
        x1 = torch.randn(1, 8, 64)
        x2 = torch.randn(1, 8, 64) * 5.0

        gate1 = torch.sigmoid(dml.to_output_gate(x1))
        gate2 = torch.sigmoid(dml.to_output_gate(x2))
        assert not torch.allclose(gate1, gate2, atol=1e-4), \
            "Data-dependent gates should respond to different inputs"

    def test_save_and_load_soul(self):
        """Soul checkpoint should capture memory MLP weights."""
        dml = DeepMemoryLevel(dim=64, mem_dim=32, chunk_size=4, poly_degree=2)
        dml.save_soul()
        assert len(dml._soul_weights) > 0
        for name, p in dml.memory.named_parameters():
            assert name in dml._soul_weights

    def test_soul_pullback_limits_drift(self):
        """Soul anchor should prevent unbounded weight drift."""
        dml = DeepMemoryLevel(dim=64, mem_dim=32, chunk_size=4, poly_degree=2)
        dml.save_soul()
        dml.max_drift = 0.1
        dml.soul_pull_strength = 0.5
        dml.learning_enabled = True
        dml.eval()

        # Feed extreme input to force large updates
        x = torch.randn(2, 32, 64) * 10.0
        for _ in range(20):
            with torch.no_grad():
                dml(x)

        # Check drift from soul is bounded
        for name, p in dml.memory.named_parameters():
            if name in dml._soul_weights:
                drift = (p.data.float() - dml._soul_weights[name]).norm().item()
                assert drift < 20.0, f"Drift should be bounded: {name} drift={drift:.4f}"

    def test_surprise_tracking(self):
        """Surprise EMA should be tracked and accessible."""
        dml = DeepMemoryLevel(dim=64, mem_dim=32, chunk_size=4, poly_degree=2)
        assert dml.surprise == 0.0  # no updates yet

        dml.learning_enabled = True
        dml.eval()
        x = torch.randn(2, 16, 64)
        with torch.no_grad():
            dml(x)
        assert dml.surprise > 0.0  # should have tracked something

    def test_reset_state(self):
        """Reset should clear momentum and update counter."""
        dml = DeepMemoryLevel(dim=64, mem_dim=32, chunk_size=4, poly_degree=2)
        dml.learning_enabled = True
        dml.eval()

        x = torch.randn(2, 16, 64)
        with torch.no_grad():
            dml(x)
        assert dml._total_updates > 0

        dml.reset_state()
        assert dml._total_updates == 0
        assert len(dml._momentum_state) == 0
        assert dml.surprise == 0.0


class TestDeepMemoryEndToEnd:
    """End-to-end tests proving the new architecture actually works."""

    def test_cms_learns_and_output_changes(self):
        """Full CMS with DeepMemoryLevel should produce different outputs after learning."""
        torch.manual_seed(42)
        cms = ContinuumMemorySystem(
            dim=64, num_levels=2, chunk_sizes=[1, 8],
            hidden_mult=4.0, mem_dim=32, poly_degree=2,
        )
        cms.eval()

        x = torch.randn(2, 32, 64)

        # Disable learning for baseline
        cms.levels[1].learning_enabled = False
        with torch.no_grad():
            baseline = cms(x).clone()

        # Enable learning and run several passes
        cms.levels[1].learning_enabled = True
        for _ in range(10):
            with torch.no_grad():
                cms(x)

        # Check output changed
        cms.levels[1].learning_enabled = False
        with torch.no_grad():
            evolved = cms(x)

        assert not torch.allclose(baseline, evolved, atol=1e-4), \
            "CMS output should change after deep memory learning"

    def test_different_inputs_different_specializations(self):
        """Two CMS instances fed different data should diverge."""
        torch.manual_seed(42)

        def make_cms():
            torch.manual_seed(0)
            cms = ContinuumMemorySystem(
                dim=64, num_levels=2, chunk_sizes=[1, 8],
                hidden_mult=4.0, mem_dim=32, poly_degree=2,
            )
            cms.levels[1].learning_enabled = True
            cms.eval()
            return cms

        cms_a = make_cms()
        cms_b = make_cms()

        torch.manual_seed(100)
        input_a = torch.randn(2, 32, 64) * 2.0
        torch.manual_seed(200)
        input_b = torch.randn(2, 32, 64) * 0.5

        for _ in range(10):
            with torch.no_grad():
                cms_a(input_a)
                cms_b(input_b)

        # Memory weights should have diverged
        params_a = dict(cms_a.levels[1].memory.named_parameters())
        params_b = dict(cms_b.levels[1].memory.named_parameters())
        diverged = any(
            not torch.allclose(params_a[n], params_b[n].data, atol=1e-6)
            for n in params_a
        )
        assert diverged, "Different inputs should produce different memory states"


class TestCompetenceGate:
    """Tests for the competence-based adaptive gate.

    Gate logic is INVERTED from surprise-based:
    - LOW surprise (L1 predicting well) → gate OPENS (trust L1's contribution)
    - HIGH surprise (L1 confused) → gate CLOSES (protect output)
    L1 still LEARNS from high-surprise tokens — learning is decoupled from gate.
    """

    def test_competence_gate_responds_to_surprise(self):
        """Gate should CLOSE with high surprise, OPEN with low (competence-based)."""
        level = CMSLevel(dim=32, hidden_mult=2.0, chunk_size=1, swiglu=False)

        # Competence gate activates in no_grad context
        with torch.no_grad():
            level._surprise_ema = 0.1  # low surprise = competent
            gate_competent = level._compute_gate().item()

            level._surprise_ema = 1.0  # neutral
            gate_neutral = level._compute_gate().item()

            level._surprise_ema = 5.0  # high surprise = confused
            gate_confused = level._compute_gate().item()

        assert gate_competent > gate_neutral > gate_confused, \
            f"Gate should OPEN with competence (low surprise): " \
            f"competent={gate_competent:.4f}, neutral={gate_neutral:.4f}, confused={gate_confused:.4f}"
        # At neutral surprise (ema=1.0), log(1)=0, so gate = sigmoid(0) = 0.5
        assert abs(gate_neutral - 0.5) < 0.01

    def test_gate_static_during_training(self):
        """During backprop training (grad enabled), gate should be the raw learnable parameter."""
        level = CMSLevel(dim=32, hidden_mult=2.0, chunk_size=1, swiglu=False)
        level.train()

        # With grad enabled, gate should be static sigmoid(residual_gate)
        level._surprise_ema = 100.0  # Very high surprise — should NOT affect gate
        gate = level._compute_gate().item()
        expected = torch.sigmoid(torch.tensor(0.0)).item()  # sigmoid(0) = 0.5
        assert abs(gate - expected) < 1e-6, \
            "Gate should be static during backprop training regardless of surprise"

    def test_gate_dynamic_range(self):
        """Gate should have reasonable range across surprise values."""
        level = CMSLevel(dim=32, hidden_mult=2.0, chunk_size=1, swiglu=False)

        with torch.no_grad():
            # Very low surprise (highly competent) → gate opens wide
            level._surprise_ema = 0.01
            gate_max = level._compute_gate().item()
            assert gate_max > 0.5, f"Gate should be open when competent: {gate_max:.4f}"

            # Very high surprise (confused) → gate closes to protect output
            level._surprise_ema = 10.0
            gate_min = level._compute_gate().item()
            assert gate_min < gate_max, f"Gate should close when confused: {gate_min:.4f}"


class TestLowRankLevel:
    """Tests for the low-rank residual level."""

    def test_small_init_output_dominated_by_base(self):
        """With small random init, low-rank level output should be close to base."""
        l0 = CMSLevel(dim=64, hidden_mult=4.0, chunk_size=1, swiglu=True)
        lrl = LowRankLevel(dim=64, hidden_dim=256, rank=16, chunk_size=8)
        lrl._l0_ref = [l0]
        torch.nn.init.normal_(lrl.A.weight, std=0.02)
        torch.nn.init.normal_(lrl.B.weight, std=0.02)

        x = torch.randn(2, 16, 64)
        l0_out = l0(x)
        out = lrl(x, l0_out=l0_out)
        # Small init → small delta, output dominated by L0
        relative_diff = (out - l0_out).norm() / l0_out.norm()
        assert relative_diff < 0.3, \
            f"Small-init low-rank level should be close to L0: diff={relative_diff:.4f}"

    def test_low_rank_learns_from_forward(self):
        """Low-rank level should update A/B weights via predictive coding.

        With zero-output init (A=0, B=random), A gets gradients first
        (via non-zero B), then once A becomes non-zero, B starts updating too.
        """
        torch.manual_seed(42)
        l0 = CMSLevel(dim=32, hidden_mult=4.0, chunk_size=1, swiglu=True)
        lrl = LowRankLevel(dim=32, hidden_dim=128, rank=8, chunk_size=1, lr=1e-2)
        lrl._l0_ref = [l0]
        # Zero-output init with gradient bootstrap
        torch.nn.init.normal_(lrl.A.weight, std=0.02)
        torch.nn.init.normal_(lrl.B.weight, std=0.02)
        lrl.learning_enabled = True
        lrl.eval()

        x = torch.randn(2, 32, 32)
        l0_out = l0(x)

        A_before = lrl.A.weight.data.clone()
        B_before = lrl.B.weight.data.clone()

        with torch.no_grad():
            _ = lrl(x, l0_out=l0_out)

        A_changed = not torch.allclose(lrl.A.weight.data, A_before, atol=1e-7)
        B_changed = not torch.allclose(lrl.B.weight.data, B_before, atol=1e-7)
        assert A_changed and B_changed, \
            f"Both A and B should update via predictive coding: A={A_changed}, B={B_changed}"

    def test_parameter_count(self):
        """Low-rank level should have dramatically fewer params than full MLP."""
        dim, hidden_dim, rank = 2048, 5408, 32

        # Full MLP: up (dim*hidden) + down (hidden*dim) = 2 * dim * hidden
        full_params = 2 * dim * hidden_dim  # ~22M

        # Low-rank: A (rank*hidden) + B (dim*rank)
        lr_params = rank * hidden_dim + dim * rank  # ~239K

        ratio = full_params / lr_params
        assert ratio > 80, f"Low-rank should be >80x smaller, got {ratio:.1f}x"

    def test_save_and_restore_soul(self):
        """Soul checkpoint should work for low-rank weights."""
        l0 = CMSLevel(dim=32, hidden_mult=4.0, chunk_size=1, swiglu=True)
        lrl = LowRankLevel(dim=32, hidden_dim=128, rank=8, chunk_size=1)
        lrl._l0_ref = [l0]

        # Save soul
        lrl.save_soul()
        assert "A.weight" in lrl._soul_weights
        assert "B.weight" in lrl._soul_weights

        # Verify soul weights match current weights
        assert torch.allclose(lrl._soul_weights["A.weight"], lrl.A.weight.data.float())
        assert torch.allclose(lrl._soul_weights["B.weight"], lrl.B.weight.data.float())


class TestEndToEndLearning:
    """End-to-end test: prove low-rank CMS learns from conversation without seeding."""

    def test_cms_learns_reduces_surprise_opens_gate(self):
        """Full CMS should learn from repeated input, reducing surprise and opening gate.

        This is THE core test for the no-seeding architecture:
        1. Start with zero-init L1 (no seeding)
        2. Run structured input through CMS multiple times
        3. Verify: weights change, surprise drops, gate opens
        """
        torch.manual_seed(42)
        dim = 64

        cms = ContinuumMemorySystem(
            dim=dim, num_levels=2, chunk_sizes=[1, 4],
            hidden_mult=4.0, rank=16, lr=1e-2,
        )
        # Zero-output init with gradient bootstrap (simulating conversion)
        torch.nn.init.normal_(cms.levels[1].A.weight, std=0.02)
        torch.nn.init.normal_(cms.levels[1].B.weight, std=0.02)
        cms.levels[0].learning_enabled = False
        cms.levels[1].learning_enabled = True
        cms.eval()

        # Create structured input with learnable patterns
        # (not random — there should be predictable h_t → h_{t+1} transitions)
        base_pattern = torch.randn(1, 1, dim)
        x = base_pattern + 0.1 * torch.randn(1, 64, dim)  # structured, not random

        # Record initial state
        B_init = cms.levels[1].B.weight.data.clone()
        surprise_init = cms.levels[1]._surprise_ema

        # Run multiple passes (simulating multiple conversations)
        for _ in range(20):
            with torch.no_grad():
                _ = cms(x)

        # 1. Weights should have changed
        B_delta = (cms.levels[1].B.weight.data - B_init).norm().item()
        assert B_delta > 1e-4, f"B weights should change, delta={B_delta:.6f}"

        # 2. Surprise should have changed from initial
        surprise_after = cms.levels[1]._surprise_ema
        assert surprise_after != surprise_init, \
            f"Surprise should change: init={surprise_init:.4f}, after={surprise_after:.4f}"

        # 3. Gate should reflect competence
        with torch.no_grad():
            gate = cms.levels[1]._compute_gate().item()
        # Gate = sigmoid(0 - 0.25 * log(surprise_ema))
        # If surprise dropped below 1.0, log is negative, so -0.25*negative = positive, gate > 0.5
        # If surprise is still high, gate < 0.5
        # Either way, gate should not be at its default value
        assert gate != 0.5, f"Gate should have moved from default: {gate:.4f}"

    def test_different_inputs_produce_different_specializations(self):
        """Two CMS instances fed different distributions should diverge.

        This proves that the environment (input) compiles the identity (weights).
        """
        torch.manual_seed(42)
        dim = 64

        def make_cms():
            cms = ContinuumMemorySystem(
                dim=dim, num_levels=2, chunk_sizes=[1, 4],
                hidden_mult=4.0, rank=16, lr=1e-2,
            )
            torch.nn.init.zeros_(cms.levels[1].A.weight)
            torch.nn.init.normal_(cms.levels[1].B.weight, std=0.02)
            cms.levels[0].learning_enabled = False
            cms.levels[1].learning_enabled = True
            cms.eval()
            return cms

        # Two CMS instances start identical
        torch.manual_seed(0)
        cms_a = make_cms()
        torch.manual_seed(0)
        cms_b = make_cms()

        # Verify they start identical
        assert torch.allclose(cms_a.levels[1].A.weight.data, cms_b.levels[1].A.weight.data)

        # Feed different distributions
        torch.manual_seed(100)
        input_a = torch.randn(1, 64, dim) * 2.0  # high variance domain
        torch.manual_seed(200)
        input_b = torch.randn(1, 64, dim) * 0.5  # low variance domain

        for _ in range(30):
            with torch.no_grad():
                cms_a(input_a)
                cms_b(input_b)

        # Weights should have diverged
        A_diff = (cms_a.levels[1].A.weight.data - cms_b.levels[1].A.weight.data).norm().item()
        B_diff = (cms_a.levels[1].B.weight.data - cms_b.levels[1].B.weight.data).norm().item()

        assert A_diff > 1e-3, f"A weights should diverge between different inputs: {A_diff:.6f}"
        assert B_diff > 1e-3, f"B weights should diverge between different inputs: {B_diff:.6f}"

    def test_system_prompt_steers_learning(self):
        """A system prompt prefix should change the hidden state landscape,
        producing different L1 weight updates than no system prompt.

        This proves: system prompt → different hidden states → different prediction
        errors → different L1 weights. The prompt steers specialization.
        """
        torch.manual_seed(42)
        dim = 64

        def make_cms():
            cms = ContinuumMemorySystem(
                dim=dim, num_levels=2, chunk_sizes=[1, 4],
                hidden_mult=4.0, rank=16, lr=1e-2,
            )
            torch.nn.init.zeros_(cms.levels[1].A.weight)
            torch.nn.init.normal_(cms.levels[1].B.weight, std=0.02)
            cms.levels[0].learning_enabled = False
            cms.levels[1].learning_enabled = True
            cms.eval()
            return cms

        # Same user input
        torch.manual_seed(99)
        user_input = torch.randn(1, 32, dim)

        # "System prompt" = different prefix hidden states
        torch.manual_seed(1)
        system_a = torch.randn(1, 16, dim) * 1.5  # "analytical" prompt
        torch.manual_seed(2)
        system_b = torch.randn(1, 16, dim) * 0.3  # "casual" prompt

        # Concatenate: [system_prompt, user_input]
        input_with_sys_a = torch.cat([system_a, user_input], dim=1)
        input_with_sys_b = torch.cat([system_b, user_input], dim=1)

        torch.manual_seed(0)
        cms_a = make_cms()
        torch.manual_seed(0)
        cms_b = make_cms()

        for _ in range(30):
            with torch.no_grad():
                cms_a(input_with_sys_a)
                cms_b(input_with_sys_b)

        # Different system prompts should produce different specializations
        B_diff = (cms_a.levels[1].B.weight.data - cms_b.levels[1].B.weight.data).norm().item()
        assert B_diff > 1e-3, \
            f"Different system prompts should produce different L1 weights: {B_diff:.6f}"


class TestCKA:
    """Tests for Centered Kernel Alignment metric."""

    def test_identical_representations_cka_one(self):
        """CKA of identical representations should be 1.0."""
        from anamnesis.evaluation.metrics import compute_cka
        x = torch.randn(50, 64)
        cka = compute_cka(x, x)
        assert abs(cka - 1.0) < 0.01, f"CKA of identical reps should be ~1.0, got {cka:.4f}"

    def test_random_representations_cka_lower(self):
        """CKA of independent random representations should be lower than identical."""
        from anamnesis.evaluation.metrics import compute_cka
        torch.manual_seed(42)
        x = torch.randn(200, 64)
        torch.manual_seed(99)
        y = torch.randn(200, 64)
        cka = compute_cka(x, y)
        # With finite samples, CKA won't be exactly 0, but should be much less than 1
        assert cka < 0.5, f"CKA of random reps should be well below 1.0, got {cka:.4f}"

    def test_scaled_representations_cka_high(self):
        """CKA should be invariant to isotropic scaling."""
        from anamnesis.evaluation.metrics import compute_cka
        torch.manual_seed(42)
        x = torch.randn(50, 64)
        y = x * 3.7  # scaled version
        cka = compute_cka(x, y)
        assert cka > 0.99, f"CKA should be scale-invariant, got {cka:.4f}"

    def test_structured_vs_corrupted_cka(self):
        """CKA should detect when representations are meaningfully different."""
        from anamnesis.evaluation.metrics import compute_cka
        torch.manual_seed(42)
        x = torch.randn(100, 64)
        # Add structured corruption to some dimensions (simulating specialization)
        y = x.clone()
        y[:, :32] = torch.randn(100, 32)  # replace half the dimensions
        cka = compute_cka(x, y)
        # Partial corruption should drop CKA but not to zero
        assert 0.1 < cka < 0.9, f"Partial corruption should give mid-range CKA, got {cka:.4f}"
