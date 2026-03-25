"""Tests for Active Inference Cluster C: Dreaming + Toroidal Flow."""

import pytest
import torch
import torch.nn as nn

from hope.core.cms import CMSLevel, ContinuumMemorySystem
from hope.active_inference.dreaming import (
    NREMConsolidation, REMExploration, DreamCycle, DreamResult,
)
from hope.active_inference.toroidal import ToroidalFlow, LevelSignal


class TestNREMConsolidation:
    def test_consolidation_reduces_energy(self):
        """NREM should prune small singular values, reducing total energy."""
        nrem = NREMConsolidation(prune_threshold=0.1)
        level = CMSLevel(dim=32, hidden_mult=4.0, chunk_size=32)

        # Add random noise to weights (simulating accumulated deltas)
        with torch.no_grad():
            for p in level.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        energy_before = sum(p.norm().item() for p in level.parameters())
        pruned = nrem.consolidate_level(level, hours_elapsed=2.0)
        energy_after = sum(p.norm().item() for p in level.parameters())

        assert energy_after <= energy_before
        assert pruned >= 0

    def test_consolidation_with_base_weights(self):
        """Consolidation with soul checkpoint should compute deltas correctly."""
        nrem = NREMConsolidation(prune_threshold=0.05)
        level = CMSLevel(dim=32, hidden_mult=2.0, chunk_size=32)

        base = {name: p.clone() for name, p in level.named_parameters()}

        # Modify weights
        with torch.no_grad():
            for p in level.parameters():
                p.add_(torch.randn_like(p) * 0.5)

        pruned = nrem.consolidate_level(level, base_weights=base, hours_elapsed=1.0)
        assert pruned >= 0

    def test_min_salience_preserves_params(self):
        """Even aggressive pruning should keep min_salience fraction."""
        nrem = NREMConsolidation(prune_threshold=0.99, min_salience=0.5)
        level = CMSLevel(dim=16, hidden_mult=2.0, chunk_size=1)

        with torch.no_grad():
            for p in level.parameters():
                p.add_(torch.randn_like(p))

        nrem.consolidate_level(level)
        # Params should still have non-zero values
        for p in level.parameters():
            if p.dim() >= 2:
                assert p.abs().sum() > 0


class TestREMExploration:
    def test_bridges_discovered(self):
        """REM should find at least some perturbations that help."""
        rem = REMExploration(noise_scale=0.1, num_perturbations=10)
        level = CMSLevel(dim=16, hidden_mult=2.0, chunk_size=1)

        call_count = [0]

        def eval_fn(module):
            call_count[0] += 1
            # Random signal — some perturbations will "help" by chance
            return 0.5 + (torch.randn(1).item() * 0.2)

        bridges, rejects = rem.explore_level(level, eval_fn)
        assert bridges + rejects == 10
        assert call_count[0] >= 10  # baseline + perturbations

    def test_revert_on_rejection(self):
        """Rejected perturbations should be fully reverted."""
        rem = REMExploration(noise_scale=1.0, num_perturbations=5)
        level = CMSLevel(dim=8, hidden_mult=2.0, chunk_size=1)

        initial_weights = {name: p.clone() for name, p in level.named_parameters()}

        def always_worse(module):
            return 0.1  # always worse than baseline

        # First call establishes baseline, rest are perturbations
        bridges, rejects = rem.explore_level(level, always_worse)
        assert bridges == 0  # nothing should be kept

        # Weights should be reverted to initial
        for name, p in level.named_parameters():
            assert torch.allclose(p, initial_weights[name], atol=1e-6), (
                f"Weight {name} wasn't reverted after all rejections"
            )


class TestDreamCycle:
    def test_full_cycle(self):
        cms = ContinuumMemorySystem(dim=16, num_levels=3, chunk_sizes=[1, 4, 16])

        call_count = [0]

        def eval_fn(module):
            call_count[0] += 1
            return 0.5 + torch.randn(1).item() * 0.1

        dreamer = DreamCycle(
            rem_noise_scale=0.01, rem_perturbations=3,
            rem_min_level=1, rem_max_level=1,
        )
        result = dreamer.dream(cms.levels, eval_fn)

        assert isinstance(result, DreamResult)
        assert result.nrem_energy_before >= 0
        assert result.rem_perturbations_tested > 0

    def test_nrem_reduces_energy(self):
        cms = ContinuumMemorySystem(dim=16, num_levels=2, chunk_sizes=[1, 8])

        # Add noise
        with torch.no_grad():
            for level in cms.levels:
                for p in level.parameters():
                    p.add_(torch.randn_like(p) * 0.5)

        dreamer = DreamCycle(rem_perturbations=0)  # NREM only
        result = dreamer.dream(cms.levels, lambda m: 0.5)

        assert result.nrem_energy_after <= result.nrem_energy_before

    def test_rem_respects_level_bounds(self):
        """REM should only explore levels within min/max bounds."""
        cms = ContinuumMemorySystem(dim=16, num_levels=4, chunk_sizes=[1, 4, 16, 64])

        explored_levels = set()

        def tracking_eval(module):
            # Track which level is being explored
            for i, level in enumerate(cms.levels):
                if module is level:
                    explored_levels.add(i)
            return 0.5

        dreamer = DreamCycle(
            rem_perturbations=2, rem_min_level=1, rem_max_level=2,
        )
        dreamer.dream(cms.levels, tracking_eval)

        # Only levels 1 and 2 should have been explored
        assert 0 not in explored_levels  # too fast
        assert 3 not in explored_levels  # too slow (identity)


class TestToroidalFlow:
    def test_no_signal_initially(self):
        flow = ToroidalFlow(num_levels=4, sustained_chunks=3)
        signals = flow.check_signals()
        assert len(signals) == 0

    def test_sustained_surprise_triggers_signal(self):
        flow = ToroidalFlow(
            num_levels=4, surprise_threshold=0.5,
            sustained_chunks=3, hold_time=0,
        )

        # Feed sustained high surprise at level 0
        for _ in range(3):
            flow.update_surprise(0, surprise=0.8)

        signals = flow.check_signals()
        assert len(signals) > 0
        assert all(s.signal_type == "increase_plasticity" for s in signals)
        # Should target levels 1, 2, 3 (all slower than level 0)
        targets = {s.target_level for s in signals}
        assert targets == {1, 2, 3}

    def test_hysteresis_prevents_rapid_signals(self):
        flow = ToroidalFlow(
            num_levels=3, surprise_threshold=0.5,
            sustained_chunks=2, hold_time=5,
        )

        # Trigger first signal
        flow.update_surprise(0, 0.8)
        flow.update_surprise(0, 0.8)
        signals1 = flow.check_signals()
        assert len(signals1) > 0

        # Immediately try again — should be blocked by hysteresis
        flow.update_surprise(0, 0.8)
        flow.update_surprise(0, 0.8)
        signals2 = flow.check_signals()
        assert len(signals2) == 0  # hold_time not elapsed

    def test_damping_reduces_strength(self):
        flow = ToroidalFlow(
            num_levels=3, sustained_chunks=2, hold_time=0, damping=0.5,
        )

        # First signal
        flow.update_surprise(0, 0.9)
        flow.update_surprise(0, 0.9)
        signals1 = flow.check_signals()
        strength1 = signals1[0].strength if signals1 else 0

        # Second signal (after enough updates)
        flow.update_surprise(0, 0.9)
        flow.update_surprise(0, 0.9)
        signals2 = flow.check_signals()
        strength2 = signals2[0].strength if signals2 else 0

        assert strength2 < strength1  # damped

    def test_convergence_restores_strength(self):
        flow = ToroidalFlow(
            num_levels=3, sustained_chunks=2, hold_time=0, damping=0.5,
        )

        # Trigger and damp
        flow.update_surprise(0, 0.9)
        flow.update_surprise(0, 0.9)
        flow.check_signals()
        damped_strength = flow._signal_strength[0]

        # Feed enough history for trend computation (needs 2*n=10 points)
        # First window: high surprise
        for _ in range(5):
            flow.update_surprise(0, 0.8)
        # Second window: decreasing surprise (convergence)
        for _ in range(5):
            flow.update_surprise(0, 0.2)

        flow.check_signals()
        assert flow._signal_strength[0] > damped_strength  # restored

    def test_apply_signals_boosts_plasticity(self):
        flow = ToroidalFlow(num_levels=3)
        signals = [
            LevelSignal(0, 2, "increase_plasticity", strength=0.8),
        ]
        gates = [0.5, 0.5, 0.5]
        modified = flow.apply_signals(signals, gates)
        assert modified[2] > gates[2]  # level 2 got boosted
        assert modified[0] == gates[0]  # level 0 unchanged
        assert modified[1] == gates[1]  # level 1 unchanged

    def test_diagnostics(self):
        flow = ToroidalFlow(num_levels=4)
        flow.update_surprise(0, 0.5)
        diag = flow.get_diagnostics()
        assert "surprise_emas" in diag
        assert len(diag["surprise_emas"]) == 4

    def test_reset(self):
        flow = ToroidalFlow(num_levels=3)
        flow.update_surprise(0, 0.9)
        flow.update_surprise(1, 0.8)
        flow.reset()
        assert all(e == 0.0 for e in flow._surprise_ema)
        assert all(s == 1.0 for s in flow._signal_strength)
