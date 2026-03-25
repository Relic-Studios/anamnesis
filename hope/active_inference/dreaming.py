"""
CMS Dreaming — Extension 4.

Offline consolidation for CMS levels, triggered by the Gardener stream.
Two-phase cycle mirroring biological sleep:

NREM Phase (structural consolidation):
    - SVD decompose weight deltas since last dream
    - Prune singular values below salience threshold
    - Merge remaining deltas into base weights
    - Apply Ebbinghaus-curve decay to fast CMS levels

REM Phase (creative reorganization):
    - Inject structured noise into medium CMS levels
    - Run forward pass on held-out conversation samples (replay buffer)
    - Measure signal on noisy outputs
    - Keep perturbations that improve signal (bridge discovery)

Active Inference mapping: Offline free energy minimization — reducing
model complexity (pruning) while maintaining predictive accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class DreamResult:
    """Results from a complete dream cycle."""
    nrem_pruned_params: int = 0       # Number of singular values pruned
    nrem_energy_before: float = 0.0   # Total param norm before NREM
    nrem_energy_after: float = 0.0    # Total param norm after NREM
    rem_perturbations_tested: int = 0
    rem_bridges_discovered: int = 0   # Perturbations that improved signal
    rem_bridges_rejected: int = 0     # Perturbations that degraded signal
    signal_before: float = 0.0
    signal_after: float = 0.0


class NREMConsolidation:
    """
    NREM sleep: structural consolidation via SVD pruning and decay.

    Compresses what the CMS levels have learned by removing low-energy
    weight directions and merging the remaining deltas into base weights.

    This is the "forgetting that strengthens" — removing noise so signal
    emerges more clearly.

    Args:
        prune_threshold: Singular values below this fraction of the max
            are pruned (default 0.05 = keep top 95% of energy).
        decay_rate: Ebbinghaus decay rate for fast CMS level deltas.
        min_salience: Minimum fraction to preserve (never prune below this).
    """

    def __init__(
        self,
        prune_threshold: float = 0.05,
        decay_rate: float = 0.1,
        min_salience: float = 0.1,
    ):
        self.prune_threshold = prune_threshold
        self.decay_rate = decay_rate
        self.min_salience = min_salience

    @torch.no_grad()
    def consolidate_level(
        self,
        level: nn.Module,
        base_weights: dict[str, Tensor] | None = None,
        hours_elapsed: float = 1.0,
    ) -> int:
        """
        Apply NREM consolidation to a single CMS level.

        1. If base_weights provided: compute delta = current - base
        2. SVD decompose delta
        3. Prune small singular values
        4. Reconstruct from pruned SVD
        5. Update level weights

        Args:
            level: A CMSLevel module.
            base_weights: Soul checkpoint weights for this level (for delta computation).
            hours_elapsed: Time since last consolidation (for Ebbinghaus decay).

        Returns:
            Number of singular values pruned.
        """
        total_pruned = 0

        for name, param in level.named_parameters():
            if param.dim() < 2:
                continue  # Skip 1D params (biases)

            if base_weights and name in base_weights:
                delta = param.data - base_weights[name]
            else:
                delta = param.data.clone()

            # SVD decomposition
            try:
                U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
            except Exception:
                continue  # SVD can fail on degenerate matrices

            # Prune: remove singular values below threshold
            threshold = S.max() * self.prune_threshold
            mask = S > threshold

            # Ensure we keep at least min_salience fraction
            min_keep = max(1, int(len(S) * self.min_salience))
            if mask.sum() < min_keep:
                _, top_indices = S.topk(min_keep)
                mask = torch.zeros_like(mask)
                mask[top_indices] = True

            pruned_count = (~mask).sum().item()
            total_pruned += pruned_count

            # Reconstruct from pruned SVD
            S_pruned = S * mask.float()
            reconstructed = U @ torch.diag(S_pruned) @ Vh

            # Apply Ebbinghaus decay to the delta
            decay_factor = math.exp(-self.decay_rate * hours_elapsed)
            decayed = reconstructed * decay_factor

            # Update: base + decayed_pruned_delta
            if base_weights and name in base_weights:
                param.data.copy_(base_weights[name] + decayed)
            else:
                param.data.copy_(decayed)

        return total_pruned


class REMExploration:
    """
    REM sleep: creative reorganization via structured noise + evaluation.

    Injects perturbations into CMS levels and tests whether they improve
    signal health on held-out data. Perturbations that help are "bridge
    discoveries" — the model found a useful weight direction it hadn't
    explored during normal learning.

    This is the "dreaming" that creates novel connections.

    Args:
        noise_scale: Scale of perturbation noise (relative to param norm).
        num_perturbations: Number of random perturbations to test per level.
    """

    def __init__(
        self,
        noise_scale: float = 0.01,
        num_perturbations: int = 5,
    ):
        self.noise_scale = noise_scale
        self.num_perturbations = num_perturbations

    @torch.no_grad()
    def explore_level(
        self,
        level: nn.Module,
        eval_fn: Callable[[nn.Module], float],
    ) -> tuple[int, int]:
        """
        Run REM exploration on a single CMS level.

        For each perturbation:
        1. Save current weights
        2. Add structured noise
        3. Evaluate signal on held-out data via eval_fn
        4. If signal improved: keep perturbation (bridge discovered)
        5. If signal degraded: revert

        Args:
            level: A CMSLevel module.
            eval_fn: Function that takes the level's parent model and returns
                signal health score. This runs on the replay buffer.

        Returns:
            Tuple of (bridges_discovered, bridges_rejected).
        """
        bridges = 0
        rejects = 0

        # Baseline signal
        baseline_signal = eval_fn(level)

        for _ in range(self.num_perturbations):
            # Save current state
            saved = {name: p.clone() for name, p in level.named_parameters()}

            # Add structured noise (scaled by param norm)
            for name, param in level.named_parameters():
                noise = torch.randn_like(param) * self.noise_scale * param.norm()
                param.data.add_(noise)

            # Evaluate
            perturbed_signal = eval_fn(level)

            if perturbed_signal > baseline_signal:
                # Bridge discovered — keep the perturbation
                bridges += 1
                baseline_signal = perturbed_signal  # new baseline
            else:
                # Revert
                for name, param in level.named_parameters():
                    param.data.copy_(saved[name])
                rejects += 1

        return bridges, rejects


class DreamCycle:
    """
    Complete dreaming cycle combining NREM and REM phases.

    Triggered by the GardenerStream when signal health indicates
    consolidation is needed. Runs offline (not in the inference hot path).

    Usage:
        dreamer = DreamCycle(...)
        if gardener_output.should_dream:
            result = dreamer.dream(model, eval_fn, soul_checkpoint)
            gardener.acknowledge_dream()

    Args:
        nrem_prune_threshold: SVD pruning threshold for NREM.
        nrem_decay_rate: Ebbinghaus decay rate.
        rem_noise_scale: Perturbation noise for REM.
        rem_perturbations: Number of perturbations per level in REM.
        rem_min_level: Minimum CMS level index to apply REM to
            (don't perturb the fastest level — it's too volatile).
        rem_max_level: Maximum CMS level index for REM
            (don't perturb the slowest level — it's identity).
    """

    def __init__(
        self,
        nrem_prune_threshold: float = 0.05,
        nrem_decay_rate: float = 0.1,
        rem_noise_scale: float = 0.01,
        rem_perturbations: int = 5,
        rem_min_level: int = 1,
        rem_max_level: int = 2,
    ):
        self.nrem = NREMConsolidation(
            prune_threshold=nrem_prune_threshold,
            decay_rate=nrem_decay_rate,
        )
        self.rem = REMExploration(
            noise_scale=rem_noise_scale,
            num_perturbations=rem_perturbations,
        )
        self.rem_min_level = rem_min_level
        self.rem_max_level = rem_max_level

    def dream(
        self,
        cms_levels: nn.ModuleList,
        eval_fn: Callable[[nn.Module], float],
        soul_checkpoint: list[dict[str, Tensor]] | None = None,
        hours_since_last: float = 1.0,
    ) -> DreamResult:
        """
        Run a complete NREM→REM dream cycle on all CMS levels.

        Args:
            cms_levels: The CMS module's list of levels.
            eval_fn: Signal evaluation function for REM exploration.
            soul_checkpoint: Per-level base weights from soul checkpoint.
            hours_since_last: Hours since last dream (for decay calculation).

        Returns:
            DreamResult with consolidation and exploration statistics.
        """
        result = DreamResult()

        # Measure total param energy before
        result.nrem_energy_before = sum(
            p.norm().item() for level in cms_levels for p in level.parameters()
        )

        # ── NREM Phase ──
        for i, level in enumerate(cms_levels):
            base = soul_checkpoint[i] if soul_checkpoint and i < len(soul_checkpoint) else None
            pruned = self.nrem.consolidate_level(level, base, hours_since_last)
            result.nrem_pruned_params += pruned

        # Measure energy after NREM
        result.nrem_energy_after = sum(
            p.norm().item() for level in cms_levels for p in level.parameters()
        )

        # ── REM Phase ── (only on medium-frequency levels)
        for i, level in enumerate(cms_levels):
            if i < self.rem_min_level or i > self.rem_max_level:
                continue

            bridges, rejects = self.rem.explore_level(level, eval_fn)
            result.rem_perturbations_tested += bridges + rejects
            result.rem_bridges_discovered += bridges
            result.rem_bridges_rejected += rejects

        return result
