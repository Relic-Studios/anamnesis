"""
Neutral Drift — Extension 5.

Continuous micro-perturbation for dormant CMS levels to prevent the
homeostatic trap (discovered in our cellular simulation studies).

Without drift, slow CMS levels that aren't being updated become rigid —
they maintain structural identity but lose the ability to learn when
their update interval arrives. This is because the loss landscape
around their current position becomes flat (local minimum deepens).

Neutral drift injects infinitesimal noise to keep the landscape navigable.
The noise magnitude scales inversely with the level's update frequency:
slower levels get less noise (they should be more stable).

Active Inference mapping: Prior diffusion — maintaining the generative
model's capacity to explain novel observations.

Biological parallel: Kimura neutral drift maintains genetic diversity
even without selective pressure.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class NeutralDrift(nn.Module):
    """
    Applies neutral drift to CMS level parameters.

    For each CMS level l with update frequency C_l:
        Between updates (when t % C_l ≠ 0):
            θ_l += ε · N(0, σ²_l)
            where σ_l = σ_base / C_l
            and ε = plasticity_gate (from gardener, default 1.0)

    Args:
        sigma_base: Base noise magnitude (default 1e-5 for bfloat16 stability).
        scale_by_frequency: Whether to scale noise by 1/chunk_size.
        enabled: Whether drift is active (disable during initial training).
    """

    def __init__(
        self,
        sigma_base: float = 1e-5,
        scale_by_frequency: bool = True,
        enabled: bool = False,
    ):
        super().__init__()
        self.sigma_base = sigma_base
        self.scale_by_frequency = scale_by_frequency
        self.enabled = enabled

    @torch.no_grad()
    def apply(
        self,
        params: dict[str, Tensor] | nn.Module,
        chunk_size: int = 1,
        plasticity_gate: float = 1.0,
    ) -> None:
        """
        Apply neutral drift to parameters.

        Args:
            params: Either a dict of tensors or an nn.Module.
            chunk_size: The CMS level's update frequency (for scaling).
            plasticity_gate: Multiplier from gardener (0=no drift, 1=full drift).
        """
        if not self.enabled:
            return

        sigma = self.sigma_base
        if self.scale_by_frequency and chunk_size > 1:
            sigma = sigma / chunk_size

        effective_sigma = sigma * plasticity_gate

        if effective_sigma <= 0:
            return

        if isinstance(params, nn.Module):
            for p in params.parameters():
                if p.requires_grad:
                    p.add_(torch.randn_like(p) * effective_sigma)
        else:
            for name, p in params.items():
                if isinstance(p, Tensor):
                    p.add_(torch.randn_like(p) * effective_sigma)

    def apply_to_cms(self, cms: nn.Module, plasticity_gate: float = 1.0) -> None:
        """
        Apply drift to all levels of a ContinuumMemorySystem.

        Automatically scales noise by each level's chunk_size.

        Args:
            cms: A ContinuumMemorySystem module.
            plasticity_gate: Global plasticity multiplier.
        """
        if not self.enabled:
            return

        for level in cms.levels:
            self.apply(level, chunk_size=level.chunk_size, plasticity_gate=plasticity_gate)

    def extra_repr(self) -> str:
        return (
            f"sigma_base={self.sigma_base}, "
            f"scale_by_freq={self.scale_by_frequency}, "
            f"enabled={self.enabled}"
        )
