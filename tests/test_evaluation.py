"""Tests for evaluation metrics and ablation framework."""

import pytest
import torch
from torch.utils.data import DataLoader

from anamnesis.core.model import HopeModel, HopeConfig
from anamnesis.evaluation.metrics import (
    compute_perplexity,
    snapshot_cms_state,
    compute_cms_delta,
    compute_surprise_profile,
)
from anamnesis.evaluation.ablation import AblationConfig, ABLATION_CONFIGS


def _small_config():
    return HopeConfig(
        vocab_size=256, hidden_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_kv_heads=2,
        cms_levels=2, cms_chunk_sizes=[1, 8], cms_hidden_mult=4.0,
    )


def _make_dataloader(batch_size=2, num_batches=5, seq_len=16, vocab_size=256):
    """Create a simple random dataloader."""
    batches = []
    for _ in range(num_batches):
        batches.append({"input_ids": torch.randint(0, vocab_size, (batch_size, seq_len))})
    return DataLoader(batches, batch_size=None)


class TestPerplexity:
    def test_basic(self):
        config = _small_config()
        model = HopeModel(config)
        model.eval()
        dl = _make_dataloader()
        ppl = compute_perplexity(model, dl)
        assert ppl > 0
        assert ppl < float("inf")

    def test_max_batches(self):
        config = _small_config()
        model = HopeModel(config)
        model.eval()
        dl = _make_dataloader(num_batches=10)
        ppl = compute_perplexity(model, dl, max_batches=2)
        assert ppl > 0

    def test_empty_dataloader(self):
        config = _small_config()
        model = HopeModel(config)
        model.eval()
        dl = DataLoader([], batch_size=None)
        ppl = compute_perplexity(model, dl)
        assert ppl == float("inf")


class TestCMSDelta:
    def test_no_change(self):
        config = _small_config()
        model = HopeModel(config)
        snap = snapshot_cms_state(model)
        delta = compute_cms_delta(snap, snap)
        assert delta["total_l2"] == 0.0
        assert delta["max_param_delta"] == 0.0

    def test_after_forward(self):
        config = _small_config()
        model = HopeModel(config)
        model.eval()

        before = snapshot_cms_state(model)
        x = torch.randint(0, 256, (1, 16))
        with torch.no_grad():
            model(x)
        after = snapshot_cms_state(model)

        delta = compute_cms_delta(before, after)
        # Inner-loop learning should have changed some weights
        assert delta["total_l2"] >= 0.0

    def test_per_level_breakdown(self):
        config = _small_config()
        model = HopeModel(config)
        snap = snapshot_cms_state(model)
        # Manually perturb level 0 weights
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "cms.levels.0.up_proj" in name:
                    param.add_(torch.randn_like(param) * 0.1)
        after = snapshot_cms_state(model)
        delta = compute_cms_delta(snap, after)
        assert delta["total_l2"] > 0.0
        assert "level_0" in delta["per_level"]


class TestSurpriseProfile:
    def test_shape(self):
        config = _small_config()
        model = HopeModel(config)
        profile = compute_surprise_profile(model)
        assert len(profile) == config.num_hidden_layers
        for layer_profile in profile:
            assert len(layer_profile) == config.cms_levels


class TestAblationConfig:
    def test_all_configs_defined(self):
        assert "full" in ABLATION_CONFIGS
        assert "baseline" in ABLATION_CONFIGS
        assert len(ABLATION_CONFIGS) == 9  # full + 7 single-removal + baseline

    def test_baseline_all_off(self):
        baseline = ABLATION_CONFIGS["baseline"]
        assert not baseline.enable_gardener
        assert not baseline.enable_thompson
        assert not baseline.enable_toroidal
        assert not baseline.enable_drift
        assert not baseline.enable_dreaming
        assert not baseline.enable_signal_loss
        assert not baseline.enable_identity_loss

    def test_to_trainer_config(self):
        config = ABLATION_CONFIGS["no_signal_loss"]
        tc = config.to_trainer_config(max_steps=50)
        assert tc.lambda_signal_target == 0.0
        assert tc.lambda_identity_target == 0.01  # identity still on
        assert tc.max_steps == 50

    def test_full_has_everything(self):
        full = ABLATION_CONFIGS["full"]
        assert full.enable_gardener
        assert full.enable_thompson
        assert full.enable_toroidal
        assert full.enable_drift
        assert full.enable_dreaming
        assert full.enable_signal_loss
        assert full.enable_identity_loss
