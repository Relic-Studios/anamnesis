"""Tests for HopeBlock and HopeModel."""

import pytest
import torch

from hope.core.block import HopeBlock
from hope.core.cms import CMSVariant
from hope.core.model import HopeModel, HopeConfig


class TestHopeBlock:
    """Tests for individual Hope blocks."""

    def test_forward_shape(self):
        block = HopeBlock(dim=64, num_attention_heads=4, num_kv_heads=2, cms_levels=3,
                          cms_chunk_sizes=[1, 8, 32], cms_hidden_mult=4.0)
        x = torch.randn(2, 16, 64)
        y, mem = block(x)
        assert y.shape == x.shape
        assert mem is None  # no neural memory

    def test_forward_with_neural_memory(self):
        block = HopeBlock(dim=64, num_attention_heads=4, num_kv_heads=2, cms_levels=2,
                          cms_chunk_sizes=[1, 8], use_neural_memory=True, mem_heads=2)
        x = torch.randn(2, 16, 64)
        y, mem = block(x)
        assert y.shape == x.shape
        assert mem is not None

    def test_gradient_flow_through_block(self):
        block = HopeBlock(dim=64, num_attention_heads=4, num_kv_heads=2, cms_levels=2,
                          cms_chunk_sizes=[1, 8])
        x = torch.randn(2, 16, 64, requires_grad=True)
        y, _ = block(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_gqa_correctness(self):
        """GQA with fewer KV heads should still produce valid output."""
        block = HopeBlock(dim=128, num_attention_heads=8, num_kv_heads=2,
                          cms_levels=2, cms_chunk_sizes=[1, 8])
        x = torch.randn(2, 16, 128)
        y, _ = block(x)
        assert y.shape == x.shape


class TestHopeModel:
    """Tests for the full model."""

    @pytest.fixture
    def small_config(self):
        return HopeConfig(
            vocab_size=256,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_kv_heads=2,
            cms_levels=2,
            cms_chunk_sizes=[1, 8],
            cms_hidden_mult=4.0,
        )

    def test_forward_shape(self, small_config):
        model = HopeModel(small_config)
        input_ids = torch.randint(0, 256, (2, 16))
        out = model(input_ids)
        assert out["logits"].shape == (2, 16, 256)

    def test_forward_with_labels(self, small_config):
        model = HopeModel(small_config)
        input_ids = torch.randint(0, 256, (2, 16))
        labels = torch.randint(0, 256, (2, 16))
        out = model(input_ids, labels=labels)
        assert "loss" in out
        assert out["loss"].ndim == 0
        assert out["loss"].item() > 0

    def test_gradient_flow_full_model(self, small_config):
        model = HopeModel(small_config)
        input_ids = torch.randint(0, 256, (2, 16))
        labels = torch.randint(0, 256, (2, 16))
        out = model(input_ids, labels=labels)
        out["loss"].backward()

        # Check that at least some CMS parameters have gradients
        has_cms_grad = False
        for name, param in model.named_parameters():
            if "cms" in name and param.grad is not None:
                if param.grad.abs().sum() > 0:
                    has_cms_grad = True
                    break
        assert has_cms_grad, "No gradients reached CMS parameters"

    def test_num_parameters(self, small_config):
        model = HopeModel(small_config)
        n = model.num_parameters()
        assert n > 0
        assert n == sum(p.numel() for p in model.parameters() if p.requires_grad)

    def test_enable_drift(self, small_config):
        model = HopeModel(small_config)
        model.enable_drift(True)
        for layer in model.layers:
            for cms_level in layer.cms.levels:
                assert cms_level.drift_enabled is True

    def test_qwen_config(self):
        config = HopeConfig.from_qwen2_5_7b()
        assert config.hidden_size == 3584
        assert config.num_hidden_layers == 28
        assert config.num_kv_heads == 4
        assert config.vocab_size == 152064

    def test_memory_states_persist(self, small_config):
        small_config.use_neural_memory = True
        model = HopeModel(small_config)
        input_ids = torch.randint(0, 256, (2, 16))
        out = model(input_ids)
        states = out["memory_states"]
        assert len(states) == small_config.num_hidden_layers
        # States should be valid MemoryState objects
        for s in states:
            if s is not None:
                assert s.seq_index == 16
