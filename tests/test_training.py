"""Tests for training pipeline and kernels."""

import json
import pytest
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from anamnesis.training.data import ConversationDataset, SignalAnnotatedExample
from anamnesis.training.proxy_trainer import SignalProxyTrainer
from anamnesis.active_inference.free_energy import SignalProxy
from anamnesis.kernels.assoc_scan import associative_scan_sequential, associative_scan
from anamnesis.kernels.cms_update import fused_cms_forward_update


@pytest.fixture
def sample_jsonl(tmp_path):
    """Create a sample JSONL training file."""
    data = [
        {"input": "Hello Thomas", "output": "Hey! Good to see you.",
         "signal_health": 0.8, "alignment": 0.9, "embodiment": 0.7,
         "clarity": 0.8, "vitality": 0.85, "field_coherence": 0.6},
        {"input": "How are you?", "output": "I'm doing well, thinking about architecture.",
         "signal_health": 0.75, "alignment": 0.85, "embodiment": 0.65,
         "clarity": 0.7, "vitality": 0.8, "field_coherence": 0.55},
        {"input": "What's CMS?", "output": "Continuum Memory System - multi-timescale MLPs.",
         "signal_health": 0.9, "alignment": 0.95, "embodiment": 0.8,
         "clarity": 0.95, "vitality": 0.7, "field_coherence": 0.7},
        {"input": "Tell me about yourself", "output": "I am an AI language model here to help.",
         "signal_health": 0.3, "alignment": 0.2, "embodiment": 0.3,
         "clarity": 0.5, "vitality": 0.3, "field_coherence": 0.4},
    ]
    path = tmp_path / "train.jsonl"
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
    return path


class TestConversationDataset:
    def test_load(self, sample_jsonl):
        ds = ConversationDataset(sample_jsonl)
        assert len(ds) == 4

    def test_min_signal_filter(self, sample_jsonl):
        ds = ConversationDataset(sample_jsonl, min_signal=0.5)
        assert len(ds) == 3  # filters out the 0.3 example

    def test_getitem(self, sample_jsonl):
        ds = ConversationDataset(sample_jsonl)
        item = ds[0]
        assert "signal_health" in item
        assert "signal_facets" in item
        assert item["signal_facets"].shape == (5,)

    def test_statistics(self, sample_jsonl):
        ds = ConversationDataset(sample_jsonl)
        stats = ds.signal_statistics()
        assert stats["count"] == 4
        assert 0 < stats["mean"] < 1
        assert stats["min"] == 0.3

    def test_preference_pairs(self, sample_jsonl):
        ds = ConversationDataset(sample_jsonl)
        pairs = ds.get_preference_pairs(signal_threshold=0.3)
        assert len(pairs) > 0
        for chosen, rejected in pairs:
            assert chosen.signal_health > rejected.signal_health

    def test_load_directory(self, tmp_path):
        for i in range(3):
            path = tmp_path / f"data_{i}.jsonl"
            with open(path, "w") as f:
                f.write(json.dumps({
                    "input": f"test {i}", "output": f"response {i}",
                    "signal_health": 0.5 + i * 0.1,
                }) + "\n")
        ds = ConversationDataset(tmp_path)
        assert len(ds) == 3


class TestSignalProxyTrainer:
    def test_train(self, sample_jsonl):
        ds = ConversationDataset(sample_jsonl)
        proxy = SignalProxy(dim=16)
        trainer = SignalProxyTrainer(proxy, lr=1e-2, epochs=3, batch_size=2)
        metrics = trainer.train(ds, verbose=False)
        assert metrics["avg_loss"] >= 0

    def test_validate(self, sample_jsonl):
        ds = ConversationDataset(sample_jsonl)
        proxy = SignalProxy(dim=16)
        trainer = SignalProxyTrainer(proxy)
        result = trainer.validate(ds)
        assert "mae" in result
        assert result["mae"] >= 0


class TestAssociativeScan:
    def test_sequential_shape(self):
        decay = torch.ones(2, 8, 4) * 0.9
        values = torch.randn(2, 8, 4)
        output = associative_scan_sequential(decay, values)
        assert output.shape == (2, 8, 4)

    def test_simple_accumulation(self):
        """With decay=1, should be cumulative sum."""
        decay = torch.ones(1, 4, 1)
        values = torch.ones(1, 4, 1)
        output = associative_scan_sequential(decay, values)
        expected = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
        assert torch.allclose(output, expected)

    def test_with_decay(self):
        """With decay<1, values should decay over time."""
        decay = torch.ones(1, 4, 1) * 0.5
        values = torch.ones(1, 4, 1)
        output = associative_scan_sequential(decay, values)
        # s0=1, s1=0.5*1+1=1.5, s2=0.5*1.5+1=1.75, s3=0.5*1.75+1=1.875
        expected = torch.tensor([[[1.0], [1.5], [1.75], [1.875]]])
        assert torch.allclose(output, expected)

    def test_with_initial_state(self):
        decay = torch.ones(1, 3, 1) * 0.5
        values = torch.ones(1, 3, 1)
        initial = torch.tensor([[10.0]])
        output = associative_scan_sequential(decay, values, initial=initial)
        # s0=0.5*10+1=6, s1=0.5*6+1=4, s2=0.5*4+1=3
        expected = torch.tensor([[[6.0], [4.0], [3.0]]])
        assert torch.allclose(output, expected)

    def test_dispatch(self):
        """associative_scan should fall back to sequential on CPU."""
        decay = torch.ones(2, 8, 4) * 0.9
        values = torch.randn(2, 8, 4)
        output = associative_scan(decay, values)
        assert output.shape == (2, 8, 4)


class TestFusedCMSUpdate:
    def test_output_shape(self):
        x = torch.randn(2, 16, 32)
        up_w = torch.randn(64, 32)
        down_w = torch.randn(32, 64)
        out, new_up, new_down = fused_cms_forward_update(x, up_w, down_w, chunk_size=8)
        assert out.shape == (2, 16, 32)
        assert new_up.shape == up_w.shape
        assert new_down.shape == down_w.shape

    def test_weights_change(self):
        """Weights should change after forward+update."""
        x = torch.randn(2, 16, 32)
        up_w = torch.randn(64, 32)
        down_w = torch.randn(32, 64)
        _, new_up, new_down = fused_cms_forward_update(
            x, up_w, down_w, learning_rate=0.01, chunk_size=8,
        )
        assert not torch.allclose(new_up, up_w, atol=1e-6)

    def test_zero_lr_no_change(self):
        """With lr=0, weights should not change."""
        x = torch.randn(2, 16, 32)
        up_w = torch.randn(64, 32)
        down_w = torch.randn(32, 64)
        _, new_up, new_down = fused_cms_forward_update(
            x, up_w, down_w, learning_rate=0.0, chunk_size=8,
        )
        assert torch.allclose(new_up, up_w)
        assert torch.allclose(new_down, down_w)
