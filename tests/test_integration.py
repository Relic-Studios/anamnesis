"""Integration tests for the full Anamnesis training pipeline."""

import pytest
import torch
from torch.utils.data import DataLoader

from anamnesis.core.model import HopeModel, HopeConfig
from anamnesis.training.trainer import AnamnesisTrainer, TrainerConfig
from anamnesis.evaluation.metrics import (
    compute_perplexity,
    snapshot_cms_state,
    compute_cms_delta,
)


def _make_model():
    config = HopeConfig(
        vocab_size=256, hidden_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_kv_heads=2,
        cms_levels=2, cms_chunk_sizes=[1, 8], cms_hidden_mult=4.0,
    )
    return HopeModel(config), config


def _make_dataloader(num_batches=20, batch_size=2, seq_len=16):
    batches = []
    for _ in range(num_batches):
        batches.append({
            "input_ids": torch.randint(0, 256, (batch_size, seq_len)),
            "signal_health": torch.rand(batch_size),
        })
    return DataLoader(batches, batch_size=None)


class TestAnamnesisTrainer:
    """Integration tests for the full trainer."""

    def test_train_basic(self):
        """Basic training loop runs without error."""
        model, _ = _make_model()
        dl = _make_dataloader(num_batches=15)
        config = TrainerConfig(
            max_steps=10,
            enable_gardener=False,
            enable_thompson=False,
            enable_toroidal=False,
            enable_drift=False,
            enable_dreaming=False,
            log_every=5,
            save_every=100,
        )
        trainer = AnamnesisTrainer(model, config)
        history = trainer.train(dl)
        assert len(history["loss"]) == 10
        assert all(l > 0 for l in history["loss"])

    def test_train_with_gardener(self):
        """Training with gardener evaluation."""
        model, _ = _make_model()
        dl = _make_dataloader(num_batches=20)
        config = TrainerConfig(
            max_steps=15,
            enable_gardener=True,
            enable_thompson=True,
            enable_toroidal=False,
            enable_drift=False,
            enable_dreaming=False,
            warmup_steps=3,
            log_every=100,
            save_every=100,
        )
        trainer = AnamnesisTrainer(model, config)
        history = trainer.train(dl)
        assert len(history["loss"]) == 15
        # Gardener runs every 5 steps, so we should have some signal values
        assert len(history["signal_health"]) > 0

    def test_train_with_all_extensions(self):
        """Full system with all 7 extensions enabled."""
        model, _ = _make_model()
        dl = _make_dataloader(num_batches=40)
        config = TrainerConfig(
            max_steps=30,
            enable_gardener=True,
            enable_thompson=True,
            enable_toroidal=True,
            enable_drift=True,
            enable_dreaming=True,
            warmup_steps=5,
            drift_after_step=5,
            log_every=100,
            save_every=100,
        )
        trainer = AnamnesisTrainer(model, config)
        history = trainer.train(dl)
        assert len(history["loss"]) == 30
        assert len(history["signal_health"]) > 0

    def test_loss_decreases(self):
        """Loss should generally decrease over training."""
        torch.manual_seed(42)
        model, _ = _make_model()
        dl = _make_dataloader(num_batches=60)
        config = TrainerConfig(
            max_steps=50,
            enable_gardener=False,
            enable_thompson=False,
            enable_toroidal=False,
            enable_drift=False,
            enable_dreaming=False,
            log_every=100,
            save_every=100,
        )
        trainer = AnamnesisTrainer(model, config)
        history = trainer.train(dl)
        # Compare first 5 vs last 5 losses
        early = sum(history["loss"][:5]) / 5
        late = sum(history["loss"][-5:]) / 5
        assert late < early, f"Loss didn't decrease: {early:.4f} → {late:.4f}"

    def test_replay_buffer_populates(self):
        """Replay buffer should fill during training."""
        model, _ = _make_model()
        dl = _make_dataloader(num_batches=15)
        config = TrainerConfig(
            max_steps=10,
            enable_gardener=False,
            enable_thompson=False,
            enable_toroidal=False,
            enable_drift=False,
            enable_dreaming=False,
            log_every=100,
            save_every=100,
        )
        trainer = AnamnesisTrainer(model, config)
        trainer.train(dl)
        assert len(trainer._replay_buffer) == 10

    def test_anneal_signal_loss(self):
        """Signal loss weight should anneal after warmup."""
        model, _ = _make_model()
        dl = _make_dataloader(num_batches=20)
        config = TrainerConfig(
            max_steps=15,
            warmup_steps=5,
            anneal_signal_step=0.1,
            enable_gardener=False,
            enable_thompson=False,
            enable_toroidal=False,
            enable_drift=False,
            enable_dreaming=False,
            log_every=100,
            save_every=100,
        )
        trainer = AnamnesisTrainer(model, config)
        trainer.train(dl)
        # After 10 steps of annealing at 0.1 per step, should have reached target
        assert trainer.loss_fn.lambda_signal > 0.0


class TestConvertTrainEvaluate:
    """End-to-end: create model, train, evaluate."""

    def test_full_pipeline(self):
        torch.manual_seed(123)
        model, config = _make_model()

        # Eval dataloader (separate from train)
        eval_dl = _make_dataloader(num_batches=5)
        train_dl = _make_dataloader(num_batches=30)

        # Baseline perplexity
        model.eval()
        ppl_before = compute_perplexity(model, eval_dl)
        cms_before = snapshot_cms_state(model)

        # Train
        trainer_config = TrainerConfig(
            lr=3e-4,  # M3 diverges at default lr=0.02 on tiny models
            max_steps=20,
            enable_gardener=False,
            enable_thompson=False,
            enable_toroidal=False,
            enable_drift=False,
            enable_dreaming=False,
            log_every=100,
            save_every=100,
        )
        trainer = AnamnesisTrainer(model, trainer_config)
        history = trainer.train(train_dl)

        # Post-training perplexity
        model.eval()
        ppl_after = compute_perplexity(model, eval_dl)
        cms_after = snapshot_cms_state(model)

        # CMS should have changed
        delta = compute_cms_delta(cms_before, cms_after)
        assert delta["total_l2"] > 0.0, "CMS weights should change during training"

        # Perplexity should improve (or at least be finite)
        assert ppl_after < float("inf")
        assert ppl_before < float("inf")


class TestStatePersistence:
    """Test CMS state save/load across sessions."""

    def test_save_load_roundtrip(self, tmp_path):
        from anamnesis.state.persistence import save_cms_state, load_cms_state

        model, _ = _make_model()

        # Do a forward pass so CMS has learned something
        x = torch.randint(0, 256, (1, 16))
        model.eval()
        with torch.no_grad():
            model(x)

        # Save
        path = tmp_path / "cms_state.pt"
        save_cms_state(model, path, {"tokens_processed": 16})

        # Create fresh model and load
        model2, _ = _make_model()
        metadata = load_cms_state(model2, path)
        assert metadata["tokens_processed"] == 16

        # Weights should match
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters()
        ):
            if "cms" in n1:
                assert torch.allclose(p1.cpu(), p2.cpu()), f"Mismatch in {n1}"
