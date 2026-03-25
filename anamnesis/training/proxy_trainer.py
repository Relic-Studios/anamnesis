"""
Signal Proxy Trainer — trains the differentiable signal approximation.

The signal proxy network learns to predict Didymus signal health scores
from model hidden states. This is a prerequisite for the gardener stream
and the inner-loop signal loss.

Training data: (text, signal_health) pairs from Didymus logs.
The proxy is small (~100K params) and trains quickly.

Anti-gaming: periodically validate against the real Didymus scorer.
If proxy-real divergence exceeds threshold, retrain.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from anamnesis.active_inference.free_energy import SignalProxy
from anamnesis.training.data import ConversationDataset


class SignalProxyTrainer:
    """
    Trains a SignalProxy network on Didymus signal annotations.

    The proxy learns: hidden_states → signal_health

    Since we don't have access to the full model's hidden states during
    proxy training, we use a lightweight text encoder (the model's own
    embeddings) to produce proxy inputs.

    Args:
        proxy: The SignalProxy network to train.
        lr: Learning rate.
        epochs: Number of training epochs.
        batch_size: Training batch size.
    """

    def __init__(
        self,
        proxy: SignalProxy,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 32,
    ):
        self.proxy = proxy
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = torch.optim.AdamW(proxy.parameters(), lr=lr)

    def train(
        self,
        dataset: ConversationDataset,
        embed_fn=None,
        verbose: bool = True,
    ) -> dict[str, float]:
        """
        Train the proxy on signal-annotated data.

        Args:
            dataset: ConversationDataset with signal annotations.
            embed_fn: Function that converts input_ids → hidden states.
                If None, uses random embeddings (for testing).
            verbose: Print progress.

        Returns:
            Dict with training metrics.
        """
        self.proxy.train()
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True,
        )

        total_loss = 0.0
        total_steps = 0

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_steps = 0

            for batch in dataloader:
                signal_target = batch["signal_health"]  # (batch,)

                # Get hidden states
                if embed_fn is not None and "input_ids" in batch:
                    with torch.no_grad():
                        hidden = embed_fn(batch["input_ids"])
                else:
                    # Fallback: use signal facets as proxy input
                    # Pad to match proxy input dim
                    facets = batch["signal_facets"]  # (batch, 5)
                    hidden = facets.unsqueeze(1)  # (batch, 1, 5)
                    # Pad to proxy dim
                    pad_size = self.proxy.dim - facets.shape[-1]
                    if pad_size > 0:
                        hidden = torch.nn.functional.pad(hidden, (0, pad_size))

                # Forward
                predicted = self.proxy(hidden)
                loss = nn.functional.mse_loss(predicted, signal_target)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_steps += 1

            avg_loss = epoch_loss / max(epoch_steps, 1)
            total_loss += epoch_loss
            total_steps += epoch_steps

            if verbose:
                print(f"  Proxy epoch {epoch + 1}/{self.epochs} | Loss: {avg_loss:.6f}")

        return {
            "avg_loss": total_loss / max(total_steps, 1),
            "total_steps": total_steps,
        }

    def validate(
        self,
        dataset: ConversationDataset,
        embed_fn=None,
    ) -> dict[str, float]:
        """
        Validate proxy predictions against real signal scores.

        Args:
            dataset: Validation dataset with real signal annotations.
            embed_fn: Embedding function.

        Returns:
            Dict with validation metrics (MAE, correlation, max_divergence).
        """
        self.proxy.eval()
        predictions = []
        targets = []

        with torch.no_grad():
            for i in range(len(dataset)):
                example = dataset[i]
                signal_target = example["signal_health"].item()

                facets = example["signal_facets"].unsqueeze(0).unsqueeze(0)
                pad_size = self.proxy.dim - facets.shape[-1]
                if pad_size > 0:
                    facets = torch.nn.functional.pad(facets, (0, pad_size))

                predicted = self.proxy(facets).item()
                predictions.append(predicted)
                targets.append(signal_target)

        if not predictions:
            return {"mae": 0.0, "max_divergence": 0.0}

        mae = sum(abs(p - t) for p, t in zip(predictions, targets)) / len(predictions)
        max_div = max(abs(p - t) for p, t in zip(predictions, targets))

        return {
            "mae": mae,
            "max_divergence": max_div,
            "num_examples": len(predictions),
        }
