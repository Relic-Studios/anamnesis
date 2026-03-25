"""Tests for Delta Gradient Descent."""

import pytest
import torch

from anamnesis.core.dgd import DeltaGradientDescent


class TestDeltaGradientDescent:
    """Tests for the DGD update rule."""

    def test_compute_update_shape(self):
        dgd = DeltaGradientDescent(dim=64)
        weight = torch.randn(128, 64)
        input_x = torch.randn(64)
        grad = torch.randn(128, 64)
        new_weight = dgd.compute_update(weight, input_x, grad)
        assert new_weight.shape == weight.shape

    def test_compute_update_batched(self):
        dgd = DeltaGradientDescent(dim=64)
        weight = torch.randn(128, 64)
        input_x = torch.randn(8, 64)  # batch of 8
        grad = torch.randn(128, 64)
        new_weight = dgd.compute_update(weight, input_x, grad)
        assert new_weight.shape == weight.shape

    def test_zero_alpha_recovers_sgd(self):
        """With α=0, DGD reduces to standard gradient descent."""
        dgd = DeltaGradientDescent(dim=64, normalize_inputs=False)
        weight = torch.randn(128, 64)
        input_x = torch.randn(64)
        grad = torch.randn(128, 64)
        lr = 0.01

        new_weight_dgd = dgd.compute_update(weight, input_x, grad, lr=lr, alpha=0.0)
        new_weight_sgd = weight - lr * grad

        assert torch.allclose(new_weight_dgd, new_weight_sgd, atol=1e-6)

    def test_nonzero_alpha_differs_from_sgd(self):
        """With α>0, DGD should produce different results than standard GD."""
        dgd = DeltaGradientDescent(dim=64)
        weight = torch.randn(128, 64)
        input_x = torch.randn(64)
        grad = torch.randn(128, 64)
        lr = 0.01

        new_weight_dgd = dgd.compute_update(weight, input_x, grad, lr=lr, alpha=0.1)
        new_weight_sgd = weight - lr * grad

        assert not torch.allclose(new_weight_dgd, new_weight_sgd, atol=1e-4)

    def test_associative_loss(self):
        """Test the associative memory loss computation."""
        output = torch.randn(8, 64)
        target = torch.randn(8, 64)
        loss = DeltaGradientDescent.compute_associative_loss(output, target)
        assert loss.ndim == 0  # scalar
        assert loss.item() >= 0  # MSE is non-negative

    def test_associative_loss_zero_for_identical(self):
        x = torch.randn(8, 64)
        loss = DeltaGradientDescent.compute_associative_loss(x, x)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_data_dependent_decay_is_directional(self):
        """The decay should be stronger in the direction of the input."""
        dgd = DeltaGradientDescent(dim=4, normalize_inputs=True)
        weight = torch.eye(4)

        # Input along first dimension
        input_x = torch.tensor([1.0, 0.0, 0.0, 0.0])
        grad = torch.zeros(4, 4)

        new_weight = dgd.compute_update(weight, input_x, grad, lr=0.0, alpha=0.5)

        # The first column should be decayed more than others
        # because the decay is (I - 0.5 * e1 * e1^T) = diag(0.5, 1, 1, 1)
        assert new_weight[0, 0] < weight[0, 0]  # decayed
        assert torch.allclose(new_weight[1, 1], weight[1, 1], atol=1e-6)  # unchanged
        assert torch.allclose(new_weight[2, 2], weight[2, 2], atol=1e-6)  # unchanged
