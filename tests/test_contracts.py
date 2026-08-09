"""Tests for shared contracts."""

from torch import nn

from modern_nn_lab.contracts import count_trainable_parameters


def test_count_trainable_parameters_ignores_frozen_parameters() -> None:
    """Only parameters optimized by gradient descent count toward the trainable budget."""

    model = nn.Linear(3, 2, bias=True)
    model.bias.requires_grad_(False)
    assert count_trainable_parameters(model) == 6
