"""Split construction and leakage-safe preprocessing."""

from __future__ import annotations

import pytest
import torch

from modern_nn_lab.experiments.data import (
    make_function_split,
    make_tabular_split,
    random_split_indices,
    standardize,
)


def test_split_indices_are_disjoint_and_complete() -> None:
    train, val, test = random_split_indices(100, seed=3)
    combined = torch.cat([train, val, test])
    assert combined.numel() == 100
    assert torch.equal(torch.sort(combined).values, torch.arange(100))


def test_split_indices_are_deterministic_per_seed() -> None:
    first = random_split_indices(50, seed=11)
    second = random_split_indices(50, seed=11)
    third = random_split_indices(50, seed=12)
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    assert not all(torch.equal(a, b) for a, b in zip(first, third, strict=True))


def test_split_indices_validate_arguments() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        random_split_indices(2)
    with pytest.raises(ValueError, match="positive"):
        random_split_indices(10, ratios=(0.0, 0.5, 0.5))
    with pytest.raises(ValueError, match="sum to 1"):
        random_split_indices(10, ratios=(0.5, 0.4, 0.4))


def test_standardize_uses_training_statistics_only() -> None:
    train = torch.tensor([[0.0], [2.0]])
    held_out = torch.tensor([[10.0]])
    scaled_train, (scaled_held_out,) = standardize(train, held_out)

    assert scaled_train.mean().abs() < 1e-6
    # Held-out data must NOT be re-centred on its own mean; that would leak.
    assert scaled_held_out.abs().item() > 1.0


def test_standardize_survives_constant_features() -> None:
    train = torch.ones(5, 2)
    scaled, _ = standardize(train)
    assert torch.isfinite(scaled).all()


def test_function_split_shapes_and_determinism() -> None:
    split = make_function_split(
        lambda x: torch.sin(x[:, :1]), name="sine", input_dim=1, n_samples=200, seed=5
    )
    n_train, n_val, n_test = split.sizes()
    assert n_train + n_val + n_test == 200
    assert split.train_targets.shape[-1] == 1
    assert split.input_dim == 1

    repeat = make_function_split(
        lambda x: torch.sin(x[:, :1]), name="sine", input_dim=1, n_samples=200, seed=5
    )
    assert split.fingerprint == repeat.fingerprint


def test_function_split_noise_changes_the_fingerprint() -> None:
    clean = make_function_split(lambda x: x, name="id", input_dim=1, n_samples=100, seed=1)
    noisy = make_function_split(
        lambda x: x, name="id", input_dim=1, n_samples=100, seed=1, noise_std=0.1
    )
    assert clean.fingerprint != noisy.fingerprint


def test_function_split_validates_arguments() -> None:
    with pytest.raises(ValueError, match="input_dim"):
        make_function_split(lambda x: x, name="x", input_dim=0)
    with pytest.raises(ValueError, match="n_samples"):
        make_function_split(lambda x: x, name="x", input_dim=1, n_samples=0)
    with pytest.raises(ValueError, match="noise_std"):
        make_function_split(lambda x: x, name="x", input_dim=1, noise_std=-1.0)
    with pytest.raises(ValueError, match="low"):
        make_function_split(lambda x: x, name="x", input_dim=1, low=1.0, high=0.0)


def test_tabular_split_standardizes_features_on_train_only() -> None:
    inputs = torch.randn(200, 4) * 5.0 + 3.0
    targets = torch.randint(0, 2, (200,))
    split = make_tabular_split(inputs, targets, name="synthetic-tabular", seed=2)

    assert split.train_inputs.mean().abs() < 1e-5
    assert split.train_targets.dtype == torch.int64  # labels are never standardized
    assert "standardized on train" in split.strategy


def test_tabular_split_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="first dimension"):
        make_tabular_split(torch.randn(10, 2), torch.randn(9), name="bad")
