"""Tests for shared reproducibility helpers."""

import numpy as np
import pytest
import torch

from modern_nn_lab.reproducibility import seed_everything


def test_seed_everything_repeats_numpy_and_torch_draws() -> None:
    """Resetting the same seed should repeat pseudo-random draws."""

    seed_everything(1729)
    numpy_first = np.random.random(4)
    torch_first = torch.rand(4)

    seed_everything(1729)
    numpy_second = np.random.random(4)
    torch_second = torch.rand(4)

    np.testing.assert_allclose(numpy_first, numpy_second)
    torch.testing.assert_close(torch_first, torch_second)


def test_negative_seed_is_rejected() -> None:
    """Negative seeds are invalid by repository contract."""

    with pytest.raises(ValueError, match="non-negative"):
        seed_everything(-1)
