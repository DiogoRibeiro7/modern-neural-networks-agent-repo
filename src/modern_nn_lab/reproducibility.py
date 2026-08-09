"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch pseudo-random generators.

    Args:
        seed: Non-negative integer seed.
        deterministic: If true, request deterministic PyTorch algorithms. This can
            reduce performance and may raise when a deterministic implementation is
            unavailable.

    Raises:
        ValueError: If ``seed`` is negative.
    """

    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic)
