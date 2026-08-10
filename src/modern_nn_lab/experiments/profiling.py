"""Computational profiling shared by every track.

The repository compares mechanisms whose *asymptotic* and *measured* costs disagree, so
every comparison must report both. These helpers deliberately separate:

- **capacity**: total and activated parameter counts;
- **measured cost**: wall-clock latency, throughput, peak memory.

Never present a measured number without the hardware descriptor from
:func:`modern_nn_lab.experiments.records.describe_hardware`.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class ParameterProfile:
    """Capacity accounting for one model.

    Attributes:
        total: All parameters, trainable or not.
        trainable: Parameters with ``requires_grad``.
        activated: Parameters used for a single example. Equals ``total`` for dense
            models; for conditional computation such as MoE it is strictly smaller and
            must be reported alongside ``total``.
        buffers: Non-parameter persistent state, for example running statistics.
    """

    total: int
    trainable: int
    activated: int
    buffers: int = 0

    @property
    def sparsity(self) -> float:
        """Fraction of total parameters *not* activated per example."""

        if self.total == 0:
            return 0.0
        return 1.0 - (self.activated / self.total)


def profile_parameters(model: nn.Module, *, activated: int | None = None) -> ParameterProfile:
    """Count parameters and buffers of ``model``.

    Args:
        model: Module to inspect.
        activated: Parameters used per example. Pass explicitly for conditional
            computation; defaults to the total parameter count for dense models.

    Returns:
        A :class:`ParameterProfile`.
    """

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers = sum(buffer.numel() for buffer in model.buffers())
    return ParameterProfile(
        total=total,
        trainable=trainable,
        activated=total if activated is None else activated,
        buffers=buffers,
    )


@dataclass(frozen=True, slots=True)
class LatencyProfile:
    """Measured inference cost.

    Attributes:
        latency_ms_mean: Mean wall-clock latency per forward call, in milliseconds.
        latency_ms_std: Standard deviation across timed repetitions.
        throughput_per_s: Examples processed per second.
        batch_size: Batch size used for the measurement.
        repeats: Number of timed repetitions.
        peak_memory_bytes: Peak allocated accelerator memory, when measurable.
    """

    latency_ms_mean: float
    latency_ms_std: float
    throughput_per_s: float
    batch_size: int
    repeats: int
    peak_memory_bytes: int | None = None


def _synchronize(device: torch.device) -> None:
    """Block until queued accelerator work has finished."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def peak_memory(device: torch.device | str = "cpu") -> Iterator[Callable[[], int | None]]:
    """Track peak allocated memory for the enclosed block.

    Peak memory is only measurable on CUDA. On other devices the returned callable
    yields ``None`` rather than a misleading number.

    Args:
        device: Device to track.

    Yields:
        A callable returning peak bytes allocated during the block, or ``None``.
    """

    resolved = torch.device(device)
    tracked = resolved.type == "cuda" and torch.cuda.is_available()
    if tracked:
        torch.cuda.synchronize(resolved)
        torch.cuda.reset_peak_memory_stats(resolved)

    result: dict[str, int | None] = {"peak": None}

    def read() -> int | None:
        return result["peak"]

    try:
        yield read
    finally:
        if tracked:
            torch.cuda.synchronize(resolved)
            result["peak"] = int(torch.cuda.max_memory_allocated(resolved))


@torch.no_grad()
def profile_inference(
    model: nn.Module,
    example: Tensor,
    *,
    repeats: int = 20,
    warmup: int = 3,
    device: torch.device | str = "cpu",
) -> LatencyProfile:
    """Measure forward-pass latency and throughput.

    The model is put in evaluation mode and gradients are disabled. Warm-up iterations
    are discarded because the first calls include allocator and kernel-selection costs
    that are not representative.

    Args:
        model: Module to time.
        example: A representative input batch, shape ``(B, ...)``.
        repeats: Timed repetitions after warm-up. Must be positive.
        warmup: Untimed warm-up iterations.
        device: Device to run on.

    Returns:
        A :class:`LatencyProfile`.

    Raises:
        ValueError: If ``repeats`` is not positive or ``example`` has no batch dimension.
    """

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if example.ndim == 0:
        raise ValueError("example must have at least a batch dimension")

    resolved = torch.device(device)
    model = model.to(resolved).eval()
    example = example.to(resolved)
    batch_size = int(example.shape[0])

    for _ in range(warmup):
        model(example)
    _synchronize(resolved)

    gc.collect()
    timings: list[float] = []
    with peak_memory(resolved) as read_peak:
        for _ in range(repeats):
            start = time.perf_counter()
            model(example)
            _synchronize(resolved)
            timings.append((time.perf_counter() - start) * 1000.0)
        measured_peak = read_peak()

    times = torch.tensor(timings, dtype=torch.float64)
    mean_ms = float(times.mean())
    std_ms = float(times.std(unbiased=False))
    throughput = float(batch_size / (mean_ms / 1000.0)) if mean_ms > 0 else float("inf")

    return LatencyProfile(
        latency_ms_mean=mean_ms,
        latency_ms_std=std_ms,
        throughput_per_s=throughput,
        batch_size=batch_size,
        repeats=repeats,
        peak_memory_bytes=measured_peak,
    )


class Stopwatch:
    """Context manager measuring wall-clock seconds.

    Example:
        >>> with Stopwatch() as watch:
        ...     _ = sum(range(1000))
        >>> watch.elapsed_s >= 0.0
        True
    """

    def __init__(self) -> None:
        """Create a stopwatch that has not started yet."""

        self._start: float | None = None
        self.elapsed_s: float = 0.0

    def __enter__(self) -> Stopwatch:
        """Start timing."""

        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Stop timing and store the elapsed duration."""

        if self._start is not None:
            self.elapsed_s = time.perf_counter() - self._start


def linear_flops(in_features: int, out_features: int, *, bias: bool = True) -> float:
    """Return multiply-accumulate FLOPs for one linear layer applied to one example.

    A multiply-accumulate is counted as two floating-point operations, which is the
    convention used throughout this repository's FLOP estimates.

    Args:
        in_features: Input width.
        out_features: Output width.
        bias: Whether a bias is added.

    Returns:
        FLOPs per example.
    """

    flops = 2.0 * in_features * out_features
    if bias:
        flops += out_features
    return flops
