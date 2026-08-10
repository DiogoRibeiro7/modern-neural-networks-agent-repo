"""Synthetic sequence diagnostics shared by the sequence-model tracks.

Every task here is designed so that a *specific* mechanism is required to solve it, and so
that a model without that mechanism fails in a legible way rather than degrading
gracefully. All tasks are causal: the label at position ``t`` never depends on inputs
after ``t``, which is what makes the causality tests in each track meaningful.

Conventions shared by every task:

- inputs are integer token ids of shape ``(B, T)``;
- targets are integer token ids of shape ``(B, T)``;
- positions that should not contribute to the loss are marked in a boolean mask of shape
  ``(B, T)``, where ``True`` means *scored*.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from modern_nn_lab.experiments.records import tensor_fingerprint

IGNORE_INDEX = -100
"""Target value marking a position that must not contribute to the loss."""


@dataclass(frozen=True, slots=True)
class SequenceSplit:
    """A train/validation/test split of a synthetic sequence task.

    Attributes:
        name: Task identifier recorded with results.
        vocab_size: Number of distinct token ids, including any special tokens.
        seq_len: Sequence length ``T``.
        train_inputs: Shape ``(N_train, T)`` integer tokens.
        train_targets: Shape ``(N_train, T)`` integer targets, ``IGNORE_INDEX`` where unscored.
        val_inputs: Shape ``(N_val, T)``.
        val_targets: Shape ``(N_val, T)``.
        test_inputs: Shape ``(N_test, T)``.
        test_targets: Shape ``(N_test, T)``.
        strategy: Description of how the split was produced.
        metadata: JSON-serializable task parameters.
    """

    name: str
    vocab_size: int
    seq_len: int
    train_inputs: Tensor
    train_targets: Tensor
    val_inputs: Tensor
    val_targets: Tensor
    test_inputs: Tensor
    test_targets: Tensor
    strategy: str
    metadata: dict[str, object]

    @property
    def fingerprint(self) -> str:
        """Content fingerprint over every tensor in the split."""

        return tensor_fingerprint(
            self.train_inputs,
            self.train_targets,
            self.val_inputs,
            self.val_targets,
            self.test_inputs,
            self.test_targets,
        )

    def sizes(self) -> tuple[int, int, int]:
        """Return ``(n_train, n_val, n_test)``."""

        return (
            int(self.train_inputs.shape[0]),
            int(self.val_inputs.shape[0]),
            int(self.test_inputs.shape[0]),
        )


def _split_counts(n_sequences: int) -> tuple[int, int, int]:
    """Split a sequence count 70/15/15, keeping every part non-empty."""

    if n_sequences < 3:
        raise ValueError("n_sequences must be at least 3")
    n_train = max(1, round(0.7 * n_sequences))
    n_val = max(1, round(0.15 * n_sequences))
    n_train = min(n_train, n_sequences - 2)
    n_val = min(n_val, n_sequences - n_train - 1)
    return n_train, n_val, n_sequences - n_train - n_val


def _assemble(
    name: str,
    inputs: Tensor,
    targets: Tensor,
    *,
    vocab_size: int,
    strategy: str,
    metadata: dict[str, object],
) -> SequenceSplit:
    """Cut generated sequences into disjoint train/validation/test splits."""

    n_train, n_val, _ = _split_counts(int(inputs.shape[0]))
    return SequenceSplit(
        name=name,
        vocab_size=vocab_size,
        seq_len=int(inputs.shape[1]),
        train_inputs=inputs[:n_train],
        train_targets=targets[:n_train],
        val_inputs=inputs[n_train : n_train + n_val],
        val_targets=targets[n_train : n_train + n_val],
        test_inputs=inputs[n_train + n_val :],
        test_targets=targets[n_train + n_val :],
        strategy=strategy,
        metadata=metadata,
    )


def make_copy_task(
    *,
    n_sequences: int = 2000,
    payload_len: int = 8,
    delay: int = 16,
    n_symbols: int = 8,
    seed: int = 1729,
) -> SequenceSplit:
    """Copy a payload after a delay.

    Layout of one sequence, with ``P`` the payload length and ``D`` the delay::

        [ payload (P) ][ blanks (D) ][ cue ][ blanks (P) ]
                                             ^ scored positions

    The model must hold ``P`` symbols across ``D`` uninformative steps and emit them in
    order once the cue arrives. A model with no persistent state cannot do better than
    chance on the scored positions.

    Args:
        n_sequences: Total sequences before splitting.
        payload_len: Number of symbols to memorize.
        delay: Uninformative steps between payload and cue.
        n_symbols: Size of the payload alphabet.
        seed: Generator seed.

    Returns:
        A :class:`SequenceSplit`. Vocabulary is ``n_symbols + 2``: symbols, then a blank
        token, then a cue token.

    Raises:
        ValueError: If any argument is not positive.
    """

    if min(n_sequences, payload_len, delay, n_symbols) <= 0:
        raise ValueError("all task dimensions must be positive")

    blank = n_symbols
    cue = n_symbols + 1
    vocab_size = n_symbols + 2
    seq_len = payload_len + delay + 1 + payload_len

    generator = torch.Generator().manual_seed(seed)
    payload = torch.randint(0, n_symbols, (n_sequences, payload_len), generator=generator)

    inputs = torch.full((n_sequences, seq_len), blank, dtype=torch.long)
    inputs[:, :payload_len] = payload
    inputs[:, payload_len + delay] = cue

    targets = torch.full((n_sequences, seq_len), IGNORE_INDEX, dtype=torch.long)
    answer_start = payload_len + delay + 1
    targets[:, answer_start : answer_start + payload_len] = payload

    return _assemble(
        "copy",
        inputs,
        targets,
        vocab_size=vocab_size,
        strategy=f"iid sequences, 70/15/15 split, seq_len={seq_len}",
        metadata={
            "task": "copy",
            "payload_len": payload_len,
            "delay": delay,
            "n_symbols": n_symbols,
            "seq_len": seq_len,
            "chance_accuracy": 1.0 / n_symbols,
        },
    )


def make_selective_recall_task(
    *,
    n_sequences: int = 2000,
    n_pairs: int = 8,
    n_keys: int = 12,
    n_values: int = 12,
    seed: int = 1729,
) -> SequenceSplit:
    """Associative recall: read back the value bound to a queried key.

    Layout::

        [ k1 v1 k2 v2 ... kP vP ][ query = k_j ][ answer ]
                                                 ^ scored

    Unlike the copy task this requires *selective* retrieval: only one of the stored
    pairs matters, and which one is revealed at the very end. A model that compresses the
    whole prefix into an undifferentiated summary cannot answer reliably.

    Args:
        n_sequences: Total sequences before splitting.
        n_pairs: Key-value pairs stored per sequence.
        n_keys: Size of the key alphabet; must be at least ``n_pairs`` so keys are unique.
        n_values: Size of the value alphabet.
        seed: Generator seed.

    Returns:
        A :class:`SequenceSplit`. Keys occupy ids ``[0, n_keys)`` and values
        ``[n_keys, n_keys + n_values)``.

    Raises:
        ValueError: If any dimension is not positive or ``n_keys < n_pairs``.
    """

    if min(n_sequences, n_pairs, n_keys, n_values) <= 0:
        raise ValueError("all task dimensions must be positive")
    if n_keys < n_pairs:
        raise ValueError("n_keys must be at least n_pairs so that keys stay unique")

    vocab_size = n_keys + n_values
    seq_len = 2 * n_pairs + 2

    generator = torch.Generator().manual_seed(seed)
    # Unique keys per sequence: an independent permutation prefix for each row.
    keys = torch.stack(
        [torch.randperm(n_keys, generator=generator)[:n_pairs] for _ in range(n_sequences)]
    )
    values = torch.randint(0, n_values, (n_sequences, n_pairs), generator=generator) + n_keys
    query_index = torch.randint(0, n_pairs, (n_sequences,), generator=generator)

    inputs = torch.zeros((n_sequences, seq_len), dtype=torch.long)
    inputs[:, 0 : 2 * n_pairs : 2] = keys
    inputs[:, 1 : 2 * n_pairs : 2] = values

    rows = torch.arange(n_sequences)
    queried_key = keys[rows, query_index]
    queried_value = values[rows, query_index]
    inputs[:, 2 * n_pairs] = queried_key
    inputs[:, 2 * n_pairs + 1] = queried_key  # the answer position re-presents the key

    targets = torch.full((n_sequences, seq_len), IGNORE_INDEX, dtype=torch.long)
    targets[:, 2 * n_pairs + 1] = queried_value

    return _assemble(
        "selective_recall",
        inputs,
        targets,
        vocab_size=vocab_size,
        strategy=f"iid sequences, 70/15/15 split, seq_len={seq_len}",
        metadata={
            "task": "selective_recall",
            "n_pairs": n_pairs,
            "n_keys": n_keys,
            "n_values": n_values,
            "seq_len": seq_len,
            "chance_accuracy": 1.0 / n_values,
        },
    )


def make_state_tracking_task(
    *,
    n_sequences: int = 2000,
    seq_len: int = 64,
    n_states: int = 2,
    seed: int = 1729,
) -> SequenceSplit:
    """Track a running modular sum: the canonical state-tracking probe.

    At every position the model must output the running sum of all inputs seen so far,
    modulo ``n_states``. With ``n_states = 2`` this is running parity. The target at
    position ``t`` depends on the entire prefix, so the task is unsolvable without a
    state that is carried forward and updated multiplicatively.

    Every position is scored, which makes it a strict test: a model that solves the task
    only near the start will show it.

    Args:
        n_sequences: Total sequences before splitting.
        seq_len: Sequence length.
        n_states: Modulus of the running sum.
        seed: Generator seed.

    Returns:
        A :class:`SequenceSplit` whose vocabulary is ``n_states``.

    Raises:
        ValueError: If any argument is not positive or ``n_states`` is below 2.
    """

    if min(n_sequences, seq_len) <= 0:
        raise ValueError("all task dimensions must be positive")
    if n_states < 2:
        raise ValueError("n_states must be at least 2")

    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randint(0, n_states, (n_sequences, seq_len), generator=generator)
    targets = torch.cumsum(inputs, dim=1) % n_states

    return _assemble(
        "state_tracking",
        inputs,
        targets,
        vocab_size=n_states,
        strategy=f"iid sequences, 70/15/15 split, seq_len={seq_len}",
        metadata={
            "task": "state_tracking",
            "n_states": n_states,
            "seq_len": seq_len,
            "chance_accuracy": 1.0 / n_states,
        },
    )


def make_rebinding_task(
    *,
    n_sequences: int = 500,
    n_pairs: int = 3,
    n_keys: int = 6,
    n_values: int = 6,
    seed: int = 1729,
    name: str = "rebinding",
) -> SequenceSplit:
    """Associative recall where the key-value mapping is *overwritten* mid-sequence.

    Layout, with ``P`` pairs::

        [ k1 a1 ... kP aP ][ q  ans_a ][ k1 b1 ... kP bP ][ q  ans_b ]
                             ^ scored                       ^ scored

    The same keys appear twice with different values. The first query is ordinary
    associative recall; the second requires *unlearning* the earlier binding within the
    same sequence, using evidence that arrived after the first answer.

    This is the diagnostic for adaptation to an abrupt distribution change inside a
    sequence. A model whose state can only accumulate will answer the second query with
    the first value; a model that can revise its state will not. Scoring the two answers
    separately is what makes the distinction visible, so the second answer's accuracy is
    reported as a secondary metric by the track suite.

    Args:
        n_sequences: Total sequences before splitting.
        n_pairs: Key-value pairs, bound twice.
        n_keys: Key alphabet size; at least ``n_pairs`` so keys stay unique.
        n_values: Value alphabet size.
        seed: Generator seed.
        name: Dataset label. Give scaled variants distinct names so they are legible in
            reports; record filenames additionally carry a content fingerprint.

    Returns:
        A :class:`SequenceSplit`. Keys occupy ``[0, n_keys)`` and values
        ``[n_keys, n_keys + n_values)``.

    Raises:
        ValueError: If a dimension is not positive or ``n_keys < n_pairs``.
    """

    if min(n_sequences, n_pairs, n_keys, n_values) <= 0:
        raise ValueError("all task dimensions must be positive")
    if n_keys < n_pairs:
        raise ValueError("n_keys must be at least n_pairs so that keys stay unique")

    vocab_size = n_keys + n_values
    block = 2 * n_pairs
    seq_len = 2 * block + 4

    generator = torch.Generator().manual_seed(seed)
    keys = torch.stack(
        [torch.randperm(n_keys, generator=generator)[:n_pairs] for _ in range(n_sequences)]
    )
    first_values = torch.randint(0, n_values, (n_sequences, n_pairs), generator=generator) + n_keys
    second_values = torch.randint(0, n_values, (n_sequences, n_pairs), generator=generator) + n_keys
    query_index = torch.randint(0, n_pairs, (n_sequences,), generator=generator)

    inputs = torch.zeros((n_sequences, seq_len), dtype=torch.long)
    inputs[:, 0:block:2] = keys
    inputs[:, 1:block:2] = first_values

    rows = torch.arange(n_sequences)
    queried_key = keys[rows, query_index]
    inputs[:, block] = queried_key
    inputs[:, block + 1] = queried_key

    rebind = block + 2
    inputs[:, rebind : rebind + block : 2] = keys
    inputs[:, rebind + 1 : rebind + block : 2] = second_values
    inputs[:, rebind + block] = queried_key
    inputs[:, rebind + block + 1] = queried_key

    targets = torch.full((n_sequences, seq_len), IGNORE_INDEX, dtype=torch.long)
    targets[:, block + 1] = first_values[rows, query_index]
    targets[:, rebind + block + 1] = second_values[rows, query_index]

    return _assemble(
        name,
        inputs,
        targets,
        vocab_size=vocab_size,
        strategy=f"iid sequences, 70/15/15 split, seq_len={seq_len}",
        metadata={
            "task": "rebinding",
            "n_pairs": n_pairs,
            "n_keys": n_keys,
            "n_values": n_values,
            "seq_len": seq_len,
            "first_answer_index": block + 1,
            "second_answer_index": rebind + block + 1,
            "chance_accuracy": 1.0 / n_values,
        },
    )


def make_needle_task(
    *,
    n_sequences: int = 500,
    n_pairs: int = 8,
    needle_index: int = 0,
    n_keys: int = 12,
    n_values: int = 8,
    repeats: int = 1,
    seed: int = 1729,
    name: str | None = None,
) -> SequenceSplit:
    """Retrieve one fact planted at a controlled distance from the query.

    Layout::

        [ k1 v1  k2 v2  ...  kP vP ][ q  ans ]
                 ^ the queried pair sits at `needle_index`      ^ scored

    Unlike :func:`make_selective_recall_task`, which queries a *random* pair, this task
    fixes *which* pair is asked for. Sweeping ``needle_index`` therefore sweeps the
    distance between writing a fact and being asked for it, which is what turns a single
    accuracy number into a **forgetting curve**.

    ``needle_index = 0`` puts the fact furthest from the query; the final index puts it
    adjacent. A model whose only memory is a sliding window of size ``W`` can answer only
    when the distance is below ``W``, so the curve's shape distinguishes "has long-term
    memory" from "has a long enough short-term window".

    Args:
        n_sequences: Total sequences before splitting.
        n_pairs: Key-value pairs stored per sequence.
        needle_index: Which pair is queried, counting from the start.
        n_keys: Key alphabet size; at least ``n_pairs`` so keys stay unique.
        n_values: Value alphabet size.
        repeats: How many times the queried pair is written. ``1`` is a one-off fact;
            higher values make it a repeated fact, which tests whether repetition
            strengthens a memory the way the source's surprise metric implies it should.
        seed: Generator seed.
        name: Dataset label. Defaults to ``needle-d{distance}``.

    Returns:
        A :class:`SequenceSplit` whose metadata carries ``distance``, the number of tokens
        between the needle's value and the query.

    Raises:
        ValueError: If a dimension is invalid, ``needle_index`` is out of range, or
            ``repeats`` exceeds the available slots.
    """

    if min(n_sequences, n_pairs, n_keys, n_values, repeats) <= 0:
        raise ValueError("all task dimensions must be positive")
    if n_keys < n_pairs:
        raise ValueError("n_keys must be at least n_pairs so that keys stay unique")
    if not 0 <= needle_index < n_pairs:
        raise ValueError(f"needle_index must lie in [0, {n_pairs}), got {needle_index}")
    if repeats > n_pairs - needle_index:
        raise ValueError("repeats exceeds the slots available after needle_index")

    vocab_size = n_keys + n_values
    seq_len = 2 * n_pairs + 2

    generator = torch.Generator().manual_seed(seed)
    keys = torch.stack(
        [torch.randperm(n_keys, generator=generator)[:n_pairs] for _ in range(n_sequences)]
    )
    values = torch.randint(0, n_values, (n_sequences, n_pairs), generator=generator) + n_keys

    # A repeated fact reuses the same key and value in the following slots.
    for offset in range(1, repeats):
        keys[:, needle_index + offset] = keys[:, needle_index]
        values[:, needle_index + offset] = values[:, needle_index]

    inputs = torch.zeros((n_sequences, seq_len), dtype=torch.long)
    inputs[:, 0 : 2 * n_pairs : 2] = keys
    inputs[:, 1 : 2 * n_pairs : 2] = values

    queried_key = keys[:, needle_index]
    inputs[:, 2 * n_pairs] = queried_key
    inputs[:, 2 * n_pairs + 1] = queried_key

    targets = torch.full((n_sequences, seq_len), IGNORE_INDEX, dtype=torch.long)
    targets[:, 2 * n_pairs + 1] = values[:, needle_index]

    # Tokens between the needle's value and the answer slot.
    distance = seq_len - 1 - (2 * needle_index + 1)

    return _assemble(
        name or f"needle-d{distance}",
        inputs,
        targets,
        vocab_size=vocab_size,
        strategy=f"iid sequences, 70/15/15 split, seq_len={seq_len}",
        metadata={
            "task": "needle",
            "n_pairs": n_pairs,
            "needle_index": needle_index,
            "distance": distance,
            "repeats": repeats,
            "n_keys": n_keys,
            "n_values": n_values,
            "seq_len": seq_len,
            "answer_index": 2 * n_pairs + 1,
            "chance_accuracy": 1.0 / n_values,
        },
    )


def masked_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Cross-entropy over scored positions only.

    Args:
        logits: Shape ``(B, T, V)`` unnormalized scores.
        targets: Shape ``(B, T)`` token ids, ``IGNORE_INDEX`` where unscored.

    Returns:
        Scalar loss averaged over scored positions.

    Raises:
        ValueError: If shapes are inconsistent.
    """

    if logits.ndim != 3 or targets.ndim != 2:
        raise ValueError(
            f"expected logits (B, T, V) and targets (B, T), got "
            f"{tuple(logits.shape)} and {tuple(targets.shape)}"
        )

    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


def masked_accuracy(logits: Tensor, targets: Tensor) -> float:
    """Top-1 accuracy over scored positions only.

    Args:
        logits: Shape ``(B, T, V)`` unnormalized scores.
        targets: Shape ``(B, T)`` token ids, ``IGNORE_INDEX`` where unscored.

    Returns:
        Fraction of scored positions predicted correctly, or ``0.0`` if none are scored.
    """

    scored = targets != IGNORE_INDEX
    if not bool(scored.any()):
        return 0.0
    predicted = logits.argmax(dim=-1)
    return float((predicted[scored] == targets[scored]).float().mean())
