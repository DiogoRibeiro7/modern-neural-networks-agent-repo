"""Mathematical invariants of the xLSTM track.

The tests that matter here are the ones a forward-pass smoke test cannot catch: that no
model can see the future, that the stabilized recurrence equals the textbook recurrence on
a hand-computable case, that the state stays finite over long sequences, and that the
gating ablation changes exactly one thing.
"""

from __future__ import annotations

import math

import pytest
import torch

from modern_nn_lab.experiments.tasks.sequence import (
    IGNORE_INDEX,
    make_copy_task,
    make_selective_recall_task,
    make_state_tracking_task,
    masked_accuracy,
    masked_cross_entropy,
)
from modern_nn_lab.tracks.xlstm import (
    XLSTM,
    CausalTransformer,
    MLSTMCell,
    RecurrentBaseline,
    SLSTMCell,
    XLSTMConfig,
    count_parameters,
    match_width_to_budget,
)

# --------------------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------------------


def test_copy_task_scores_only_the_answer_span() -> None:
    split = make_copy_task(n_sequences=60, payload_len=4, delay=6, n_symbols=5, seed=0)
    scored = split.train_targets != IGNORE_INDEX
    assert int(scored.sum(dim=1).unique().item()) == 4
    # The scored span sits strictly after the cue.
    first_scored = scored.float().argmax(dim=1)
    assert int(first_scored.unique().item()) == 4 + 6 + 1


def test_copy_task_answer_matches_the_payload() -> None:
    split = make_copy_task(n_sequences=40, payload_len=3, delay=5, n_symbols=6, seed=1)
    payload = split.train_inputs[:, :3]
    answers = split.train_targets[:, -3:]
    assert torch.equal(payload, answers)


def test_selective_recall_answer_is_the_value_bound_to_the_query() -> None:
    split = make_selective_recall_task(n_sequences=50, n_pairs=4, n_keys=6, n_values=7, seed=2)
    inputs, targets = split.train_inputs, split.train_targets
    keys = inputs[:, 0:8:2]
    values = inputs[:, 1:8:2]
    query = inputs[:, 8]
    answer = targets[:, 9]

    for row in range(inputs.shape[0]):
        position = int((keys[row] == query[row]).nonzero()[0])
        assert int(answer[row]) == int(values[row, position])


def test_selective_recall_keys_are_unique_within_a_sequence() -> None:
    split = make_selective_recall_task(n_sequences=30, n_pairs=5, n_keys=9, n_values=5, seed=3)
    keys = split.train_inputs[:, 0:10:2]
    for row in range(keys.shape[0]):
        assert len(set(keys[row].tolist())) == 5


def test_state_tracking_target_is_the_running_modular_sum() -> None:
    split = make_state_tracking_task(n_sequences=20, seq_len=12, n_states=3, seed=4)
    expected = torch.cumsum(split.train_inputs, dim=1) % 3
    assert torch.equal(split.train_targets, expected)


def test_state_tracking_targets_depend_on_the_whole_prefix() -> None:
    # Flipping the first token must change every later target: this is what makes the
    # task unsolvable without carried state.
    split = make_state_tracking_task(n_sequences=8, seq_len=8, n_states=2, seed=5)
    flipped = split.train_inputs.clone()
    flipped[:, 0] = 1 - flipped[:, 0]
    original = torch.cumsum(split.train_inputs, dim=1) % 2
    changed = torch.cumsum(flipped, dim=1) % 2
    assert bool((original != changed).all())


def test_tasks_split_disjointly_and_fingerprint_stably() -> None:
    first = make_copy_task(n_sequences=100, seed=7)
    second = make_copy_task(n_sequences=100, seed=7)
    assert first.fingerprint == second.fingerprint
    assert sum(first.sizes()) == 100
    assert first.fingerprint != make_copy_task(n_sequences=100, seed=8).fingerprint


def test_masked_loss_and_accuracy_ignore_unscored_positions() -> None:
    logits = torch.zeros(2, 3, 4)
    targets = torch.full((2, 3), IGNORE_INDEX)
    targets[:, 2] = 1
    assert masked_cross_entropy(logits, targets) == pytest.approx(math.log(4.0))

    logits[:, 2, 1] = 10.0  # correct only at the scored position
    assert masked_accuracy(logits, targets) == pytest.approx(1.0)


def test_masked_loss_validates_shapes() -> None:
    with pytest.raises(ValueError, match="expected logits"):
        masked_cross_entropy(torch.zeros(2, 3), torch.zeros(2, 3, dtype=torch.long))


def test_task_arguments_are_validated() -> None:
    with pytest.raises(ValueError, match="positive"):
        make_copy_task(n_sequences=0)
    with pytest.raises(ValueError, match="n_keys"):
        make_selective_recall_task(n_pairs=8, n_keys=4)
    with pytest.raises(ValueError, match="n_states"):
        make_state_tracking_task(n_states=1)


# --------------------------------------------------------------------------------------
# Cells: hand-computable recurrence and stability
# --------------------------------------------------------------------------------------


def test_slstm_matches_a_manual_two_step_calculation() -> None:
    torch.manual_seed(0)
    cell = SLSTMCell(2, 3, input_gate="exponential", forget_gate="sigmoid")
    inputs = torch.randn(1, 2, 2)  # (B, T, D)

    state = cell.initial_state(1)
    outputs = []
    for step in range(2):
        output, state = cell(inputs[:, step], state)
        outputs.append(output)

    # Reference implementation written straight from the equations, without the
    # stabilizer. It is numerically safe for two steps and must agree exactly.
    cell_state = torch.zeros(1, 3)
    normalizer = torch.ones(1, 3)
    hidden = torch.zeros(1, 3)
    for step in range(2):
        pre = cell.input_projection(inputs[:, step]) + cell.recurrent_projection(hidden)
        candidate, input_pre, forget_pre, output_pre = pre.chunk(4, dim=-1)
        gate_i = torch.exp(input_pre)
        gate_f = torch.sigmoid(forget_pre)
        cell_state = gate_f * cell_state + gate_i * torch.tanh(candidate)
        normalizer = gate_f * normalizer + gate_i
        hidden = torch.sigmoid(output_pre) * (cell_state / normalizer)
        assert torch.allclose(outputs[step], hidden, atol=1e-5)


def test_mlstm_matches_a_manual_two_step_calculation() -> None:
    torch.manual_seed(0)
    cell = MLSTMCell(4, heads=2, input_gate="exponential", forget_gate="sigmoid")
    inputs = torch.randn(1, 2, 4)

    state = cell.initial_state(1)
    outputs = []
    for step in range(2):
        output, state = cell(inputs[:, step], state)
        outputs.append(output)

    memory = torch.zeros(1, 2, 2, 2)
    normalizer = torch.zeros(1, 2, 2)
    for step in range(2):
        token = inputs[:, step]
        query = cell.query(token).view(1, 2, 2)
        key = cell.key(token).view(1, 2, 2) / math.sqrt(2)
        value = cell.value(token).view(1, 2, 2)
        input_pre, forget_pre = cell.gates(token).chunk(2, dim=-1)
        # Unstabilized reference, straight from the published equations: exponential
        # input gate, sigmoid forget gate, and the covering floor at a bare 1.
        gate_i = torch.exp(input_pre).unsqueeze(-1)
        gate_f = torch.sigmoid(forget_pre).unsqueeze(-1)

        memory = gate_f.unsqueeze(-1) * memory + gate_i.unsqueeze(-1) * (
            value.unsqueeze(-1) * key.unsqueeze(-2)
        )
        normalizer = gate_f * normalizer + gate_i * key
        retrieved = torch.einsum("bhij,bhj->bhi", memory, query)
        denominator = torch.einsum("bhi,bhi->bh", normalizer, query).abs().clamp_min(1.0)
        expected = torch.sigmoid(cell.output_gate(token)) * (
            retrieved / denominator.unsqueeze(-1)
        ).reshape(1, 4)
        assert torch.allclose(outputs[step], expected, atol=1e-5)


def test_stabilizer_keeps_long_sequences_finite() -> None:
    # Without the running maximum, exp() of a growing input-gate pre-activation
    # overflows within a few dozen steps. Large inputs make that failure certain.
    torch.manual_seed(0)
    cell = SLSTMCell(4, 8, input_gate="exponential", forget_gate="sigmoid")
    inputs = torch.randn(2, 400, 4) * 5.0

    state = cell.initial_state(2)
    for step in range(400):
        output, state = cell(inputs[:, step], state)
        assert torch.isfinite(output).all()
    assert torch.isfinite(state.cell).all()
    assert torch.isfinite(state.normalizer).all()
    assert torch.isfinite(state.stabilizer).all()


def test_mlstm_state_stays_finite_over_long_sequences() -> None:
    torch.manual_seed(0)
    cell = MLSTMCell(8, heads=2)
    inputs = torch.randn(2, 400, 8) * 5.0
    state = cell.initial_state(2)
    for step in range(400):
        output, state = cell(inputs[:, step], state)
        assert torch.isfinite(output).all()
    assert torch.isfinite(state.memory).all()


def test_initial_state_is_a_true_reset() -> None:
    # Running a sequence, then resetting, must reproduce the very first output exactly.
    torch.manual_seed(0)
    cell = SLSTMCell(3, 4)
    inputs = torch.randn(2, 5, 3)

    state = cell.initial_state(2)
    first_output, _ = cell(inputs[:, 0], state)

    state = cell.initial_state(2)
    for step in range(5):
        _, state = cell(inputs[:, step], state)

    state = cell.initial_state(2)
    reset_output, _ = cell(inputs[:, 0], state)
    assert torch.equal(first_output, reset_output)


def test_gating_ablation_changes_only_the_input_gate() -> None:
    torch.manual_seed(0)
    exponential = SLSTMCell(3, 4, input_gate="exponential")
    torch.manual_seed(0)
    sigmoid = SLSTMCell(3, 4, input_gate="sigmoid")

    assert torch.equal(exponential.input_projection.weight, sigmoid.input_projection.weight)
    inputs = torch.randn(2, 3)
    assert not torch.allclose(
        exponential(inputs, exponential.initial_state(2))[0],
        sigmoid(inputs, sigmoid.initial_state(2))[0],
    )


def test_sigmoid_gate_is_bounded_while_exponential_gate_is_not() -> None:
    # This is the mechanism in one line: the sigmoid gate cannot exceed one, so a late
    # token can never outweigh everything accumulated before it.
    from modern_nn_lab.tracks.xlstm.cells import _log_gate

    large = torch.tensor([5.0])
    assert float(torch.exp(_log_gate(large, "sigmoid"))) <= 1.0
    assert float(torch.exp(_log_gate(large, "exponential"))) > 100.0


def test_cells_validate_input_shapes() -> None:
    cell = SLSTMCell(3, 4)
    with pytest.raises(ValueError, match="expected shape"):
        cell(torch.zeros(2, 5), cell.initial_state(2))

    matrix_cell = MLSTMCell(4, heads=2)
    with pytest.raises(ValueError, match="expected shape"):
        matrix_cell(torch.zeros(2, 3), matrix_cell.initial_state(2))


def test_cells_validate_construction() -> None:
    with pytest.raises(ValueError, match="positive"):
        SLSTMCell(0, 4)
    with pytest.raises(ValueError, match="divisible"):
        MLSTMCell(5, heads=2)
    with pytest.raises(ValueError, match="unknown gate kind"):
        SLSTMCell(2, 2, input_gate="softmax")(  # type: ignore[arg-type]
            torch.zeros(1, 2), SLSTMCell(2, 2).initial_state(1)
        )


# --------------------------------------------------------------------------------------
# Models: causality, shapes, determinism
# --------------------------------------------------------------------------------------


def build_models(vocab_size: int = 6) -> dict[str, torch.nn.Module]:
    torch.manual_seed(0)
    return {
        "xlstm_mlstm": XLSTM(vocab_size, d_model=16, n_blocks=1, heads=2),
        "xlstm_slstm": XLSTM(vocab_size, d_model=16, n_blocks=1, block_kinds=("slstm",)),
        "lstm": RecurrentBaseline(vocab_size, d_model=16, n_layers=1, kind="lstm"),
        "gru": RecurrentBaseline(vocab_size, d_model=16, n_layers=1, kind="gru"),
        "transformer": CausalTransformer(vocab_size, d_model=16, n_layers=1, n_heads=2),
    }


@pytest.mark.parametrize("name", list(build_models()))
def test_models_are_causal(name: str) -> None:
    """Changing a token must never alter any output at an earlier position."""

    torch.manual_seed(0)
    model = build_models()[name].eval()
    tokens = torch.randint(0, 6, (2, 9))

    with torch.no_grad():
        original = model(tokens)
        edited = tokens.clone()
        edited[:, 6] = (edited[:, 6] + 1) % 6
        modified = model(edited)

    assert torch.allclose(original[:, :6], modified[:, :6], atol=1e-6)
    assert not torch.allclose(original[:, 6:], modified[:, 6:], atol=1e-6)


@pytest.mark.parametrize("name", list(build_models()))
def test_models_produce_the_documented_shape(name: str) -> None:
    model = build_models()[name]
    assert model(torch.randint(0, 6, (3, 7))).shape == (3, 7, 6)


@pytest.mark.parametrize("name", list(build_models()))
def test_models_have_finite_gradients(name: str) -> None:
    model = build_models()[name]
    logits = model(torch.randint(0, 6, (2, 8)))
    logits.pow(2).mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_model_initialization_is_deterministic() -> None:
    torch.manual_seed(3)
    first = XLSTM(6, d_model=16, n_blocks=1)
    torch.manual_seed(3)
    second = XLSTM(6, d_model=16, n_blocks=1)
    tokens = torch.randint(0, 6, (2, 5))
    assert torch.equal(first(tokens), second(tokens))


def test_model_serialization_round_trip() -> None:
    torch.manual_seed(0)
    config = XLSTMConfig(d_model=16, n_blocks=1)
    model = XLSTM(6, d_model=config.d_model, n_blocks=config.n_blocks, heads=config.heads)
    tokens = torch.randint(0, 6, (2, 5))
    expected = model(tokens)

    restored = XLSTM(6, d_model=config.d_model, n_blocks=config.n_blocks, heads=config.heads)
    restored.load_state_dict(model.state_dict())
    assert torch.allclose(restored(tokens), expected, atol=1e-7)


def test_model_construction_is_validated() -> None:
    with pytest.raises(ValueError, match="positive"):
        XLSTM(0)
    with pytest.raises(ValueError, match="n_blocks"):
        XLSTM(6, n_blocks=0)
    with pytest.raises(ValueError, match="block_kinds"):
        XLSTM(6, n_blocks=2, block_kinds=("mlstm",))
    with pytest.raises(ValueError, match="unknown baseline"):
        RecurrentBaseline(6, kind="rnn")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="divisible"):
        CausalTransformer(6, d_model=10, n_heads=4)


def test_transformer_rejects_sequences_beyond_its_positional_table() -> None:
    model = CausalTransformer(6, d_model=16, n_heads=2, max_len=8)
    with pytest.raises(ValueError, match="exceeds max_len"):
        model(torch.zeros(1, 9, dtype=torch.long))


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="d_model"):
        XLSTMConfig(d_model=0)
    with pytest.raises(ValueError, match="n_blocks"):
        XLSTMConfig(n_blocks=0)
    with pytest.raises(ValueError, match="divisible"):
        XLSTMConfig(d_model=10, heads=4)
    with pytest.raises(ValueError, match="block_kinds"):
        XLSTMConfig(n_blocks=2, block_kinds=("mlstm",))


def test_width_matching_lands_near_the_target_budget() -> None:
    target = count_parameters(XLSTM(10, d_model=32, n_blocks=1))
    width = match_width_to_budget(
        target, lambda w: CausalTransformer(10, d_model=w, n_layers=1, n_heads=2)
    )
    matched = count_parameters(CausalTransformer(10, d_model=width, n_layers=1, n_heads=2))
    assert abs(matched - target) <= 0.2 * target


def test_width_matching_reports_when_nothing_can_be_built() -> None:
    def always_invalid(width: int) -> torch.nn.Module:
        raise ValueError("no")

    with pytest.raises(ValueError, match="no candidate width"):
        match_width_to_budget(100, always_invalid)
