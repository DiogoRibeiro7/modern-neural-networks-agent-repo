"""Invariants of the Prior-Fitted Network track.

The structural properties asserted here are what make in-context prediction meaningful: a
query must not see other queries, order must not matter, and prediction must not involve
any gradient step. A model that violated any of them could score well for reasons that
have nothing to do with the mechanism.
"""

from __future__ import annotations

import pytest
import torch

from modern_nn_lab.tracks.pfn import (
    ImbalancedPrior,
    LinearPrior,
    MLPPrior,
    PFNConfig,
    PriorFittedNetwork,
    XORPrior,
    build_prior,
    pretraining_advantage_note,
    tabpfn_available,
)

# --------------------------------------------------------------------------------------
# Priors
# --------------------------------------------------------------------------------------


def test_priors_produce_the_documented_shapes() -> None:
    generator = torch.Generator().manual_seed(0)
    batch = LinearPrior(n_features=4).sample(
        batch_size=3, n_context=7, n_query=5, generator=generator
    )
    assert batch.context_inputs.shape == (3, 7, 4)
    assert batch.context_labels.shape == (3, 7)
    assert batch.query_inputs.shape == (3, 5, 4)
    assert batch.query_labels.shape == (3, 5)
    assert batch.n_features == 4
    assert batch.n_context == 7


def test_each_task_in_a_batch_has_its_own_labelling_function() -> None:
    """Tasks must be independent, or the model could learn one shared rule."""

    generator = torch.Generator().manual_seed(0)
    batch = LinearPrior(n_features=3).sample(
        batch_size=16, n_context=40, n_query=4, generator=generator
    )
    # The same point would receive different labels under different tasks, so the
    # per-task class balance should vary rather than being identical everywhere.
    balances = batch.context_labels.float().mean(dim=1)
    assert float(balances.std()) > 0.01


def test_linear_prior_is_linearly_separable() -> None:
    """A logistic regression should recover a linear prior's rule almost perfectly."""

    from sklearn.linear_model import LogisticRegression

    generator = torch.Generator().manual_seed(1)
    batch = LinearPrior(n_features=3).sample(
        batch_size=1, n_context=200, n_query=100, generator=generator
    )
    model = LogisticRegression(max_iter=1000).fit(
        batch.context_inputs[0].numpy(), batch.context_labels[0].numpy()
    )
    score = model.score(batch.query_inputs[0].numpy(), batch.query_labels[0].numpy())
    assert score > 0.95


def test_xor_prior_defeats_a_linear_model() -> None:
    """The out-of-prior family must genuinely be out of reach for a linear rule."""

    from sklearn.linear_model import LogisticRegression

    generator = torch.Generator().manual_seed(2)
    scores = []
    for _ in range(5):
        batch = XORPrior(n_features=3).sample(
            batch_size=1, n_context=200, n_query=100, generator=generator
        )
        model = LogisticRegression(max_iter=1000).fit(
            batch.context_inputs[0].numpy(), batch.context_labels[0].numpy()
        )
        scores.append(model.score(batch.query_inputs[0].numpy(), batch.query_labels[0].numpy()))
    assert sum(scores) / len(scores) < 0.7


def test_label_noise_flips_roughly_the_requested_fraction() -> None:
    clean = LinearPrior(n_features=3, label_noise=0.0)
    noisy = LinearPrior(n_features=3, label_noise=0.3)

    # One task per draw, with the generators re-seeded together: the points and the rule
    # are then drawn identically, and the noise flips are the only source of disagreement.
    # (Sampling several tasks from one generator would not work — the noisy prior consumes
    # extra random numbers, so the two streams would diverge after the first task.)
    flipped = 0
    total = 0
    for seed in range(20):
        clean_batch = clean.sample(
            batch_size=1, n_context=50, n_query=1, generator=torch.Generator().manual_seed(seed)
        )
        noisy_batch = noisy.sample(
            batch_size=1, n_context=50, n_query=1, generator=torch.Generator().manual_seed(seed)
        )
        assert torch.equal(clean_batch.context_inputs, noisy_batch.context_inputs)
        flipped += int((clean_batch.context_labels != noisy_batch.context_labels).sum())
        total += clean_batch.context_labels.numel()

    assert 0.25 < flipped / total < 0.35


def test_imbalanced_prior_hits_its_target_rate() -> None:
    """The class-imbalance study needs the imbalance to be the only thing that changed."""

    generator = torch.Generator().manual_seed(4)
    prior = ImbalancedPrior(n_features=3, positive_rate=0.2)
    batch = prior.sample(batch_size=20, n_context=100, n_query=1, generator=generator)
    rate = float(batch.context_labels.float().mean())
    assert 0.15 < rate < 0.25

    # Thresholding a linear score keeps the boundary linear, so a linear model still
    # recovers it: the study varies balance, not separability.
    from sklearn.linear_model import LogisticRegression

    single = ImbalancedPrior(n_features=3, positive_rate=0.2).sample(
        batch_size=1, n_context=300, n_query=100, generator=torch.Generator().manual_seed(7)
    )
    model = LogisticRegression(max_iter=1000).fit(
        single.context_inputs[0].numpy(), single.context_labels[0].numpy()
    )
    assert model.score(single.query_inputs[0].numpy(), single.query_labels[0].numpy()) > 0.9


def test_blank_features_masks_the_requested_fraction_and_copies() -> None:
    """Missingness must be applied identically to every model, and must not mutate."""

    from modern_nn_lab.experiments.tracks.pfn import blank_features

    inputs = torch.randn(4, 50, 3)
    original = inputs.clone()
    masked = blank_features(inputs, 0.3, torch.Generator().manual_seed(0))

    assert torch.equal(inputs, original), "blank_features must not modify its argument"
    blanked = float((masked == 0.0).float().mean())
    assert 0.2 < blanked < 0.4
    assert blank_features(inputs, 0.0, torch.Generator()) is inputs


def test_prior_fit_cache_is_keyed_by_feature_width() -> None:
    """A four-feature model cannot answer a sixteen-feature task."""

    from modern_nn_lab.experiments.tracks import pfn as suite
    from modern_nn_lab.tracks.pfn import PFNExperimentConfig

    settings = PFNExperimentConfig(prior_fitting_steps=2, tasks_per_step=2, n_features=3)
    narrow, _ = suite.fit_prior_cached(build_prior("linear", n_features=3), settings, 0)
    wide, _ = suite.fit_prior_cached(build_prior("linear", n_features=6), settings, 0, n_features=6)

    assert narrow.config.n_features == 3
    assert wide.config.n_features == 6
    assert narrow is not wide


def test_prior_registry_and_validation() -> None:
    assert build_prior("linear", n_features=3).name == "linear"
    assert build_prior("mlp", n_features=3).name == "mlp"
    assert build_prior("xor", n_features=3).name == "xor"
    assert build_prior("imbalanced", n_features=3, positive_rate=0.3).name == "imbalanced0.3"
    with pytest.raises(KeyError, match="unknown prior"):
        build_prior("nope", n_features=3)
    with pytest.raises(TypeError):
        build_prior("linear", n_features=3, positive_rate=0.3)
    with pytest.raises(ValueError, match="positive_rate"):
        ImbalancedPrior(n_features=3, positive_rate=1.0)
    with pytest.raises(ValueError, match="n_features"):
        LinearPrior(n_features=0)
    with pytest.raises(ValueError, match="label_noise"):
        LinearPrior(n_features=3, label_noise=1.5)
    with pytest.raises(ValueError, match="hidden"):
        MLPPrior(n_features=3, hidden=0)
    with pytest.raises(ValueError, match="at least two features"):
        XORPrior(n_features=1).label(torch.randn(4, 1), torch.Generator())
    with pytest.raises(ValueError, match="must be positive"):
        LinearPrior(n_features=3).sample(
            batch_size=0, n_context=2, n_query=2, generator=torch.Generator()
        )


# --------------------------------------------------------------------------------------
# The attention structure that makes in-context prediction well posed
# --------------------------------------------------------------------------------------


def build_model(**overrides: object) -> PriorFittedNetwork:
    torch.manual_seed(0)
    settings: dict[str, object] = {"n_features": 3, "d_model": 32, "n_layers": 2, "n_heads": 4}
    settings.update(overrides)
    return PriorFittedNetwork(PFNConfig(**settings))  # type: ignore[arg-type]


def test_attention_mask_lets_queries_read_context_and_themselves_only() -> None:
    model = build_model()
    mask = model.attention_mask(3, 2, torch.device("cpu"))
    allowed = mask == 0.0

    # Context rows: context only.
    assert allowed[0].tolist() == [True, True, True, False, False]
    # Query rows: all context, plus itself, never the other query.
    assert allowed[3].tolist() == [True, True, True, True, False]
    assert allowed[4].tolist() == [True, True, True, False, True]


def test_one_query_cannot_influence_another() -> None:
    """The defining property: each query is answered independently."""

    model = build_model().eval()
    generator = torch.Generator().manual_seed(0)
    batch = LinearPrior(n_features=3).sample(
        batch_size=2, n_context=6, n_query=4, generator=generator
    )

    with torch.no_grad():
        original = model(batch.context_inputs, batch.context_labels, batch.query_inputs)
        edited = batch.query_inputs.clone()
        edited[:, 2] += 5.0
        modified = model(batch.context_inputs, batch.context_labels, edited)

    untouched = [0, 1, 3]
    assert torch.allclose(original[:, untouched], modified[:, untouched], atol=1e-6)
    assert not torch.allclose(original[:, 2], modified[:, 2], atol=1e-6)


def test_predictions_are_invariant_to_context_order() -> None:
    """A dataset is a set. Row order must not matter, and there is no positional encoding."""

    model = build_model().eval()
    generator = torch.Generator().manual_seed(0)
    batch = LinearPrior(n_features=3).sample(
        batch_size=1, n_context=8, n_query=3, generator=generator
    )

    permutation = torch.randperm(8)
    with torch.no_grad():
        original = model(batch.context_inputs, batch.context_labels, batch.query_inputs)
        shuffled = model(
            batch.context_inputs[:, permutation],
            batch.context_labels[:, permutation],
            batch.query_inputs,
        )
    assert torch.allclose(original, shuffled, atol=1e-5)


def test_context_labels_actually_matter() -> None:
    """If flipping the context labels changed nothing, the model would be ignoring it."""

    model = build_model().eval()
    generator = torch.Generator().manual_seed(0)
    batch = LinearPrior(n_features=3).sample(
        batch_size=1, n_context=8, n_query=3, generator=generator
    )
    with torch.no_grad():
        original = model(batch.context_inputs, batch.context_labels, batch.query_inputs)
        flipped = model(batch.context_inputs, 1 - batch.context_labels, batch.query_inputs)
    assert not torch.allclose(original, flipped, atol=1e-5)


def test_prediction_does_not_change_parameters() -> None:
    """Prediction on a new dataset is a forward pass, not an optimization."""

    model = build_model()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    generator = torch.Generator().manual_seed(0)
    batch = LinearPrior(n_features=3).sample(
        batch_size=2, n_context=6, n_query=3, generator=generator
    )
    probabilities = model.predict_proba(
        batch.context_inputs, batch.context_labels, batch.query_inputs
    )

    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name]), f"{name} moved during prediction"
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2, 3), atol=1e-5)


def test_predict_proba_restores_training_mode() -> None:
    model = build_model()
    model.train()
    generator = torch.Generator().manual_seed(0)
    batch = LinearPrior(n_features=3).sample(
        batch_size=1, n_context=4, n_query=2, generator=generator
    )
    model.predict_proba(batch.context_inputs, batch.context_labels, batch.query_inputs)
    assert model.training


def test_model_validates_shapes_and_configuration() -> None:
    model = build_model()
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(2, 3), torch.zeros(2, 3, dtype=torch.long), torch.randn(2, 2, 3))
    with pytest.raises(ValueError, match="features"):
        model(torch.randn(1, 2, 9), torch.zeros(1, 2, dtype=torch.long), torch.randn(1, 2, 9))
    with pytest.raises(ValueError, match="agree on batch"):
        model(torch.randn(1, 4, 3), torch.zeros(1, 3, dtype=torch.long), torch.randn(1, 2, 3))

    with pytest.raises(ValueError, match="n_features"):
        PFNConfig(n_features=0)
    with pytest.raises(ValueError, match="divisible"):
        PFNConfig(d_model=10, n_heads=4)
    with pytest.raises(ValueError, match="feedforward"):
        PFNConfig(feedforward=0)


def test_a_prior_fitted_model_beats_chance_in_context() -> None:
    """A short prior-fitting run must actually learn to use the context."""

    from modern_nn_lab.experiments.tracks.pfn import evaluate_pfn, fit_prior
    from modern_nn_lab.tracks.pfn import PFNExperimentConfig

    settings = PFNExperimentConfig(
        prior_fitting_steps=250, tasks_per_step=16, n_features=3, eval_tasks=40
    )
    prior = build_prior("linear", n_features=3)
    model, losses = fit_prior(prior, settings, seed=0)

    assert losses[-1] < losses[0]
    score, _ = evaluate_pfn(
        model,
        prior,
        n_context=settings.train_context,
        n_query=settings.n_query,
        n_tasks=settings.eval_tasks,
        generator=torch.Generator().manual_seed(99),
    )
    assert score > 0.6


# --------------------------------------------------------------------------------------
# Deliverable B: the official checkpoint, which is a different kind of object
# --------------------------------------------------------------------------------------


def test_reference_adapter_reports_availability_without_importing_eagerly() -> None:
    assert isinstance(tabpfn_available(), bool)


def test_pretraining_advantage_note_states_the_asymmetry() -> None:
    """Any table containing TabPFN must carry this disclosure."""

    note = pretraining_advantage_note()
    assert "pre-trained" in note
    assert "accounted for" in note
    assert "Neither the pre-training data nor its compute is measured" in note


def test_reference_adapter_fails_with_instructions_when_unavailable() -> None:
    if tabpfn_available():  # pragma: no cover - only when the optional extra is installed
        pytest.skip("the optional tabpfn extra is installed")

    from modern_nn_lab.tracks.pfn.reference import build_tabpfn_classifier

    with pytest.raises(ImportError, match="license"):
        build_tabpfn_classifier()
