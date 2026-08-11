"""Invariants of the JEPA track.

The acceptance criterion is that the report must explain what is predicted in
representation space and why trivial collapse is or is not prevented. Explanation is prose,
but the claims underneath it are testable, and they are tested here:

- the objective really does have a trivial solution — a constant encoder scores zero loss;
- the mechanism that prevents it really is the stop-gradient, not the loss;
- the collapse metrics really do detect collapse, including the case where one of them
  does not;
- the dataset really does separate content from nuisance, so the probes measure what they
  are supposed to.
"""

from __future__ import annotations

import pytest
import torch

from modern_nn_lab.tracks.jepa import (
    JEPA,
    Autoencoder,
    ContrastiveLearner,
    JEPAConfig,
    RawFeatures,
    collapse_report,
    effective_rank,
    generate,
    jepa_loss,
    linear_probe,
    normalized_effective_rank,
    representation_variance,
    sample_masks,
)

# --------------------------------------------------------------------------------------
# The trivial solution really is trivial
# --------------------------------------------------------------------------------------


def test_a_constant_encoder_achieves_zero_loss() -> None:
    """The objection the whole track exists to answer, demonstrated rather than asserted.

    If the encoder outputs the same vector for every input, the predictor need only output
    that vector and the prediction error is exactly zero. A low loss is therefore not
    evidence of a good representation.
    """

    constant = torch.full((4, 6, 8), 0.7)
    target_mask = torch.ones((4, 6), dtype=torch.bool)
    assert float(jepa_loss(constant, constant.clone(), target_mask)) == pytest.approx(0.0)


def test_the_variance_hinge_punishes_the_constant_solution() -> None:
    """The direct mechanism: a collapsed target is no longer free."""

    constant = torch.full((4, 6, 8), 0.7)
    target_mask = torch.ones((4, 6), dtype=torch.bool)

    without = float(jepa_loss(constant, constant.clone(), target_mask))
    with_hinge = float(
        jepa_loss(constant, constant.clone(), target_mask, variance_weight=1.0, variance_floor=1.0)
    )
    assert without == pytest.approx(0.0)
    assert with_hinge > 0.9


def test_the_variance_hinge_is_satisfied_by_a_spread_representation() -> None:
    torch.manual_seed(0)
    spread = torch.randn(64, 6, 8) * 2.0
    target_mask = torch.ones((64, 6), dtype=torch.bool)

    penalized = float(
        jepa_loss(spread, spread.clone(), target_mask, variance_weight=1.0, variance_floor=1.0)
    )
    assert penalized == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------------------
# The anti-collapse mechanism is the stop-gradient, not the loss
# --------------------------------------------------------------------------------------


def test_the_ema_target_receives_no_gradient() -> None:
    """If gradient reached the target, the loss could be reduced by moving it."""

    model = JEPA(JEPAConfig(d_patch=10, anti_collapse="ema"))
    assert model.target_encoder is not None
    assert all(not p.requires_grad for p in model.target_encoder.parameters())

    patches = torch.randn(8, 6, 10)
    context_mask = torch.ones((8, 6), dtype=torch.bool)
    predicted, targets = model(patches, context_mask)
    assert not targets.requires_grad
    assert predicted.requires_grad


def test_without_a_target_encoder_the_gradient_flows_into_the_targets() -> None:
    """The configuration that collapses, shown to be structurally different."""

    model = JEPA(JEPAConfig(d_patch=10, anti_collapse="none"))
    assert model.target_encoder is None

    patches = torch.randn(8, 6, 10)
    context_mask = torch.ones((8, 6), dtype=torch.bool)
    _, targets = model(patches, context_mask)
    assert targets.requires_grad


def test_the_ema_update_moves_the_target_towards_the_online_encoder() -> None:
    model = JEPA(JEPAConfig(d_patch=10, anti_collapse="ema", ema_decay=0.9))
    assert model.target_encoder is not None

    with torch.no_grad():
        for parameter in model.encoder.parameters():
            parameter.add_(1.0)

    before = [p.clone() for p in model.target_encoder.parameters()]
    model.update_target()
    after = list(model.target_encoder.parameters())

    for old, new, online in zip(before, after, model.encoder.parameters(), strict=True):
        expected = 0.9 * old + 0.1 * online
        assert torch.allclose(new, expected, atol=1e-6)


def test_the_ema_update_is_a_no_op_without_a_target_encoder() -> None:
    model = JEPA(JEPAConfig(d_patch=10, anti_collapse="none"))
    model.update_target()  # must not raise


def test_training_a_jepa_without_anti_collapse_actually_collapses() -> None:
    """The end-to-end demonstration, which is the point of keeping the ablation."""

    from modern_nn_lab.experiments.tracks.jepa import train_model
    from modern_nn_lab.tracks.jepa import JEPAExperimentConfig

    settings = JEPAExperimentConfig(n_samples=400, steps=300, d_patch=10, n_patches=6)
    train, _ = generate(
        n_samples=settings.n_samples,
        n_patches=settings.n_patches,
        d_patch=settings.d_patch,
        seed=settings.data_seed,
    ).split(0.7)

    collapsed = train_model(
        "jepa",
        JEPAConfig(d_patch=10, anti_collapse="none"),
        train,
        settings,
        seed=0,
        n_targets=3,
    )
    healthy = train_model(
        "jepa",
        JEPAConfig(d_patch=10, anti_collapse="ema"),
        train,
        settings,
        seed=0,
        n_targets=3,
    )

    with torch.no_grad():
        collapsed_std = representation_variance(collapsed.represent(train.patches))
        healthy_std = representation_variance(healthy.represent(train.patches))

    assert collapsed_std < 0.05, f"expected collapse, got std {collapsed_std:.4f}"
    assert healthy_std > 10 * collapsed_std


# --------------------------------------------------------------------------------------
# The collapse metrics, including the one that fails
# --------------------------------------------------------------------------------------


def test_variance_detects_total_collapse() -> None:
    constant = torch.full((100, 8), 0.3)
    assert representation_variance(constant) == pytest.approx(0.0, abs=1e-7)


def test_effective_rank_detects_dimensional_collapse() -> None:
    """Every sample on one line: variance looks fine, rank does not."""

    torch.manual_seed(0)
    direction = torch.randn(8)
    on_a_line = torch.randn(200, 1) * direction

    assert representation_variance(on_a_line) > 0.1
    assert effective_rank(on_a_line) == pytest.approx(1.0, abs=0.05)
    assert normalized_effective_rank(on_a_line) < 0.2


def test_effective_rank_is_maximal_for_isotropic_features() -> None:
    torch.manual_seed(0)
    isotropic = torch.randn(4000, 8)
    assert effective_rank(isotropic) == pytest.approx(8.0, rel=0.05)
    assert normalized_effective_rank(isotropic) == pytest.approx(1.0, rel=0.05)


def test_effective_rank_does_not_detect_total_collapse() -> None:
    """The trap this repository walked into, pinned so the pairing is never dropped.

    A constant encoder leaves only floating-point noise, which is isotropic — so the
    effective rank reads *high* on a fully collapsed representation. The rank must always be
    read beside the standard deviation, never instead of it.
    """

    torch.manual_seed(0)
    collapsed = torch.full((500, 8), 0.3) + 1e-6 * torch.randn(500, 8)

    assert representation_variance(collapsed) < 1e-4
    assert normalized_effective_rank(collapsed) > 0.5, (
        "if this ever fails, the rank has started detecting total collapse and the "
        "warning in metrics.py can be revisited"
    )


def test_collapse_report_returns_both_measures() -> None:
    torch.manual_seed(0)
    report = collapse_report(torch.randn(100, 8))
    assert set(report) == {
        "representation_std",
        "effective_rank",
        "normalized_effective_rank",
    }


def test_effective_rank_of_an_all_zero_representation_is_one() -> None:
    assert effective_rank(torch.zeros(50, 8)) == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# The linear probe
# --------------------------------------------------------------------------------------


def test_the_probe_recovers_a_linear_function_exactly() -> None:
    torch.manual_seed(0)
    features = torch.randn(500, 6)
    weights = torch.randn(6, 3)
    targets = features @ weights + 0.5

    score = linear_probe(features[:400], targets[:400], features[400:], targets[400:])
    assert score > 0.99


def test_the_probe_scores_zero_on_unrelated_targets() -> None:
    torch.manual_seed(0)
    features = torch.randn(500, 6)
    targets = torch.randn(500, 3)
    score = linear_probe(features[:400], targets[:400], features[400:], targets[400:])
    assert score < 0.2


def test_the_probe_scores_zero_on_a_collapsed_representation() -> None:
    """A collapsed representation must not be rescued by the ridge term."""

    torch.manual_seed(0)
    collapsed = torch.full((500, 6), 0.2)
    targets = torch.randn(500, 3)
    assert linear_probe(collapsed[:400], targets[:400], collapsed[400:], targets[400:]) < 0.1


def test_the_probe_is_deterministic() -> None:
    """Closed form, so no probe seed can be tuned into a better-looking number."""

    torch.manual_seed(0)
    features = torch.randn(200, 5)
    targets = torch.randn(200, 2)
    first = linear_probe(features[:150], targets[:150], features[150:], targets[150:])
    second = linear_probe(features[:150], targets[:150], features[150:], targets[150:])
    assert first == second


# --------------------------------------------------------------------------------------
# The dataset separates content from nuisance
# --------------------------------------------------------------------------------------


def test_generated_data_has_the_documented_shapes() -> None:
    dataset = generate(n_samples=50, n_patches=6, d_patch=10, d_content=4, d_nuisance=3, seed=0)
    assert dataset.patches.shape == (50, 6, 10)
    assert dataset.content.shape == (50, 4)
    assert dataset.nuisance.shape == (50, 6, 3)
    assert dataset.n_samples == 50
    assert dataset.n_patches == 6
    assert dataset.d_patch == 10


def test_content_is_shared_across_patches_and_nuisance_is_not() -> None:
    """The design property that makes the whole task well posed."""

    dataset = generate(n_samples=400, n_patches=6, d_patch=10, noise=0.0, seed=1)

    # One patch predicts another's content well, because content is shared...
    first, second = dataset.patches[:, 0], dataset.patches[:, 1]
    content_score = linear_probe(
        first[:300], dataset.content[:300], first[300:], dataset.content[300:]
    )
    assert content_score > 0.7

    # ...but not another patch's nuisance, because nuisance is drawn independently.
    cross_nuisance = linear_probe(
        first[:300], dataset.nuisance[:300, 1], first[300:], dataset.nuisance[300:, 1]
    )
    assert cross_nuisance < 0.2

    # A patch does carry its *own* nuisance, which is why discarding it is a real choice.
    own_nuisance = linear_probe(
        second[:300], dataset.nuisance[:300, 1], second[300:], dataset.nuisance[300:, 1]
    )
    assert own_nuisance > 0.5


def test_masks_are_complementary_and_the_right_size() -> None:
    generator = torch.Generator().manual_seed(0)
    context, target = sample_masks(64, 8, n_targets=3, generator=generator)

    assert torch.equal(context, ~target)
    assert torch.equal(target.sum(dim=1), torch.full((64,), 3))
    assert torch.equal(context.sum(dim=1), torch.full((64,), 5))
    assert not bool((context & target).any()), "a patch must never be both"


def test_masking_varies_across_samples() -> None:
    generator = torch.Generator().manual_seed(0)
    _, target = sample_masks(200, 8, n_targets=3, generator=generator)
    assert len({tuple(row) for row in target.tolist()}) > 10


def test_mask_sampling_validates_its_target_count() -> None:
    generator = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="n_targets"):
        sample_masks(4, 6, n_targets=6, generator=generator)
    with pytest.raises(ValueError, match="n_targets"):
        sample_masks(4, 6, n_targets=0, generator=generator)


def test_generation_validates_its_sizes() -> None:
    with pytest.raises(ValueError, match="positive"):
        generate(n_samples=0)
    with pytest.raises(ValueError, match="noise"):
        generate(n_samples=4, noise=-1.0)


# --------------------------------------------------------------------------------------
# The models and the baselines
# --------------------------------------------------------------------------------------


def test_every_model_produces_a_representation_per_sample() -> None:
    config = JEPAConfig(d_patch=10)
    patches = torch.randn(7, 6, 10)

    assert JEPA(config).represent(patches).shape == (7, config.d_representation)
    assert Autoencoder(config).represent(patches).shape == (7, config.d_representation)
    assert ContrastiveLearner(config).represent(patches).shape == (7, config.d_representation)
    assert RawFeatures(config).represent(patches).shape == (7, 10)


def test_an_identity_predictor_is_the_zero_depth_configuration() -> None:
    model = JEPA(JEPAConfig(d_patch=10, n_predictor_layers=0))
    assert isinstance(model.predictor, torch.nn.Identity)

    hidden = torch.randn(4, 16)
    assert torch.equal(model.predictor(hidden), hidden)


def test_the_jepa_requires_at_least_one_context_patch() -> None:
    model = JEPA(JEPAConfig(d_patch=10))
    with pytest.raises(ValueError, match="context patch"):
        model(torch.randn(3, 6, 10), torch.zeros((3, 6), dtype=torch.bool))


def test_the_autoencoder_reconstructs_its_input_shape() -> None:
    model = Autoencoder(JEPAConfig(d_patch=10))
    patches = torch.randn(5, 6, 10)
    assert model(patches).shape == patches.shape


def test_the_contrastive_loss_is_lower_for_aligned_views() -> None:
    torch.manual_seed(0)
    model = ContrastiveLearner(JEPAConfig(d_patch=10)).eval()

    # Two patches of the same sample share content, so their embeddings should align more
    # than two patches drawn from unrelated samples.
    dataset = generate(n_samples=128, n_patches=6, d_patch=10, noise=0.0, seed=2)
    aligned = dataset.patches[:, :2]
    shuffled = torch.stack([dataset.patches[:, 0], dataset.patches.roll(1, 0)[:, 1]], dim=1)

    with torch.no_grad():
        assert float(model(aligned)) < float(model(shuffled))


def test_the_contrastive_learner_needs_two_patches() -> None:
    model = ContrastiveLearner(JEPAConfig(d_patch=10))
    with pytest.raises(ValueError, match="two patches"):
        model(torch.randn(4, 1, 10))


def test_raw_features_has_no_parameters() -> None:
    assert list(RawFeatures(JEPAConfig(d_patch=10)).parameters()) == []


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="widths"):
        JEPAConfig(d_representation=0)
    with pytest.raises(ValueError, match="n_encoder_layers"):
        JEPAConfig(n_encoder_layers=0)
    with pytest.raises(ValueError, match="n_predictor_layers"):
        JEPAConfig(n_predictor_layers=-1)
    with pytest.raises(ValueError, match="ema_decay"):
        JEPAConfig(ema_decay=1.0)
    with pytest.raises(ValueError, match="temperature"):
        JEPAConfig(temperature=0.0)


def test_the_variance_weight_is_only_applied_by_the_variance_mechanism() -> None:
    """Otherwise the EMA variant would be quietly carrying two mechanisms."""

    assert JEPAConfig(anti_collapse="ema", variance_weight=5.0).effective_variance_weight == 0.0
    assert (
        JEPAConfig(anti_collapse="variance", variance_weight=5.0).effective_variance_weight == 5.0
    )
    assert JEPAConfig(anti_collapse="none", variance_weight=5.0).effective_variance_weight == 0.0


def test_build_model_rejects_an_unknown_name() -> None:
    from modern_nn_lab.experiments.tracks.jepa import build_model

    with pytest.raises(ValueError, match="unknown model"):
        build_model("nope", JEPAConfig(d_patch=10))
