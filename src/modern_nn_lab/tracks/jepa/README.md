# JEPA / Predictive Representation Learning

**Claim level: research prototype.** A compact joint-embedding predictive architecture that
predicts representations of masked patches rather than reconstructing them, with the
baselines and collapse diagnostics needed to tell whether that produces anything useful.

## What exactly is predicted

Patches are split into a visible **context** set and a hidden **target** set. The context
encoder embeds the visible patches and pools them; the predictor must then produce each
hidden patch's **representation** — never its observed values:

```
ẑ_p = g_φ( mean of f_θ(context patches) )
z_p = stopgrad( f_ξ(x_p) )                    ξ = EMA of θ
L   = mean over hidden p of ‖ẑ_p − z_p‖²
```

The target comes from a *separate* encoder `f_ξ` under a stop-gradient. Nothing in the loss
mentions the observations, which is the whole point: the model is free to discard anything
about a patch that another patch could not have predicted.

## Why trivial collapse is or is not prevented

**The loss alone does not prevent it.** If `f(x) = k` for every input, the predictor outputs
`k` and the loss is exactly zero. That is not a hypothesis — `test_a_constant_encoder_
achieves_zero_loss` asserts it, and the `anti_collapse="none"` variant is trained and
reported so the collapse is shown rather than described.

**What prevents it here** is the stop-gradient paired with an EMA target encoder:

```
ξ ← τ·ξ + (1−τ)·θ
```

No gradient reaches `ξ`, so the loss cannot reduce itself by moving the target; the online
encoder chases a lagged copy of itself. This is an empirical stabilizer, **not a proof** —
collapse remains reachable in principle, so the report treats "collapse did not occur" as a
measurement rather than a guarantee.

`anti_collapse="variance"` provides a contrasting mechanism: an explicit hinge holding each
dimension's standard deviation above a floor. A direct constraint rather than a dynamics
argument, and comparing the two is one of the ablations.

## ⚠ Effective rank does not detect total collapse

Both collapse metrics are necessary, and the reason is not obvious.

**Variance** misses *dimensional* collapse: healthy per-dimension spread is compatible with
every sample lying on one line.

**Effective rank** misses *total* collapse. When the encoder becomes constant, what remains
is floating-point noise — and that noise is **isotropic**, so the covariance has roughly
equal eigenvalues and the rank comes out *high*. In this repository's own measurements a
fully collapsed representation reported a normalized effective rank of ≈0.75 while its
standard deviation was 0.001. Reading the rank alone would have called it healthy.

They are therefore always reported together, standard deviation first, and
`test_effective_rank_does_not_detect_total_collapse` pins the trap so the pairing is never
dropped.

## Measuring representation quality without a downstream task

The generative factors are known and split into two kinds:

- **content** — shared across all patches of a sample;
- **nuisance** — drawn independently per patch.

That asymmetry makes the task well posed: content is exactly the part of one patch another
patch can predict, and nuisance is exactly the part it cannot. It also gives the invariance
analysis a ground truth. "Invariance to nuisance" is the R² of a probe *trying* to recover
the nuisance factors, where a low score is the good outcome — evidence rather than an
impression.

Both probes are closed-form ridge regressions, so the numbers depend on the representation
alone: no probe learning rate, no probe seed, nothing to tune into a better-looking result.

| File | What it holds |
| --- | --- |
| [data.py](data.py) | Latent-factor generator and complementary mask sampling |
| [model.py](model.py) | `JEPA`, `Autoencoder`, `ContrastiveLearner`, `RawFeatures` |
| [metrics.py](metrics.py) | Collapse metrics and the closed-form linear probe |
| [config.py](config.py) | `JEPAConfig`, `JEPAExperimentConfig` |

## Baselines

`autoencoder` — reconstructs raw patches, and **cannot collapse**: a constant code
reconstructs nothing. Expected to retain the nuisance a JEPA should discard.

`contrastive` — InfoNCE over two patches of a sample. The other standard answer to collapse:
explicit repulsion.

`raw-features` — mean of the raw patches, no learning. The floor, and on a task this simple
a higher bar than it sounds.

## Running it

```bash
modern-nn run-track jepa          # writes results/jepa/
python scripts/report_jepa.py     # regenerates every table in reports/jepa.md
```

See [reports/jepa.md](../../../../reports/jepa.md) for results.
