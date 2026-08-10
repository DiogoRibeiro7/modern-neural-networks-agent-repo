# Relational foundation-model prototype

**Claim level: research prototype.** A compact model that learns over linked tables without
flattening them first, plus the baselines needed to tell whether that buys anything. This is
**not** a reproduction of any relational foundation model, and no pre-trained relational
checkpoint is invoked anywhere in the track.

## The representation

A row stays a row. Each prediction point `(entity, t*)` becomes a small set of typed,
linked rows rather than a feature vector:

| Slot | Contents |
| --- | --- |
| `0` | the target entity's own columns |
| `1 … K` | its most recent visible events (`orders`) |
| `K+1 … 2K` | the row each of those events points at (`products`) |
| `2K+1 … 2K+S` | rows from the distractor table (`signals`) |

Every row carries its table's type flag, a presence mask, elapsed time since the event, its
numeric and categorical columns, and the slot of the row it references. That last channel is
the foreign key, and it is what makes a two-hop path distinguishable from two unrelated rows
sitting in the same bag.

| File | What it holds |
| --- | --- |
| [schema.py](schema.py) | `Column`, `Table`, `ForeignKey`, `Database` — typed columns and links |
| [generator.py](generator.py) | The five diagnostic regimes, and the leakage canary |
| [sampler.py](sampler.py) | **The single time-gating chokepoint**, and `Normalizer` |
| [model.py](model.py) | `RelationalEncoder` and the target-only floor |
| [features.py](features.py) | Leakage-safe aggregation — the flattening being argued against |
| [trace.py](trace.py) | Which relational paths can reach a prediction, and which did |

## Temporal leakage

The rule is: a row is visible to prediction `(e, t*)` if it is static, or if `t_row < t*`.
Note the strict inequality — a row timestamped exactly at the prediction instant is not
visible, and admitting simultaneous events is an easily-missed leak.

It is enforced in [sampler.py](sampler.py) and nowhere else. Every model *and every
baseline*, including the GBDT's engineered features, reads its inputs from there. If each
model applied its own filter, the leakage rules would be per-model discipline and the tests
would only cover whichever model they happened to call.

Three things are gated on the same rule, because the prompt names all three:

- **features** — only visible rows are encoded;
- **neighbourhoods** — an invisible event does not reveal the product it points at;
- **normalization** — statistics are fitted on training rows only.

A fourth is handled in the suite: the train/validation/test split is **chronological**, so a
model cannot learn from a period it is later scored on.

### The canary

Asserting "we do not leak" proves nothing if there is nothing to find. Every generated
database therefore contains post-timestamp orders whose amount encodes the label exactly.
They are legitimate rows — they simply have not happened yet. `leakage_canary_strength`
reports what a leaking pipeline would score (1.00 in every regime), and the tests first show
that an ungated sampler *does* find them before asserting that the real one cannot.

## The five regimes

| Regime | The label depends on | What it detects |
| --- | --- | --- |
| `one_hop` | the entity's own events | any use of a related table at all |
| `multi_hop` | attributes of what those events point at | following foreign keys past one hop |
| `temporal` | recent events only; older ones point the other way | using timestamps rather than ignoring them |
| `irrelevant` | one-hop signal, buried under 5× more distractor rows | robustness to unrelated tables |
| `cold_start` | static attributes only — there is no history | graceful degradation |

`cold_start` is the regime where the target-only floor is also the ceiling, and a model
scoring above it there would be evidence of a bug, not of skill.

## What varies, one flag at a time

`use_types` (per-table row projections), `use_links` (message passing along foreign keys —
off is the homogeneous-GNN baseline), and `use_time` (elapsed-time channel and gate). Each
ablation is width-searched to match the prototype's parameter count, because removing the
typed encoders deletes three quarters of them and comparing at a shared width would confound
mechanism with capacity.

Two message-passing rounds is the minimum that can reach a linked row's attributes, and
[tests/test_relational.py](../../../../tests/test_relational.py) asserts exactly that: a
one-round model's prediction does not move when a product's price changes, and a two-round
model's does.

## Running it

```bash
modern-nn run-track relational       # writes results/relational/
python scripts/report_relational.py  # regenerates every table in reports/relational.md
```

See [reports/relational.md](../../../../reports/relational.md) for results.
