# Roadmap

## Where this repository stands

All eleven architecture tracks and the cross-track integration are complete, and the program
milestones in [`docs/milestones.md`](docs/milestones.md) — M0 through M6, including the exit
condition — are satisfied: every track meets its local acceptance criteria, and the
cross-track report regenerates entirely from saved machine-readable results.

That is a statement about **completeness of the scaffold**, not about scientific weight. What
exists is eleven carefully-instrumented prototypes on synthetic data, 987 validated records,
and an honest account of what each does and does not establish. The roadmap below is ordered
by how much each item would change that second sentence.

## What this repository is, and is not

**It is** a controlled environment for asking whether a mechanism does what it is said to do,
with baselines that are given a fair chance and diagnostics that can falsify the headline.
Its most reliable output so far is methodological: in six of eleven tracks a plausible result
turned out to be an artefact of the baseline or the measurement, and each was caught by a
control rather than by the task metric.

**It is not** a benchmark, a reproduction of any paper, or evidence about behaviour at scale.
Five tracks were built without reading their primary source. Every dataset is synthetic and
was designed to isolate the mechanism under test. No track ran on an accelerator.

## Priorities

Ordered by effect on what the repository can legitimately claim, not by effort.

### 1. Read the five unread primary sources

**Tracks 07 (PFN), 08 (Relational), 09 (Sparse MoE), 10 (Flow Matching), 11 (JEPA).**

This is the highest-value item and among the cheapest. Those five tracks were built from the
track prompts, so a misreading in a prompt would propagate into the implementation
undetected, and nothing in them may be cited as evidence about the work they are named after.
Reading the sources would let the claim audit reconsider each — several implementations are
likely faithful enough for `educational implementation`, and the Flow Matching track may
support `compact reproduction`, which is where its registry entry originally aimed.

Concretely: read the source, add the equation-to-code mapping the first six tracks carry, and
re-run `scripts/report_synthesis.py` to regenerate the claim audit.

### 2. Put one real dataset behind at least three tracks

Every result rests on data this repository generated to suit the mechanism. That makes the
tests clean and makes generalization entirely unevidenced. The tracks where this bites
hardest, and where a real dataset is cheapest:

- **PFN** — OpenML tabular tasks, which is the setting the method exists for;
- **Relational** — any genuinely multi-table dataset, since the synthetic schema is a
  two-hop star and §8 of its report shows the sampler was doing most of the relational work;
- **JEPA** — real images or audio, where content and nuisance are not cleanly separable by
  construction as they are here.

Expect the conclusions to move. The relational and JEPA tracks both found their novel
mechanism matching rather than beating simpler baselines on synthetic data; real data usually
widens such gaps in one direction or the other.

### 3. Run on an accelerator

Three separate claims are currently **untestable rather than untested**:

- **Mamba-3's MIMO state** — its benefit is decode-time arithmetic intensity, which a Python
  scan on CPU cannot exhibit. This is recorded as untestable, not as a negative result.
- **Sparse MoE throughput** — sparse dispatch is a gather, small matmuls, and a scatter here;
  the setting where sparsity pays needs fused kernels and expert parallelism.
- **Peak memory** — unmeasured across all 987 records, because `profiling.peak_memory`
  correctly returns `None` on CPU rather than guessing.

One accelerator pass would convert all three from gaps into measurements.

### 4. Vary the dataset seed, not only the training seed

Most tracks fix the data-generating seed and vary initialization, so the reported intervals
are narrower than they look and say nothing about variability across problem draws. The
Nested Learning track already documents a version of this failure, where deterministic
learners on a fixed stream produced five identical runs that *read* as evidence of stability.

### 5. Reproduce the committed records by re-running every suite

The reproducibility audit verifies that the artefacts are internally consistent — records
validate, reports derive from them, hashes are stable — but **not** that today's code
reproduces the committed numbers. Determinism is exercised per track by seeded tests, which
checks that a run repeats rather than that it repeats *what is committed*. A scheduled full
re-run comparing against the configuration hashes would close this.

### 6. Populate `flops_per_sample` across tracks

The schema field is empty in all 987 records. Sparse MoE computes analytic FLOP counts but
stores them in track-local configuration, so the one trustworthy cost number in the
repository is also the least accessible. Analytic FLOPs are computable for most tracks and,
unlike throughput, are not corrupted by machine contention.

### 7. Per-track next experiments

Each report closes with its own list, and those are the sharpest questions in the repository
because they were written against results. The ones that would most change a conclusion:

- **Relational** — give the baselines a genuinely *unjoined* view. The current "flat"
  baselines consume relationally sampled neighbourhoods, so the track measured foreign-key
  propagation against pooling over a correct join, not relational modelling against
  flattening.
- **Sparse MoE** — expert-choice routing, which removes the capacity-drop failure mode by
  construction rather than by tuning.
- **PFN** — sweep context size on the out-of-prior family, to test the claim that more
  context cannot rescue a misspecified prior.
- **Flow Matching** — extend the closed-form reference to a Gaussian *mixture*, so the
  error-separation analysis covers a multi-modal case.
- **JEPA** — a factor structure where content and nuisance are correlated, which is what real
  data always presents and what the clean split here deliberately avoids.

## Explicit non-goals

- **No aggregate leaderboard.** The tracks measure accuracies, mean squared errors, energy
  distances and probe R-squareds on unrelated data. Averaging them would be arithmetic on
  incompatible units, and the ordering would reflect which tracks chose metrics with larger
  dynamic range. This is a standing prohibition, not a deferred feature.
- **No scale chasing.** The repository exists to make mechanisms legible, and a larger model
  would make the diagnostics harder to read, not easier.
- **No state-of-the-art claims.** Forbidden by [`docs/claim_policy.md`](docs/claim_policy.md)
  without evidence this repository is not structured to produce.
- **No claim-level upgrade without the corresponding work.** A label moves only when the
  evidence behind it moves — three levels were downgraded during the integration audit for
  exactly this reason.

## How to pick up any of this

1. `reports/cross_track_synthesis.md` — what every track established, and the gaps.
2. `STATUS.md` — the eight remaining scientific gaps, and the defects worth carrying forward.
3. `results/artefacts/experiment_index.json` — the exact configuration behind any number.
4. `docs/claim_policy.md` — what may be claimed at each level, and what each requires.
