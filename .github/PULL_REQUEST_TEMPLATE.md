# Summary

<!-- What changed and why. One track or one piece of shared infrastructure per PR. -->

## Type of change

- [ ] `feat` — new mechanism, baseline, task, or infrastructure
- [ ] `fix` — corrects incorrect behaviour or an incorrect result
- [ ] `docs` — documentation, specification, or report only
- [ ] `test` — tests only
- [ ] `refactor` / `perf` / `chore` / `ci` / `build`

## Scientific checklist

Delete the section if this PR contains no experimental result.

- [ ] Primary sources verified and recorded in `docs/source_registry.md`, including deviations.
- [ ] Track specification (`src/modern_nn_lab/tracks/<track>/README.md`) is current.
- [ ] Invariant tests added, not just forward-pass smoke tests.
- [ ] At least one baseline compared under a **matched parameter budget**.
- [ ] At least one ablation neutralizing the claimed mechanism.
- [ ] Multiple seeds with reported uncertainty, per `docs/experiment_contract.md`.
- [ ] Raw records written under `results/`; every figure and table is regenerable from them.
- [ ] Claim level stated and permitted by `docs/claim_policy.md`.
- [ ] Failures, divergences, and instabilities are reported rather than omitted.

Claim level for new results: <!-- educational implementation | compact reproduction | reference integration | research prototype | n/a -->

## Quality gate

```text
<!-- Paste the tail of: make check -->
```

- [ ] `ruff check .`
- [ ] `ruff format --check .`
- [ ] `mypy src`
- [ ] `pytest`

## Status update

- [ ] `STATUS.md` updated with completed work, tests run, unresolved failures, and the next
      atomic milestone.
- [ ] `CHANGELOG.md` updated under `Unreleased`.

## Notes for reviewers

<!-- Known limitations, deliberate approximations, and anything you want challenged. -->
