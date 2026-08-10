# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Because this is a research repository, entries also record **claim levels** for new
experimental results, as defined in [`docs/claim_policy.md`](docs/claim_policy.md).

## [Unreleased]

### Added

- MIT license, contribution guide, code of conduct, security policy, and citation metadata.
- Issue and pull-request templates, `CODEOWNERS`, and Dependabot configuration.
- CodeQL analysis workflow and an extended CI matrix with coverage reporting.
- `.editorconfig` and `.gitattributes` for consistent cross-platform checkouts.

### Changed

- Pre-commit now runs the standard hygiene hooks and `mypy` alongside Ruff.
- `pyproject.toml` declares project URLs, classifiers, keywords, and coverage settings.

## [0.1.0] - 2026-08-08

### Added

- Initial research scaffold: package layout, track registry, typed contracts,
  reproducibility helpers, CLI, scaffold validator, and CI.
- Repository documentation: architecture, benchmark protocol, claim policy, experiment
  contract, mathematical notation, milestones, source registry, and track matrix.
- Track prompts for eleven architecture tracks plus the final integration prompt.

[Unreleased]: https://github.com/DiogoRibeiro7/modern-neural-networks-agent-repo/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DiogoRibeiro7/modern-neural-networks-agent-repo/releases/tag/v0.1.0
