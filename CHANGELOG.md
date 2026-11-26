# Changelog

## [Unreleased]

### Fixed
- Distinct predefined fitting/validation/final-test partitions, instead of reused validation/test rows.
- Historical online setup disabled and stale outputs cleared.

### Added
- Strict FER CSV/pixel/label/duplicate checks, original-width CPU CNN CLI and explicit final scoring.
- Actual synthetic backend/CLI/data tests, uv lock, Ruff/mypy/hooks, data-only Docker/CI.
- Honest privacy, consent, bias, dataset-label and model-quality limits.


### Engineering follow-up
- Use the committed uv lock in CI and key dependency-installer caching by that lock.
- Complete copy-only profile README drafts and clarify historical versus current verification.

### Real-data CSV route follow-up
- Pinned fetch script for the original FER2013 challenge CSV mirror and a reviewed-dedup preparation script with a removal ledger.
- Batched final-test prediction in `evaluate`; the previous single forward pass over every PrivateTest row exhausted memory on real-sized data. `evaluate` now also validates every row but materializes only PrivateTest arrays, so the documented full-CSV route is verified on a 1-CPU 2 GB machine.
- Documented verified fetch-to-final-test commands, metrics and a separately licensed pretrained baseline link.

### Synthetic demonstration follow-up

Added standalone invented-input backend demo, checked JSON output and documented limits. Opt-in ML container added where a model backend exists; default lightweight image unchanged. Hosted container verification is required separately.

## Portable notebook follow-up

Replace platform-specific notebook execution with safe-default local package entry points, explicit local input checks and opt-in training. Add notebook execution/argv/failure regressions. Historical model scores and full-scale data training are not claimed reproduced.
