# Verification record

Original issue: the notebook randomly split all rows and reused `X_test,y_test` during validation and final evaluation. This pass respects explicit Training/PublicTest/PrivateTest values and moves final scoring into a separate command. Historical online setup/credentials disabled, outputs cleared, `.keras` safe reload used. Full notebook remains unrerun, not a tested standalone path.

## Commands

- `uv sync --locked --extra dev --extra ml`: working locked Linuxx86_64 environment.
- `TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 uv run pytest --cov=fer_research --cov-report=term-missing`:20passed,94% coverage.
- `uv run ruff check src tests`, `uv run ruff format --check src tests`, `uv run mypy src`, `uv run pre-commit run --all-files`:pass.
- `uv run python scripts/synthetic_demo.py`, followed by `uv run fer-research check-data --data synthetic-demo/fer.csv` or `--config .env.example`:2Training,1PublicTest,1PrivateTest.
- `TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 uv run fer-research train synthetic-demo/fer.csv synthetic-demo/trained --epochs 1 --batch-size 2`:actual CNN fit, saved local artifact/finite validation metrics. PrivateTest not scored.
- `TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 uv run fer-research evaluate synthetic-demo/fer.csv synthetic-demo/trained/model.keras`:explicit PrivateTest-only synthetic scoring.
- Module/console/help commands pass. Fresh light install excludes TensorFlow and skips its tests explicitly.

All test images are invented arrays. Actual original-width CNN forward, training, safe save/reload, evaluation and CLI run; a separate capture test confirms train labels0/1 and validation label2 are used, never test label3. No real FER dataset was downloaded or trained. No real emotion-recognition accuracy is established. Aggregate accuracy is not per-class fairness or a valid measure of a person's feelings.

Hosted CI builds/runs a light non-root data-check image; no local-daemon result is claimed. No ML backend in Docker. CI pip/direct pins, local uv scoped lock; no claim of cross-platform/GPU testing. Public-only ordinary hosted Linux job, no paid services or external training.

## Phase notes

Installable src package, meaningful data/model/CLI split, typed functions, actionable guards/logging, opt-in dotenv and example, ignored secrets/face/model artifacts, runtime/dev/optionalML groups and uv lock, actual tests and quality tooling, public-only CI, multi-stage data image, tested usage and architecture, changelog and separate notebook fix, copy-only metadata. No unused helper stubs. No main merge/tag/release/license change or fake history. Web-editor file commits do not pretend a phase-perfect local sequence.

## Deliberate changes and limits

Strict CSV replaces implicit Colab uploads. Require all predefined partitions, reject exact duplicate pixels everywhere, never drop silently. A real corpus may fail until duplicate-label/partition review. Only structure is read across all partitions during training; PrivateTest labels are not used to optimize/validate. No subject IDs, no near-duplicate or subject-independence proof. No frozen external test, full notebook run, real-data macro-F1/confusion/bias/calibration, human review, production inference/camera service, or demographic/generalization result. Repeated explicit test evaluation still compromises test independence. Local `.keras` must be trusted. Artifacts are not saved transactionally. Dataset categories must not be treated as mental state, medical diagnosis, personality or intent. No face/model/data is published.


## Gap-closure follow-up (24 November 2026)

The CI source now uses uv 0.13.8 and the committed transitive lock, rather than a new pip resolution of direct pins. Dependency installer caching is keyed by uv.lock. The local matching command is `uv sync --locked --extra dev --extra ml`, followed by the existing tests/quality commands. This does not cache or publish private datasets, weights or model outputs. Public-repository-only guards remain. The changed workflow's hosted outcome must be checked after publication; prior green runs do not validate a changed workflow.

A complete copy-only project-focused profile README draft is appended to presentation.md. No profile account edits, main merge, PR, release or license change is implied by that text. Historical notebook/full-data/quality limitations remain open, not replaced by a passing synthetic test.
