# FER Category Research

An explicit local CPU CNN workflow for FER2013-style category labels. It validates predefined partitions and keeps final-test scoring separate from fitting. It does **not** infer a person's true feelings, health, character or intent, and is not a surveillance, hiring or diagnostic tool.

## Install and invented demo

Python3.11/3.12, Linuxx86_64, uv installed:

```sh
uv sync --locked --extra dev
uv run fer-research --help
uv run python scripts/synthetic_demo.py
uv run fer-research check-data --data synthetic-demo/fer.csv
uv run fer-research check-data --config .env.example
```

Actual output:

```json
{"Training": 2, "PublicTest": 1, "PrivateTest": 1}
```

The example is constant grayscale arrays, not faces. It is only a wiring test and refuses an existing `synthetic-demo` folder. No data, credentials, pretrained checkpoint or automatic download is included. Explicit `--data` wins over an opt-in dotenv file's `FER_CSV`. The light install runs CSV checks without TensorFlow.

## CSV contract and leakage controls

UTF-8 columns `emotion,pixels,Usage`. Require integer labels0..6 and exactly2304 integer pixel values0..255 per row. Pixel arrays reshape to48x48x1 and scale by255. Required Usage values are exactly `Training`, `PublicTest`, `PrivateTest`; all must be nonempty. Missing, malformed or unknown rows fail, without a random-split fallback. Exact duplicate pixel arrays are rejected even within one partition, because contradictory labels and cross-partition copies need review, not silent dropping. This stricter rule can reject the original corpus until duplicates are reviewed.

The original notebook used the same held-out rows both for fitting-time validation and its reported final test. This pass uses `PublicTest` only for validation and reserves `PrivateTest` scoring for a separate command. Full CSV structure is checked during training, but final-test labels are never passed into fitting or validation. Repeated final-test scoring still turns it into development evidence; code cannot enforce scientific independence on its own. No subject IDs are present, so partition names do not prove person/source/domain independence. Near duplicates are not detected.

Dataset label order is Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral. These are dataset annotations, not verified internal emotions. Real-world use needs consent, privacy, demographic/bias and context review. No camera capture or face detection is implemented.

## Explicit CPU training and final test

```sh
uv sync --locked --extra dev --extra ml
TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 uv run fer-research train synthetic-demo/fer.csv synthetic-demo/trained --epochs 1 --batch-size 2
TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 uv run fer-research evaluate synthetic-demo/fer.csv synthetic-demo/trained/model.keras
```

These exact commands were tested on invented arrays only. Training refuses an existing output directory, uses original three-convolution CNN widths32/64/64 and seven-category softmax, saves `model.keras` plus `validation.json` after finite metrics. CPU-only, no silent GPU or remote process. `evaluate` loads an explicitly supplied trusted local `.keras` model with `safe_mode=True` and scores only PrivateTest. No remote or pickle loading is offered. A model file must be trusted even with safe mode.

Metrics are sparse cross-entropy and aggregate classification accuracy. No real FER accuracy, macro-F1, calibrated confidence, demographic fairness, subject-independent or real-world quality is claimed. Invented demo accuracy has no useful model-quality meaning. Large real-data training can take significant time/memory.

## Engineering checks

```sh
TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 uv run pytest --cov=fer_research --cov-report=term-missing
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
uv run pre-commit run --all-files
uv run python -m fer_research --help
```

20 meaningful tests,94% statement coverage, including actual original-width CNN forward/train/save/load/evaluate, CLI execution, label/pixel validation and a test that captures the labels used in fitting/validation to exclude PrivateTest. Light-only install explicitly skips the TensorFlow module. Coverage is not model quality.

## Docker

```sh
docker build --tag fer-research:ci .
docker run --rm fer-research:ci --help
docker run --rm -v "$PWD/synthetic-demo:/data:ro" fer-research:ci check-data --data /data/fer.csv
```

Multi-stage slim non-root runtime, no TensorFlow/dev dependencies. It checks data only, not fitting/evaluation. Hosted CI verifies build/run; no local Docker daemon is claimed. Public-repository-only ordinary Linux workflow, no paid APIs, hosted training, deployment or private billable runners. Local uv lock is Linuxx86_64-scoped; CI pip uses pinned direct dependencies, not the full transitive lock. Windows/macOS/ARM/GPU are unverified.

## Layout

```text
CSV -> core.read -> validated predefined partitions
Training -> explicit CPU model.train <- PublicTest validation
trusted local model + PrivateTest -> explicit model.evaluate
Typer CLI -> JSON output / separate stderr errors
```

`core.py` validates arrays/labels/partitions. `model.py` owns lazy TensorFlow fitting and explicit final scoring. `cli.py` is thin. Historical notebook output is cleared; online credential/download setup disabled and leakage repaired, but full notebook execution remains unverified. The CLI does not execute it.

Next work: review a licensed real corpus and duplicates, freeze an untouched subject-independent test where possible, measure per-class/macro metrics, inspect confusion/error/bias slices and document consent and use limitations. Do not turn dataset categories into claims about a person.

[Verification](docs/verification.md), [changelog](CHANGELOG.md), [copy-only presentation](docs/presentation.md). No main merge, license change, release/tag, real face/model/data publication.


## License scope

New project-owned package/engineering files use the owner's MIT default. See LICENSE and LICENSE-SCOPE.md. This is not permission to relicense historical notebooks, third-party code, datasets, models or weights. Their provenance and terms still need review. Earlier verification entries saying no license was added describe the initial pilot, not this follow-up.

## Checked synthetic wiring demo

See [commands, saved output and scope](docs/synthetic-demo.md). This is invented-input execution evidence, not model accuracy or a real-data gallery.

## Portable notebook follow-up

The root notebook now uses the supported local package with safe defaults and explicit data/training opt-ins. It replaces the platform-specific experiment rather than reproducing historical results. See `docs/notebook-migration.md` and the notebook settings. Earlier descriptions of live notebook downloads/hard-coded paths are historical and superseded. No dataset or weight rights are granted.


## Ordinary-image inference follow-up

See [commands, artifact contracts and limits](docs/fix-pass.md). This adds explicit local inference, not a new model-quality claim.


## Real-data route and pretrained baseline

See [commands, pinned source, duplicate-review ledger, verified metrics and limits](docs/real-data-csv.md). A fresh clone can fetch the pinned original-challenge CSV, prepare it with counted duplicate review, train the original-width CNN on 27473 rows and score 0.5067 accuracy on the reserved 3268-row PrivateTest partition once. A separately licensed pretrained baseline is linked with its own pinned weights hash; its card numbers use a different split and protocol and are not comparable.
