# FER2013 Category Research

A local CPU CNN workflow for FER2013-style image category labels, with strict validation of the predefined partitions and final-test scoring kept separate from fitting. **It does not infer a person's real feelings, health or intent, and is not a surveillance, hiring or diagnostic tool.**

## Results (honest)

- Fresh-clone full run (25 Nov 2026, one CPU core): **0.5067 accuracy on the reserved 3,268-row PrivateTest partition**, trained on 27,473 rows with counted duplicate review.
- Quick 2-epoch demo run: 0.42 test accuracy in under 100 s.
- A separately licensed pretrained baseline is linked in `docs/real-data-csv.md`; its card numbers use a different split/protocol and are not comparable.
- The historical notebook reused the same held-out rows for validation and its reported test score; this version validates on PublicTest and reserves PrivateTest for a separate explicit command.
- 20 tests, 94% statement coverage, including a test that captures which labels were used in fitting to exclude PrivateTest. No macro-F1, calibration, demographic fairness or subject-independent evaluation is claimed.

## Quickstart (fresh clone, about 5 minutes)

Needs Python 3.11 or 3.12 (locked wheels have no 3.13+ build), internet for the first install/downloads, ~2 GB free RAM.

```sh
uv sync --python 3.11 --locked --extra dev --extra ml
uv run --no-sync python scripts/fetch_fer2013.py --output fer-src                 # 300 MB, pinned mirror, hash checked
uv run --no-sync python scripts/prepare_fer2013.py --input fer-src/icml_face_data.csv --output fer-prepared
uv run --no-sync fer-research train fer-prepared/fer2013.csv fer-trained --epochs 2 --batch-size 128
uv run --no-sync fer-research evaluate fer-prepared/fer2013.csv fer-trained/model.keras
uv run --no-sync fer-research predict FACE_CROP.png fer-trained/model.keras
```

`evaluate` prints one prediction per test row, so redirect it to a file. FER2013 has its own terms; see `LICENSE-SCOPE.md`.

## Limits

- Dataset labels (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) are annotations, not verified emotions. Real-world use needs consent, privacy and bias review. No camera capture or face detection is included.
- No subject IDs exist in the data, so partitions do not prove person independence; near-duplicates are not detected. Repeated final-test scoring turns it into development evidence.
- Exact duplicate pixel arrays are rejected even within one partition, which can reject the original corpus until duplicates are reviewed.
- Verified on Linux x86_64; the container runs data checks only (no TensorFlow). Free hosted CI covers tests, lint, CLI and Docker build.

## License

MIT for the new project-owned code (see `LICENSE` and `LICENSE-SCOPE.md`). The dataset, pretrained baseline and historical notebook keep their own terms; no real faces, models or data are published. Details and evidence: `docs/verification.md`, `docs/real-data-csv.md`, `CHANGELOG.md`.
