# Real-data CSV route, 25 November 2026

A fresh-clone route from a pinned public download to trained model output on the
original FER2013 challenge CSV, with predefined partitions kept. This is weak
expression-label classification, not internal feelings, fairness validation or a
production system. Consent, privacy, demographic/bias and context review limits
in the README still apply.

## Source and rights

Mirror: https://huggingface.co/datasets/DerrickUnleashed/FER-2013 revision
5b1d92d29dea477a4c13af5f928b2baee3744951, file `icml_face_data.csv`, 301072768
bytes, SHA-256
2c7ce224ef03ec2e2e522d10b93a069900b26fa1c028174d91d5f1f8013a2884. On 25
November 2026 the served bytes matched this pinned hash, which is also the
mirror's recorded LFS content hash. The mirror's card declares MIT; that is the
uploader's declaration and does not clear the original web-face provenance.
Original challenge rules:
https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/rules
. Attribution: Goodfellow et al. 2013, https://arxiv.org/abs/1307.0414 . No raw
faces, predictions on identifiable people or model weights are redistributed.
This CSV mirror keeps the original `Training`/`PublicTest`/`PrivateTest`
partitions (28709/3589/3589 rows), unlike the image-folder mirror in
improvement.md, so this route measures on the original held-out partitions.

## Duplicate review, not silent dropping

`scripts/prepare_fer2013.py` validates every raw row under the same contract as
`fer_research.core.read`, then removes exact-duplicate pixel rows by keeping the
first occurrence and counting every removal in `ledger.json`. Observed on the
pinned file: 1853 duplicate rows removed (1210 same label and partition, 643
with conflicting label or partition), 34034 rows kept
(Training 27473, PublicTest 3293, PrivateTest 3268). The raw file and ledger
stay local.

## Verified run

Python 3.12, Linux x86_64, 1 CPU, TensorFlow 2.18.0, locked environment, seed
42, 25 November 2026. `check-data` accepted the prepared CSV. Training 10
epochs at batch 128 took about four minutes:
`validation_loss 1.3199, validation_accuracy 0.5120` (PublicTest only).
`evaluate` scored the reserved partition once:
`test_loss 1.3075, test_accuracy 0.5067` (PrivateTest 3268 rows).
`predict` on a real held-out happy crop returned Happy at 0.9905. These are
original-partition, deduplicated numbers; they are not comparable to the
historical random-split 54.47%, to the folder-mirror development numbers in
improvement.md and fix-pass.md, or to pretrained-model cards. Repeated final
scoring would turn PrivateTest into development evidence; do not tune on it.

```sh
uv sync --locked --extra dev --extra ml
uv run python scripts/fetch_fer2013.py --output fer-src
uv run python scripts/prepare_fer2013.py --input fer-src/icml_face_data.csv --output fer-prepared
uv run fer-research check-data --data fer-prepared/fer2013.csv
TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 uv run fer-research train fer-prepared/fer2013.csv fer-trained --epochs 10 --batch-size 128
TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 uv run fer-research evaluate fer-prepared/fer2013.csv fer-trained/model.keras
uv run fer-research predict FACE_CROP.png fer-trained/model.keras
```

## Pretrained baseline, no training required

Anyone can instead download already-trained weights without this project
publishing any: https://huggingface.co/trpakov/vit-face-expression revision
ef0bc6fc34241b6587e7e009e7711357be28c024, license Apache-2.0 per its card front
matter, a ViT fine-tuned from `google/vit-base-patch16-224-in21k` on the FER2013
image-folder mirror per its card. `model.safetensors` is 343239356 bytes,
SHA-256
48acf03b1fd90ea45c5e91eb3be2f364cee1c1342639962bd4eae1eac2ad2f93; the download
and hash were verified on 25 November 2026. Its card reports validation accuracy
0.7113 and test accuracy 0.7116. This is a separate baseline: different
architecture, training split and evaluation protocol, so those numbers must not
be compared with this repository's figures above. It needs PyTorch/Transformers
and its own environment review; it is not vendored or re-licensed here. Face
image rights caveats apply to any inputs it is run on.
