# Expanded real-data FER continuation, 24 November 2026

This is still weak expression-label classification, not internal feelings, fairness validation or a production system. Original source/rights and no-redistribution caveats in improvement.md remain. Source: https://www.kaggle.com/datasets/msambare/fer2013 , original paper https://arxiv.org/abs/1307.0414 . No raw faces or model weights published. Dataset mirror DbCL/ODbL declaration does not waive privacy/publicity rights of each image.

## Training and development

Preserve identical677internal dev images from prior script. Expand training from2704to12304images, classcap2000except304disgust images. Exact-pixel dedup across train/dev. Remaining training excludes dev slice. Start from the previous30epoch best checkpoint. Eight epochs of flip/3pixel-shift augmentation at Adam .0003, then12class-weighted epochs, then12moreclass-weighted epochs. Each stage loads preceding best devmacroF1 checkpoint and restarts the optimizer and seed42data order. This is an explicitly multi-stage, dev-selected continuation, not a fresh single run or controlled ablation. DevmacroF1 .45157/accuracy .46233 vs previous .37314/.37814. All three portable stages independently reproduced every epoch metric of the exploratory stages exactly.

## Frozen-model comparison

Only after fixing both checkpoints, read first100unique images per class from mirror test directory, excluding exact pixel hashes of ALL original train images.679images total (disgust79, others100). Previous model accuracy .32695/macroF1 .32016; expanded .46686/.45647. No training or tuning on that subset, no claim of full official benchmark or identity-independent evaluation. This set is now exposed and must not be reused as an untouched test for future work. Per-class F1 and confusion matrix below make weak classes visible, fear F1 only .2264. Evaluator rerun exactly reproduced the comparison.

## Explicit portable sequence

Use existing locked repository environment with dev/ml extras and source mirror root. Checkpoints stay local. First reproduce the previous script from improvement.md, then:

```sh
uv run --no-sync python scripts/expanded_cpu.py --data SOURCE --checkpoint PREVIOUS30EPOCH/best.keras --output EXPAND8 --epochs 8
uv run --no-sync python scripts/expanded_cpu.py --data SOURCE --checkpoint EXPAND8/best.keras --output BALANCED12 --epochs 12 --balanced
uv run --no-sync python scripts/expanded_cpu.py --data SOURCE --checkpoint BALANCED12/best.keras --output BALANCED12MORE --epochs 12 --balanced
uv run --no-sync python scripts/compare_fixed_cpu.py --data SOURCE --previous PREVIOUS30EPOCH/best.keras --expanded BALANCED12MORE/best.keras --output FIXED_COMPARISON
```

Import/help are side-effect free, no downloads, outputs must be new. Source acquisition and accepting dataset terms are separate. Script-selected checkpoint history and explicit steps matter: do not present just the last12epochs as the entire training budget. Future improvement needs fresh evaluation rather than optimizing this now-exposed subset.

```json
{
  "selection": "first100unique/class sorted files, excludes exacthashofALLtrainimages",
  "images": 679,
  "counts": [
    100,
    79,
    100,
    100,
    100,
    100,
    100
  ],
  "labels": [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
  ],
  "duplicates_excluded": 108,
  "metrics": [
    {
      "model": "previous",
      "accuracy": 0.3269513991163476,
      "macro_f1": 0.3201575454777452,
      "per_class_f1": [
        0.19753086419753085,
        0.32954545454545453,
        0.11976047904191617,
        0.5252525252525253,
        0.3794466403162055,
        0.24888888888888888,
        0.4406779661016949
      ],
      "confusion": [
        [
          16,
          22,
          8,
          7,
          22,
          18,
          7
        ],
        [
          10,
          29,
          7,
          3,
          16,
          11,
          3
        ],
        [
          10,
          13,
          10,
          12,
          18,
          24,
          13
        ],
        [
          2,
          4,
          7,
          52,
          18,
          11,
          6
        ],
        [
          6,
          9,
          8,
          7,
          48,
          20,
          2
        ],
        [
          11,
          8,
          11,
          10,
          25,
          28,
          7
        ],
        [
          7,
          12,
          16,
          7,
          6,
          13,
          39
        ]
      ]
    },
    {
      "model": "expanded",
      "accuracy": 0.46686303387334316,
      "macro_f1": 0.45646551912135197,
      "per_class_f1": [
        0.35555555555555557,
        0.4834123222748815,
        0.22641509433962265,
        0.6764705882352942,
        0.48677248677248675,
        0.3251231527093596,
        0.6415094339622641
      ],
      "confusion": [
        [
          32,
          21,
          6,
          6,
          5,
          22,
          8
        ],
        [
          8,
          51,
          3,
          2,
          1,
          8,
          6
        ],
        [
          11,
          16,
          18,
          10,
          10,
          17,
          18
        ],
        [
          9,
          6,
          4,
          69,
          6,
          5,
          1
        ],
        [
          7,
          13,
          4,
          11,
          46,
          15,
          4
        ],
        [
          10,
          17,
          13,
          4,
          16,
          33,
          7
        ],
        [
          3,
          8,
          11,
          2,
          5,
          3,
          68
        ]
      ]
    }
  ],
  "limits": "one-time testsubset comparison after fixing both checkpoints; no test tuning; dataset/identity provenance not independent; now exposed not untouched for future; not emotion/fairness validation"
}
```
