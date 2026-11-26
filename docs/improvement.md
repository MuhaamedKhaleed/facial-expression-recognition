# Measured development alternative, 24 November 2026

Real-data CNN on the Kaggle mirror train folder, class-capped and exact-pixel deduplicated before internal split.2704train/677dev,30epochs fresh, devaccuracy .37814/macroF1 .37314. Previous pooled model and interrupted larger model are not this result. Test folder untouched. This is weak internal expression-label classification, not actual internal emotion, fairness or officialbenchmark validation. Mirror page declares DbCL/ODbL; original paper documents web-image provenance, with no blanket individual-image privacy/publicity clearance implied. No raw faces or model weights redistributed. Dataset access: https://www.kaggle.com/datasets/msambare/fer2013 . Original rules: https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/rules . Attribution: Goodfellow etal2013 https://arxiv.org/abs/1307.0414 . Terms: http://opendatacommons.org/licenses/dbcl/1.0/ and https://opendatacommons.org/licenses/odbl/1-0/ .

Only explicit local execution. No download/import side effects. Use a new output directory.

```sh
uv sync --locked --extra dev --extra ml
uv run --no-sync python scripts/improvement_cpu.py --data LOCAL_SOURCE --output NEW_OUTPUT
```

The portable script was rerun and reproduced the recorded metrics. Environment Python3.11 Linux x86_64 CPU. Raw data and trained weights stay outside GitHub. Original baseline and limitations remain in quality-cpu.md.

```json
{
  "source": "https://www.kaggle.com/datasets/msambare/fer2013",
  "source_version": 1,
  "license": "Kaggle page DbCL/ODbL; original web-face provenance retained, no face redistribution",
  "labels": [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
  ],
  "train_images": 2704,
  "dev_images": 677,
  "train_counts": [
    400,
    304,
    400,
    400,
    400,
    400,
    400
  ],
  "dev_counts": [
    100,
    77,
    100,
    100,
    100,
    100,
    100
  ],
  "duplicates_excluded": 1236,
  "history": [
    {
      "epoch": 1,
      "dev_accuracy": 0.15066469719350073,
      "dev_macro_f1": 0.049853672512944355,
      "seconds": 2.5667049884796143
    },
    {
      "epoch": 2,
      "dev_accuracy": 0.22304283604135894,
      "dev_macro_f1": 0.1459103661755876,
      "seconds": 4.132406949996948
    },
    {
      "epoch": 3,
      "dev_accuracy": 0.2259970457902511,
      "dev_macro_f1": 0.18872735507763436,
      "seconds": 5.659886360168457
    },
    {
      "epoch": 4,
      "dev_accuracy": 0.2998522895125554,
      "dev_macro_f1": 0.2193354399506788,
      "seconds": 7.25877833366394
    },
    {
      "epoch": 5,
      "dev_accuracy": 0.2968980797636632,
      "dev_macro_f1": 0.23996710668909152,
      "seconds": 8.829026460647583
    },
    {
      "epoch": 6,
      "dev_accuracy": 0.32644017725258495,
      "dev_macro_f1": 0.2999937220767828,
      "seconds": 10.405875444412231
    },
    {
      "epoch": 7,
      "dev_accuracy": 0.32644017725258495,
      "dev_macro_f1": 0.2906204776627335,
      "seconds": 11.99985671043396
    },
    {
      "epoch": 8,
      "dev_accuracy": 0.35893648449039883,
      "dev_macro_f1": 0.33320582258747594,
      "seconds": 13.603062629699707
    },
    {
      "epoch": 9,
      "dev_accuracy": 0.3559822747415066,
      "dev_macro_f1": 0.33053843204454675,
      "seconds": 15.227452516555786
    },
    {
      "epoch": 10,
      "dev_accuracy": 0.3663220088626292,
      "dev_macro_f1": 0.3566965462528203,
      "seconds": 16.803019285202026
    },
    {
      "epoch": 11,
      "dev_accuracy": 0.34268833087149186,
      "dev_macro_f1": 0.3186016040941072,
      "seconds": 18.406965494155884
    },
    {
      "epoch": 12,
      "dev_accuracy": 0.3604135893648449,
      "dev_macro_f1": 0.35169739600668815,
      "seconds": 19.997705221176147
    },
    {
      "epoch": 13,
      "dev_accuracy": 0.3722304283604136,
      "dev_macro_f1": 0.3680633431493215,
      "seconds": 21.55704617500305
    },
    {
      "epoch": 14,
      "dev_accuracy": 0.36336779911373707,
      "dev_macro_f1": 0.3487112899817387,
      "seconds": 23.141995668411255
    },
    {
      "epoch": 15,
      "dev_accuracy": 0.38847858197932056,
      "dev_macro_f1": 0.3816015354419538,
      "seconds": 24.72798204421997
    },
    {
      "epoch": 16,
      "dev_accuracy": 0.3412112259970458,
      "dev_macro_f1": 0.32168010462335445,
      "seconds": 26.33930993080139
    },
    {
      "epoch": 17,
      "dev_accuracy": 0.3766617429837518,
      "dev_macro_f1": 0.3608995447521291,
      "seconds": 27.94670033454895
    },
    {
      "epoch": 18,
      "dev_accuracy": 0.3722304283604136,
      "dev_macro_f1": 0.3608120914398949,
      "seconds": 29.471163988113403
    },
    {
      "epoch": 19,
      "dev_accuracy": 0.37813884785819796,
      "dev_macro_f1": 0.36158413148018725,
      "seconds": 31.014134883880615
    },
    {
      "epoch": 20,
      "dev_accuracy": 0.35450516986706054,
      "dev_macro_f1": 0.3472514037386091,
      "seconds": 32.59182691574097
    },
    {
      "epoch": 21,
      "dev_accuracy": 0.3722304283604136,
      "dev_macro_f1": 0.37237849507625703,
      "seconds": 34.11470603942871
    },
    {
      "epoch": 22,
      "dev_accuracy": 0.3663220088626292,
      "dev_macro_f1": 0.3664438312918857,
      "seconds": 35.65730404853821
    },
    {
      "epoch": 23,
      "dev_accuracy": 0.3707533234859675,
      "dev_macro_f1": 0.3547207258064722,
      "seconds": 37.19804525375366
    },
    {
      "epoch": 24,
      "dev_accuracy": 0.3663220088626292,
      "dev_macro_f1": 0.3672254705367847,
      "seconds": 38.72886657714844
    },
    {
      "epoch": 25,
      "dev_accuracy": 0.3559822747415066,
      "dev_macro_f1": 0.3504331200406318,
      "seconds": 40.258864641189575
    },
    {
      "epoch": 26,
      "dev_accuracy": 0.3825701624815362,
      "dev_macro_f1": 0.3700015675097546,
      "seconds": 41.844865560531616
    },
    {
      "epoch": 27,
      "dev_accuracy": 0.3810930576070901,
      "dev_macro_f1": 0.375671667426243,
      "seconds": 43.42674660682678
    },
    {
      "epoch": 28,
      "dev_accuracy": 0.3707533234859675,
      "dev_macro_f1": 0.36742149971998045,
      "seconds": 45.029821157455444
    },
    {
      "epoch": 29,
      "dev_accuracy": 0.3766617429837518,
      "dev_macro_f1": 0.36894533355562914,
      "seconds": 46.615593671798706
    },
    {
      "epoch": 30,
      "dev_accuracy": 0.37813884785819796,
      "dev_macro_f1": 0.3731384049530536,
      "seconds": 48.20525360107422
    }
  ],
  "test_read": false,
  "limits": "internal stratified subset from train directory only, not official PublicTest/PrivateTest; 30epochs fresh, no interrupted/resume state selected on dev, no fairness/external/clinical or internal-emotion validity; no weights or faces published",
  "portable_rerun_verified": true
}
```
