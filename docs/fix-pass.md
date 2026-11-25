# Ordinary image inference, 25 November 2026

The original-width CNN package now provides `fer-research predict FACE_CROP.png MODEL/model.keras`. Install locked `ml` extras first. This command accepts a supplied face crop, not a scene, camera or tracking stream. Grayscale resize48x48 and division by255 match CSV training. It returns dataset category probabilities, not internal feelings or calibrated confidence.

Training writes `labels.json` beside `model.keras`. Predict requires this exact ordered seven-category manifest and refuses missing metadata instead of guessing. Original CSV order is Angry,Disgust,Fear,Happy,Sad,Surprise,Neutral. The separate alphabetical folder-training experiments use angry,disgust,fear,happy,neutral,sad,surprise. Do not copy the wrong label manifest beside a model. Original historical .h5 weights are not included or newly recovered.

Tests exercise image decode/preprocessing, actual original-width CNN save/reload, JSON CLI, category manifest and invalid shape. This adds usable inference, not new face-quality evidence. The previously exposed679-image alternative comparison remains development evidence and does not compare against original historical54.47%.

## New original-width CPU run

Explicit5epoch fresh original-width32/64/64CNN,dense64 trained on12304unique train-folder images with the same677internal dev images as earlier alternative work. No augmentation, batch128,seed42. Final dev accuracy.39143,macroF1.33266. This is not a reproduction of the historical random-split54.47% or a fair equal-budget alternative ranking. The interrupted10epoch attempt produced no artifact; this completed5epoch run is the reported result. Official test folder was not read. Reused internal dev is not an untouched benchmark. Script `scripts/original_objective_cpu.py` reproduces the selection/run, writes pixel-hash split ledger, label order and model hash; artifacts remain local. No quality improvement asserted.
