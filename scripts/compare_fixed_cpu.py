"""Fixed-checkpoint FER comparison, exposes selected test subset."""


def main():
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--expanded", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Choose a new output directory.")
    if not args.previous.is_file() or not args.expanded.is_file():
        p.error("Two local checkpoints required.")
    import os

    for k in ["OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"]:
        os.environ[k] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import hashlib
    import json
    from pathlib import Path

    import numpy as np
    import tensorflow as tf
    from PIL import Image

    root = args.data
    labels = sorted(p.name for p in (root / "train").iterdir() if p.is_dir())
    seen = set()
    for name in labels:
        for p in sorted((root / "train" / name).glob("*")):
            a = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
            seen.add(hashlib.sha256(a.tobytes()).hexdigest())
    xs = []
    ys = []
    counts = []
    rows = []
    duplicates = 0
    for c, name in enumerate(labels):
        count = 0
        for p in sorted((root / "test" / name).glob("*")):
            a = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
            h = hashlib.sha256(a.tobytes()).hexdigest()
            if h in seen:
                duplicates += 1
                continue
            seen.add(h)
            xs.append(a)
            ys.append(c)
            rows.append(dict(file=name + "/" + p.name, sha256=h))
            count += 1
            if count == 100:
                break
        counts.append(count)
    x = np.asarray(xs, dtype=np.float32)[..., None] / 255
    y = np.asarray(ys)
    metrics = []
    for label, checkpoint in [
        ("previous", str(args.previous)),
        ("expanded", str(args.expanded)),
    ]:
        m = tf.keras.models.load_model(checkpoint)
        ps = []
        for i in range(0, len(x), 64):
            ps.extend(m(x[i : i + 64], training=False).numpy().argmax(1).tolist())
        pred = np.asarray(ps)
        conf = np.zeros((7, 7), dtype=int)
        f = []
        for a, b in zip(y, pred, strict=False):
            conf[a, b] += 1
        for c in range(7):
            tp = conf[c, c]
            fp = conf[:, c].sum() - tp
            fn = conf[c, :].sum() - tp
            f.append(float(2 * tp / max(1, 2 * tp + fp + fn)))
        metrics.append(
            dict(
                model=label,
                accuracy=float((pred == y).mean()),
                macro_f1=float(np.mean(f)),
                per_class_f1=f,
                confusion=conf.tolist(),
            )
        )
    out = args.output
    out.mkdir(exist_ok=False)
    (out / "report.json").write_text(
        json.dumps(
            dict(
                source="sameFER2013mirror/testdirectory",
                selection=(
                    "first100unique/class sorted files, "
                    "excludes exacthashofALLtrainimages"
                ),
                images=len(y),
                counts=counts,
                labels=labels,
                duplicates_excluded=duplicates,
                metrics=metrics,
                limits=(
                    "one-time testsubset comparison after fixing both checkpoints; "
                    "no test tuning; dataset/identity provenance not independent; "
                    "now exposed not untouched for future; "
                    "not emotion/fairness validation"
                ),
            ),
            indent=2,
        )
    )
    (out / "manifest.json").write_text(json.dumps(rows, indent=2))
    print(metrics)


if __name__ == "__main__":
    main()
