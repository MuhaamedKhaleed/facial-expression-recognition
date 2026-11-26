"""Explicit local development experiment. No downloads or import-time work."""


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new directory.")
    """Local FER mirror subset. No claims of emotion reading or official-test split."""
    import os

    for k in ["OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"]:
        os.environ[k] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import hashlib
    import json
    import time
    from pathlib import Path

    import numpy as np
    import tensorflow as tf
    from PIL import Image

    root = args.data
    out = args.output
    out.mkdir(exist_ok=False)
    labels = sorted(p.name for p in (root / "train").iterdir() if p.is_dir())
    rng = np.random.default_rng(42)
    seen = set()
    xs = []
    ys = []
    vs = []
    vys = []
    duplicates = 0
    # Deterministic hash dedup across train/dev; train-folder internal split, test fo
    for c, name in enumerate(labels):
        files = sorted((root / "train" / name).glob("*"))
        rng.shuffle(files)
        accepted = []
        for p in files:
            a = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
            h = hashlib.sha256(a.tobytes()).hexdigest()
            if h in seen:
                duplicates += 1
                continue
            seen.add(h)
            accepted.append((p, a))
        n = min(400, int(len(accepted) * 0.8))
        for _p, a in accepted[:n]:
            xs.append(a)
            ys.append(c)
        for _p, a in accepted[n : n + 100]:
            vs.append(a)
            vys.append(c)
    x = np.asarray(xs, dtype=np.float32)[..., None] / 255
    y = np.asarray(ys)
    vx = np.asarray(vs, dtype=np.float32)[..., None] / 255
    vy = np.asarray(vys)
    tf.keras.utils.set_random_seed(42)
    L = tf.keras.layers
    m = tf.keras.Sequential(
        [
            L.Input((48, 48, 1)),
            L.Conv2D(32, 3, activation="relu"),
            L.MaxPooling2D(),
            L.Conv2D(64, 3, activation="relu"),
            L.MaxPooling2D(),
            L.Conv2D(32, 3, activation="relu"),
            L.Flatten(),
            L.Dense(32, activation="relu"),
            L.Dense(7, activation="softmax"),
        ]
    )
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    start = time.time()
    history = []
    best = -1
    for epoch in range(30):
        order = rng.permutation(len(x))
        for i in range(0, len(x), 64):
            ix = order[i : i + 64]
            m.train_on_batch(x[ix], y[ix])
        p = m(vx, training=False).numpy().argmax(1)
        acc = float((p == vy).mean())
        f = []
        for c in range(7):
            tp = ((p == c) & (vy == c)).sum()
            fp = ((p == c) & (vy != c)).sum()
            fn = ((p != c) & (vy == c)).sum()
            f.append(float(2 * tp / max(1, 2 * tp + fp + fn)))
        r = dict(
            epoch=epoch + 1,
            dev_accuracy=acc,
            dev_macro_f1=float(np.mean(f)),
            seconds=time.time() - start,
        )
        history.append(r)
        if r["dev_macro_f1"] > best:
            best = r["dev_macro_f1"]
            m.save(out / "best.keras")
        report = dict(
            source="https://www.kaggle.com/datasets/msambare/fer2013",
            source_version=1,
            license=(
                "Kaggle page DbCL/ODbL; original web-face provenance retained, no "
                "face redistribution "
            ),
            labels=labels,
            train_images=len(y),
            dev_images=len(vy),
            train_counts=np.bincount(y, minlength=7).tolist(),
            dev_counts=np.bincount(vy, minlength=7).tolist(),
            duplicates_excluded=duplicates,
            history=history,
            test_read=False,
            limits=(
                "internal stratified subset from train directory only, not "
                "official PublicTest/PrivateTest; 30epochs fresh, no "
                "interrupted/resume state selected on dev, no "
                "fairness/external/clinical or internal-emotion validity; no "
                "weights or faces published "
            ),
        )
        (out / "report.json").write_text(json.dumps(report, indent=2))
        print(r, flush=True)


if __name__ == "__main__":
    main()
