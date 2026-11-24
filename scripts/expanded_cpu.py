"""Explicit local FER expansion. Same internal dev, never official test folder."""


def main():
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--balanced", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        p.error("Choose a new output directory.")
    if not args.checkpoint.is_file() or not (args.data / "train").is_dir():
        p.error("Local checkpoint and train directory required.")
    if not 1 <= args.epochs <= 50:
        p.error("Epochs must be1to50.")
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
    for c, name in enumerate(labels):
        files = sorted((root / "train" / name).glob("*"))
        rng.shuffle(files)
        accepted = []
        for p in files:
            a = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
            h = hashlib.sha256(a.tobytes()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            accepted.append(a)
        n = min(400, int(len(accepted) * 0.8))
        for a in accepted[n : n + 100]:
            vs.append(a)
            vys.append(c)
        training = accepted[:n] + accepted[n + 100 :]
        for a in training[:2000]:
            xs.append(a)
            ys.append(c)
    x = np.asarray(xs, dtype=np.float32)[..., None] / 255
    y = np.asarray(ys)
    vx = np.asarray(vs, dtype=np.float32)[..., None] / 255
    vy = np.asarray(vys)
    tf.keras.utils.set_random_seed(42)
    m = tf.keras.models.load_model(str(args.checkpoint))
    m.compile(
        optimizer=tf.keras.optimizers.Adam(0.0003),
        loss="sparse_categorical_crossentropy",
    )
    history = []
    best = -1
    start = time.time()
    for epoch in range(args.epochs):
        order = rng.permutation(len(x))
        for i in range(0, len(x), 64):
            ix = order[i : i + 64]
            a = x[ix].copy()
            flip = rng.random(len(a)) < 0.5
            a[flip] = a[flip, :, ::-1, :]
            # Up to3pixel shifts, edge fill, no dev augmentation.
            dy, dx = rng.integers(-3, 4, size=2)
            a = np.pad(a, ((0, 0), (3, 3), (3, 3), (0, 0)), mode="edge")[
                :, 3 + dy : 51 + dy, 3 + dx : 51 + dx, :
            ]
            weights = (
                len(y) / (7 * np.bincount(y))[y[ix]]
                if args.balanced
                else np.ones(len(ix))
            )
            m.train_on_batch(a, y[ix], sample_weight=weights.astype(np.float32))
        pred = m(vx, training=False).numpy().argmax(1)
        f = []
        for c in range(7):
            tp = ((pred == c) & (vy == c)).sum()
            fp = ((pred == c) & (vy != c)).sum()
            fn = ((pred != c) & (vy == c)).sum()
            f.append(float(2 * tp / max(1, 2 * tp + fp + fn)))
        r = dict(
            epoch=epoch + 1,
            accuracy=float((pred == vy).mean()),
            macro_f1=float(np.mean(f)),
            seconds=time.time() - start,
        )
        history.append(r)
        if r["macro_f1"] > best:
            best = r["macro_f1"]
            m.save(out / "best.keras")
        (out / "report.json").write_text(
            json.dumps(
                dict(
                    scope=(
                        "expanded12304train same677internaldev, continuation from "
                        "supplied "
                        "checkpoint, flip/shift augmentation"
                    ),
                    balanced=args.balanced,
                    epochs=args.epochs,
                    train=len(y),
                    dev=len(vy),
                    train_counts=np.bincount(y).tolist(),
                    history=history,
                    test_read=False,
                    limits=(
                        "reused internal dev selection, not official test or "
                        "emotion-reading/fairness proof"
                    ),
                ),
                indent=2,
            )
        )
        print(r, flush=True)


if __name__ == "__main__":
    main()
