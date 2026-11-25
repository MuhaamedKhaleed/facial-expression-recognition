"""Explicit original-width FER CNN run on prior internal development split only."""


def main():
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=5)
    args = p.parse_args()
    if args.output.exists() or not 1 <= args.epochs <= 20:
        p.error("New output and epochs1..20required.")
    import hashlib
    import json
    import os
    import time

    for k in ["OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"]:
        os.environ[k] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import numpy as np
    from PIL import Image

    from fer_research.model import build

    labels = sorted(x.name for x in (args.data / "train").iterdir() if x.is_dir())
    if labels != ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]:
        p.error("Expected seven known folder categories.")
    rng = np.random.default_rng(42)
    seen = set()
    xs = []
    ys = []
    vs = []
    vys = []
    ledger = []
    for c, name in enumerate(labels):
        files = sorted((args.data / "train" / name).glob("*"))
        rng.shuffle(files)
        accepted = []
        for path in files:
            with Image.open(path) as image:
                a = np.asarray(image.convert("L"), dtype=np.uint8)
            if a.shape != (48, 48):
                raise ValueError("Expected native48x48 source.")
            h = hashlib.sha256(a.tobytes()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            accepted.append((a, h))
        n = min(400, int(len(accepted) * 0.8))
        dev = accepted[n : n + 100]
        train = (accepted[:n] + accepted[n + 100 :])[:2000]
        for a, h in dev:
            vs.append(a)
            vys.append(c)
            ledger.append(["dev", c, h])
        for a, h in train:
            xs.append(a)
            ys.append(c)
            ledger.append(["train", c, h])
    x = np.asarray(xs, dtype=np.float32)[..., None] / 255
    y = np.asarray(ys)
    vx = np.asarray(vs, dtype=np.float32)[..., None] / 255
    vy = np.asarray(vys)
    model = build(42)
    args.output.mkdir(parents=True)
    history = []
    start = time.monotonic()
    for epoch in range(args.epochs):
        order = rng.permutation(len(x))
        for i in range(0, len(x), 128):
            ix = order[i : i + 128]
            model.train_on_batch(x[ix], y[ix])
        predictions = model(vx, training=False).numpy().argmax(1)
        fs = []
        for c in range(7):
            tp = ((predictions == c) & (vy == c)).sum()
            fp = ((predictions == c) & (vy != c)).sum()
            fn = ((predictions != c) & (vy == c)).sum()
            fs.append(float(2 * tp / max(1, 2 * tp + fp + fn)))
        row = {
            "epoch": epoch + 1,
            "dev_accuracy": float((predictions == vy).mean()),
            "dev_macro_f1": float(np.mean(fs)),
            "seconds": time.monotonic() - start,
        }
        history.append(row)
        print(row, flush=True)
    model.save(args.output / "model.keras")
    (args.output / "labels.json").write_text(json.dumps([s.title() for s in labels]))
    (args.output / "split-hashes.json").write_text(json.dumps(ledger))
    result = {
        "architecture": "original32/64/64conv,dense64,7softmax",
        "protocol": "same12304train677internaldev as earlier alternative, exactpixel dedup",
        "source": "https://www.kaggle.com/datasets/msambare/fer2013",
        "seed": 42,
        "batch_size": 128,
        "epochs": args.epochs,
        "train": len(y),
        "dev": len(vy),
        "augmentation": False,
        "history": history,
        "test_folder_read": False,
        "limits": "different protocol from historical randomsplit54.47%, not replication; reuseddev not untouched benchmark or emotion inference",
        "model_sha256": hashlib.sha256(
            (args.output / "model.keras").read_bytes()
        ).hexdigest(),
    }
    (args.output / "report.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
