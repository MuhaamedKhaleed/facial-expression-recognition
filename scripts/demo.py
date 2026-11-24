"""Explicit synthetic wiring demo. No real-data accuracy or quality claim."""

import json
import os
import sys
from pathlib import Path


def main() -> None:
    for key, value in {
        "CUDA_VISIBLE_DEVICES": "-1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }.items():
        os.environ.setdefault(key, value)

    out = Path(sys.argv[1] if len(sys.argv) == 2 else "synthetic-run")
    if len(sys.argv) > 2:
        raise SystemExit("Usage: python scripts/demo.py [NEW_OUTPUT_DIRECTORY]")
    out.mkdir(parents=True, exist_ok=False)
    import tensorflow as tf

    from fer_research.model import evaluate, train

    tf.keras.utils.set_random_seed(42)
    path = out / "fer.csv"
    path.write_text(
        "emotion,pixels,Usage\n"
        + "\n".join(
            f"{i},{' '.join([str(i * 20)] * 2304)},{usage}"
            for i, usage in enumerate(
                ["Training", "Training", "PublicTest", "PrivateTest"]
            )
        )
        + "\n"
    )
    scores = train(path, out / "model", 1, 2)
    test = evaluate(path, out / "model/model.keras")
    assert "test_accuracy" not in scores and test["test_rows"] == 1
    result = {"training": scores, "explicit_synthetic_test": test}
    scope = "invented_constant_pixels_not_faces_not_human_internal_emotions"
    report = {
        "synthetic_only": True,
        "scope": scope,
        "quality_claim": False,
        "result": result,
    }
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"synthetic_only": True, "report": str(out / "report.json")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
