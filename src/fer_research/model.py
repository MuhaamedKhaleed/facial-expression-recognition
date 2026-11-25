"""Original FER CNN, explicit CPU fitting and separate final-test evaluation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .core import LABELS, read


def backend() -> Any:
    """Lazy CPU-only optional TensorFlow backend."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ValueError("Install uv sync --extra ml for TensorFlow.") from exc
    tf.config.set_visible_devices([], "GPU")
    return tf


def build(seed: int = 42) -> Any:
    """Retain original three-convolution48x48 CNN and seven-category head."""
    tf = backend()
    tf.keras.utils.set_random_seed(seed)
    layers = tf.keras.layers
    model = tf.keras.Sequential(
        [
            layers.Input((48, 48, 1)),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(7, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def train(
    data: Path, output: Path, epochs: int = 10, batch_size: int = 32, seed: int = 42
) -> dict[str, float | int]:
    """Fit Training only, inspect PublicTest only, do not score PrivateTest."""
    if output.exists():
        raise ValueError("Output exists, choose a new directory.")
    if epochs < 1 or batch_size < 1:
        raise ValueError("Positive epochs and batch size required.")
    splits = read(data)
    x, y = splits["Training"]
    vx, vy = splits["PublicTest"]
    model = build(seed)
    for _ in range(epochs):
        for start in range(0, len(x), batch_size):
            loss = float(
                model.train_on_batch(
                    x[start : start + batch_size], y[start : start + batch_size]
                )[0]
            )
            if not math.isfinite(loss):
                raise ValueError("Nonfinite training loss; nothing saved.")
    loss, accuracy = map(float, model.evaluate(vx, vy, verbose=0))
    if not math.isfinite(loss) or not math.isfinite(accuracy):
        raise ValueError("Nonfinite validation metrics; nothing saved.")
    result = {
        "validation_loss": loss,
        "validation_accuracy": accuracy,
        "train_rows": len(x),
        "validation_rows": len(vx),
    }
    output.mkdir(parents=True)
    model.save(output / "model.keras")
    (output / "labels.json").write_text(json.dumps(list(LABELS)))
    (output / "validation.json").write_text(json.dumps(result))
    return result


def evaluate(data: Path, artifact: Path) -> dict[str, Any]:
    """Explicitly score PrivateTest, never invoked from training."""
    if artifact.suffix != ".keras" or not artifact.is_file():
        raise ValueError("Supply a trusted local .keras model file.")
    x, y = read(data)["PrivateTest"]
    tf = backend()
    model = tf.keras.models.load_model(artifact, safe_mode=True)
    if model.input_shape[1:] != (48, 48, 1) or model.output_shape[-1] != 7:
        raise ValueError("Model must have48x48x1 input and seven-category output.")
    loss, accuracy = map(float, model.evaluate(x, y, verbose=0))
    prediction = model(x, training=False).numpy()
    if not math.isfinite(loss) or not math.isfinite(accuracy):
        raise ValueError("Nonfinite test metrics.")
    return {
        "test_loss": loss,
        "test_accuracy": accuracy,
        "test_rows": len(x),
        "labels": list(LABELS),
        "predicted_indices": prediction.argmax(axis=1).tolist(),
    }


def predict(image: Path, artifact: Path) -> dict[str, Any]:
    """Classify one explicit face crop, not a whole scene or internal emotion.

    The caller supplies a face crop. No face detection, tracking or camera use.
    Match CSV preprocessing: grayscale, 48x48 and pixel values divided by255.
    """
    import numpy as np
    from PIL import Image

    if artifact.suffix != ".keras" or not artifact.is_file():
        raise ValueError("Supply a trusted local .keras model file.")
    with Image.open(image) as source:
        pixels = (
            np.asarray(
                source.convert("L").resize((48, 48), Image.Resampling.BILINEAR),
                dtype=np.float32,
            ).reshape(1, 48, 48, 1)
            / 255
        )
    labels_path = artifact.parent / "labels.json"
    if not labels_path.is_file():
        raise ValueError(
            "Missing labels.json. Recover training label order; do not guess."
        )
    labels = json.loads(labels_path.read_text())
    if not isinstance(labels, list) or len(labels) != 7 or set(labels) != set(LABELS):
        raise ValueError(
            "labels.json must list the seven training categories exactly once."
        )
    tf = backend()
    model = tf.keras.models.load_model(artifact, safe_mode=True, compile=False)
    if model.input_shape[1:] != (48, 48, 1) or model.output_shape != (None, 7):
        raise ValueError("Model must have48x48x1 input and seven-category output.")
    scores = np.asarray(model(pixels, training=False)).reshape(-1)
    if (
        scores.shape != (7,)
        or not np.isfinite(scores).all()
        or np.any(scores < 0)
        or np.any(scores > 1)
        or not np.isclose(scores.sum(), 1, atol=1e-4)
    ):
        raise ValueError("Model must return seven finite softmax probabilities.")
    return {
        "category": labels[int(scores.argmax())],
        "probabilities": dict(zip(labels, map(float, scores), strict=True)),
        "warning": "Dataset category only, not evidence of internal emotion. "
        "Input must be a face crop; no face detection is performed.",
    }
