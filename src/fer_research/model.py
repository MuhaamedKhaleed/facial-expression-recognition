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
