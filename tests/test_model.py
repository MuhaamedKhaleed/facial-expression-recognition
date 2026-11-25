import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from fer_research.cli import app
from fer_research.core import read
from fer_research.model import build, evaluate, train

tf = pytest.importorskip("tensorflow")


def data_file(tmp_path: Path) -> Path:
    path = tmp_path / "data.csv"
    path.write_text(
        "emotion,pixels,Usage\n"
        + "\n".join(
            f"{i},{' '.join([str(i * 20)] * 2304)},{usage}"
            for i, usage in enumerate(
                ["Training", "Training", "PublicTest", "PrivateTest"]
            )
        )
    )
    return path


def test_actual_cnn_train_reload_and_explicit_test(tmp_path: Path) -> None:
    data = data_file(tmp_path)
    splits = read(data)
    model = build()
    assert model(splits["Training"][0]).shape == (2, 7)
    scores = train(data, tmp_path / "model", 1, 2)
    assert scores["train_rows"] == 2 and scores["validation_rows"] == 1
    assert "test_accuracy" not in scores and np.isfinite(scores["validation_loss"])
    assert json.loads((tmp_path / "model" / "validation.json").read_text()) == scores
    result = evaluate(data, tmp_path / "model" / "model.keras")
    assert result["test_rows"] == 1 and 0 <= result["test_accuracy"] <= 1
    assert len(result["predicted_indices"]) == 1
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "train",
            str(data),
            str(tmp_path / "cli"),
            "--epochs",
            "1",
            "--batch-size",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "test_accuracy" not in json.loads(result.output)
    result = runner.invoke(
        app, ["evaluate", str(data), str(tmp_path / "cli" / "model.keras")]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["test_rows"] == 1
    with pytest.raises(ValueError):
        train(data, tmp_path / "model")
    with pytest.raises(ValueError):
        train(data, tmp_path / "new", 0)
    with pytest.raises(ValueError):
        evaluate(data, tmp_path / "missing.keras")


def test_training_never_consumes_test_labels(tmp_path: Path) -> None:
    # PrivateTest stays structurally checked, never passed to fitting/validation.
    from unittest.mock import patch

    data = data_file(tmp_path)
    splits = read(data)
    captured = []

    class Model:
        def train_on_batch(self, x, y):
            captured.extend(y.tolist())
            return [0.5, 0.5]

        def evaluate(self, x, y, verbose=0):
            assert y.tolist() == [2]
            return [0.4, 1.0]

        def save(self, path):
            Path(path).write_text("mock artifact")

    with patch("fer_research.model.build", return_value=Model()):
        train(data, tmp_path / "capture", 1, 2)
    assert captured == splits["Training"][1].tolist() == [0, 1]


def test_predict_crop_reload_and_cli(tmp_path: Path) -> None:
    from PIL import Image

    from fer_research.model import predict

    artifact = tmp_path / "model.keras"
    build().save(artifact)
    (tmp_path / "labels.json").write_text(
        json.dumps(
            list(reversed(__import__("fer_research.core", fromlist=["LABELS"]).LABELS))
        )
    )
    image = tmp_path / "crop.png"
    Image.new("RGB", (67, 59), (120, 90, 60)).save(image)
    result = predict(image, artifact)
    assert len(result["probabilities"]) == 7
    assert sum(result["probabilities"].values()) == pytest.approx(1)
    assert result["category"] in result["probabilities"]
    assert "not evidence" in result["warning"]
    response = CliRunner().invoke(app, ["predict", str(image), str(artifact)])
    assert response.exit_code == 0, response.output
    assert json.loads(response.output) == result
    with pytest.raises(ValueError):
        predict(image, tmp_path / "missing.keras")
    bad = tf.keras.Sequential(
        [
            tf.keras.layers.Input((48, 48, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(6),
        ]
    )
    bad.save(tmp_path / "bad.keras")
    with pytest.raises(ValueError):
        predict(image, tmp_path / "bad.keras")
