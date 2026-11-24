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
