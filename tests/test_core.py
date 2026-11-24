import csv
from pathlib import Path

import numpy as np
import pytest

from fer_research.core import LABELS, read


def write(path: Path, rows: list[list[str | int]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["emotion", "pixels", "Usage"])
        writer.writerows(rows)


def valid() -> list[list[str | int]]:
    return [
        [i, " ".join([str(i * 20)] * 2304), usage]
        for i, usage in enumerate(["Training", "Training", "PublicTest", "PrivateTest"])
    ]


def test_read_partitions(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    write(path, valid())
    data = read(path)
    assert list(data) == ["Training", "PublicTest", "PrivateTest"]
    assert [len(x) for x, y in data.values()] == [2, 1, 1]
    x, y = data["Training"]
    assert x.shape == (2, 48, 48, 1)
    assert x.dtype == np.float32 and y.dtype == np.int64
    assert y.tolist() == [0, 1] and x[1].mean() == pytest.approx(20 / 255)
    assert LABELS == ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")


@pytest.mark.parametrize(
    "column,value",
    [
        (0, "bad"),
        (0, 7),
        (0, -1),
        (1, " ".join(["0"] * 2303)),
        (1, " ".join(["256"] * 2304)),
        (1, " ".join(["-1"] * 2304)),
        (1, "x"),
        (2, "Unknown"),
    ],
)
def test_bad_row(tmp_path: Path, column: int, value: str | int) -> None:
    rows = valid()
    rows[0][column] = value
    path = tmp_path / "data.csv"
    write(path, rows)
    with pytest.raises(ValueError):
        read(path)


def test_no_duplicate_leakage(tmp_path: Path) -> None:
    rows = valid()
    rows[2][1] = rows[0][1]
    path = tmp_path / "data.csv"
    write(path, rows)
    with pytest.raises(ValueError, match="Duplicate"):
        read(path)
    write(path, valid()[:2])
    with pytest.raises(ValueError, match="three"):
        read(path)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "pixels,Usage\na,Training\n",
        "emotion,pixels,Usage,emotion\n",
        "emotion,pixels,Usage\n0,pixels\n",
        "emotion,pixels,Usage\n0,pixels,Training,extra\n",
    ],
)
def test_invalid_csv(tmp_path: Path, text: str) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(text)
    with pytest.raises(ValueError):
        read(path)


def test_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read(tmp_path / "missing")
