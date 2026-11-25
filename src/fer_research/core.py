"""FER2013 CSV validation without final-test leakage or silent row dropping."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

LABELS = ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")
PARTITIONS = ("Training", "PublicTest", "PrivateTest")
Data = dict[str, tuple[NDArray[np.float32], NDArray[np.int64]]]


def read(path: Path, *, only: str | None = None) -> Data:
    """Require valid2304-pixel rows, labels0..6, explicit Usage partitions.

    Reject exact pixel duplicates anywhere, including across partitions. This
    protects this experiment, not a claim that the source dataset is clean.
    With ``only``, every row is still validated and duplicate-checked, but only
    that partition's arrays are materialized, so final scoring of a large
    corpus does not hold all partitions in memory.
    """
    if only is not None and only not in PARTITIONS:
        raise ValueError(f"Unknown partition {only}.")
    wanted = PARTITIONS if only is None else (only,)
    values: dict[str, list[list[int]]] = {p: [] for p in wanted}
    labels: dict[str, list[int]] = {p: [] for p in wanted}
    seen = set()
    with path.open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream)
        headers = rows.fieldnames or []
        if len(headers) != len(set(headers)) or not {
            "emotion",
            "pixels",
            "Usage",
        } <= set(headers):
            raise ValueError("CSV needs unique emotion,pixels,Usage headers.")
        for i, row in enumerate(rows, 2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"Malformed row{i}.")
            usage = row["Usage"].strip()
            if usage not in PARTITIONS:
                raise ValueError(f"Unknown Usage at row{i}; no random fallback split.")
            try:
                label = int(row["emotion"])
                pixels = [int(x) for x in row["pixels"].split()]
            except ValueError as exc:
                raise ValueError(f"Noninteger label/pixels at row{i}.") from exc
            if (
                not 0 <= label < 7
                or len(pixels) != 2304
                or any(p < 0 or p > 255 for p in pixels)
            ):
                raise ValueError(
                    f"Need label0..6 and2304 integer pixels0..255 at row{i}."
                )
            key = hashlib.sha256(bytes(pixels)).digest()
            if key in seen:
                raise ValueError(
                    f"Duplicate pixels at row{i}; "
                    "review labels/partitions before fitting."
                )
            seen.add(key)
            if usage in values:
                values[usage].append(pixels)
                labels[usage].append(label)
    if any(not values[p] for p in wanted):
        if only is None:
            raise ValueError("All three Usage partitions must be nonempty.")
        raise ValueError(f"Partition {only} must be nonempty.")
    return {
        p: (
            np.asarray(values[p], dtype=np.float32).reshape(-1, 48, 48, 1) / 255,
            np.asarray(labels[p], dtype=np.int64),
        )
        for p in wanted
    }
