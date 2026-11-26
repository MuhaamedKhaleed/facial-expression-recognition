"""Thin FER research CLI, no camera/surveillance/automatic downloads."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer
from dotenv import dotenv_values

from .core import read

app = typer.Typer(
    help="Research-only FER categories, predefined splits, no real emotion inference."
)
logger = logging.getLogger(__name__)


def run(action: Callable[[], Any]) -> None:
    """Print JSON separately from expected input-error logs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        typer.echo(json.dumps(action()))
    except (ValueError, OSError, TypeError) as exc:
        logger.error("%s", exc)
        raise typer.Exit(2) from exc


def resolve(data: Path | None, config: Path | None) -> Path:
    """Read only explicitly selected dotenv FER_CSV, with --data precedence."""
    if data:
        return data
    if config:
        if not config.is_file():
            raise ValueError("Config missing; supply existing dotenv.")
        value = dotenv_values(config).get("FER_CSV")
        if value:
            return Path(value)
    raise ValueError("Supply --data or --config containing FER_CSV.")


@app.command()
def check_data(data: Path | None = None, config: Path | None = None) -> None:
    """Validate every row, predefined Usage and duplicate pixels."""
    run(lambda: {p: len(x) for p, (x, _) in read(resolve(data, config)).items()})


@app.command()
def train(
    data: Path, output: Path, epochs: int = 10, batch_size: int = 32, seed: int = 42
) -> None:
    """Deliberately fit CPU model, never score final PrivateTest."""

    def action() -> dict[str, float | int]:
        from .model import train as fit

        return fit(data, output, epochs, batch_size, seed)

    run(action)


@app.command()
def evaluate(data: Path, model: Path) -> None:
    """Explicit final PrivateTest scoring, do not use repeatedly for tuning."""

    def action() -> dict[str, Any]:
        from .model import evaluate as score

        return score(data, model)

    run(action)


@app.command()
def predict(image: Path, model: Path) -> None:
    """Classify one supplied face crop, not an internal emotion or whole scene."""

    def action() -> dict[str, Any]:
        from .model import predict as classify

        return classify(image, model)

    run(action)
