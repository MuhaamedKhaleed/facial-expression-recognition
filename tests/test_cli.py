import json
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from fer_research.cli import app

runner = CliRunner()


def test_help_lazy() -> None:
    code = "import sys;import fer_research.cli;assert 'tensorflow' not in sys.modules"
    assert (
        subprocess.run([sys.executable, "-c", code], capture_output=True).returncode
        == 0
    )
    for args in [[], ["train"], ["evaluate"]]:
        assert runner.invoke(app, [*args, "--help"]).exit_code == 0


def test_data_and_config(tmp_path: Path) -> None:
    data = tmp_path / "fer.csv"
    data.write_text(
        "emotion,pixels,Usage\n"
        + "\n".join(
            f"{i},{' '.join([str(i)] * 2304)},{usage}"
            for i, usage in enumerate(["Training", "PublicTest", "PrivateTest"])
        )
    )
    config = tmp_path / ".env"
    config.write_text(f"FER_CSV={data}\n")
    result = runner.invoke(app, ["check-data", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "Training": 1,
        "PublicTest": 1,
        "PrivateTest": 1,
    }
    config.write_text("OTHER=none\n")
    assert (
        runner.invoke(
            app, ["check-data", "--data", str(data), "--config", str(config)]
        ).exit_code
        == 0
    )
    for args in [
        [],
        ["--config", str(config)],
        ["--config", str(tmp_path / "missing")],
        ["--data", str(tmp_path / "missing")],
    ]:
        result = runner.invoke(app, ["check-data", *args])
        assert result.exit_code == 2
        assert "Traceback" not in result.output
