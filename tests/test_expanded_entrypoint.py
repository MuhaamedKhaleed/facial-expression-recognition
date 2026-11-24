import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("filename", ["expanded_cpu.py", "compare_fixed_cpu.py"])
def test_import_and_help(filename):
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location("fer_optional", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
    result = subprocess.run(
        [sys.executable, str(path), "--help"], text=True, capture_output=True
    )
    assert result.returncode == 0
    assert "--output" in result.stdout
