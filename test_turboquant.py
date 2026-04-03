"""Compatibility entry point for the pytest-based TurboQuant test suite."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    for env_name in (".testenv", ".venv"):
        local_python = repo_root / env_name / "bin" / "python"
        if not local_python.exists():
            continue

        command = [
            str(local_python),
            "-m",
            "pytest",
            "tests/test_core.py",
            "tests/test_stability.py",
            "-v",
        ]
        return subprocess.call(command)

    if importlib.util.find_spec("pytest") is None:
        print(
            "pytest is not installed. Install dependencies, then run "
            "`python3 -m pytest tests/ -v`."
        )
        return 1

    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_core.py",
        "tests/test_stability.py",
        "-v",
    ]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
