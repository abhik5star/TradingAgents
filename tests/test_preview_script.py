"""`scripts/preview.sh`: the recovery path has to be boring and re-runnable.

The dashboard is often the first thing someone runs, and often after a wiped checkout —
so the script that starts it is worth a couple of cheap assertions: it parses, it is
executable, and it starts the demo rather than inventing a second way to launch things.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preview.sh"


def test_script_exists_and_is_executable():
    assert SCRIPT.is_file()
    assert os.access(SCRIPT, os.X_OK), "preview.sh must be runnable as ./scripts/preview.sh"


def test_script_parses():
    bash = shutil_which("bash")
    if bash is None:
        pytest.skip("bash is not available")
    result = subprocess.run([bash, "-n", str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_script_starts_the_demo_and_nothing_of_its_own():
    text = SCRIPT.read_text()
    assert "research demo" in text
    # The demo is the only launcher: a second entry point here would drift from the CLI.
    assert text.count("exec .venv/bin/tradingagents") == 1
    assert "--dir" in text and "--port" in text and "--host" in text
    # It may create the environment, but it must never install into the system Python.
    assert not re.search(r"^\s*(sudo\s+)?(python3?|pip3?)\s+(-m\s+)?pip\s+install", text,
                         re.MULTILINE)
    assert ".venv/bin/pip install" in text


def test_script_defaults_to_the_documented_port():
    text = SCRIPT.read_text()
    assert 'PORT="${PORT:-32909}"' in text
    assert 'HOST="${HOST:-0.0.0.0}"' in text


def test_script_refuses_a_checkout_without_the_research_module():
    text = SCRIPT.read_text()
    assert "tradingagents/research/demo.py" in text
    assert "exit 1" in text


def shutil_which(name: str) -> str | None:
    from shutil import which

    return which(name)
