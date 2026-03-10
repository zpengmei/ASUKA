from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / ".git").exists() and (parent / "asuka").is_dir():
            return parent
    raise RuntimeError("Could not locate repository root from test path")


REPO_ROOT = _repo_root()


@pytest.mark.parametrize(
    ("module", "args", "needle"),
    [
        ("asuka.cli.doctor", [], "ASUKA environment check"),
        ("asuka.cli.kernels", ["--help"], "Print a report"),
        ("asuka.cli.cuda_audit", ["--help"], "usage"),
    ],
)
def test_cli_module_smoke(module: str, args: list[str], needle: str):
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(REPO_ROOT))
    proc = subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    combined = (proc.stdout + proc.stderr).lower()
    assert str(needle).lower() in combined
