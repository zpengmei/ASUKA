from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / ".git").exists() and (parent / "asuka").is_dir():
            return parent
    raise RuntimeError("Could not locate repository root from test path")


REPO_ROOT = _repo_root()


def test_import_cycle_budget_check_passes_current_baseline():
    script = REPO_ROOT / ".github" / "scripts" / "check_import_cycle_budget.py"
    baseline = REPO_ROOT / ".github" / "import_cycle_budget.json"
    proc = subprocess.run(
        [sys.executable, str(script), "--baseline", str(baseline), "--package-root", str(REPO_ROOT / "asuka")],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Import-cycle budget check passed." in proc.stdout
