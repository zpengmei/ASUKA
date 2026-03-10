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


def _run_py(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def test_import_asuka_does_not_eager_import_heavy_modules():
    code = (
        "import sys\n"
        "import asuka\n"
        "print(int('asuka.solver' in sys.modules))\n"
        "print(int('asuka.mcscf.casci' in sys.modules))\n"
    )
    proc = _run_py(code)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    assert lines[-2:] == ["0", "0"]


def test_lazy_exports_load_on_demand():
    code = (
        "import sys\n"
        "import asuka\n"
        "_ = asuka.GUGAFCISolver\n"
        "print(int('asuka.solver' in sys.modules))\n"
        "_ = asuka.autotune\n"
        "print(int('asuka.mcscf.casci' in sys.modules))\n"
    )
    proc = _run_py(code)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    assert lines[-2:] == ["1", "1"]
