from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / ".git").exists() and (parent / "asuka").is_dir():
            return parent
    raise RuntimeError("Could not locate repository root")
