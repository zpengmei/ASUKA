#!/usr/bin/env python3
"""Fail when a CuPy-using test file lacks an explicit pytest.mark.cuda marker."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = ROOT / "tests"

# Conservative CuPy usage detection for test files.
_CUPY_USAGE_PATTERNS = (
    re.compile(r"\bimport\s+cupy\b"),
    re.compile(r"\bimport\s+cupy\s+as\b"),
    re.compile(r"pytest\.importorskip\(\s*[\"']cupy[\"']"),
    re.compile(r"\bcupy\s*=\s*pytest\.importorskip\("),
)

# Any explicit use of pytest.mark.cuda in the file is acceptable.
_CUDA_MARKER_PATTERN = re.compile(r"pytest\.mark\.cuda")


def _uses_cupy(text: str) -> bool:
    return any(p.search(text) for p in _CUPY_USAGE_PATTERNS)


def main() -> int:
    offenders: list[Path] = []
    for test_file in sorted(TESTS_ROOT.rglob("test_*.py")):
        text = test_file.read_text(encoding="utf-8")
        if not _uses_cupy(text):
            continue
        if not _CUDA_MARKER_PATTERN.search(text):
            offenders.append(test_file)

    if offenders:
        print("Found CuPy-using tests without pytest.mark.cuda:")
        for path in offenders:
            print(f" - {path.relative_to(ROOT)}")
        return 1

    print("All CuPy-using tests have explicit pytest.mark.cuda coverage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
