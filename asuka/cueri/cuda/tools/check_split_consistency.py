#!/usr/bin/env python3
"""Verify split CUDA sources are consistent with their monolithic counterparts.

The checker round-trips each configured monolithic source through the existing
splitters in a temporary directory and compares the generated part files
byte-for-byte against the checked-in part files.

Usage:
    python check_split_consistency.py
    python check_split_consistency.py --show-diff
    python check_split_consistency.py --only wave2
"""
from __future__ import annotations

import argparse
import contextlib
import difflib
import io
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import split_large_kernels as _split_large  # noqa: E402
import split_wave_files as _split_wave  # noqa: E402


@dataclass(frozen=True)
class CheckSpec:
    name: str
    kind: str
    src: Path
    parts: int
    split_parts: int | None = None


REPO_ROOT = THIS_DIR.parents[3]
SRC_DIR = REPO_ROOT / "asuka" / "cueri" / "cuda" / "ext" / "src"
GEN_DIR = SRC_DIR / "generated"

DEFAULT_CHECKS: tuple[CheckSpec, ...] = (
    CheckSpec("bindings", "bindings", SRC_DIR / "cueri_cuda_ext.cpp", 9),
    CheckSpec("df_deriv", "large", SRC_DIR / "cueri_cuda_kernels_df_deriv.cu", 17),
    CheckSpec("step2", "large", SRC_DIR / "cueri_cuda_kernels_step2.cu", 4),
    CheckSpec("rys_generic", "large", SRC_DIR / "cueri_cuda_kernels_rys_generic.cu", 1),
    CheckSpec("rys_generic_deriv", "large", SRC_DIR / "cueri_cuda_kernels_rys_generic_deriv.cu", 2),
    CheckSpec("wave1", "wave", GEN_DIR / "cueri_cuda_kernels_wave1_generated.cu", 12),
    CheckSpec("wave2", "wave", GEN_DIR / "cueri_cuda_kernels_wave2_generated.cu", 9, split_parts=12),
)


def _split_to_temp(spec: CheckSpec, tmp_src: Path) -> list[Path]:
    split_parts = int(spec.split_parts or spec.parts)
    with contextlib.redirect_stderr(io.StringIO()):
        if spec.kind == "large":
            return _split_large.split_file(tmp_src, split_parts, no_cmake=True)
        if spec.kind == "wave":
            return _split_wave.split_file(tmp_src, split_parts)
    raise ValueError(f"Unknown check kind: {spec.kind}")


def _print_diff(expected: str, actual: str, *, max_lines: int) -> None:
    diff = difflib.unified_diff(
        expected.splitlines(),
        actual.splitlines(),
        fromfile="generated",
        tofile="checked_in",
        n=2,
    )
    for line in list(diff)[:max_lines]:
        print(line)


def _extract_part_numbers(pattern: str, text: str) -> list[int]:
    return sorted(int(match.group(1)) for match in re.finditer(pattern, text, flags=re.MULTILINE))


def run_bindings_check(spec: CheckSpec) -> bool:
    print(f"[check] {spec.name}: {spec.src}")
    ok = True
    expected = list(range(1, int(spec.parts) + 1))

    entry_text = spec.src.read_text(encoding="utf-8")
    decls = _extract_part_numbers(r"^void cueri_bind_part(\d+)\(py::module_& m\);$", entry_text)
    calls = _extract_part_numbers(r"^\s*cueri_bind_part(\d+)\(m\);$", entry_text)

    if decls == expected:
        print(f"  OK declarations {decls[0]}..{decls[-1]}")
    else:
        ok = False
        print(f"  MISMATCH declarations expected={expected} actual={decls}")

    if calls == expected:
        print(f"  OK module_calls {calls[0]}..{calls[-1]}")
    else:
        ok = False
        print(f"  MISMATCH module_calls expected={expected} actual={calls}")

    part_dir = spec.src.parent
    existing_parts = sorted(
        int(path.stem.removeprefix("cueri_cuda_ext_part"))
        for path in part_dir.glob("cueri_cuda_ext_part*.cpp")
        if path.stem.removeprefix("cueri_cuda_ext_part").isdigit()
    )
    if existing_parts == expected:
        print(f"  OK part_files {existing_parts[0]}..{existing_parts[-1]}")
    else:
        ok = False
        print(f"  MISMATCH part_files expected={expected} actual={existing_parts}")

    for idx in expected:
        part_path = part_dir / f"cueri_cuda_ext_part{idx}.cpp"
        if not part_path.exists():
            ok = False
            print(f"  MISSING {part_path.name}")
            continue
        part_text = part_path.read_text(encoding="utf-8")
        signature = f"void cueri_bind_part{idx}(py::module_& m) {{"
        if signature in part_text:
            print(f"  OK {part_path.name}")
        else:
            ok = False
            print(f"  MISMATCH {part_path.name} missing_signature={signature!r}")

    cmake_path = spec.src.parent.parent / "CMakeLists.txt"
    cmake_text = cmake_path.read_text(encoding="utf-8")
    entry_path = "src/cueri_cuda_ext.cpp"
    if entry_path in cmake_text:
        print("  OK CMake entrypoint")
    else:
        ok = False
        print(f"  MISMATCH CMake missing_entry={entry_path!r}")
    cmake_parts = _extract_part_numbers(r"^\s+src/cueri_cuda_ext_part(\d+)\.cpp$", cmake_text)
    if cmake_parts == expected:
        print(f"  OK CMake part_entries {cmake_parts[0]}..{cmake_parts[-1]}")
    else:
        ok = False
        print(f"  MISMATCH CMake part_entries expected={expected} actual={cmake_parts}")

    return ok


def run_check(spec: CheckSpec, *, show_diff: bool, diff_lines: int) -> bool:
    if spec.kind == "bindings":
        return run_bindings_check(spec)

    print(f"[check] {spec.name}: {spec.src}")
    with TemporaryDirectory() as td:
        tmp_src = Path(td) / spec.src.name
        copy2(spec.src, tmp_src)
        generated_paths = _split_to_temp(spec, tmp_src)
        checked_in_paths = [spec.src.parent / path.name for path in generated_paths]

        ok = True
        for gen_path, checked_path in zip(generated_paths, checked_in_paths):
            generated = gen_path.read_text(encoding="utf-8")
            checked_in = checked_path.read_text(encoding="utf-8")
            if generated == checked_in:
                print(f"  OK {checked_path.name}")
                continue

            ok = False
            print(f"  MISMATCH {checked_path.name}")
            if show_diff:
                _print_diff(generated, checked_in, max_lines=diff_lines)

        expected_parts = spec.parts
        actual_parts = len(generated_paths)
        if actual_parts != expected_parts:
            ok = False
            print(f"  MISMATCH part_count expected={expected_parts} generated={actual_parts}")

        return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Limit checks to named groups: bindings, df_deriv, step2, rys_generic, rys_generic_deriv, wave1, wave2",
    )
    parser.add_argument("--show-diff", action="store_true", help="Show unified diffs for mismatches")
    parser.add_argument(
        "--diff-lines",
        type=int,
        default=80,
        help="Maximum number of diff lines to print per mismatched file",
    )
    args = parser.parse_args(argv)

    selected = set(args.only)
    checks = [spec for spec in DEFAULT_CHECKS if not selected or spec.name in selected]
    if not checks:
        parser.error("No checks selected")

    all_ok = True
    for spec in checks:
        all_ok = run_check(spec, show_diff=bool(args.show_diff), diff_lines=int(args.diff_lines)) and all_ok

    if all_ok:
        print("All split sources are consistent.")
        return 0

    print("Split consistency check failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
