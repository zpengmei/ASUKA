#!/usr/bin/env python3
"""Split cuERI wave-generated CUDA files into parallel-compilable TUs.

Usage:
    python split_wave_files.py [--parts N] [wave1.cu] [wave2.cu] ...

Each output file is a self-contained .cu that can be compiled independently.
The original file is kept; new split files are written alongside it.
CMakeLists.txt in the parent ext/ directory is updated automatically.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _find_func_end(lines: list[str], start: int) -> int:
    """Return index one past the closing brace of the function starting at `start`."""
    depth = 0
    seen_open = False
    for i in range(start, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if depth > 0:
            seen_open = True
        if seen_open and depth == 0:
            return i + 1
    raise ValueError(f"No matching closing brace for block starting at line {start}")


def _kernel_trailing_gap(block_lines: list[str]) -> list[str]:
    """Return the lines AFTER the __global__ body in a kernel block.

    These are device helper functions that appear after the LAST __global__ body
    in the block but before the next kernel's template line. They belong to the
    current block but may be needed by the following part.
    """
    rx_global = re.compile(r"^__global__")
    body_end = None
    for i, line in enumerate(block_lines):
        if rx_global.match(line):
            body_end = _find_func_end(block_lines, i)
    if body_end is not None:
        next_kernel = len(block_lines)
        for j in range(body_end, len(block_lines)):
            stripped = block_lines[j].strip()
            if stripped == "template <int NROOTS>" or rx_global.match(block_lines[j]):
                next_kernel = j
                break
        return block_lines[body_end:next_kernel]
    return []


def _find_line(lines: list[str], pattern: str) -> int:
    """Return first 0-based index matching regex pattern."""
    rx = re.compile(pattern)
    for i, line in enumerate(lines):
        if rx.search(line):
            return i
    raise ValueError(f"Pattern {pattern!r} not found in file")


def _find_line_exact(lines: list[str], text: str, start: int = 0) -> int:
    for i in range(start, len(lines)):
        if lines[i].rstrip() == text:
            return i
    raise ValueError(f"Exact line {text!r} not found after line {start}")


def _kernel_names_in_order(lines: list[str], ns_start: int, ns_end: int) -> list[str]:
    """Return kernel names in source order (e.g. 'psds', 'ppds', ...)."""
    rx = re.compile(r"^__global__ void KernelERI_(\w+?)_flat\s*\(")
    names: list[str] = []
    for line in lines[ns_start:ns_end]:
        m = rx.match(line)
        if m:
            names.append(m.group(1))
    return names


def _kernel_block_bounds(
    lines: list[str], ns_start: int, ns_end: int, kernel_names: list[str]
) -> list[tuple[int, int]]:
    """Return [(start, end)] line indices (inclusive start, exclusive end) for
    each kernel block within the namespace.  The first block starts at the end
    of the shared helpers; subsequent blocks start where the previous ends.
    shared helpers end just before the first eval_* function of kernel 0.
    """
    # Find where the shared helpers end: right after the closing } of
    # compute_G_stride_fixed (the only template helper in the shared area).
    # Strategy: find each kernel's first device helper (eval_<name>_x).
    rx_global = re.compile(r"^__global__ void KernelERI_(\w+?)_flat\s*\(")

    # Collect the line index of each __global__ marker (absolute in file)
    global_lines: dict[str, int] = {}
    for i in range(ns_start, ns_end):
        m = rx_global.match(lines[i])
        if m:
            name = m.group(1)
            if name not in global_lines:
                global_lines[name] = i

    # For each kernel the "template <int NROOTS>" line comes just before __global__
    # so the kernel block (including its template line) starts at i-1.
    kernel_template_lines: list[int] = []
    for name in kernel_names:
        gl = global_lines[name]
        # The template line is just above the __global__ line
        assert lines[gl - 1].strip() == "template <int NROOTS>", (
            f"Expected 'template <int NROOTS>' before __global__ for kernel {name}, "
            f"got: {lines[gl-1]!r}"
        )
        kernel_template_lines.append(gl - 1)

    # Now find where the shared helpers end = first kernel's template line
    shared_end = kernel_template_lines[0]  # exclusive

    # Kernel block boundaries: each block starts at template line, ends at start of next
    bounds: list[tuple[int, int]] = []
    for idx, start in enumerate(kernel_template_lines):
        if idx + 1 < len(kernel_template_lines):
            end = kernel_template_lines[idx + 1]
        else:
            end = ns_end  # up to (but not including) '}  // namespace'
        bounds.append((start, end))

    return bounds, shared_end


def _extern_blocks(
    lines: list[str], ns_end: int, kernel_names: list[str]
) -> dict[str, list[str]]:
    """Return a dict mapping kernel name -> list of extern "C" launcher lines.

    Each kernel has 3 launchers: _launch_stream, _warp_launch_stream,
    _multiblock_launch_stream.  They appear in the same order as the kernels,
    grouped in blocks separated by blank lines.
    """
    # Match any extern "C" cueri_eri_* function opener
    rx_start = re.compile(r'^extern "C" cudaError_t (cueri_eri_\w+)\(')

    def _kernel_for_func(func_name: str) -> str | None:
        """Map e.g. 'cueri_eri_psds_warp_launch_stream' -> 'psds'."""
        # Strip 'cueri_eri_' prefix
        suffix = func_name[len("cueri_eri_"):]  # e.g. 'psds_warp_launch_stream'
        # Find the longest kernel name that is a prefix of suffix
        best: str | None = None
        for name in kernel_names:
            if suffix.startswith(name) and (best is None or len(name) > len(best)):
                best = name
        return best

    result: dict[str, list[str]] = {name: [] for name in kernel_names}

    current_kernel: str | None = None
    current_block: list[str] = []
    all_blocks: list[tuple[str, list[str]]] = []

    for line in lines[ns_end + 1:]:
        m = rx_start.match(line)
        if m:
            # flush previous block
            if current_kernel is not None:
                all_blocks.append((current_kernel, current_block))
            kern = _kernel_for_func(m.group(1))
            current_kernel = kern
            current_block = [line]
        elif current_kernel is not None:
            current_block.append(line)

    if current_kernel is not None:
        all_blocks.append((current_kernel, current_block))

    for kern, block in all_blocks:
        if kern in result:
            result[kern].extend(block)

    return result


# ---------------------------------------------------------------------------
# Main split logic
# ---------------------------------------------------------------------------

def split_file(src: Path, n_parts: int) -> list[Path]:
    text = src.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=False)

    # Locate structural landmarks
    ns_start = _find_line(lines, r"^namespace \{")
    ns_end = _find_line(lines, r"^\}  // namespace")

    # Includes = everything before the namespace
    includes: list[str] = lines[:ns_start]  # does NOT include 'namespace {'

    # Shared helpers inside namespace (after 'namespace {', before first kernel)
    kernel_names = _kernel_names_in_order(lines, ns_start, ns_end)
    if not kernel_names:
        print(f"  No kernels found in {src.name}, skipping.", file=sys.stderr)
        return []

    bounds, shared_end = _kernel_block_bounds(lines, ns_start, ns_end, kernel_names)
    # Shared section: lines[ns_start .. shared_end)  (includes 'namespace {' line)
    shared_lines: list[str] = lines[ns_start:shared_end]

    # Extern "C" launcher blocks per kernel
    extern_by_kernel = _extern_blocks(lines, ns_end, kernel_names)

    # Balance kernels across parts: greedy bin-packing by line count.
    # Keep exactly `n_parts` non-empty groups when possible so the requested
    # split count stays reproducible even if the last few kernels are tiny.
    n_kernels = len(kernel_names)
    kernel_sizes = [(bounds[i][1] - bounds[i][0]) for i in range(n_kernels)]
    n_parts = min(n_parts, n_kernels)

    # Simple balanced split: divide by target line count, but force a cut when
    # each remaining group must receive exactly one remaining kernel.
    total_lines = sum(kernel_sizes)
    target = total_lines / n_parts

    groups: list[list[int]] = []  # list of kernel indices per part
    current_group: list[int] = []
    current_sum = 0
    for i, sz in enumerate(kernel_sizes):
        current_group.append(i)
        current_sum += sz
        remaining_items = n_kernels - (i + 1)
        remaining_groups = n_parts - len(groups) - 1
        # Start a new group when we hit the target, unless it's the last
        # group. Also force a cut when we must leave one kernel per remaining
        # group to avoid silently collapsing the requested split count.
        if len(groups) < n_parts - 1 and (
            current_sum >= target * (len(groups) + 1) or remaining_items == remaining_groups
        ):
            groups.append(current_group)
            current_group = []
    if current_group:
        groups.append(current_group)
    # If we have too many groups (due to rounding), merge trailing ones
    while len(groups) > n_parts:
        groups[-2].extend(groups[-1])
        groups.pop()

    # Write split files
    out_paths: list[Path] = []
    stem = src.stem  # e.g. 'cueri_cuda_kernels_wave1_generated'
    for part_idx, group in enumerate(groups):
        part_num = part_idx + 1
        out_path = src.parent / f"{stem}_part{part_num}.cu"

        # Build file content
        out_lines: list[str] = []

        # File header comment
        first_name = kernel_names[group[0]]
        last_name = kernel_names[group[-1]]
        out_lines.append(
            f"// Auto-split from {src.name} (part {part_num}/{len(groups)}:"
            f" {first_name}..{last_name})"
        )
        out_lines.append(f"// Do not edit — regenerate with split_wave_files.py")
        out_lines.append("")

        # Includes + namespace { + shared helpers
        out_lines.extend(includes)
        out_lines.extend(shared_lines)

        # Bridge code: trailing device helpers from the previous part's last kernel block.
        # These helper functions appear after the __global__ body of the last kernel in
        # the previous part, and are needed by the first kernel in this part (same angular
        # class).  Since they are __device__ __forceinline__ in an anonymous namespace
        # (internal linkage), duplicating them is safe (no ODR violation).
        if part_idx > 0:
            prev_last_kidx = groups[part_idx - 1][-1]
            prev_start, prev_end = bounds[prev_last_kidx]
            prev_block = lines[prev_start:prev_end]
            bridge = _kernel_trailing_gap(prev_block)
            if bridge:
                out_lines.append("// Bridge: device helpers from previous part (needed here).")
                out_lines.extend(bridge)

        # Kernel blocks for this part (still inside the namespace)
        for kidx in group:
            start, end = bounds[kidx]
            out_lines.extend(lines[start:end])

        # Close the anonymous namespace
        out_lines.append("}  // namespace")
        out_lines.append("")

        # Extern "C" launchers for this part's kernels
        for kidx in group:
            name = kernel_names[kidx]
            out_lines.extend(extern_by_kernel[name])

        out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        n_kerns = len(group)
        total_written = sum(kernel_sizes[k] for k in group)
        print(
            f"  Wrote {out_path.name}"
            f" ({n_kerns} kernels, ~{total_written + len(shared_lines)} lines)",
            file=sys.stderr,
        )
        out_paths.append(out_path)

    return out_paths


# ---------------------------------------------------------------------------
# CMakeLists.txt updater
# ---------------------------------------------------------------------------

def _cmake_list_path(src: Path) -> Path:
    """Walk up from src to find CMakeLists.txt containing the source file reference."""
    for parent in [src.parent, src.parent.parent, src.parent.parent.parent]:
        candidate = parent / "CMakeLists.txt"
        if candidate.exists():
            rel = src.relative_to(parent)
            if str(rel).replace("\\", "/") in candidate.read_text():
                return candidate
    raise FileNotFoundError(f"Could not find CMakeLists.txt referencing {src}")


def update_cmake(cmake_path: Path, original_src: Path, split_paths: list[Path]) -> None:
    """Replace the original source entry in CMakeLists.txt with split files."""
    text = cmake_path.read_text(encoding="utf-8")
    cmake_dir = cmake_path.parent

    # Build the original relative path as it appears in CMakeLists
    orig_rel = str(original_src.relative_to(cmake_dir)).replace("\\", "/")

    # Build replacement lines
    new_entries = "\n".join(
        f"    {str(p.relative_to(cmake_dir)).replace(chr(92), '/')}"
        for p in split_paths
    )

    if orig_rel not in text:
        print(
            f"  WARNING: {orig_rel!r} not found in {cmake_path}; CMakeLists not updated.",
            file=sys.stderr,
        )
        return

    new_text = text.replace(f"    {orig_rel}", new_entries)
    cmake_path.write_text(new_text, encoding="utf-8")
    print(f"  Updated {cmake_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", help="Wave .cu files to split (default: both wave files)")
    ap.add_argument("--parts", type=int, default=6, help="Number of output parts per file (default: 6)")
    ap.add_argument("--no-cmake", action="store_true", help="Skip CMakeLists.txt update")
    args = ap.parse_args(argv)

    this_dir = Path(__file__).resolve().parent
    gen_dir = this_dir.parent / "ext" / "src" / "generated"

    if args.files:
        sources = [Path(f).resolve() for f in args.files]
    else:
        sources = sorted(gen_dir.glob("cueri_cuda_kernels_wave*_generated.cu"))

    if not sources:
        ap.error("No wave files found. Pass explicit paths or run from the tools/ directory.")

    for src in sources:
        print(f"\nSplitting {src.name} into {args.parts} parts ...", file=sys.stderr)
        split_paths = split_file(src, args.parts)
        if not split_paths:
            continue
        if not args.no_cmake:
            try:
                cmake = _cmake_list_path(src)
                update_cmake(cmake, src, split_paths)
            except FileNotFoundError as e:
                print(f"  WARNING: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
