#!/usr/bin/env python3
"""Split large cuERI CUDA kernel files into parallel-compilable TUs.

Handles files with two structures:

Structure A (e.g. rys_generic.cu):
  <includes>
  namespace {
    <preamble>
    [__global__ kernel [static inline helper ...]]+
  }  // namespace
  [extern "C" launcher]+   ← launchers call __global__ kernels directly

Structure B (e.g. df_deriv.cu):
  <includes>
  namespace {
    <preamble>
    [__global__ kernel]+
    [static inline dispatch_helper]+  ← helpers call kernels, appear AFTER all kernels
  }  // namespace
  [extern "C" launcher]+   ← launchers call dispatch helpers (two-level)

Each split TU contains:
  includes + preamble + kernel group + attached helpers + }  // namespace
  + extern "C" launchers that (transitively) reference those kernels.

Usage:
    python split_large_kernels.py [--parts N] [file.cu ...]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


_RX_LINE_COMMENT = re.compile(r"//[^\n]*")
_RX_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def _strip_comments(body: str) -> str:
    """Remove C/C++ comments to avoid false-positive name matches in comments."""
    body = _RX_BLOCK_COMMENT.sub("", body)
    body = _RX_LINE_COMMENT.sub("", body)
    return body


def _name_in_body(name: str, body: str) -> bool:
    """True iff `name` appears in `body` as a standalone identifier (not as a prefix of a longer name)."""
    return bool(re.search(re.escape(name) + r"(?!\w)", body))


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _find_line(lines: list[str], pattern: str, start: int = 0) -> int:
    rx = re.compile(pattern)
    for i in range(start, len(lines)):
        if rx.search(lines[i]):
            return i
    raise ValueError(f"Pattern {pattern!r} not found after line {start}")


def _find_func_end(lines: list[str], start: int) -> int:
    """Return index one past the closing brace of a function starting at `start`.

    Multi-line function signatures mean the opening ``{`` may appear many lines
    after ``start``.  We wait until we have seen at least one ``{`` before
    treating a return-to-zero depth as end-of-function.
    """
    depth = 0
    seen_open = False
    for i in range(start, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if depth > 0:
            seen_open = True
        if seen_open and depth == 0:
            return i + 1
    raise ValueError(f"No matching closing brace for block starting at line {start}")


def _extract_func_names_in_block(lines_block: list[str]) -> set[str]:
    """Extract all non-kernel function names defined in a block of lines."""
    rx = re.compile(
        r"^(?:static\s+inline|inline|__device__(?:\s+inline)?|template\s*<[^>]*>\s*(?:static\s+inline|inline|__device__))"
        r"\s+\S+\s+(\w+)\s*\("
    )
    result: set[str] = set()
    for line in lines_block:
        m = rx.match(line)
        if m:
            result.add(m.group(1))
    return result


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def _parse_structure(lines: list[str]):
    """Parse a cuERI CUDA kernel file into structural sections.

    Returns
    -------
    header_lines : list[str]
        Lines before ``namespace {``.
    preamble_lines : list[str]
        ``namespace {`` line + shared constants/device helpers up to first kernel.
    kernel_groups : list[tuple[str, list[str]]]
        (kernel_name, all_lines_including_attached_inline_helpers)
        Inline helpers that appear BETWEEN kernels are attached to the preceding kernel.
    dispatch_helpers : list[tuple[str, list[str], set[str]]]
        (func_name, lines, kernels_referenced)
        Static-inline helpers that appear AFTER all kernels (Structure B only).
    extern_blocks : list[tuple[str, list[str], set[str]]]
        (func_name, lines, kernel_names_directly_or_transitively_referenced)
    """
    rx_global = re.compile(r"^__global__\s+\S+\s+(\w+)\s*\(")
    rx_template = re.compile(r"^template\s*<")
    rx_inline = re.compile(r"^(?:static\s+inline|__device__\s+inline|__device__)\s+\S+\s+(\w+)\s*\(")

    ns_start = _find_line(lines, r"^namespace \{")
    ns_end = _find_line(lines, r"^\}  // namespace", start=ns_start)

    header_lines = lines[:ns_start]

    # Locate all __global__ kernels within the namespace.
    # For each, also find the immediately-preceding `template <...>` line (if any).
    # The kernel block starts at the template line so it is self-contained.
    kernel_positions: list[tuple[int, str]] = []  # (block_start, kernel_name)
    for i in range(ns_start, ns_end):
        m = rx_global.match(lines[i])
        if m:
            # Check for a preceding template line (ignore blank lines)
            j = i - 1
            while j >= ns_start and lines[j].strip() == "":
                j -= 1
            block_start = j if (j >= ns_start and rx_template.match(lines[j])) else i
            kernel_positions.append((block_start, m.group(1)))

    if not kernel_positions:
        raise ValueError("No __global__ kernels found in namespace")

    # Preamble = namespace { ... up to (but not including) the first kernel block start.
    first_kernel_block_start = kernel_positions[0][0]
    preamble_lines = lines[ns_start:first_kernel_block_start]

    # Locate inline helper blocks BETWEEN kernels or AFTER all kernels.
    # We scan for `static inline`, `inline void`, etc. in the region of the namespace
    # that is NOT a kernel.
    inline_positions: list[tuple[int, str]] = []  # (line_idx, func_name)
    for i in range(first_kernel_block_start, ns_end):
        m = rx_inline.match(lines[i])
        if m:
            inline_positions.append((i, m.group(1)))
    # Also detect `inline <type> func(` pattern (e.g. `inline void launch_...`)
    rx_plain_inline = re.compile(r"^inline\s+\S+\s+(\w+)\s*\(")
    for i in range(first_kernel_block_start, ns_end):
        m = rx_plain_inline.match(lines[i])
        if m and not rx_global.match(lines[i]):
            if not any(pos == i for pos, _ in inline_positions):
                inline_positions.append((i, m.group(1)))
    inline_positions.sort()

    # Build "checkpoints": sorted list of (line_idx, kind, name) for kernels and inlines
    checkpoints: list[tuple[int, str, str]] = []
    for pos, name in kernel_positions:
        checkpoints.append((pos, "kernel", name))
    for pos, name in inline_positions:
        # Only include if this inline is NOT inside a kernel body
        # (i.e. its position is ≥ kernel block_start but < next kernel block_start)
        # We add it regardless and filter later
        checkpoints.append((pos, "inline", name))
    checkpoints.sort()

    # Assign extents: each checkpoint ends where the next one begins
    checkpoint_extents: list[tuple[int, str, str, int, int]] = []
    for idx, (pos, kind, name) in enumerate(checkpoints):
        end = checkpoints[idx + 1][0] if idx + 1 < len(checkpoints) else ns_end
        checkpoint_extents.append((pos, kind, name, pos, end))

    # Build kernel groups: kernel + inline helpers that are "inside" the kernel
    # (i.e. between this kernel's block_start and the next kernel's block_start).
    # Inline helpers AFTER the last kernel's block_start become dispatch helpers.
    last_kernel_pos = kernel_positions[-1][0]

    kernel_groups: list[tuple[str, list[str]]] = []
    dispatch_helpers: list[tuple[str, list[str], set[str]]] = []
    dispatch_helper_lines: list[list[str]] = []  # raw lines for all dispatch helpers

    # Process checkpoints in order
    current_kernel: str | None = None
    current_kernel_lines: list[str] = []

    for pos, kind, name, start, end in checkpoint_extents:
        if kind == "kernel":
            # Flush previous kernel group
            if current_kernel is not None:
                kernel_groups.append((current_kernel, current_kernel_lines))
            current_kernel = name
            current_kernel_lines = list(lines[start:end])
        elif kind == "inline":
            if current_kernel is not None and pos < last_kernel_pos:
                # Inline between kernels → attach to current kernel group
                # (But only if there's no kernel starting at or after this inline's start
                #  that would make it a dispatch helper; we already sorted so this is correct)
                current_kernel_lines.extend(lines[start:end])
            else:
                # Dispatch helper (after all kernels)
                block_lines = list(lines[start:end])
                dispatch_helper_lines.append(block_lines)

    # Flush last kernel
    if current_kernel is not None:
        kernel_groups.append((current_kernel, current_kernel_lines))

    # Build dispatch helpers with kernel refs
    kernel_names_all = {name for name, _ in kernel_groups}
    for block_lines in dispatch_helper_lines:
        body = "\n".join(block_lines)
        first_line = block_lines[0] if block_lines else ""
        m = rx_inline.match(first_line) or re.match(r"^(?:inline|static)\s+\S+\s+(\w+)\s*\(", first_line)
        fname = m.group(1) if m else ""
        kernel_refs = {kn for kn in kernel_names_all if _name_in_body(kn, body)}
        dispatch_helpers.append((fname, block_lines, kernel_refs))

    # Collect all kernel names
    kernel_names = {name for name, _ in kernel_groups}

    # Build comprehensive callable → kernel_name mapping.
    # Covers:
    #   (a) non-kernel functions attached to a kernel group (inline helpers between kernels)
    #   (b) dispatch helpers after all kernels (Structure B)
    # Key: callable name; Value: set of kernel_names that "own" this callable.
    callable_to_kernel: dict[str, set[str]] = {}

    # (a) Functions defined inside each kernel group block
    for kname, klines in kernel_groups:
        for fname in _extract_func_names_in_block(klines):
            callable_to_kernel.setdefault(fname, set()).add(kname)

    # (b) Dispatch helpers — already parsed, find kernel refs by body scan
    for hname, hlines, hkernels in dispatch_helpers:
        callable_to_kernel.setdefault(hname, set()).update(hkernels)
        # Also scan the dispatch helper body for calls to kernel-group callables
        body = "\n".join(hlines)
        for cname, ckernels in list(callable_to_kernel.items()):
            if cname != hname and _name_in_body(cname, body):
                callable_to_kernel[hname].update(ckernels)

    # Parse everything after namespace: static helpers + extern "C" launchers.
    # Some files have `static` (non-extern "C") dispatch functions between
    # `}  // namespace` and the first `extern "C"` — treat these like dispatch_helpers.
    rx_extern = re.compile(r'^extern\s+"C"\s+\S+\s+(\w+)\s*\(')
    rx_static_fn = re.compile(r'^(?:static|template\s*<[^>]*>\s*static)\s+\S+\s+(\w+)\s*\(')

    post_ns_statics: list[tuple[str, list[str]]] = []  # non-extern-C static helpers
    extern_raw: list[tuple[str, list[str]]] = []  # (func_name, lines)

    i = ns_end + 1
    while i < len(lines):
        m_extern = rx_extern.match(lines[i])
        m_static = rx_static_fn.match(lines[i])
        if m_extern:
            fname = m_extern.group(1)
            end_i = _find_func_end(lines, i)
            extern_raw.append((fname, lines[i:end_i]))
            i = end_i
        elif m_static:
            fname = m_static.group(1)
            end_i = _find_func_end(lines, i)
            post_ns_statics.append((fname, lines[i:end_i]))
            i = end_i
        else:
            i += 1

    # Add post-namespace statics to the callable→kernel mapping
    for sname, slines in post_ns_statics:
        body = "\n".join(slines)
        krefs = {kn for kn in kernel_names if _name_in_body(kn, body)}
        # Also via existing callable_to_kernel
        for cname, ckernels in list(callable_to_kernel.items()):
            if _name_in_body(cname, body):
                krefs.update(ckernels)
        callable_to_kernel.setdefault(sname, set()).update(krefs)

    extern_names = {fname for fname, _ in extern_raw}

    def _resolve_refs(body: str) -> set[str]:
        """Return set of kernel names transitively referenced by a function body."""
        refs: set[str] = set()
        # Direct kernel names in body (word-boundary match)
        refs.update(kn for kn in kernel_names if _name_in_body(kn, body))
        # Via callable-to-kernel mapping
        for cname, ckernels in callable_to_kernel.items():
            if _name_in_body(cname, body):
                refs.update(ckernels)
        return refs

    # First pass: direct + callable-map resolution
    extern_refs: dict[str, set[str]] = {}
    for fname, fblock in extern_raw:
        body = "\n".join(fblock)
        extern_refs[fname] = _resolve_refs(body)

    # Second pass: propagate through extern-to-extern delegation
    changed = True
    while changed:
        changed = False
        for fname, fblock in extern_raw:
            body = "\n".join(fblock)
            for other in extern_names:
                if other != fname and _name_in_body(other, body):
                    new_refs = extern_refs[other] - extern_refs[fname]
                    if new_refs:
                        extern_refs[fname] |= new_refs
                        changed = True

    extern_blocks = [
        (fname, fblock, extern_refs[fname]) for fname, fblock in extern_raw
    ]

    return header_lines, preamble_lines, kernel_groups, dispatch_helpers, post_ns_statics, extern_blocks


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def _balance_groups(sizes: list[int], n_parts: int) -> list[list[int]]:
    n = len(sizes)
    n_parts = min(n_parts, n)
    total = sum(sizes)
    target = total / n_parts

    groups: list[list[int]] = []
    current: list[int] = []
    current_sum = 0
    for i, sz in enumerate(sizes):
        current.append(i)
        current_sum += sz
        if current_sum >= target * (len(groups) + 1) and len(groups) < n_parts - 1:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    while len(groups) > n_parts:
        groups[-2].extend(groups.pop())
    return groups


def split_file(src: Path, n_parts: int, no_cmake: bool = False) -> list[Path]:
    text = src.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=False)

    header_lines, preamble_lines, kernel_groups, dispatch_helpers, post_ns_statics, extern_blocks = (
        _parse_structure(lines)
    )

    n_kernels = len(kernel_groups)
    if n_kernels == 0:
        print(f"  No kernels found in {src.name}, skipping.", file=sys.stderr)
        return []

    n_parts = min(n_parts, n_kernels)
    kernel_sizes = [len(block) for _, block in kernel_groups]
    groups = _balance_groups(kernel_sizes, n_parts)

    # Merge groups that would share the same extern "C" symbol — duplicate extern "C"
    # symbols cause linker errors.  Keep merging until no extern spans multiple groups.
    merged = True
    while merged:
        merged = False
        for _, _, refs in extern_blocks:
            if not refs:
                continue
            # Find which group indices this extern touches
            touching = [
                gi for gi, group in enumerate(groups)
                if refs & {kernel_groups[kidx][0] for kidx in group}
            ]
            if len(touching) > 1:
                # Merge all touching groups into the first one
                base = touching[0]
                for other in reversed(touching[1:]):
                    groups[base].extend(groups.pop(other))
                groups[base].sort()
                merged = True
                break

    if len(groups) < n_parts:
        print(
            f"  Note: extern-merged to {len(groups)} parts (from {n_parts}).",
            file=sys.stderr,
        )

    # Cross-group body reference merge:
    # If a kernel group's block contains a reference to a kernel defined in another
    # group (e.g. via an inline dispatch helper that calls kernels from multiple
    # groups), those groups must be merged to avoid undefined template instantiations.
    body_merged = True
    while body_merged:
        body_merged = False
        for gi in range(len(groups)):
            # Strip comments so kernel names in comments don't trigger false merges
            group_body = _strip_comments("\n".join(
                line for kidx in groups[gi] for line in kernel_groups[kidx][1]
            ))
            for gj in range(len(groups)):
                if gj == gi:
                    continue
                other_kernel_names = {kernel_groups[kidx][0] for kidx in groups[gj]}
                if any(_name_in_body(kn, group_body) for kn in other_kernel_names):
                    # Merge gj into gi — handle index shift from pop
                    base = min(gi, gj)
                    other = max(gi, gj)
                    groups[base].extend(groups.pop(other))
                    groups[base].sort()
                    body_merged = True
                    break
            if body_merged:
                break

    if len(groups) < n_parts:
        print(
            f"  Note: merged to {len(groups)} parts (from {n_parts}) to avoid cross-part kernel references.",
            file=sys.stderr,
        )

    out_paths: list[Path] = []
    stem = src.stem

    # Dispatch helpers: assign each to the group that owns its kernels.
    # Helpers with NO kernel refs (pure utilities) go in every group so that
    # any extern "C" launcher in any part can call them.
    def _dispatch_helpers_for_group(group_kernel_names: set[str]) -> list[list[str]]:
        result = []
        for hname, hlines, hkernels in dispatch_helpers:
            if not hkernels or (hkernels & group_kernel_names):
                result.append(hlines)
        return result

    for part_idx, group in enumerate(groups):
        part_num = part_idx + 1
        out_path = src.parent / f"{stem}_part{part_num}.cu"

        group_kernel_names = {kernel_groups[kidx][0] for kidx in group}

        out_lines: list[str] = []
        first_name = kernel_groups[group[0]][0]
        last_name = kernel_groups[group[-1]][0]
        out_lines.append(
            f"// Auto-split from {src.name} (part {part_num}/{len(groups)}:"
            f" {first_name}..{last_name})"
        )
        out_lines.append("// Do not edit — regenerate with split_large_kernels.py")
        out_lines.append("")

        # Header (includes)
        out_lines.extend(header_lines)

        # Preamble: namespace { + shared constants/helpers
        out_lines.extend(preamble_lines)

        # Kernel blocks (with attached inline helpers)
        for kidx in group:
            _, klines = kernel_groups[kidx]
            out_lines.extend(klines)

        # Dispatch helpers for this group's kernels (Structure B)
        for hlines in _dispatch_helpers_for_group(group_kernel_names):
            out_lines.extend(hlines)

        # Close namespace
        out_lines.append("}  // namespace")
        out_lines.append("")

        # Post-namespace static helpers whose bodies reference this group's kernels
        for _sname, slines in post_ns_statics:
            body = "\n".join(slines)
            if any(_name_in_body(kn, body) for kn in group_kernel_names):
                out_lines.extend(slines)
                out_lines.append("")

        # Extern "C" launchers that (transitively) reference any kernel in this group
        for fname, fblock, refs in extern_blocks:
            if refs & group_kernel_names:
                out_lines.extend(fblock)
                out_lines.append("")

        out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        n_kerns = len(group)
        approx_lines = sum(kernel_sizes[k] for k in group) + len(preamble_lines)
        print(
            f"  Wrote {out_path.name} ({n_kerns} kernel(s), ~{approx_lines} lines)",
            file=sys.stderr,
        )
        out_paths.append(out_path)

    # Sanity check
    all_extern_names = {fname for fname, _, _ in extern_blocks}
    assigned = set()
    for part_idx, group in enumerate(groups):
        group_kernel_names = {kernel_groups[kidx][0] for kidx in group}
        for fname, _, refs in extern_blocks:
            if refs & group_kernel_names:
                assigned.add(fname)
    unassigned = all_extern_names - assigned
    if unassigned:
        print(
            f"  WARNING: {len(unassigned)} extern(s) not assigned: {sorted(unassigned)}",
            file=sys.stderr,
        )
    else:
        print(f"  All {len(all_extern_names)} extern(s) assigned. OK.", file=sys.stderr)

    if not no_cmake:
        _update_cmake(src, out_paths)

    return out_paths


# ---------------------------------------------------------------------------
# CMakeLists.txt updater
# ---------------------------------------------------------------------------

def _update_cmake(original_src: Path, split_paths: list[Path]) -> None:
    for parent in [
        original_src.parent,
        original_src.parent.parent,
        original_src.parent.parent.parent,
    ]:
        candidate = parent / "CMakeLists.txt"
        if not candidate.exists():
            continue
        text = candidate.read_text(encoding="utf-8")
        orig_rel = str(original_src.relative_to(parent)).replace("\\", "/")
        if orig_rel not in text:
            continue
        new_entries = "\n".join(
            f"    {str(p.relative_to(parent)).replace(chr(92), '/')}"
            for p in split_paths
        )
        new_text = text.replace(f"    {orig_rel}", new_entries)
        candidate.write_text(new_text, encoding="utf-8")
        print(f"  Updated {candidate}", file=sys.stderr)
        return
    print(
        f"  WARNING: could not find CMakeLists.txt referencing {original_src.name}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", help="CUDA source files to split")
    ap.add_argument(
        "--parts", type=int, default=4, help="Number of output parts per file (default: 4)"
    )
    ap.add_argument("--no-cmake", action="store_true", help="Skip CMakeLists.txt update")
    args = ap.parse_args(argv)

    this_dir = Path(__file__).resolve().parent
    src_dir = this_dir.parent / "ext" / "src"

    if args.files:
        sources = [Path(f).resolve() for f in args.files]
    else:
        sources = [
            src_dir / "cueri_cuda_kernels_df_deriv.cu",
            src_dir / "cueri_cuda_kernels_rys_generic.cu",
        ]

    for src in sources:
        if not src.exists():
            print(f"  ERROR: {src} not found", file=sys.stderr)
            continue
        print(f"\nSplitting {src.name} into {args.parts} parts ...", file=sys.stderr)
        split_file(src, args.parts, no_cmake=args.no_cmake)


if __name__ == "__main__":
    main()
