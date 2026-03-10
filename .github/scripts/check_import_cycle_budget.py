#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def _module_name_from_path(path: Path, package_root: Path) -> str:
    rel = path.relative_to(package_root)
    mod = ".".join(rel.with_suffix("").parts)
    if mod.endswith(".__init__"):
        mod = mod[: -len(".__init__")]
    return f"{package_root.name}.{mod}" if mod else package_root.name


def _package_key(modname: str) -> str:
    parts = modname.split(".")
    if len(parts) < 2:
        return parts[0]
    return ".".join(parts[:2])


def _iter_python_files(package_root: Path) -> Iterable[Path]:
    for path in package_root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def _resolve_relative(base_mod: str, level: int, module: str | None) -> str:
    base_parts = base_mod.split(".")
    if base_mod.endswith(".__init__"):
        pkg_parts = base_parts[:-1]
    else:
        pkg_parts = base_parts[:-1]

    if level <= 0:
        return module or ""

    trim = level - 1
    if trim > 0:
        if trim >= len(pkg_parts):
            pkg_parts = []
        else:
            pkg_parts = pkg_parts[:-trim]

    mod = (module or "").strip(".")
    if mod:
        return ".".join([*pkg_parts, mod])
    return ".".join(pkg_parts)


def _build_package_graph(package_root: Path) -> dict[str, set[str]]:
    edges: dict[str, set[str]] = defaultdict(set)
    for py in _iter_python_files(package_root):
        modname = _module_name_from_path(py, package_root)
        src = _package_key(modname)
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except Exception:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if not (name == package_root.name or name.startswith(f"{package_root.name}.")):
                        continue
                    dst = _package_key(name)
                    if dst != src:
                        edges[src].add(dst)
            elif isinstance(node, ast.ImportFrom):
                resolved = _resolve_relative(modname, int(node.level or 0), node.module)
                if not (resolved == package_root.name or resolved.startswith(f"{package_root.name}.")):
                    continue
                dst = _package_key(resolved)
                if dst != src:
                    edges[src].add(dst)

    return edges


def _tarjan_scc(edges: dict[str, set[str]]) -> list[list[str]]:
    nodes: set[str] = set(edges)
    for dsts in edges.values():
        nodes.update(dsts)

    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    idx: dict[str, int] = {}
    low: dict[str, int] = {}
    out: list[list[str]] = []

    def strongconnect(v: str) -> None:
        nonlocal index
        idx[v] = index
        low[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)

        for w in edges.get(v, ()):
            if w not in idx:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], idx[w])

        if low[v] == idx[v]:
            comp: list[str] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                comp.append(w)
                if w == v:
                    break
            out.append(sorted(comp))

    for node in sorted(nodes):
        if node not in idx:
            strongconnect(node)

    out.sort(key=lambda comp: (-len(comp), comp))
    return out


def _load_baseline(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("baseline must be a JSON object")
    if "core_packages" not in data or "max_core_scc_size" not in data:
        raise ValueError("baseline must contain 'core_packages' and 'max_core_scc_size'")
    return data


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Check package import-cycle budget against a baseline.")
    p.add_argument("--baseline", type=Path, default=Path(".github/import_cycle_budget.json"))
    p.add_argument("--package-root", type=Path, default=Path("asuka"))
    args = p.parse_args(argv)

    baseline = _load_baseline(args.baseline)
    core_packages = {str(x) for x in baseline["core_packages"]}
    max_core_scc_size = int(baseline["max_core_scc_size"])

    edges = _build_package_graph(args.package_root)
    sccs = _tarjan_scc(edges)

    largest = sccs[0] if sccs else []
    print(f"Largest SCC size: {len(largest)}")
    if largest:
        print("Largest SCC members:", ", ".join(largest))

    core_scc_size = 0
    core_scc_members: list[str] = []
    for comp in sccs:
        overlap = core_packages.intersection(comp)
        if len(overlap) > len(core_scc_members):
            core_scc_members = sorted(overlap)
            core_scc_size = len(comp)

    print(f"Core SCC size: {core_scc_size}")
    if core_scc_members:
        print("Core SCC overlap:", ", ".join(core_scc_members))

    if core_scc_size > max_core_scc_size:
        print(
            f"ERROR: core SCC size {core_scc_size} exceeds budget {max_core_scc_size}.",
            file=sys.stderr,
        )
        return 1

    print("Import-cycle budget check passed.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
