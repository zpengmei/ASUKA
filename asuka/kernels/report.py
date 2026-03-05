from __future__ import annotations

"""Aggregated kernel report.

The intent is to provide a single "what native kernels exist?" view without
forcing import-time failures when optional CUDA extensions are absent.
"""

from datetime import datetime, timezone
import json
import platform
import sys
from typing import Any, TextIO


def kernel_report() -> dict[str, Any]:
    """Return a JSON-serializable report of native extension availability."""

    # Local imports to keep the module lightweight at import time.
    from . import cueri, guga, hf_df_jk, orbitals

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "extensions": {
            "cueri": cueri.probe(),
            "hf_df_jk": hf_df_jk.probe(),
            "orbitals": orbitals.probe(),
            "guga": guga.probe(),
        },
    }


def _summarize_ext(ext: dict[str, Any]) -> tuple[int, int]:
    """Return (n_available, n_total) for a probe() dict."""

    syms = ext.get("symbols", {})
    if not isinstance(syms, dict):
        return 0, 0
    n_total = len(syms)
    n_ok = int(sum(1 for v in syms.values() if bool(v)))
    return n_ok, n_total


def print_kernel_report(
    *,
    full: bool = False,
    json_output: bool = False,
    file: TextIO | None = None,
) -> None:
    """Print a human-readable kernel report (or JSON if requested)."""

    out = sys.stdout if file is None else file
    rep = kernel_report()
    if json_output:
        print(json.dumps(rep, indent=2, sort_keys=True), file=out)
        return

    print("ASUKA native kernel report", file=out)
    print(f"- timestamp_utc: {rep.get('timestamp_utc')}", file=out)
    print(f"- python: {rep.get('python')}", file=out)
    print(f"- platform: {rep.get('platform')}", file=out)

    exts = rep.get("extensions", {})
    if not isinstance(exts, dict):
        return

    def _print_one(name: str, ext: dict[str, Any]) -> None:
        present = bool(ext.get("present", False))
        err = ext.get("import_error", None)
        n_ok, n_total = _summarize_ext(ext)
        status = "OK" if present else "MISSING"
        suffix = f" ({n_ok}/{n_total} symbols)" if n_total else ""
        if present and err:
            # Present + error should not happen, but show it if it does.
            status = "PARTIAL"
        if err and not present:
            suffix = f" ({err})"
        print(f"- {name}: {status}{suffix}", file=out)

        if not full:
            return

        cats = ext.get("categories", {})
        syms = ext.get("symbols", {})
        if isinstance(cats, dict) and isinstance(syms, dict) and cats:
            for cat, names in sorted(cats.items(), key=lambda kv: str(kv[0])):
                if not isinstance(names, list):
                    continue
                ok = sum(1 for s in names if bool(syms.get(s, False)))
                tot = len(names)
                print(f"  - {cat}: {ok}/{tot}", file=out)
                for s in names:
                    ok_s = bool(syms.get(s, False))
                    mark = "yes" if ok_s else "no"
                    print(f"    - {s}: {mark}", file=out)
            return

        if isinstance(syms, dict) and syms:
            for s, ok_s in sorted(syms.items(), key=lambda kv: str(kv[0])):
                mark = "yes" if bool(ok_s) else "no"
                print(f"  - {s}: {mark}", file=out)

    # Top-level extensions.
    for key in ("cueri", "hf_df_jk", "orbitals"):
        ext = exts.get(key, {})
        if isinstance(ext, dict):
            _print_one(key, ext)

    # cuGUGA has a nested structure: core + linalg.
    g = exts.get("guga", {})
    if isinstance(g, dict):
        core = g.get("core", {})
        if isinstance(core, dict):
            _print_one("guga.core", core)
        linalg = g.get("linalg", {})
        if isinstance(linalg, dict):
            _print_one("guga.linalg", linalg)


__all__ = [
    "kernel_report",
    "print_kernel_report",
]

