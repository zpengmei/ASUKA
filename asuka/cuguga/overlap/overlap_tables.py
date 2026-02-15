from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# =============================================================================
# Helper functions and expression parsing for Dobrautz thesis overlap tables
# (Tables A.4 / A.5).
#
# This module provides:
# - A,C,B,D helpers (Eq. 3.84 and Eq. A.37)
# - parsing of the table mini-language (tA(...), ±/∓, √2, μ, etc.)
# - a small lookup container for W^x(Q; d', d, Δb, b)
# =============================================================================

sqrt2 = math.sqrt(2.0)
t = 1.0 / sqrt2


def A(b: int | float, x: int | float, y: int | float) -> float:
    """A(b,x,y) = sqrt((b+x)/(b+y)) (thesis Eq. 3.84)."""

    num = float(b) + float(x)
    den = float(b) + float(y)
    if den == 0.0:
        raise ZeroDivisionError(f"A: b+y=0 for b={b}, y={y}")
    arg = num / den
    if arg < 0.0:
        raise ValueError(f"A: negative sqrt argument for b={b}, x={x}, y={y}")
    return math.sqrt(arg)


def C(b: int | float, x: int | float) -> float:
    """C(b,x) = sqrt((b+x-1)(b+x+1)) / (b+x) (thesis Eq. 3.84)."""

    den = float(b) + float(x)
    if den == 0.0:
        raise ZeroDivisionError(f"C: b+x=0 for b={b}, x={x}")
    arg = (float(b) + float(x) - 1.0) * (float(b) + float(x) + 1.0)
    if arg < 0.0:
        raise ValueError(f"C: negative sqrt argument for b={b}, x={x}")
    return math.sqrt(arg) / den


def B(b: int | float, p: int | float, q: int | float) -> float:
    """B(p,q) = sqrt(2 / ((b+p)(b+q))) (thesis Eq. A.37)."""

    den = (float(b) + float(p)) * (float(b) + float(q))
    if den == 0.0:
        raise ZeroDivisionError(f"B: denominator 0 for b={b}, p={p}, q={q}")
    arg = 2.0 / den
    if arg < 0.0:
        raise ValueError(f"B: negative sqrt argument for b={b}, p={p}, q={q}")
    return math.sqrt(arg)


def D(b: int | float, p: int | float) -> float:
    """D(p) = sqrt((b+p-1)(b+p+2)/((b+p)(b+p+1))) (thesis Eq. A.37)."""

    den = (float(b) + float(p)) * (float(b) + float(p) + 1.0)
    if den == 0.0:
        raise ZeroDivisionError(f"D: denominator 0 for b={b}, p={p}")
    arg = (float(b) + float(p) - 1.0) * (float(b) + float(p) + 2.0) / den
    if arg < 0.0:
        raise ValueError(f"D: negative sqrt argument for b={b}, p={p}")
    return math.sqrt(arg)


_re_A = re.compile(r"A\(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)")
_re_C = re.compile(r"C\(\s*([+-]?\d+)\s*\)")
_re_B = re.compile(r"B\(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)")
_re_D = re.compile(r"D\(\s*([+-]?\d+)\s*\)")


def _normalize_expr(expr: str) -> str:
    s = expr.strip()
    s = s.replace("−", "-")
    s = s.replace("µ", "mu").replace("μ", "mu")
    s = s.replace("√2", "sqrt2")
    s = s.replace("½", "0.5")
    s = re.sub(r"\s+", "", s)

    # Implicit multiplication in the table notation.
    s = s.replace("tA", "t*A")
    s = re.sub(r"\bt\(", "t*(", s)
    return s


def _inject_b(s: str) -> str:
    # Convert helper calls with implicit b dependence:
    #   A(x,y) -> A(b,x,y), etc.
    s = _re_A.sub(r"A(b,\1,\2)", s)
    s = _re_C.sub(r"C(b,\1)", s)
    s = _re_B.sub(r"B(b,\1,\2)", s)
    s = _re_D.sub(r"D(b,\1)", s)
    return s


@dataclass(frozen=True)
class TableExpr:
    """Parsed expression for one overlap-table cell.

    Some entries have leading ±/∓, whose sign depends on whether the segment type uses
    the "upper" or "lower" sign convention (Dobrautz Table A.4 starred segment types).

    sign_mode:
      0  : no special sign
      +1 : leading '±'  => factor = sign
      -1 : leading '∓'  => factor = -sign
    """

    py_expr: str
    sign_mode: int = 0
    raw: str = ""

    def eval(self, *, b: int, mu: float = 0.0, sign: int = +1) -> float:
        if self.sign_mode == +1:
            factor = float(sign)
        elif self.sign_mode == -1:
            factor = float(-sign)
        else:
            factor = 1.0

        env = {
            "b": float(b),
            "mu": float(mu),
            "t": float(t),
            "sqrt2": float(sqrt2),
            "A": A,
            "C": C,
            "B": B,
            "D": D,
            "math": math,
        }
        val = eval(self.py_expr, {"__builtins__": {}}, env)
        return factor * float(val)


def parse_table_expr(expr: str | None) -> TableExpr | None:
    """Parse a single cell expression.

    Returns None for '-' (meaning "no entry").
    """

    if expr is None:
        return None
    if expr.strip() in ("", "-"):
        return None

    s = _normalize_expr(expr)
    sign_mode = 0
    if s.startswith("±"):
        sign_mode = +1
        s = s[1:]
    elif s.startswith("∓"):
        sign_mode = -1
        s = s[1:]

    s = _inject_b(s)
    return TableExpr(py_expr=s, sign_mode=sign_mode, raw=expr)


OverlapKey = tuple[str, int, Optional[int], int, int]  # (seg_code, delta_b, x, dprime, d)


@dataclass
class OverlapTables:
    """Storage for overlap-range two-body segment values W^x(...) from Tables A.4/A.5."""

    _data: dict[OverlapKey, TableExpr] | None = None

    def __post_init__(self) -> None:
        if self._data is None:
            self._data = {}

    @staticmethod
    def _sign_from_seg_code(seg_code: str) -> int:
        # Convention: seg codes ending with '*' use "lower sign" (Table A.4 caption).
        return -1 if seg_code.endswith("*") else +1

    def set(
        self,
        seg_code: str,
        delta_b: int,
        x: Optional[int],
        dprime: int,
        d: int,
        expr: str,
    ) -> None:
        te = parse_table_expr(expr)
        if te is None:
            return
        key = (str(seg_code), int(delta_b), None if x is None else int(x), int(dprime), int(d))
        self._data[key] = te

    def get_expr(
        self, seg_code: str, delta_b: int, x: int, dprime: int, d: int
    ) -> TableExpr | None:
        key = (str(seg_code), int(delta_b), int(x), int(dprime), int(d))
        te = self._data.get(key)
        if te is not None:
            return te
        key_any = (str(seg_code), int(delta_b), None, int(dprime), int(d))
        return self._data.get(key_any)

    def value(
        self,
        seg_code: str,
        delta_b: int,
        x: int,
        dprime: int,
        d: int,
        *,
        b: int,
        mu: float = 0.0,
    ) -> float:
        te = self.get_expr(seg_code, delta_b, x, dprime, d)
        if te is None:
            return 0.0
        sign = self._sign_from_seg_code(seg_code)
        return te.eval(b=int(b), mu=float(mu), sign=sign)


def load_overlap_tables_json(path: str | Path) -> OverlapTables:
    """Load overlap-table entries from a JSON file.

    Expected schema (see `pyscf-forge/guga_overlap_mapping_pack/docs/JSON_SCHEMA.md`):
      {"entries": [{"seg":..., "x":..., "db":..., "dp":..., "d":..., "expr":...}, ...]}

    Entries with empty expr (""), missing expr, or expr=="-" are ignored.
    """

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = data.get("entries", [])
    tab = OverlapTables()
    for e in entries:
        expr = str(e.get("expr", "")).strip()
        if not expr or expr == "-":
            continue
        tab.set(
            str(e["seg"]),
            int(e["db"]),
            None if e.get("x", None) is None else int(e["x"]),
            int(e["dp"]),
            int(e["d"]),
            expr,
        )
    return tab
