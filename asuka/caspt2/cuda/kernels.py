from __future__ import annotations

from dataclasses import dataclass
from typing import Any


from asuka.caspt2.cuda._ext import ext


def apply_h0diag_sr(
    *,
    y: Any,
    x: Any,
    bd: Any,
    id: Any,
    real_shift: float = 0.0,
    imag_shift: float = 0.0,
    alpha: float = 1.0,
    beta: float = 0.0,
    denom_tol: float = 1e-14,
) -> None:
    ext.apply_h0diag_sr_f64(y, x, bd, id, real_shift, imag_shift, alpha, beta, denom_tol)


def apply_precond_sr(
    *,
    out: Any,
    r: Any,
    bd: Any,
    id: Any,
    real_shift: float = 0.0,
    imag_shift: float = 0.0,
    scale: float = 1.0,
    denom_tol: float = 1e-14,
) -> None:
    ext.apply_precond_sr_f64(out, r, bd, id, real_shift, imag_shift, scale, denom_tol)


def mltsca(
    imltop: int,
    lst1: Any,
    lst2: Any,
    x: Any,
    f: Any,
    y: Any,
    *,
    val1: tuple[float, float],
    val2: tuple[float, float],
) -> None:
    ext.mltsca_f64(int(imltop), lst1, lst2, x, f, y, tuple(val1), tuple(val2))


def mltdxp(
    imltop: int,
    lst1: Any,
    lst2: Any,
    x: Any,
    f: Any,
    y: Any,
    *,
    val1: tuple[float, float],
    val2: tuple[float, float],
) -> None:
    ext.mltdxp_f64(int(imltop), lst1, lst2, x, f, y, tuple(val1), tuple(val2))


def mltmv(
    imltop: int,
    lst1: Any,
    x: Any,
    f: Any,
    y: Any,
    *,
    val1: tuple[float, float],
) -> None:
    ext.mltmv_f64(int(imltop), lst1, x, f, y, tuple(val1))


def mltr1(
    imltop: int,
    lst1: Any,
    x: Any,
    f: Any,
    y: Any,
    *,
    val1: tuple[float, float],
) -> None:
    ext.mltr1_f64(int(imltop), lst1, x, f, y, tuple(val1))


def ddot(x: Any, y: Any) -> float:
    return float(ext.ddot_f64(x, y))


@dataclass(frozen=True)
class ListSoA:
    """Coupling list stored as SoA int32 array on device: shape (4, n)."""

    data: Any

