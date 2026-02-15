from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import scipy.linalg

from .labels import AOInfo

Scheme = Literal["lowdin", "mulliken"]
GroupBy = Literal["ao", "shell", "atom", "l", "atom_l"]


def _asnumpy(a: Any) -> np.ndarray:
    """Best-effort convert numpy/cupy to numpy float64."""

    try:
        import cupy as cp  # type: ignore
    except Exception:
        cp = None
    if cp is not None and isinstance(a, cp.ndarray):  # type: ignore[attr-defined]
        a = cp.asnumpy(a)
    return np.asarray(a, dtype=np.float64)


def lowdin_sqrt_times_vec(S: np.ndarray, v: np.ndarray, *, tol: float = 1e-12) -> np.ndarray:
    """Compute (S^{1/2}) v without forming S^{1/2} explicitly."""

    S = np.asarray(S, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64).reshape((-1,))
    if S.ndim != 2 or S.shape[0] != S.shape[1] or S.shape[0] != v.size:
        raise ValueError("dimension mismatch in lowdin_sqrt_times_vec")

    s, U = scipy.linalg.eigh(S)
    if np.any(s <= float(tol)):
        raise ValueError("overlap matrix not positive definite (or ill-conditioned)")
    Utv = U.T @ v
    return U @ (np.sqrt(s) * Utv)


def mo_ao_weights(
    C: Any,
    S: np.ndarray,
    mo: int,
    *,
    scheme: Scheme = "lowdin",
    lowdin_tol: float = 1e-12,
) -> np.ndarray:
    """Return per-AO weights for a single MO index.

    Parameters
    ----------
    C
        MO coefficients in AO basis, shape (nao,nmo). Can be numpy or cupy.
    S
        AO overlap (nao,nao), numpy.
    mo
        MO index.
    scheme
        - "lowdin": weights = (S^{1/2} c)^2 (non-negative, sums ~ 1)
        - "mulliken": weights = c * (S c) (sums ~ 1, can be slightly negative)
    """

    Cn = _asnumpy(C)
    S = np.asarray(S, dtype=np.float64)
    if Cn.ndim != 2:
        raise ValueError("C must be 2D (nao,nmo)")
    nao, nmo = map(int, Cn.shape)
    if S.shape != (nao, nao):
        raise ValueError("S shape mismatch")
    mo = int(mo)
    if mo < 0 or mo >= nmo:
        raise IndexError("mo out of range")

    c = Cn[:, mo].copy()
    scheme_s = str(scheme).strip().lower()
    if scheme_s == "lowdin":
        ctilde = lowdin_sqrt_times_vec(S, c, tol=float(lowdin_tol))
        w = ctilde * ctilde
    elif scheme_s == "mulliken":
        Sc = S @ c
        w = c * Sc
    else:
        raise ValueError("scheme must be 'lowdin' or 'mulliken'")

    s = float(np.sum(w))
    if s != 0.0 and np.isfinite(s):
        w = w / s
    return np.asarray(w, dtype=np.float64)


@dataclass(frozen=True)
class ContribRow:
    key: str
    weight: float
    count: int = 1
    extra: dict[str, Any] | None = None


def group_contrib(
    ao_infos: list[AOInfo],
    weights: np.ndarray,
    *,
    groupby: GroupBy = "atom",
) -> list[ContribRow]:
    """Group per-AO weights into AO/shell/atom/l buckets."""

    weights = np.asarray(weights, dtype=np.float64).reshape((-1,))
    if len(ao_infos) != int(weights.size):
        raise ValueError("ao_infos length mismatch with weights")

    groupby_s = str(groupby).strip().lower()
    if groupby_s not in {"ao", "shell", "atom", "l", "atom_l"}:
        raise ValueError("invalid groupby")

    acc: dict[str, tuple[float, int, dict[str, Any]]] = {}
    for info, w in zip(ao_infos, weights):
        if groupby_s == "ao":
            key = info.label
            meta = {"ao": info.ao, "shell": info.shell, "atom": info.atom, "l": info.l, "comp": (info.lx, info.ly, info.lz)}
        elif groupby_s == "shell":
            key = f"{info.element}{info.atom+1} shell{info.shell} l={info.l}"
            meta = {"shell": info.shell, "atom": info.atom, "l": info.l}
        elif groupby_s == "atom":
            key = f"{info.element}{info.atom+1}"
            meta = {"atom": info.atom, "element": info.element}
        elif groupby_s == "l":
            key = f"l={info.l}"
            meta = {"l": info.l}
        else:  # atom_l
            key = f"{info.element}{info.atom+1} l={info.l}"
            meta = {"atom": info.atom, "element": info.element, "l": info.l}

        if key in acc:
            sw, cnt, m = acc[key]
            acc[key] = (sw + float(w), cnt + 1, m)
        else:
            acc[key] = (float(w), 1, meta)

    rows = [ContribRow(key=k, weight=v[0], count=v[1], extra=v[2]) for k, v in acc.items()]
    rows.sort(key=lambda r: abs(float(r.weight)), reverse=True)
    return rows


def top_contrib(
    rows: list[ContribRow],
    *,
    top: int | None = 10,
    thresh: float | None = None,
) -> list[ContribRow]:
    out = rows
    if thresh is not None:
        t = float(thresh)
        out = [r for r in out if abs(float(r.weight)) >= t]
    if top is not None:
        out = out[: int(top)]
    return out


__all__ = [
    "ContribRow",
    "GroupBy",
    "Scheme",
    "group_contrib",
    "lowdin_sqrt_times_vec",
    "mo_ao_weights",
    "top_contrib",
]

