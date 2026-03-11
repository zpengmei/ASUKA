from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .contrib import ContribRow, GroupBy, Scheme, group_contrib, mo_ao_weights, top_contrib
from .cube import write_mo_cube
from .labels import AOInfo, build_ao_info


def _asnumpy(a: Any) -> np.ndarray:
    try:
        import cupy as cp  # type: ignore
    except Exception:
        cp = None
    if cp is not None and isinstance(a, cp.ndarray):  # type: ignore[attr-defined]
        a = cp.asnumpy(a)
    return np.asarray(a)


def _get_xp(*arrs: Any) -> tuple[Any, bool]:
    try:
        import cupy as cp  # type: ignore
    except Exception:
        cp = None
    if cp is not None:
        for a in arrs:
            if isinstance(a, cp.ndarray):  # type: ignore[attr-defined]
                return cp, True
    return np, False


def _as_xp(xp: Any, a: Any, *, dtype: Any) -> Any:
    return xp.asarray(a, dtype=dtype)


def _symmetrize(xp: Any, a: Any) -> Any:
    return 0.5 * (a + a.T)


def _orthogonalizer_from_s(S: Any, *, eps: float = 1e-12) -> Any:
    xp, _ = _get_xp(S)
    S = _as_xp(xp, S, dtype=xp.float64)
    S = _symmetrize(xp, S)
    s, U = xp.linalg.eigh(S)
    if bool(xp.any(s <= float(eps))):
        raise ValueError("S is not positive definite (small/negative eigenvalues)")
    return U @ xp.diag(s ** (-0.5)) @ U.T


def _spatialize_uhf_mo_coeff(*, S_ao: Any, mo_coeff: tuple[Any, Any], mo_occ: tuple[Any, Any]) -> tuple[Any, Any]:
    """Build spatial natural orbitals from UHF (Ca,Cb)/(occ_a,occ_b)."""

    Ca, Cb = mo_coeff
    occ_a, occ_b = mo_occ
    xp, _ = _get_xp(Ca, Cb, occ_a, occ_b, S_ao)
    Ca = _as_xp(xp, Ca, dtype=xp.float64)
    Cb = _as_xp(xp, Cb, dtype=xp.float64)
    occ_a = _as_xp(xp, occ_a, dtype=xp.float64).ravel()
    occ_b = _as_xp(xp, occ_b, dtype=xp.float64).ravel()
    S = _as_xp(xp, S_ao, dtype=xp.float64)

    if Ca.ndim != 2 or Cb.ndim != 2:
        raise ValueError("mo_coeff must be (Ca,Cb) with 2D arrays")
    if Ca.shape != Cb.shape:
        raise ValueError("Ca/Cb shape mismatch")
    nao, nmo = map(int, Ca.shape)
    if S.shape != (nao, nao):
        raise ValueError("S_ao shape mismatch with mo_coeff")
    if occ_a.shape != (nmo,) or occ_b.shape != (nmo,):
        raise ValueError("mo_occ shape mismatch with mo_coeff")

    Da = (Ca * occ_a[None, :]) @ Ca.T
    Db = (Cb * occ_b[None, :]) @ Cb.T
    D = _symmetrize(xp, Da + Db)

    X = _orthogonalizer_from_s(S)
    D_orth = _symmetrize(xp, X.T @ D @ X)
    occ_no, U = xp.linalg.eigh(D_orth)
    idx = xp.argsort(occ_no)[::-1]
    occ_no = occ_no[idx]
    U = U[:, idx]

    C_spatial = _as_xp(xp, X @ U, dtype=xp.float64)
    occ_no = _as_xp(xp, occ_no, dtype=xp.float64)
    return C_spatial, occ_no


@dataclass(frozen=True)
class MOInspector:
    mol: Any
    ao_basis: Any
    S: np.ndarray
    C: Any  # numpy or cupy
    mo_energy: np.ndarray | None
    mo_occ: np.ndarray | None
    ncore: int | None
    ncas: int | None
    ao_info: list[AOInfo]

    @property
    def nao(self) -> int:
        return int(self.S.shape[0])

    @property
    def nmo(self) -> int:
        return int(_asnumpy(self.C).shape[1])

    @classmethod
    def from_scf_out(cls, scf_out: Any) -> "MOInspector":
        C = getattr(scf_out.scf, "mo_coeff", None)
        if C is None:
            raise ValueError("scf_out.scf.mo_coeff missing")
        S = np.asarray(scf_out.int1e.S, dtype=np.float64)
        ao_info = build_ao_info(scf_out.mol, scf_out.ao_basis)

        mo_energy_raw = getattr(scf_out.scf, "mo_energy", None)
        mo_occ_raw = getattr(scf_out.scf, "mo_occ", None)
        mo_energy = _asnumpy(mo_energy_raw) if mo_energy_raw is not None and not isinstance(mo_energy_raw, tuple) else None
        mo_occ = _asnumpy(mo_occ_raw) if mo_occ_raw is not None and not isinstance(mo_occ_raw, tuple) else None

        if isinstance(C, tuple):
            if not isinstance(mo_occ_raw, tuple) or len(mo_occ_raw) != 2:
                raise ValueError("UHF mo_coeff=(Ca,Cb) requires scf_out.scf.mo_occ=(occ_a,occ_b)")
            C, occ_no = _spatialize_uhf_mo_coeff(S_ao=S, mo_coeff=C, mo_occ=mo_occ_raw)
            mo_occ = _asnumpy(occ_no)
            mo_energy = None

        return cls(
            mol=scf_out.mol,
            ao_basis=scf_out.ao_basis,
            S=S,
            C=C,
            mo_energy=mo_energy,
            mo_occ=mo_occ,
            ncore=None,
            ncas=None,
            ao_info=ao_info,
        )

    @classmethod
    def from_casscf(
        cls,
        scf_out: Any,
        casscf: Any,
    ) -> "MOInspector":
        base = cls.from_scf_out(scf_out)
        return cls(
            mol=base.mol,
            ao_basis=base.ao_basis,
            S=base.S,
            C=casscf.mo_coeff,
            mo_energy=base.mo_energy,  # CASSCF orbitals have no unique eps; keep SCF eps for context
            mo_occ=base.mo_occ,
            ncore=int(casscf.ncore),
            ncas=int(casscf.ncas),
            ao_info=base.ao_info,
        )

    @classmethod
    def from_casci(
        cls,
        scf_out: Any,
        casci: Any,
    ) -> "MOInspector":
        base = cls.from_scf_out(scf_out)
        return cls(
            mol=base.mol,
            ao_basis=base.ao_basis,
            S=base.S,
            C=casci.mo_coeff,
            mo_energy=base.mo_energy,
            mo_occ=base.mo_occ,
            ncore=int(casci.ncore),
            ncas=int(casci.ncas),
            ao_info=base.ao_info,
        )

    def orbital_summary(self, *, max_rows: int | None = 50) -> str:
        Cn = _asnumpy(self.C)
        nmo = int(Cn.shape[1])
        nshow = nmo if max_rows is None else min(nmo, int(max_rows))

        lines: list[str] = []
        lines.append(" idx   type     occ        eps")
        lines.append("----  -------  -------  -----------")

        for i in range(nshow):
            typ = "mo"
            if self.ncore is not None and self.ncas is not None:
                ncore = int(self.ncore)
                ncas = int(self.ncas)
                if i < ncore:
                    typ = "core"
                elif i < ncore + ncas:
                    typ = "active"
                else:
                    typ = "virt"

            if self.mo_occ is not None:
                try:
                    occ = f"{float(np.ravel(self.mo_occ)[i]):7.3f}"
                except Exception:
                    occ = "   n/a "
            else:
                occ = "   n/a "

            if self.mo_energy is not None:
                try:
                    eps = f"{float(np.ravel(self.mo_energy)[i]):11.6f}"
                except Exception:
                    eps = "      n/a  "
            else:
                eps = "      n/a  "

            lines.append(f"{i:4d}  {typ:7s}  {occ}  {eps}")
        if nshow < nmo:
            lines.append(f"... ({nmo-nshow} more)")
        return "\n".join(lines)

    def ao_contrib(
        self,
        *,
        mo: int,
        scheme: Scheme = "lowdin",
        groupby: GroupBy = "atom",
        top: int | None = 10,
        thresh: float | None = None,
    ) -> list[ContribRow]:
        w = mo_ao_weights(self.C, self.S, int(mo), scheme=scheme)
        rows = group_contrib(self.ao_info, w, groupby=groupby)
        return top_contrib(rows, top=top, thresh=thresh)

    @staticmethod
    def format_contrib(rows: list[ContribRow], *, max_key: int = 24) -> str:
        lines: list[str] = []
        lines.append(" key".ljust(max_key + 2) + "weight    count")
        lines.append("-" * (max_key + 2 + 16))
        for r in rows:
            k = (r.key[: max_key - 1] + "…") if len(r.key) > max_key else r.key
            lines.append(f" {k:<{max_key}s}  {r.weight: .6f}  {r.count:5d}")
        return "\n".join(lines)

    def write_mo_cube(self, path: str, *, mo: int, spacing: float = 0.25, padding: float = 4.0) -> None:
        write_mo_cube(path, self.mol, self.ao_basis, self.C, int(mo), spacing=float(spacing), padding=float(padding))

    def write_active_space_cubes(self, out_dir: str, *, spacing: float = 0.25, padding: float = 4.0) -> None:
        if self.ncore is None or self.ncas is None:
            raise ValueError("active-space info missing (ncore/ncas not set)")
        import os

        os.makedirs(out_dir, exist_ok=True)
        ncore = int(self.ncore)
        ncas = int(self.ncas)
        for i in range(ncore, ncore + ncas):
            self.write_mo_cube(os.path.join(out_dir, f"mo_{i:03d}.cube"), mo=i, spacing=spacing, padding=padding)


__all__ = ["MOInspector"]
