from __future__ import annotations

from typing import Any

import numpy as np

from asuka.cuguga.drt import DRT


def build_cas_drt_for_ic_res(ic_res: Any, *, n_act: int) -> DRT:
    """Build a CAS DRT consistent with `ci_cas` ordering for phase-3 contractions."""

    drt_work = getattr(ic_res, "drt_work", None)
    if drt_work is None:
        raise NotImplementedError("Phase-3 dm2 blocks require ic_res.drt_work (semi-direct backend)")
    if not isinstance(drt_work, DRT):
        raise TypeError("ic_res.drt_work must be a DRT instance")

    nelec = int(getattr(drt_work, "nelec"))
    twos = int(getattr(drt_work, "twos_target"))

    spaces = getattr(ic_res, "spaces", None)
    orbsym_act = None
    if spaces is not None:
        orbsym = getattr(spaces, "orbsym", None)
        if orbsym is not None:
            orbsym_act = np.asarray(orbsym, dtype=np.int32).ravel()[: int(n_act)].tolist()

    wfnsym = None
    try:
        wfnsym = int(np.asarray(getattr(drt_work, "node_sym"))[int(drt_work.leaf)])
    except Exception:  # pragma: no cover
        wfnsym = None

    from asuka.cuguga import build_drt  # noqa: PLC0415

    return build_drt(norb=int(n_act), nelec=nelec, twos_target=twos, orbsym=orbsym_act, wfnsym=wfnsym)


def infer_n_act_n_virt(ic_res: Any) -> tuple[int, int]:
    spaces = getattr(ic_res, "spaces", None)
    if spaces is None:
        raise TypeError("ic_res missing OrbitalSpaces")
    n_act = int(getattr(spaces, "n_internal"))
    n_virt = int(getattr(spaces, "n_external"))
    if n_act < 0 or n_virt < 0:
        raise ValueError("invalid orbital spaces (negative sizes)")
    return n_act, n_virt


def require_internal_external_contiguous(spaces: Any, *, n_act: int, n_virt: int) -> None:
    internal = np.asarray(getattr(spaces, "internal"), dtype=np.int32).ravel()
    external = np.asarray(getattr(spaces, "external"), dtype=np.int32).ravel()
    if internal.size != int(n_act) or external.size != int(n_virt):
        raise ValueError("orbital spaces do not match n_act/n_virt")

    want_internal = np.arange(int(n_act), dtype=np.int32)
    want_external = np.arange(int(n_act), int(n_act) + int(n_virt), dtype=np.int32)
    if not bool(np.all(internal == want_internal)) or not bool(np.all(external == want_external)):
        raise NotImplementedError(
            "Phase-3 dm2 blocks currently require contiguous correlated ordering: internal=0..n_act-1, external=n_act.."
        )


def cas_dm123_for_ic_res(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    n_act: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(dm1_int, dm2_int, dm3_int)` for the reference CAS wavefunction."""

    drt_work = getattr(ic_res, "drt_work", None)
    if drt_work is None:
        raise NotImplementedError("Phase-3 dm2 blocks require ic_res.drt_work (semi-direct backend)")
    if not isinstance(drt_work, DRT):
        raise TypeError("ic_res.drt_work must be a DRT instance")

    nelec = int(getattr(drt_work, "nelec"))
    twos = int(getattr(drt_work, "twos_target"))

    ci_cas = np.asarray(ci_cas, dtype=np.float64).ravel()
    nrm = float(np.linalg.norm(ci_cas))
    if not np.isfinite(nrm) or nrm <= 0.0:
        raise ValueError("ci_cas must have nonzero finite norm")
    ci_cas = np.asarray(ci_cas / nrm, dtype=np.float64)

    from asuka import GUGAFCISolver  # noqa: PLC0415

    cas = GUGAFCISolver(twos=twos)
    dm1, dm2, dm3 = cas.make_rdm123(ci_cas, norb=int(n_act), nelec=nelec, reorder=True)
    dm1 = np.asarray(dm1, dtype=np.float64, order="C")
    dm2 = np.asarray(dm2, dtype=np.float64, order="C")
    dm3 = np.asarray(dm3, dtype=np.float64, order="C")
    return dm1, dm2, dm3


def cas_dm23_for_ic_res(ic_res: Any, *, ci_cas: np.ndarray, n_act: int) -> tuple[np.ndarray, np.ndarray]:
    """Return `(dm2_int, dm3_int)` for the reference CAS wavefunction."""

    _dm1, dm2, dm3 = cas_dm123_for_ic_res(ic_res, ci_cas=ci_cas, n_act=n_act)
    return dm2, dm3


__all__ = [
    "build_cas_drt_for_ic_res",
    "infer_n_act_n_virt",
    "require_internal_external_contiguous",
    "cas_dm123_for_ic_res",
    "cas_dm23_for_ic_res",
]
