"""Spin-adapted CISD (CSF/DRT) implemented via ASUKA's uncontracted MRCISD kernel.

This module operates directly on MO-basis
integrals in an ``[occupied][virtual]`` ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from asuka.cuguga.drt import DRT, build_drt
from asuka.mrci.mrcisd import (
    MRCISDResult,
    MRCISDResultMulti,
    build_drt_mrcisd,
    mrcisd_kernel,
)


@dataclass(frozen=True)
class CISDResult:
    """Single-root CISD result (spin-adapted CSF basis)."""

    converged: bool
    e_cisd: float
    ci: np.ndarray
    drt: DRT
    ci_ref0: np.ndarray
    ref_idx: np.ndarray
    diagnostics: dict[str, float]


@dataclass(frozen=True)
class CISDResultMulti:
    """Multi-root CISD result (spin-adapted CSF basis)."""

    converged: np.ndarray  # (nroots,) bool
    e_cisd: np.ndarray  # (nroots,) float
    ci: list[np.ndarray]
    drt: DRT
    ci_ref0: list[np.ndarray]
    ref_idx: np.ndarray
    overlap_ref_root: np.ndarray
    diagnostics: list[dict[str, float]]


@dataclass
class GUGACISDSolver:
    """Solver-style wrapper for CISD (mirrors other ASUKA solver APIs).

    This class stores the problem definition and solver options; call :meth:`kernel`
    with MO-basis integrals to run the calculation.
    """

    n_occ: int
    n_virt: int
    nelec: int
    twos: int

    # Reference specification (pick one)
    ci_ref_occ: np.ndarray | None = None
    ref_steps_occ: Sequence[int | str] | None = None

    # Symmetry
    orbsym: Sequence[int] | None = None
    wfnsym: int | None = None

    # Options
    ecore: float = 0.0
    nroots: int = 1
    max_virt_e: int = 2
    hop_backend: str | None = None
    tol: float = 1e-10
    lindep: float = 1e-14
    max_cycle: int = 100
    max_space: int = 12
    max_memory: float = 4000.0
    contract_nthreads: int = 1
    contract_blas_nthreads: int | None = 1
    precompute_epq: bool = True
    verbose: int | None = None

    def kernel(self, h1e: Any, eri: Any) -> CISDResult | CISDResultMulti:
        return cisd_kernel(
            h1e=h1e,
            eri=eri,
            n_occ=int(self.n_occ),
            n_virt=int(self.n_virt),
            nelec=int(self.nelec),
            twos=int(self.twos),
            ci_ref_occ=self.ci_ref_occ,
            ref_steps_occ=self.ref_steps_occ,
            orbsym=self.orbsym,
            wfnsym=self.wfnsym,
            ecore=float(self.ecore),
            nroots=int(self.nroots),
            max_virt_e=int(self.max_virt_e),
            hop_backend=self.hop_backend,
            tol=float(self.tol),
            lindep=float(self.lindep),
            max_cycle=int(self.max_cycle),
            max_space=int(self.max_space),
            max_memory=float(self.max_memory),
            contract_nthreads=int(self.contract_nthreads),
            contract_blas_nthreads=self.contract_blas_nthreads,
            precompute_epq=bool(self.precompute_epq),
            verbose=self.verbose,
        )


def build_drt_cisd(
    *,
    n_occ: int,
    n_virt: int,
    nelec: int,
    twos: int,
    orbsym: Sequence[int] | None = None,
    wfnsym: int | None = None,
    max_virt_e: int = 2,
) -> DRT:
    """Build the restricted CISD DRT for an ``[occupied][virtual]`` ordering.

    Notes
    -----
    This is the same construction used by uncontracted MRCISD, with the
    "active" block identified with the *occupied* orbitals.
    """

    return build_drt_mrcisd(
        n_act=int(n_occ),
        n_virt=int(n_virt),
        nelec=int(nelec),
        twos=int(twos),
        orbsym=orbsym,
        wfnsym=wfnsym,
        max_virt_e=int(max_virt_e),
    )


def _infer_ref_steps_occ(*, n_occ: int, nelec: int, twos: int) -> np.ndarray:
    """Infer a single-CSF reference path in the occupied-only DRT.

    Supported default references
    ----------------------------
    1) Closed-shell RHF-like:
        nelec == 2*n_occ and twos == 0  ->  D...D
    2) High-spin ROHF-like (all unpaired aligned):
        n_socc = 2*n_occ - nelec, twos == n_socc  ->  D...D U...U

    Returns
    -------
    steps : np.ndarray
        int8 array of length n_occ containing DRT step codes:
        0='E', 1='U', 2='L', 3='D'.
    """

    n_occ_i = int(n_occ)
    nelec_i = int(nelec)
    twos_i = int(twos)

    if n_occ_i < 0:
        raise ValueError("n_occ must be >= 0")
    if nelec_i < 0:
        raise ValueError("nelec must be >= 0")
    if twos_i < 0:
        raise ValueError("twos must be >= 0")

    # Closed-shell
    if nelec_i == 2 * n_occ_i and twos_i == 0:
        return np.full(n_occ_i, 3, dtype=np.int8)  # D steps

    # High-spin ROHF-like: assume occupied orbitals are either doubly or singly occupied.
    n_socc = 2 * n_occ_i - nelec_i
    n_docc = nelec_i - n_occ_i
    if (
        0 <= n_socc <= n_occ_i
        and 0 <= n_docc <= n_occ_i
        and n_docc + n_socc == n_occ_i
        and twos_i == n_socc
    ):
        steps = np.empty(n_occ_i, dtype=np.int8)
        steps[:n_docc] = 3  # D
        steps[n_docc:] = 1  # U (maximal spin coupling)
        return steps

    raise ValueError(
        "Unable to infer a single-CSF CISD reference from (n_occ, nelec, twos). "
        "Provide ref_steps_occ (DRT path in the occupied block) or ci_ref_occ explicitly. "
        "Defaults only cover closed-shell RHF (D...D) and high-spin ROHF (D...D U...U)."
    )


def _ref_ci_from_steps(*, drt_occ: DRT, steps_occ: Sequence[int | str]) -> tuple[np.ndarray, int]:
    """Return a unit vector CI reference in the occupied-only DRT basis."""

    idx = int(drt_occ.path_to_index(steps_occ))
    ci = np.zeros(int(drt_occ.ncsf), dtype=np.float64)
    ci[idx] = 1.0
    return ci, idx


def cisd_kernel(
    *,
    h1e: Any,
    eri: Any,
    n_occ: int,
    n_virt: int,
    nelec: int,
    twos: int,
    # Reference specification (pick one)
    ci_ref_occ: np.ndarray | None = None,
    ref_steps_occ: Sequence[int | str] | None = None,
    # Symmetry
    orbsym: Sequence[int] | None = None,
    wfnsym: int | None = None,
    # Options
    ecore: float = 0.0,
    nroots: int = 1,
    max_virt_e: int = 2,
    hop_backend: str | None = None,
    tol: float = 1e-10,
    lindep: float = 1e-14,
    max_cycle: int = 100,
    max_space: int = 12,
    max_memory: float = 4000.0,
    contract_nthreads: int = 1,
    contract_blas_nthreads: int | None = 1,
    precompute_epq: bool = True,
    verbose: int | None = None,
) -> CISDResult | CISDResultMulti:
    """Solve spin-adapted CISD in a restricted DRT basis.

    Parameters
    ----------
    h1e, eri
        MO-basis 1e/2e integrals in the *same* spatial-orbital ordering
        ``[occupied][virtual]``.
    n_occ, n_virt
        Orbital partition of the correlated space.
    nelec, twos
        Total correlated electrons and target ``2S``.
    ci_ref_occ
        Reference CI vector in the occupied-only DRT basis (length = ncsf_occ).
        If provided, takes precedence over ``ref_steps_occ``.
    ref_steps_occ
        Reference DRT path (length = n_occ) using step codes ``{'E','U','L','D'}``
        or integer step indices ``0..3``.

    Returns
    -------
    CISDResult | CISDResultMulti
        Energy is returned as ``e_cisd`` (includes ``ecore`` shift).

    Notes
    -----
    Internally this is implemented by calling the uncontracted MRCISD kernel
    with ``n_act = n_occ`` and ``max_virt_e = 2``.
    """

    n_occ_i = int(n_occ)
    n_virt_i = int(n_virt)
    nelec_i = int(nelec)
    twos_i = int(twos)
    nroots_i = int(nroots)
    max_virt_e_i = int(max_virt_e)

    if nroots_i != 1 and not isinstance(ci_ref_occ, (list, tuple)):
        raise ValueError(
            "cisd_kernel: nroots != 1 requires providing ci_ref_occ as a list/tuple "
            "of occupied-space reference vectors (one per root guess)."
        )

    # Normalize symmetry labels (if provided) for the correlated ordering.
    orbsym_arr: tuple[int, ...] | None
    if orbsym is None:
        orbsym_arr = None
    else:
        arr = np.asarray(orbsym, dtype=np.int32).ravel()
        if int(arr.size) != n_occ_i + n_virt_i:
            raise ValueError(
                f"orbsym has wrong length {int(arr.size)} (expected {n_occ_i + n_virt_i})"
            )
        orbsym_arr = tuple(int(x) for x in arr.tolist())

    orbsym_occ = None if orbsym_arr is None else orbsym_arr[:n_occ_i]

    # Build the occupied-only DRT to validate/build the reference.
    drt_occ = build_drt(
        norb=n_occ_i,
        nelec=nelec_i,
        twos_target=twos_i,
        orbsym=orbsym_occ,
        wfnsym=wfnsym,
    )

    if ci_ref_occ is not None:
        if isinstance(ci_ref_occ, (list, tuple)):
            ci_cas = [np.asarray(v, dtype=np.float64).ravel() for v in ci_ref_occ]
        else:
            ci_cas = np.asarray(ci_ref_occ, dtype=np.float64).ravel()
        if not isinstance(ci_cas, list):
            if int(ci_cas.size) != int(drt_occ.ncsf):
                raise ValueError(
                    f"ci_ref_occ has wrong size {int(ci_cas.size)} (expected {int(drt_occ.ncsf)})"
                )
    else:
        if ref_steps_occ is None:
            ref_steps_occ = _infer_ref_steps_occ(n_occ=n_occ_i, nelec=nelec_i, twos=twos_i)
        ci_cas, _ = _ref_ci_from_steps(drt_occ=drt_occ, steps_occ=ref_steps_occ)

    out = mrcisd_kernel(
        h1e=h1e,
        eri=eri,
        n_act=n_occ_i,
        n_virt=n_virt_i,
        nelec=nelec_i,
        twos=twos_i,
        ci_cas=ci_cas,
        nroots=nroots_i,
        ecore=float(ecore),
        orbsym_act=orbsym_occ,
        orbsym_corr=orbsym_arr,
        wfnsym=wfnsym,
        max_virt_e=max_virt_e_i,
        hop_backend=hop_backend,
        tol=tol,
        lindep=lindep,
        max_cycle=max_cycle,
        max_space=max_space,
        max_memory=max_memory,
        contract_nthreads=contract_nthreads,
        contract_blas_nthreads=contract_blas_nthreads,
        precompute_epq=precompute_epq,
        verbose=verbose,
    )

    if isinstance(out, MRCISDResult):
        return CISDResult(
            converged=bool(out.converged),
            e_cisd=float(out.e_mrci),
            ci=np.asarray(out.ci, dtype=np.float64),
            drt=out.drt,
            ci_ref0=np.asarray(out.ci_ref0, dtype=np.float64),
            ref_idx=np.asarray(out.ref_idx, dtype=np.int32),
            diagnostics=dict(out.diagnostics),
        )

    if not isinstance(out, MRCISDResultMulti):  # pragma: no cover
        raise TypeError("internal error: unexpected return type from mrcisd_kernel")

    return CISDResultMulti(
        converged=np.asarray(out.converged, dtype=bool),
        e_cisd=np.asarray(out.e_mrci, dtype=np.float64),
        ci=[np.asarray(v, dtype=np.float64) for v in out.ci],
        drt=out.drt,
        ci_ref0=[np.asarray(v, dtype=np.float64) for v in out.ci_ref0],
        ref_idx=np.asarray(out.ref_idx, dtype=np.int32),
        overlap_ref_root=np.asarray(out.overlap_ref_root, dtype=np.float64),
        diagnostics=[dict(d) for d in out.diagnostics],
    )


def cisd(
    h1e: Any,
    eri: Any,
    *,
    n_occ: int,
    n_virt: int,
    nelec: int,
    twos: int,
    ci_ref_occ: np.ndarray | None = None,
    ref_steps_occ: Sequence[int | str] | None = None,
    orbsym: Sequence[int] | None = None,
    wfnsym: int | None = None,
    ecore: float = 0.0,
    nroots: int = 1,
    max_virt_e: int = 2,
    hop_backend: str | None = None,
    tol: float = 1e-10,
    lindep: float = 1e-14,
    max_cycle: int = 100,
    max_space: int = 12,
    max_memory: float = 4000.0,
    contract_nthreads: int = 1,
    contract_blas_nthreads: int | None = 1,
    precompute_epq: bool = True,
    verbose: int | None = None,
) -> CISDResult | CISDResultMulti:
    """Convenience wrapper around :func:`cisd_kernel` (integrals-first API)."""

    return cisd_kernel(
        h1e=h1e,
        eri=eri,
        n_occ=n_occ,
        n_virt=n_virt,
        nelec=nelec,
        twos=twos,
        ci_ref_occ=ci_ref_occ,
        ref_steps_occ=ref_steps_occ,
        orbsym=orbsym,
        wfnsym=wfnsym,
        ecore=ecore,
        nroots=nroots,
        max_virt_e=max_virt_e,
        hop_backend=hop_backend,
        tol=tol,
        lindep=lindep,
        max_cycle=max_cycle,
        max_space=max_space,
        max_memory=max_memory,
        contract_nthreads=contract_nthreads,
        contract_blas_nthreads=contract_blas_nthreads,
        precompute_epq=precompute_epq,
        verbose=verbose,
    )
