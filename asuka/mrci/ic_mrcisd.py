from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from asuka.cuguga import build_drt
from asuka.mrci.generalized_davidson import GeneralizedDavidsonResult
from asuka.mrci.ic_basis import (
    ICDoubles,
    ICSingles,
    OrbitalSpaces,
    SCDoubles,
    SCSingles,
    enumerate_ic_doubles,
    enumerate_ic_singles,
    enumerate_sc_doubles,
    enumerate_sc_singles,
    filter_ic_doubles_by_norm,
    filter_ic_singles_by_norm,
)
from asuka.mrci.ic_sigma_rdm import ICRefSinglesRDM
from asuka.mrci.ic_sigma_semidirect import (
    ICRefSinglesDoublesSemiDirect,
    ICRefSinglesSemiDirect,
    ICStronglyContractedSemiDirect,
    ICStronglyContractedSemiDirectOTF,
)
from asuka.mrci.mrcisd import build_drt_mrcisd, embed_cas_ci_into_mrcisd
from asuka.solver import GUGAFCISolver


Contraction = Literal["fic", "sc"]
Backend = Literal["semi_direct", "rdm"]
SCBackend = Literal["dense", "otf"]
SolverMode = Literal["davidson", "dense", "auto"]


@dataclass(frozen=True)
class ICMRCISDResult:
    """Result of a contracted ic-MRCISD solve in the non-orthogonal CCF basis.

    Attributes
    ----------
    converged : bool
        Whether the solver converged.
    e : float
        Electronic energy in the correlated space (no ecore shift).
    e_tot : float
        Total energy including frozen core and nuclear repulsion.
    c : np.ndarray
        Contracted coefficients (S-normalized).
    spaces : OrbitalSpaces
        Definition of internal and external orbital spaces.
    singles : ICSingles | SCSingles
        Label set for singles.
    doubles : ICDoubles | SCDoubles
        Label set for doubles.
    backend : str
        Backend used ("semi_direct" or "rdm").
    drt_work : Any | None
        Restricted DRT used for the workspace, if applicable.
    niter : int
        Number of iterations performed.
    residual_norm : float
        Norm of the residual vector.
    diagnostics : dict[str, float]
        Additional diagnostics (weights, timings, etc.).
    """

    converged: bool
    e: float
    e_tot: float
    c: np.ndarray

    spaces: OrbitalSpaces
    singles: ICSingles | SCSingles
    doubles: ICDoubles | SCDoubles

    backend: str
    drt_work: Any | None

    niter: int
    residual_norm: float
    diagnostics: dict[str, float]


def _normalize_ci(ci: np.ndarray) -> np.ndarray:
    ci = np.asarray(ci, dtype=np.float64).ravel()
    n = float(np.linalg.norm(ci))
    if not np.isfinite(n) or n <= 0.0:
        raise ValueError("ci vector must have nonzero finite norm")
    return np.asarray(ci / n, dtype=np.float64)


def _sector_weights(overlap, c: np.ndarray, *, n_singles: int, n_doubles: int) -> dict[str, float]:
    c = np.asarray(c, dtype=np.float64).ravel()
    if int(c.size) != 1 + int(n_singles) + int(n_doubles):
        raise ValueError("c has wrong length for given n_singles/n_doubles")

    c0 = float(c[0])
    w_ref = c0 * c0

    w_singles = 0.0
    if int(n_singles):
        tmp = np.zeros_like(c)
        tmp[1 : 1 + int(n_singles)] = c[1 : 1 + int(n_singles)]
        rho = np.asarray(overlap(tmp), dtype=np.float64).ravel()
        w_singles = float(np.dot(tmp[1 : 1 + int(n_singles)], rho[1 : 1 + int(n_singles)]))

    w_doubles = 0.0
    if int(n_doubles):
        tmp = np.zeros_like(c)
        tmp[1 + int(n_singles) :] = c[1 + int(n_singles) :]
        rho = np.asarray(overlap(tmp), dtype=np.float64).ravel()
        w_doubles = float(np.dot(tmp[1 + int(n_singles) :], rho[1 + int(n_singles) :]))

    w_total = float(np.dot(c, np.asarray(overlap(c), dtype=np.float64).ravel()))
    return {
        "w_ref": float(w_ref),
        "w_singles": float(w_singles),
        "w_doubles": float(w_doubles),
        "w_total": float(w_total),
    }


def _ref_energy_from_rdms(*, h1e: np.ndarray, eri4: np.ndarray, gamma: np.ndarray, dm2: np.ndarray) -> float:
    """Return E_ref = <Psi0|H|Psi0> in the spin-free E_pq formalism.

    Uses the `GUGAFCISolver.make_rdm12` convention:
      dm2[p,q,r,s] = <E_{p q} E_{r s}> - δ_{q r} <E_{p s}>.
    """

    h1e = np.asarray(h1e, dtype=np.float64)
    eri4 = np.asarray(eri4, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    nI = int(gamma.shape[0])
    if gamma.shape != (nI, nI):
        raise ValueError("gamma must be square")
    if dm2.shape != (nI, nI, nI, nI):
        raise ValueError("dm2 shape must match gamma")
    if h1e.shape[0] < nI or h1e.shape[1] < nI:
        raise ValueError("h1e is too small for internal dimension")
    if eri4.shape[0] < nI:
        raise ValueError("eri4 is too small for internal dimension")

    h_int = h1e[:nI, :nI]
    eri_int = eri4[:nI, :nI, :nI, :nI]
    e1 = float(np.einsum("pq,pq->", h_int, gamma, optimize=True))
    e2 = 0.5 * float(np.einsum("pqrs,pqrs->", eri_int, dm2, optimize=True))
    return float(e1 + e2)


def ic_mrcisd_kernel(
    *,
    h1e: Any,
    eri: Any,
    n_act: int,
    n_virt: int,
    nelec: int,
    twos: int,
    ci_cas: np.ndarray,
    ecore: float = 0.0,
    orbsym: Sequence[int] | None = None,
    wfnsym: int | None = None,
    max_virt_e: int = 2,
    hop_backend: str = "augmented",
    contraction: Contraction = "fic",
    backend: Backend = "semi_direct",
    sc_backend: SCBackend = "otf",
    symmetry: bool = True,
    allow_same_external: bool = True,
    allow_same_internal: bool = True,
    norm_min_singles: float = 0.0,
    norm_min_doubles: float = 0.0,
    tol: float = 1e-10,
    max_cycle: int = 80,
    max_space: int = 25,
    s_tol: float = 1e-12,
    solver: SolverMode = "davidson",
    dense_nlab_max: int = 250,
    contract_nthreads: int = 1,
    contract_blas_nthreads: int | None = 1,
    precompute_epq: bool = True,
) -> ICMRCISDResult:
    """Solve a state-specific contracted ic-MRCISD generalized eigenproblem.

    Parameters
    ----------
    h1e : Any
        One-electron integrals in the correlated MO basis.
    eri : Any
        Two-electron integrals in the correlated MO basis.
    n_act : int
        Number of active orbitals.
    n_virt : int
        Number of virtual orbitals.
    nelec : int
        Total number of electrons.
    twos : int
        Spin multiplicity (2S).
    ci_cas : np.ndarray
        Initial guess vector from CASCI.
    ecore : float, optional
        Frozen-core energy shift. Default is 0.0.
    orbsym : Sequence[int] | None, optional
        Orbital symmetry labels.
    wfnsym : int | None, optional
        Wavefunction symmetry label.
    max_virt_e : int, optional
        Maximum number of electrons in the virtual space. Default is 2.
    hop_backend : str, optional
        Hop backend ("augmented" or "fast"). Default is "augmented".
    contraction : {"fic", "sc"}, optional
        Contraction scheme. Default is "fic".
    backend : {"semi_direct", "rdm"}, optional
        Algorithm backend. Default is "semi_direct".
    sc_backend : {"dense", "otf"}, optional
        Backend for strongly contracted variant. Default is "otf".
    symmetry : bool, optional
        Whether to use symmetry. Default is True.
    allow_same_external : bool, optional
        Whether to allow same external orbitals. Default is True.
    allow_same_internal : bool, optional
        Whether to allow same internal orbitals. Default is True.
    norm_min_singles : float, optional
        Minimum norm threshold for singles. Default is 0.0.
    norm_min_doubles : float, optional
        Minimum norm threshold for doubles. Default is 0.0.
    tol : float, optional
        Convergence tolerance. Default is 1e-10.
    max_cycle : int, optional
        Maximum number of iterations. Default is 80.
    max_space : int, optional
        Maximum subspace dimensionality. Default is 25.
    s_tol : float, optional
        Singularity tolerance for the overlap matrix. Default is 1e-12.
    solver : {"davidson", "dense", "auto"}, optional
        Eigensolver method. Default is "davidson".
    dense_nlab_max : int, optional
        Maximum dimension for dense solver. Default is 250.
    contract_nthreads : int, optional
        Number of threads for contraction. Default is 1.
    contract_blas_nthreads : int | None, optional
        Number of threads for BLAS. Default is 1.
    precompute_epq : bool, optional
        Whether to precompute EPQ actions. Default is True.

    Returns
    -------
    ICMRCISDResult
        Result object containing energies and wavefunctions.
    """

    n_act = int(n_act)
    n_virt = int(n_virt)
    nelec = int(nelec)
    twos = int(twos)
    max_virt_e = int(max_virt_e)

    if n_act < 0 or n_virt < 0:
        raise ValueError("n_act and n_virt must be >= 0")
    if nelec < 0:
        raise ValueError("nelec must be >= 0")
    if max_virt_e < 0:
        raise ValueError("max_virt_e must be >= 0")

    contraction_s = str(contraction).strip().lower()
    backend_s = str(backend).strip().lower()
    sc_backend_s = str(sc_backend).strip().lower()
    if contraction_s not in ("fic", "sc"):
        raise ValueError("contraction must be 'fic' or 'sc'")
    if backend_s not in ("semi_direct", "rdm"):
        raise ValueError("backend must be 'semi_direct' or 'rdm'")
    if sc_backend_s not in ("dense", "otf"):
        raise ValueError("sc_backend must be 'dense' or 'otf'")
    solver_s = str(solver).strip().lower()
    if solver_s not in ("davidson", "dense", "auto"):
        raise ValueError("solver must be 'davidson', 'dense', or 'auto'")
    hop_backend_s = str(hop_backend).strip().lower()
    if hop_backend_s not in ("fast", "augmented"):
        raise ValueError("hop_backend must be 'fast' or 'augmented'")
    dense_nlab_max = int(dense_nlab_max)
    if dense_nlab_max < 1:
        raise ValueError("dense_nlab_max must be >= 1")

    ci_cas = _normalize_ci(ci_cas)

    # Internal (CAS) RDMs are required for norm screening and for the RDM backend.
    cas = GUGAFCISolver(twos=twos)
    gamma, dm2 = cas.make_rdm12(ci_cas, norb=n_act, nelec=nelec)
    gamma = np.asarray(gamma, dtype=np.float64, order="C")
    dm2 = np.asarray(dm2, dtype=np.float64, order="C")

    norb_corr = n_act + n_virt
    orbsym_corr = None
    if orbsym is not None:
        orbsym_corr = np.asarray(orbsym, dtype=np.int32).ravel()
        if int(orbsym_corr.size) != norb_corr:
            raise ValueError("orbsym must have length n_act + n_virt (correlated orbital space)")

    spaces = OrbitalSpaces(
        internal=np.arange(n_act, dtype=np.int32),
        external=np.arange(n_act, n_act + n_virt, dtype=np.int32),
        orbsym=orbsym_corr,
    )

    singles: ICSingles | SCSingles
    doubles: ICDoubles | SCDoubles

    n_singles_raw = 0
    n_doubles_raw = 0

    if contraction_s == "fic":
        singles_f = enumerate_ic_singles(spaces, symmetry=bool(symmetry))
        doubles_f = enumerate_ic_doubles(
            spaces,
            symmetry=bool(symmetry),
            allow_same_external=bool(allow_same_external),
            allow_same_internal=bool(allow_same_internal),
        )
        n_singles_raw = int(singles_f.nlab)
        n_doubles_raw = int(doubles_f.nlab)

        if float(norm_min_singles) > 0.0:
            singles_f = filter_ic_singles_by_norm(singles_f, gamma=gamma, norm_min=float(norm_min_singles))
        if float(norm_min_doubles) > 0.0:
            doubles_f = filter_ic_doubles_by_norm(doubles_f, dm2=dm2, norm_min=float(norm_min_doubles))

        singles = singles_f
        doubles = doubles_f
    else:
        singles = enumerate_sc_singles(spaces, symmetry=bool(symmetry))
        doubles = enumerate_sc_doubles(
            spaces,
            symmetry=bool(symmetry),
            allow_same_external=bool(allow_same_external),
            allow_same_internal=bool(allow_same_internal),
        )
        n_singles_raw = int(singles.nlab)
        n_doubles_raw = int(doubles.nlab)

    # Semi-direct backends require a restricted uncontracted DRT workspace and an embedded reference vector.
    drt_work = None
    psi0 = None
    hop_map = None
    contract_executor: ThreadPoolExecutor | None = None
    contract_ws = None
    if backend_s == "semi_direct":
        orbsym_act = None if orbsym_corr is None else orbsym_corr[:n_act].tolist()
        drt_cas = build_drt(
            norb=n_act,
            nelec=nelec,
            twos_target=twos,
            orbsym=orbsym_act,
            wfnsym=wfnsym,
        )
        drt_work = build_drt_mrcisd(
            n_act=n_act,
            n_virt=n_virt,
            nelec=nelec,
            twos=twos,
            orbsym=None if orbsym_corr is None else orbsym_corr.tolist(),
            wfnsym=wfnsym,
            max_virt_e=max_virt_e,
        )
        _ci0, psi0, _ref_idx = embed_cas_ci_into_mrcisd(drt_cas=drt_cas, drt_mrci=drt_work, ci_cas=ci_cas, n_virt=n_virt)

        if hop_backend_s == "augmented":
            from asuka.mrci.projected_hop import build_subspace_map  # noqa: PLC0415

            drt_hop = build_drt_mrcisd(
                n_act=n_act,
                n_virt=n_virt,
                nelec=nelec,
                twos=twos,
                orbsym=None if orbsym_corr is None else orbsym_corr.tolist(),
                wfnsym=wfnsym,
                max_virt_e=int(max_virt_e) + 1,
            )
            hop_map = build_subspace_map(drt_full=drt_hop, drt_sub=drt_work)

        if int(contract_nthreads) > 1:
            contract_executor = ThreadPoolExecutor(max_workers=int(contract_nthreads))
        try:
            from asuka.contract import ContractWorkspace  # noqa: PLC0415

            contract_ws = ContractWorkspace()
        except Exception:  # pragma: no cover
            contract_ws = None

    def _solve_dense_lowest(h: np.ndarray, s: np.ndarray, *, s_tol: float) -> tuple[float, np.ndarray]:
        h = 0.5 * (h + h.T)
        s = 0.5 * (s + s.T)
        evals_s, evecs_s = np.linalg.eigh(s)
        keep = evals_s > float(s_tol)
        if not np.any(keep):
            raise np.linalg.LinAlgError("overlap matrix is numerically singular")
        u = evecs_s[:, keep]
        t = u @ np.diag(1.0 / np.sqrt(evals_s[keep]))
        h_ortho = 0.5 * ((t.T @ h @ t) + (t.T @ h @ t).T)
        evals_h, evecs_h = np.linalg.eigh(h_ortho)
        idx = int(np.argmin(evals_h))
        e0 = float(evals_h[idx])
        x0 = t @ evecs_h[:, idx]
        x0 = np.asarray(x0, dtype=np.float64).ravel()
        # Normalize in the S-metric.
        n2 = float(np.dot(x0, s @ x0))
        if n2 <= 0.0:  # pragma: no cover
            raise np.linalg.LinAlgError("lowest-root eigenvector has non-positive S-norm")
        x0 = x0 / np.sqrt(n2)
        return float(e0), np.ascontiguousarray(x0, dtype=np.float64)

    # Build backend operator and solve.
    ws = None
    try:
        if contraction_s == "fic":
            if backend_s == "rdm":
                if int(n_virt) != 0:
                    # ICRefSinglesRDM currently only covers [ref+singles] and expects dense ERIs.
                    raise NotImplementedError("backend='rdm' currently supports only n_virt==0 or ref+singles prototypes")
                e_ref = _ref_energy_from_rdms(
                    h1e=np.asarray(h1e, dtype=np.float64),
                    eri4=np.asarray(eri, dtype=np.float64),
                    gamma=gamma,
                    dm2=dm2,
                )
                ws = ICRefSinglesRDM(
                    h1e=h1e,
                    eri=eri,
                    e_ref=float(e_ref),
                    gamma=gamma,
                    dm2=dm2,
                    spaces=spaces,
                    singles=singles,  # type: ignore[arg-type]
                )
            else:
                if drt_work is None or psi0 is None:
                    raise RuntimeError("internal error: missing semi-direct workspace")
                ws = ICRefSinglesDoublesSemiDirect(
                    drt=drt_work,
                    h1e=h1e,
                    eri=eri,
                    psi0=psi0,
                    singles=singles,  # type: ignore[arg-type]
                    doubles=doubles,  # type: ignore[arg-type]
                    contract_nthreads=int(contract_nthreads),
                    contract_blas_nthreads=contract_blas_nthreads,
                    precompute_epq=bool(precompute_epq),
                    hop_map=hop_map,
                    contract_executor=contract_executor,
                    contract_workspace=contract_ws,
                )
        else:
            if backend_s != "semi_direct":
                raise NotImplementedError("backend='rdm' is not implemented for strongly contracted ic-MRCI")
            if drt_work is None or psi0 is None:
                raise RuntimeError("internal error: missing semi-direct workspace")
            if sc_backend_s == "dense":
                ws = ICStronglyContractedSemiDirect(
                    drt=drt_work,
                    h1e=h1e,
                    eri=eri,
                    psi0=psi0,
                    internal=np.arange(n_act, dtype=np.int32),
                    singles=singles,  # type: ignore[arg-type]
                    doubles=doubles,  # type: ignore[arg-type]
                    allow_same_internal=bool(allow_same_internal),
                    contract_nthreads=int(contract_nthreads),
                    contract_blas_nthreads=contract_blas_nthreads,
                    precompute_epq=bool(precompute_epq),
                    hop_map=hop_map,
                    contract_executor=contract_executor,
                    contract_workspace=contract_ws,
                )
            else:
                ws = ICStronglyContractedSemiDirectOTF(
                    drt=drt_work,
                    h1e=h1e,
                    eri=eri,
                    psi0=psi0,
                    internal=np.arange(n_act, dtype=np.int32),
                    singles=singles,  # type: ignore[arg-type]
                    doubles=doubles,  # type: ignore[arg-type]
                    allow_same_internal=bool(allow_same_internal),
                    contract_nthreads=int(contract_nthreads),
                    contract_blas_nthreads=contract_blas_nthreads,
                    precompute_epq=bool(precompute_epq),
                    hop_map=hop_map,
                    contract_executor=contract_executor,
                    contract_workspace=contract_ws,
                )

        x0 = np.zeros(int(ws.nlab), dtype=np.float64)  # type: ignore[union-attr]
        x0[0] = 1.0

        nlab = int(ws.nlab)  # type: ignore[union-attr]
        do_dense = solver_s == "dense" or (solver_s == "auto" and nlab <= dense_nlab_max)
        if do_dense:
            h = np.empty((nlab, nlab), dtype=np.float64)
            s = np.empty((nlab, nlab), dtype=np.float64)
            for j in range(nlab):
                ej = np.zeros(nlab, dtype=np.float64)
                ej[j] = 1.0
                h[:, j] = np.asarray(ws.sigma(ej), dtype=np.float64).ravel()  # type: ignore[arg-type]
                s[:, j] = np.asarray(ws.overlap(ej), dtype=np.float64).ravel()  # type: ignore[arg-type]
            e_dense, c_dense = _solve_dense_lowest(h, s, s_tol=float(s_tol))
            r = h @ c_dense - float(e_dense) * (s @ c_dense)
            res = GeneralizedDavidsonResult(
                converged=True,
                e=float(e_dense),
                x=c_dense,
                niter=1,
                residual_norm=float(np.linalg.norm(r)),
            )
        else:
            res = ws.solve(  # type: ignore[assignment]
                x0=x0,
                tol=float(tol),
                max_cycle=int(max_cycle),
                max_space=int(max_space),
                s_tol=float(s_tol),
            )
    finally:
        if contract_executor is not None:
            contract_executor.shutdown(wait=True)

    e = float(res.e)
    e_tot = e + float(ecore)
    c = np.ascontiguousarray(res.x, dtype=np.float64)

    n_singles = int(singles.nlab)
    n_doubles = int(doubles.nlab)

    diag: dict[str, float] = {}
    diag.update(
        {
            "n_act": float(n_act),
            "n_virt": float(n_virt),
            "nelec": float(nelec),
            "twos": float(twos),
            "max_virt_e": float(max_virt_e),
            "allow_same_external": 1.0 if bool(allow_same_external) else 0.0,
            "allow_same_internal": 1.0 if bool(allow_same_internal) else 0.0,
            "nlab": float(int(ws.nlab)),  # type: ignore[union-attr]
            "n_singles_raw": float(n_singles_raw),
            "n_doubles_raw": float(n_doubles_raw),
            "n_singles": float(n_singles),
            "n_doubles": float(n_doubles),
        }
    )
    diag.update(_sector_weights(ws.overlap, c, n_singles=n_singles, n_doubles=n_doubles))  # type: ignore[arg-type]

    return ICMRCISDResult(
        converged=bool(res.converged),
        e=float(e),
        e_tot=float(e_tot),
        c=c,
        spaces=spaces,
        singles=singles,
        doubles=doubles,
        backend=str(backend_s),
        drt_work=drt_work,
        niter=int(res.niter),
        residual_norm=float(res.residual_norm),
        diagnostics=diag,
    )
