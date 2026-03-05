from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from asuka.cuguga import build_drt
from asuka.mrci.generalized_davidson import (
    GeneralizedDavidsonResult,
    GeneralizedDavidsonResultMulti,
    generalized_davidson,
)
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
from asuka.mrci.ic_sigma_rdm import ICDenseContractedRDM, ICRefSinglesRDM
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


@dataclass(frozen=True)
class ICMRCISDResultMulti:
    """Result of a shared-basis multi-root contracted ic-MRCISD solve."""

    converged: np.ndarray
    e: np.ndarray
    e_tot: np.ndarray
    c: list[np.ndarray]

    spaces: list[OrbitalSpaces]
    singles: list[ICSingles | SCSingles]
    doubles: list[ICDoubles | SCDoubles]
    allow_same_internal: bool
    backend: str
    drt_work: Any | None
    block_slices: list[tuple[int, int]]
    overlap_ref_root: np.ndarray

    niter: int
    residual_norm: np.ndarray
    diagnostics: list[dict[str, float]]


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
            if backend_s == "rdm" and int(n_virt) == 0:
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
                ws_base = ICRefSinglesDoublesSemiDirect(
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
                ws = ICDenseContractedRDM(ws_base) if backend_s == "rdm" else ws_base
        else:
            if drt_work is None or psi0 is None:
                raise RuntimeError("internal error: missing semi-direct workspace")
            if sc_backend_s == "dense":
                ws_base = ICStronglyContractedSemiDirect(
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
                ws_base = ICStronglyContractedSemiDirectOTF(
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
            ws = ICDenseContractedRDM(ws_base) if backend_s == "rdm" else ws_base

        x0 = np.zeros(int(ws.nlab), dtype=np.float64)  # type: ignore[union-attr]
        x0[0] = 1.0

        nlab = int(ws.nlab)  # type: ignore[union-attr]
        do_dense = bool(backend_s == "rdm") or solver_s == "dense" or (solver_s == "auto" and nlab <= dense_nlab_max)
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
            "rdm_specialized": 1.0 if bool(backend_s == "rdm" and contraction_s == "fic" and int(n_virt) == 0) else 0.0,
            "rdm_dense_general": 1.0 if bool(backend_s == "rdm" and not (contraction_s == "fic" and int(n_virt) == 0)) else 0.0,
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


@dataclass(frozen=True)
class _ICComponentSpec:
    ws: Any
    spaces: OrbitalSpaces
    singles: ICSingles | SCSingles
    doubles: ICDoubles | SCDoubles
    drt_work: Any | None
    n_singles_raw: int
    n_doubles_raw: int


class _ICUnionWorkspace:
    """Shared contracted basis formed by concatenating reference-specific bases."""

    def __init__(self, *, components: Sequence[_ICComponentSpec], hop):
        comps = list(components)
        if not comps:
            raise ValueError("components must be non-empty")
        self.components = comps
        self._hop = hop
        self.block_slices: list[tuple[int, int]] = []
        self.reference_offsets: list[int] = []

        start = 0
        for comp in self.components:
            nlab = int(comp.ws.nlab)
            stop = int(start + nlab)
            self.block_slices.append((int(start), int(stop)))
            self.reference_offsets.append(int(start))
            start = stop
        self.nlab = int(start)

    def expand(self, c: np.ndarray) -> np.ndarray:
        c = np.asarray(c, dtype=np.float64).ravel()
        if int(c.size) != int(self.nlab):
            raise ValueError("contracted coefficient vector has wrong length")
        out = None
        for comp, (start, stop) in zip(self.components, self.block_slices, strict=True):
            y = np.asarray(comp.ws.expand(c[int(start) : int(stop)]), dtype=np.float64)
            out = y if out is None else out + y
        if out is None:  # pragma: no cover
            raise RuntimeError("internal error: union basis produced no vector")
        return np.asarray(out, dtype=np.float64)

    def project(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64).ravel()
        out = np.empty((int(self.nlab),), dtype=np.float64)
        for comp, (start, stop) in zip(self.components, self.block_slices, strict=True):
            out[int(start) : int(stop)] = np.asarray(comp.ws.project(z), dtype=np.float64).ravel()
        return out

    def overlap(self, c: np.ndarray) -> np.ndarray:
        return self.project(self.expand(c))

    def sigma(self, c: np.ndarray) -> np.ndarray:
        return self.project(self._hop(self.expand(c)))

    def _compute_diag_precond(self) -> np.ndarray:
        out = np.empty((int(self.nlab),), dtype=np.float64)
        for comp, (start, stop) in zip(self.components, self.block_slices, strict=True):
            out[int(start) : int(stop)] = np.asarray(comp.ws._compute_diag_precond(), dtype=np.float64).ravel()
        return out


def _build_ic_component_workspace(
    *,
    h1e: Any,
    eri: Any,
    n_act: int,
    n_virt: int,
    nelec: int,
    twos: int,
    ci_cas: np.ndarray,
    orbsym_corr: np.ndarray | None,
    wfnsym: int | None,
    max_virt_e: int,
    hop_backend_s: str,
    contraction_s: str,
    sc_backend_s: str,
    symmetry: bool,
    allow_same_external: bool,
    allow_same_internal: bool,
    norm_min_singles: float,
    norm_min_doubles: float,
    contract_nthreads: int,
    contract_blas_nthreads: int | None,
    precompute_epq: bool,
) -> _ICComponentSpec:
    ci_cas = _normalize_ci(ci_cas)
    cas = GUGAFCISolver(twos=int(twos))
    gamma, dm2 = cas.make_rdm12(ci_cas, norb=int(n_act), nelec=int(nelec))
    gamma = np.asarray(gamma, dtype=np.float64, order="C")
    dm2 = np.asarray(dm2, dtype=np.float64, order="C")

    spaces = OrbitalSpaces(
        internal=np.arange(int(n_act), dtype=np.int32),
        external=np.arange(int(n_act), int(n_act) + int(n_virt), dtype=np.int32),
        orbsym=orbsym_corr,
    )

    if str(contraction_s) == "fic":
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

    orbsym_act = None if orbsym_corr is None else np.asarray(orbsym_corr[: int(n_act)], dtype=np.int32).tolist()
    drt_cas = build_drt(
        norb=int(n_act),
        nelec=int(nelec),
        twos_target=int(twos),
        orbsym=orbsym_act,
        wfnsym=wfnsym,
    )
    drt_work = build_drt_mrcisd(
        n_act=int(n_act),
        n_virt=int(n_virt),
        nelec=int(nelec),
        twos=int(twos),
        orbsym=None if orbsym_corr is None else np.asarray(orbsym_corr, dtype=np.int32).tolist(),
        wfnsym=wfnsym,
        max_virt_e=int(max_virt_e),
    )
    _ci0, psi0, _ref_idx = embed_cas_ci_into_mrcisd(
        drt_cas=drt_cas,
        drt_mrci=drt_work,
        ci_cas=ci_cas,
        n_virt=int(n_virt),
    )

    hop_map = None
    if str(hop_backend_s) in {"augmented", "cuda"}:
        from asuka.mrci.projected_hop import build_subspace_map  # noqa: PLC0415

        drt_hop = build_drt_mrcisd(
            n_act=int(n_act),
            n_virt=int(n_virt),
            nelec=int(nelec),
            twos=int(twos),
            orbsym=None if orbsym_corr is None else np.asarray(orbsym_corr, dtype=np.int32).tolist(),
            wfnsym=wfnsym,
            max_virt_e=int(max_virt_e) + 1,
        )
        hop_map = build_subspace_map(drt_full=drt_hop, drt_sub=drt_work)

    if str(contraction_s) == "fic":
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
            contract_executor=None,
            contract_workspace=None,
        )
    elif str(sc_backend_s) == "dense":
        ws = ICStronglyContractedSemiDirect(
            drt=drt_work,
            h1e=h1e,
            eri=eri,
            psi0=psi0,
            internal=np.arange(int(n_act), dtype=np.int32),
            singles=singles,  # type: ignore[arg-type]
            doubles=doubles,  # type: ignore[arg-type]
            allow_same_internal=bool(allow_same_internal),
            contract_nthreads=int(contract_nthreads),
            contract_blas_nthreads=contract_blas_nthreads,
            precompute_epq=bool(precompute_epq),
            hop_map=hop_map,
            contract_executor=None,
            contract_workspace=None,
        )
    else:
        ws = ICStronglyContractedSemiDirectOTF(
            drt=drt_work,
            h1e=h1e,
            eri=eri,
            psi0=psi0,
            internal=np.arange(int(n_act), dtype=np.int32),
            singles=singles,  # type: ignore[arg-type]
            doubles=doubles,  # type: ignore[arg-type]
            allow_same_internal=bool(allow_same_internal),
            contract_nthreads=int(contract_nthreads),
            contract_blas_nthreads=contract_blas_nthreads,
            precompute_epq=bool(precompute_epq),
            hop_map=hop_map,
            contract_executor=None,
            contract_workspace=None,
        )
    return _ICComponentSpec(
        ws=ws,
        spaces=spaces,
        singles=singles,
        doubles=doubles,
        drt_work=drt_work,
        n_singles_raw=int(n_singles_raw),
        n_doubles_raw=int(n_doubles_raw),
    )


def _build_uncontracted_hop(
    *,
    drt_work,
    h1e: Any,
    eri: Any,
    hop_backend_s: str,
    hop_map: Any | None,
    contract_nthreads: int,
    contract_blas_nthreads: int | None,
):
    try:
        from asuka.integrals.df_integrals import DFMOIntegrals, DeviceDFMOIntegrals  # noqa: PLC0415
    except Exception:  # pragma: no cover
        DFMOIntegrals = None  # type: ignore[assignment]
        DeviceDFMOIntegrals = None  # type: ignore[assignment]

    use_cuda = bool(str(hop_backend_s) == "cuda")
    if use_cuda:
        if DeviceDFMOIntegrals is None or not isinstance(eri, DeviceDFMOIntegrals):
            raise ValueError("hop_backend='cuda' currently requires DeviceDFMOIntegrals")
        import cupy as cp  # noqa: PLC0415
        from asuka.cuda.cuda_backend import (  # noqa: PLC0415
            build_epq_action_table_combined_device,
            make_device_drt,
            make_device_state_cache,
        )
        from asuka.cuda.mrci_hop import (  # noqa: PLC0415
            CudaMrciHopWorkspace,
            hop_cuda_epq_table,
            hop_cuda_projected,
        )

        norb = int(np.asarray(h1e).shape[0])
        nops = int(norb) * int(norb)
        h_eff_d = cp.asarray(h1e, dtype=cp.float64) - 0.5 * cp.asarray(eri.j_ps, dtype=cp.float64)
        l_full_d = None if eri.l_full is None else cp.asarray(eri.l_full, dtype=cp.float64)
        eri_mat_t_d = None
        if l_full_d is None:
            if eri.eri_mat is None:
                raise ValueError("DeviceDFMOIntegrals must provide l_full or eri_mat")
            eri_mat_d = cp.asarray(eri.eri_mat, dtype=cp.float64)
            eri_mat_t_d = 0.5 * eri_mat_d.T
        if hop_map is not None:
            drt_full = hop_map.drt_full
            drt_dev_full = make_device_drt(drt_full)
            state_dev_full = make_device_state_cache(drt=drt_full, drt_dev=drt_dev_full)
            epq_table_full = build_epq_action_table_combined_device(
                drt=drt_full, drt_dev=drt_dev_full, state_dev=state_dev_full
            )
            sub_to_full_d = cp.asarray(hop_map.sub_to_full, dtype=cp.int64)
            naux = None if l_full_d is None else int(l_full_d.shape[1])
            ws_full = CudaMrciHopWorkspace.auto(ncsf=int(drt_full.ncsf), nops=int(nops), naux=naux, sym_pair=False)
            x_full_buf = cp.empty((int(drt_full.ncsf),), dtype=cp.float64)
            y_full_buf = cp.empty((int(drt_full.ncsf),), dtype=cp.float64)

            def hop(x: np.ndarray) -> np.ndarray:
                x_d = cp.asarray(x, dtype=cp.float64)
                y_d = hop_cuda_projected(
                    drt_full=drt_full,
                    drt_dev_full=drt_dev_full,
                    state_dev_full=state_dev_full,
                    epq_table_full=epq_table_full,
                    h_eff=h_eff_d,
                    eri_mat_t_full=eri_mat_t_d,
                    l_full_full=l_full_d,
                    x_sub=x_d,
                    sub_to_full=sub_to_full_d,
                    x_full_buf=x_full_buf,
                    y_full_buf=y_full_buf,
                    workspace_full=ws_full,
                    sym_pair=False,
                )
                return np.asarray(cp.asnumpy(y_d), dtype=np.float64)

            return hop

        drt_dev = make_device_drt(drt_work)
        state_dev = make_device_state_cache(drt=drt_work, drt_dev=drt_dev)
        epq_table = build_epq_action_table_combined_device(drt=drt_work, drt_dev=drt_dev, state_dev=state_dev)
        naux = None if l_full_d is None else int(l_full_d.shape[1])
        ws = CudaMrciHopWorkspace.auto(ncsf=int(drt_work.ncsf), nops=int(nops), naux=naux, sym_pair=False)

        def hop(x: np.ndarray) -> np.ndarray:
            x_d = cp.asarray(x, dtype=cp.float64)
            y_d = hop_cuda_epq_table(
                drt=drt_work,
                drt_dev=drt_dev,
                state_dev=state_dev,
                epq_table=epq_table,
                h_eff=h_eff_d,
                eri_mat_t=eri_mat_t_d,
                l_full=l_full_d,
                x=x_d,
                workspace=ws,
                sym_pair=False,
            )
            return np.asarray(cp.asnumpy(y_d), dtype=np.float64)

        return hop

    if DeviceDFMOIntegrals is not None and isinstance(eri, DeviceDFMOIntegrals):
        raise ValueError("DeviceDFMOIntegrals requires hop_backend='cuda'")

    def hop(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        if hop_map is not None:
            from asuka.mrci.projected_hop import projected_contract_h_csf_multi  # noqa: PLC0415

            y = projected_contract_h_csf_multi(
                mapping=hop_map,
                h1e=h1e,
                eri=eri,
                xs_sub=[x],
                precompute_epq_full=False,
                nthreads=int(contract_nthreads),
                blas_nthreads=contract_blas_nthreads,
                executor=None,
                workspace=None,
            )[0]
            return np.asarray(y, dtype=np.float64)
        if DFMOIntegrals is not None and isinstance(eri, DFMOIntegrals):
            from asuka.integrals.contract_df import contract_h_csf_multi_df as _contract_df  # noqa: PLC0415

            y = _contract_df(
                drt_work,
                h1e,
                eri,
                [x],
                precompute_epq=False,
                nthreads=int(contract_nthreads),
                blas_nthreads=contract_blas_nthreads,
                executor=None,
                workspace=None,
            )[0]
            return np.asarray(y, dtype=np.float64)

        from asuka.contract import contract_h_csf_multi as _contract_dense  # noqa: PLC0415

        y = _contract_dense(
            drt_work,
            h1e,
            eri,
            [x],
            precompute_epq=False,
            nthreads=int(contract_nthreads),
            blas_nthreads=contract_blas_nthreads,
            executor=None,
            workspace=None,
        )[0]
        return np.asarray(y, dtype=np.float64)

    return hop


def ic_mrcisd_kernel_multi(
    *,
    h1e: Any,
    eri: Any,
    n_act: int,
    n_virt: int,
    nelec: int,
    twos: int,
    ci_cas: Sequence[np.ndarray],
    nroots: int | None = None,
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
    solver: SolverMode = "auto",
    dense_nlab_max: int = 250,
    contract_nthreads: int = 1,
    contract_blas_nthreads: int | None = 1,
    precompute_epq: bool = True,
) -> ICMRCISDResultMulti:
    """Solve a shared-basis multi-root contracted ic-MRCISD problem."""

    ci_list = [np.asarray(v, dtype=np.float64).ravel() for v in ci_cas]
    if not ci_list:
        raise ValueError("ci_cas must be non-empty")
    nroots_i = len(ci_list) if nroots is None else int(nroots)
    if nroots_i < 1:
        raise ValueError("nroots must be >= 1")
    if nroots_i > len(ci_list):
        raise ValueError("nroots must be <= len(ci_cas)")

    backend_s = str(backend).strip().lower()
    if backend_s not in ("semi_direct", "rdm"):
        raise ValueError("backend must be 'semi_direct' or 'rdm'")

    contraction_s = str(contraction).strip().lower()
    sc_backend_s = str(sc_backend).strip().lower()
    hop_backend_s = str(hop_backend).strip().lower()
    if hop_backend_s not in ("fast", "augmented", "cuda"):
        raise ValueError("hop_backend must be 'fast', 'augmented', or 'cuda'")
    solver_s = str(solver).strip().lower()
    if solver_s not in ("davidson", "dense", "auto"):
        raise ValueError("solver must be 'davidson', 'dense', or 'auto'")

    n_act = int(n_act)
    n_virt = int(n_virt)
    nelec = int(nelec)
    twos = int(twos)
    max_virt_e = int(max_virt_e)
    norb_corr = int(n_act + n_virt)
    orbsym_corr = None
    if orbsym is not None:
        orbsym_corr = np.asarray(orbsym, dtype=np.int32).ravel()
        if int(orbsym_corr.size) != int(norb_corr):
            raise ValueError("orbsym must have length n_act + n_virt")

    components = [
        _build_ic_component_workspace(
            h1e=h1e,
            eri=eri,
            n_act=n_act,
            n_virt=n_virt,
            nelec=nelec,
            twos=twos,
            ci_cas=ci,
            orbsym_corr=orbsym_corr,
            wfnsym=wfnsym,
            max_virt_e=max_virt_e,
            hop_backend_s=hop_backend_s,
            contraction_s=contraction_s,
            sc_backend_s=sc_backend_s,
            symmetry=bool(symmetry),
            allow_same_external=bool(allow_same_external),
            allow_same_internal=bool(allow_same_internal),
            norm_min_singles=float(norm_min_singles),
            norm_min_doubles=float(norm_min_doubles),
            contract_nthreads=int(contract_nthreads),
            contract_blas_nthreads=contract_blas_nthreads,
            precompute_epq=bool(precompute_epq),
        )
        for ci in ci_list
    ]
    drt_work = components[0].drt_work
    if drt_work is None:
        raise RuntimeError("internal error: missing semi-direct DRT")
    hop = _build_uncontracted_hop(
        drt_work=drt_work,
        h1e=h1e,
        eri=eri,
        hop_backend_s=hop_backend_s,
        hop_map=getattr(components[0].ws, "hop_map", None),
        contract_nthreads=int(contract_nthreads),
        contract_blas_nthreads=contract_blas_nthreads,
    )
    union_ws = _ICUnionWorkspace(components=components, hop=hop)

    x0 = []
    for root in range(int(nroots_i)):
        guess = np.zeros((int(union_ws.nlab),), dtype=np.float64)
        guess[union_ws.reference_offsets[root]] = 1.0
        x0.append(guess)

    do_dense = bool(backend_s == "rdm") or solver_s == "dense" or (solver_s == "auto" and int(union_ws.nlab) <= int(dense_nlab_max))
    if do_dense:
        h = np.empty((int(union_ws.nlab), int(union_ws.nlab)), dtype=np.float64)
        s = np.empty((int(union_ws.nlab), int(union_ws.nlab)), dtype=np.float64)
        for j in range(int(union_ws.nlab)):
            ej = np.zeros((int(union_ws.nlab),), dtype=np.float64)
            ej[j] = 1.0
            h[:, j] = np.asarray(union_ws.sigma(ej), dtype=np.float64).ravel()
            s[:, j] = np.asarray(union_ws.overlap(ej), dtype=np.float64).ravel()
        s_sym = 0.5 * (s + s.T)
        evals_s, evecs_s = np.linalg.eigh(s_sym)
        keep = evals_s > float(s_tol)
        if not np.any(keep):
            raise np.linalg.LinAlgError("overlap matrix is numerically singular")
        t = evecs_s[:, keep] @ np.diag(1.0 / np.sqrt(evals_s[keep]))
        h_ortho = 0.5 * ((t.T @ h @ t) + (t.T @ h @ t).T)
        evals_h, evecs_h = np.linalg.eigh(h_ortho)
        nkeep = min(int(nroots_i), int(evals_h.size))
        e_dense = np.asarray(evals_h[:nkeep], dtype=np.float64)
        c_dense = np.asarray(t @ evecs_h[:, :nkeep], dtype=np.float64)
        c_roots = []
        rnorm = np.zeros((int(nroots_i),), dtype=np.float64)
        for root in range(int(nroots_i)):
            c = np.asarray(c_dense[:, root], dtype=np.float64).ravel()
            sc = np.asarray(s @ c, dtype=np.float64).ravel()
            snorm2 = float(np.dot(c, sc))
            if snorm2 <= 0.0:
                raise np.linalg.LinAlgError("dense generalized eigenvector has non-positive S-norm")
            c = c / np.sqrt(snorm2)
            r = h @ c - float(e_dense[root]) * (s @ c)
            rnorm[root] = float(np.linalg.norm(r))
            c_roots.append(np.ascontiguousarray(c, dtype=np.float64))
        res = GeneralizedDavidsonResultMulti(
            converged=np.ones((int(nroots_i),), dtype=bool),
            e=np.asarray(e_dense, dtype=np.float64),
            x=c_roots,
            niter=1,
            residual_norm=np.asarray(rnorm, dtype=np.float64),
        )
    else:
        precond = None
        try:
            diag_h = union_ws._compute_diag_precond()

            def precond(r: np.ndarray, e: float) -> np.ndarray:
                denom = diag_h - float(e)
                denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
                return np.asarray(r, dtype=np.float64) / denom

        except Exception:
            precond = None

        res = generalized_davidson(
            union_ws.sigma,
            union_ws.overlap,
            x0,
            precond=precond,
            tol=float(tol),
            max_cycle=int(max_cycle),
            max_space=int(max_space),
            s_tol=float(s_tol),
            nroots=int(nroots_i),
        )

    e = np.asarray(res.e, dtype=np.float64).ravel()[: int(nroots_i)].copy()
    c_roots = [np.ascontiguousarray(v, dtype=np.float64) for v in res.x[: int(nroots_i)]]
    overlap_ref_root = np.zeros((len(union_ws.reference_offsets), int(nroots_i)), dtype=np.float64)
    diagnostics: list[dict[str, float]] = []
    for root, c in enumerate(c_roots):
        rho = np.asarray(union_ws.overlap(c), dtype=np.float64).ravel()
        diag = {
            "nlab": float(int(union_ws.nlab)),
            "n_blocks": float(len(union_ws.block_slices)),
            "w_total": float(np.dot(c, rho)),
        }
        for k, offset in enumerate(union_ws.reference_offsets):
            ov = float(rho[int(offset)])
            overlap_ref_root[k, root] = float(ov * ov)
            diag[f"w_block_{k}"] = float(
                np.dot(
                    c[int(union_ws.block_slices[k][0]) : int(union_ws.block_slices[k][1])],
                    rho[int(union_ws.block_slices[k][0]) : int(union_ws.block_slices[k][1])],
                )
            )
        diagnostics.append(diag)

    return ICMRCISDResultMulti(
        converged=np.asarray(res.converged, dtype=bool).ravel()[: int(nroots_i)].copy(),
        e=e,
        e_tot=e + float(ecore),
        c=c_roots,
        spaces=[comp.spaces for comp in components],
        singles=[comp.singles for comp in components],
        doubles=[comp.doubles for comp in components],
        allow_same_internal=bool(allow_same_internal),
        backend=str(backend_s),
        drt_work=drt_work,
        block_slices=[(int(a), int(b)) for a, b in union_ws.block_slices],
        overlap_ref_root=np.asarray(overlap_ref_root, dtype=np.float64),
        niter=int(res.niter),
        residual_norm=np.asarray(res.residual_norm, dtype=np.float64).ravel()[: int(nroots_i)].copy(),
        diagnostics=diagnostics,
    )


def expand_ic_mrcisd_multi_root(
    ic_res: ICMRCISDResultMulti,
    *,
    ci_cas: Sequence[np.ndarray],
    root: int,
) -> tuple[Any, np.ndarray]:
    """Expand a shared-basis contracted root to the uncontracted CSF vector."""

    drt_work = getattr(ic_res, "drt_work", None)
    if drt_work is None:
        raise ValueError("ic_res.drt_work is required")
    root_i = int(root)
    coeff = np.asarray(ic_res.c[root_i], dtype=np.float64).ravel()
    if int(coeff.size) != int(sum(int(b) - int(a) for a, b in ic_res.block_slices)):
        raise ValueError("ic_res root coefficient length mismatch with block_slices")
    if int(len(ci_cas)) != int(len(ic_res.block_slices)):
        raise ValueError("ci_cas must have one reference vector per contracted basis block")

    from asuka.cuguga import build_drt  # noqa: PLC0415

    out = np.zeros((int(drt_work.ncsf),), dtype=np.float64)
    spaces0 = ic_res.spaces[0]
    n_act = int(getattr(spaces0, "n_internal"))
    n_virt = int(getattr(spaces0, "n_external"))
    orbsym_act = None
    orbsym_corr = getattr(spaces0, "orbsym", None)
    if orbsym_corr is not None:
        orbsym_act = np.asarray(orbsym_corr, dtype=np.int32).ravel()[: int(n_act)].tolist()
    wfnsym = None
    try:
        wfnsym = int(np.asarray(getattr(drt_work, "node_sym"))[int(drt_work.leaf)])
    except Exception:
        wfnsym = None

    for idx, ((start, stop), ci_ref, spaces, singles, doubles) in enumerate(
        zip(ic_res.block_slices, ci_cas, ic_res.spaces, ic_res.singles, ic_res.doubles, strict=True)
    ):
        if int(getattr(spaces, "n_internal")) != int(n_act) or int(getattr(spaces, "n_external")) != int(n_virt):
            raise ValueError("all blocks must share the same internal/external partition")
        ci_ref = _normalize_ci(np.asarray(ci_ref, dtype=np.float64))
        drt_cas = build_drt(
            norb=int(n_act),
            nelec=int(drt_work.nelec),
            twos_target=int(drt_work.twos_target),
            orbsym=orbsym_act,
            wfnsym=wfnsym,
        )
        _ci0, psi0, _ref_idx = embed_cas_ci_into_mrcisd(
            drt_cas=drt_cas,
            drt_mrci=drt_work,
            ci_cas=ci_ref,
            n_virt=int(n_virt),
        )
        c_blk = coeff[int(start) : int(stop)]
        if isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles):
            ws = ICRefSinglesDoublesSemiDirect(
                drt=drt_work,
                h1e=None,
                eri=None,
                psi0=psi0,
                singles=singles,
                doubles=doubles,
                precompute_epq=False,
            )
        elif isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles):
            ws = ICStronglyContractedSemiDirectOTF(
                drt=drt_work,
                h1e=None,
                eri=None,
                psi0=psi0,
                internal=np.arange(int(n_act), dtype=np.int32),
                singles=singles,
                doubles=doubles,
                allow_same_internal=bool(ic_res.allow_same_internal),
                precompute_epq=False,
            )
        else:  # pragma: no cover
            raise TypeError(f"unsupported contracted labels for block {idx}")
        out += np.asarray(ws.expand(c_blk), dtype=np.float64)
    return drt_work, np.asarray(out, dtype=np.float64, order="C")
