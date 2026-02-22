from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.soc.wigner import wigner_3j_twos

_M_ORDER = (-1, 0, +1)
_TM_ORDER = (-2, 0, +2)
_TM_TO_M_INDEX = {-2: 0, 0: 1, 2: 2}


@dataclass(frozen=True)
class SpinFreeState:
    twos: int
    energy: float
    drt: DRT
    ci: np.ndarray

    def __post_init__(self) -> None:
        if int(self.twos) < 0:
            raise ValueError("twos must be >= 0")
        if (int(self.twos) - int(self.drt.twos_target)) != 0:
            raise ValueError("SpinFreeState.twos must match drt.twos_target")
        ci = np.asarray(self.ci)
        if ci.ndim != 1:
            raise ValueError("ci must be a 1D array")
        if int(ci.size) != int(self.drt.ncsf):
            raise ValueError("ci has wrong length for drt")


@dataclass(frozen=True)
class SOCIntegrals:
    """Effective one-electron SOC integrals in the active MO basis."""

    h_xyz: np.ndarray | None = None  # (3, norb, norb), complex
    h_m: np.ndarray | None = None  # (3, norb, norb), complex, m in (-1,0,+1) order

    def __post_init__(self) -> None:
        if self.h_xyz is None and self.h_m is None:
            raise ValueError("must provide either h_xyz or h_m")
        if self.h_xyz is not None:
            arr = np.asarray(self.h_xyz)
            if arr.shape[0] != 3 or arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
                raise ValueError("h_xyz must have shape (3, norb, norb)")
        if self.h_m is not None:
            arr = np.asarray(self.h_m)
            if arr.shape[0] != 3 or arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
                raise ValueError("h_m must have shape (3, norb, norb)")


@dataclass(frozen=True)
class SpinFreeStateInteractionResult:
    """Spin-free state interaction result for a non-orthogonal state basis."""

    energies: np.ndarray  # (nroot,)
    mixing: np.ndarray  # (nstate, nroot), columns are eigenstates in the input basis
    overlap: np.ndarray  # (nstate, nstate)
    hamiltonian: np.ndarray  # (nstate, nstate)
    states: list[SpinFreeState]  # (nroot,), orthonormalized CI vectors with energies


@dataclass(frozen=True)
class SOCRASSIResult:
    """SOC-SI result after a spin-free RASSI-style orthonormalization step."""

    spinfree: SpinFreeStateInteractionResult
    so_energies: np.ndarray  # (nss,)
    so_vectors: np.ndarray  # (nss,nss), spin-component basis
    so_basis: list[tuple[int, int]]  # (state_index, tm=2M)


def soc_xyz_to_spherical(h_xyz: np.ndarray) -> np.ndarray:
    """Convert Cartesian SOC integrals (x,y,z) to spherical components (m=-1,0,+1)."""

    h = np.asarray(h_xyz)
    if h.ndim != 3 or h.shape[0] != 3 or h.shape[1] != h.shape[2]:
        raise ValueError("h_xyz must have shape (3, norb, norb)")
    hx = h[0].astype(np.complex128, copy=False)
    hy = h[1].astype(np.complex128, copy=False)
    hz = h[2].astype(np.complex128, copy=False)

    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    hm_p1 = -(hx + 1j * hy) * inv_sqrt2
    hm_0 = hz
    hm_m1 = (hx - 1j * hy) * inv_sqrt2
    return np.stack([hm_m1, hm_0, hm_p1], axis=0)


def build_si_basis(states: Sequence[SpinFreeState]) -> list[tuple[int, int]]:
    """Return SI basis as a list of (state_index, tm=2M)."""

    basis: list[tuple[int, int]] = []
    for i, st in enumerate(states):
        twos = int(st.twos)
        for tm in range(-twos, twos + 1, 2):
            basis.append((int(i), int(tm)))
    return basis


def build_si_hamiltonian_from_Gm(
    states: Sequence[SpinFreeState],
    Gm: np.ndarray,
    *,
    include_diag: bool = True,
    symmetrize: bool = True,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Assemble the SOC SI Hamiltonian from precomputed reduced couplings G_m(IJ).

    Parameters
    ----------
    states
        Spin-free states (each with a well-defined total spin `twos=2S`).
    Gm
        Array with shape (nstates, nstates, 3) storing G_m(IJ) for m in (-1,0,+1) order.
    include_diag
        If True, include spin-free energies on the diagonal.
    symmetrize
        If True, enforce H <- (H + H^\dagger)/2 as a Phase-1 safety check.
    """

    states = list(states)
    nstates = int(len(states))
    Gm = np.asarray(Gm, dtype=np.complex128)
    if Gm.shape != (nstates, nstates, 3):
        raise ValueError("Gm must have shape (nstates, nstates, 3)")

    basis = build_si_basis(states)
    dim = int(len(basis))
    H = np.zeros((dim, dim), dtype=np.complex128)

    if include_diag:
        for row, (i, _tm) in enumerate(basis):
            H[row, row] += complex(states[i].energy)

    for row, (i, tm_i) in enumerate(basis):
        twos_i = int(states[i].twos)
        for col, (j, tm_j) in enumerate(basis):
            tm = int(tm_i - tm_j)
            midx = _TM_TO_M_INDEX.get(tm)
            if midx is None:
                continue
            gm = Gm[i, j, midx]
            if gm == 0:
                continue

            # (-1)^(S_i - M_i) with doubled integers.
            phase = -1.0 if (((twos_i - int(tm_i)) // 2) & 1) else 1.0
            threej = wigner_3j_twos(twos_i, 2, int(states[j].twos), -int(tm_i), int(tm), int(tm_j))
            if threej == 0.0:
                continue
            # Our triplet-CSF engine convention for reduced RMEs implies an extra normalization
            # in the SI assembly: divide by sqrt(2*S_j+1) = sqrt(twos_j+1).
            inv_norm = 1.0 / math.sqrt(float(int(states[j].twos) + 1))
            H[row, col] += phase * threej * inv_norm * gm

    if symmetrize:
        H = 0.5 * (H + H.conj().T)
    return H, basis


def compute_si_adjoint_weights(
    states: Sequence[SpinFreeState],
    basis_index: Sequence[tuple[int, int]],
    si_vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SI adjoint weights w_I and eta_m(IJ) for a single SO-mixed SI eigenvector.

    This is the SI-level Hellmann–Feynman adjoint for the SOC-SI assembly used in this module:

        H_{I,tm_I;J,tm_J} = (-1)^{S_I-M_I}
                            ( S_I  1  S_J ; -M_I  m  M_J ) * G_m(IJ) / sqrt(2S_J+1)

    where `twos=2S`, `tm=2M`, and `m=(tm_I-tm_J)/2` with `tm_I-tm_J ∈ {-2,0,+2}`.

    For a normalized SI eigenvector v, these weights satisfy:

        v† (dH) v = Σ_I w_I dE_I  +  Σ_{I,J,m} eta_m(IJ) dG_m(IJ)

    with:
        w_I = Σ_M |v_{I,M}|^2
        eta_m(IJ) = Σ_{M_I} v_{I,M_I}* v_{J,M_I+m} (-1)^{S_I-M_I}
                    ( S_I  1  S_J ; -M_I  m  M_I+m ) / sqrt(2S_J+1)
    """

    states = list(states)
    nstates = int(len(states))
    basis_index = list(basis_index)

    v = np.asarray(si_vector, dtype=np.complex128).ravel()
    if int(v.size) != int(len(basis_index)):
        raise ValueError("si_vector has wrong length for basis_index")

    # Fast lookup for SI rows.
    row_of: dict[tuple[int, int], int] = {}
    for row, (i, tm) in enumerate(basis_index):
        key = (int(i), int(tm))
        if key in row_of:
            raise ValueError("basis_index contains duplicate (state,tm) entries")
        row_of[key] = int(row)

    w_state = np.zeros(nstates, dtype=np.float64)
    for row, (i, _tm) in enumerate(basis_index):
        w_state[int(i)] += float((v[row].conjugate() * v[row]).real)

    eta = np.zeros((3, nstates, nstates), dtype=np.complex128)

    for i, st_i in enumerate(states):
        twos_i = int(st_i.twos)
        for tm_i in range(-twos_i, twos_i + 1, 2):
            row_i = row_of.get((int(i), int(tm_i)))
            if row_i is None:
                continue
            v_i_conj = v[row_i].conjugate()
            if v_i_conj == 0:
                continue

            # (-1)^(S_i - M_i) with doubled integers.
            phase = -1.0 if (((twos_i - int(tm_i)) // 2) & 1) else 1.0

            for j, st_j in enumerate(states):
                twos_j = int(st_j.twos)
                for tm in _TM_ORDER:
                    tm_j = int(tm_i - tm)
                    if tm_j < -twos_j or tm_j > twos_j:
                        continue
                    if (tm_j - (-twos_j)) & 1:
                        continue
                    row_j = row_of.get((int(j), int(tm_j)))
                    if row_j is None:
                        continue
                    threej = wigner_3j_twos(twos_i, 2, twos_j, -int(tm_i), int(tm), int(tm_j))
                    if threej == 0.0:
                        continue
                    midx = _TM_TO_M_INDEX[int(tm)]
                    inv_norm = 1.0 / math.sqrt(float(twos_j + 1))
                    eta[midx, int(i), int(j)] += v_i_conj * v[row_j] * phase * threej * inv_norm

    return w_state, eta


def soc_state_interaction_from_Gm(
    states: Sequence[SpinFreeState],
    Gm: np.ndarray,
    *,
    include_diag: bool = True,
    symmetrize: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Diagonalize the SI Hamiltonian assembled from precomputed G_m(IJ)."""

    H, basis = build_si_hamiltonian_from_Gm(states, Gm, include_diag=include_diag, symmetrize=symmetrize)
    e, c = np.linalg.eigh(H)
    return e, c, basis


def _normalize_cuda_gm_strategy(strategy: str) -> str:
    mode = str(strategy).strip().lower()
    if mode not in ("auto", "apply_gemm", "direct_reduction"):
        raise ValueError("cuda_gm_strategy must be one of: 'auto', 'apply_gemm', 'direct_reduction'")
    return mode


def soc_state_interaction(
    states: Sequence[SpinFreeState],
    hso: SOCIntegrals,
    *,
    include_diag: bool = True,
    block_nops: int = 8,
    symmetrize: bool = True,
    backend: str = "cpu",
    cuda_threads: int = 128,
    cuda_sync: bool = True,
    cuda_fallback_to_cpu: bool = True,
    cuda_gm_strategy: str = "auto",
    cuda_gm_direct_max_nb_nk: int = 256,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Compute SOC SI by building G_m(IJ) from triplet TRDMs, then assembling/diagonalizing H.

    Phase 1 notes
    -------------
    Cross-DRT (S' = S±1) couplings are supported via a correctness-first DFS enumerator and
    are not yet optimized for large CI spaces.
    """

    states = list(states)
    if not states:
        raise ValueError("states is empty")

    norb = int(states[0].drt.norb)
    nelec = int(states[0].drt.nelec)
    for st in states:
        if int(st.drt.norb) != norb:
            raise ValueError("all states must have the same norb")
        if int(st.drt.nelec) != nelec:
            raise ValueError("all states must have the same nelec")

    if hso.h_m is not None:
        h_m = np.asarray(hso.h_m, dtype=np.complex128)
    else:
        h_m = soc_xyz_to_spherical(np.asarray(hso.h_xyz, dtype=np.complex128))
    if h_m.shape != (3, norb, norb):
        raise ValueError("SOC integrals have wrong shape for the states' norb")

    mode = str(backend).strip().lower()
    if mode not in ("cpu", "cuda", "auto"):
        raise ValueError("backend must be one of: 'cpu', 'cuda', 'auto'")
    gm_strategy = _normalize_cuda_gm_strategy(cuda_gm_strategy)
    cuda_gm_direct_max_nb_nk = int(cuda_gm_direct_max_nb_nk)
    if cuda_gm_direct_max_nb_nk < 1:
        raise ValueError("cuda_gm_direct_max_nb_nk must be >= 1")

    use_cuda = False
    if mode in ("cuda", "auto"):
        try:
            from asuka.soc.cuda_backend import has_soc_cuda  # noqa: PLC0415

            use_cuda = mode == "cuda" or bool(has_soc_cuda())
        except Exception:
            use_cuda = mode == "cuda"

    nstates = int(len(states))
    Gm = np.zeros((nstates, nstates, 3), dtype=np.complex128)

    by_drt: dict[DRT, list[int]] = {}
    for idx, st in enumerate(states):
        by_drt.setdefault(st.drt, []).append(int(idx))

    def _build_gm_cpu() -> None:
        from asuka.soc.trdm import trans_trdm1_triplet_all_streaming  # noqa: PLC0415

        for drt_bra, bra_ids in by_drt.items():
            bra_cis = [states[i].ci for i in bra_ids]
            bra_idx = np.asarray(bra_ids, dtype=np.int32)
            for drt_ket, ket_ids in by_drt.items():
                ket_cis = [states[j].ci for j in ket_ids]
                ket_idx = np.asarray(ket_ids, dtype=np.int32)
                u_blk = trans_trdm1_triplet_all_streaming(
                    drt_bra,
                    drt_ket,
                    bra_cis,
                    ket_cis,
                    block_nops=int(block_nops),
                )
                Gm_blk = np.einsum("mpq,ikpq->ikm", h_m, u_blk, optimize=True)
                Gm[np.ix_(bra_idx, ket_idx)] = Gm_blk

    if not use_cuda:
        _build_gm_cpu()
        return soc_state_interaction_from_Gm(states, Gm, include_diag=include_diag, symmetrize=symmetrize)

    try:
        import cupy as cp  # type: ignore
    except Exception as e:  # pragma: no cover
        if mode == "cuda" and not bool(cuda_fallback_to_cpu):
            raise RuntimeError("backend='cuda' requires CuPy") from e
        _build_gm_cpu()
        return soc_state_interaction_from_Gm(states, Gm, include_diag=include_diag, symmetrize=symmetrize)

    try:
        from asuka.soc.cuda_backend import (  # noqa: PLC0415
            _apply_contracted_triplet_all_m_cuda_inner,
            build_gm_soc_m_block_cuda,
            prepare_soc_device_context,
        )

        h_m_d = cp.asarray(h_m, dtype=cp.complex128)
        for drt_bra, bra_ids in by_drt.items():
            cbra = np.stack([np.asarray(states[i].ci, dtype=np.float64) for i in bra_ids], axis=1)
            cbra_d = cp.asarray(cbra, dtype=cp.float64)
            bra_idx = np.asarray(bra_ids, dtype=np.int32)
            for drt_ket, ket_ids in by_drt.items():
                cket = np.stack([np.asarray(states[j].ci, dtype=np.float64) for j in ket_ids], axis=1)
                ket_idx = np.asarray(ket_ids, dtype=np.int32)
                use_direct = False
                if gm_strategy == "direct_reduction":
                    use_direct = True
                elif gm_strategy == "auto":
                    use_direct = (len(bra_ids) * len(ket_ids)) <= int(cuda_gm_direct_max_nb_nk)

                if use_direct:
                    try:
                        gm_re_d, gm_im_d = build_gm_soc_m_block_cuda(
                            drt_bra,
                            drt_ket,
                            cbra,
                            cket,
                            h_m_d,
                            threads=int(cuda_threads),
                            sync=bool(cuda_sync),
                            use_epq_table_if_possible=True,
                        )
                        gm_blk = np.asarray(cp.asnumpy(gm_re_d), dtype=np.float64) + 1j * np.asarray(
                            cp.asnumpy(gm_im_d), dtype=np.float64
                        )  # (3, nb, nk)
                        Gm[np.ix_(bra_idx, ket_idx)] = np.transpose(gm_blk, (1, 2, 0))
                        continue
                    except Exception:
                        if mode == "cuda" and gm_strategy == "direct_reduction" and not bool(cuda_fallback_to_cpu):
                            raise

                # Fallback: per-ket apply + GEMM with hoisted device context
                ctx = prepare_soc_device_context(
                    drt_bra, drt_ket, h_m_d, threads=int(cuda_threads),
                )
                ncsf_bra = int(drt_bra.ncsf)
                out_re_buf = cp.empty((3, ncsf_bra), dtype=cp.float64)
                out_im_buf = cp.empty((3, ncsf_bra), dtype=cp.float64)
                cket_d = cp.ascontiguousarray(cp.asarray(cket, dtype=cp.float64))  # (ncsf_ket, nk)
                gm_accum = cp.empty((len(ket_ids), len(bra_ids), 3), dtype=cp.complex128)
                for k_local, j in enumerate(ket_ids):
                    _apply_contracted_triplet_all_m_cuda_inner(
                        ctx,
                        cket_d[:, k_local],
                        out_re=out_re_buf,
                        out_im=out_im_buf,
                        sync=False,
                    )
                    gm_accum[k_local] = cbra_d.T @ (out_re_buf + 1j * out_im_buf).T  # (nb, 3)
                gm_blk_h = np.asarray(cp.asnumpy(gm_accum), dtype=np.complex128)
                for k_local, j in enumerate(ket_ids):
                    for local_bra, i in enumerate(bra_ids):
                        Gm[int(i), int(j), :] = gm_blk_h[k_local, local_bra]
    except Exception:
        if mode == "cuda" and not bool(cuda_fallback_to_cpu):
            raise
        _build_gm_cpu()

    return soc_state_interaction_from_Gm(states, Gm, include_diag=include_diag, symmetrize=symmetrize)


def solve_spinfree_state_interaction(
    h: np.ndarray,
    s: np.ndarray,
    *,
    metric_tol: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the generalized spin-free SI problem ``H C = S C E`` with metric filtering.

    Returns eigenvalues ``E`` and S-orthonormal eigenvectors ``C`` such that:
    - ``C.T @ S @ C = I``
    - ``C.T @ H @ C = diag(E)``

    Near-null metric modes of ``S`` are dropped using ``metric_tol`` relative to
    the largest eigenvalue of ``S``.
    """

    h = np.asarray(h, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("h must be a square 2D array")
    if s.shape != h.shape:
        raise ValueError("s must have the same shape as h")

    metric_tol = float(metric_tol)
    if metric_tol < 0.0:
        raise ValueError("metric_tol must be >= 0")

    h = 0.5 * (h + h.T)
    s = 0.5 * (s + s.T)

    w, v = np.linalg.eigh(s)
    w = np.asarray(w, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w_max = float(np.max(w)) if int(w.size) else 0.0
    if not math.isfinite(w_max) or w_max <= 0.0:
        raise np.linalg.LinAlgError("overlap matrix S is singular (no positive metric modes)")

    keep = w > metric_tol * w_max
    if not np.any(keep):
        raise np.linalg.LinAlgError("overlap matrix S is singular under metric_tol filtering")

    w_keep = w[keep]
    v_keep = v[:, keep]
    x = v_keep / np.sqrt(w_keep)[None, :]

    h_ortho = x.T @ h @ x
    h_ortho = 0.5 * (h_ortho + h_ortho.T)

    evals, u = np.linalg.eigh(h_ortho)
    c = x @ u
    return np.asarray(evals, dtype=np.float64), np.asarray(c, dtype=np.float64)


def build_spinfree_state_interaction_matrices(
    states: Sequence[SpinFreeState],
    *,
    h1e: np.ndarray,
    eri: np.ndarray,
    ecore: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build spin-free (H,S) matrices for a set of CSF states sharing the same active basis."""

    states = list(states)
    if not states:
        raise ValueError("states is empty")

    drt0 = states[0].drt
    norb = int(drt0.norb)
    ncsf = int(drt0.ncsf)
    twos = int(states[0].twos)
    for st in states:
        if st.drt is not drt0:
            if int(st.drt.norb) != norb or int(st.drt.nelec) != int(drt0.nelec) or int(st.drt.twos_target) != int(drt0.twos_target):
                raise ValueError("all states must have the same DRT (norb/nelec/twos)")
        if int(st.twos) != twos:
            raise ValueError("all states must have the same twos")
        if np.asarray(st.ci).size != ncsf:
            raise ValueError("state CI has wrong length for DRT")

    h1e = np.asarray(h1e, dtype=np.float64)
    if h1e.shape != (norb, norb):
        raise ValueError("h1e must have shape (norb,norb)")
    eri = np.asarray(eri, dtype=np.float64)
    if eri.shape != (norb, norb, norb, norb):
        raise ValueError("eri must have shape (norb,norb,norb,norb)")

    ci_mat = np.stack([np.asarray(st.ci, dtype=np.float64) for st in states], axis=0)  # (nstate,ncsf)
    s = ci_mat @ ci_mat.T

    from asuka.contract import contract_h_csf  # noqa: PLC0415

    h_ci = np.stack(
        [contract_h_csf(drt0, h1e, eri, ci_mat[i], precompute_epq=True) for i in range(int(ci_mat.shape[0]))],
        axis=0,
    )  # (nstate,ncsf) rows are H|ci_i>
    h = ci_mat @ h_ci.T
    h += float(ecore) * s

    # Numerical cleanup.
    h = 0.5 * (h + h.T)
    s = 0.5 * (s + s.T)
    return h, s


def spinfree_state_interaction_from_states(
    states: Sequence[SpinFreeState],
    *,
    h1e: np.ndarray,
    eri: np.ndarray,
    ecore: float,
    metric_tol: float = 1e-14,
) -> SpinFreeStateInteractionResult:
    """Solve spin-free SI for a (possibly non-orthogonal) set of CSF states."""

    states = list(states)
    if not states:
        raise ValueError("states is empty")

    h, s = build_spinfree_state_interaction_matrices(states, h1e=h1e, eri=eri, ecore=float(ecore))
    evals, u = solve_spinfree_state_interaction(h, s, metric_tol=float(metric_tol))

    ci_in = np.stack([np.asarray(st.ci, dtype=np.float64) for st in states], axis=0)  # (nstate,ncsf)
    ci_out = u.T @ ci_in  # (nroot,ncsf)

    drt = states[0].drt
    twos = int(states[0].twos)
    out_states: list[SpinFreeState] = []
    for i in range(int(evals.size)):
        out_states.append(SpinFreeState(twos=twos, energy=float(evals[i]), drt=drt, ci=ci_out[i]))

    return SpinFreeStateInteractionResult(
        energies=np.asarray(evals, dtype=np.float64),
        mixing=np.asarray(u, dtype=np.float64),
        overlap=np.asarray(s, dtype=np.float64),
        hamiltonian=np.asarray(h, dtype=np.float64),
        states=out_states,
    )


def soc_state_interaction_rassi(
    states: Sequence[SpinFreeState],
    hso: SOCIntegrals,
    *,
    h1e: np.ndarray,
    eri: np.ndarray,
    ecore: float,
    metric_tol: float = 1e-14,
    include_diag: bool = True,
    block_nops: int = 8,
    symmetrize: bool = True,
) -> SOCRASSIResult:
    """SOC-SI after a spin-free RASSI-style orthonormalization step in the input state subspace."""

    sf = spinfree_state_interaction_from_states(
        states,
        h1e=h1e,
        eri=eri,
        ecore=float(ecore),
        metric_tol=float(metric_tol),
    )
    e_so, c_so, basis = soc_state_interaction(
        sf.states,
        hso,
        include_diag=bool(include_diag),
        block_nops=int(block_nops),
        symmetrize=bool(symmetrize),
    )
    return SOCRASSIResult(
        spinfree=sf,
        so_energies=np.asarray(e_so, dtype=np.complex128),
        so_vectors=np.asarray(c_so, dtype=np.complex128),
        so_basis=list(basis),
    )
