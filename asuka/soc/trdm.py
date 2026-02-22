from __future__ import annotations

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.soc.si import SpinFreeState
from asuka.soc.triplet_op import apply_triplet_pq


def _normalize_backend(backend: str) -> str:
    mode = str(backend).strip().lower()
    if mode not in ("cpu", "cuda", "auto"):
        raise ValueError("backend must be one of: 'cpu', 'cuda', 'auto'")
    return mode


def _tri_ok_twos(tj1: int, tj2: int, tj3: int) -> bool:
    tj1 = int(tj1)
    tj2 = int(tj2)
    tj3 = int(tj3)
    if tj1 < 0 or tj2 < 0 or tj3 < 0:
        return False
    if (tj1 + tj2 + tj3) & 1:
        return False
    if tj3 < abs(tj1 - tj2) or tj3 > tj1 + tj2:
        return False
    return True


def _resolve_soc_cuda_apply(backend: str):
    """Return ``(cp_module, cuda_apply_fn)`` for SOC CUDA, or ``(None, None)``."""

    mode = _normalize_backend(backend)
    if mode == "cpu":
        return None, None

    from asuka.soc.cuda_backend import has_soc_cuda  # noqa: PLC0415

    if mode == "auto" and not bool(has_soc_cuda()):
        return None, None

    import cupy as cp  # type: ignore  # noqa: PLC0415
    from asuka.soc.cuda_backend import apply_contracted_triplet_all_m_cuda  # noqa: PLC0415

    return cp, apply_contracted_triplet_all_m_cuda


def trans_trdm1_triplet_streaming(
    drt_bra: DRT,
    drt_ket: DRT,
    ci_bra: np.ndarray,
    ci_ket: np.ndarray,
    *,
    block_nops: int = 8,
) -> np.ndarray:
    """Compute the reduced triplet transition 1-body density u[p,q] (rank-1, reduced).

    Convention
    ----------
    Returns `u[p,q] = <bra|| T^(1)_{q p} ||ket>` (adjoint qp indexing, matching
    `asuka.rdm.stream.trans_rdm1_*` style).
    """

    if int(drt_bra.norb) != int(drt_ket.norb):
        raise ValueError("drt_bra and drt_ket must have the same norb")

    norb = int(drt_ket.norb)
    ncsf_bra = int(drt_bra.ncsf)
    ncsf_ket = int(drt_ket.ncsf)

    bra = np.asarray(ci_bra, dtype=np.float64).ravel()
    ket = np.asarray(ci_ket, dtype=np.float64).ravel()
    if int(bra.size) != ncsf_bra or int(ket.size) != ncsf_ket:
        raise ValueError("ci_bra/ci_ket have wrong length for the provided DRTs")

    nops = norb * norb
    block_nops = int(block_nops)
    if block_nops < 1:
        raise ValueError("block_nops must be >= 1")
    block_nops = min(block_nops, nops)

    u = np.zeros((norb, norb), dtype=np.float64)

    # Total-spin selection rule for rank-1: if violated, the entire TRDM is zero.
    if not _tri_ok_twos(int(drt_bra.twos_target), 2, int(drt_ket.twos_target)):
        return u

    for i0 in range(0, nops, block_nops):
        i1 = min(nops, i0 + block_nops)
        blk = int(i1 - i0)
        Tblk = np.empty((blk, ncsf_bra), dtype=np.float64)
        for t in range(blk):
            pq = int(i0 + t)
            p_out = pq // norb
            q_out = pq - p_out * norb
            if p_out == q_out:
                Tblk[t].fill(0.0)
                continue
            # u[p,q] = <bra||T_{q p}||ket>
            apply_triplet_pq(drt_bra, drt_ket, ket, p=int(q_out), q=int(p_out), out=Tblk[t])
        u_flat = Tblk @ bra
        for t in range(blk):
            pq = int(i0 + t)
            p_out = pq // norb
            q_out = pq - p_out * norb
            u[p_out, q_out] = float(u_flat[t])

    return u


def trans_trdm1_triplet_all_streaming(
    drt_bra: DRT,
    drt_ket: DRT,
    ci_list_bra: list[np.ndarray],
    ci_list_ket: list[np.ndarray] | None = None,
    *,
    block_nops: int = 8,
) -> np.ndarray:
    """Compute reduced triplet transition 1-RDMs for many bras/kets with one operator sweep per ket.

    Convention matches `trans_trdm1_triplet_streaming`:
    - u[bra, ket, p, q] = <bra|| T^(1)_{q p} ||ket>
    """

    if int(drt_bra.norb) != int(drt_ket.norb):
        raise ValueError("drt_bra and drt_ket must have the same norb")

    norb = int(drt_ket.norb)
    nops = norb * norb
    ncsf_bra = int(drt_bra.ncsf)
    ncsf_ket = int(drt_ket.ncsf)

    if ci_list_ket is None:
        ci_list_ket = ci_list_bra

    if len(ci_list_bra) == 0 or len(ci_list_ket) == 0:
        raise ValueError("ci_list_bra/ci_list_ket must be non-empty")

    cbra = np.stack([np.asarray(c, dtype=np.float64).ravel() for c in ci_list_bra], axis=1)
    cket_list = [np.asarray(c, dtype=np.float64).ravel() for c in ci_list_ket]
    if int(cbra.shape[0]) != ncsf_bra:
        raise ValueError("ci_list_bra vectors have wrong length")
    for c in cket_list:
        if int(c.size) != ncsf_ket:
            raise ValueError("ci_list_ket vectors have wrong length")

    block_nops = int(block_nops)
    if block_nops < 1:
        raise ValueError("block_nops must be >= 1")
    block_nops = min(block_nops, nops)

    u = np.zeros((cbra.shape[1], len(cket_list), norb, norb), dtype=np.float64)

    if not _tri_ok_twos(int(drt_bra.twos_target), 2, int(drt_ket.twos_target)):
        return u

    # Workspace A holds a block of rows of T_{q p} |ket> in the bra CSF basis.
    A = np.empty((block_nops, ncsf_bra), dtype=np.float64)

    for k, cket in enumerate(cket_list):
        V = np.empty((nops, cbra.shape[1]), dtype=np.float64)
        for i0 in range(0, nops, block_nops):
            i1 = min(nops, i0 + block_nops)
            nb = int(i1 - i0)
            for t, pq in enumerate(range(i0, i1)):
                p_out = pq // norb
                q_out = pq - p_out * norb
                if p_out == q_out:
                    A[t].fill(0.0)
                else:
                    apply_triplet_pq(drt_bra, drt_ket, cket, p=int(q_out), q=int(p_out), out=A[t])
            V[i0:i1] = A[:nb] @ cbra

        u[:, k] = V.T.reshape(cbra.shape[1], norb, norb)

    return u


def build_rho_soc_m_streaming(
    states: list[SpinFreeState],
    eta: np.ndarray,
    *,
    block_nops: int = 8,
    eps: float = 0.0,
    backend: str = "cpu",
    cuda_threads: int = 128,
    cuda_sync: bool = True,
    cuda_fallback_to_cpu: bool = True,
) -> np.ndarray:
    """Build effective reduced SOC densities rho^(m)[p,q] without storing all u^{IJ}.

    Definitions
    ----------
    Given the SI adjoint weights eta_m(IJ) (m in {-1,0,+1}) and reduced triplet TRDMs
    u^{IJ}[p,q] = <I||T^(1)_{q p}||J>, build:

        rho^(m)[p,q] = Σ_{I,J} eta_m(IJ) * u^{IJ}[p,q].

    Returns
    -------
    rho_m
        Complex array with shape (3, norb, norb) in m=(-1,0,+1) order.
    """

    if not states:
        raise ValueError("states is empty")

    norb = int(states[0].drt.norb)
    nelec = int(states[0].drt.nelec)
    for st in states:
        if int(st.drt.norb) != norb:
            raise ValueError("all states must have the same norb")
        if int(st.drt.nelec) != nelec:
            raise ValueError("all states must have the same nelec")

    nstates = int(len(states))
    eta = np.asarray(eta, dtype=np.complex128)
    if eta.shape != (3, nstates, nstates):
        raise ValueError("eta must have shape (3, nstates, nstates)")

    mode = _normalize_backend(backend)
    rho = np.zeros((3, norb, norb), dtype=np.complex128)

    by_drt: dict[DRT, list[int]] = {}
    for idx, st in enumerate(states):
        by_drt.setdefault(st.drt, []).append(int(idx))

    cp_mod = None
    cuda_rho_block = None
    cuda_enabled = False
    if mode in ("cuda", "auto"):
        try:
            import cupy as cp  # type: ignore  # noqa: PLC0415
            from asuka.soc.cuda_backend import (  # noqa: PLC0415
                build_rho_soc_m_block_cuda,
                has_soc_cuda,
            )

            if mode == "cuda" or bool(has_soc_cuda()):
                cp_mod = cp  # type: ignore[assignment]
                cuda_rho_block = build_rho_soc_m_block_cuda
                cuda_enabled = True
        except Exception:
            if mode == "cuda" and not bool(cuda_fallback_to_cpu):
                raise

    for drt_bra, bra_ids in by_drt.items():
        bra_cis = [states[i].ci for i in bra_ids]
        bra_idx = np.asarray(bra_ids, dtype=np.int32)
        c_bra_mat = np.stack([np.asarray(states[i].ci, dtype=np.float64) for i in bra_ids], axis=1)
        for drt_ket, ket_ids in by_drt.items():
            ket_cis = [states[j].ci for j in ket_ids]
            ket_idx = np.asarray(ket_ids, dtype=np.int32)
            c_ket_mat = np.stack([np.asarray(states[j].ci, dtype=np.float64) for j in ket_ids], axis=1)

            eta_sub = eta[:, bra_idx, :][:, :, ket_idx]  # (3, nb, nk)
            if float(eps) > 0.0 and float(np.max(np.abs(eta_sub))) <= float(eps):
                continue

            if cuda_enabled:
                try:
                    rho_re_d, rho_im_d = cuda_rho_block(
                        drt_bra,
                        drt_ket,
                        c_bra_mat,
                        c_ket_mat,
                        eta_sub,
                        threads=int(cuda_threads),
                        sync=bool(cuda_sync),
                        use_epq_table_if_possible=True,
                    )
                    rho += np.asarray(cp_mod.asnumpy(rho_re_d), dtype=np.float64) + 1j * np.asarray(
                        cp_mod.asnumpy(rho_im_d), dtype=np.float64
                    )
                    continue
                except Exception:
                    if mode == "cuda" and not bool(cuda_fallback_to_cpu):
                        raise
                    cuda_enabled = False

            u_blk = trans_trdm1_triplet_all_streaming(
                drt_bra,
                drt_ket,
                bra_cis,
                ket_cis,
                block_nops=int(block_nops),
            )  # (nb, nk, norb, norb)
            rho += np.einsum("mij,ijpq->mpq", eta_sub, u_blk, optimize=True)

    return rho


def apply_contracted_triplet_all_m(
    drt_bra: DRT,
    drt_ket: DRT,
    ci_ket: np.ndarray,
    h_m: np.ndarray,
    *,
    block_nops: int = 8,
    backend: str = "cpu",
    cuda_threads: int = 128,
    cuda_sync: bool = True,
    cuda_fallback_to_cpu: bool = True,
) -> np.ndarray:
    """Apply the triplet operator contracted with spherical SOC integrals.

    Computes, in the bra CSF basis:

        out_m[m] = Σ_{p,q} h_m[m,p,q] * (T^(1)_{q p} |ket>)

    where `m` is in (-1,0,+1) order and we use the qp convention
    `u[p,q] = <bra||T^(1)_{q p}||ket>`.

    Returns
    -------
    out
        Complex array with shape (3, ncsf_bra).
    """

    backend = _normalize_backend(backend)
    if backend in ("cuda", "auto"):
        try:
            from asuka.soc.cuda_backend import (  # noqa: PLC0415
                apply_contracted_triplet_all_m_cuda,
                has_soc_cuda,
            )

            if backend == "cuda" or has_soc_cuda():
                try:  # optional dependency
                    import cupy as cp  # type: ignore
                except Exception as e:  # pragma: no cover
                    if backend == "cuda" and not bool(cuda_fallback_to_cpu):
                        raise RuntimeError("CuPy is required for backend='cuda'") from e
                else:
                    y_re, y_im = apply_contracted_triplet_all_m_cuda(
                        drt_bra,
                        drt_ket,
                        c_ket=ci_ket,
                        h_m=h_m,
                        threads=int(cuda_threads),
                        sync=bool(cuda_sync),
                        use_epq_table_if_possible=True,
                        fallback_to_cpu=bool(cuda_fallback_to_cpu),
                        block_nops_cpu=int(block_nops),
                    )
                    y = cp.asnumpy(y_re) + 1j * cp.asnumpy(y_im)
                    return np.asarray(y, dtype=np.complex128)
        except Exception:
            if backend == "cuda" and not bool(cuda_fallback_to_cpu):
                raise

    if int(drt_bra.norb) != int(drt_ket.norb):
        raise ValueError("drt_bra and drt_ket must have the same norb")

    norb = int(drt_ket.norb)
    nops = norb * norb
    ncsf_bra = int(drt_bra.ncsf)
    ncsf_ket = int(drt_ket.ncsf)

    h_m = np.asarray(h_m, dtype=np.complex128)
    if h_m.shape != (3, norb, norb):
        raise ValueError("h_m must have shape (3, norb, norb)")

    ket = np.asarray(ci_ket, dtype=np.float64).ravel()
    if int(ket.size) != ncsf_ket:
        raise ValueError("ci_ket has wrong length for drt_ket")

    out = np.zeros((3, ncsf_bra), dtype=np.complex128)

    # Total-spin selection rule for rank-1: if violated, the entire contracted apply is zero.
    if not _tri_ok_twos(int(drt_bra.twos_target), 2, int(drt_ket.twos_target)):
        return out

    block_nops = int(block_nops)
    if block_nops < 1:
        raise ValueError("block_nops must be >= 1")
    block_nops = min(block_nops, nops)

    A = np.empty((block_nops, ncsf_bra), dtype=np.float64)
    p_idx = np.empty(block_nops, dtype=np.int32)
    q_idx = np.empty(block_nops, dtype=np.int32)

    for i0 in range(0, nops, block_nops):
        i1 = min(nops, i0 + block_nops)
        nb = int(i1 - i0)

        for t in range(nb):
            pq = int(i0 + t)
            p_out = pq // norb
            q_out = pq - p_out * norb
            p_idx[t] = int(p_out)
            q_idx[t] = int(q_out)

            if p_out == q_out:
                A[t].fill(0.0)
            else:
                apply_triplet_pq(drt_bra, drt_ket, ket, p=int(q_out), q=int(p_out), out=A[t])

        h_blk = h_m[:, p_idx[:nb], q_idx[:nb]]  # (3, nb)
        out += h_blk @ A[:nb]

    return out


def build_ci_rhs_soc_bra_term(
    states: list[SpinFreeState],
    eta: np.ndarray,
    h_m: np.ndarray,
    *,
    block_nops: int = 8,
    eps: float = 0.0,
    backend: str = "cpu",
    cuda_threads: int = 128,
    cuda_sync: bool = True,
    cuda_fallback_to_cpu: bool = True,
) -> list[np.ndarray]:
    """Build the CI-space RHS vectors for the SOC term (bra-side contribution only).

    For SOC-SI gradients we define:

      F_SOC = Σ_{I,J,m} eta_m(IJ) * G_m(IJ)
      G_m(IJ) = c_I^T Ω_m^{IJ} c_J
      Ω_m^{IJ} = Σ_{p,q} h_m[m,p,q] * T^(1)_{q p}

    This function computes the contribution from varying the *bra* CI vector c_K:

      b_K^bra = ∂F_SOC/∂c_K  (bra term)
             = Σ_{J,m} eta_m(KJ) * Ω_m^{KJ} c_J

    The transpose/ket-side contribution (K appearing as the ket index) is handled separately once the
    reduced-RME symmetry phase is pinned down (or a reverse-apply kernel is implemented).

    Returns
    -------
    rhs_bra
        List of length nstates; rhs_bra[K] has shape (ncsf_K,) and dtype complex128.
    """

    if not states:
        raise ValueError("states is empty")

    norb = int(states[0].drt.norb)
    nelec = int(states[0].drt.nelec)
    for st in states:
        if int(st.drt.norb) != norb:
            raise ValueError("all states must have the same norb")
        if int(st.drt.nelec) != nelec:
            raise ValueError("all states must have the same nelec")

    nstates = int(len(states))
    eta = np.asarray(eta, dtype=np.complex128)
    if eta.shape != (3, nstates, nstates):
        raise ValueError("eta must have shape (3, nstates, nstates)")

    h_m = np.asarray(h_m, dtype=np.complex128)
    if h_m.shape != (3, norb, norb):
        raise ValueError("h_m must have shape (3, norb, norb)")

    mode = _normalize_backend(backend)

    rhs: list[np.ndarray] = [np.zeros(int(st.drt.ncsf), dtype=np.complex128) for st in states]

    by_drt: dict[DRT, list[int]] = {}
    for idx, st in enumerate(states):
        by_drt.setdefault(st.drt, []).append(int(idx))

    cp_mod = None
    cuda_enabled = False
    try:
        if mode in ("cuda", "auto"):
            from asuka.soc.cuda_backend import has_soc_cuda  # noqa: PLC0415

            if mode == "cuda" or bool(has_soc_cuda()):
                import cupy as _cp  # type: ignore  # noqa: PLC0415

                cp_mod = _cp
                cuda_enabled = True
    except Exception:
        if mode == "cuda" and not bool(cuda_fallback_to_cpu):
            raise

    for drt_bra, bra_ids in by_drt.items():
        bra_idx = np.asarray(bra_ids, dtype=np.int32)
        ncsf_bra = int(drt_bra.ncsf)

        if cuda_enabled:
            try:
                from asuka.soc.cuda_backend import (  # noqa: PLC0415
                    _apply_contracted_triplet_all_m_cuda_inner,
                    prepare_soc_device_context,
                )

                rhs_blk_d = cp_mod.zeros((len(bra_ids), ncsf_bra), dtype=cp_mod.complex128)
                out_re_buf = cp_mod.empty((3, ncsf_bra), dtype=cp_mod.float64)
                out_im_buf = cp_mod.empty((3, ncsf_bra), dtype=cp_mod.float64)
                for drt_ket, ket_ids in by_drt.items():
                    ket_idx = np.asarray(ket_ids, dtype=np.int32)
                    eta_sub = eta[:, bra_idx, :][:, :, ket_idx]  # (3, nb, nk)
                    if float(eps) > 0.0 and float(np.max(np.abs(eta_sub))) <= float(eps):
                        continue
                    nk = int(len(ket_ids))
                    ctx = prepare_soc_device_context(
                        drt_bra, drt_ket, h_m, threads=int(cuda_threads),
                    )
                    cket_all_d = cp_mod.ascontiguousarray(
                        cp_mod.asarray(
                            np.stack([np.asarray(states[int(j)].ci, dtype=np.float64) for j in ket_ids], axis=1),
                            dtype=cp_mod.float64,
                        )
                    )  # (ncsf_ket, nk)
                    y_blk_d = cp_mod.empty((3, ncsf_bra, nk), dtype=cp_mod.complex128)
                    for k_local in range(nk):
                        _apply_contracted_triplet_all_m_cuda_inner(
                            ctx,
                            cket_all_d[:, k_local],
                            out_re=out_re_buf,
                            out_im=out_im_buf,
                            sync=False,
                        )
                        y_blk_d[:, :, k_local] = out_re_buf + 1j * out_im_buf
                    eta_sub_d = cp_mod.asarray(eta_sub, dtype=cp_mod.complex128)
                    rhs_blk_d += cp_mod.einsum("mbk,mxk->bx", eta_sub_d, y_blk_d)
                rhs_blk = np.asarray(cp_mod.asnumpy(rhs_blk_d), dtype=np.complex128)
            except Exception:
                if mode == "cuda" and not bool(cuda_fallback_to_cpu):
                    raise
                cuda_enabled = False
                rhs_blk = np.zeros((len(bra_ids), ncsf_bra), dtype=np.complex128)
                for drt_ket, ket_ids in by_drt.items():
                    for j in ket_ids:
                        eta_sub = eta[:, bra_idx, int(j)]  # (3, nb)
                        if float(eps) > 0.0 and float(np.max(np.abs(eta_sub))) <= float(eps):
                            continue
                        vec_m = apply_contracted_triplet_all_m(
                            drt_bra,
                            drt_ket,
                            states[int(j)].ci,
                            h_m,
                            block_nops=int(block_nops),
                            backend=str(mode),
                            cuda_threads=int(cuda_threads),
                            cuda_sync=bool(cuda_sync),
                            cuda_fallback_to_cpu=bool(cuda_fallback_to_cpu),
                        )  # (3, ncsf_bra)
                        rhs_blk += np.einsum("mb,mx->bx", eta_sub, vec_m, optimize=True)
        else:
            rhs_blk = np.zeros((len(bra_ids), ncsf_bra), dtype=np.complex128)
            for drt_ket, ket_ids in by_drt.items():
                for j in ket_ids:
                    eta_sub = eta[:, bra_idx, int(j)]  # (3, nb)
                    if float(eps) > 0.0 and float(np.max(np.abs(eta_sub))) <= float(eps):
                        continue
                    vec_m = apply_contracted_triplet_all_m(
                        drt_bra,
                        drt_ket,
                        states[int(j)].ci,
                        h_m,
                        block_nops=int(block_nops),
                        backend=str(mode),
                        cuda_threads=int(cuda_threads),
                        cuda_sync=bool(cuda_sync),
                        cuda_fallback_to_cpu=bool(cuda_fallback_to_cpu),
                    )  # (3, ncsf_bra)
                    rhs_blk += np.einsum("mb,mx->bx", eta_sub, vec_m, optimize=True)

        for local, k in enumerate(bra_ids):
            rhs[int(k)] = rhs_blk[local]

    return rhs


def _triplet_transpose_scale_twos(*, twos_bra: int, twos_ket: int) -> float:
    """Scale relating transpose vs swapped-bra/ket triplet reduced matrices (qp convention).

    Empirically (and consistent with reduced-tensor algebra in our SI convention), the CSF-level reduced
    matrices satisfy:

      (T^(1)_{q p}[bra<-ket])^T = scale * T^(1)_{p q}[ket<-bra]

    with:
      scale = +1                                if twos_bra == twos_ket
      scale = -sqrt((twos_ket+1)/(twos_bra+1))  if |twos_bra - twos_ket| == 2
      scale = 0                                 otherwise (rank-1 triangle violation)
    """

    twos_bra = int(twos_bra)
    twos_ket = int(twos_ket)
    if twos_bra < 0 or twos_ket < 0:
        return 0.0
    if twos_bra == twos_ket:
        return 1.0
    if abs(twos_bra - twos_ket) == 2:
        return -float(np.sqrt((twos_ket + 1) / (twos_bra + 1)))
    return 0.0


def build_ci_rhs_soc_full(
    states: list[SpinFreeState],
    eta: np.ndarray,
    h_m: np.ndarray,
    *,
    block_nops: int = 8,
    eps: float = 0.0,
    backend: str = "cpu",
    cuda_threads: int = 128,
    cuda_sync: bool = True,
    cuda_fallback_to_cpu: bool = True,
) -> list[np.ndarray]:
    """Build the full CI-space RHS vectors for the SOC term (bra + ket contributions).

    This returns b_K = ∂/∂c_K [ Σ_{I,J,m} eta_m(IJ) * G_m(IJ) ] for real CI coefficients c_K, where:

      G_m(IJ) = c_I^T Ω_m^{IJ} c_J
      Ω_m^{IJ} = Σ_{p,q} h_m[m,p,q] * T^(1)_{q p}

    The result splits into:

      b_K = Σ_{J,m} eta_m(KJ) Ω_m^{KJ} c_J   +   Σ_{I,m} eta_m(IK) (Ω_m^{IK})^T c_I

    The transpose term is computed using a reduced-RME symmetry factor consistent with the current CSF engine
    implementation; see `_triplet_transpose_scale_twos`.

    Returns
    -------
    rhs
        List of length nstates; rhs[K] has shape (ncsf_K,) and dtype complex128.
    """

    mode = _normalize_backend(backend)

    rhs = build_ci_rhs_soc_bra_term(
        states,
        eta,
        h_m,
        block_nops=int(block_nops),
        eps=float(eps),
        backend=str(mode),
        cuda_threads=int(cuda_threads),
        cuda_sync=bool(cuda_sync),
        cuda_fallback_to_cpu=bool(cuda_fallback_to_cpu),
    )

    if not states:
        return rhs

    norb = int(states[0].drt.norb)
    h_m = np.asarray(h_m, dtype=np.complex128)
    if h_m.shape != (3, norb, norb):
        raise ValueError("h_m must have shape (3, norb, norb)")
    h_m_T = np.swapaxes(h_m, 1, 2)

    nstates = int(len(states))
    eta = np.asarray(eta, dtype=np.complex128)
    if eta.shape != (3, nstates, nstates):
        raise ValueError("eta must have shape (3, nstates, nstates)")

    by_drt: dict[DRT, list[int]] = {}
    for idx, st in enumerate(states):
        by_drt.setdefault(st.drt, []).append(int(idx))

    cp_mod = None
    cuda_enabled = False
    try:
        if mode in ("cuda", "auto"):
            from asuka.soc.cuda_backend import has_soc_cuda  # noqa: PLC0415

            if mode == "cuda" or bool(has_soc_cuda()):
                import cupy as _cp  # type: ignore  # noqa: PLC0415

                cp_mod = _cp
                cuda_enabled = True
    except Exception:
        if mode == "cuda" and not bool(cuda_fallback_to_cpu):
            raise

    for drt_target, ket_ids in by_drt.items():
        ket_idx = np.asarray(ket_ids, dtype=np.int32)
        twos_ket = np.asarray([int(states[k].twos) for k in ket_ids], dtype=np.int32)
        ncsf_target = int(drt_target.ncsf)

        if cuda_enabled:
            try:
                from asuka.soc.cuda_backend import (  # noqa: PLC0415
                    _apply_contracted_triplet_all_m_cuda_inner,
                    prepare_soc_device_context,
                )

                rhs_target_d = cp_mod.asarray(
                    np.stack([rhs[int(k)] for k in ket_ids], axis=0),
                    dtype=cp_mod.complex128,
                )  # (nk, ncsf_target)
                out_re_buf = cp_mod.empty((3, ncsf_target), dtype=cp_mod.float64)
                out_im_buf = cp_mod.empty((3, ncsf_target), dtype=cp_mod.float64)
                for drt_i, bra_ids in by_drt.items():
                    bra_idx = np.asarray(bra_ids, dtype=np.int32)
                    eta_blk = eta[:, bra_idx, :][:, :, ket_idx]  # (3, ni, nk)
                    if float(eps) > 0.0 and float(np.max(np.abs(eta_blk))) <= float(eps):
                        continue
                    ni = int(len(bra_ids))
                    scales = np.empty((ni, int(ket_idx.size)), dtype=np.float64)
                    for i_local, i in enumerate(bra_ids):
                        twos_i = int(states[int(i)].twos)
                        scales[i_local, :] = np.asarray(
                            [_triplet_transpose_scale_twos(twos_bra=twos_i, twos_ket=int(tk)) for tk in twos_ket],
                            dtype=np.float64,
                        )
                    eta_scaled = eta_blk * scales[None, :, :]
                    if float(np.max(np.abs(eta_scaled))) == 0.0:
                        continue

                    ctx = prepare_soc_device_context(
                        drt_target, drt_i, h_m_T, threads=int(cuda_threads),
                    )
                    cbra_all_d = cp_mod.ascontiguousarray(
                        cp_mod.asarray(
                            np.stack([np.asarray(states[int(i)].ci, dtype=np.float64) for i in bra_ids], axis=1),
                            dtype=cp_mod.float64,
                        )
                    )  # (ncsf_i, ni)
                    y_blk_d = cp_mod.empty((3, ncsf_target, ni), dtype=cp_mod.complex128)
                    for i_local in range(ni):
                        _apply_contracted_triplet_all_m_cuda_inner(
                            ctx,
                            cbra_all_d[:, i_local],
                            out_re=out_re_buf,
                            out_im=out_im_buf,
                            sync=False,
                        )
                        y_blk_d[:, :, i_local] = out_re_buf + 1j * out_im_buf

                    eta_scaled_d = cp_mod.asarray(eta_scaled, dtype=cp_mod.complex128)
                    rhs_target_d += cp_mod.einsum("mik,mxi->kx", eta_scaled_d, y_blk_d)

                rhs_target = np.asarray(cp_mod.asnumpy(rhs_target_d), dtype=np.complex128)
                for k_local, k in enumerate(ket_ids):
                    rhs[int(k)] = rhs_target[k_local]
            except Exception:
                if mode == "cuda" and not bool(cuda_fallback_to_cpu):
                    raise
                cuda_enabled = False
                for drt_i, bra_ids in by_drt.items():
                    for i in bra_ids:
                        eta_sub = eta[:, int(i), ket_idx]  # (3, nk)
                        if float(eps) > 0.0 and float(np.max(np.abs(eta_sub))) <= float(eps):
                            continue

                        twos_i = int(states[int(i)].twos)
                        scales = np.asarray(
                            [_triplet_transpose_scale_twos(twos_bra=twos_i, twos_ket=int(tk)) for tk in twos_ket],
                            dtype=np.float64,
                        )
                        if float(np.max(np.abs(scales))) == 0.0:
                            continue

                        vec_m = apply_contracted_triplet_all_m(
                            drt_target,
                            drt_i,
                            states[int(i)].ci,
                            h_m_T,
                            block_nops=int(block_nops),
                            backend=str(mode),
                            cuda_threads=int(cuda_threads),
                            cuda_sync=bool(cuda_sync),
                            cuda_fallback_to_cpu=bool(cuda_fallback_to_cpu),
                        )  # (3, ncsf_target)

                        eta_scaled = eta_sub * scales[None, :]
                        contrib = np.einsum("mk,mx->kx", eta_scaled, vec_m, optimize=True)  # (nk, ncsf_target)
                        for k_local, k in enumerate(ket_ids):
                            rhs[int(k)] += contrib[k_local]
        else:
            for drt_i, bra_ids in by_drt.items():
                for i in bra_ids:
                    eta_sub = eta[:, int(i), ket_idx]  # (3, nk)
                    if float(eps) > 0.0 and float(np.max(np.abs(eta_sub))) <= float(eps):
                        continue

                    twos_i = int(states[int(i)].twos)
                    scales = np.asarray(
                        [_triplet_transpose_scale_twos(twos_bra=twos_i, twos_ket=int(tk)) for tk in twos_ket],
                        dtype=np.float64,
                    )
                    if float(np.max(np.abs(scales))) == 0.0:
                        continue

                    vec_m = apply_contracted_triplet_all_m(
                        drt_target,
                        drt_i,
                        states[int(i)].ci,
                        h_m_T,
                        block_nops=int(block_nops),
                        backend=str(mode),
                        cuda_threads=int(cuda_threads),
                        cuda_sync=bool(cuda_sync),
                        cuda_fallback_to_cpu=bool(cuda_fallback_to_cpu),
                    )  # (3, ncsf_target)

                    eta_scaled = eta_sub * scales[None, :]
                    contrib = np.einsum("mk,mx->kx", eta_scaled, vec_m, optimize=True)  # (nk, ncsf_target)
                    for k_local, k in enumerate(ket_ids):
                        rhs[int(k)] += contrib[k_local]

    return rhs
