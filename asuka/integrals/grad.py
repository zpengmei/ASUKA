from __future__ import annotations

import time
import numpy as np
from typing import Any

from asuka.integrals.df_adjoint import chol_lower_adjoint, df_whiten_adjoint

def _choose_aux_block_naux(*, nao: int, naux: int, target_bytes: int) -> int:
    """Choose a conservative aux block size for (qblk,nao,nao) float64 chunks."""

    nao = int(nao)
    naux = int(naux)
    if nao <= 0 or naux <= 0:
        raise ValueError("invalid nao/naux")
    target_bytes = int(target_bytes)
    if target_bytes <= 0:
        raise ValueError("target_bytes must be > 0")

    bytes_per_Q = int(nao) * int(nao) * 8
    return max(1, min(int(naux), int(target_bytes // max(1, bytes_per_Q))))


def _contract_bar_with_B_streamed(
    bar_L_ao: np.ndarray,
    B: Any,
    *,
    backend: str,
    aux_block_naux: int,
) -> float:
    """Return sum_{Q,mu,nu} bar_L[Q,mu,nu] * B[mu,nu,Q] with aux streaming."""

    naux, nao, nao2 = map(int, bar_L_ao.shape)
    if nao != nao2:
        raise ValueError("bar_L_ao must have shape (naux, nao, nao)")

    aux_block_naux = max(1, int(aux_block_naux))

    backend_s = str(backend).strip().lower()
    if backend_s == "cpu":
        BQ = np.transpose(np.asarray(B, dtype=np.float64), (2, 0, 1))
        if tuple(BQ.shape) != (naux, nao, nao):
            raise ValueError("B shape mismatch")
        acc = 0.0
        for q0 in range(0, naux, aux_block_naux):
            q1 = min(naux, int(q0) + int(aux_block_naux))
            acc += float(np.sum(bar_L_ao[int(q0) : int(q1)] * BQ[int(q0) : int(q1)]))
        return float(acc)

    if backend_s == "cuda":
        try:
            import cupy as cp  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError("backend='cuda' requires CuPy") from e

        BQ = cp.transpose(B, (2, 0, 1))
        if tuple(map(int, BQ.shape)) != (naux, nao, nao):
            raise ValueError("B shape mismatch")
        acc = cp.float64(0.0)
        for q0 in range(0, naux, aux_block_naux):
            q1 = min(naux, int(q0) + int(aux_block_naux))
            bar_q = cp.asarray(bar_L_ao[int(q0) : int(q1)], dtype=cp.float64)
            acc += cp.sum(bar_q * BQ[int(q0) : int(q1)])
            bar_q = None
        return float(acc.item())

    raise ValueError("backend must be 'cpu' or 'cuda'")


def contract_3c_deriv(
    mol,
    auxmol,
    d_3c: np.ndarray,
) -> np.ndarray:
    """Legacy helper (analytic DF derivatives).

    This function used to rely on PySCF derivative integral kernels.
    The standalone path in this repo uses finite differences on the whitened DF
    factors instead; see `compute_df_gradient_contributions_tiled(...)`.
    """

    _ = (mol, auxmol, d_3c)
    raise NotImplementedError(
        "contract_3c_deriv is not available in the standalone runtime. "
        "Use compute_df_gradient_contributions_tiled(...), which performs a "
        "finite-difference contraction on cuERI-built DF factors."
    )


def contract_2c_deriv(
    mol,
    auxmol,
    d_2c: np.ndarray,
) -> np.ndarray:
    """Legacy helper (analytic DF derivatives).

    See `contract_3c_deriv` for rationale. The standalone gradient contraction
    path uses finite differences on the whitened DF factors.
    """

    _ = (mol, auxmol, d_2c)
    raise NotImplementedError(
        "contract_2c_deriv is not available in the standalone runtime. "
        "Use compute_df_gradient_contributions_tiled(...), which performs a "
        "finite-difference contraction on cuERI-built DF factors."
    )


def compute_df_gradient_contributions_analytic_packed_bases(
    ao_basis,
    aux_basis,
    *,
    atom_coords_bohr: np.ndarray,
    B_ao: np.ndarray,
    bar_L_ao: np.ndarray,
    L_chol: Any | None = None,
    backend: str = "cpu",
    df_threads: int = 0,
    profile: dict | None = None,
) -> np.ndarray:
    """Compute DF 2e nuclear-gradient contraction analytically (no finite differences).

    This replaces the legacy FD-on-B contraction by:

      1) Backprop through whitening + metric Cholesky to obtain adjoints
         (bar_X, bar_V) for the *unwhitened* DF primitives:
           - X(μ,ν,P) = (μν|P)
           - V(P,Q)   = (P|Q)
      2) Contract analytic integral derivatives:
           dE/dR = (dX/dR)⋅bar_X + (dV/dR)⋅bar_V

    Parameters
    ----------
    L_chol
        Optional pre-computed lower Cholesky factor of the aux metric V.
        When provided, the gradient uses *exactly the same* L as the forward
        DF build, eliminating run-to-run non-determinism from recomputing
        V on GPU (where 1-ULP kernel non-determinism is amplified by an
        ill-conditioned V into ~1e-2 gradient errors).

    Notes
    -----
    - `backend="cpu"` uses cuERI CPU derivative tiles.
    - `backend="cuda"` uses cuERI CUDA derivative contraction kernels (requires the cuERI CUDA extension).
    - The returned gradient is in Eh/Bohr and has shape (natm, 3).
    """

    t0_total = time.perf_counter() if profile is not None else 0.0

    backend_s = str(backend).strip().lower()
    if backend_s not in {"cpu", "cuda"}:
        raise ValueError("backend must be 'cpu' or 'cuda'")

    atom_coords_bohr = np.asarray(atom_coords_bohr, dtype=np.float64)
    if atom_coords_bohr.ndim != 2 or atom_coords_bohr.shape[1] != 3:
        raise ValueError("atom_coords_bohr must have shape (natm, 3)")
    natm = int(atom_coords_bohr.shape[0])
    if natm <= 0:
        return np.zeros((0, 3), dtype=np.float64)

    if backend_s == "cuda":
        try:
            import cupy as cp  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError("backend='cuda' requires CuPy") from e

        B_ao = cp.ascontiguousarray(cp.asarray(B_ao, dtype=cp.float64))
        bar_L_ao = cp.ascontiguousarray(cp.asarray(bar_L_ao, dtype=cp.float64))
    else:
        B_ao = np.asarray(B_ao, dtype=np.float64, order="C")
        bar_L_ao = np.asarray(bar_L_ao, dtype=np.float64, order="C")

    if B_ao.ndim != 3:
        raise ValueError("B_ao must have shape (nao, nao, naux)")
    nao0, nao1, naux = map(int, B_ao.shape)
    if nao0 != nao1:
        raise ValueError("B_ao must have shape (nao, nao, naux)")

    if bar_L_ao.shape != (naux, nao0, nao0):
        raise ValueError("bar_L_ao must have shape (naux, nao, nao)")

    # Build SA shell->atom maps for AO and aux bases (used for final accumulation).
    from asuka.integrals.int1e_cart import shell_to_atom_map  # noqa: PLC0415

    ao_shell_atom = shell_to_atom_map(ao_basis, atom_coords_bohr=atom_coords_bohr)
    aux_shell_atom = shell_to_atom_map(aux_basis, atom_coords_bohr=atom_coords_bohr)

    t_shell_atom = time.perf_counter() if profile is not None else 0.0

    # Build combined (AO + aux + dummy) basis and shell-pair tables (same as CPU DF build).
    from asuka.integrals.cueri_df_cpu import _build_df_combined_basis_and_shell_pairs  # noqa: PLC0415
    from asuka.cueri.cart import ncart  # noqa: PLC0415
    from asuka.cueri.pair_tables_cpu import build_pair_tables_cpu  # noqa: PLC0415

    basis_all, sp_all, nsp_ao, n_shell_ao, n_shell_aux = _build_df_combined_basis_and_shell_pairs(ao_basis, aux_basis)
    aux_sp0 = int(nsp_ao)

    t_combined_basis = time.perf_counter() if profile is not None else 0.0

    pt = build_pair_tables_cpu(basis_all, sp_all, threads=int(df_threads), profile=None)

    t_pair_tables = time.perf_counter() if profile is not None else 0.0

    # ---- metric V and Cholesky L ----
    # We need the aux-metric Cholesky factor L for df_whiten_adjoint. For CPU
    # we build V via the CPU ERI tiles (same as the DF builder). For CUDA we
    # build V (and L) directly on GPU via cuERI DF primitives.

    shell_cxyz_all = np.asarray(basis_all.shell_cxyz, dtype=np.float64, order="C")
    shell_l_all = np.asarray(basis_all.shell_l, dtype=np.int32, order="C")
    shell_prim_start_all = np.asarray(basis_all.shell_prim_start, dtype=np.int32, order="C")
    shell_nprim_all = np.asarray(basis_all.shell_nprim, dtype=np.int32, order="C")
    shell_ao_start_all = np.asarray(basis_all.shell_ao_start, dtype=np.int32, order="C")
    prim_exp_all = np.asarray(basis_all.prim_exp, dtype=np.float64, order="C")

    sp_A_all = np.asarray(sp_all.sp_A, dtype=np.int32, order="C")
    sp_B_all = np.asarray(sp_all.sp_B, dtype=np.int32, order="C")
    sp_pair_start_all = np.asarray(sp_all.sp_pair_start, dtype=np.int32, order="C")
    sp_npair_all = np.asarray(sp_all.sp_npair, dtype=np.int32, order="C")

    pair_eta_all = np.asarray(pt.pair_eta, dtype=np.float64, order="C")
    pair_Px_all = np.asarray(pt.pair_Px, dtype=np.float64, order="C")
    pair_Py_all = np.asarray(pt.pair_Py, dtype=np.float64, order="C")
    pair_Pz_all = np.asarray(pt.pair_Pz, dtype=np.float64, order="C")
    pair_cK_all = np.asarray(pt.pair_cK, dtype=np.float64, order="C")

    aux_shell_l = np.asarray(aux_basis.shell_l, dtype=np.int32, order="C").ravel()
    aux_shell_ao_start = np.asarray(aux_basis.shell_ao_start, dtype=np.int32, order="C").ravel()

    by_l: dict[int, list[int]] = {}
    for sh in range(int(n_shell_aux)):
        by_l.setdefault(int(aux_shell_l[sh]), []).append(int(sh))

    # Pre-build spCD arrays per aux-l group to reduce Python overhead in the
    # derivative contractions (hot loops over AO shell pairs).
    spCD_by_l: dict[int, np.ndarray] = {}
    shells_by_l: dict[int, np.ndarray] = {}
    for lq, q_shells in by_l.items():
        q_arr = np.asarray(q_shells, dtype=np.int32)
        shells_by_l[int(lq)] = q_arr
        spCD_by_l[int(lq)] = (aux_sp0 + q_arr).astype(np.int32, copy=False)

    if backend_s == "cpu":
        # Build V(P,Q) on CPU via ERI tiles (P*1|Q*1), matching the CPU DF builder.
        try:
            from asuka.cueri import _eri_rys_cpu as _ext  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError("CPU ERI extension is required for analytic DF gradient contraction on backend='cpu'") from e

        eri_batch = getattr(_ext, "eri_rys_tile_cart_sp_batch_cy", None)
        if eri_batch is None:  # pragma: no cover
            raise RuntimeError("CPU ERI extension is missing batch tile entry points; rebuild the extension")

        V = np.zeros((naux, naux), dtype=np.float64)
        for psh in range(int(n_shell_aux)):
            lp = int(aux_shell_l[psh])
            nP = int(ncart(lp))
            p0 = int(aux_shell_ao_start[psh])
            spAB = int(aux_sp0 + psh)

            for lq, spCD in spCD_by_l.items():
                nQ = int(ncart(int(lq)))
                q_shells = shells_by_l[int(lq)]
                q_list = [int(q) for q in q_shells if int(q) <= int(psh)]
                if not q_list:
                    continue

                spCD_sub = (aux_sp0 + np.asarray(q_list, dtype=np.int32)).astype(np.int32, copy=False)
                tiles = eri_batch(
                    shell_cxyz_all,
                    shell_l_all,
                    sp_A_all,
                    sp_B_all,
                    sp_pair_start_all,
                    sp_npair_all,
                    pair_eta_all,
                    pair_Px_all,
                    pair_Py_all,
                    pair_Pz_all,
                    pair_cK_all,
                    int(spAB),
                    spCD_sub,
                    int(df_threads),
                )

                for t, qsh in enumerate(q_list):
                    q0 = int(aux_shell_ao_start[int(qsh)])
                    block = np.asarray(tiles[int(t)], dtype=np.float64, order="C").reshape((nP, nQ))
                    V[p0 : p0 + nP, q0 : q0 + nQ] = block
                    if int(qsh) != int(psh):
                        V[q0 : q0 + nQ, p0 : p0 + nP] = block.T

        t_metric_build = time.perf_counter() if profile is not None else 0.0

        if L_chol is not None:
            L = np.asarray(L_chol, dtype=np.float64, order="C")
            V = L @ L.T
        else:
            V = 0.5 * (V + V.T)
            L = np.linalg.cholesky(V)

        t_metric_chol = time.perf_counter() if profile is not None else 0.0
    else:
        # Build V and its Cholesky on GPU. This avoids requiring the CPU ERI
        # extension for the CUDA backend and keeps df_whiten_adjoint on-device.
        import cupy as cp  # noqa: PLC0415

        from asuka.cueri import df as cueri_df  # noqa: PLC0415

        if L_chol is not None:
            # Reuse the forward-pass Cholesky factor — avoids recomputing V
            # and eliminates non-determinism from GPU 2c2e kernel + ill-conditioned V.
            L = cp.ascontiguousarray(cp.asarray(L_chol, dtype=cp.float64))
            # Still need V for the 2c2e derivative adjoint; reconstruct from L.
            V = L @ L.T
            if profile is not None:
                cp.cuda.get_current_stream().synchronize()
                t_metric_build = time.perf_counter()
                t_metric_chol = t_metric_build
            else:
                t_metric_build = 0.0
                t_metric_chol = 0.0
        else:
            V = cueri_df.metric_2c2e_basis(aux_basis, stream=None, backend="gpu_rys", mode="warp", threads=256)
            # Regularize: AutoAux bases can have cond(V) > 1e17, amplifying
            # 1-ULP GPU kernel non-determinism into ~1e-2 gradient errors
            # through the Cholesky factor. Apply the same diagonal shift as
            # the forward DF builder (asuka/integrals/cueri_df.py).
            _v_diag = cp.diag(V)
            _v_shift = max(float(cp.max(cp.abs(_v_diag))) * 1e-14, 1e-12)
            V[cp.diag_indices_from(V)] += _v_shift
            if profile is not None:
                cp.cuda.get_current_stream().synchronize()
                t_metric_build = time.perf_counter()
            else:
                t_metric_build = 0.0

            L = cp.linalg.cholesky(V)
            if profile is not None:
                cp.cuda.get_current_stream().synchronize()
                t_metric_chol = time.perf_counter()
            else:
                t_metric_chol = 0.0

    # ---- DF adjoints: (bar_B, L) -> (bar_X, bar_V) ----
    # bar_L_ao is stored as (naux,nao,nao); df_whiten_adjoint expects (nao,nao,naux).
    if backend_s == "cpu":
        bar_B = np.transpose(bar_L_ao, (1, 2, 0))
        bar_B = np.asarray(bar_B, dtype=np.float64, order="C")
        B_ao_c = np.asarray(B_ao, dtype=np.float64, order="C")
        L_c = np.asarray(L, dtype=np.float64, order="C")
    else:
        import cupy as cp  # noqa: PLC0415

        bar_B = cp.ascontiguousarray(bar_L_ao.transpose((1, 2, 0)))
        B_ao_c = cp.ascontiguousarray(B_ao)
        L_c = cp.ascontiguousarray(L)

    bar_X, bar_L = df_whiten_adjoint(B_ao_c, bar_B, L_c)
    bar_V = chol_lower_adjoint(L_c, bar_L)

    if profile is not None and backend_s == "cuda":
        import cupy as cp  # noqa: PLC0415

        cp.cuda.get_current_stream().synchronize()
    t_df_adjoint = time.perf_counter() if profile is not None else 0.0

    # X(μ,ν,P) is symmetric in (μ,ν); enforce symmetric adjoint.
    bar_X = 0.5 * (bar_X + bar_X.transpose((1, 0, 2)))
    if backend_s == "cpu":
        bar_X = np.asarray(bar_X, dtype=np.float64, order="C")
        bar_V = np.asarray(bar_V, dtype=np.float64, order="C")
    else:
        import cupy as cp  # noqa: PLC0415

        bar_X = cp.ascontiguousarray(bar_X)
        bar_V = cp.ascontiguousarray(bar_V)

    bar_X_flat = bar_X.reshape((nao0 * nao0, naux))

    # ---- contracted integral derivatives ----
    # CPU path accumulates directly into a NumPy array. CUDA path accumulates on
    # device to avoid per-batch host sync/copies (a major performance killer).
    grad = np.zeros((natm, 3), dtype=np.float64)
    grad_dev = None

    if backend_s == "cpu":
        # 3c2e: loop AO shell pairs and batch over aux shells grouped by l.
        for spAB in range(int(nsp_ao)):
            shA = int(sp_A_all[int(spAB)])
            shB = int(sp_B_all[int(spAB)])
            fac = 2.0 if shA != shB else 1.0
            atomA = int(ao_shell_atom[int(shA)])
            atomB = int(ao_shell_atom[int(shB)])

            for lq, spCD_batch in spCD_by_l.items():
                q_shells = shells_by_l[int(lq)]
                out_batch = _ext.df_int3c2e_deriv_contracted_cart_sp_batch_cy(
                    shell_cxyz_all,
                    shell_prim_start_all,
                    shell_nprim_all,
                    shell_l_all,
                    shell_ao_start_all,
                    prim_exp_all,
                    sp_A_all,
                    sp_B_all,
                    sp_pair_start_all,
                    sp_npair_all,
                    pair_eta_all,
                    pair_Px_all,
                    pair_Py_all,
                    pair_Pz_all,
                    pair_cK_all,
                    int(spAB),
                    spCD_batch,
                    int(nao0),
                    bar_X_flat,
                )

                # Accumulate: center C varies per aux shell.
                for t, qsh in enumerate(q_shells):
                    atomC = int(aux_shell_atom[int(qsh)])
                    grad[atomA] += fac * out_batch[int(t), 0, :]
                    grad[atomB] += fac * out_batch[int(t), 1, :]
                    grad[atomC] += fac * out_batch[int(t), 2, :]
    else:
        try:
            import cupy as cp  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError("backend='cuda' requires CuPy") from e

        try:
            from asuka.cueri import _cueri_cuda_ext as _ext_cuda  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "cuERI CUDA extension is required for analytic DF gradient contraction; "
                "build via `python -m asuka.cueri.build_cuda_ext`"
            ) from e

        # Upload static tables once.
        shell_cx_dev = cp.ascontiguousarray(cp.asarray(shell_cxyz_all[:, 0], dtype=cp.float64))
        shell_cy_dev = cp.ascontiguousarray(cp.asarray(shell_cxyz_all[:, 1], dtype=cp.float64))
        shell_cz_dev = cp.ascontiguousarray(cp.asarray(shell_cxyz_all[:, 2], dtype=cp.float64))

        shell_prim_start_dev = cp.ascontiguousarray(cp.asarray(shell_prim_start_all, dtype=cp.int32))
        shell_nprim_dev = cp.ascontiguousarray(cp.asarray(shell_nprim_all, dtype=cp.int32))
        shell_ao_start_dev = cp.ascontiguousarray(cp.asarray(shell_ao_start_all, dtype=cp.int32))
        shell_l_host = np.asarray(shell_l_all, dtype=np.int32).ravel()

        prim_exp_dev = cp.ascontiguousarray(cp.asarray(prim_exp_all, dtype=cp.float64))

        sp_A_dev = cp.ascontiguousarray(cp.asarray(sp_A_all, dtype=cp.int32))
        sp_B_dev = cp.ascontiguousarray(cp.asarray(sp_B_all, dtype=cp.int32))
        sp_pair_start_dev = cp.ascontiguousarray(cp.asarray(sp_pair_start_all, dtype=cp.int32))
        sp_npair_dev = cp.ascontiguousarray(cp.asarray(sp_npair_all, dtype=cp.int32))

        pair_eta_dev = cp.ascontiguousarray(cp.asarray(pair_eta_all, dtype=cp.float64))
        pair_Px_dev = cp.ascontiguousarray(cp.asarray(pair_Px_all, dtype=cp.float64))
        pair_Py_dev = cp.ascontiguousarray(cp.asarray(pair_Py_all, dtype=cp.float64))
        pair_Pz_dev = cp.ascontiguousarray(cp.asarray(pair_Pz_all, dtype=cp.float64))
        pair_cK_dev = cp.ascontiguousarray(cp.asarray(pair_cK_all, dtype=cp.float64))

        bar_X_dev = cp.ascontiguousarray(cp.asarray(bar_X_flat.reshape(-1), dtype=cp.float64))
        bar_V_dev = cp.ascontiguousarray(cp.asarray(bar_V.reshape(-1), dtype=cp.float64))

        spCD_by_l_dev = {int(lq): cp.ascontiguousarray(cp.asarray(spCD_batch, dtype=cp.int32)) for lq, spCD_batch in spCD_by_l.items()}

        # Combined AO+aux shell→atom map for batched atomicAdd kernel.
        shell_atom_all = np.concatenate([
            np.asarray(ao_shell_atom, dtype=np.int32),
            np.asarray(aux_shell_atom, dtype=np.int32),
        ])
        shell_atom_dev = cp.ascontiguousarray(cp.asarray(shell_atom_all, dtype=cp.int32))

        # Group AO shell pairs by (la, lb) angular momentum class.
        spAB_by_lab: dict[tuple[int, int], list[int]] = {}
        for spAB_i in range(int(nsp_ao)):
            shA = int(sp_A_all[int(spAB_i)])
            shB = int(sp_B_all[int(spAB_i)])
            key = (int(shell_l_host[shA]), int(shell_l_host[shB]))
            if key not in spAB_by_lab:
                spAB_by_lab[key] = []
            spAB_by_lab[key].append(spAB_i)
        spAB_by_lab_dev = {
            lab: cp.ascontiguousarray(cp.asarray(indices, dtype=cp.int32))
            for lab, indices in spAB_by_lab.items()
        }

        grad_dev = cp.zeros((natm, 3), dtype=cp.float64)

        threads = 256
        stream_ptr = int(cp.cuda.get_current_stream().ptr)

        # 3c2e derivative contraction — batched: one kernel per (la,lb,lq) class.
        grad_dev_flat = grad_dev.reshape(-1)  # (natm*3,) view for atomicAdd kernel
        _3c_class_times: list[tuple[int, int, int, int, int, float]] = []
        for (la_, lb_), spAB_class_dev in spAB_by_lab_dev.items():
            n_spAB = int(spAB_class_dev.shape[0])
            for lq, spCD_dev in spCD_by_l_dev.items():
                nt = int(spCD_dev.shape[0])
                if nt == 0 or n_spAB == 0:
                    continue
                if profile is not None:
                    cp.cuda.get_current_stream().synchronize()
                    _t_cls0 = time.perf_counter()
                _ext_cuda.df_int3c2e_deriv_contracted_cart_allsp_atomgrad_inplace_device(
                    spAB_class_dev,
                    spCD_dev,
                    sp_A_dev,
                    sp_B_dev,
                    sp_pair_start_dev,
                    sp_npair_dev,
                    shell_cx_dev,
                    shell_cy_dev,
                    shell_cz_dev,
                    shell_prim_start_dev,
                    shell_nprim_dev,
                    shell_ao_start_dev,
                    prim_exp_dev,
                    pair_eta_dev,
                    pair_Px_dev,
                    pair_Py_dev,
                    pair_Pz_dev,
                    pair_cK_dev,
                    int(nao0),
                    int(naux),
                    int(la_),
                    int(lb_),
                    int(lq),
                    bar_X_dev,
                    shell_atom_dev,
                    grad_dev_flat,
                    int(threads),
                    int(stream_ptr),
                    False,
                )
                if profile is not None:
                    cp.cuda.get_current_stream().synchronize()
                    _3c_class_times.append((int(la_), int(lb_), int(lq), n_spAB, nt, time.perf_counter() - _t_cls0))

    if backend_s == "cuda":
        # Synchronize so that async 3c2e kernels complete before 2c2e metric section.
        import cupy as cp  # noqa: PLC0415

        cp.cuda.get_current_stream().synchronize()

    t_3c_deriv = time.perf_counter() if profile is not None else 0.0

    if backend_s == "cpu":
        # 2c2e metric term: loop aux shell pairs (Q<=P), batch over q shells grouped by l.
        for psh in range(int(n_shell_aux)):
            lp = int(aux_shell_l[int(psh)])
            spAB = int(aux_sp0 + psh)
            atomP = int(aux_shell_atom[int(psh)])
            for lq, q_shells in shells_by_l.items():
                q_list = [int(q) for q in q_shells if int(q) <= int(psh)]
                if not q_list:
                    continue
                spCD_batch = (aux_sp0 + np.asarray(q_list, dtype=np.int32)).astype(np.int32, copy=False)
                out_batch = _ext.df_metric_2c2e_deriv_contracted_cart_sp_batch_cy(
                    shell_cxyz_all,
                    shell_prim_start_all,
                    shell_nprim_all,
                    shell_l_all,
                    shell_ao_start_all,
                    prim_exp_all,
                    sp_A_all,
                    sp_B_all,
                    sp_pair_start_all,
                    sp_npair_all,
                    pair_eta_all,
                    pair_Px_all,
                    pair_Py_all,
                    pair_Pz_all,
                    pair_cK_all,
                    int(spAB),
                    spCD_batch,
                    int(nao0),
                    bar_V,
                )

                for t, qsh in enumerate(q_list):
                    fac = 2.0 if int(qsh) != int(psh) else 1.0
                    atomQ = int(aux_shell_atom[int(qsh)])
                    grad[atomP] += fac * out_batch[int(t), 0, :]
                    grad[atomQ] += fac * out_batch[int(t), 1, :]
    else:
        import cupy as cp  # noqa: PLC0415
        from asuka.cueri import _cueri_cuda_ext as _ext_cuda  # noqa: PLC0415

        threads = 256
        stream_ptr = int(cp.cuda.get_current_stream().ptr)

        # 2c2e metric derivative contraction — one kernel per (la, lc) class.
        #
        # Uses the allsp_atomgrad kernel which takes arrays of spAB and spCD,
        # uses a 2D grid (ntasks, n_spAB), and accumulates via atomicAdd
        # directly into grad_dev.  Processing the full (P,Q) matrix (not just
        # the upper triangle) is correct because each off-diagonal pair
        # (P,Q) and (Q,P) together contribute the same as fac=2 on the upper
        # triangle.
        #
        # For C10H12 / cc-pVDZ this reduces ~2100 kernel launches to ~16.

        # Reuse the combined (AO + aux) shell→atom map already on device.
        # sp_A[spAB] returns a global shell index, so the atom lookup must
        # cover both AO and aux shells — using shell_atom_dev (built above).

        # Group aux shell-pair indices by angular momentum.
        _spAB_by_l: dict[int, Any] = {}
        _spCD_by_l: dict[int, Any] = {}
        for lval in sorted(shells_by_l.keys()):
            q_arr = np.asarray(shells_by_l[int(lval)], dtype=np.int32)
            sp_arr = (aux_sp0 + q_arr).astype(np.int32, copy=False)
            _spAB_by_l[int(lval)] = cp.ascontiguousarray(cp.asarray(sp_arr, dtype=cp.int32))
            _spCD_by_l[int(lval)] = _spAB_by_l[int(lval)]  # same arrays

        grad_dev_flat = grad_dev.reshape(-1)

        for lp, spAB_class_dev in _spAB_by_l.items():
            n_spAB = int(spAB_class_dev.shape[0])
            for lq, spCD_class_dev in _spCD_by_l.items():
                ntasks_lq = int(spCD_class_dev.shape[0])
                if n_spAB == 0 or ntasks_lq == 0:
                    continue
                _ext_cuda.df_metric_2c2e_deriv_contracted_cart_allsp_atomgrad_inplace_device(
                    spAB_class_dev,
                    spCD_class_dev,
                    sp_A_dev,
                    sp_B_dev,
                    sp_pair_start_dev,
                    sp_npair_dev,
                    shell_cx_dev,
                    shell_cy_dev,
                    shell_cz_dev,
                    shell_prim_start_dev,
                    shell_nprim_dev,
                    shell_ao_start_dev,
                    prim_exp_dev,
                    pair_eta_dev,
                    pair_Px_dev,
                    pair_Py_dev,
                    pair_Pz_dev,
                    pair_cK_dev,
                    int(nao0),
                    int(naux),
                    int(lp),
                    int(lq),
                    bar_V_dev,
                    shell_atom_dev,
                    grad_dev_flat,
                    int(threads),
                    int(stream_ptr),
                    False,
                )

    t_2c_deriv = time.perf_counter() if profile is not None else 0.0

    if backend_s == "cuda":
        import cupy as cp  # noqa: PLC0415

        if profile is not None:
            cp.cuda.get_current_stream().synchronize()
            t_2c_deriv = time.perf_counter()

        # Copy back once.
        if grad_dev is None:  # pragma: no cover
            raise RuntimeError("internal error: grad_dev not initialized for backend='cuda'")
        grad = np.asarray(cp.asnumpy(grad_dev), dtype=np.float64)

    if profile is not None:
        profile.clear()
        profile["backend"] = backend_s
        profile["df_threads"] = int(df_threads)
        profile["natm"] = int(natm)
        profile["nao"] = int(nao0)
        profile["naux"] = int(naux)
        profile["t_shell_atom_s"] = float(t_shell_atom - t0_total)
        profile["t_combined_basis_s"] = float(t_combined_basis - t_shell_atom)
        profile["t_pair_tables_s"] = float(t_pair_tables - t_combined_basis)
        profile["t_metric_build_s"] = float(t_metric_build - t_pair_tables)
        profile["t_metric_chol_s"] = float(t_metric_chol - t_metric_build)
        profile["t_df_adjoint_s"] = float(t_df_adjoint - t_metric_chol)
        profile["t_3c_deriv_s"] = float(t_3c_deriv - t_df_adjoint)
        profile["t_2c_deriv_s"] = float(t_2c_deriv - t_3c_deriv)
        profile["t_total_s"] = float(t_2c_deriv - t0_total)
        if _3c_class_times:
            profile["_3c_class_times"] = _3c_class_times  # [(la, lb, lq, n_spAB, n_spCD, time_s), ...]

    return grad

def compute_df_gradient_contributions_tiled(
    mol,
    auxbasis: Any,
    bar_L_ao: np.ndarray,
    L_ao: np.ndarray,
    *,
    backend: str = "cpu",
    df_config=None,
    df_threads: int = 0,
    delta_bohr: float = 1e-5,
    expand_contractions: bool = True,
    profile: dict | None = None,
) -> np.ndarray:
    """Compute nuclear gradients from DF integral derivatives.

    This is the DF-only contraction stage used by unrelaxed (frozen-orbital/CI)
    CASPT2 gradients.

    Implementation
    --------------
    Prefer analytic cuERI derivative contractions on the *whitened* DF factors `B`
    (aka `cderi`). If analytic derivative kernels are unavailable, fall back to
    central finite differences on `B`:

        dE/dR ≈ bar_B : dB/dR

    Both paths keep the runtime self-contained (no reliance on PySCF derivative
    integral kernels).
    """

    # Preserve the historical signature. `L_ao` is not needed for the FD path,
    # but we keep it for shape validation.
    bar_L_ao = np.asarray(bar_L_ao, dtype=np.float64)
    L_ao = np.asarray(L_ao, dtype=np.float64)
    if bar_L_ao.shape != L_ao.shape:
        raise ValueError("bar_L_ao and L_ao shape mismatch")

    # Fast path: native ASUKA molecule container.
    try:
        from asuka.frontend.molecule import Molecule  # noqa: PLC0415
    except Exception:  # pragma: no cover
        Molecule = None  # type: ignore[assignment]

    if Molecule is not None and isinstance(mol, Molecule):
        return compute_df_gradient_contributions_fd_molecule(
            mol,
            auxbasis=auxbasis,
            bar_L_ao=bar_L_ao,
            expand_contractions=bool(expand_contractions),
            backend=str(backend),
            df_config=df_config,
            df_threads=int(df_threads),
            delta_bohr=float(delta_bohr),
            profile=profile,
        )

    # Mol-like path (e.g. an external Mole object): pack AO/aux bases without importing
    # external dependencies.
    from asuka.cueri.mol_basis import pack_cart_shells_from_mol  # noqa: PLC0415
    from asuka.integrals.int1e_cart import nao_cart_from_basis  # noqa: PLC0415
    from asuka.frontend.basis_bse import load_autoaux_shells, load_basis_shells  # noqa: PLC0415
    from asuka.frontend.basis_packer import pack_cart_basis, parse_pyscf_basis_dict  # noqa: PLC0415

    if not bool(getattr(mol, "cart", False)):
        raise NotImplementedError("DF gradient FD path currently requires cart=True")

    # Atom list in Bohr (rely on mol-like methods).
    natm = int(getattr(mol, "natm"))
    if natm <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    if not callable(getattr(mol, "atom_symbol", None)) or not callable(getattr(mol, "atom_coord", None)):
        raise TypeError("mol must provide natm, atom_symbol(i), and atom_coord(i)")

    def _clean_element_symbol(sym_raw: str) -> str:
        """Return a best-effort element symbol from a mol-like `atom_symbol` string.

        Some `mol` implementations (notably from Molden imports) return labels like
        ``'Li1'`` / ``'H2'`` from `atom_symbol(i)`. Basis Set
        Exchange expects plain element symbols (``'Li'``, ``'H'``), so strip common
        numeric suffixes and normalize capitalization.
        """

        import re

        s = str(sym_raw).strip()
        m = re.match(r"^([A-Za-z]{1,2})", s)
        if m is None:
            return s
        el = m.group(1)
        if len(el) == 1:
            return el.upper()
        return el[0].upper() + el[1:].lower()

    atoms_bohr: list[tuple[str, np.ndarray]] = []
    for ia in range(natm):
        sym = _clean_element_symbol(str(mol.atom_symbol(int(ia))))
        xyz = np.asarray(mol.atom_coord(int(ia)), dtype=np.float64).reshape((3,))
        atoms_bohr.append((sym, xyz))

    atom_coords_bohr = np.asarray([xyz for _sym, xyz in atoms_bohr], dtype=np.float64).reshape((natm, 3))
    elements = sorted({sym for sym, _xyz in atoms_bohr})

    ao_basis = pack_cart_shells_from_mol(mol, expand_contractions=bool(expand_contractions))

    # Aux basis: packed BasisCartSoA, explicit dict, or BSE-resolved string.
    if hasattr(auxbasis, "shell_cxyz") and hasattr(auxbasis, "shell_l") and hasattr(auxbasis, "prim_exp"):
        aux_basis = auxbasis
    elif isinstance(auxbasis, dict):
        aux_shells = parse_pyscf_basis_dict(auxbasis, elements=elements)
        aux_basis = pack_cart_basis(atoms_bohr, aux_shells, expand_contractions=bool(expand_contractions))
    elif isinstance(auxbasis, str):
        aux_name = str(auxbasis)
        try:
            aux_shells = load_basis_shells(aux_name, elements=elements)
        except Exception as e_bse:
            basis_name = getattr(mol, "basis", None)
            if isinstance(basis_name, str):
                base = str(aux_name).strip()
                for suf in ("-jkfit", "-jfit", "-rifit", "-ri", "-mp2fit"):
                    if base.lower().endswith(suf):
                        base = base[: -len(suf)]
                        break
                base = base or str(basis_name)
                try:
                    _aux_name2, aux_shells = load_autoaux_shells(base, elements=elements)
                except Exception:
                    aux_shells = None
            else:
                aux_shells = None

            if aux_shells is None:
                # Standalone fallback for common PySCF-only aux basis aliases.
                # In PySCF, "weigend" and "weigend+etb" are identical to BSE's
                # `def2-universal-jfit` for all elements.
                if str(aux_name).strip().lower() in ("weigend+etb", "weigend", "etb"):
                    aux_shells = load_basis_shells("def2-universal-jfit", elements=elements)
                else:
                    raise RuntimeError(
                        f"failed to load auxiliary basis '{aux_name}' via Basis Set Exchange. "
                        "ASUKA no longer falls back to PySCF's basis library. "
                        "Install `basis_set_exchange` (e.g. `pip install asuka[frontend]`) "
                        "or provide an explicit per-element aux basis dict."
                    ) from e_bse
        aux_basis = pack_cart_basis(atoms_bohr, aux_shells, expand_contractions=bool(expand_contractions))
    else:
        raise TypeError("auxbasis must be a string name, a basis dict, or a packed aux basis object")

    nao = nao_cart_from_basis(ao_basis)
    naux = nao_cart_from_basis(aux_basis)
    if bar_L_ao.shape != (naux, nao, nao):
        raise ValueError(f"bar_L_ao shape mismatch: expected {(naux, nao, nao)}, got {bar_L_ao.shape}")

    # Fast path: analytic derivative contraction with cuERI kernels.
    #
    # L_ao is stored as (naux,nao,nao); analytic kernels expect B_ao as (nao,nao,naux).
    B_ao = np.transpose(L_ao, (1, 2, 0))
    try:
        return compute_df_gradient_contributions_analytic_packed_bases(
            ao_basis,
            aux_basis,
            atom_coords_bohr=atom_coords_bohr,
            B_ao=B_ao,
            bar_L_ao=bar_L_ao,
            backend=str(backend),
            df_threads=int(df_threads),
            profile=profile,
        )
    except RuntimeError as e:
        # Fallback: finite differences on the whitened DF factors `B` when analytic
        # derivative kernels are unavailable (e.g. extension not built).
        msg = str(e).lower()
        if "extension" not in msg and "missing" not in msg and "required" not in msg:
            raise
        return compute_df_gradient_contributions_fd_packed_bases(
            ao_basis,
            aux_basis,
            atom_coords_bohr=atom_coords_bohr,
            bar_L_ao=bar_L_ao,
            backend=str(backend),
            df_config=df_config,
            df_threads=int(df_threads),
            delta_bohr=float(delta_bohr),
            profile=profile,
        )


def _basis_cart_shifted_by_atom(
    basis,
    *,
    shell_atom: np.ndarray,
    atom: int,
    disp_bohr_xyz: np.ndarray,
):
    from asuka.cueri.basis_cart import BasisCartSoA  # noqa: PLC0415

    if not isinstance(basis, BasisCartSoA):
        raise TypeError("basis must be a BasisCartSoA")

    shell_atom = np.asarray(shell_atom, dtype=np.int32).ravel()
    nshell = int(basis.shell_cxyz.shape[0])
    if shell_atom.shape != (nshell,):
        raise ValueError("shell_atom shape mismatch")

    disp = np.asarray(disp_bohr_xyz, dtype=np.float64).reshape((3,))
    if not bool(np.any(disp)):
        return basis

    atom_i = int(atom)
    if atom_i < 0:
        raise ValueError("atom must be >= 0")

    shell_cxyz = np.asarray(basis.shell_cxyz, dtype=np.float64, order="C").copy()
    mask = shell_atom == np.int32(atom_i)
    shell_cxyz[mask] += disp[None, :]

    return BasisCartSoA(
        shell_cxyz=shell_cxyz,
        shell_prim_start=basis.shell_prim_start,
        shell_nprim=basis.shell_nprim,
        shell_l=basis.shell_l,
        shell_ao_start=basis.shell_ao_start,
        prim_exp=basis.prim_exp,
        prim_coef=basis.prim_coef,
        source_bas_id=basis.source_bas_id,
        source_ctr_id=basis.source_ctr_id,
    )


def compute_df_gradient_contributions_fd_packed_bases(
    ao_basis,
    aux_basis,
    *,
    atom_coords_bohr: np.ndarray,
    bar_L_ao: np.ndarray,
    backend: str = "cpu",
    df_config=None,
    df_threads: int = 0,
    delta_bohr: float = 1e-4,
    profile: dict | None = None,
) -> np.ndarray:
    """DF integral derivative contraction via finite differences.

    This computes the contraction-only contribution:

        dE/dR ≈ bar_L : dL/dR

    by central finite differences on the whitened DF factors `L_ao` (aka `B`).
    The DF build is performed by cuERI on either CPU or CUDA.

    Notes
    -----
    - This is primarily a correctness / baseline path to remove PySCF derivative
      integral dependencies (`int3c2e_ip*`, `int2c2e_ip*`).
    - Cost scales as O(natm * 3) DF builds.
    """

    import time

    atom_coords_bohr = np.asarray(atom_coords_bohr, dtype=np.float64)
    if atom_coords_bohr.ndim != 2 or atom_coords_bohr.shape[1] != 3:
        raise ValueError("atom_coords_bohr must have shape (natm, 3)")
    natm = int(atom_coords_bohr.shape[0])
    if natm <= 0:
        raise ValueError("natm must be > 0")

    bar_L_ao = np.asarray(bar_L_ao, dtype=np.float64)
    if bar_L_ao.ndim != 3:
        raise ValueError("bar_L_ao must have shape (naux, nao, nao)")
    naux, nao, nao2 = map(int, bar_L_ao.shape)
    if nao != nao2:
        raise ValueError("bar_L_ao must have shape (naux, nao, nao)")

    backend_s = str(backend).strip().lower()
    if backend_s not in {"cpu", "cuda"}:
        raise ValueError("backend must be 'cpu' or 'cuda'")

    if df_config is not None:
        from asuka.integrals.cueri_df import CuERIDFConfig  # noqa: PLC0415

        if not isinstance(df_config, CuERIDFConfig):
            raise TypeError("df_config must be a CuERIDFConfig or None")

    delta = float(delta_bohr)
    if delta <= 0.0:
        raise ValueError("delta_bohr must be > 0")

    t0 = time.perf_counter() if profile is not None else 0.0

    from asuka.integrals.int1e_cart import shell_to_atom_map  # noqa: PLC0415

    ao_shell_atom = shell_to_atom_map(ao_basis, atom_coords_bohr=atom_coords_bohr)
    aux_shell_atom = shell_to_atom_map(aux_basis, atom_coords_bohr=atom_coords_bohr)

    if backend_s == "cuda":
        try:
            import cupy as cp  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError("backend='cuda' requires CuPy") from e

        from asuka.integrals.cueri_df import build_df_B_from_cueri_packed_bases  # noqa: PLC0415

        def _build_B(ao_b, aux_b):
            return build_df_B_from_cueri_packed_bases(ao_b, aux_b, config=df_config, profile=None)

        aux_block = _choose_aux_block_naux(nao=nao, naux=naux, target_bytes=64 * 1024 * 1024)

    else:
        from asuka.integrals.cueri_df_cpu import build_df_B_from_cueri_packed_bases_cpu  # noqa: PLC0415

        def _build_B(ao_b, aux_b):
            return build_df_B_from_cueri_packed_bases_cpu(ao_b, aux_b, threads=int(df_threads), profile=None)

        aux_block = _choose_aux_block_naux(nao=nao, naux=naux, target_bytes=128 * 1024 * 1024)

    grad = np.zeros((natm, 3), dtype=np.float64)
    disp = np.zeros((3,), dtype=np.float64)
    for ia in range(natm):
        for ax in range(3):
            disp[:] = 0.0
            disp[ax] = delta

            ao_p = _basis_cart_shifted_by_atom(ao_basis, shell_atom=ao_shell_atom, atom=ia, disp_bohr_xyz=+disp)
            aux_p = _basis_cart_shifted_by_atom(aux_basis, shell_atom=aux_shell_atom, atom=ia, disp_bohr_xyz=+disp)
            ao_m = _basis_cart_shifted_by_atom(ao_basis, shell_atom=ao_shell_atom, atom=ia, disp_bohr_xyz=-disp)
            aux_m = _basis_cart_shifted_by_atom(aux_basis, shell_atom=aux_shell_atom, atom=ia, disp_bohr_xyz=-disp)

            # Reduce peak memory: do not hold B(+h) and B(-h) simultaneously.
            Bp = _build_B(ao_p, aux_p)
            Ep = _contract_bar_with_B_streamed(bar_L_ao, Bp, backend=backend_s, aux_block_naux=int(aux_block))
            Bp = None

            Bm = _build_B(ao_m, aux_m)
            Em = _contract_bar_with_B_streamed(bar_L_ao, Bm, backend=backend_s, aux_block_naux=int(aux_block))
            Bm = None

            grad[ia, ax] = (Ep - Em) / (2.0 * delta)

    if profile is not None:
        profile.clear()
        profile["backend"] = backend_s
        profile["df_threads"] = int(df_threads)
        profile["delta_bohr"] = float(delta)
        profile["natm"] = int(natm)
        profile["aux_block_naux"] = int(aux_block)
        profile["t_total_s"] = float(time.perf_counter() - t0)

    return grad


def compute_df_gradient_contributions_fd_molecule(
    mol: Any,
    *,
    auxbasis: Any,
    bar_L_ao: np.ndarray,
    basis: Any | None = None,
    expand_contractions: bool = True,
    backend: str = "cpu",
    df_config=None,
    df_threads: int = 0,
    delta_bohr: float = 1e-4,
    profile: dict | None = None,
) -> np.ndarray:
    """DF gradient contraction from a `frontend.Molecule`."""

    from asuka.frontend.molecule import Molecule  # noqa: PLC0415

    if not isinstance(mol, Molecule):
        raise TypeError("mol must be asuka.frontend.molecule.Molecule for the standalone DF FD path")

    from asuka.frontend.df import build_df_bases_cart  # noqa: PLC0415

    ao_basis, aux_basis, _aux_name = build_df_bases_cart(
        mol,
        basis=mol.basis if basis is None else basis,
        auxbasis=auxbasis,
        expand_contractions=bool(expand_contractions),
    )
    coords = np.asarray([xyz for _sym, xyz in mol.atoms_bohr], dtype=np.float64).reshape((mol.natm, 3))
    return compute_df_gradient_contributions_fd_packed_bases(
        ao_basis,
        aux_basis,
        atom_coords_bohr=coords,
        bar_L_ao=bar_L_ao,
        backend=str(backend),
        df_config=df_config,
        df_threads=int(df_threads),
        delta_bohr=float(delta_bohr),
        profile=profile,
    )
