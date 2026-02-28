from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from asuka.cueri.cart import ncart
from asuka.cueri.pair_tables_cpu import build_pair_tables_cpu
from asuka.integrals.cueri_df_cpu import _build_df_combined_basis_and_shell_pairs
from asuka.integrals.df_adjoint import chol_lower_adjoint, df_whiten_adjoint, df_whiten_adjoint_Qmn
from asuka.integrals.int1e_cart import nao_cart_from_basis, shell_to_atom_map

Backend = Literal["cpu", "cuda"]


@dataclass
class DFGradContractionContext:
    backend: Backend
    ao_basis: Any
    aux_basis: Any
    atom_coords_bohr: np.ndarray
    df_threads: int

    natm: int
    nao: int
    naux: int

    ao_shell_atom: np.ndarray
    aux_shell_atom: np.ndarray
    aux_shell_l: np.ndarray

    basis_all: Any
    sp_all: Any
    pt_all: Any
    nsp_ao: int
    n_shell_aux: int
    aux_sp0: int

    shell_cxyz_all: np.ndarray
    shell_l_all: np.ndarray
    shell_prim_start_all: np.ndarray
    shell_nprim_all: np.ndarray
    shell_ao_start_all: np.ndarray
    prim_exp_all: np.ndarray

    sp_A_all: np.ndarray
    sp_B_all: np.ndarray
    sp_pair_start_all: np.ndarray
    sp_npair_all: np.ndarray

    pair_eta_all: np.ndarray
    pair_Px_all: np.ndarray
    pair_Py_all: np.ndarray
    pair_Pz_all: np.ndarray
    pair_cK_all: np.ndarray

    shells_by_l: dict[int, np.ndarray]
    spCD_by_l: dict[int, np.ndarray]
    metric_batches: list[tuple[int, int, int, int, np.ndarray, np.ndarray, np.ndarray]]

    L_metric: Any
    cpu: dict[str, Any] | None = None
    cuda: dict[str, Any] | None = None

    @staticmethod
    def build(
        ao_basis: Any,
        aux_basis: Any,
        *,
        atom_coords_bohr: np.ndarray,
        backend: Backend,
        df_threads: int = 0,
        L_chol: Any | None = None,
    ) -> "DFGradContractionContext":
        backend_s = str(backend).strip().lower()
        if backend_s not in ("cpu", "cuda"):
            raise ValueError("backend must be 'cpu' or 'cuda'")

        atom_coords = np.asarray(atom_coords_bohr, dtype=np.float64)
        if atom_coords.ndim != 2 or atom_coords.shape[1] != 3:
            raise ValueError("atom_coords_bohr must have shape (natm, 3)")
        natm = int(atom_coords.shape[0])
        if natm <= 0:
            raise ValueError("atom_coords_bohr must have natm > 0")

        ao_shell_atom = np.asarray(shell_to_atom_map(ao_basis, atom_coords_bohr=atom_coords), dtype=np.int32)
        aux_shell_atom = np.asarray(shell_to_atom_map(aux_basis, atom_coords_bohr=atom_coords), dtype=np.int32)

        basis_all, sp_all, nsp_ao, _n_shell_ao, n_shell_aux = _build_df_combined_basis_and_shell_pairs(ao_basis, aux_basis)
        pt_all = build_pair_tables_cpu(basis_all, sp_all, threads=int(df_threads), profile=None)
        aux_sp0 = int(nsp_ao)

        nao = int(nao_cart_from_basis(ao_basis))
        naux = int(nao_cart_from_basis(aux_basis))

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

        pair_eta_all = np.asarray(pt_all.pair_eta, dtype=np.float64, order="C")
        pair_Px_all = np.asarray(pt_all.pair_Px, dtype=np.float64, order="C")
        pair_Py_all = np.asarray(pt_all.pair_Py, dtype=np.float64, order="C")
        pair_Pz_all = np.asarray(pt_all.pair_Pz, dtype=np.float64, order="C")
        pair_cK_all = np.asarray(pt_all.pair_cK, dtype=np.float64, order="C")

        aux_shell_l = np.asarray(aux_basis.shell_l, dtype=np.int32, order="C").ravel()
        by_l: dict[int, list[int]] = {}
        for sh in range(int(n_shell_aux)):
            by_l.setdefault(int(aux_shell_l[sh]), []).append(int(sh))

        shells_by_l: dict[int, np.ndarray] = {}
        spCD_by_l: dict[int, np.ndarray] = {}
        for lq, q_shells in by_l.items():
            q_arr = np.asarray(q_shells, dtype=np.int32)
            shells_by_l[int(lq)] = q_arr
            spCD_by_l[int(lq)] = (aux_sp0 + q_arr).astype(np.int32, copy=False)

        metric_batches: list[tuple[int, int, int, int, np.ndarray, np.ndarray, np.ndarray]] = []
        for psh in range(int(n_shell_aux)):
            lp = int(aux_shell_l[int(psh)])
            atomP = int(aux_shell_atom[int(psh)])
            spAB = int(aux_sp0 + int(psh))
            for lq, q_shells in shells_by_l.items():
                q_arr = np.asarray(q_shells, dtype=np.int32)
                q_list = q_arr[q_arr <= np.int32(int(psh))]
                if int(q_list.size) == 0:
                    continue
                spCD_batch = (aux_sp0 + q_list).astype(np.int32, copy=False)
                atomQ = np.asarray(aux_shell_atom[q_list], dtype=np.int32, order="C")
                fac = np.full((int(q_list.size),), 2.0, dtype=np.float64)
                fac[q_list == np.int32(int(psh))] = 1.0
                metric_batches.append((int(spAB), int(lp), int(atomP), int(lq), spCD_batch, atomQ, fac))

        cpu_ctx: dict[str, Any] | None = None
        if backend_s == "cpu":
            try:
                from asuka.cueri import _eri_rys_cpu as _ext  # noqa: PLC0415
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "CPU ERI extension is required for analytic DF gradient contraction on backend='cpu'"
                ) from e

            eri_batch = getattr(_ext, "eri_rys_tile_cart_sp_batch_cy", None)
            fn_3c = getattr(_ext, "df_int3c2e_deriv_contracted_cart_sp_batch_cy", None)
            fn_2c = getattr(_ext, "df_metric_2c2e_deriv_contracted_cart_sp_batch_cy", None)
            if eri_batch is None or fn_3c is None or fn_2c is None:  # pragma: no cover
                raise RuntimeError("CPU ERI extension is missing DF derivative entry points; rebuild the extension")

            aux_shell_ao_start = np.asarray(aux_basis.shell_ao_start, dtype=np.int32, order="C").ravel()
            V = np.zeros((naux, naux), dtype=np.float64)
            for psh in range(int(n_shell_aux)):
                lp = int(aux_shell_l[int(psh)])
                nP = int(ncart(lp))
                p0 = int(aux_shell_ao_start[int(psh)])
                spAB = int(aux_sp0 + psh)

                for lq, q_shells in shells_by_l.items():
                    nQ = int(ncart(int(lq)))
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

            V = 0.5 * (V + V.T)
            L_metric = np.linalg.cholesky(V)
            cpu_ctx = {"fn_3c": fn_3c, "fn_2c": fn_2c}
        else:
            try:
                import cupy as cp  # noqa: PLC0415
            except Exception as e:  # pragma: no cover
                raise RuntimeError("backend='cuda' requires CuPy") from e

            from asuka.cueri import df as cueri_df  # noqa: PLC0415

            if L_chol is not None:
                L_metric = cp.ascontiguousarray(cp.asarray(L_chol, dtype=cp.float64))
            else:
                V = cueri_df.metric_2c2e_basis(aux_basis, stream=None, backend="gpu_rys", mode="warp", threads=256)
                _v_diag = cp.diag(V)
                _v_shift = max(float(cp.max(cp.abs(_v_diag))) * 1e-14, 1e-12)
                V[cp.diag_indices_from(V)] += _v_shift
                L_metric = cp.linalg.cholesky(V)

        ctx = DFGradContractionContext(
            backend=backend_s,  # type: ignore[arg-type]
            ao_basis=ao_basis,
            aux_basis=aux_basis,
            atom_coords_bohr=atom_coords,
            df_threads=int(df_threads),
            natm=natm,
            nao=int(nao),
            naux=int(naux),
            ao_shell_atom=ao_shell_atom,
            aux_shell_atom=aux_shell_atom,
            aux_shell_l=aux_shell_l,
            basis_all=basis_all,
            sp_all=sp_all,
            pt_all=pt_all,
            nsp_ao=int(nsp_ao),
            n_shell_aux=int(n_shell_aux),
            aux_sp0=int(aux_sp0),
            shell_cxyz_all=shell_cxyz_all,
            shell_l_all=shell_l_all,
            shell_prim_start_all=shell_prim_start_all,
            shell_nprim_all=shell_nprim_all,
            shell_ao_start_all=shell_ao_start_all,
            prim_exp_all=prim_exp_all,
            sp_A_all=sp_A_all,
            sp_B_all=sp_B_all,
            sp_pair_start_all=sp_pair_start_all,
            sp_npair_all=sp_npair_all,
            pair_eta_all=pair_eta_all,
            pair_Px_all=pair_Px_all,
            pair_Py_all=pair_Py_all,
            pair_Pz_all=pair_Pz_all,
            pair_cK_all=pair_cK_all,
            shells_by_l=shells_by_l,
            spCD_by_l=spCD_by_l,
            metric_batches=metric_batches,
            L_metric=L_metric,
            cpu=cpu_ctx,
            cuda=None,
        )
        if backend_s == "cuda":
            ctx._init_cuda()
        return ctx

    def _init_cuda(self) -> None:
        import cupy as cp  # noqa: PLC0415

        try:
            from asuka.cueri import _cueri_cuda_ext as _ext_cuda  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "cuERI CUDA extension is required for analytic DF gradient contraction; "
                "build via `python -m asuka.cueri.build_cuda_ext`"
            ) from e

        spCD_by_l_dev = {
            int(lq): cp.ascontiguousarray(cp.asarray(spCD_batch, dtype=cp.int32)) for lq, spCD_batch in self.spCD_by_l.items()
        }
        atomC_by_l_dev = {
            int(lq): cp.ascontiguousarray(cp.asarray(self.aux_shell_atom[np.asarray(q_shells, dtype=np.int32)], dtype=cp.int32))
            for lq, q_shells in self.shells_by_l.items()
        }
        metric_batches_dev: list[tuple[int, int, int, int, Any, Any, Any]] = []
        for spAB, lp, atomP, lq, spCD_batch, atomQ, fac in self.metric_batches:
            metric_batches_dev.append(
                (
                    int(spAB),
                    int(lp),
                    int(atomP),
                    int(lq),
                    cp.ascontiguousarray(cp.asarray(spCD_batch, dtype=cp.int32)),
                    cp.ascontiguousarray(cp.asarray(atomQ, dtype=cp.int32)),
                    cp.ascontiguousarray(cp.asarray(fac, dtype=cp.float64)),
                )
            )

        # Combined AO+aux shell→atom map (AO shells first, then aux shells).
        # shellA/B from sp_A/sp_B for AO pairs index into [0, n_ao_shells).
        # shellC from sp_A for aux pairs indexes into [n_ao_shells, n_ao_shells+n_aux_shells).
        shell_atom_all = np.concatenate([
            np.asarray(self.ao_shell_atom, dtype=np.int32),
            np.asarray(self.aux_shell_atom, dtype=np.int32),
        ])
        shell_atom_dev = cp.ascontiguousarray(cp.asarray(shell_atom_all, dtype=cp.int32))

        # Pre-group AO shell pairs by (la, lb) angular momentum class for batched kernel.
        shell_l_np = np.asarray(self.shell_l_all, dtype=np.int32).ravel()
        sp_A_np = np.asarray(self.sp_A_all, dtype=np.int32).ravel()
        sp_B_np = np.asarray(self.sp_B_all, dtype=np.int32).ravel()
        spAB_by_lab: dict[tuple[int, int], list[int]] = {}
        for spAB_i in range(int(self.nsp_ao)):
            shA = int(sp_A_np[spAB_i])
            shB = int(sp_B_np[spAB_i])
            key = (int(shell_l_np[shA]), int(shell_l_np[shB]))
            if key not in spAB_by_lab:
                spAB_by_lab[key] = []
            spAB_by_lab[key].append(spAB_i)
        spAB_by_lab_dev = {
            lab: cp.ascontiguousarray(cp.asarray(indices, dtype=cp.int32))
            for lab, indices in spAB_by_lab.items()
        }

        self.cuda = {
            "_ext": _ext_cuda,
            "shell_cx": cp.ascontiguousarray(cp.asarray(self.shell_cxyz_all[:, 0], dtype=cp.float64)),
            "shell_cy": cp.ascontiguousarray(cp.asarray(self.shell_cxyz_all[:, 1], dtype=cp.float64)),
            "shell_cz": cp.ascontiguousarray(cp.asarray(self.shell_cxyz_all[:, 2], dtype=cp.float64)),
            "shell_prim_start": cp.ascontiguousarray(cp.asarray(self.shell_prim_start_all, dtype=cp.int32)),
            "shell_nprim": cp.ascontiguousarray(cp.asarray(self.shell_nprim_all, dtype=cp.int32)),
            "shell_ao_start": cp.ascontiguousarray(cp.asarray(self.shell_ao_start_all, dtype=cp.int32)),
            "prim_exp": cp.ascontiguousarray(cp.asarray(self.prim_exp_all, dtype=cp.float64)),
            "sp_A": cp.ascontiguousarray(cp.asarray(self.sp_A_all, dtype=cp.int32)),
            "sp_B": cp.ascontiguousarray(cp.asarray(self.sp_B_all, dtype=cp.int32)),
            "sp_pair_start": cp.ascontiguousarray(cp.asarray(self.sp_pair_start_all, dtype=cp.int32)),
            "sp_npair": cp.ascontiguousarray(cp.asarray(self.sp_npair_all, dtype=cp.int32)),
            "pair_eta": cp.ascontiguousarray(cp.asarray(self.pair_eta_all, dtype=cp.float64)),
            "pair_Px": cp.ascontiguousarray(cp.asarray(self.pair_Px_all, dtype=cp.float64)),
            "pair_Py": cp.ascontiguousarray(cp.asarray(self.pair_Py_all, dtype=cp.float64)),
            "pair_Pz": cp.ascontiguousarray(cp.asarray(self.pair_Pz_all, dtype=cp.float64)),
            "pair_cK": cp.ascontiguousarray(cp.asarray(self.pair_cK_all, dtype=cp.float64)),
            "spCD_by_l": spCD_by_l_dev,
            "atomC_by_l": atomC_by_l_dev,
            "metric_batches": metric_batches_dev,
            "shell_l_host": shell_l_np,
            "shell_atom": shell_atom_dev,
            "spAB_by_lab": spAB_by_lab_dev,
        }

    # ------------------------------------------------------------------
    # Internal: CPU kernel loops extracted for reuse
    # ------------------------------------------------------------------
    def _contract_cpu_from_adjoints(self, bar_X_flat: np.ndarray, bar_V: np.ndarray) -> np.ndarray:
        """Run 3c + 2c CPU kernel loops given pre-computed Cartesian adjoints.

        Parameters
        ----------
        bar_X_flat : np.ndarray, shape (nao_cart * nao_cart, naux)
        bar_V : np.ndarray, shape (naux, naux)
        """
        grad = np.zeros((self.natm, 3), dtype=np.float64)
        if self.cpu is None:  # pragma: no cover
            raise RuntimeError("internal error: CPU function table missing")
        fn_3c = self.cpu["fn_3c"]
        fn_2c = self.cpu["fn_2c"]

        for spAB in range(int(self.nsp_ao)):
            shA = int(self.sp_A_all[int(spAB)])
            shB = int(self.sp_B_all[int(spAB)])
            fac = 2.0 if shA != shB else 1.0
            atomA = int(self.ao_shell_atom[int(shA)])
            atomB = int(self.ao_shell_atom[int(shB)])

            for lq, spCD_batch in self.spCD_by_l.items():
                q_shells = self.shells_by_l[int(lq)]
                out_batch = fn_3c(
                    self.shell_cxyz_all,
                    self.shell_prim_start_all,
                    self.shell_nprim_all,
                    self.shell_l_all,
                    self.shell_ao_start_all,
                    self.prim_exp_all,
                    self.sp_A_all,
                    self.sp_B_all,
                    self.sp_pair_start_all,
                    self.sp_npair_all,
                    self.pair_eta_all,
                    self.pair_Px_all,
                    self.pair_Py_all,
                    self.pair_Pz_all,
                    self.pair_cK_all,
                    int(spAB),
                    spCD_batch,
                    int(self.nao),
                    bar_X_flat,
                )
                out_batch = np.asarray(out_batch, dtype=np.float64)
                for t, qsh in enumerate(q_shells.tolist()):
                    atomC = int(self.aux_shell_atom[int(qsh)])
                    grad[atomA] += fac * out_batch[int(t), 0, :]
                    grad[atomB] += fac * out_batch[int(t), 1, :]
                    grad[atomC] += fac * out_batch[int(t), 2, :]

        for spAB, lp, atomP, lq, spCD_batch, atomQ, fac in self.metric_batches:
            out_batch = fn_2c(
                self.shell_cxyz_all,
                self.shell_prim_start_all,
                self.shell_nprim_all,
                self.shell_l_all,
                self.shell_ao_start_all,
                self.prim_exp_all,
                self.sp_A_all,
                self.sp_B_all,
                self.sp_pair_start_all,
                self.sp_npair_all,
                self.pair_eta_all,
                self.pair_Px_all,
                self.pair_Py_all,
                self.pair_Pz_all,
                self.pair_cK_all,
                int(spAB),
                spCD_batch,
                int(self.nao),
                bar_V,
            )
            out_batch = np.asarray(out_batch, dtype=np.float64)
            grad[int(atomP)] += np.sum(out_batch[:, 0, :] * fac[:, None], axis=0)
            np.add.at(grad, atomQ, out_batch[:, 1, :] * fac[:, None])

        return np.asarray(grad, dtype=np.float64)

    def contract(self, *, B_ao: Any, bar_L_ao: Any) -> np.ndarray:
        if self.backend == "cuda":
            import cupy as cp  # noqa: PLC0415

            grad_dev = self.contract_device(B_ao=B_ao, bar_L_ao=bar_L_ao)
            return np.asarray(cp.asnumpy(grad_dev), dtype=np.float64)

        B = np.asarray(B_ao, dtype=np.float64, order="C")
        bar_L = np.asarray(bar_L_ao, dtype=np.float64, order="C")

        if B.ndim != 3:
            raise ValueError("B_ao must have shape (nao, nao, naux)")
        nao0, nao1, naux = map(int, B.shape)
        if nao0 != nao1:
            raise ValueError("B_ao must have shape (nao, nao, naux)")
        if nao0 != int(self.nao) or naux != int(self.naux):
            raise ValueError("B_ao shape mismatch with context")

        if tuple(map(int, bar_L.shape)) != (int(self.naux), int(self.nao), int(self.nao)):
            raise ValueError("bar_L_ao must have shape (naux, nao, nao)")

        bar_L_c = np.asarray(bar_L, dtype=np.float64, order="C")
        bar_X, bar_Lchol = df_whiten_adjoint_Qmn(B, bar_L_c, self.L_metric)
        bar_V = chol_lower_adjoint(self.L_metric, bar_Lchol)
        bar_X = 0.5 * (bar_X + bar_X.transpose((1, 0, 2)))
        bar_X = np.asarray(bar_X, dtype=np.float64, order="C")
        bar_V = np.asarray(bar_V, dtype=np.float64, order="C")
        bar_X_flat = bar_X.reshape((self.nao * self.nao, self.naux))

        return self._contract_cpu_from_adjoints(bar_X_flat, bar_V)

    def contract_sph(self, *, B_sph: Any, bar_L_sph: Any, T_c2s: Any) -> np.ndarray:
        """Contract DF gradient with spherical-basis B and bar_L.

        Computes the DF adjoint (bar_X, bar_V) in the smaller spherical basis,
        then transforms bar_X to Cartesian for the 3c derivative kernel.
        The 2c (metric) contribution is basis-independent (no AO indices).

        Parameters
        ----------
        B_sph : array, shape (nao_sph, nao_sph, naux)
            Whitened DF 3-index integrals in spherical AO basis.
        bar_L_sph : array, shape (naux, nao_sph, nao_sph)
            Adjoint of whitened DF factors in spherical AO basis.
        T_c2s : array, shape (nao_cart, nao_sph)
            Cart-to-spherical transformation matrix.

        Returns
        -------
        np.ndarray, shape (natm, 3)
            Nuclear gradient contribution from DF 2e integrals.
        """
        if self.backend == "cuda":
            import cupy as cp  # noqa: PLC0415

            grad_dev = self.contract_device_sph(B_sph=B_sph, bar_L_sph=bar_L_sph, T_c2s=T_c2s)
            return np.asarray(cp.asnumpy(grad_dev), dtype=np.float64)

        T = np.asarray(T_c2s, dtype=np.float64)
        B = np.asarray(B_sph, dtype=np.float64, order="C")
        bar_L = np.asarray(bar_L_sph, dtype=np.float64, order="C")

        # 1. Compute adjoints in spherical basis (smaller, faster)
        bar_X_sph, bar_Lchol = df_whiten_adjoint_Qmn(B, bar_L, self.L_metric)
        bar_V = chol_lower_adjoint(self.L_metric, bar_Lchol)

        # 2. Transform bar_X to Cartesian for 3c kernel: bar_X_cart[mu,nu,Q] = T @ bar_X_sph @ T^T
        bar_X_cart = np.einsum("mi,ijQ,nj->mnQ", T, bar_X_sph, T, optimize=True)
        bar_X_cart = 0.5 * (bar_X_cart + bar_X_cart.transpose((1, 0, 2)))
        bar_X_flat = np.asarray(bar_X_cart.reshape((self.nao * self.nao, self.naux)), dtype=np.float64, order="C")
        bar_V = np.asarray(bar_V, dtype=np.float64, order="C")

        # 3. Reuse existing kernel loops
        return self._contract_cpu_from_adjoints(bar_X_flat, bar_V)

    # ------------------------------------------------------------------
    # Internal: CUDA kernel loops extracted for reuse
    # ------------------------------------------------------------------
    def _contract_device_from_adjoints(self, bar_X_dev: Any, bar_V_dev: Any) -> Any:
        """Run 3c + 2c CUDA kernel loops given pre-computed Cartesian adjoints on device.

        Parameters
        ----------
        bar_X_dev : cupy.ndarray, shape (nao_cart * nao_cart * naux,), flat
        bar_V_dev : cupy.ndarray, shape (naux * naux,), flat
        """
        import cupy as cp  # noqa: PLC0415

        if self.cuda is None:
            raise RuntimeError("CUDA static context is not initialized")
        cuda = self.cuda
        _ext = cuda["_ext"]

        grad_dev = cp.zeros((self.natm, 3), dtype=cp.float64)
        threads = 256
        stream_ptr = int(cp.cuda.get_current_stream().ptr)

        grad_dev_flat = grad_dev.reshape(-1)
        for (la_, lb_), spAB_class_dev in cuda["spAB_by_lab"].items():
            n_spAB = int(spAB_class_dev.shape[0])
            for lq, spCD_dev in cuda["spCD_by_l"].items():
                nt = int(spCD_dev.shape[0])
                if nt == 0 or n_spAB == 0:
                    continue
                _ext.df_int3c2e_deriv_contracted_cart_allsp_atomgrad_inplace_device(
                    spAB_class_dev,
                    spCD_dev,
                    cuda["sp_A"],
                    cuda["sp_B"],
                    cuda["sp_pair_start"],
                    cuda["sp_npair"],
                    cuda["shell_cx"],
                    cuda["shell_cy"],
                    cuda["shell_cz"],
                    cuda["shell_prim_start"],
                    cuda["shell_nprim"],
                    cuda["shell_ao_start"],
                    cuda["prim_exp"],
                    cuda["pair_eta"],
                    cuda["pair_Px"],
                    cuda["pair_Py"],
                    cuda["pair_Pz"],
                    cuda["pair_cK"],
                    int(self.nao),
                    int(self.naux),
                    int(la_),
                    int(lb_),
                    int(lq),
                    bar_X_dev,
                    cuda["shell_atom"],
                    grad_dev_flat,
                    int(threads),
                    int(stream_ptr),
                    False,
                )

        work_2c: dict[int, Any] = {}
        for spAB, lp, atomP, lq, spCD_dev, atomQ_dev, fac_dev in cuda["metric_batches"]:
            nt = int(spCD_dev.shape[0])
            if nt == 0:
                continue
            out_dev = work_2c.get(nt)
            if out_dev is None:
                out_dev = cp.empty((nt * 6,), dtype=cp.float64)
                work_2c[nt] = out_dev
            _ext.df_metric_2c2e_deriv_contracted_cart_sp_batch_inplace_device(
                int(spAB),
                spCD_dev,
                cuda["sp_A"],
                cuda["sp_B"],
                cuda["sp_pair_start"],
                cuda["sp_npair"],
                cuda["shell_cx"],
                cuda["shell_cy"],
                cuda["shell_cz"],
                cuda["shell_prim_start"],
                cuda["shell_nprim"],
                cuda["shell_ao_start"],
                cuda["prim_exp"],
                cuda["pair_eta"],
                cuda["pair_Px"],
                cuda["pair_Py"],
                cuda["pair_Pz"],
                cuda["pair_cK"],
                int(self.nao),
                int(self.naux),
                int(lp),
                int(lq),
                bar_V_dev,
                out_dev,
                int(threads),
                int(stream_ptr),
                False,
            )
            out_batch_dev = out_dev.reshape((nt, 2, 3))
            grad_dev[int(atomP)] += cp.sum(out_batch_dev[:, 0, :] * fac_dev[:, None], axis=0)
            valsQ = out_batch_dev[:, 1, :] * fac_dev[:, None]
            cp.add.at(grad_dev[:, 0], atomQ_dev, valsQ[:, 0])
            cp.add.at(grad_dev[:, 1], atomQ_dev, valsQ[:, 1])
            cp.add.at(grad_dev[:, 2], atomQ_dev, valsQ[:, 2])

        return grad_dev

    def contract_device(self, *, B_ao: Any, bar_L_ao: Any) -> Any:
        """CUDA-only version of :meth:`contract` that returns the gradient on device.

        Returns
        -------
        cupy.ndarray
            Gradient array on device with shape (natm, 3) and dtype float64.
        """
        if self.backend != "cuda":
            raise NotImplementedError("contract_device is only available for backend='cuda'")

        import cupy as cp  # noqa: PLC0415

        B = cp.asarray(B_ao, dtype=cp.float64)
        if not B.flags.c_contiguous:
            B = cp.ascontiguousarray(B)
        bar_L = cp.asarray(bar_L_ao, dtype=cp.float64)
        if not bar_L.flags.c_contiguous:
            bar_L = cp.ascontiguousarray(bar_L)

        if B.ndim != 3:
            raise ValueError("B_ao must have shape (nao, nao, naux)")
        nao0, nao1, naux = map(int, B.shape)
        if nao0 != nao1:
            raise ValueError("B_ao must have shape (nao, nao, naux)")
        if nao0 != int(self.nao) or naux != int(self.naux):
            raise ValueError("B_ao shape mismatch with context")

        if tuple(map(int, bar_L.shape)) != (int(self.naux), int(self.nao), int(self.nao)):
            raise ValueError("bar_L_ao must have shape (naux, nao, nao)")

        bar_X, bar_Lchol = df_whiten_adjoint_Qmn(B, bar_L, self.L_metric)
        del bar_L
        bar_V = chol_lower_adjoint(self.L_metric, bar_Lchol)
        del bar_Lchol

        bar_X = 0.5 * (bar_X + bar_X.transpose((1, 0, 2)))
        bar_X_dev = cp.ascontiguousarray(bar_X.reshape(-1), dtype=cp.float64)
        del bar_X
        bar_V_dev = cp.ascontiguousarray(bar_V.reshape(-1), dtype=cp.float64)
        del bar_V

        return self._contract_device_from_adjoints(bar_X_dev, bar_V_dev)

    def contract_device_sph(self, *, B_sph: Any, bar_L_sph: Any, T_c2s: Any) -> Any:
        """CUDA spherical variant of :meth:`contract_device`.

        Parameters
        ----------
        B_sph : cupy.ndarray, shape (nao_sph, nao_sph, naux)
        bar_L_sph : cupy.ndarray, shape (naux, nao_sph, nao_sph)
        T_c2s : array, shape (nao_cart, nao_sph)

        Returns
        -------
        cupy.ndarray, shape (natm, 3)
        """
        if self.backend != "cuda":
            raise NotImplementedError("contract_device_sph is only available for backend='cuda'")

        import cupy as cp  # noqa: PLC0415

        B = cp.asarray(B_sph, dtype=cp.float64)
        if not B.flags.c_contiguous:
            B = cp.ascontiguousarray(B)
        bar_L = cp.asarray(bar_L_sph, dtype=cp.float64)
        if not bar_L.flags.c_contiguous:
            bar_L = cp.ascontiguousarray(bar_L)
        T = cp.asarray(T_c2s, dtype=cp.float64)

        # 1. Compute adjoints in spherical basis
        bar_X_sph, bar_Lchol = df_whiten_adjoint_Qmn(B, bar_L, self.L_metric)
        del bar_L
        bar_V = chol_lower_adjoint(self.L_metric, bar_Lchol)
        del bar_Lchol

        # 2. Transform bar_X to Cartesian: bar_X_cart[mu,nu,Q] = T @ bar_X_sph @ T^T
        bar_X_cart = cp.einsum("mi,ijQ,nj->mnQ", T, bar_X_sph, T, optimize=True)
        del bar_X_sph
        bar_X_cart = 0.5 * (bar_X_cart + bar_X_cart.transpose((1, 0, 2)))
        bar_X_dev = cp.ascontiguousarray(bar_X_cart.reshape(-1), dtype=cp.float64)
        del bar_X_cart
        bar_V_dev = cp.ascontiguousarray(bar_V.reshape(-1), dtype=cp.float64)
        del bar_V

        # 3. Reuse existing kernel loops
        return self._contract_device_from_adjoints(bar_X_dev, bar_V_dev)
