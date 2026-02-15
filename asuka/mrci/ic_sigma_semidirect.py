from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.mrci.generalized_davidson import GeneralizedDavidsonResult, generalized_davidson1
from asuka.mrci.ic_basis import ICDoubles, ICSingles, SCDoubles, SCSingles
from asuka.cuguga.oracle import _csr_for_epq, _get_epq_action_cache, _restore_eri_4d


_STEP_TO_OCC_F64 = np.asarray([0.0, 1.0, 1.0, 2.0], dtype=np.float64)  # E,U,L,D


def _validate_hop_map(drt: DRT, hop_map: Any | None) -> None:
    if hop_map is None:
        return
    if getattr(hop_map, "drt_sub", None) is not drt:
        raise ValueError("hop_map.drt_sub must be the same object as ws.drt")


def _apply_epq(
    drt: DRT,
    cache,
    x: np.ndarray,
    *,
    p: int,
    q: int,
    out: np.ndarray,
) -> None:
    """Fill out[:] = E_pq |x> using cached CSC-like action tables."""

    p = int(p)
    q = int(q)
    x = np.asarray(x, dtype=np.float64).ravel()
    if int(out.size) != int(x.size) or int(out.size) != int(drt.ncsf):
        raise ValueError("x/out must have length drt.ncsf")

    if p == q:
        out[:] = _STEP_TO_OCC_F64[cache.steps[:, p]] * x
        return

    csr = _csr_for_epq(cache, drt, p, q)
    out.fill(0.0)
    indptr = csr.indptr
    indices = csr.indices
    data = csr.data
    for j in range(int(x.size)):
        xj = float(x[j])
        if xj == 0.0:
            continue
        start = int(indptr[j])
        end = int(indptr[j + 1])
        if start == end:
            continue
        out[indices[start:end]] += data[start:end] * xj


@dataclass
class ICRefSinglesSemiDirect:
    """Semi-direct ic-MRCI backend for the [reference + singles] contracted space.

    Expands contracted vectors into an uncontracted CSF basis (DRT space),
    applies H there, then projects back.

    Attributes
    ----------
    drt : DRT
        Discrete Reaction Field.
    h1e : Any
        One-electron integrals.
    eri : Any
        Two-electron integrals.
    psi0 : np.ndarray
        Embedded reference vector in ``drt`` basis.
    singles : ICSingles
        Singles label set.
    contract_nthreads : int, optional
        Number of threads for contraction.
    contract_blas_nthreads : int | None, optional
        Number of BLAS threads.
    precompute_epq : bool, optional
        Precompute EPQ actions.
    hop_map : Any | None, optional
        Hop mapping for projected contraction.
    contract_executor : ThreadPoolExecutor | None, optional
        Executor for parallel contraction.
    contract_workspace : Any | None, optional
        Workspace for contraction.
    use_gpu : bool, optional
        Use GPU for contraction.
    """

    drt: DRT
    h1e: Any
    eri: Any
    psi0: np.ndarray  # embedded reference vector in `drt` basis (length ncsf)
    singles: ICSingles
    contract_nthreads: int = 1
    contract_blas_nthreads: int | None = 1
    precompute_epq: bool = True
    hop_map: Any | None = None
    contract_executor: ThreadPoolExecutor | None = None
    contract_workspace: Any | None = None
    use_gpu: bool = False

    _gpu_setup_done: bool = False
    _gpu_resources: dict[str, Any] = field(default_factory=dict)


    _basis_singles: np.ndarray | None = None  # (ncsf, nlab)

    def __post_init__(self) -> None:
        self.psi0 = np.asarray(self.psi0, dtype=np.float64).ravel()
        if int(self.psi0.size) != int(self.drt.ncsf):
            raise ValueError("psi0 must have length drt.ncsf")

        n = float(np.linalg.norm(self.psi0))
        if not np.isfinite(n) or n <= 0.0:
            raise ValueError("psi0 must have nonzero norm")
        self.psi0 = self.psi0 / n

        _validate_hop_map(self.drt, self.hop_map)

        if self.precompute_epq:
            from asuka.cuguga.oracle import precompute_epq_actions  # noqa: PLC0415

            # Precompute all E_pq actions (small-space convenience).
            precompute_epq_actions(self.drt)
            if self.hop_map is not None and getattr(self.hop_map, "drt_full", None) is not self.drt:
                precompute_epq_actions(self.hop_map.drt_full)

        self._gpu_setup_done = False
        self._gpu_resources = {}


    @property
    def nlab(self) -> int:
        return 1 + int(self.singles.nlab)

    def _hop_uncontracted(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        if int(x.size) != int(self.drt.ncsf):
            raise ValueError("uncontracted vector has wrong length")

        if self.use_gpu:
            return self._hop_gpu(x)


        if self.hop_map is not None:
            from asuka.mrci.projected_hop import projected_contract_h_csf_multi  # noqa: PLC0415

            return projected_contract_h_csf_multi(
                mapping=self.hop_map,
                h1e=self.h1e,
                eri=self.eri,
                xs_sub=x,
                precompute_epq_full=False,
                nthreads=int(self.contract_nthreads),
                blas_nthreads=self.contract_blas_nthreads,
                executor=self.contract_executor,
                workspace=self.contract_workspace,
            )[0]

        try:
            from asuka.integrals.df_integrals import DFMOIntegrals  # noqa: PLC0415
        except Exception:  # pragma: no cover
            DFMOIntegrals = None  # type: ignore[assignment]

        if DFMOIntegrals is not None and isinstance(self.eri, DFMOIntegrals):
            from asuka.integrals.contract_df import contract_h_csf_multi_df as _contract_df  # noqa: PLC0415

            return _contract_df(
                self.drt,
                self.h1e,
                self.eri,
                [x],
                precompute_epq=False,
                nthreads=int(self.contract_nthreads),
                blas_nthreads=self.contract_blas_nthreads,
                executor=self.contract_executor,
                workspace=self.contract_workspace,
            )[0]

        from asuka.contract import contract_h_csf_multi as _contract_dense  # noqa: PLC0415

        return _contract_dense(
            self.drt,
            self.h1e,
            self.eri,
            [x],
            precompute_epq=False,
            nthreads=int(self.contract_nthreads),
            blas_nthreads=self.contract_blas_nthreads,
            executor=self.contract_executor,
            workspace=self.contract_workspace,
        )[0]

    def _hop_gpu(self, x: np.ndarray) -> np.ndarray:
        import cupy as cp  # noqa: PLC0415
        from asuka.cuda.cuda_backend import (  # noqa: PLC0415
            make_device_drt,
            make_device_state_cache,
        )
        from asuka.cuda.mrci_hop import (  # noqa: PLC0415
            CudaMrciHopWorkspace,
            hop_cuda_epq_table,
            hop_cuda_projected,
        )
        from asuka.cuguga.oracle import _get_epq_action_cache, precompute_epq_actions  # noqa: PLC0415

        if not self._gpu_setup_done:
            norb = int(self.drt.norb)
            eri_4d = _restore_eri_4d(self.eri, norb)
            h_eff = self.h1e - 0.5 * np.einsum("pqqs->ps", eri_4d)
            eri_mat_t = 0.5 * eri_4d.reshape(norb * norb, -1).T

            self._gpu_resources["h_eff_d"] = cp.asarray(h_eff, dtype=cp.float64)
            self._gpu_resources["eri_mat_t_d"] = cp.asarray(eri_mat_t, dtype=cp.float64)

            from asuka.cuda.cuda_backend import (  # noqa: PLC0415
                build_epq_action_table_combined_device,
                make_device_drt,
                make_device_state_cache,
            )

            if self.hop_map is not None:
                drt_full = self.hop_map.drt_full
                drt_dev_full = make_device_drt(drt_full)
                state_dev_full = make_device_state_cache(drt=drt_full, drt_dev=drt_dev_full)
                self._gpu_resources["drt_dev_full"] = drt_dev_full
                self._gpu_resources["state_dev_full"] = state_dev_full
                self._gpu_resources["epq_table_full"] = build_epq_action_table_combined_device(
                    drt=drt_full, drt_dev=drt_dev_full, state_dev=state_dev_full
                )
                self._gpu_resources["sub_to_full_d"] = cp.asarray(self.hop_map.sub_to_full, dtype=cp.int64)
                self._gpu_resources["hop_ws_full"] = CudaMrciHopWorkspace.auto(
                    ncsf=int(drt_full.ncsf),
                    nops=int(norb) * int(norb),
                    sym_pair=bool(int(os.getenv("CUGUGA_MRCI_CUDA_SYM_PAIR", "0"))),
                )
                self._gpu_resources["x_full_buf"] = cp.empty((int(drt_full.ncsf),), dtype=cp.float64)
                self._gpu_resources["y_full_buf"] = cp.empty((int(drt_full.ncsf),), dtype=cp.float64)
            else:
                drt_dev = make_device_drt(self.drt)
                state_dev = make_device_state_cache(drt=self.drt, drt_dev=drt_dev)
                self._gpu_resources["drt_dev"] = drt_dev
                self._gpu_resources["state_dev"] = state_dev
                self._gpu_resources["epq_table"] = build_epq_action_table_combined_device(
                    drt=self.drt, drt_dev=drt_dev, state_dev=state_dev
                )
                self._gpu_resources["hop_ws"] = CudaMrciHopWorkspace.auto(
                    ncsf=int(self.drt.ncsf),
                    nops=int(norb) * int(norb),
                    sym_pair=bool(int(os.getenv("CUGUGA_MRCI_CUDA_SYM_PAIR", "0"))),
                )
            self._gpu_setup_done = True

        x_d = cp.asarray(x, dtype=cp.float64)
        if self.hop_map is not None:
            y_d = hop_cuda_projected(
                drt_full=self.hop_map.drt_full,
                drt_dev_full=self._gpu_resources["drt_dev_full"],
                state_dev_full=self._gpu_resources["state_dev_full"],
                epq_table_full=self._gpu_resources["epq_table_full"],
                h_eff=self._gpu_resources["h_eff_d"],
                eri_mat_t_full=self._gpu_resources["eri_mat_t_d"],
                x_sub=x_d,
                sub_to_full=self._gpu_resources["sub_to_full_d"],
                x_full_buf=self._gpu_resources["x_full_buf"],
                y_full_buf=self._gpu_resources["y_full_buf"],
                workspace_full=self._gpu_resources["hop_ws_full"],
            )
        else:
            y_d = hop_cuda_epq_table(
                drt=self.drt,
                drt_dev=self._gpu_resources["drt_dev"],
                state_dev=self._gpu_resources["state_dev"],
                epq_table=self._gpu_resources["epq_table"],
                h_eff=self._gpu_resources["h_eff_d"],
                eri_mat_t=self._gpu_resources["eri_mat_t_d"],
                x=x_d,
                workspace=self._gpu_resources["hop_ws"],
            )
        return cp.asnumpy(y_d)


    def _build_singles_basis(self) -> np.ndarray:
        if self._basis_singles is not None:
            return self._basis_singles

        ncsf = int(self.drt.ncsf)
        nlab = int(self.singles.nlab)
        if nlab == 0:
            self._basis_singles = np.zeros((ncsf, 0), dtype=np.float64)
            return self._basis_singles

        cache = _get_epq_action_cache(self.drt)
        out = np.empty((ncsf, nlab), dtype=np.float64)

        tmp = np.empty(ncsf, dtype=np.float64)
        for k in range(nlab):
            a = int(self.singles.a[k])
            r = int(self.singles.r[k])
            _apply_epq(self.drt, cache, self.psi0, p=a, q=r, out=tmp)
            out[:, k] = tmp

        self._basis_singles = np.asarray(out, dtype=np.float64, order="C")
        return self._basis_singles

    def expand(self, c: np.ndarray) -> np.ndarray:
        """Expand contracted coefficients to an uncontracted CSF vector.

        Parameters
        ----------
        c : np.ndarray
            Contracted coefficients. Shape: (nlab,).

        Returns
        -------
        y : np.ndarray
            Uncontracted CSF vector. Shape: (ncsf,).
        """

        c = np.asarray(c, dtype=np.float64).ravel()
        if int(c.size) != int(self.nlab):
            raise ValueError("contracted coefficient vector has wrong length")

        c0 = float(c[0])
        cs = c[1:]
        y = c0 * self.psi0
        if int(cs.size):
            b = self._build_singles_basis()
            y = y + b @ cs
        return np.asarray(y, dtype=np.float64)

    def project(self, z: np.ndarray) -> np.ndarray:
        """Project an uncontracted CSF vector onto the contracted basis.

        Parameters
        ----------
        z : np.ndarray
            Uncontracted CSF vector. Shape: (ncsf,).

        Returns
        -------
        c : np.ndarray
            Contracted coefficients. Shape: (nlab,).
        """

        z = np.asarray(z, dtype=np.float64).ravel()
        if int(z.size) != int(self.drt.ncsf):
            raise ValueError("uncontracted vector has wrong length")

        out = np.empty(self.nlab, dtype=np.float64)
        out[0] = float(np.dot(self.psi0, z))
        if int(self.singles.nlab):
            b = self._build_singles_basis()
            out[1:] = b.T @ z
        return out

    def overlap(self, c: np.ndarray) -> np.ndarray:
        """Return rho(c) = S c (contracted overlap matvec).

        Parameters
        ----------
        c : np.ndarray
            Contracted coefficients.

        Returns
        -------
        rho : np.ndarray
            Overlap-transformed coefficients.
        """

        y = self.expand(c)
        return self.project(y)

    def _compute_diag_precond(self) -> np.ndarray:
        """Compute diagonal preconditioner vector for contracted basis (singles only)."""
        import numpy as np

        nlab = self.nlab
        d = np.zeros(nlab, dtype=np.float64)

        # Compute reference energy E_ref = <psi0|H|psi0>
        hpsi0 = self._hop_uncontracted(self.psi0)
        E_ref = float(np.dot(self.psi0, hpsi0))
        d[0] = E_ref

        # Get integrals in 4-index form
        from asuka.cuguga.oracle import _restore_eri_4d
        h1e = np.asarray(self.h1e, dtype=np.float64)
        eri4 = _restore_eri_4d(self.eri, h1e.shape[0]).astype(np.float64, copy=False)

        # Get singles labels
        singles = self.singles

        # For singles: d = E_ref * S_μμ + (h_aa - h_rr + g_arar)
        if singles.nlab > 0:
            a_arr = np.asarray(singles.a, dtype=np.int32)
            r_arr = np.asarray(singles.r, dtype=np.int32)

            # Get diagonal of overlap S_μμ from basis norms
            basis_s = self._build_singles_basis()  # (ncsf, n_singles)
            S_diag_s = np.sum(basis_s * basis_s, axis=0)  # column norms squared

            # Compute approximate H diagonal
            for idx in range(singles.nlab):
                a = int(a_arr[idx])
                r = int(r_arr[idx])
                delta_e = (h1e[a, a] - h1e[r, r] + eri4[a, r, a, r])
                d[1 + idx] = E_ref * S_diag_s[idx] + delta_e

        return d

    def sigma(self, c: np.ndarray) -> np.ndarray:
        """Return sigma(c) = H c (contracted sigma matvec).

        Parameters
        ----------
        c : np.ndarray
            Contracted coefficients.

        Returns
        -------
        sigma : np.ndarray
            Contracted sigma vector.
        """

        y = self.expand(c)
        hy = self._hop_uncontracted(y)
        return self.project(hy)

    def solve(
        self,
        *,
        x0: np.ndarray | None = None,
        tol: float = 1e-10,
        max_cycle: int = 80,
        max_space: int = 25,
        s_tol: float = 1e-12,
    ) -> GeneralizedDavidsonResult:
        """Solve the lowest-root generalized eigenproblem in the contracted space.

        Parameters
        ----------
        x0 : np.ndarray | None, optional
            Initial guess.
        tol : float, optional
            Convergence tolerance.
        max_cycle : int, optional
            Maximum iterations.
        max_space : int, optional
            Maximum subspace size.
        s_tol : float, optional
            Overlap singularity tolerance.

        Returns
        -------
        GeneralizedDavidsonResult
            Result object.
        """

        if x0 is None:
            x0 = np.zeros(self.nlab, dtype=np.float64)
            x0[0] = 1.0
        else:
            x0 = np.asarray(x0, dtype=np.float64).ravel()
            if int(x0.size) != int(self.nlab):
                raise ValueError("x0 has wrong length")

        # Compute diagonal preconditioner vector
        diag_h = self._compute_diag_precond()

        def precond(r: np.ndarray, e: float) -> np.ndarray:
            """Diagonal preconditioner: r / (diag_h - e) with regularization."""
            denom = diag_h - float(e)
            # Regularize small denominators similar to uncontracted MRCISD
            denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
            return r / denom

        return generalized_davidson1(
            self.sigma,
            self.overlap,
            x0,
            precond=precond,
            tol=float(tol),
            max_cycle=int(max_cycle),
            max_space=int(max_space),
            s_tol=float(s_tol),
        )


@dataclass
class ICRefSinglesDoublesSemiDirect:
    """Semi-direct ic-MRCI backend for the [reference + singles + doubles] contracted space.

    Expands contracted vectors into an uncontracted CSF basis (DRT space),
    applies H there, then projects back.

    Attributes
    ----------
    drt : DRT
        Discrete Reaction Field.
    h1e : Any
        One-electron integrals.
    eri : Any
        Two-electron integrals.
    psi0 : np.ndarray
        Embedded reference vector in ``drt`` basis.
    singles : ICSingles
        Singles label set.
    doubles : ICDoubles
        Doubles label set.
    contract_nthreads : int, optional
        Number of threads for contraction.
    contract_blas_nthreads : int | None, optional
        Number of BLAS threads.
    precompute_epq : bool, optional
        Precompute EPQ actions.
    hop_map : Any | None, optional
        Hop mapping for projected contraction.
    contract_executor : ThreadPoolExecutor | None, optional
        Executor for parallel contraction.
    contract_workspace : Any | None, optional
        Workspace for contraction.
    use_gpu : bool, optional
        Use GPU for contraction.
    """

    drt: DRT
    h1e: Any
    eri: Any
    psi0: np.ndarray  # embedded reference vector in `drt` basis (length ncsf)
    singles: ICSingles
    doubles: ICDoubles
    contract_nthreads: int = 1
    contract_blas_nthreads: int | None = 1
    precompute_epq: bool = True
    hop_map: Any | None = None
    contract_executor: ThreadPoolExecutor | None = None
    contract_workspace: Any | None = None
    use_gpu: bool = False

    _gpu_setup_done: bool = False
    _gpu_resources: dict[str, Any] = field(default_factory=dict)


    _basis_singles: np.ndarray | None = None  # (ncsf, nlab_singles)
    _basis_doubles: np.ndarray | None = None  # (ncsf, nlab_doubles)

    def __post_init__(self) -> None:
        self.psi0 = np.asarray(self.psi0, dtype=np.float64).ravel()
        if int(self.psi0.size) != int(self.drt.ncsf):
            raise ValueError("psi0 must have length drt.ncsf")

        n = float(np.linalg.norm(self.psi0))
        if not np.isfinite(n) or n <= 0.0:
            raise ValueError("psi0 must have nonzero norm")
        self.psi0 = self.psi0 / n

        _validate_hop_map(self.drt, self.hop_map)

        if self.precompute_epq:
            from asuka.cuguga.oracle import precompute_epq_actions  # noqa: PLC0415

            # Precompute all E_pq actions (small-space convenience).
            precompute_epq_actions(self.drt)
            if self.hop_map is not None and getattr(self.hop_map, "drt_full", None) is not self.drt:
                precompute_epq_actions(self.hop_map.drt_full)

        self._gpu_setup_done = False
        self._gpu_resources = {}


    @property
    def nlab(self) -> int:
        return 1 + int(self.singles.nlab) + int(self.doubles.nlab)

    @property
    def _ns(self) -> int:
        return int(self.singles.nlab)

    @property
    def _nd(self) -> int:
        return int(self.doubles.nlab)

    def _hop_uncontracted(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        if int(x.size) != int(self.drt.ncsf):
            raise ValueError("uncontracted vector has wrong length")

        if self.use_gpu:
            return self._hop_gpu(x)


        if self.hop_map is not None:
            from asuka.mrci.projected_hop import projected_contract_h_csf_multi  # noqa: PLC0415

            return projected_contract_h_csf_multi(
                mapping=self.hop_map,
                h1e=self.h1e,
                eri=self.eri,
                xs_sub=x,
                precompute_epq_full=False,
                nthreads=int(self.contract_nthreads),
                blas_nthreads=self.contract_blas_nthreads,
                executor=self.contract_executor,
                workspace=self.contract_workspace,
            )[0]

        try:
            from asuka.integrals.df_integrals import DFMOIntegrals  # noqa: PLC0415
        except Exception:  # pragma: no cover
            DFMOIntegrals = None  # type: ignore[assignment]

        if DFMOIntegrals is not None and isinstance(self.eri, DFMOIntegrals):
            from asuka.integrals.contract_df import contract_h_csf_multi_df as _contract_df  # noqa: PLC0415

            return _contract_df(
                self.drt,
                self.h1e,
                self.eri,
                [x],
                precompute_epq=False,
                nthreads=int(self.contract_nthreads),
                blas_nthreads=self.contract_blas_nthreads,
                executor=self.contract_executor,
                workspace=self.contract_workspace,
            )[0]

        from asuka.contract import contract_h_csf_multi as _contract_dense  # noqa: PLC0415

        return _contract_dense(
            self.drt,
            self.h1e,
            self.eri,
            [x],
            precompute_epq=False,
            nthreads=int(self.contract_nthreads),
            blas_nthreads=self.contract_blas_nthreads,
            executor=self.contract_executor,
            workspace=self.contract_workspace,
        )[0]

    def _hop_gpu(self, x: np.ndarray) -> np.ndarray:
        import cupy as cp  # noqa: PLC0415
        from asuka.cuda.cuda_backend import (  # noqa: PLC0415
            make_device_drt,
            make_device_state_cache,
        )
        from asuka.cuda.mrci_hop import (  # noqa: PLC0415
            CudaMrciHopWorkspace,
            hop_cuda_epq_table,
            hop_cuda_projected,
        )
        from asuka.cuguga.oracle import _get_epq_action_cache, precompute_epq_actions  # noqa: PLC0415

        if not self._gpu_setup_done:
            norb = int(self.drt.norb)
            eri_4d = _restore_eri_4d(self.eri, norb)
            h_eff = self.h1e - 0.5 * np.einsum("pqqs->ps", eri_4d)
            eri_mat_t = 0.5 * eri_4d.reshape(norb * norb, -1).T

            self._gpu_resources["h_eff_d"] = cp.asarray(h_eff, dtype=cp.float64)
            self._gpu_resources["eri_mat_t_d"] = cp.asarray(eri_mat_t, dtype=cp.float64)

            from asuka.cuda.cuda_backend import (  # noqa: PLC0415
                build_epq_action_table_combined_device,
                make_device_drt,
                make_device_state_cache,
            )

            if self.hop_map is not None:
                drt_full = self.hop_map.drt_full
                drt_dev_full = make_device_drt(drt_full)
                state_dev_full = make_device_state_cache(drt=drt_full, drt_dev=drt_dev_full)
                self._gpu_resources["drt_dev_full"] = drt_dev_full
                self._gpu_resources["state_dev_full"] = state_dev_full
                self._gpu_resources["epq_table_full"] = build_epq_action_table_combined_device(
                    drt=drt_full, drt_dev=drt_dev_full, state_dev=state_dev_full
                )
                self._gpu_resources["sub_to_full_d"] = cp.asarray(self.hop_map.sub_to_full, dtype=cp.int64)
                self._gpu_resources["hop_ws_full"] = CudaMrciHopWorkspace.auto(
                    ncsf=int(drt_full.ncsf),
                    nops=int(norb) * int(norb),
                    sym_pair=bool(int(os.getenv("CUGUGA_MRCI_CUDA_SYM_PAIR", "0"))),
                )
                self._gpu_resources["x_full_buf"] = cp.empty((int(drt_full.ncsf),), dtype=cp.float64)
                self._gpu_resources["y_full_buf"] = cp.empty((int(drt_full.ncsf),), dtype=cp.float64)
            else:
                drt_dev = make_device_drt(self.drt)
                state_dev = make_device_state_cache(drt=self.drt, drt_dev=drt_dev)
                self._gpu_resources["drt_dev"] = drt_dev
                self._gpu_resources["state_dev"] = state_dev
                self._gpu_resources["epq_table"] = build_epq_action_table_combined_device(
                    drt=self.drt, drt_dev=drt_dev, state_dev=state_dev
                )
                self._gpu_resources["hop_ws"] = CudaMrciHopWorkspace.auto(
                    ncsf=int(self.drt.ncsf),
                    nops=int(norb) * int(norb),
                    sym_pair=bool(int(os.getenv("CUGUGA_MRCI_CUDA_SYM_PAIR", "0"))),
                )
            self._gpu_setup_done = True

        x_d = cp.asarray(x, dtype=cp.float64)
        if self.hop_map is not None:
            y_d = hop_cuda_projected(
                drt_full=self.hop_map.drt_full,
                drt_dev_full=self._gpu_resources["drt_dev_full"],
                state_dev_full=self._gpu_resources["state_dev_full"],
                epq_table_full=self._gpu_resources["epq_table_full"],
                h_eff=self._gpu_resources["h_eff_d"],
                eri_mat_t_full=self._gpu_resources["eri_mat_t_d"],
                x_sub=x_d,
                sub_to_full=self._gpu_resources["sub_to_full_d"],
                x_full_buf=self._gpu_resources["x_full_buf"],
                y_full_buf=self._gpu_resources["y_full_buf"],
                workspace_full=self._gpu_resources["hop_ws_full"],
            )
        else:
            y_d = hop_cuda_epq_table(
                drt=self.drt,
                drt_dev=self._gpu_resources["drt_dev"],
                state_dev=self._gpu_resources["state_dev"],
                epq_table=self._gpu_resources["epq_table"],
                h_eff=self._gpu_resources["h_eff_d"],
                eri_mat_t=self._gpu_resources["eri_mat_t_d"],
                x=x_d,
                workspace=self._gpu_resources["hop_ws"],
            )
        return cp.asnumpy(y_d)


    def _build_singles_basis(self) -> np.ndarray:
        if self._basis_singles is not None:
            return self._basis_singles

        ncsf = int(self.drt.ncsf)
        nlab = int(self.singles.nlab)
        if nlab == 0:
            self._basis_singles = np.zeros((ncsf, 0), dtype=np.float64)
            return self._basis_singles

        cache = _get_epq_action_cache(self.drt)
        out = np.empty((ncsf, nlab), dtype=np.float64)

        tmp = np.empty(ncsf, dtype=np.float64)
        for k in range(nlab):
            a = int(self.singles.a[k])
            r = int(self.singles.r[k])
            _apply_epq(self.drt, cache, self.psi0, p=a, q=r, out=tmp)
            out[:, k] = tmp

        self._basis_singles = np.asarray(out, dtype=np.float64, order="C")
        return self._basis_singles

    def _build_doubles_basis(self) -> np.ndarray:
        if self._basis_doubles is not None:
            return self._basis_doubles

        ncsf = int(self.drt.ncsf)
        nlab = int(self.doubles.nlab)
        if nlab == 0:
            self._basis_doubles = np.zeros((ncsf, 0), dtype=np.float64)
            return self._basis_doubles

        cache = _get_epq_action_cache(self.drt)
        out = np.empty((ncsf, nlab), dtype=np.float64)

        tmp1 = np.empty(ncsf, dtype=np.float64)
        tmp2 = np.empty(ncsf, dtype=np.float64)
        for k in range(nlab):
            a = int(self.doubles.a[k])
            b = int(self.doubles.b[k])
            r = int(self.doubles.r[k])
            s = int(self.doubles.s[k])

            _apply_epq(self.drt, cache, self.psi0, p=b, q=s, out=tmp1)
            _apply_epq(self.drt, cache, tmp1, p=a, q=r, out=tmp2)
            out[:, k] = tmp2

        self._basis_doubles = np.asarray(out, dtype=np.float64, order="C")
        return self._basis_doubles

    def expand(self, c: np.ndarray) -> np.ndarray:
        """Expand contracted coefficients to an uncontracted CSF vector.

        Parameters
        ----------
        c : np.ndarray
            Contracted coefficients. Shape: (nlab,).

        Returns
        -------
        y : np.ndarray
            Uncontracted CSF vector. Shape: (ncsf,).
        """

        c = np.asarray(c, dtype=np.float64).ravel()
        if int(c.size) != int(self.nlab):
            raise ValueError("contracted coefficient vector has wrong length")

        c0 = float(c[0])
        cs = c[1 : 1 + self._ns]
        cd = c[1 + self._ns :]

        y = c0 * self.psi0
        if int(cs.size):
            y = y + self._build_singles_basis() @ cs
        if int(cd.size):
            y = y + self._build_doubles_basis() @ cd
        return np.asarray(y, dtype=np.float64)

    def project(self, z: np.ndarray) -> np.ndarray:
        """Project an uncontracted CSF vector onto the contracted basis.

        Parameters
        ----------
        z : np.ndarray
            Uncontracted CSF vector. Shape: (ncsf,).

        Returns
        -------
        c : np.ndarray
            Contracted coefficients. Shape: (nlab,).
        """

        z = np.asarray(z, dtype=np.float64).ravel()
        if int(z.size) != int(self.drt.ncsf):
            raise ValueError("uncontracted vector has wrong length")

        out = np.empty(self.nlab, dtype=np.float64)
        out[0] = float(np.dot(self.psi0, z))
        if int(self._ns):
            b = self._build_singles_basis()
            out[1 : 1 + self._ns] = b.T @ z
        if int(self._nd):
            b = self._build_doubles_basis()
            out[1 + self._ns :] = b.T @ z
        return out

    def overlap(self, c: np.ndarray) -> np.ndarray:
        """Return rho(c) = S c (contracted overlap matvec).

        Parameters
        ----------
        c : np.ndarray
            Contracted coefficients.

        Returns
        -------
        rho : np.ndarray
            Overlap-transformed coefficients.
        """

        y = self.expand(c)
        return self.project(y)

    def _compute_diag_precond(self) -> np.ndarray:
        """Compute diagonal preconditioner vector for contracted basis.

        Returns a vector d of length nlab such that d[i] approximates
        <μ_i|H|μ_i> where |μ_i> are the contracted basis functions.
        """
        import numpy as np

        nlab = self.nlab
        d = np.zeros(nlab, dtype=np.float64)

        # Compute reference energy E_ref = <psi0|H|psi0>
        hpsi0 = self._hop_uncontracted(self.psi0)
        E_ref = float(np.dot(self.psi0, hpsi0))
        d[0] = E_ref

        # Get integrals in 4-index form
        from asuka.cuguga.oracle import _restore_eri_4d
        h1e = np.asarray(self.h1e, dtype=np.float64)
        eri4 = _restore_eri_4d(self.eri, h1e.shape[0]).astype(np.float64, copy=False)

        # Get singles and doubles labels
        singles = self.singles
        doubles = self.doubles

        # For singles: d = E_ref * S_μμ + (h_aa - h_rr + g_arar)
        if singles.nlab > 0:
            a_arr = np.asarray(singles.a, dtype=np.int32)
            r_arr = np.asarray(singles.r, dtype=np.int32)

            # Get diagonal of overlap S_μμ from basis norms
            basis_s = self._build_singles_basis()  # (ncsf, n_singles)
            S_diag_s = np.sum(basis_s * basis_s, axis=0)  # column norms squared

            # Compute approximate H diagonal
            for idx in range(singles.nlab):
                a = int(a_arr[idx])
                r = int(r_arr[idx])
                delta_e = (h1e[a, a] - h1e[r, r] + eri4[a, r, a, r])
                d[1 + idx] = E_ref * S_diag_s[idx] + delta_e

        # For doubles: d = E_ref * S_μμ + (h_aa + h_bb - h_rr - h_ss
        #               + g_arar + g_bsbs + g_asas + g_brbr - g_arbs - g_asbr)
        if doubles.nlab > 0:
            a_arr = np.asarray(doubles.a, dtype=np.int32)
            b_arr = np.asarray(doubles.b, dtype=np.int32)
            r_arr = np.asarray(doubles.r, dtype=np.int32)
            s_arr = np.asarray(doubles.s, dtype=np.int32)

            # Get diagonal of overlap S_μμ from basis norms
            basis_d = self._build_doubles_basis()  # (ncsf, n_doubles)
            S_diag_d = np.sum(basis_d * basis_d, axis=0)  # column norms squared

            # Compute approximate H diagonal
            offset = 1 + singles.nlab
            for idx in range(doubles.nlab):
                a = int(a_arr[idx])
                b = int(b_arr[idx])
                r = int(r_arr[idx])
                s = int(s_arr[idx])

                # Approximate excitation energy
                delta_e = (h1e[a, a] + h1e[b, b] - h1e[r, r] - h1e[s, s] +
                          eri4[a, r, a, r] + eri4[b, s, b, s] +
                          eri4[a, s, a, s] + eri4[b, r, b, r] -
                          eri4[a, r, b, s] - eri4[a, s, b, r])

                d[offset + idx] = E_ref * S_diag_d[idx] + delta_e

        return d

    def sigma(self, c: np.ndarray) -> np.ndarray:
        """Return sigma(c) = H c (contracted sigma matvec).

        Parameters
        ----------
        c : np.ndarray
            Contracted coefficients.

        Returns
        -------
        sigma : np.ndarray
            Contracted sigma vector.
        """

        y = self.expand(c)
        hy = self._hop_uncontracted(y)
        return self.project(hy)

    def solve(
        self,
        *,
        x0: np.ndarray | None = None,
        tol: float = 1e-10,
        max_cycle: int = 80,
        max_space: int = 25,
        s_tol: float = 1e-12,
    ) -> GeneralizedDavidsonResult:
        """Solve the lowest-root generalized eigenproblem in the contracted space.

        Parameters
        ----------
        x0 : np.ndarray | None, optional
            Initial guess.
        tol : float, optional
            Convergence tolerance.
        max_cycle : int, optional
            Maximum iterations.
        max_space : int, optional
            Maximum subspace size.
        s_tol : float, optional
            Overlap singularity tolerance.

        Returns
        -------
        GeneralizedDavidsonResult
            Result object.
        """

        if x0 is None:
            x0 = np.zeros(self.nlab, dtype=np.float64)
            x0[0] = 1.0
        else:
            x0 = np.asarray(x0, dtype=np.float64).ravel()
            if int(x0.size) != int(self.nlab):
                raise ValueError("x0 has wrong length")

        # Compute diagonal preconditioner vector
        diag_h = self._compute_diag_precond()

        def precond(r: np.ndarray, e: float) -> np.ndarray:
            """Diagonal preconditioner: r / (diag_h - e) with regularization."""
            denom = diag_h - float(e)
            # Regularize small denominators similar to uncontracted MRCISD
            denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
            return r / denom

        return generalized_davidson1(
            self.sigma,
            self.overlap,
            x0,
            precond=precond,
            tol=float(tol),
            max_cycle=int(max_cycle),
            max_space=int(max_space),
            s_tol=float(s_tol),
        )


@dataclass
class ICStronglyContractedSemiDirect:
    """Semi-direct ic-MRCI backend for a strongly contracted [ref + singles + doubles] space.

    Strongly contracted basis definition (first pass):
      - singles: |a> = Σ_r E_ar |Psi0>
      - doubles: |ab> = Σ_{r,s} E_ar E_bs |Psi0> (and for a=b, sum only r<=s)

    This is a correctness/validation backend; it explicitly builds the contracted
    basis vectors in the uncontracted restricted-DRT space and uses expand/H/project.
    """

    drt: DRT
    h1e: Any
    eri: Any
    psi0: np.ndarray  # embedded reference vector in `drt` basis (length ncsf)
    internal: np.ndarray  # (nI,), int32 internal orbital ids (in `drt` orbital indexing)
    singles: SCSingles
    doubles: SCDoubles
    allow_same_internal: bool = True
    contract_nthreads: int = 1
    contract_blas_nthreads: int | None = 1
    precompute_epq: bool = True
    hop_map: Any | None = None
    contract_executor: ThreadPoolExecutor | None = None
    contract_workspace: Any | None = None
    use_gpu: bool = False

    _gpu_setup_done: bool = False
    _gpu_resources: dict[str, Any] = field(default_factory=dict)


    _basis_singles: np.ndarray | None = None  # (ncsf, nlab_singles)
    _basis_doubles: np.ndarray | None = None  # (ncsf, nlab_doubles)

    def __post_init__(self) -> None:
        self.psi0 = np.asarray(self.psi0, dtype=np.float64).ravel()
        if int(self.psi0.size) != int(self.drt.ncsf):
            raise ValueError("psi0 must have length drt.ncsf")

        n = float(np.linalg.norm(self.psi0))
        if not np.isfinite(n) or n <= 0.0:
            raise ValueError("psi0 must have nonzero norm")
        self.psi0 = self.psi0 / n

        internal = np.unique(np.asarray(self.internal, dtype=np.int32).ravel())
        internal.sort()
        if int(internal.size) == 0:
            raise ValueError("internal must be non-empty for SC contractions")
        object.__setattr__(self, "internal", internal)

        object.__setattr__(self, "allow_same_internal", bool(self.allow_same_internal))

        _validate_hop_map(self.drt, self.hop_map)

        if self.precompute_epq:
            from asuka.cuguga.oracle import precompute_epq_actions  # noqa: PLC0415

            precompute_epq_actions(self.drt)
            if self.hop_map is not None and getattr(self.hop_map, "drt_full", None) is not self.drt:
                precompute_epq_actions(self.hop_map.drt_full)

        self._gpu_setup_done = False
        self._gpu_resources = {}


    @property
    def nlab(self) -> int:
        return 1 + int(self.singles.nlab) + int(self.doubles.nlab)

    @property
    def _ns(self) -> int:
        return int(self.singles.nlab)

    @property
    def _nd(self) -> int:
        return int(self.doubles.nlab)

    def _hop_uncontracted(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        if int(x.size) != int(self.drt.ncsf):
            raise ValueError("uncontracted vector has wrong length")

        if self.use_gpu:
            return self._hop_gpu(x)


        if self.hop_map is not None:
            from asuka.mrci.projected_hop import projected_contract_h_csf_multi  # noqa: PLC0415

            return projected_contract_h_csf_multi(
                mapping=self.hop_map,
                h1e=self.h1e,
                eri=self.eri,
                xs_sub=x,
                precompute_epq_full=False,
                nthreads=int(self.contract_nthreads),
                blas_nthreads=self.contract_blas_nthreads,
                executor=self.contract_executor,
                workspace=self.contract_workspace,
            )[0]

        try:
            from asuka.integrals.df_integrals import DFMOIntegrals  # noqa: PLC0415
        except Exception:  # pragma: no cover
            DFMOIntegrals = None  # type: ignore[assignment]

        if DFMOIntegrals is not None and isinstance(self.eri, DFMOIntegrals):
            from asuka.integrals.contract_df import contract_h_csf_multi_df as _contract_df  # noqa: PLC0415

            return _contract_df(
                self.drt,
                self.h1e,
                self.eri,
                [x],
                precompute_epq=False,
                nthreads=int(self.contract_nthreads),
                blas_nthreads=self.contract_blas_nthreads,
                executor=self.contract_executor,
                workspace=self.contract_workspace,
            )[0]

        from asuka.contract import contract_h_csf_multi as _contract_dense  # noqa: PLC0415

        return _contract_dense(
            self.drt,
            self.h1e,
            self.eri,
            [x],
            precompute_epq=False,
            nthreads=int(self.contract_nthreads),
            blas_nthreads=self.contract_blas_nthreads,
            executor=self.contract_executor,
            workspace=self.contract_workspace,
        )[0]

    def _hop_gpu(self, x: np.ndarray) -> np.ndarray:
        import cupy as cp  # noqa: PLC0415
        from asuka.cuda.cuda_backend import (  # noqa: PLC0415
            make_device_drt,
            make_device_state_cache,
        )
        from asuka.cuda.mrci_hop import (  # noqa: PLC0415
            CudaMrciHopWorkspace,
            hop_cuda_epq_table,
            hop_cuda_projected,
        )
        from asuka.cuguga.oracle import _get_epq_action_cache, precompute_epq_actions  # noqa: PLC0415

        if not self._gpu_setup_done:
            norb = int(self.drt.norb)

            eri_4d = _restore_eri_4d(self.eri, norb)
            h_eff = self.h1e - 0.5 * np.einsum("pqqs->ps", eri_4d)
            eri_mat_t = 0.5 * eri_4d.reshape(norb * norb, -1).T

            self._gpu_resources["h_eff_d"] = cp.asarray(h_eff, dtype=cp.float64)
            self._gpu_resources["eri_mat_t_d"] = cp.asarray(eri_mat_t, dtype=cp.float64)

            from asuka.cuda.cuda_backend import (  # noqa: PLC0415
                build_epq_action_table_combined_device,
                make_device_drt,
                make_device_state_cache,
            )

            if self.hop_map is not None:
                drt_full = self.hop_map.drt_full
                drt_dev_full = make_device_drt(drt_full)
                state_dev_full = make_device_state_cache(drt=drt_full, drt_dev=drt_dev_full)
                self._gpu_resources["drt_dev_full"] = drt_dev_full
                self._gpu_resources["state_dev_full"] = state_dev_full
                self._gpu_resources["epq_table_full"] = build_epq_action_table_combined_device(
                    drt=drt_full, drt_dev=drt_dev_full, state_dev=state_dev_full
                )
                self._gpu_resources["sub_to_full_d"] = cp.asarray(self.hop_map.sub_to_full, dtype=cp.int64)
                self._gpu_resources["hop_ws_full"] = CudaMrciHopWorkspace.auto(
                    ncsf=int(drt_full.ncsf),
                    nops=int(norb) * int(norb),
                    sym_pair=bool(int(os.getenv("CUGUGA_MRCI_CUDA_SYM_PAIR", "0"))),
                )
                self._gpu_resources["x_full_buf"] = cp.empty((int(drt_full.ncsf),), dtype=cp.float64)
                self._gpu_resources["y_full_buf"] = cp.empty((int(drt_full.ncsf),), dtype=cp.float64)
            else:
                drt_dev = make_device_drt(self.drt)
                state_dev = make_device_state_cache(drt=self.drt, drt_dev=drt_dev)
                self._gpu_resources["drt_dev"] = drt_dev
                self._gpu_resources["state_dev"] = state_dev
                self._gpu_resources["epq_table"] = build_epq_action_table_combined_device(
                    drt=self.drt, drt_dev=drt_dev, state_dev=state_dev
                )
                self._gpu_resources["hop_ws"] = CudaMrciHopWorkspace.auto(
                    ncsf=int(self.drt.ncsf),
                    nops=int(norb) * int(norb),
                    sym_pair=bool(int(os.getenv("CUGUGA_MRCI_CUDA_SYM_PAIR", "0"))),
                )
            self._gpu_setup_done = True

        x_d = cp.asarray(x, dtype=cp.float64)
        if self.hop_map is not None:
            y_d = hop_cuda_projected(
                drt_full=self.hop_map.drt_full,
                drt_dev_full=self._gpu_resources["drt_dev_full"],
                state_dev_full=self._gpu_resources["state_dev_full"],
                epq_table_full=self._gpu_resources["epq_table_full"],
                h_eff=self._gpu_resources["h_eff_d"],
                eri_mat_t_full=self._gpu_resources["eri_mat_t_d"],
                x_sub=x_d,
                sub_to_full=self._gpu_resources["sub_to_full_d"],
                x_full_buf=self._gpu_resources["x_full_buf"],
                y_full_buf=self._gpu_resources["y_full_buf"],
                workspace_full=self._gpu_resources["hop_ws_full"],
            )
        else:
            y_d = hop_cuda_epq_table(
                drt=self.drt,
                drt_dev=self._gpu_resources["drt_dev"],
                state_dev=self._gpu_resources["state_dev"],
                epq_table=self._gpu_resources["epq_table"],
                h_eff=self._gpu_resources["h_eff_d"],
                eri_mat_t=self._gpu_resources["eri_mat_t_d"],
                x=x_d,
                workspace=self._gpu_resources["hop_ws"],
            )
        return cp.asnumpy(y_d)


    def _build_singles_basis(self) -> np.ndarray:
        if self._basis_singles is not None:
            return self._basis_singles

        ncsf = int(self.drt.ncsf)
        nlab = int(self.singles.nlab)
        if nlab == 0:
            self._basis_singles = np.zeros((ncsf, 0), dtype=np.float64)
            return self._basis_singles

        cache = _get_epq_action_cache(self.drt)
        out = np.empty((ncsf, nlab), dtype=np.float64)

        tmp = np.empty(ncsf, dtype=np.float64)
        acc = np.empty(ncsf, dtype=np.float64)
        for k in range(nlab):
            a = int(self.singles.a[k])
            acc.fill(0.0)
            for r in self.internal.tolist():
                _apply_epq(self.drt, cache, self.psi0, p=a, q=int(r), out=tmp)
                acc += tmp
            out[:, k] = acc

        self._basis_singles = np.asarray(out, dtype=np.float64, order="C")
        return self._basis_singles

    def _build_doubles_basis(self) -> np.ndarray:
        if self._basis_doubles is not None:
            return self._basis_doubles

        ncsf = int(self.drt.ncsf)
        nlab = int(self.doubles.nlab)
        if nlab == 0:
            self._basis_doubles = np.zeros((ncsf, 0), dtype=np.float64)
            return self._basis_doubles

        cache = _get_epq_action_cache(self.drt)
        out = np.empty((ncsf, nlab), dtype=np.float64)

        tmp1 = np.empty(ncsf, dtype=np.float64)
        tmp2 = np.empty(ncsf, dtype=np.float64)
        acc = np.empty(ncsf, dtype=np.float64)
        internal = self.internal.tolist()

        for k in range(nlab):
            a = int(self.doubles.a[k])
            b = int(self.doubles.b[k])
            acc.fill(0.0)

            if a == b:
                for ir, r in enumerate(internal):
                    start_s = ir if self.allow_same_internal else ir + 1
                    for s in internal[start_s:]:
                        _apply_epq(self.drt, cache, self.psi0, p=a, q=int(s), out=tmp1)
                        _apply_epq(self.drt, cache, tmp1, p=a, q=int(r), out=tmp2)
                        acc += tmp2
            else:
                for r in internal:
                    _apply_epq(self.drt, cache, self.psi0, p=a, q=int(r), out=tmp1)
                    for s in internal:
                        if (not self.allow_same_internal) and int(r) == int(s):
                            continue
                        _apply_epq(self.drt, cache, tmp1, p=b, q=int(s), out=tmp2)
                        acc += tmp2

            out[:, k] = acc

        self._basis_doubles = np.asarray(out, dtype=np.float64, order="C")
        return self._basis_doubles

    def expand(self, c: np.ndarray) -> np.ndarray:
        """Expand contracted coefficients to an uncontracted CSF vector.

        Parameters
        ----------
        c : np.ndarray
            Contracted coefficients. Shape: (nlab,).

        Returns
        -------
        y : np.ndarray
            Uncontracted CSF vector. Shape: (ncsf,).
        """

        c = np.asarray(c, dtype=np.float64).ravel()
        if int(c.size) != int(self.nlab):
            raise ValueError("contracted coefficient vector has wrong length")

        c0 = float(c[0])
        cs = c[1 : 1 + self._ns]
        cd = c[1 + self._ns :]

        y = c0 * self.psi0
        if int(cs.size):
            y = y + self._build_singles_basis() @ cs
        if int(cd.size):
            y = y + self._build_doubles_basis() @ cd
        return np.asarray(y, dtype=np.float64)

    def project(self, z: np.ndarray) -> np.ndarray:
        """Project an uncontracted CSF vector onto the contracted basis.

        Parameters
        ----------
        z : np.ndarray
            Uncontracted CSF vector. Shape: (ncsf,).

        Returns
        -------
        c : np.ndarray
            Contracted coefficients. Shape: (nlab,).
        """

        z = np.asarray(z, dtype=np.float64).ravel()
        if int(z.size) != int(self.drt.ncsf):
            raise ValueError("uncontracted vector has wrong length")

        out = np.empty(self.nlab, dtype=np.float64)
        out[0] = float(np.dot(self.psi0, z))
        if int(self._ns):
            b = self._build_singles_basis()
            out[1 : 1 + self._ns] = b.T @ z
        if int(self._nd):
            b = self._build_doubles_basis()
            out[1 + self._ns :] = b.T @ z
        return out

    def overlap(self, c: np.ndarray) -> np.ndarray:
        """Return rho(c) = S c (contracted overlap matvec).

        Parameters
        ----------
        c : np.ndarray
            Contracted coefficients.

        Returns
        -------
        rho : np.ndarray
            Overlap-transformed coefficients.
        """

        y = self.expand(c)
        return self.project(y)

    def sigma(self, c: np.ndarray) -> np.ndarray:
        """Return sigma(c) = H c (contracted sigma matvec).

        Parameters
        ----------
        c : np.ndarray
            Contracted coefficients.

        Returns
        -------
        sigma : np.ndarray
            Contracted sigma vector.
        """

        y = self.expand(c)
        hy = self._hop_uncontracted(y)
        return self.project(hy)

    def _compute_diag_precond(self) -> np.ndarray:
        """Compute diagonal preconditioner vector for contracted basis.

        Returns
        -------
        d : np.ndarray
            Diagonal preconditioner. Shape: (nlab,).

        Notes
        -----
        Currently returns zero vector (no diagonal preconditioning).
        """

        return np.zeros(self.nlab, dtype=np.float64)

    def solve(
        self,
        *,
        x0: np.ndarray | None = None,
        tol: float = 1e-10,
        max_cycle: int = 80,
        max_space: int = 25,
        s_tol: float = 1e-12,
    ) -> GeneralizedDavidsonResult:
        """Solve the lowest-root generalized eigenproblem in the contracted space.

        Parameters
        ----------
        x0 : np.ndarray | None, optional
            Initial guess.
        tol : float, optional
            Convergence tolerance.
        max_cycle : int, optional
            Maximum iterations.
        max_space : int, optional
            Maximum subspace size.
        s_tol : float, optional
            Overlap singularity tolerance.

        Returns
        -------
        GeneralizedDavidsonResult
            Result object.
        """

        if x0 is None:
            x0 = np.zeros(self.nlab, dtype=np.float64)
            x0[0] = 1.0
        else:
            x0 = np.asarray(x0, dtype=np.float64).ravel()
            if int(x0.size) != int(self.nlab):
                raise ValueError("x0 has wrong length")

        # Compute diagonal preconditioner vector
        diag_h = self._compute_diag_precond()

        def precond(r: np.ndarray, e: float) -> np.ndarray:
            """Diagonal preconditioner: r / (diag_h - e) with regularization."""
            denom = diag_h - float(e)
            # Regularize small denominators similar to uncontracted MRCISD
            denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
            return r / denom

        return generalized_davidson1(
            self.sigma,
            self.overlap,
            x0,
            precond=precond,
            tol=float(tol),
            max_cycle=int(max_cycle),
            max_space=int(max_space),
            s_tol=float(s_tol),
        )


@dataclass
class ICStronglyContractedSemiDirectOTF:
    """Semi-direct strongly contracted backend that does not build explicit SC basis matrices.

    This class computes `expand(c)` and `project(z)` on the fly using repeated `E_{pq}`
    applications and dot products, avoiding persistent storage of length-`ncsf` SC
    basis vectors.

    Strongly contracted basis definition (first pass):
      - singles: |a> = Σ_r E_ar |Psi0>
      - doubles: |ab> = Σ_{r,s} E_ar E_bs |Psi0> (and for a=b, sum only r<=s)
    """

    drt: DRT
    h1e: Any
    eri: Any
    psi0: np.ndarray  # embedded reference vector in `drt` basis (length ncsf)
    internal: np.ndarray  # (nI,), int32 internal orbital ids (in `drt` orbital indexing)
    singles: SCSingles
    doubles: SCDoubles
    allow_same_internal: bool = True
    contract_nthreads: int = 1
    contract_blas_nthreads: int | None = 1
    precompute_epq: bool = True
    hop_map: Any | None = None
    contract_executor: ThreadPoolExecutor | None = None
    contract_workspace: Any | None = None
    use_gpu: bool = False

    _gpu_setup_done: bool = False
    _gpu_resources: dict[str, Any] = field(default_factory=dict)


    def __post_init__(self) -> None:
        self.psi0 = np.asarray(self.psi0, dtype=np.float64).ravel()
        if int(self.psi0.size) != int(self.drt.ncsf):
            raise ValueError("psi0 must have length drt.ncsf")

        n = float(np.linalg.norm(self.psi0))
        if not np.isfinite(n) or n <= 0.0:
            raise ValueError("psi0 must have nonzero norm")
        self.psi0 = self.psi0 / n

        internal = np.unique(np.asarray(self.internal, dtype=np.int32).ravel())
        internal.sort()
        if int(internal.size) == 0:
            raise ValueError("internal must be non-empty for SC contractions")
        object.__setattr__(self, "internal", internal)

        object.__setattr__(self, "allow_same_internal", bool(self.allow_same_internal))

        _validate_hop_map(self.drt, self.hop_map)

        if self.precompute_epq:
            from asuka.cuguga.oracle import precompute_epq_actions  # noqa: PLC0415

            precompute_epq_actions(self.drt)
            if self.hop_map is not None and getattr(self.hop_map, "drt_full", None) is not self.drt:
                precompute_epq_actions(self.hop_map.drt_full)

        self._gpu_setup_done = False
        self._gpu_resources = {}


    @property
    def nlab(self) -> int:
        return 1 + int(self.singles.nlab) + int(self.doubles.nlab)

    @property
    def _ns(self) -> int:
        return int(self.singles.nlab)

    @property
    def _nd(self) -> int:
        return int(self.doubles.nlab)

    def _hop_uncontracted(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        if int(x.size) != int(self.drt.ncsf):
            raise ValueError("uncontracted vector has wrong length")

        if self.use_gpu:
            return self._hop_gpu(x)


        if self.hop_map is not None:
            from asuka.mrci.projected_hop import projected_contract_h_csf_multi  # noqa: PLC0415

            return projected_contract_h_csf_multi(
                mapping=self.hop_map,
                h1e=self.h1e,
                eri=self.eri,
                xs_sub=x,
                precompute_epq_full=False,
                nthreads=int(self.contract_nthreads),
                blas_nthreads=self.contract_blas_nthreads,
                executor=self.contract_executor,
                workspace=self.contract_workspace,
            )[0]

        try:
            from asuka.integrals.df_integrals import DFMOIntegrals  # noqa: PLC0415
        except Exception:  # pragma: no cover
            DFMOIntegrals = None  # type: ignore[assignment]

        if DFMOIntegrals is not None and isinstance(self.eri, DFMOIntegrals):
            from asuka.integrals.contract_df import contract_h_csf_multi_df as _contract_df  # noqa: PLC0415

            return _contract_df(
                self.drt,
                self.h1e,
                self.eri,
                [x],
                precompute_epq=False,
                nthreads=int(self.contract_nthreads),
                blas_nthreads=self.contract_blas_nthreads,
                executor=self.contract_executor,
                workspace=self.contract_workspace,
            )[0]

        from asuka.contract import contract_h_csf_multi as _contract_dense  # noqa: PLC0415

        return _contract_dense(
            self.drt,
            self.h1e,
            self.eri,
            [x],
            precompute_epq=False,
            nthreads=int(self.contract_nthreads),
            blas_nthreads=self.contract_blas_nthreads,
            executor=self.contract_executor,
            workspace=self.contract_workspace,
        )[0]

    def _hop_gpu(self, x: np.ndarray) -> np.ndarray:
        import cupy as cp  # noqa: PLC0415
        from asuka.cuda.cuda_backend import (  # noqa: PLC0415
            make_device_drt,
            make_device_state_cache,
        )
        from asuka.cuda.mrci_hop import (  # noqa: PLC0415
            CudaMrciHopWorkspace,
            hop_cuda_epq_table,
            hop_cuda_projected,
        )
        from asuka.cuguga.oracle import _get_epq_action_cache, precompute_epq_actions  # noqa: PLC0415

        if not self._gpu_setup_done:
            norb = int(self.drt.norb)

            eri_4d = _restore_eri_4d(self.eri, norb)
            h_eff = self.h1e - 0.5 * np.einsum("pqqs->ps", eri_4d)
            eri_mat_t = 0.5 * eri_4d.reshape(norb * norb, -1).T

            self._gpu_resources["h_eff_d"] = cp.asarray(h_eff, dtype=cp.float64)
            self._gpu_resources["eri_mat_t_d"] = cp.asarray(eri_mat_t, dtype=cp.float64)

            from asuka.cuda.cuda_backend import (  # noqa: PLC0415
                build_epq_action_table_combined_device,
                make_device_drt,
                make_device_state_cache,
            )

            if self.hop_map is not None:
                drt_full = self.hop_map.drt_full
                drt_dev_full = make_device_drt(drt_full)
                state_dev_full = make_device_state_cache(drt=drt_full, drt_dev=drt_dev_full)
                self._gpu_resources["drt_dev_full"] = drt_dev_full
                self._gpu_resources["state_dev_full"] = state_dev_full
                self._gpu_resources["epq_table_full"] = build_epq_action_table_combined_device(
                    drt=drt_full, drt_dev=drt_dev_full, state_dev=state_dev_full
                )
                self._gpu_resources["sub_to_full_d"] = cp.asarray(self.hop_map.sub_to_full, dtype=cp.int64)
                self._gpu_resources["hop_ws_full"] = CudaMrciHopWorkspace.auto(
                    ncsf=int(drt_full.ncsf),
                    nops=int(norb) * int(norb),
                    sym_pair=bool(int(os.getenv("CUGUGA_MRCI_CUDA_SYM_PAIR", "0"))),
                )
                self._gpu_resources["x_full_buf"] = cp.empty((int(drt_full.ncsf),), dtype=cp.float64)
                self._gpu_resources["y_full_buf"] = cp.empty((int(drt_full.ncsf),), dtype=cp.float64)
            else:
                drt_dev = make_device_drt(self.drt)
                state_dev = make_device_state_cache(drt=self.drt, drt_dev=drt_dev)
                self._gpu_resources["drt_dev"] = drt_dev
                self._gpu_resources["state_dev"] = state_dev
                self._gpu_resources["epq_table"] = build_epq_action_table_combined_device(
                    drt=self.drt, drt_dev=drt_dev, state_dev=state_dev
                )
                self._gpu_resources["hop_ws"] = CudaMrciHopWorkspace.auto(
                    ncsf=int(self.drt.ncsf),
                    nops=int(norb) * int(norb),
                    sym_pair=bool(int(os.getenv("CUGUGA_MRCI_CUDA_SYM_PAIR", "0"))),
                )
            self._gpu_setup_done = True

        x_d = cp.asarray(x, dtype=cp.float64)
        if self.hop_map is not None:
            y_d = hop_cuda_projected(
                drt_full=self.hop_map.drt_full,
                drt_dev_full=self._gpu_resources["drt_dev_full"],
                state_dev_full=self._gpu_resources["state_dev_full"],
                epq_table_full=self._gpu_resources["epq_table_full"],
                h_eff=self._gpu_resources["h_eff_d"],
                eri_mat_t_full=self._gpu_resources["eri_mat_t_d"],
                x_sub=x_d,
                sub_to_full=self._gpu_resources["sub_to_full_d"],
                x_full_buf=self._gpu_resources["x_full_buf"],
                y_full_buf=self._gpu_resources["y_full_buf"],
                workspace_full=self._gpu_resources["hop_ws_full"],
            )
        else:
            y_d = hop_cuda_epq_table(
                drt=self.drt,
                drt_dev=self._gpu_resources["drt_dev"],
                state_dev=self._gpu_resources["state_dev"],
                epq_table=self._gpu_resources["epq_table"],
                h_eff=self._gpu_resources["h_eff_d"],
                eri_mat_t=self._gpu_resources["eri_mat_t_d"],
                x=x_d,
                workspace=self._gpu_resources["hop_ws"],
            )
        return cp.asnumpy(y_d)


    def expand(self, c: np.ndarray) -> np.ndarray:
        """Expand contracted coefficients to an uncontracted CSF vector.

        Parameters
        ----------
        c : np.ndarray
            Contracted coefficients. Shape: (nlab,).

        Returns
        -------
        y : np.ndarray
            Uncontracted CSF vector. Shape: (ncsf,).
        """

        c = np.asarray(c, dtype=np.float64).ravel()
        if int(c.size) != int(self.nlab):
            raise ValueError("contracted coefficient vector has wrong length")

        c0 = float(c[0])
        cs = c[1 : 1 + self._ns]
        cd = c[1 + self._ns :]

        y = c0 * self.psi0
        if (not int(cs.size)) and (not int(cd.size)):
            return np.asarray(y, dtype=np.float64)

        cache = _get_epq_action_cache(self.drt)
        ncsf = int(self.drt.ncsf)
        tmp1 = np.empty(ncsf, dtype=np.float64)
        tmp2 = np.empty(ncsf, dtype=np.float64)
        acc = np.empty(ncsf, dtype=np.float64)

        internal = self.internal.tolist()

        for k in range(int(cs.size)):
            ck = float(cs[k])
            if ck == 0.0:
                continue
            a = int(self.singles.a[k])
            acc.fill(0.0)
            for r in internal:
                _apply_epq(self.drt, cache, self.psi0, p=a, q=int(r), out=tmp1)
                acc += tmp1
            y = y + ck * acc

        for k in range(int(cd.size)):
            ck = float(cd[k])
            if ck == 0.0:
                continue
            a = int(self.doubles.a[k])
            b = int(self.doubles.b[k])
            acc.fill(0.0)

            if a == b:
                for ir, r in enumerate(internal):
                    start_s = ir if self.allow_same_internal else ir + 1
                    for s in internal[start_s:]:
                        _apply_epq(self.drt, cache, self.psi0, p=a, q=int(s), out=tmp1)
                        _apply_epq(self.drt, cache, tmp1, p=a, q=int(r), out=tmp2)
                        acc += tmp2
            else:
                for r in internal:
                    _apply_epq(self.drt, cache, self.psi0, p=a, q=int(r), out=tmp1)
                    for s in internal:
                        if (not self.allow_same_internal) and int(r) == int(s):
                            continue
                        _apply_epq(self.drt, cache, tmp1, p=b, q=int(s), out=tmp2)
                        acc += tmp2

            y = y + ck * acc

        return np.asarray(y, dtype=np.float64)

    def project(self, z: np.ndarray) -> np.ndarray:
        """Project an uncontracted CSF vector onto the contracted basis.

        Parameters
        ----------
        z : np.ndarray
            Uncontracted CSF vector. Shape: (ncsf,).

        Returns
        -------
        c : np.ndarray
            Contracted coefficients. Shape: (nlab,).
        """

        z = np.asarray(z, dtype=np.float64).ravel()
        if int(z.size) != int(self.drt.ncsf):
            raise ValueError("uncontracted vector has wrong length")

        out = np.zeros(self.nlab, dtype=np.float64)
        out[0] = float(np.dot(self.psi0, z))
        if self.nlab == 1:
            return out

        cache = _get_epq_action_cache(self.drt)
        ncsf = int(self.drt.ncsf)
        tmp1 = np.empty(ncsf, dtype=np.float64)
        tmp2 = np.empty(ncsf, dtype=np.float64)
        internal = self.internal.tolist()

        # Singles: p_a = <Psi0| Σ_r E_ra |z>
        for k in range(self._ns):
            a = int(self.singles.a[k])
            acc = 0.0
            for r in internal:
                _apply_epq(self.drt, cache, z, p=int(r), q=a, out=tmp1)
                acc += float(np.dot(self.psi0, tmp1))
            out[1 + k] = float(acc)

        # Doubles: p_ab = <Psi0| Σ_{r,s} E_sb E_ra |z>
        base = 1 + self._ns
        for k in range(self._nd):
            a = int(self.doubles.a[k])
            b = int(self.doubles.b[k])
            acc = 0.0
            if a == b:
                for ir, r in enumerate(internal):
                    start_s = ir if self.allow_same_internal else ir + 1
                    _apply_epq(self.drt, cache, z, p=int(r), q=a, out=tmp1)  # E_ra|z>
                    for s in internal[start_s:]:
                        _apply_epq(self.drt, cache, tmp1, p=int(s), q=b, out=tmp2)  # E_sb E_ra |z>
                        acc += float(np.dot(self.psi0, tmp2))
            else:
                for r in internal:
                    _apply_epq(self.drt, cache, z, p=int(r), q=a, out=tmp1)  # E_ra|z>
                    for s in internal:
                        if (not self.allow_same_internal) and int(r) == int(s):
                            continue
                        _apply_epq(self.drt, cache, tmp1, p=int(s), q=b, out=tmp2)  # E_sb E_ra |z>
                        acc += float(np.dot(self.psi0, tmp2))
            out[base + k] = float(acc)

        return out

    def overlap(self, c: np.ndarray) -> np.ndarray:
        """Return rho(c) = S c (contracted overlap matvec).

        Parameters
        ----------
        c : np.ndarray
            Contracted coefficients.

        Returns
        -------
        rho : np.ndarray
            Overlap-transformed coefficients.
        """

        y = self.expand(c)
        return self.project(y)

    def sigma(self, c: np.ndarray) -> np.ndarray:
        """Return sigma(c) = H c (contracted sigma matvec).

        Parameters
        ----------
        c : np.ndarray
            Contracted coefficients.

        Returns
        -------
        sigma : np.ndarray
            Contracted sigma vector.
        """

        y = self.expand(c)
        hy = self._hop_uncontracted(y)
        return self.project(hy)

    def _compute_diag_precond(self) -> np.ndarray:
        """Compute diagonal preconditioner vector for contracted basis.

        Returns
        -------
        d : np.ndarray
            Diagonal preconditioner. Shape: (nlab,).

        Notes
        -----
        Currently returns zero vector (no diagonal preconditioning).
        """

        return np.zeros(self.nlab, dtype=np.float64)

    def solve(
        self,
        *,
        x0: np.ndarray | None = None,
        tol: float = 1e-10,
        max_cycle: int = 80,
        max_space: int = 25,
        s_tol: float = 1e-12,
    ) -> GeneralizedDavidsonResult:
        """Solve the lowest-root generalized eigenproblem in the contracted space.

        Parameters
        ----------
        x0 : np.ndarray | None, optional
            Initial guess.
        tol : float, optional
            Convergence tolerance.
        max_cycle : int, optional
            Maximum iterations.
        max_space : int, optional
            Maximum subspace size.
        s_tol : float, optional
            Overlap singularity tolerance.

        Returns
        -------
        GeneralizedDavidsonResult
            Result object.
        """

        if x0 is None:
            x0 = np.zeros(self.nlab, dtype=np.float64)
            x0[0] = 1.0
        else:
            x0 = np.asarray(x0, dtype=np.float64).ravel()
            if int(x0.size) != int(self.nlab):
                raise ValueError("x0 has wrong length")

        # Compute diagonal preconditioner vector
        diag_h = self._compute_diag_precond()

        def precond(r: np.ndarray, e: float) -> np.ndarray:
            """Diagonal preconditioner: r / (diag_h - e) with regularization."""
            denom = diag_h - float(e)
            # Regularize small denominators similar to uncontracted MRCISD
            denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
            return r / denom

        return generalized_davidson1(
            self.sigma,
            self.overlap,
            x0,
            precond=precond,
            tol=float(tol),
            max_cycle=int(max_cycle),
            max_space=int(max_space),
            s_tol=float(s_tol),
        )
