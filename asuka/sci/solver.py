"""GUGASCISolver — Selected CI solver implementing ASUKA's fcisolver protocol.

Subclasses GUGAFCISolver and overrides kernel() to use CIPSI/HB-SCI.
All RDM, contract_2e, absorb_h1e methods are inherited and work unchanged
since kernel() returns a standard dense CI vector.

Usage:
    solver = GUGASCISolver(twos=0, nroots=1, max_ncsf=10000)
    result = run_casci_df(scf_out, ..., fcisolver=solver)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from asuka.solver import GUGAFCISolver


class GUGASCISolver(GUGAFCISolver):
    """Selected CI solver (CIPSI / HB-SCI) as a drop-in for GUGAFCISolver.

    Runs SCI to build a compact variational wavefunction, then expands
    the sparse CI vector to dense so that all inherited RDM / sigma
    methods work without modification.

    Parameters
    ----------
    twos : int
        Target 2*S_z.
    nroots : int
        Number of roots.
    max_ncsf : int
        Maximum selected space size.
    init_ncsf : int
        Initial selected space size.
    grow_by : int
        CSFs added per macro iteration.
    max_iter : int
        Maximum macro growth iterations.
    selection_mode : str
        ``'heat_bath'`` (HB-SCI) or ``'frontier_hash'`` (CIPSI).
    hb_epsilon : float
        Heat-bath screening threshold.
    hf_seed : bool
        Seed the initial space with the HF determinant.
    """

    _is_sci = True

    def __init__(
        self,
        *,
        twos: int | None = None,
        nroots: int = 1,
        max_ncsf: int = 10000,
        init_ncsf: int = 256,
        grow_by: int = 2000,
        max_iter: int = 10,
        selection_mode: str = "heat_bath",
        hb_epsilon: float = 1e-4,
        hb_eps_schedule: str = "fixed",
        hb_eps_init: float = 1e-3,
        hb_eps_final: float = 1e-5,
        davidson_tol: float = 1e-7,
        davidson_max_cycle: int = 40,
        davidson_max_space: int = 16,
        hf_seed: bool = True,
        backend: str = "auto",
        orbsym=None,
        wfnsym=None,
    ):
        super().__init__(twos=twos, nroots=nroots, orbsym=orbsym, wfnsym=wfnsym)
        self.sci_max_ncsf = max_ncsf
        self.sci_init_ncsf = init_ncsf
        self.sci_grow_by = grow_by
        self.sci_max_iter = max_iter
        self.sci_selection_mode = selection_mode
        self.sci_hb_epsilon = hb_epsilon
        self.sci_hb_eps_schedule = hb_eps_schedule
        self.sci_hb_eps_init = hb_eps_init
        self.sci_hb_eps_final = hb_eps_final
        self.sci_davidson_tol = davidson_tol
        self.sci_davidson_max_cycle = davidson_max_cycle
        self.sci_davidson_max_space = davidson_max_space
        self.sci_hf_seed = hf_seed
        self.sci_backend = backend

    def _resolve_backend(self) -> str:
        """Return the effective CIPSI backend, promoting 'auto' to cuda_key64 when GPU is available."""
        backend = str(self.sci_backend)
        if backend == "auto":
            try:
                import cupy as cp
                if int(cp.cuda.runtime.getDeviceCount()) > 0:
                    backend = "cuda_key64"
            except Exception:
                pass
        return backend

    def kernel(
        self,
        h1e,
        eri,
        norb: int,
        nelec: int | tuple[int, int],
        ci0=None,
        ecore: float = 0.0,
        nroots: int | None = None,
        **kwargs,
    ):
        """Run SCI, return (energy, dense_ci) — same contract as GUGAFCISolver."""
        from asuka.cuguga.drt import build_drt
        from asuka.integrals.df_integrals import DFMOIntegrals, DeviceDFMOIntegrals
        from asuka.sci.gpu_cipsi import run_cipsi_trials

        if nroots is None:
            nroots = self.nroots

        neleca, nelecb, nelec_total, sz_twos = self._normalize_nelec(nelec)
        twos = self._get_twos_target(neleca, nelecb)

        # Convert CuPy arrays to numpy (CASSCF Newton adapter may pass GPU arrays)
        try:
            import cupy as cp
            if isinstance(h1e, cp.ndarray):
                h1e = cp.asnumpy(h1e)
            if isinstance(eri, cp.ndarray):
                eri = np.asarray(cp.asnumpy(eri))
        except ImportError:
            pass

        backend = self._resolve_backend()
        if isinstance(eri, DFMOIntegrals) and not isinstance(eri, DeviceDFMOIntegrals):
            try:
                import cupy as cp
                eri = eri.to_device(cp, with_eri_mat=False)
            except Exception:
                pass

        drt = build_drt(
            norb=int(norb), nelec=nelec_total, twos_target=twos,
            orbsym=self.orbsym, wfnsym=self.wfnsym,
        )

        # Build HF seed
        cipsi_ci0 = None
        if self.sci_hf_seed:
            cipsi_ci0 = self._build_hf_ci0(drt, int(norb), nelec_total)

        res = run_cipsi_trials(
            drt, h1e, eri,
            ecore=float(ecore),
            nroots=int(nroots),
            ci0=cipsi_ci0,
            init_ncsf=int(self.sci_init_ncsf),
            max_ncsf=int(self.sci_max_ncsf),
            grow_by=int(self.sci_grow_by),
            max_iter=int(self.sci_max_iter),
            selection_mode=str(self.sci_selection_mode),
            hb_epsilon=float(self.sci_hb_epsilon),
            hb_eps_schedule=str(self.sci_hb_eps_schedule),
            hb_eps_init=float(self.sci_hb_eps_init),
            hb_eps_final=float(self.sci_hb_eps_final),
            davidson_max_cycle=int(self.sci_davidson_max_cycle),
            davidson_max_space=int(self.sci_davidson_max_space),
            davidson_tol=float(self.sci_davidson_tol),
            backend=backend,
            epq_mode="no_epq_support_aware",
            workspace_kwargs={
                "projected_solver_gpu": True,
                "projected_solver_matrix_free": True,
            },
        )

        sel_idx = np.asarray(res.sel_idx, dtype=np.int64)
        ci_sel = np.asarray(res.ci_sel, dtype=np.float64)

        # Store sparse state for efficient GPU RDM computation
        self._sci_drt = drt
        self._sci_sel_idx = sel_idx
        self._sci_ci_sel = ci_sel
        self._sci_nroots = int(nroots)

        # Cache GPU-resident DRT for the T-matrix kernel
        try:
            from asuka.cuda.cuda_backend import make_device_drt
            self._sci_drt_dev = make_device_drt(drt)
        except Exception:
            self._sci_drt_dev = None

        # Build dense CI vector and store for external use (e.g., CASPT2).
        # During CASSCF iterations the CASSCF driver only needs a CI identity
        # sentinel: make_rdm12 ignores civec and uses the stored sparse arrays.
        # Returning the full 449M-element dense array causes two expensive ops
        # per CASSCF macro-iteration:
        #   1. c.copy() in casscf.py triggers ~878k OS page faults (~3.6s in WSL2)
        #   2. _to_xp_f64(ci1, cp) uploads 3.59 GB over PCIe per micro-iter (~2s)
        # We therefore return a 1-element stub to the CASSCF driver and store
        # the real dense CI in self.ci for post-CASSCF use.
        ncsf = int(drt.ncsf)
        if nroots == 1:
            e_tot = float(np.asarray(res.e_var, dtype=np.float64).ravel()[0])
            self.converged = True
            self.eci = e_tot
            # Build the dense CI only when it fits comfortably in memory
            # (threshold ~8 GB). For spaces like CAS(22,22) with 79.5B CSFs
            # (636 GB), we skip the allocation and keep only the sparse rep.
            _MAX_DENSE_NCSF = 1_000_000_000
            if ncsf <= _MAX_DENSE_NCSF:
                ci_dense = np.zeros(ncsf, dtype=np.float64)
                c = ci_sel[:, 0] if ci_sel.ndim == 2 else ci_sel
                ci_dense[sel_idx] = c
                self.ci = ci_dense
            else:
                self.ci = np.ones(1, dtype=np.float64)
            # Return stub: CASSCF driver passes civec back to make_rdm12 which
            # ignores it (uses stored _sci_sel_idx/_sci_ci_sel). This avoids
            # the page-fault storm and PCIe upload on every CASSCF iteration.
            return e_tot, np.ones(1, dtype=np.float64)
        else:
            ci_list = []
            for r in range(nroots):
                ci_dense = np.zeros(ncsf, dtype=np.float64)
                ci_dense[sel_idx] = ci_sel[:, r]
                ci_list.append(ci_dense)
            e_tot = np.asarray(res.e_var, dtype=np.float64).ravel()[:nroots]
            self.converged = True
            self.eci = e_tot
            self.ci = ci_list
            return e_tot, ci_list

    def make_rdm12(self, civec, norb: int, nelec, **kwargs):
        """Compute (dm1, dm2) on GPU using the pairwise T-matrix kernel.

        GPU end-to-end: CSF data materialised on GPU, T matrix accumulated
        with atomicAdd, dm1/dm2 from cuBLAS GEMM. Returns numpy arrays for
        compatibility with the CASSCF driver.

        Falls back to CPU sparse_rdm if GPU is unavailable.
        """
        drt = getattr(self, "_sci_drt", None)
        sel_idx = getattr(self, "_sci_sel_idx", None)
        ci_sel = getattr(self, "_sci_ci_sel", None)
        drt_dev = getattr(self, "_sci_drt_dev", None)

        if drt is None or sel_idx is None or ci_sel is None:
            return super().make_rdm12(civec, norb, nelec, **kwargs)

        root = kwargs.pop("root", None)
        if ci_sel.ndim == 2:
            nroots_stored = ci_sel.shape[1]
            if root is None:
                # Caller didn't pass root= (e.g. make_state_averaged_rdms).
                # Identify which root civec corresponds to by matching sparse
                # entries against each stored column.
                try:
                    import cupy as _cp
                    if isinstance(civec, _cp.ndarray):
                        civec_arr = _cp.asnumpy(civec).astype(np.float64).ravel()
                    else:
                        civec_arr = np.asarray(civec, dtype=np.float64).ravel()
                except ImportError:
                    civec_arr = np.asarray(civec, dtype=np.float64).ravel()
                root = 0
                if (nroots_stored > 1 and civec_arr.size > 0 and len(sel_idx) > 0
                        and int(civec_arr.size) > int(sel_idx.max())):
                    ci_at_sel = civec_arr[sel_idx]
                    best_err = np.inf
                    for r in range(nroots_stored):
                        err = float(np.max(np.abs(ci_at_sel - ci_sel[:, r])))
                        if err < best_err:
                            best_err, root = err, r
            c = ci_sel[:, int(root)]
        else:
            c = ci_sel

        # GPU path: requires CuPy and compiled T-matrix kernel
        if drt_dev is not None:
            try:
                import cupy as cp
                from asuka.sci.gpu_rdm import make_rdm12_gpu
                dm1_d, dm2_d = make_rdm12_gpu(drt, drt_dev, sel_idx, c, cp)
                return cp.asnumpy(dm1_d), cp.asnumpy(dm2_d)
            except Exception:
                pass  # fall through to CPU path

        # CPU fallback
        from asuka.sci.sparse_rdm import make_rdm12_selected
        return make_rdm12_selected(drt, sel_idx, c)

    def approx_kernel(self, h1e, eri, norb, nelec, ci0=None, ecore=0.0, nroots=1, **kwargs):
        """Approximate CI solve for CASSCF micro-iterations.

        Reuses the selected space from the last kernel() call and re-solves
        the eigenvalue problem in that fixed subspace with the new integrals.
        This avoids re-running the full SCI growth loop at every micro-iteration,
        giving a short warm-started Davidson solve instead of a full HB-SCI run.

        Falls back to full kernel() if no selected space is cached yet.
        """
        from asuka.sci.gpu_cipsi import run_cipsi_trials

        drt = getattr(self, "_sci_drt", None)
        sel_idx = getattr(self, "_sci_sel_idx", None)
        ci_sel = getattr(self, "_sci_ci_sel", None)

        if drt is None or sel_idx is None or ci_sel is None:
            return self.kernel(h1e, eri, norb, nelec, ci0=ci0, ecore=ecore, nroots=nroots, **kwargs)

        # Discard irrelevant kwargs the CI solver doesn't accept
        kwargs.pop("tol", None)
        kwargs.pop("max_memory", None)
        kwargs.pop("max_cycle", None)
        kwargs.pop("return_cupy", None)

        neleca, nelecb, nelec_total, _ = self._normalize_nelec(nelec)
        twos = self._get_twos_target(neleca, nelecb)
        nroots_use = int(nroots) if nroots is not None else self.nroots

        # Convert CuPy arrays to numpy
        try:
            import cupy as cp
            if isinstance(h1e, cp.ndarray):
                h1e = cp.asnumpy(h1e)
            if isinstance(eri, cp.ndarray):
                eri = np.asarray(cp.asnumpy(eri))
        except ImportError:
            pass

        # Warm-start from previous CI coefficients
        nsel = int(sel_idx.size)
        if ci_sel.ndim == 2:
            prev_c = [
                (sel_idx.copy(), np.asarray(ci_sel[:, r], dtype=np.float64))
                for r in range(min(nroots_use, ci_sel.shape[1]))
            ]
        else:
            prev_c = [(sel_idx.copy(), np.asarray(ci_sel, dtype=np.float64))]

        # Re-diagonalize in fixed selected space: no growth, warm-started Davidson
        res = run_cipsi_trials(
            drt, h1e, eri,
            ecore=float(ecore),
            nroots=nroots_use,
            ci0=prev_c,
            init_ncsf=nsel,
            max_ncsf=nsel,
            grow_by=0,
            max_iter=0,
            selection_mode=str(self.sci_selection_mode),
            hb_epsilon=float(self.sci_hb_epsilon),
            davidson_max_cycle=int(self.sci_davidson_max_cycle),
            davidson_max_space=int(self.sci_davidson_max_space),
            davidson_tol=float(self.sci_davidson_tol),
            backend=self._resolve_backend(),
            epq_mode="no_epq_support_aware",
            workspace_kwargs={
                "projected_solver_gpu": True,
                "projected_solver_matrix_free": True,
            },
        )

        # Update stored CI coefficients (selection unchanged)
        ci_sel_new = np.asarray(res.ci_sel, dtype=np.float64)
        self._sci_ci_sel = ci_sel_new

        ncsf = int(drt.ncsf)
        if nroots_use == 1:
            e_tot = float(np.asarray(res.e_var, dtype=np.float64).ravel()[0])
            # Return stub: make_rdm12 uses _sci_ci_sel, not civec.
            # Avoids PCIe upload of 3.59 GB per CASSCF micro-iteration.
            return e_tot, np.ones(1, dtype=np.float64)
        else:
            ci_list = []
            for r in range(nroots_use):
                ci_dense = np.zeros(ncsf, dtype=np.float64)
                ci_dense[sel_idx] = ci_sel_new[:, r]
                ci_list.append(ci_dense)
            e_tot = np.asarray(res.e_var, dtype=np.float64).ravel()[:nroots_use]
            return e_tot, ci_list

    def make_rdm1(self, civec, norb: int, nelec, **kwargs):
        """Compute 1-RDM using sparse RDM."""
        dm1, _dm2 = self.make_rdm12(civec, norb, nelec, **kwargs)
        return dm1

    @staticmethod
    def _build_hf_ci0(drt, norb: int, nelec: int):
        nocc = nelec // 2
        remaining = nelec - 2 * nocc
        steps = []
        for i in range(norb):
            if i < nocc:
                steps.append("D")
            elif i == nocc and remaining > 0:
                steps.append("S")
            else:
                steps.append("E")
        try:
            hf_idx = int(drt.path_to_index(steps))
            return [(np.array([hf_idx], dtype=np.int64), np.array([1.0]))]
        except Exception:
            return None
