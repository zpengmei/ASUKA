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

        # Ensure integrals are GPU-resident DF for the CUDA CIPSI path.
        # run_cipsi_trials with dense np.ndarray ERI + backend='auto' falls
        # back to the CPU row oracle which is catastrophically slow for large
        # active spaces.  Promote to DeviceDFMOIntegrals whenever possible.
        backend = str(self.sci_backend)
        if isinstance(eri, DFMOIntegrals) and not isinstance(eri, DeviceDFMOIntegrals):
            try:
                import cupy as cp
                eri = eri.to_device(cp, with_eri_mat=False)
            except Exception:
                pass
        if isinstance(eri, DeviceDFMOIntegrals) and backend == "auto":
            backend = "cuda_key64"

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

        # Expand sparse CI to dense (ncsf,) vectors — makes inherited RDMs work
        sel_idx = np.asarray(res.sel_idx, dtype=np.int64)
        ci_sel = np.asarray(res.ci_sel, dtype=np.float64)
        ncsf = int(drt.ncsf)

        if nroots == 1:
            ci_dense = np.zeros(ncsf, dtype=np.float64)
            c = ci_sel[:, 0] if ci_sel.ndim == 2 else ci_sel
            ci_dense[sel_idx] = c
            e_tot = float(np.asarray(res.e_var, dtype=np.float64).ravel()[0])
            self.converged = True
            self.eci = e_tot
            self.ci = ci_dense
            return e_tot, ci_dense
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
