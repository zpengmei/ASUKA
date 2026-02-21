from __future__ import annotations

"""CASSCF drivers (DF, GPU-assisted).

Status
------
This is an initial, energy-only CASSCF implementation intended to enable an
end-to-end SCF->CASCI->CASSCF workflow for small/medium systems.

The current driver uses a simple macro-iteration loop:
  1) Solve CASCI for fixed orbitals (cuERI active-space DF integrals + GUGA solver)
  2) Build the orbital gradient from CAS RDMs and DF factors
  3) Update orbitals via a Cayley-transform rotation in the non-redundant blocks

Limitations (current)
---------------------
- RHF/ROHF orbitals only (UHF is not supported yet).
- Nuclear gradients are available via :mod:`asuka.mcscf.nuc_grad_df` for DF-based
  validation. The DF 2e term is analytic on supported backends and may
  fall back to finite differences on the DF factors `B` when analytic derivative
  kernels are unavailable.
"""

from dataclasses import dataclass
from typing import Any, Sequence
import time
import math

import numpy as np

from asuka.hf import df_scf as _df_scf
from asuka.solver import GUGAFCISolver

from asuka.frontend.molecule import Molecule
from asuka.frontend.scf import RHFDFRunResult, ROHFDFRunResult, UHFDFRunResult

from .casci import CASCIResult, run_casci_df, eval_casci_energy_df
from .orbital_grad import allowed_rotation_mask, cayley_update, orbital_gradient_dense, orbital_gradient_df
from .state_average import (
    ci_as_list,
    fix_ci_phases,
    make_state_averaged_rdms,
    match_roots_by_overlap,
    normalize_weights,
)
from .uhf_guess import spatialize_uhf_mo_coeff


def _as_xp_f64(xp, a):
    return xp.asarray(a, dtype=xp.float64)


def _infer_max_l_from_ao_basis(ao_basis: Any) -> int:
    shell_l = getattr(ao_basis, "shell_l", None)
    if shell_l is None:
        return 5
    arr = np.asarray(shell_l, dtype=np.int32).ravel()
    if int(arr.size) == 0:
        return 0
    return int(np.max(arr))


class _LBFGSHistory:
    """Minimal L-BFGS history (NumPy/CuPy agnostic)."""

    def __init__(self, *, max_vec: int, curvature_tol: float = 1e-14):
        max_vec = int(max_vec)
        if max_vec < 0:
            raise ValueError("max_vec must be >= 0")
        curvature_tol = float(curvature_tol)
        if curvature_tol <= 0.0:
            raise ValueError("curvature_tol must be > 0")

        self.max_vec = max_vec
        self.curvature_tol = curvature_tol
        self._s: list[Any] = []
        self._y: list[Any] = []
        self._rho: list[float] = []

    @property
    def enabled(self) -> bool:
        return self.max_vec > 0

    def update(self, s: Any, y: Any) -> bool:
        if not self.enabled:
            return False
        ys = float((y * s).sum().item())
        if ys <= self.curvature_tol:
            return False
        self._s.append(s.copy())
        self._y.append(y.copy())
        self._rho.append(1.0 / ys)
        if len(self._s) > self.max_vec:
            self._s.pop(0)
            self._y.pop(0)
            self._rho.pop(0)
        return True

    def direction(self, g: Any, *, h0_inv: Any | None = None) -> Any:
        """Return p ≈ -H g via 2-loop recursion."""

        if not self.enabled or len(self._s) == 0:
            return -g if h0_inv is None else -(h0_inv * g)

        q = g.copy()
        alpha: list[float] = []
        for s, y, rho in zip(reversed(self._s), reversed(self._y), reversed(self._rho)):
            a = float(rho) * float((s * q).sum().item())
            alpha.append(a)
            q = q - float(a) * y

        r = q if h0_inv is None else (h0_inv * q)
        for s, y, rho, a in zip(self._s, self._y, self._rho, reversed(alpha)):
            b = float(rho) * float((y * r).sum().item())
            r = r + s * (float(a) - float(b))
        return -r

    def clear(self) -> None:
        self._s.clear()
        self._y.clear()
        self._rho.clear()


class _DIISHistory:
    """Small Pulay DIIS helper for orbital-step extrapolation."""

    def __init__(self, *, max_vec: int, min_vec: int = 3, regularization: float = 1e-12):
        max_vec = int(max_vec)
        min_vec = int(min_vec)
        regularization = float(regularization)
        if max_vec < 0:
            raise ValueError("max_vec must be >= 0")
        if min_vec < 2:
            raise ValueError("min_vec must be >= 2")
        if regularization < 0.0:
            raise ValueError("regularization must be >= 0")

        self.max_vec = max_vec
        self.min_vec = min_vec
        self.regularization = regularization
        self._x: list[Any] = []
        self._e: list[Any] = []

    @property
    def enabled(self) -> bool:
        return self.max_vec > 0

    @property
    def ready(self) -> bool:
        return self.enabled and len(self._x) >= self.min_vec and len(self._e) >= self.min_vec

    def push(self, x: Any, e: Any) -> None:
        if not self.enabled:
            return
        self._x.append(x.copy())
        self._e.append(e.copy())
        if len(self._x) > self.max_vec:
            self._x.pop(0)
            self._e.pop(0)

    def extrapolate(self) -> Any | None:
        if not self.ready:
            return None
        m = len(self._x)
        B = np.empty((m + 1, m + 1), dtype=np.float64)
        B.fill(-1.0)
        B[m, m] = 0.0
        for i in range(m):
            ei = self._e[i]
            for j in range(i, m):
                val = float((ei * self._e[j]).sum().item())
                B[i, j] = val
                B[j, i] = val
        if self.regularization > 0.0:
            for i in range(m):
                B[i, i] += self.regularization
        rhs = np.zeros((m + 1,), dtype=np.float64)
        rhs[m] = -1.0
        try:
            coeff = np.linalg.solve(B, rhs)[:m]
        except np.linalg.LinAlgError:
            return None
        out = self._x[0] * 0.0
        for c, x in zip(coeff, self._x):
            out = out + float(c) * x
        return out

    def clear(self) -> None:
        self._x.clear()
        self._e.clear()


def _qune_predict_tmin(
    *,
    e_last: float,
    e_now: float,
    fp_last: float,
    fp_now: float,
) -> tuple[float, float, bool]:
    """Cubic line-search proxy used by OpenMolcas QUNE.

    Returns
    -------
    tmin
        Selected minimum position in the scalar coordinate where ``t=1`` denotes
        the current point and ``t=0`` denotes the previous point.
    e_min
        Predicted minimum energy value at ``tmin``.
    has_local_min
        Whether a local minimum inside the broad search window was found.
    """

    c0 = float(e_last)
    c1 = float(fp_last)
    c2 = 3.0 * (float(e_now) - float(e_last)) - 2.0 * float(fp_last) - float(fp_now)
    c3 = -2.0 * (float(e_now) - float(e_last)) + float(fp_last) + float(fp_now)

    n_local = 0
    t_local = 0.0
    p = 3.0 * c1 * c3
    q = c2 * c2
    if abs(p) > 1.0e-3 * q:
        if q > p:
            n_local = 1
            t_local = (math.sqrt(max(q - p, 0.0)) - c2) / (3.0 * c3)
    elif abs(c2) > 1.0e-3 * abs(c1):
        if c2 > 0.0:
            n_local = 1
            t_local = -c1 / (2.0 * c2)

    t0 = -0.5
    t1 = 2.5
    if n_local == 1:
        e_local = c0 + t_local * (c1 + t_local * (c2 + t_local * c3))
        if t_local > t1 or t_local < t0:
            n_local = 0
        else:
            e_min = float(e_local)
            t_min = float(t_local)
    if n_local != 1:
        e0 = c0 + t0 * (c1 + t0 * (c2 + t0 * c3))
        e1 = c0 + t1 * (c1 + t1 * (c2 + t1 * c3))
        if e0 < e1:
            t_min = float(t0)
            e_min = float(e0)
        else:
            t_min = float(t1)
            e_min = float(e1)
        if n_local == 1:
            e_local = c0 + t_local * (c1 + t_local * (c2 + t_local * c3))
            if e_local < e_min:
                t_min = float(t_local)
                e_min = float(e_local)
            else:
                n_local = 0

    return float(t_min), float(e_min), bool(n_local == 1)


def _estimate_basis_cart_nao(basis: Any) -> int:
    shell_start = np.asarray(getattr(basis, "shell_ao_start"), dtype=np.int64).ravel()
    shell_l = np.asarray(getattr(basis, "shell_l"), dtype=np.int64).ravel()
    if shell_start.size == 0:
        return 0
    if shell_start.shape != shell_l.shape:
        raise ValueError("shell_ao_start/shell_l shape mismatch")
    nfunc = ((shell_l + 1) * (shell_l + 2) // 2).astype(np.int64, copy=False)
    return int(np.max(shell_start + nfunc))


def _estimate_b_whitened_nbytes(ao_basis: Any, aux_basis: Any) -> int:
    try:
        nao = int(_estimate_basis_cart_nao(ao_basis))
        naux = int(_estimate_basis_cart_nao(aux_basis))
    except Exception:
        return 0
    if nao <= 0 or naux <= 0:
        return 0
    return int(nao) * int(nao) * int(naux) * 8


@dataclass(frozen=True)
class CASSCFResult:
    mol: Molecule
    basis_name: str
    auxbasis_name: str
    converged: bool
    niter: int
    ncore: int
    ncas: int
    nelecas: int | tuple[int, int]
    nroots: int
    root_weights: np.ndarray
    e_roots: np.ndarray
    e_tot: float
    ecore: float
    ci: Any
    mo_coeff: Any
    grad_norm: float
    casci: CASCIResult
    profile: dict | None = None
    scf_out: Any | None = None


def casscf_orbital_gradient_df(
    scf_out: RHFDFRunResult | ROHFDFRunResult | UHFDFRunResult,
    casscf: CASSCFResult,
    *,
    fcisolver: GUGAFCISolver | None = None,
    twos: int | None = None,
    **solver_kwargs,
):
    """Compute the DF-based CASSCF orbital gradient (antisymmetric matrix).

    For a converged CASSCF wavefunction, the returned non-redundant gradient
    norm should be small.
    """

    nroots = int(casscf.nroots)
    weights = normalize_weights(casscf.root_weights, nroots=nroots)
    ci_list = ci_as_list(casscf.ci, nroots=nroots)

    if fcisolver is None:
        if twos is None:
            twos = int(getattr(casscf.mol, "spin", 0))
        fcisolver = GUGAFCISolver(twos=int(twos), nroots=int(nroots))
    else:
        if getattr(fcisolver, "nroots", None) != int(nroots):
            try:
                fcisolver.nroots = int(nroots)
            except Exception:
                pass

    dm1_act, dm2_act = make_state_averaged_rdms(
        fcisolver,
        ci_list,
        weights,
        ncas=int(casscf.ncas),
        nelecas=casscf.nelecas,
        solver_kwargs=solver_kwargs,
    )

    return orbital_gradient_df(
        scf_out,
        C=casscf.mo_coeff,
        ncore=int(casscf.ncore),
        ncas=int(casscf.ncas),
        dm1_act=dm1_act,
        dm2_act=dm2_act,
    )


def casscf_orbital_gradient_dense(
    scf_out: RHFDFRunResult | ROHFDFRunResult | UHFDFRunResult,
    casscf: CASSCFResult,
    *,
    fcisolver: GUGAFCISolver | None = None,
    twos: int | None = None,
    dense_eps_ao: float = 0.0,
    dense_max_tile_bytes: int = 256 * 1024 * 1024,
    dense_cpu_threads: int = 0,
    dense_cpu_blas_nthreads: int | None = None,
    dense_cpu_p_block_nmo: int = 64,
    dense_gpu_threads: int = 256,
    profile: dict | None = None,
    **solver_kwargs,
):
    """Compute the dense-consistent CASSCF orbital gradient (antisymmetric matrix)."""

    nroots = int(casscf.nroots)
    weights = normalize_weights(casscf.root_weights, nroots=nroots)
    ci_list = ci_as_list(casscf.ci, nroots=nroots)

    if fcisolver is None:
        if twos is None:
            twos = int(getattr(casscf.mol, "spin", 0))
        fcisolver = GUGAFCISolver(twos=int(twos), nroots=int(nroots))
    else:
        if getattr(fcisolver, "nroots", None) != int(nroots):
            try:
                fcisolver.nroots = int(nroots)
            except Exception:
                pass

    dm1_act, dm2_act = make_state_averaged_rdms(
        fcisolver,
        ci_list,
        weights,
        ncas=int(casscf.ncas),
        nelecas=casscf.nelecas,
        solver_kwargs=solver_kwargs,
    )

    return orbital_gradient_dense(
        scf_out,
        C=casscf.mo_coeff,
        ncore=int(casscf.ncore),
        ncas=int(casscf.ncas),
        dm1_act=dm1_act,
        dm2_act=dm2_act,
        dense_eps_ao=float(dense_eps_ao),
        dense_max_tile_bytes=int(dense_max_tile_bytes),
        dense_cpu_threads=int(dense_cpu_threads),
        dense_cpu_blas_nthreads=None if dense_cpu_blas_nthreads is None else int(dense_cpu_blas_nthreads),
        dense_cpu_p_block_nmo=int(dense_cpu_p_block_nmo),
        dense_gpu_threads=int(dense_gpu_threads),
        profile=profile,
    )


def run_casscf_df(
    scf_out: RHFDFRunResult | ROHFDFRunResult | UHFDFRunResult,
    *,
    ncore: int,
    ncas: int,
    nelecas: int | tuple[int, int],
    mo_coeff: Any | None = None,
    mo_coeff0: Any | None = None,
    rotation_mask: Any | None = None,
    ci0: Any | None = None,
    guess: Any | None = None,
    fcisolver: GUGAFCISolver | None = None,
    twos: int | None = None,
    nroots: int = 1,
    root_weights: Sequence[float] | None = None,
    casci_backend: str = "df",
    matvec_backend: str = "cuda_eri_mat",
    want_eri_mat: bool = True,
    aux_block_naux: int = 256,
    max_tile_bytes: int = 256 * 1024 * 1024,
    use_cueri_b_cache: bool = True,
    max_b_cache_bytes: int = 512 * 1024 * 1024,
    dense_gpu_threads: int = 256,
    dense_gpu_eps_ao: float = 0.0,
    dense_gpu_ao_rep: str = "auto",
    dense_gpu_builder_mol: Any | None = None,
    dense_exact_jk: bool | str = "auto",
    max_cycle_macro: int = 30,
    tol: float = 1e-8,
    conv_tol_grad: float | None = None,
    max_stepsize: float = 0.02,
    damp: float = 1.0,
    orbital_optimizer: str = "ah",
    lbfgs_history: int = 10,
    lbfgs_descent_guard: bool = False,
    lbfgs_descent_guard_clear_history: bool = True,
    diis_enabled: bool = True,
    diis_start_cycle: int = 3,
    diis_space: int = 8,
    step_rejection_enabled: bool = True,
    step_rejection_tol: float = 1e-6,
    step_rejection_factor: float = 0.7,
    step_rejection_min_stepsize: float = 1e-4,
    step_recovery_enabled: bool = False,
    step_recovery_factor: float = 1.2,
    step_recovery_interval: int = 5,
    step_norm_scale_alpha: float = 0.0,
    qune_enabled: bool = False,
    qune_ls_max_consecutive: int = 2,
    qune_close_threshold: float = 0.4,
    qune_uphill_scale: float = 0.7,
    ah_level_shift: float = 1e-8,
    ah_conv_tol: float = 1e-12,
    ah_max_cycle: int = 30,
    ah_lindep: float = 1e-14,
    ah_start_tol: float = 2.5,
    ah_start_cycle: int = 3,
    ah_max_cycle_micro: int = 4,
    ah_grad_trust_region: float = 3.0,
    ah_kf_interval: int = 4,
    ah_kf_trust_region: float = 3.0,
    ah_max_cycle_micro_cap: int | None = 4,
    ah_conv_tol_grad: float = 1e-4,
    ah_conv_tol_energy: float = 1e-7,
    ah_ci_update: str = "orthonormalize",
    ci_max_cycle_inner: int | None = 2,
    dense_cpu_eps_ao: float = 0.0,
    dense_cpu_eps_mo: float = 0.0,
    dense_cpu_threads: int = 0,
    dense_cpu_blas_nthreads: int | None = None,
    profile: dict | None = None,
    **solver_kwargs,
) -> CASSCFResult:
    """Run a DF-CASSCF optimization.

    Notes
    -----
    - Supports state-specific (`nroots=1`) and state-averaged (`nroots>1`) orbital
      optimization via weighted active-space RDMs (`root_weights`).
    - The orbital gradient is DF-based (from `scf_out.df_B`). If `casci_backend`
      is `"dense_cpu"`, only the *active-space* 2e integrals in the CI solve are
      exact; the mean-field pieces remain DF-based.
    """

    nroots = int(nroots)
    if nroots < 1:
        raise ValueError("nroots must be >= 1")
    weights = normalize_weights(root_weights, nroots=nroots)

    if not bool(getattr(scf_out.scf, "converged", False)):
        raise RuntimeError("SCF must be converged before CASSCF")

    ncore = int(ncore)
    ncas = int(ncas)
    if ncore < 0:
        raise ValueError("ncore must be >= 0")
    if ncas <= 0:
        raise ValueError("ncas must be > 0")

    if mo_coeff is not None:
        if mo_coeff0 is not None:
            raise ValueError("provide only one of mo_coeff and mo_coeff0")
        mo_coeff0 = mo_coeff

    tol = float(tol)
    if tol <= 0.0:
        raise ValueError("tol must be > 0")
    if conv_tol_grad is None:
        conv_tol_grad = float(np.sqrt(tol))
    conv_tol_grad = float(conv_tol_grad)
    if conv_tol_grad <= 0.0:
        raise ValueError("conv_tol_grad must be > 0")

    max_cycle_macro = int(max_cycle_macro)
    if max_cycle_macro < 1:
        raise ValueError("max_cycle_macro must be >= 1")

    max_stepsize = float(max_stepsize)
    if max_stepsize <= 0.0:
        raise ValueError("max_stepsize must be > 0")

    damp = float(damp)
    if damp <= 0.0:
        raise ValueError("damp must be > 0")

    use_cueri_b_cache = bool(use_cueri_b_cache)
    max_b_cache_bytes = int(max_b_cache_bytes)
    if max_b_cache_bytes < 0:
        raise ValueError("max_b_cache_bytes must be >= 0")

    orbital_optimizer = str(orbital_optimizer).strip().lower()
    if orbital_optimizer not in {"jacobi", "lbfgs", "ah"}:
        raise ValueError("orbital_optimizer must be one of: 'jacobi', 'lbfgs', 'ah'")
    lbfgs_history = int(lbfgs_history)
    if lbfgs_history < 0:
        raise ValueError("lbfgs_history must be >= 0")
    lbfgs_descent_guard = bool(lbfgs_descent_guard)
    lbfgs_descent_guard_clear_history = bool(lbfgs_descent_guard_clear_history)
    diis_enabled = bool(diis_enabled)
    diis_start_cycle = int(diis_start_cycle)
    if diis_start_cycle < 1:
        raise ValueError("diis_start_cycle must be >= 1")
    diis_space = int(diis_space)
    if diis_space < 0:
        raise ValueError("diis_space must be >= 0")
    step_rejection_enabled = bool(step_rejection_enabled)
    step_rejection_tol = float(step_rejection_tol)
    if step_rejection_tol < 0.0:
        raise ValueError("step_rejection_tol must be >= 0")
    step_rejection_factor = float(step_rejection_factor)
    if not (0.0 < step_rejection_factor < 1.0):
        raise ValueError("step_rejection_factor must satisfy 0 < factor < 1")
    step_rejection_min_stepsize = float(step_rejection_min_stepsize)
    if step_rejection_min_stepsize <= 0.0:
        raise ValueError("step_rejection_min_stepsize must be > 0")
    step_recovery_enabled = bool(step_recovery_enabled)
    step_recovery_factor = float(step_recovery_factor)
    if step_recovery_factor <= 1.0:
        raise ValueError("step_recovery_factor must be > 1")
    step_recovery_interval = int(step_recovery_interval)
    if step_recovery_interval < 1:
        raise ValueError("step_recovery_interval must be >= 1")
    step_norm_scale_alpha = float(step_norm_scale_alpha)
    if step_norm_scale_alpha < 0.0:
        raise ValueError("step_norm_scale_alpha must be >= 0")
    qune_enabled = bool(qune_enabled)
    qune_ls_max_consecutive = int(qune_ls_max_consecutive)
    if qune_ls_max_consecutive < 0:
        raise ValueError("qune_ls_max_consecutive must be >= 0")
    qune_close_threshold = float(qune_close_threshold)
    if qune_close_threshold <= 0.0:
        raise ValueError("qune_close_threshold must be > 0")
    qune_uphill_scale = float(qune_uphill_scale)
    if not (0.0 < qune_uphill_scale <= 1.0):
        raise ValueError("qune_uphill_scale must satisfy 0 < scale <= 1")
    ah_level_shift = float(ah_level_shift)
    ah_conv_tol = float(ah_conv_tol)
    ah_max_cycle = int(ah_max_cycle)
    ah_lindep = float(ah_lindep)
    ah_start_tol = float(ah_start_tol)
    ah_start_cycle = int(ah_start_cycle)
    ah_max_cycle_micro = int(ah_max_cycle_micro)
    ah_grad_trust_region = float(ah_grad_trust_region)
    ah_kf_interval = int(ah_kf_interval)
    ah_kf_trust_region = float(ah_kf_trust_region)
    ah_max_cycle_micro_cap_i = None if ah_max_cycle_micro_cap is None else int(ah_max_cycle_micro_cap)
    ah_conv_tol_grad = float(ah_conv_tol_grad)
    ah_conv_tol_energy = float(ah_conv_tol_energy)
    ah_ci_update = str(ah_ci_update).strip().lower()
    if ah_conv_tol <= 0.0:
        raise ValueError("ah_conv_tol must be > 0")
    if ah_max_cycle < 1:
        raise ValueError("ah_max_cycle must be >= 1")
    if ah_lindep <= 0.0:
        raise ValueError("ah_lindep must be > 0")
    if ah_start_tol <= 0.0:
        raise ValueError("ah_start_tol must be > 0")
    if ah_start_cycle < 1:
        raise ValueError("ah_start_cycle must be >= 1")
    if ah_max_cycle_micro < 1:
        raise ValueError("ah_max_cycle_micro must be >= 1")
    if ah_grad_trust_region <= 0.0:
        raise ValueError("ah_grad_trust_region must be > 0")
    if ah_kf_interval < 1:
        raise ValueError("ah_kf_interval must be >= 1")
    if ah_kf_trust_region <= 0.0:
        raise ValueError("ah_kf_trust_region must be > 0")
    if ah_max_cycle_micro_cap_i is not None and ah_max_cycle_micro_cap_i < 1:
        raise ValueError("ah_max_cycle_micro_cap must be >= 1 when set")
    if ah_conv_tol_grad <= 0.0:
        raise ValueError("ah_conv_tol_grad must be > 0")
    if ah_conv_tol_energy <= 0.0:
        raise ValueError("ah_conv_tol_energy must be > 0")
    if ah_ci_update not in {"pyscf", "orthonormalize"}:
        raise ValueError("ah_ci_update must be one of: 'pyscf', 'orthonormalize'")

    casci_backend_s = str(casci_backend).strip().lower()
    if casci_backend_s not in {"df", "dense_cpu", "dense_gpu"}:
        raise ValueError("casci_backend must be one of: 'df', 'dense_cpu', 'dense_gpu'")

    matvec_backend_s = str(matvec_backend).strip().lower()
    if casci_backend_s == "dense_cpu" and matvec_backend_s != "contract":
        raise ValueError("casci_backend='dense_cpu' currently requires matvec_backend='contract'")
    if casci_backend_s == "dense_gpu" and matvec_backend_s not in {"cuda_eri_mat", "cuda"}:
        raise ValueError("casci_backend='dense_gpu' currently requires matvec_backend='cuda_eri_mat' (or 'cuda')")

    dense_exact_jk_mode = dense_exact_jk
    if isinstance(dense_exact_jk_mode, str):
        mode = dense_exact_jk_mode.strip().lower()
        if mode in {"", "auto"}:
            mol_obj = getattr(scf_out, "mol", None)
            dense_exact_jk = bool(casci_backend_s in {"dense_cpu", "dense_gpu"} and hasattr(mol_obj, "intor"))
        elif mode in {"1", "true", "yes", "on"}:
            dense_exact_jk = True
        elif mode in {"0", "false", "no", "off"}:
            dense_exact_jk = False
        else:
            raise ValueError("dense_exact_jk must be bool or one of {'auto','on','off'}")
    else:
        dense_exact_jk = bool(dense_exact_jk_mode)

    if guess is not None:
        if mo_coeff0 is None:
            mo_coeff0 = getattr(guess, "mo_coeff", None)
        if ci0 is None:
            ci0 = getattr(guess, "ci", None)

    C = mo_coeff0 if mo_coeff0 is not None else getattr(scf_out.scf, "mo_coeff", None)
    if C is None:
        raise ValueError("scf_out.scf.mo_coeff is missing")
    if isinstance(C, tuple):
        mo_occ = getattr(scf_out.scf, "mo_occ", None)
        if not isinstance(mo_occ, tuple) or len(mo_occ) != 2:
            raise ValueError("UHF mo_coeff=(Ca,Cb) requires scf_out.scf.mo_occ=(occ_a,occ_b)")
        C, _occ_no = spatialize_uhf_mo_coeff(S_ao=scf_out.int1e.S, mo_coeff=C, mo_occ=mo_occ)

    # Determine array backend (cupy vs numpy).  df_B may be None for dense HF.
    _xp_probe = scf_out.df_B if scf_out.df_B is not None else getattr(scf_out, "ao_eri", C)
    xp, _is_gpu = _df_scf._get_xp(_xp_probe, C)  # noqa: SLF001
    C = _as_xp_f64(xp, C)

    if casci_backend_s == "df":
        # DF CASCI can run on CPU (DFMOIntegrals + contract/row_oracle_df) or on CUDA
        # (DeviceDFMOIntegrals + cuda/cuda_eri_mat). Pick the path based on `matvec_backend`.
        if matvec_backend_s in {"cuda_eri_mat", "cuda"}:
            if not bool(_is_gpu):
                raise ValueError("matvec_backend='cuda' requires scf_out with GPU DF factors (CuPy arrays)")
        elif matvec_backend_s not in {"contract", "row_oracle_df"}:
            raise ValueError("casci_backend='df' requires matvec_backend in {'contract','row_oracle_df','cuda','cuda_eri_mat'}")

    nmo = int(C.shape[1])
    nocc = ncore + ncas
    if nocc > nmo:
        raise ValueError("ncore+ncas exceeds number of MOs")

    if rotation_mask is None:
        allowed = allowed_rotation_mask(nmo, ncore, ncas)
    else:
        allowed = rotation_mask
        if xp is not np:
            try:
                import cupy as cp  # type: ignore
            except Exception:
                cp = None
            if cp is not None and isinstance(allowed, cp.ndarray):  # type: ignore[attr-defined]
                allowed = cp.asnumpy(allowed)
        allowed = np.asarray(allowed, dtype=bool)
        if allowed.shape != (nmo, nmo):
            raise ValueError("rotation_mask must have shape (nmo, nmo)")
        allowed &= np.tril(np.ones((nmo, nmo), dtype=bool), k=-1)

    allowed_xp = xp.asarray(allowed)
    allowed_nvar = int(np.count_nonzero(allowed))

    lbfgs = _LBFGSHistory(max_vec=lbfgs_history) if orbital_optimizer == "lbfgs" else None
    diis = _DIISHistory(max_vec=diis_space) if diis_enabled else None
    lbfgs_prev_g = None
    lbfgs_prev_step = None
    b_cache_enabled = bool(
        use_cueri_b_cache
        and max_b_cache_bytes > 0
        and casci_backend_s == "df"
        and matvec_backend_s in {"cuda_eri_mat", "cuda"}
    )
    b_cache_est_nbytes = _estimate_b_whitened_nbytes(scf_out.ao_basis, scf_out.aux_basis) if b_cache_enabled else 0
    if b_cache_enabled and b_cache_est_nbytes > 0 and b_cache_est_nbytes > max_b_cache_bytes:
        b_cache_enabled = False

    if fcisolver is None:
        if twos is None:
            twos = int(getattr(scf_out.mol, "spin", 0))
        fcisolver_use = GUGAFCISolver(twos=int(twos), nroots=int(nroots))
    else:
        fcisolver_use = fcisolver
        if getattr(fcisolver_use, "nroots", None) != int(nroots):
            try:
                fcisolver_use.nroots = int(nroots)
            except Exception:
                pass

    # Enable kernel profiling on the solver when CASSCF profiling is active
    if profile is not None:
        fcisolver_use.kernel_profile = True
        fcisolver_use.kernel_profile_cuda_sync = True

    # Propagate GPU backends to the solver so that trans_rdm12 / contract_2e
    # (called directly by the AH orbital optimizer) also run on GPU.
    if matvec_backend_s in {"cuda_eri_mat", "cuda"}:
        fcisolver_use.matvec_backend = str(matvec_backend_s)
        # rdm_backend="auto" will now pick CUDA because matvec_backend starts
        # with "cuda".  Force it explicitly for clarity.
        if str(getattr(fcisolver_use, "rdm_backend", "auto")).strip().lower() == "auto":
            fcisolver_use.rdm_backend = "cuda"

    dense_cpu_builder = None
    dense_gpu_builder = None
    if casci_backend_s == "dense_cpu":
        from asuka.cueri.active_space_dense_cpu import CuERIActiveSpaceDenseCPUBuilder  # noqa: PLC0415

        dense_cpu_builder = CuERIActiveSpaceDenseCPUBuilder(
            ao_basis=scf_out.ao_basis,
            max_l=int(_infer_max_l_from_ao_basis(scf_out.ao_basis)),
            max_tile_bytes=int(max_tile_bytes),
            threads=int(dense_cpu_threads),
        )
    elif casci_backend_s == "dense_gpu":
        from asuka.cueri.active_space_dense_gpu import CuERIActiveSpaceDenseGPUBuilder  # noqa: PLC0415

        dense_gpu_builder = CuERIActiveSpaceDenseGPUBuilder(
            mol=dense_gpu_builder_mol if dense_gpu_builder_mol is not None else getattr(scf_out, "mol", None),
            ao_basis=scf_out.ao_basis,
            ao_rep=str(dense_gpu_ao_rep),
            threads=int(dense_gpu_threads),
            max_tile_bytes=int(max_tile_bytes),
            eps_ao=float(dense_gpu_eps_ao),
        )

    e_last = None
    e_ref = None
    C_ref = C.copy()
    converged = False
    casci_out: CASCIResult | None = None
    grad_norm = float("inf")
    prev_ci_list: list[np.ndarray] | None = None
    if ci0 is not None:
        prev_ci_list = ci_as_list(ci0, nroots=nroots)
    e_roots = np.zeros((nroots,), dtype=np.float64)
    ci_out: Any = None
    cached_b_whitened = None
    if b_cache_enabled:
        # Seed the active-space DF cache from the SCF DF factors when possible.
        # This avoids rebuilding metric/int3c2e/whitening inside DF-CASCI/CASSCF.
        B_seed = getattr(scf_out, "df_B", None)
        if B_seed is not None:
            try:
                import cupy as cp  # type: ignore
            except Exception:  # pragma: no cover
                cp = None  # type: ignore
            if cp is not None and isinstance(B_seed, cp.ndarray):  # type: ignore[attr-defined]
                if B_seed.dtype == cp.float64 and B_seed.ndim == 3:  # type: ignore[attr-defined]
                    nao_b, nao_b1, _naux_b = map(int, B_seed.shape)
                    if nao_b == nao_b1 and nao_b == int(C.shape[0]) and int(B_seed.nbytes) <= int(max_b_cache_bytes):
                        if hasattr(B_seed, "flags") and not bool(B_seed.flags.c_contiguous):
                            B_seed = cp.ascontiguousarray(B_seed)
                        cached_b_whitened = B_seed
                        if profile is not None:
                            profile["cueri_b_cache_seed_source"] = "scf_out.df_B"
                            profile["cueri_b_cache_resident"] = True
    b_cache_disabled_reason: str | None = None
    if use_cueri_b_cache and not b_cache_enabled:
        if max_b_cache_bytes <= 0:
            b_cache_disabled_reason = "max_b_cache_bytes<=0"
        elif casci_backend_s != "df":
            b_cache_disabled_reason = f"casci_backend={casci_backend_s}"
        elif matvec_backend_s not in {"cuda_eri_mat", "cuda"}:
            b_cache_disabled_reason = f"matvec_backend={matvec_backend_s}"
        elif b_cache_est_nbytes > max_b_cache_bytes:
            b_cache_disabled_reason = "estimated_cache_too_large"
    if profile is not None:
        profile["cueri_b_cache_enabled"] = bool(b_cache_enabled)
        profile["cueri_b_cache_max_bytes"] = int(max_b_cache_bytes)
        profile["cueri_b_cache_estimated_bytes"] = int(b_cache_est_nbytes)
        profile["orbital_diis_enabled"] = bool(diis is not None and diis.enabled)
        profile["orbital_diis_start_cycle"] = int(diis_start_cycle)
        profile["orbital_diis_space"] = int(diis_space)
        profile["step_rejection_enabled"] = bool(step_rejection_enabled)
        profile["step_rejection_tol"] = float(step_rejection_tol)
        profile["step_rejection_factor"] = float(step_rejection_factor)
        profile["step_rejection_min_stepsize"] = float(step_rejection_min_stepsize)
        profile["step_recovery_enabled"] = bool(step_recovery_enabled)
        profile["step_recovery_factor"] = float(step_recovery_factor)
        profile["step_recovery_interval"] = int(step_recovery_interval)
        profile["step_norm_scale_alpha"] = float(step_norm_scale_alpha)
        profile["lbfgs_descent_guard"] = bool(lbfgs_descent_guard)
        profile["lbfgs_descent_guard_clear_history"] = bool(lbfgs_descent_guard_clear_history)
        profile["step_rejection_count"] = 0
        profile["qune_enabled"] = bool(qune_enabled)
        profile["qune_ls_max_consecutive"] = int(qune_ls_max_consecutive)
        profile["qune_close_threshold"] = float(qune_close_threshold)
        profile["qune_uphill_scale"] = float(qune_uphill_scale)
        profile["ah_enabled"] = bool(orbital_optimizer == "ah")
        profile["ah_max_cycle"] = int(ah_max_cycle)
        profile["ah_max_cycle_micro"] = int(ah_max_cycle_micro)
        profile["ah_max_cycle_micro_cap"] = None if ah_max_cycle_micro_cap_i is None else int(ah_max_cycle_micro_cap_i)
        profile["ah_conv_tol_grad"] = float(ah_conv_tol_grad)
        profile["ah_conv_tol_energy"] = float(ah_conv_tol_energy)
        profile["ah_ci_update"] = str(ah_ci_update)
        profile["dense_gpu_ao_rep"] = str(dense_gpu_ao_rep)
        profile["dense_exact_jk"] = bool(dense_exact_jk)
        if b_cache_disabled_reason is not None:
            profile["cueri_b_cache_disabled_reason"] = str(b_cache_disabled_reason)
    max_stepsize_cur = float(max_stepsize)
    n_step_rejected = 0
    n_consecutive_rejected = 0
    accepted_since_reject = 0
    _n_e_stall = 0
    qune_prev_step = None
    qune_prev_energy = None
    qune_prev_fp = None
    qune_nls = 0
    ah_x0_guess = None
    ah_df_B_np = None
    ah_hcore_np = None
    ah_conv_tol_grad_eff = max(float(conv_tol_grad), float(ah_conv_tol_grad))
    ah_conv_tol_energy_eff = max(float(tol), float(ah_conv_tol_energy))

    for it in range(1, int(max_cycle_macro) + 1):
        _t_iter_start = time.perf_counter() if profile is not None else 0.0

        # 1) CASCI (CI solve + active-space integrals)
        # On iter > 1 with warm-start CI, use reduced Davidson iterations
        _casci_kwargs = dict(solver_kwargs)
        if it > 1 and ci_max_cycle_inner is not None and prev_ci_list is not None:
            _casci_kwargs.setdefault("max_cycle", int(ci_max_cycle_inner))
        casci_profile = profile.setdefault("casci", {}) if profile is not None else None
        casci_cache_out = None
        if b_cache_enabled and cached_b_whitened is None:
            casci_cache_out = {}
        cached_b_in = cached_b_whitened if b_cache_enabled else None
        if casci_backend_s == "df":
            if matvec_backend_s in {"cuda_eri_mat", "cuda"}:
                casci_out = run_casci_df(
                    scf_out,
                    ncore=int(ncore),
                    ncas=int(ncas),
                    nelecas=nelecas,
                    mo_coeff=C,
                    ci0=prev_ci_list,
                    fcisolver=fcisolver_use,
                    twos=twos,
                    nroots=int(nroots),
                    matvec_backend=str(matvec_backend_s),
                    want_eri_mat=bool(want_eri_mat),
                    aux_block_naux=int(aux_block_naux),
                    max_tile_bytes=int(max_tile_bytes),
                    profile=casci_profile,
                    cached_b_whitened=cached_b_in,
                    cache_out=casci_cache_out,
                    **_casci_kwargs,
                )
            else:
                from .casci import run_casci_df_cpu  # noqa: PLC0415

                casci_out = run_casci_df_cpu(
                    scf_out,
                    ncore=int(ncore),
                    ncas=int(ncas),
                    nelecas=nelecas,
                    mo_coeff=C,
                    ci0=prev_ci_list,
                    fcisolver=fcisolver_use,
                    twos=twos,
                    nroots=int(nroots),
                    matvec_backend=str(matvec_backend_s),
                    profile=casci_profile,
                    **_casci_kwargs,
                )
        elif casci_backend_s == "dense_cpu":
            from .casci import run_casci_dense_cpu  # noqa: PLC0415

            casci_out = run_casci_dense_cpu(
                scf_out,
                ncore=int(ncore),
                ncas=int(ncas),
                nelecas=nelecas,
                mo_coeff=C,
                ci0=prev_ci_list,
                fcisolver=fcisolver_use,
                twos=twos,
                nroots=int(nroots),
                matvec_backend=str(matvec_backend_s),
                eps_ao=float(dense_cpu_eps_ao),
                eps_mo=float(dense_cpu_eps_mo),
                threads=int(dense_cpu_threads),
                blas_nthreads=None if dense_cpu_blas_nthreads is None else int(dense_cpu_blas_nthreads),
                max_tile_bytes=int(max_tile_bytes),
                builder=dense_cpu_builder,
                profile=casci_profile,
                **_casci_kwargs,
            )
        else:
            from .casci import run_casci_dense_gpu  # noqa: PLC0415

            casci_out = run_casci_dense_gpu(
                scf_out,
                ncore=int(ncore),
                ncas=int(ncas),
                nelecas=nelecas,
                mo_coeff=C,
                ci0=prev_ci_list,
                fcisolver=fcisolver_use,
                twos=twos,
                nroots=int(nroots),
                matvec_backend=str(matvec_backend_s),
                dense_gpu_ao_rep=str(dense_gpu_ao_rep),
                dense_gpu_builder_mol=dense_gpu_builder_mol,
                dense_gpu_builder=dense_gpu_builder,
                dense_exact_jk=bool(dense_exact_jk),
                threads=int(dense_gpu_threads),
                eps_ao=float(dense_gpu_eps_ao),
                max_tile_bytes=int(max_tile_bytes),
                profile=casci_profile,
                **_casci_kwargs,
            )

        if b_cache_enabled and cached_b_whitened is None and isinstance(casci_cache_out, dict):
            cached_candidate = casci_cache_out.get("cached_b_whitened", None)
            if cached_candidate is not None:
                cand_bytes = int(getattr(cached_candidate, "nbytes", 0))
                if cand_bytes <= int(max_b_cache_bytes):
                    cached_b_whitened = cached_candidate
                else:
                    b_cache_enabled = False
                    b_cache_disabled_reason = "materialized_cache_too_large"
                    if profile is not None:
                        profile["cueri_b_cache_disabled_reason"] = str(b_cache_disabled_reason)
            elif "cache_build_failed" in casci_cache_out:
                b_cache_enabled = False
                b_cache_disabled_reason = "cache_build_failed"
                if profile is not None:
                    profile["cueri_b_cache_disabled_reason"] = str(casci_cache_out["cache_build_failed"])
        if profile is not None:
            profile["cueri_b_cache_resident"] = bool(cached_b_whitened is not None)

        # Root energies/CI (track ordering by overlap for weighted SA cases).
        _t_post_casci = time.perf_counter() if profile is not None else 0.0
        if nroots == 1:
            e_roots = np.asarray([float(casci_out.e_tot)], dtype=np.float64)
        else:
            e_roots = np.asarray(casci_out.e_tot, dtype=np.float64).ravel()
            if int(e_roots.size) != nroots:
                raise RuntimeError("CASCI returned unexpected number of root energies")

        ci_list = ci_as_list(casci_out.ci, nroots=nroots)
        if prev_ci_list is not None and nroots > 1:
            perm = match_roots_by_overlap(prev_ci_list, ci_list)
            e_roots = e_roots[perm]
            ci_list = [ci_list[int(j)] for j in perm.tolist()]
            fix_ci_phases(prev_ci_list, ci_list)

        prev_ci_list = [c.copy() for c in ci_list]
        ci_out = ci_list if nroots > 1 else ci_list[0]

        e_avg = float(np.dot(weights, e_roots))
        _t_pre_rdm = time.perf_counter() if profile is not None else 0.0

        if step_rejection_enabled and e_ref is not None and float(e_avg) > float(e_ref) + float(step_rejection_tol):
            n_step_rejected += 1
            n_consecutive_rejected += 1
            accepted_since_reject = 0
            if n_consecutive_rejected >= 5:
                # Cascade cap: accept the step and reset step size
                e_ref = float(e_avg)
                max_stepsize_cur = float(max_stepsize)
                n_consecutive_rejected = 0
            else:
                max_stepsize_cur = max(float(step_rejection_min_stepsize), float(max_stepsize_cur) * float(step_rejection_factor))
                C = C_ref.copy()
            if lbfgs is not None:
                lbfgs.clear()
            lbfgs_prev_g = None
            lbfgs_prev_step = None
            if diis is not None:
                diis.clear()
            qune_prev_step = None
            qune_prev_energy = None
            qune_prev_fp = None
            qune_nls = 0
            if profile is not None:
                _t_iter_end = time.perf_counter()
                hist = profile.setdefault("history", [])
                if isinstance(hist, list):
                    ws_reused = None
                    ws_rebuild_mismatches = None
                    if isinstance(casci_profile, dict):
                        if "solver_matvec_cuda_ws_reused" in casci_profile:
                            ws_reused = bool(casci_profile.get("solver_matvec_cuda_ws_reused"))
                        if "solver_matvec_cuda_ws_rebuild_mismatches" in casci_profile:
                            _m = casci_profile.get("solver_matvec_cuda_ws_rebuild_mismatches")
                            ws_rebuild_mismatches = None if _m is None else list(_m)
                    hist.append(
                        {
                            "iter": int(it),
                            "e_avg": float(e_avg),
                            "e_roots": np.asarray(e_roots, dtype=np.float64).copy(),
                            "weights": np.asarray(weights, dtype=np.float64).copy(),
                            "grad_norm": float("nan"),
                            "t_casci_s": float(_t_post_casci - _t_iter_start),
                            "t_root_sort_s": float(_t_pre_rdm - _t_post_casci),
                            "t_rdm_s": 0.0,
                            "t_orbgrad_s": 0.0,
                            "t_conv_check_s": 0.0,
                            "t_orb_update_s": 0.0,
                            "t_iter_total_s": float(_t_iter_end - _t_iter_start),
                            "solver_matvec_cuda_ws_reused": ws_reused,
                            "solver_matvec_cuda_ws_rebuild_mismatches": ws_rebuild_mismatches,
                            "cueri_b_cache_resident": bool(cached_b_whitened is not None),
                            "orbital_diis_used": False,
                            "step_rejected": True,
                            "max_stepsize_cur": float(max_stepsize_cur),
                            "qune_step_mode": "reject",
                        }
                    )
                    profile["step_rejection_count"] = int(n_step_rejected)
            continue
        n_consecutive_rejected = 0
        if (
            step_recovery_enabled
            and e_ref is not None
            and float(max_stepsize_cur) < float(max_stepsize)
            and float(step_recovery_factor) > 1.0
        ):
            accepted_since_reject += 1
            if accepted_since_reject >= int(step_recovery_interval):
                max_stepsize_cur = min(float(max_stepsize), float(max_stepsize_cur) * float(step_recovery_factor))
                accepted_since_reject = 0

        if orbital_optimizer == "ah":
            try:
                from asuka.mcscf.newton_df import DFNewtonCASSCFAdapter  # noqa: PLC0415
                from asuka.mcscf.newton_casscf import update_orb_ci as _update_orb_ci  # noqa: PLC0415
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("orbital_optimizer='ah' requires asuka.mcscf.newton_casscf support") from exc
            try:
                import cupy as _cp_ah  # type: ignore
            except Exception:
                _cp_ah = None

            if ah_df_B_np is None:
                _raw_df_B = getattr(scf_out, "df_B", None)
                if _raw_df_B is not None:
                    if _cp_ah is not None and isinstance(_raw_df_B, _cp_ah.ndarray):
                        ah_df_B_np = _cp_ah.asnumpy(_raw_df_B)
                    else:
                        ah_df_B_np = np.asarray(_raw_df_B, dtype=np.float64)
            if ah_hcore_np is None:
                H0 = getattr(scf_out.int1e, "hcore")
                if _cp_ah is not None and isinstance(H0, _cp_ah.ndarray):
                    H0 = _cp_ah.asnumpy(H0)
                ah_hcore_np = np.asarray(H0, dtype=np.float64)

            C_np = C
            if _cp_ah is not None and isinstance(C_np, _cp_ah.ndarray):
                C_np = _cp_ah.asnumpy(C_np)
            C_np = np.asarray(C_np, dtype=np.float64)

            # Create a mixed-precision copy of the solver for the AH operator.
            # The Krylov solver is iterative and tolerant of ~1e-3 relative error,
            # so TF32 GEMM + FP32 coefficients are safe and roughly 2× faster.
            import copy as _copy
            ah_fcisolver = _copy.copy(fcisolver_use)
            if matvec_backend_s in {"cuda_eri_mat", "cuda"}:
                # Give the AH solver its own workspace caches so FP32 workspaces
                # don't overwrite the CASCI solver's FP64 workspaces.
                ah_fcisolver._matvec_cuda_ws_cache = {}
                ah_fcisolver._matvec_cuda_state_cache = {}
                ah_fcisolver._rdm_cuda_ws_cache = {}
                ah_fcisolver.matvec_cuda_gemm_backend = "gemmex_tf32"
                ah_fcisolver.matvec_cuda_fp32_coeff_data = True
                # Keep rdm_cuda_gemm_backend at FP64 — make_rdm12 workspace
                # requires FP64 dtype, and gemmex_tf32 is only valid for FP32.
                # trans_rdm12 is a smaller cost and stays FP64.

            # Resolve ao_eri for dense-mode AH (when df_B is None).
            _ah_ao_eri = None
            if ah_df_B_np is None and casci_backend_s in ("dense_gpu", "dense_cpu"):
                _ah_ao_eri = getattr(scf_out, "ao_eri", None)

            mc_ah = DFNewtonCASSCFAdapter(
                df_B=ah_df_B_np,
                ao_eri=_ah_ao_eri,
                hcore_ao=ah_hcore_np,
                ncore=int(ncore),
                ncas=int(ncas),
                nelecas=nelecas,
                mo_coeff=C_np,
                fcisolver=ah_fcisolver,
                dense_gpu_builder=dense_gpu_builder if casci_backend_s in ("dense_gpu", "dense_cpu") else None,
                weights=[float(w) for w in np.asarray(weights, dtype=np.float64).ravel().tolist()],
                frozen=None,
                internal_rotation=False,
                extrasym=None,
            )
            mc_ah.max_stepsize = float(max_stepsize_cur)
            mc_ah.ah_level_shift = float(ah_level_shift)
            mc_ah.ah_conv_tol = float(ah_conv_tol)
            mc_ah.ah_max_cycle = int(ah_max_cycle)
            mc_ah.ah_lindep = float(ah_lindep)
            mc_ah.ah_start_tol = float(ah_start_tol)
            mc_ah.ah_start_cycle = int(ah_start_cycle)
            mc_ah.max_cycle_micro = int(ah_max_cycle_micro)
            mc_ah.ah_grad_trust_region = float(ah_grad_trust_region)
            mc_ah.kf_interval = int(ah_kf_interval)
            mc_ah.kf_trust_region = float(ah_kf_trust_region)
            mc_ah.ah_max_cycle_micro_cap = None if ah_max_cycle_micro_cap_i is None else int(ah_max_cycle_micro_cap_i)
            mc_ah.verbose = 0

            _t_pre_ah = time.perf_counter() if profile is not None else 0.0
            eris_ah = mc_ah.ao2mo(C_np)
            ci_ah_in = ci_list if nroots > 1 else ci_list[0]
            U_ah, ci_ah_out, grad_norm_ah, ah_stat, ah_x0_guess = _update_orb_ci(
                mc_ah,
                C_np,
                ci_ah_in,
                eris_ah,
                x0_guess=ah_x0_guess,
                conv_tol_grad=float(ah_conv_tol_grad_eff),
                max_stepsize=float(max_stepsize_cur),
                verbose=0,
                weights=[float(w) for w in np.asarray(weights, dtype=np.float64).ravel().tolist()],
                gauge="none",
                convention="pyscf2",
                strict_weights=False,
                enforce_absorb_h1e_direct=True,
                implementation="internal",
                ci_update=str(ah_ci_update),
            )
            grad_norm = float(grad_norm_ah)
            de = float("inf") if e_last is None else abs(float(e_avg) - float(e_last))
            if e_last is not None and de < ah_conv_tol_energy_eff and grad_norm < ah_conv_tol_grad_eff:
                converged = True
            # Stall detection: energy converged for 3+ iters but grad slightly above threshold
            if not converged and e_last is not None:
                if de < max(tol * 100, ah_conv_tol_energy_eff * 10) and grad_norm < ah_conv_tol_grad_eff * 10:
                    _n_e_stall += 1
                else:
                    _n_e_stall = 0
                if _n_e_stall >= 3:
                    converged = True
            if converged:
                e_last = e_avg
                if profile is not None:
                    _t_iter_end = time.perf_counter()
                    hist = profile.setdefault("history", [])
                    if isinstance(hist, list):
                        hist.append(
                            {
                                "iter": int(it),
                                "e_avg": float(e_avg),
                                "e_roots": np.asarray(e_roots, dtype=np.float64).copy(),
                                "weights": np.asarray(weights, dtype=np.float64).copy(),
                                "grad_norm": float(grad_norm),
                                "t_casci_s": float(_t_post_casci - _t_iter_start),
                                "t_root_sort_s": float(_t_pre_rdm - _t_post_casci),
                                "t_rdm_s": 0.0,
                                "t_orbgrad_s": 0.0,
                                "t_conv_check_s": 0.0,
                                "t_orb_update_s": float(_t_iter_end - _t_pre_ah),
                                "t_iter_total_s": float(_t_iter_end - _t_iter_start),
                                "orbital_diis_used": False,
                                "lbfgs_fallback_used": False,
                                "qune_step_mode": "AH",
                                "ah_micro_iters": int(getattr(ah_stat, "imic", 0)),
                                "ah_hop_iters": int(getattr(ah_stat, "tot_hop", 0)),
                                "ah_keyframes": int(getattr(ah_stat, "tot_kf", 0)),
                            }
                        )
                break

            C_np = np.asarray(C_np, dtype=np.float64) @ np.asarray(U_ah, dtype=np.float64)
            if _cp_ah is not None and isinstance(C, _cp_ah.ndarray):
                C = _cp_ah.asarray(C_np, dtype=C.dtype)
            else:
                C = np.asarray(C_np, dtype=np.float64)
            ci_ah_list = ci_as_list(ci_ah_out, nroots=nroots)
            prev_ci_list = [np.asarray(c, dtype=np.float64).copy() for c in ci_ah_list]

            e_last = e_avg
            e_ref = e_avg
            C_ref = C.copy()

            if profile is not None:
                _t_iter_end = time.perf_counter()
                hist = profile.setdefault("history", [])
                if isinstance(hist, list):
                    hist.append(
                        {
                            "iter": int(it),
                            "e_avg": float(e_avg),
                            "e_roots": np.asarray(e_roots, dtype=np.float64).copy(),
                            "weights": np.asarray(weights, dtype=np.float64).copy(),
                            "grad_norm": float(grad_norm),
                            "t_casci_s": float(_t_post_casci - _t_iter_start),
                            "t_root_sort_s": float(_t_pre_rdm - _t_post_casci),
                            "t_rdm_s": 0.0,
                            "t_orbgrad_s": 0.0,
                            "t_conv_check_s": 0.0,
                            "t_orb_update_s": float(_t_iter_end - _t_pre_ah),
                            "t_iter_total_s": float(_t_iter_end - _t_iter_start),
                            "orbital_diis_used": False,
                            "lbfgs_fallback_used": False,
                            "qune_step_mode": "AH",
                            "ah_micro_iters": int(getattr(ah_stat, "imic", 0)),
                            "ah_hop_iters": int(getattr(ah_stat, "tot_hop", 0)),
                            "ah_keyframes": int(getattr(ah_stat, "tot_kf", 0)),
                        }
                    )
            continue

        dm1_act, dm2_act = make_state_averaged_rdms(
            fcisolver_use,
            ci_list,
            weights,
            ncas=int(ncas),
            nelecas=nelecas,
            solver_kwargs=solver_kwargs,
        )

        orbgrad_profile = profile.setdefault("orbgrad", {}) if profile is not None else None
        _t_pre_orbgrad = time.perf_counter() if profile is not None else 0.0
        if casci_backend_s == "df":
            gmat, grad_norm, eps = orbital_gradient_df(
                scf_out,
                C=C,
                ncore=int(ncore),
                ncas=int(ncas),
                dm1_act=dm1_act,
                dm2_act=dm2_act,
                allowed=allowed,
                profile=orbgrad_profile,
            )
        else:
            eps_ao_use = float(dense_cpu_eps_ao) if casci_backend_s == "dense_cpu" else float(dense_gpu_eps_ao)
            threads_use = int(dense_cpu_threads) if casci_backend_s == "dense_cpu" else int(dense_gpu_threads)
            blas_use = None if dense_cpu_blas_nthreads is None else int(dense_cpu_blas_nthreads)

            gmat, grad_norm, eps = orbital_gradient_dense(
                scf_out,
                C=C,
                ncore=int(ncore),
                ncas=int(ncas),
                dm1_act=dm1_act,
                dm2_act=dm2_act,
                allowed=allowed,
                dense_eps_ao=float(eps_ao_use),
                dense_max_tile_bytes=int(max_tile_bytes),
                dense_cpu_threads=int(threads_use),
                dense_cpu_blas_nthreads=blas_use,
                dense_cpu_p_block_nmo=64,
                dense_gpu_threads=int(threads_use),
                dense_gpu_builder=dense_gpu_builder,
                dense_exact_jk=bool(dense_exact_jk),
                profile=orbgrad_profile,
            )
        g_vec = gmat[allowed_xp].ravel()
        _t_post_orbgrad = time.perf_counter() if profile is not None else 0.0

        if profile is not None:
            ws_reused = None
            ws_rebuild_mismatches = None
            b_cache_hit_iter = None
            b_cache_populated_iter = None
            if isinstance(casci_profile, dict):
                if "solver_matvec_cuda_ws_reused" in casci_profile:
                    ws_reused = bool(casci_profile.get("solver_matvec_cuda_ws_reused"))
                if "solver_matvec_cuda_ws_rebuild_mismatches" in casci_profile:
                    _m = casci_profile.get("solver_matvec_cuda_ws_rebuild_mismatches")
                    ws_rebuild_mismatches = None if _m is None else list(_m)
                active_df_prof = casci_profile.get("active_df", {})
                if isinstance(active_df_prof, dict):
                    cueri_prof = active_df_prof.get("cueri_active_df", {})
                    if isinstance(cueri_prof, dict):
                        if "b_cache_hit" in cueri_prof:
                            b_cache_hit_iter = bool(cueri_prof.get("b_cache_hit"))
                        if "b_cache_populated" in cueri_prof:
                            b_cache_populated_iter = bool(cueri_prof.get("b_cache_populated"))
            hist = profile.setdefault("history", [])
            _t_post_conv_check = time.perf_counter()
            if isinstance(hist, list):
                hist.append(
                    {
                        "iter": int(it),
                        "e_avg": float(e_avg),
                        "e_roots": np.asarray(e_roots, dtype=np.float64).copy(),
                        "weights": np.asarray(weights, dtype=np.float64).copy(),
                        "grad_norm": float(grad_norm),
                        "t_casci_s": float(_t_post_casci - _t_iter_start),
                        "t_root_sort_s": float(_t_pre_rdm - _t_post_casci),
                        "t_rdm_s": float(_t_pre_orbgrad - _t_pre_rdm),
                        "t_orbgrad_s": float(_t_post_orbgrad - _t_pre_orbgrad),
                        "t_conv_check_s": float(_t_post_conv_check - _t_post_orbgrad),
                        "solver_matvec_cuda_ws_reused": ws_reused,
                        "solver_matvec_cuda_ws_rebuild_mismatches": ws_rebuild_mismatches,
                        "cueri_b_cache_hit": b_cache_hit_iter,
                        "cueri_b_cache_populated": b_cache_populated_iter,
                        "cueri_b_cache_resident": bool(cached_b_whitened is not None),
                        "step_rejected": False,
                        "max_stepsize_cur": float(max_stepsize_cur),
                    }
                )

        if e_last is not None:
            de = abs(float(e_avg) - float(e_last))
            if de < tol and grad_norm < conv_tol_grad:
                converged = True
                e_last = e_avg
                break
        e_last = e_avg
        e_ref = e_avg
        C_ref = C.copy()

        # 3) Orbital update (Cayley transform).
        #    Default: L-BFGS in the non-redundant rotation variables with a diagonal
        #    (energy-difference) preconditioner as H0.
        denom = eps[:, None] - eps[None, :]
        denom_floor = 1e-8
        denom = xp.where(xp.abs(denom) < denom_floor, denom_floor, xp.abs(denom))

        inv_h = 1.0 / denom[allowed_xp].ravel()

        step_vec_grad = (-damp) * (g_vec * inv_h)
        lbfgs_fallback_used = False
        if orbital_optimizer == "lbfgs" and lbfgs is not None and lbfgs.enabled:
            if lbfgs_prev_g is not None and lbfgs_prev_step is not None and allowed_nvar:
                y = g_vec - lbfgs_prev_g
                lbfgs.update(lbfgs_prev_step, y)
            step_vec_lbfgs = damp * lbfgs.direction(g_vec, h0_inv=inv_h)
            if lbfgs_descent_guard and allowed_nvar:
                proj = float((g_vec * step_vec_lbfgs).sum().item())
                if (not np.isfinite(proj)) or proj >= 0.0:
                    step_vec = step_vec_grad
                    lbfgs_fallback_used = True
                    if lbfgs_descent_guard_clear_history:
                        lbfgs.clear()
                        lbfgs_prev_g = None
                        lbfgs_prev_step = None
                else:
                    step_vec = step_vec_lbfgs
            else:
                step_vec = step_vec_lbfgs
        else:
            step_vec = step_vec_grad

        qune_mode = "QN" if orbital_optimizer == "lbfgs" else "SX"
        if (
            qune_enabled
            and allowed_nvar
            and qune_prev_step is not None
            and qune_prev_energy is not None
            and qune_prev_fp is not None
        ):
            fp_now = 2.0 * float((g_vec * qune_prev_step).sum().item())
            tmin, emin, _has_local = _qune_predict_tmin(
                e_last=float(qune_prev_energy),
                e_now=float(e_avg),
                fp_last=float(qune_prev_fp),
                fp_now=float(fp_now),
            )
            epred_ls = float(emin - float(e_avg))
            epred_qn = 0.5 * float((g_vec * step_vec).sum().item())
            epred_sx = 0.5 * float((g_vec * step_vec_grad).sum().item())
            if (
                abs(float(tmin - 1.0)) < float(qune_close_threshold)
                or abs(float(epred_ls)) < 1.0e-8
                or qune_nls >= int(qune_ls_max_consecutive)
            ):
                if epred_sx < epred_qn:
                    step_vec = step_vec_grad
                    qune_mode = "SX"
                else:
                    qune_mode = "QN"
                qune_nls = 0
            else:
                if epred_ls > epred_qn:
                    if epred_sx < epred_qn:
                        step_vec = step_vec_grad
                        qune_mode = "SX"
                    else:
                        qune_mode = "QN"
                    qune_nls = 0
                elif epred_ls > epred_sx:
                    step_vec = step_vec_grad
                    qune_mode = "SX"
                    qune_nls = 0
                else:
                    step_vec = step_vec + float(tmin - 1.0) * qune_prev_step
                    qune_mode = "LS"
                    qune_nls += 1
            if float(e_avg) > float(qune_prev_energy):
                step_vec = step_vec * float(qune_uphill_scale)

        diis_used = False
        if (
            diis is not None
            and diis.enabled
            and allowed_nvar
            and int(step_vec.size) == int(g_vec.size)
            and int(step_vec.size) > 0
        ):
            diis.push(step_vec, g_vec)
            if int(it) >= int(diis_start_cycle):
                step_vec_diis = diis.extrapolate()
                if step_vec_diis is not None:
                    step_vec = step_vec_diis
                    diis_used = True

        max_abs = float(xp.max(xp.abs(step_vec)).item()) if allowed_nvar else 0.0
        if max_abs > max_stepsize_cur:
            step_vec = step_vec * (float(max_stepsize_cur) / float(max_abs))
        if allowed_nvar and float(step_norm_scale_alpha) > 0.0:
            step_norm = float(xp.linalg.norm(step_vec).item())
            step_scale = 1.0 / (1.0 + float(step_norm_scale_alpha) * step_norm)
            step_vec = step_vec * float(step_scale)

        step_lower = xp.zeros((nmo, nmo), dtype=xp.float64)
        step_lower[allowed_xp] = step_vec
        step_lower = xp.tril(step_lower, k=-1)
        A = step_lower - step_lower.T

        U = cayley_update(xp, A)
        C = C @ U
        if orbital_optimizer == "lbfgs" and lbfgs is not None and lbfgs.enabled:
            lbfgs_prev_g = g_vec.copy()
            lbfgs_prev_step = step_vec.copy()
        qune_prev_step = step_vec.copy() if allowed_nvar else None
        qune_prev_energy = float(e_avg)
        qune_prev_fp = 2.0 * float((g_vec * step_vec).sum().item()) if allowed_nvar else None

        # Record orbital-update timing in the last history entry
        if profile is not None:
            _t_iter_end = time.perf_counter()
            hist = profile.get("history")
            if isinstance(hist, list) and hist:
                hist[-1]["t_orb_update_s"] = float(_t_iter_end - _t_post_conv_check)
                hist[-1]["t_iter_total_s"] = float(_t_iter_end - _t_iter_start)
                hist[-1]["orbital_diis_used"] = bool(diis_used)
                hist[-1]["lbfgs_fallback_used"] = bool(lbfgs_fallback_used)
                hist[-1]["qune_step_mode"] = str(qune_mode)

    if casci_out is None:  # pragma: no cover
        raise RuntimeError("internal error: CASSCF loop did not produce a CASCI result")

    # Final full CASCI to get accurate energy when inner iterations used
    # reduced Davidson cycles.
    if ci_max_cycle_inner is not None and converged and casci_backend_s == "df":
        cached_b_in = cached_b_whitened if b_cache_enabled else None
        casci_out = run_casci_df(
            scf_out,
            ncore=int(ncore),
            ncas=int(ncas),
            nelecas=nelecas,
            mo_coeff=C,
            ci0=prev_ci_list,
            fcisolver=fcisolver_use,
            twos=twos,
            nroots=int(nroots),
            matvec_backend=str(matvec_backend_s),
            want_eri_mat=bool(want_eri_mat),
            aux_block_naux=int(aux_block_naux),
            max_tile_bytes=int(max_tile_bytes),
            cached_b_whitened=cached_b_in,
            **solver_kwargs,
        )
        if nroots == 1:
            e_roots = np.asarray([float(casci_out.e_tot)], dtype=np.float64)
        else:
            e_roots = np.asarray(casci_out.e_tot, dtype=np.float64).ravel()
        ci_list = ci_as_list(casci_out.ci, nroots=nroots)
        if prev_ci_list is not None and nroots > 1:
            perm = match_roots_by_overlap(prev_ci_list, ci_list)
            e_roots = e_roots[perm]
            ci_list = [ci_list[int(j)] for j in perm.tolist()]
            fix_ci_phases(prev_ci_list, ci_list)
        ci_out = ci_list if nroots > 1 else ci_list[0]
        e_last = float(np.dot(weights, e_roots))

    return CASSCFResult(
        mol=scf_out.mol,
        basis_name=str(scf_out.basis_name),
        auxbasis_name=str(scf_out.auxbasis_name),
        converged=bool(converged),
        niter=int(it),
        ncore=int(ncore),
        ncas=int(ncas),
        nelecas=nelecas,
        nroots=int(nroots),
        root_weights=np.asarray(weights, dtype=np.float64),
        e_roots=np.asarray(e_roots, dtype=np.float64),
        e_tot=float(e_last if e_last is not None else float(np.dot(weights, e_roots))),
        ecore=float(casci_out.ecore),
        ci=ci_out if ci_out is not None else casci_out.ci,
        mo_coeff=C,
        grad_norm=float(grad_norm),
        casci=casci_out,
        profile=profile,
        scf_out=scf_out,
    )


def run_casscf_dense_cpu(
    scf_out: RHFDFRunResult | ROHFDFRunResult,
    *,
    ncore: int,
    ncas: int,
    nelecas: int | tuple[int, int],
    fcisolver: GUGAFCISolver | None = None,
    twos: int | None = None,
    nroots: int = 1,
    root_weights: Sequence[float] | None = None,
    matvec_backend: str = "contract",
    max_tile_bytes: int = 256 * 1024 * 1024,
    max_cycle_macro: int = 30,
    tol: float = 1e-8,
    conv_tol_grad: float | None = None,
    max_stepsize: float = 0.02,
    damp: float = 1.0,
    orbital_optimizer: str = "ah",
    lbfgs_history: int = 10,
    dense_cpu_eps_ao: float = 0.0,
    dense_cpu_eps_mo: float = 0.0,
    dense_cpu_threads: int = 0,
    dense_cpu_blas_nthreads: int | None = None,
    profile: dict | None = None,
    **solver_kwargs,
) -> CASSCFResult:
    """Convenience wrapper for CASSCF with dense CPU active-space ERIs in the CI solve."""

    return run_casscf_df(
        scf_out,
        ncore=int(ncore),
        ncas=int(ncas),
        nelecas=nelecas,
        fcisolver=fcisolver,
        twos=twos,
        nroots=int(nroots),
        root_weights=root_weights,
        casci_backend="dense_cpu",
        matvec_backend=str(matvec_backend),
        want_eri_mat=False,
        aux_block_naux=0,
        max_tile_bytes=int(max_tile_bytes),
        max_cycle_macro=int(max_cycle_macro),
        tol=float(tol),
        conv_tol_grad=conv_tol_grad,
        max_stepsize=float(max_stepsize),
        damp=float(damp),
        orbital_optimizer=str(orbital_optimizer),
        lbfgs_history=int(lbfgs_history),
        dense_cpu_eps_ao=float(dense_cpu_eps_ao),
        dense_cpu_eps_mo=float(dense_cpu_eps_mo),
        dense_cpu_threads=int(dense_cpu_threads),
        dense_cpu_blas_nthreads=None if dense_cpu_blas_nthreads is None else int(dense_cpu_blas_nthreads),
        profile=profile,
        **solver_kwargs,
    )


def run_casscf(
    scf_out: RHFDFRunResult | ROHFDFRunResult | UHFDFRunResult,
    *,
    ncore: int,
    ncas: int,
    nelecas: int | tuple[int, int],
    backend: str = "cuda",
    df: bool = True,
    guess: Any | None = None,
    matvec_backend: str | None = None,
    **kwargs,
) -> CASSCFResult:
    """Unified CASSCF driver over (backend, df) switches."""

    backend_s = str(backend).strip().lower()
    if backend_s not in {"cpu", "cuda"}:
        raise ValueError("backend must be 'cpu' or 'cuda'")
    df_b = bool(df)

    C0 = getattr(scf_out.scf, "mo_coeff", None)
    _xp_probe = scf_out.df_B if scf_out.df_B is not None else getattr(scf_out, "ao_eri", C0)
    _, is_gpu = _df_scf._get_xp(_xp_probe, C0)  # noqa: SLF001
    if backend_s == "cuda" and not bool(is_gpu):
        raise ValueError("backend='cuda' requires scf_out with GPU arrays (use frontend.run_*_df or run_*_dense with backend='cuda')")

    if matvec_backend is None:
        matvec_backend = "cuda_eri_mat" if backend_s == "cuda" else "contract"

    if guess is not None:
        kwargs = dict(kwargs)
        kwargs.setdefault("mo_coeff0", getattr(guess, "mo_coeff", None))
        kwargs.setdefault("ci0", getattr(guess, "ci", None))
    elif kwargs:
        kwargs = dict(kwargs)

    if not df_b:
        kwargs.setdefault("dense_exact_jk", "auto")

    if df_b:
        return run_casscf_df(
            scf_out,
            ncore=int(ncore),
            ncas=int(ncas),
            nelecas=nelecas,
            casci_backend="df",
            matvec_backend=str(matvec_backend),
            **kwargs,
        )

    if backend_s == "cuda":
        return run_casscf_df(
            scf_out,
            ncore=int(ncore),
            ncas=int(ncas),
            nelecas=nelecas,
            casci_backend="dense_gpu",
            matvec_backend=str(matvec_backend),
            **kwargs,
        )

    return run_casscf_df(
        scf_out,
        ncore=int(ncore),
        ncas=int(ncas),
        nelecas=nelecas,
        casci_backend="dense_cpu",
        matvec_backend=str(matvec_backend),
        **kwargs,
    )


__all__ = [
    "CASSCFResult",
    "casscf_orbital_gradient_df",
    "casscf_orbital_gradient_dense",
    "run_casscf",
    "run_casscf_df",
    "run_casscf_dense_cpu",
]
