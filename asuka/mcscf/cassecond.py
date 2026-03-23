"""Second-order CASSCF optimizer (CASSecond).

Uses an augmented Hessian approach with exact diagonal Hessian
preconditioning and micro-iterations.  The algorithm follows:

    T. Shiozaki, WIREs Comput. Mol. Sci. 8, e1331 (2018).

Implementation based on the BAGEL electronic structure package
(https://github.com/qsimulate-open/bagel, GPLv3+).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False


def _get_xp(device: str = "auto"):
    """Return the array module (cupy or numpy) based on device preference."""
    if device == "cpu" or not _HAS_CUPY:
        return np
    return cp


def _to_np(x):
    """Convert any array (cupy or numpy) to numpy."""
    if _HAS_CUPY and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def _to_xp(x, xp):
    """Convert array to the target array module."""
    if xp is np:
        return _to_np(x)
    return xp.asarray(x, dtype=xp.float64)



def _einsum(*args, **kwargs):
    """Dispatch einsum to cupy or numpy based on input arrays."""
    for a in args[1:]:  # skip the subscript string
        if _HAS_CUPY and isinstance(a, cp.ndarray):
            return cp.einsum(*args, **kwargs)
    return np.einsum(*args, **kwargs)


def _np_einsum(*args, **kwargs):
    """Einsum that always returns a numpy array (for RotFile-compatible results)."""
    result = _einsum(*args, **kwargs)
    return _to_np(result)

from asuka.mcscf.rotfile import RotFile
from asuka.mcscf.aughess import AugHess
from asuka.mcscf.qvec import build_qvec


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CASSecondResult:
    converged: bool
    niter: int
    e_tot: float
    e_roots: np.ndarray
    mo_coeff: np.ndarray
    mo_energy: np.ndarray | None
    ci: Any
    ncore: int
    ncas: int
    nelecas: int | tuple[int, int]
    nroots: int
    occup: np.ndarray | None = None
    profile: dict | None = None


# ---------------------------------------------------------------------------
# DF helper operations
# ---------------------------------------------------------------------------
# ASUKA stores B_ao as (nao, nao, naux) already J^{-1/2}-whitened.
# Contracting (mu nu | P) with MO coefficients gives half/full transforms.
# Since B is whitened, sum_P L[mu,i,P] L[nu,j,P] = (mu i | nu j) with the
# Coulomb metric already applied once per index.

def _half_transform(B_ao, C):
    """Half-transform: half[mu, i, P] = sum_nu B[mu, nu, P] C[nu, i].

    Returns shape (nao, norb, naux).
    """
    return _einsum("mnP,ni->miP", B_ao, C, optimize=True)


def _full_transform(half, C2):
    """Full transform: full[i, j, P] = sum_mu C2[mu, j] half[mu, i, P].

    Parameters
    ----------
    half : (nao, n1, naux)
    C2 : (nao, n2)

    Returns shape (n1, n2, naux).
    """
    return _einsum("mu,miP->iuP", C2, half, optimize=True)



def _form_4index_diagonal_from_full(full):
    """Given full[i, j, P], return M[i, j] = sum_P full[i, j, P]^2."""
    return _einsum("ijP,ijP->ij", full, full, optimize=True)


def _form_4index_diagonal_part(full):
    """Compute the diagonal part of the 4-index tensor from DF full-transform.

    Given full[k, i, P] with b1=n1, b2=n2:
        result[k + n1*j, i] = sum_P full[k, i, P] * full[j, i, P]

    Returns shape (n1*n1, n2).
    """
    n1, n2, naux = full.shape
    result = np.zeros((n1 * n1, n2), dtype=np.float64)
    for i in range(n2):
        for j in range(n1):
            for k in range(n1):
                result[k + n1 * j, i] = np.dot(full[k, i, :], full[j, i, :])
    return result


def _apply_2rdm(full, rdm2):
    """Apply 2-RDM: result[t, u, P] = sum_{x,y} full[x, y, P] rdm2[x,y,t,u].

    Parameters
    ----------
    full : (nact, nact, naux)
    rdm2 : (nact, nact, nact, nact)

    Returns (nact, nact, naux).
    """
    na = full.shape[0]
    naux = full.shape[2]
    full_flat = full.reshape(na * na, naux)
    rdm2_flat = rdm2.reshape(na * na, na * na)
    out_flat = rdm2_flat.T @ full_flat  # (na*na, naux)
    return out_flat.reshape(na, na, naux)


def _form_2index(half1, full2):
    """Compute 2-index intermediate: M[mu, u] = sum_{t, P} half1[mu, t, P] * full2[t, u, P].

    Parameters
    ----------
    half1 : (nao, n1, naux)
    full2 : (n1, n2, naux)

    Returns (nao, n2).
    """
    return _einsum("mtP,tuP->mu", half1, full2, optimize=True)



def _compute_gd_from_density(B_ao, D):
    """Compute g(D) = J(D) - 0.5*K(D) from the AO density and whitened DF factors.

    Parameters
    ----------
    B_ao : (nao, nao, naux) -- whitened DF factors
    D : (nao, nao) -- AO density matrix (need not be symmetric)

    Returns
    -------
    gd : (nao, nao)
    """
    J, K = _compute_JK_from_density(B_ao, D)
    return J - 0.5 * K


def _compute_JK_from_density(B_ao, D):
    """Compute J(D) and K(D) separately."""
    # Coulomb
    rho = _einsum("mnP,mn->P", B_ao, D, optimize=True)
    J = _einsum("mnP,P->mn", B_ao, rho, optimize=True)

    # Exchange: K[mu,nu] = sum_{rho,sigma,P} B[mu,rho,P] * D[rho,sigma] * B[sigma,nu,P]
    tmp = _einsum("mrP,rs->msP", B_ao, D, optimize=True)  # (nao, nao, naux)
    K = _einsum("msP,snP->mn", tmp, B_ao, optimize=True)
    K = 0.5 * (K + K.T)

    return J, K


def _compute_gd_from_halfs(B_ao: np.ndarray, half_t: np.ndarray, half_ref: np.ndarray,
                            C_partner: np.ndarray) -> np.ndarray:
    """Compute g(D) = J(D) - 0.5*K from half-transforms.

    Compute g(D) = J(D) - 0.5*K(D) from DF half-transforms:
      gd = df->compute_Jop(halft, pcoefft)   // Coulomb
      ex0 = halfjj->form_2index(halft, 1.0)  // Exchange
      gd -= 0.5 * sym(ex0)

    Parameters
    ----------
    B_ao : (nao, nao, naux) -- whitened DF factors
    half_t : (nao, n_orb, naux) -- half-transform of trial coefficients
    half_ref : (nao, n_orb, naux) -- half-transform of reference coefficients (JJ-applied)
    C_partner : (nao, n_orb) -- partner orbital coefficients

    The density is D = tcoeff @ C_partner.T where tcoeff is implicit in half_t.
    J and K must share the SAME orbital index dimension n_orb.
    """
    nao = B_ao.shape[0]
    n_orb = half_t.shape[1]

    # Coulomb: J_Pmu = sum_{nu,i} half_t[nu, i, P] * C_partner[nu, i]
    # rho_P = sum_{nu,i} half_t[nu,i,P] * C_partner[nu,i]
    rho = _einsum("niP,ni->P", half_t, C_partner, optimize=True)
    J = _einsum("mnP,P->mn", B_ao, rho, optimize=True)

    # Exchange: K[mu,nu] = sum_{i,P} half_ref[mu,i,P] * half_t[nu,i,P]
    K = _einsum("miP,niP->mn", half_ref, half_t, optimize=True)
    K = 0.5 * (K + K.T)

    return J - 0.5 * K


# ---------------------------------------------------------------------------
# CASSecond optimizer
# ---------------------------------------------------------------------------

class CASSecond:
    """Second-order CASSCF optimizer with augmented Hessian orbital optimization."""

    def __init__(
        self,
        scf_out: Any,
        ncore: int,
        ncas: int,
        nelecas: int | tuple[int, int],
        *,
        nroots: int = 1,
        root_weights: np.ndarray | None = None,
        max_iter: int = 100,
        thresh: float = 1e-8,
        max_micro_iter: int = 100,
        thresh_micro: float | None = None,
        thresh_microstep: float = 1e-4,
        mo_coeff: np.ndarray | None = None,
        ci0: Any = None,
        fcisolver: Any | None = None,
        twos: int | None = None,
        natocc: bool = True,
        verbose: bool = True,
    ):
        self.scf_out = scf_out
        self.ncore = int(ncore)
        self.ncas = int(ncas)
        self.nelecas = nelecas
        self.nroots = int(nroots)
        self.root_weights = root_weights
        self.max_iter = int(max_iter)
        self.thresh = float(thresh)
        self.max_micro_iter = int(max_micro_iter)
        self.thresh_micro = float(thresh_micro) if thresh_micro is not None else float(thresh * 0.5)
        self.thresh_microstep = float(thresh_microstep)
        self.natocc = bool(natocc)
        self.verbose = bool(verbose)

        # Extract from SCF result and place on target device
        # Use GPU for all DF contractions if cupy is available
        self.xp = np  # orbital optimization on CPU; FCI solve uses GPU

        hcore_np = np.asarray(_to_np(scf_out.int1e.hcore), dtype=np.float64)
        B_ao_np = _to_np(scf_out.df_B)
        if B_ao_np.ndim == 2:
            from asuka.integrals.df_packed_s2 import unpack_Qp_to_mnQ
            nao_from_coeff = _to_np(scf_out.mo_coeff).shape[0] if hasattr(scf_out, 'mo_coeff') else int(hcore_np.shape[0])
            B_ao_np = np.asarray(unpack_Qp_to_mnQ(B_ao_np, nao=nao_from_coeff), dtype=np.float64)
        if B_ao_np.ndim != 3:
            raise ValueError("df_B must have shape (nao, nao, naux)")

        self.hcore = _to_xp(hcore_np, self.xp)
        self.B_ao = _to_xp(B_ao_np, self.xp)
        self.enuc = float(scf_out.mol.energy_nuc())

        # MO coefficients
        if mo_coeff is not None:
            self.coeff = _to_xp(_to_np(mo_coeff), self.xp)
        else:
            self.coeff = _to_xp(_to_np(scf_out.mo_coeff), self.xp)

        nao, nmo = self.coeff.shape
        self.nao = nao
        self.nmo = nmo
        self.nclosed = self.ncore
        self.nact = self.ncas
        self.nvirt = nmo - self.nclosed - self.nact
        self.nocc = self.nclosed + self.nact

        if self.nvirt < 0:
            raise ValueError("ncore + ncas exceeds nmo")
        if self.nact <= 0:
            raise ValueError("ncas must be > 0")

        # FCI solver
        if fcisolver is not None:
            self.fcisolver = fcisolver
        else:
            from asuka.solver import GUGAFCISolver
            if twos is None:
                twos = int(getattr(scf_out.mol, "spin", 0))
            self.fcisolver = GUGAFCISolver(twos=int(twos), nroots=int(nroots))

        self.ci = ci0
        self.energy = None
        self.rdm1_av = None
        self.rdm2_av = None

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # ------------------------------------------------------------------ CASCI
    def _solve_casci(self) -> None:
        """Solve CASCI for current orbitals and compute RDMs."""
        C = self.coeff
        C_core = C[:, :self.nclosed]
        C_act = C[:, self.nclosed:self.nocc]

        # Core energy and effective 1e Hamiltonian
        h_mo = C.T @ self.hcore @ C

        # Core Fock in AO
        if self.nclosed > 0:
            D_core = 2.0 * (C_core @ C_core.T)
            gd_core = _compute_gd_from_density(self.B_ao, D_core)
            # Closed Fock = hcore + J_core - 0.5*K_core (but we want the
            # mean-field for the core).
            # the reference implementation: cfockao = hcore + 2*J_core - K_core  (for 2-electron doubly occ)
            #   J_core uses D_core = 2*C_core@C_core.T
            # With g(D_core) = J(D_core) - 0.5*K(D_core), and D_core = 2*Cc@Cc.T:
            #   g(D_core) = 2*J_single - 0.5*(2*K_single) = 2*J_single - K_single
            # cfock_ao = hcore + g(D_core)  is NOT right, because g(D) = J(D)-0.5K(D)
            # and for closed shell we want hcore + 2J - K.
            # With D = 2*Cc@Cc.T: J(D) = 2*J_1, K(D) = 2*K_1
            # g(D) = 2J_1 - 0.5*2*K_1 = 2J_1 - K_1
            # So cfock_ao = hcore + g(D_core) = hcore + 2J - K.  Correct!
            vhf_core = gd_core  # J(D_core) - 0.5*K(D_core)
        else:
            vhf_core = np.zeros((self.nao, self.nao), dtype=np.float64)

        # ecore = enuc + Tr(D_core @ (hcore + 0.5*vhf_core))
        #       = enuc + sum_i 2*h_ii + sum_i (2J_ii - K_ii)
        if self.nclosed > 0:
            ecore = self.enuc + _einsum("ij,ji->", D_core, self.hcore + 0.5 * vhf_core)
        else:
            ecore = self.enuc

        # Active integrals
        cfock_ao = self.hcore + vhf_core
        h1eff = C_act.T @ cfock_ao @ C_act  # (ncas, ncas)

        # 2e integrals in active space via DF
        # (tu|xy) = sum_P L[t,u,P] L[x,y,P]
        # L[t,u,P] = sum_{mu,nu} C_act[mu,t] B[mu,nu,P] C_act[nu,u]
        half_act = _half_transform(self.B_ao, C_act)  # (nao, nact, naux)
        L_tuP = _einsum("mt,muP->tuP", C_act, half_act)  # (nact, nact, naux)

        # ERI (tu|xy) = sum_P L[t,u,P] * L[x,y,P]
        nact = self.nact
        naux = self.B_ao.shape[2]
        L_flat = L_tuP.reshape(nact * nact, naux)
        eri_act = (L_flat @ L_flat.T).reshape(nact, nact, nact, nact)

        # Solve FCI (solver expects numpy arrays)
        e_roots, ci = self.fcisolver.kernel(
            _to_np(h1eff), _to_np(eri_act), nact, self.nelecas,
            ci0=self.ci, ecore=float(ecore), nroots=self.nroots,
        )
        if self.nroots == 1:
            e_roots = np.atleast_1d(e_roots)

        self.energy = e_roots.copy()
        self.ecore = float(ecore)
        self.h1eff = h1eff
        self.eri_act = eri_act

        # Compute RDMs (keep on CPU — they're small and used in RotFile operations)
        if self.nroots == 1:
            ci_for_rdm = ci
            dm1, dm2 = self.fcisolver.make_rdm12(ci_for_rdm, nact, self.nelecas)
            self.rdm1_av = np.asarray(_to_np(dm1), dtype=np.float64)
            self.rdm2_av = np.asarray(_to_np(dm2), dtype=np.float64)
        else:
            weights = self.root_weights
            if weights is None:
                weights = np.ones(self.nroots, dtype=np.float64) / self.nroots
            else:
                weights = np.asarray(weights, dtype=np.float64)
                weights = weights / weights.sum()

            dm1_av = np.zeros((nact, nact), dtype=np.float64)
            dm2_av = np.zeros((nact, nact, nact, nact), dtype=np.float64)
            for iroot in range(self.nroots):
                d1, d2 = self.fcisolver.make_rdm12(ci[iroot], nact, self.nelecas)
                dm1_av += weights[iroot] * np.asarray(d1, dtype=np.float64)
                dm2_av += weights[iroot] * np.asarray(d2, dtype=np.float64)
            self.rdm1_av = np.asarray(dm1_av, dtype=np.float64)
            self.rdm2_av = np.asarray(dm2_av, dtype=np.float64)

        self.ci = ci

    # ---------------------------------------------------------------- natorb
    def _trans_natorb(self) -> None:
        """Transform to natural orbitals within the active space.

        Translation of ``CASSecond::trans_natorb`` (cassecond.cc:439-462).
        """
        rdm1 = _to_np(self.rdm1_av)
        nact = self.nact
        # Diagonalize 2*I - rdm1 to get natural orbitals
        trans = 2.0 * np.eye(nact) - rdm1
        occup, U = np.linalg.eigh(trans)

        if self.natocc:
            occ_print = np.where(occup < 2.0, 2.0 - occup, 0.0)
            self._log("")
            self._log("  ========       state-averaged       ========")
            self._log("  ======== natural occupation numbers ========")
            for idx, occ_val in enumerate(occ_print):
                self._log(f"   Orbital {idx} : {occ_val:.4f}")
            self._log("  ============================================")

        # Rotate RDMs (stay on CPU)
        self.rdm1_av = U.T @ rdm1 @ U
        self.rdm2_av = np.einsum("pa,qb,pqrs,rc,sd->abcd", U, U, self.rdm2_av, U, U,
                                  optimize=True)

        U_dev = _to_xp(U, self.xp)

        # Rotate MO coefficients (active block only)
        xp = self.xp
        U_dev = _to_xp(U, xp)
        cnew = self.coeff.copy()
        cnew[:, self.nclosed:self.nocc] = self.coeff[:, self.nclosed:self.nocc] @ U_dev
        self.coeff = cnew

    # ---------------------------------------------------------------- Fock
    def _build_fock(self) -> tuple[np.ndarray, np.ndarray]:
        """Build cfock and afock in MO basis.

        cfock_ao = hcore + 2*J_closed - K_closed = hcore + g(D_closed)
        afock_ao = J_active(dm1) - 0.5*K_active(dm1) = g(D_active) - J_active*0 ...

        the reference implementation convention:
          cfock_ao = core_fock = hcore + 2J_c - K_c
          afock_ao = active_fock = J_a(dm1) - 0.5*K_a(dm1)

        With g(D) = J(D) - 0.5*K(D):
          cfock_ao = hcore + g(2*C_c@C_c.T) = hcore + 2J_c - K_c  (correct)
          afock_ao = g(C_a @ dm1 @ C_a.T) = J(D_a) - 0.5*K(D_a)  (correct)
        """
        C = self.coeff
        C_core = C[:, :self.nclosed]
        C_act = C[:, self.nclosed:self.nocc]

        if self.nclosed > 0:
            D_core = 2.0 * (C_core @ C_core.T)
            cfock_ao = self.hcore + _compute_gd_from_density(self.B_ao, D_core)
        else:
            cfock_ao = self.hcore.copy()

        rdm1_dev = _to_xp(self.rdm1_av, self.xp)
        D_act = C_act @ rdm1_dev @ C_act.T
        afock_ao = _compute_gd_from_density(self.B_ao, D_act)

        cfock = _to_np(C.T @ cfock_ao @ C)
        afock = _to_np(C.T @ afock_ao @ C)

        return cfock, afock

    # ------------------------------------------------------------ gradient
    def _compute_gradient(
        self, cfock: np.ndarray, afock: np.ndarray, qxr: np.ndarray
    ) -> RotFile:
        """Compute orbital gradient.  Translation of cassecond.cc:188-216."""
        nclosed = self.nclosed
        nact = self.nact
        nvirt = self.nvirt
        nocc = self.nocc
        rdm1 = _to_np(self.rdm1_av)

        sigma = RotFile(nclosed, nact, nvirt)

        # VC block: 4*(cfock + afock)[virt, closed]
        if nclosed:
            vc_mat = 4.0 * (cfock[nocc:, :nclosed] + afock[nocc:, :nclosed])
            sigma.ax_plus_y_vc(1.0, vc_mat)

        # VA block
        va_mat = np.zeros((nvirt, nact), dtype=np.float64)
        for i in range(nact):
            va_mat[:, i] += 2.0 * qxr[nocc:, i]
            for j in range(nact):
                va_mat[:, i] += 2.0 * rdm1[j, i] * cfock[nocc:, nclosed + j]
        sigma.ax_plus_y_va(1.0, va_mat)

        # CA block
        if nclosed:
            ca_mat = np.zeros((nclosed, nact), dtype=np.float64)
            for i in range(nact):
                ca_mat[:, i] += 4.0 * cfock[:nclosed, nclosed + i]
                ca_mat[:, i] += 4.0 * afock[:nclosed, nclosed + i]
                ca_mat[:, i] -= 2.0 * qxr[:nclosed, i]
                for j in range(nact):
                    ca_mat[:, i] -= 2.0 * rdm1[j, i] * cfock[:nclosed, nclosed + j]
            sigma.ax_plus_y_ca(1.0, ca_mat)

        return sigma

    # ------------------------------------------------------------ denom
    def _compute_denom(
        self,
        cfock: np.ndarray,
        afock: np.ndarray,
    ) -> RotFile:
        """Compute the diagonal Hessian for preconditioning.

        Diagonal Hessian (preconditioner) for the augmented Hessian solver.
        The full the reference implementation version uses extensive DF half/full transforms for the
        exact 2e diagonal.  Here we implement the dominant 1e part from
        cassecond.cc:229-240, plus the leading 2e correction from the
        active-space integrals.  This is sufficient for convergence (the
        diagonal Hessian is only a preconditioner).
        """
        nclosed = self.nclosed
        nact = self.nact
        nvirt = self.nvirt
        nocc = self.nocc

        denom = RotFile(nclosed, nact, nvirt)
        rdm1 = self.rdm1_av

        fock = cfock + afock
        fcaa = cfock[nclosed:nocc, nclosed:nocc]
        fcd = fcaa @ rdm1  # (nact, nact)

        # CA block (cassecond.cc:232-234)
        for i in range(nact):
            for j in range(nclosed):
                val = (4.0 * fock[i + nclosed, i + nclosed]
                       - 4.0 * fock[j, j]
                       - 2.0 * fcd[i, i]
                       + 2.0 * cfock[j, j] * rdm1[i, i])
                denom.set_ele_ca(j, i, denom.ele_ca(j, i) + val)

        # VC block (cassecond.cc:235-237)
        for i in range(nclosed):
            for j in range(nvirt):
                val = 4.0 * fock[j + nocc, j + nocc] - 4.0 * fock[i, i]
                denom.set_ele_vc(j, i, denom.ele_vc(j, i) + val)

        # VA block (cassecond.cc:238-240)
        for i in range(nact):
            for j in range(nvirt):
                val = 2.0 * cfock[j + nocc, j + nocc] * rdm1[i, i] - 2.0 * fcd[i, i]
                denom.set_ele_va(j, i, denom.ele_va(j, i) + val)

        # 2e corrections from active-space ERIs and 2-RDM
        # These correspond to the remaining terms in cassecond.cc:258-313.
        # We approximate the most important terms using the active-active
        # integrals that are already available.

        rdm2 = self.rdm2_av
        eri = self.eri_act  # (nact, nact, nact, nact)

        # e2 term (cassecond.cc:276-282): for each active orbital i,
        #   e2 = -2 * sum_{t,x,y} eri[t,i,x,y] * rdm2[t,i,x,y]
        #        -2 * sum_{x,y} eri[i,i,x,y] * rdm2[i,i,x,y]  -- column i
        # mo2e is (nact, nact, nact, nact) = eri in active space.
        for i in range(nact):
            e2 = float(-2.0 * _to_np(
                _np_einsum("tuv,tuv->", eri[:, :, :, i], rdm2[:, :, :, i])
            ))
            for j in range(nvirt):
                denom.set_ele_va(j, i, denom.ele_va(j, i) + e2)
            for j in range(nclosed):
                denom.set_ele_ca(j, i, denom.ele_ca(j, i) + e2)

        # Additional 2e terms for the VC block using the DF factors
        if nclosed > 0:
            C = _to_np(self.coeff)
            C_core = C[:, :nclosed]
            C_virt = C[:, nocc:]
            # (ii|jj) diagonal for VC: half-transform closed, second-transform virtual
            half_c = _half_transform(self.B_ao, C_core)  # (nao, nclosed, naux)
            full_cv = _full_transform(half_c, C_virt)    # (nclosed, nvirt, naux)
            diag_cv = _form_4index_diagonal_from_full(full_cv)  # (nclosed, nvirt)
            # the reference implementation adds 12.0 * diag_cv.T to VC
            denom.ax_plus_y_vc(12.0, _to_np(diag_cv.T))

            # Exchange diagonal for VC: (ij|ij) type
            # full_cc[i,j,P] for closed-closed, then contract with B to get (ij|ij)
            full_cc = _full_transform(half_c, C_core)  # (nclosed, nclosed, naux)
            for i in range(nclosed):
                # the reference implementation: tmp = B contracted with full_cc diagonal column,
                # then vcoeff % tmp * vcoeff diagonal
                # This gives -(ij|ij) for the virt-closed exchange.
                # Reconstruct: tmp_ao[mu,nu] = sum_P B[mu,nu,P] * full_cc[i,i,P]
                tmp_ao = _np_einsum("mnP,P->mn", self.B_ao, full_cc[i, i, :])
                tmp0 = C_virt.T @ tmp_ao @ C_virt  # (nvirt, nvirt)
                vc_col = -4.0 * np.diag(tmp0)
                o = denom._off_vc()
                denom._data[o + i * nvirt : o + (i + 1) * nvirt] += vc_col

        # 2e terms for VA and CA from active half-transforms
        C_act = _to_np(self.coeff)[:, nclosed:nocc]
        C_virt = self.coeff[:, nocc:]
        half_a = _half_transform(self.B_ao, C_act)  # (nao, nact, naux)
        full_aa = _full_transform(half_a, C_act)    # (nact, nact, naux)

        # apply_2rdm to full_aa
        vgaa = _apply_2rdm(full_aa, rdm2)  # (nact, nact, naux)

        for i in range(nact):
            # Reconstruct AO matrix from diagonal slice of vgaa
            tmp_ao = _np_einsum("mnP,P->mn", self.B_ao, vgaa[i, i, :])
            tmp_v = _to_np(C_virt.T @ tmp_ao @ C_virt)
            o_va = denom._off_va()
            denom._data[o_va + i * nvirt : o_va + (i + 1) * nvirt] += 2.0 * np.diag(tmp_v)
            if nclosed > 0:
                C_core = self.coeff[:, :nclosed]
                tmp_c = _to_np(C_core.T @ tmp_ao @ C_core)
                o_ca = denom._off_ca()
                denom._data[o_ca + i * nclosed : o_ca + (i + 1) * nclosed] += 2.0 * np.diag(tmp_c)

        # rdmk term (cassecond.cc:284-290)
        rdm2_np = _to_np(rdm2)
        rdmk = np.zeros((nact * nact, nact), dtype=np.float64)
        for i in range(nact):
            for j in range(nact):
                for k in range(nact):
                    rdmk[k + nact * j, i] = rdm2_np[k, i, j, i] + rdm2_np[k, i, i, j]

        # VA contribution from rdmk (cassecond.cc:289-290)
        # form_4index_diagonal_part returns (b1*b1, b2) = (nact*nact, nvirt)
        full_av = _full_transform(half_a, C_virt)  # (nact, nvirt, naux)
        diag_part_av = _form_4index_diagonal_part(full_av)  # (nact*nact, nvirt)
        # rdmk.T @ diag_part_av = (nact, nvirt), then .T = (nvirt, nact)
        denom.ax_plus_y_va(2.0, _to_np((rdmk.T @ _to_np(diag_part_av)).T))

        if nclosed > 0:
            C_core = self.coeff[:, :nclosed]
            full_ac = _full_transform(half_a, C_core)  # (nact, nclosed, naux)
            diag_part_ac = _form_4index_diagonal_part(full_ac)  # (nact*nact, nclosed)
            # mcaa = transpose -> (nclosed, nact*nact)
            mcaa = diag_part_ac.T  # (nclosed, nact*nact)
            # ax_plus_y_ca(2.0, mcaa * rdmk) where mcaa (nclosed, nact*nact) @ rdmk (nact*nact, nact) => (nclosed, nact)
            denom.ax_plus_y_ca(2.0, _to_np(mcaa) @ rdmk)

            # mcaad = mcaa - mcaa_reshaped * rdm1 (cassecond.cc:295-298)
            mcaad = mcaa.copy()  # (nclosed, nact*nact)
            # dgemm: mcaad[nclosed*nact, nact] -= mcaa[nclosed*nact, nact] @ rdm1[nact, nact]
            mcaa_r = mcaa.reshape(nclosed * nact, nact)
            mcaad_r = mcaad.reshape(nclosed * nact, nact)
            mcaad_r[:] = mcaa_r - mcaa_r @ rdm1
            # For each active i: add 12.0 * mcaad[:, i + nact*i] to CA column i
            for i in range(nact):
                col_idx = i + nact * i  # diagonal of the (nact, nact) block for each closed
                col = mcaad[:, col_idx]  # (nclosed,)
                o_ca = denom._off_ca()
                denom._data[o_ca + i * nclosed : o_ca + (i + 1) * nclosed] += 12.0 * col

        # Remaining term (cassecond.cc:301-312): density-correction for CA
        if nclosed > 0:
            C_core = self.coeff[:, :nclosed]
            # vgaa_corr = rdm1-transform of full_aa minus full_aa
            full_aa_rdm = _np_einsum("ab,bjP->ajP", rdm1, full_aa)  # transform_occ1(rdm1)
            vgaa_corr = full_aa_rdm - full_aa  # (nact, nact, naux)
            for i in range(nact):
                tmp_ao = _np_einsum("mnP,P->mn", self.B_ao, vgaa_corr[i, i, :])
                tmp_c = C_core.T @ tmp_ao @ C_core
                o_ca = denom._off_ca()
                denom._data[o_ca + i * nclosed : o_ca + (i + 1) * nclosed] += 4.0 * np.diag(tmp_c)

        return denom

    # --------------------------------------------------------- apply_denom
    @staticmethod
    def _apply_denom(grad: RotFile, denom: RotFile, shift: float, scale: float) -> RotFile:
        """Apply diagonal preconditioner.  cassecond.cc:179-185."""
        out = grad.copy()
        for i in range(out.size):
            d = denom.data[i] * scale + shift
            if abs(d) > 1.0e-12:
                out.data[i] /= d
        return out

    # --------------------------------------------------------- hess_trial
    def _compute_hess_trial(
        self,
        trot: RotFile,
        cfock: np.ndarray,
        afock: np.ndarray,
        qxr: np.ndarray,
    ) -> RotFile:
        """Compute Hessian times trial vector.  cassecond.cc:319-436.

        This implements both the 1-electron and 2-electron terms.
        """
        nclosed = self.nclosed
        nact = self.nact
        nvirt = self.nvirt
        nocc = self.nocc

        sigma = trot.clone()

        va = trot.va_mat()  # (nvirt, nact)
        ca = trot.ca_mat() if nclosed else None  # (nclosed, nact)
        vc = trot.vc_mat() if nclosed else None  # (nvirt, nclosed)

        # Fock sub-blocks
        fcaa = cfock[nclosed:nocc, nclosed:nocc]
        faaa = afock[nclosed:nocc, nclosed:nocc]
        fcva = cfock[nocc:, nclosed:nocc]
        fava = afock[nocc:, nclosed:nocc]
        fcvv = cfock[nocc:, nocc:]
        favv = afock[nocc:, nocc:]
        if nclosed:
            fccc = cfock[:nclosed, :nclosed]
            facc = afock[:nclosed, :nclosed]
            fcca = cfock[:nclosed, nclosed:nocc]
            faca = afock[:nclosed, nclosed:nocc]
            fcvc = cfock[nocc:, :nclosed]
            favc = afock[nocc:, :nclosed]

        C = _to_np(self.coeff)
        C_core = C[:, :nclosed] if nclosed else None
        C_act = C[:, nclosed:nocc]
        C_virt = C[:, nocc:]
        rdm1 = self.rdm1_av
        rdm2 = self.rdm2_av

        # ---- 2-electron g(D) terms via DF ----
        # half-transforms of reference orbitals
        if nclosed:
            half_closed = _half_transform(self.B_ao, C_core)  # (nao, nclosed, naux)
        half_act = _half_transform(self.B_ao, C_act)   # (nao, nact, naux)

        # g(t_vc) and g(t_ca) -- cassecond.cc:357-366
        if nclosed:
            # tcoeff = vcoeff * vc + acoeff * ca.T  -- (nao, nclosed)
            tcoeff = C_virt @ vc + C_act @ ca.T
            half_t = _half_transform(self.B_ao, tcoeff)  # (nao, nclosed, naux)
            gt = _compute_gd_from_halfs(self.B_ao, half_t, half_closed, C_core)
            sigma.ax_plus_y_ca(32.0, C_core.T @ gt @ C_act)
            sigma.ax_plus_y_vc(32.0, C_virt.T @ gt @ C_core)
            sigma.ax_plus_y_va(16.0, C_virt.T @ gt @ C_act @ rdm1)
            sigma.ax_plus_y_ca(-16.0, C_core.T @ gt @ C_act @ rdm1)

        # g(t_va - t_ca) * rdm1 -- cassecond.cc:367-376
        if nclosed:
            tcoeff_a = C_virt @ va - C_core @ ca
        else:
            tcoeff_a = C_virt @ va
        if nclosed:
            # halfta = half-transform of tcoeff_a: (nao, nact, naux)
            halfta = _half_transform(self.B_ao, tcoeff_a)
            # halftad = halfta contracted with rdm1 over orbital index
            halftad = _np_einsum("miP,ij->mjP", halfta, rdm1)  # (nao, nact, naux)
            gt_a = _compute_gd_from_halfs(self.B_ao, halftad, half_act, C_act)
            sigma.ax_plus_y_ca(16.0, C_core.T @ gt_a @ C_act)
            sigma.ax_plus_y_vc(16.0, C_virt.T @ gt_a @ C_core)
        else:
            tcoeff_a = C_virt @ va

        # ---- Q-vector terms (cassecond.cc:377-391) ----
        qaa = qxr[nclosed:nocc, :]  # (nact, nact)
        # va ^ qaa = va @ qaa.T
        sigma.ax_plus_y_va(-2.0, va @ qaa.T)
        # va * qaa = va @ qaa
        sigma.ax_plus_y_va(-2.0, va @ qaa)
        if nclosed:
            qva = qxr[nocc:, :]     # (nvirt, nact)
            qca = qxr[:nclosed, :]  # (nclosed, nact)
            # va ^ qca = va @ qca.T
            sigma.ax_plus_y_vc(-2.0, va @ qca.T)
            # vc * qca = vc @ qca
            sigma.ax_plus_y_va(-2.0, vc @ qca)
            # vc % qva = vc.T @ qva
            sigma.ax_plus_y_ca(-2.0, vc.T @ qva)
            # qva ^ ca = qva @ ca.T
            sigma.ax_plus_y_vc(-2.0, qva @ ca.T)
            # ca ^ qaa = ca @ qaa.T
            sigma.ax_plus_y_ca(-2.0, ca @ qaa.T)
            # ca * qaa = ca @ qaa
            sigma.ax_plus_y_ca(-2.0, ca @ qaa)

        # ---- Q' and Q'' terms (cassecond.cc:393-407) ----
        # Reuse half_act from above; build full active-active transform
        full_aa = _full_transform(half_act, C_act)  # (nact, nact, naux)

        # halfta for tcoeff_a (may already exist from above, but recompute if needed)
        halfta_qp = _half_transform(self.B_ao, tcoeff_a)  # (nao, nact, naux)
        fullta = _full_transform(halfta_qp, C_act)        # (nact, nact, naux)
        # swap: fulltas[u, t, P] = fullta[t, u, P]
        fulltas = fullta.transpose(1, 0, 2).copy()        # (nact, nact, naux)
        fullta_sym = fullta + fulltas  # (nact, nact, naux)

        # Apply 2-RDM
        fullaaD = _apply_2rdm(full_aa, rdm2)   # (nact, nact, naux)
        fulltaD = _apply_2rdm(fullta_sym, rdm2)  # (nact, nact, naux)

        # qp = halfa.form_2index(fulltaD) = sum_{t,P} half_act[mu,t,P] * fulltaD[t,u,P]
        qp = _form_2index(half_act, fulltaD)   # (nao, nact)
        # qpp = halfta.form_2index(fullaaD)
        qpp = _form_2index(halfta_qp, fullaaD)  # (nao, nact)

        qpqpp = qp + qpp  # (nao, nact)
        # vcoeff % (qp + qpp) = C_virt.T @ (qp + qpp)
        sigma.ax_plus_y_va(4.0, C_virt.T @ qpqpp)
        if nclosed:
            sigma.ax_plus_y_ca(-4.0, C_core.T @ qpqpp)

        # ---- 1-electron contributions (cassecond.cc:410-432) ----
        sigma.ax_plus_y_va(4.0, fcvv @ va @ rdm1)
        sigma.ax_plus_y_va(-2.0, va @ (rdm1 @ fcaa + fcaa @ rdm1))
        if nclosed:
            sigma.ax_plus_y_ca(8.0, ca @ (fcaa + faaa))
            sigma.ax_plus_y_ca(8.0, vc.T @ (fcva + fava))
            sigma.ax_plus_y_vc(-8.0, vc @ (fccc + facc))
            sigma.ax_plus_y_va(-4.0, vc @ (fcca + faca))
            # va ^ (fcca + faca) = va @ (fcca + faca).T
            sigma.ax_plus_y_vc(-4.0, va @ (fcca + faca).T)
            sigma.ax_plus_y_ca(-2.0, ca @ (rdm1 @ fcaa + fcaa @ rdm1))
            sigma.ax_plus_y_vc(8.0, (fcvv + favv) @ vc)
            sigma.ax_plus_y_ca(-8.0, (fccc + facc) @ ca)
            sigma.ax_plus_y_va(4.0, (fcvc + favc) @ ca)
            # (fcvc + favc) % va = (fcvc + favc).T @ va
            sigma.ax_plus_y_ca(4.0, (fcvc + favc).T @ va)
            # (fcva + fava) ^ ca = (fcva + fava) @ ca.T
            sigma.ax_plus_y_vc(8.0, (fcva + fava) @ ca.T)
            sigma.ax_plus_y_ca(4.0, fccc @ ca @ rdm1)
            # fcvc % va = fcvc.T @ va
            sigma.ax_plus_y_ca(-4.0, fcvc.T @ va @ rdm1)
            sigma.ax_plus_y_va(-4.0, fcvc @ ca @ rdm1)
            # fcva * rdm1 ^ ca = (fcva @ rdm1) @ ca.T
            sigma.ax_plus_y_vc(-2.0, (fcva @ rdm1) @ ca.T)
            # va * rdm1 ^ fcca = (va @ rdm1) @ fcca.T
            sigma.ax_plus_y_vc(-2.0, (va @ rdm1) @ fcca.T)
            # vc % fcva * rdm1 = vc.T @ fcva @ rdm1
            sigma.ax_plus_y_ca(-2.0, vc.T @ fcva @ rdm1)
            sigma.ax_plus_y_va(-2.0, vc @ fcca @ rdm1)

        # Overall 0.5 factor (cassecond.cc:434)
        sigma.scale(0.5)
        return sigma

    # ================================================================ compute
    def compute(self) -> CASSecondResult:
        """Run the second-order CASSCF optimization.  cassecond.cc:35-176."""
        assert self.nvirt > 0 and self.nact > 0

        converged = False
        profile = {}

        for iteration in range(self.max_iter):
            # 1. CASCI + RDMs
            self._solve_casci()
            self._trans_natorb()

            # 2. Build Fock matrices
            cfock, afock = self._build_fock()

            # 3. Q-vector
            qxr = _to_np(build_qvec(
                self.nmo, self.nact, self.coeff, self.nclosed,
                self.B_ao, self.rdm2_av,
            ))

            # 4. Gradient
            grad = self._compute_gradient(cfock, afock, qxr)

            # 5. Check convergence
            gradient = grad.rms()
            e_str = "  ".join(f"{e:.10f}" for e in self.energy)
            self._log(
                f"  Iter {iteration:3d}   E = {e_str}   "
                f"|grad| = {gradient:.2e}"
            )
            if gradient < self.thresh:
                converged = True
                self._log("")
                self._log("    * Second-order optimization converged. *")
                self._log("")
                break

            # 6. Diagonal Hessian
            denom = self._compute_denom(cfock, afock)

            # 7. AugHess micro-iterations
            solver = AugHess(self.max_micro_iter, grad, maxstepsize=0.1)
            trot = self._apply_denom(grad, denom, 0.001, 1.0)
            trot.normalize()

            for miter in range(self.max_micro_iter):
                sigma_vec = self._compute_hess_trial(trot, cfock, afock, qxr)
                residual, lam, eps, stepsize = solver.compute_residual(trot, sigma_vec)
                err = residual.norm() / lam
                self._log(
                    f"         res : {err:8.2e}   "
                    f"lamb: {lam:8.2e}   "
                    f"eps : {eps:8.2e}   "
                    f"step: {stepsize:8.2e}"
                )
                if err < max(self.thresh_micro, stepsize * self.thresh_microstep):
                    break

                trot = self._apply_denom(residual, denom, -eps, 1.0 / lam)
                for _ in range(10):
                    n = solver.orthog(trot)
                    if n > 0.25:
                        break

            # 8. Matrix exponential rotation (cassecond.cc:114-128)
            sol = solver.civec()
            # Scale step if norm is too large
            step_norm = sol.norm()
            max_step = 0.15
            if step_norm > max_step:
                sol.scale(max_step / step_norm)

            A = sol.unpack()  # (nmo, nmo) antisymmetric
            W = A @ A          # W = A^2 (symmetric, negative semi-definite)
            eig_w, V = np.linalg.eigh(W)

            # Build cos and sinc of sqrt(|eig|)
            nmo = A.shape[0]
            wc = V.copy()  # will hold V * cos(tau)
            ws = V.copy()  # will hold V * sinc(tau)
            for i in range(nmo):
                tau = np.sqrt(abs(eig_w[i]))
                wc[:, i] *= np.cos(tau)
                ws[:, i] *= (np.sin(tau) / tau if tau > 1.0e-15 else 1.0)

            # R = wc @ V.T + ws @ V.T @ A  (matrix exponential exp(A))
            R = wc @ V.T + ws @ V.T @ A

            self.coeff = self.coeff @ _to_xp(R, self.xp)

            if iteration == self.max_iter - 1:
                self._log("")
                self._log("    * Max iteration reached.  Convergence not reached! *")
                self._log("")

        # Semi-canonical orbitals (block-diagonalize within closed and virtual)
        if self.max_iter > 0:
            mo_energy = self._semi_canonical_orb()
        else:
            mo_energy = None

        # Final CASCI with converged orbitals
        if self.nact > 0:
            self._solve_casci()

        return CASSecondResult(
            converged=converged,
            niter=iteration + 1 if 'iteration' in dir() else 0,
            e_tot=float(self.energy[0]) if self.energy is not None else 0.0,
            e_roots=self.energy.copy() if self.energy is not None else np.array([]),
            mo_coeff=self.coeff.copy(),
            mo_energy=mo_energy,
            ci=self.ci,
            ncore=self.ncore,
            ncas=self.ncas,
            nelecas=self.nelecas,
            nroots=self.nroots,
            profile=profile,
        )

    # -------------------------------------------------------- semi-canonical
    def _semi_canonical_orb(self) -> np.ndarray:
        """Block-diagonalize Fock within closed and virtual spaces."""
        cfock, afock = self._build_fock()
        fock = cfock + afock

        C = self.coeff.copy()
        nclosed = self.nclosed
        nocc = self.nocc
        nmo = self.nmo

        fock_np = _to_np(fock)
        mo_energy = np.diag(fock_np).copy()

        # Diagonalize within closed block
        if nclosed > 1:
            fcc = fock_np[:nclosed, :nclosed]
            eig_c, Uc = np.linalg.eigh(fcc)
            C[:, :nclosed] = C[:, :nclosed] @ _to_xp(Uc, self.xp)
            mo_energy[:nclosed] = eig_c

        # Diagonalize within virtual block
        nvirt = nmo - nocc
        if nvirt > 1:
            fvv = fock_np[nocc:, nocc:]
            eig_v, Uv = np.linalg.eigh(fvv)
            C[:, nocc:] = C[:, nocc:] @ _to_xp(Uv, self.xp)
            mo_energy[nocc:] = eig_v

        self.coeff = C
        return mo_energy
