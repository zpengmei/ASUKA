from __future__ import annotations

"""Dense-integral helpers for the Newton-CASSCF operator (gen_g_hop).

This module mirrors :mod:`asuka.mcscf.newton_df` but uses **exact dense 4-center**
integrals built directly from cuERI **CPU tiles** (no AO-ERI tensor materialization)
to construct the intermediates required by
:func:`asuka.mcscf.newton_casscf.gen_g_hop_internal`.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from asuka.mcscf.dense_eri_cpu import (
    DenseERI4cDerivContractionCPUCache,
    build_dense_eri4c_deriv_contraction_cache_cpu,
    dense_ppaa_papa_from_tiles_cpu,
    dense_vhf_ao_from_tiles_cpu,
)
from asuka.mcscf.orbital_grad import cayley_update


def _asnumpy_f64(a: Any) -> np.ndarray:
    """Ensure array is numpy.float64, converting from cupy if needed.

    Parameters
    ----------
    a : Any
        Input array (numpy or cupy).

    Returns
    -------
    np.ndarray
        Numpy array with float64 dtype.
    """
    try:
        import cupy as cp  # type: ignore
    except Exception:
        cp = None  # type: ignore
    if cp is not None and isinstance(a, cp.ndarray):  # type: ignore[attr-defined]
        a = cp.asnumpy(a)
    return np.asarray(a, dtype=np.float64)


@dataclass(frozen=True)
class DenseNewtonERIs:
    """Minimal eris-like container for `newton_casscf.gen_g_hop_internal` (dense 4c).

    Attributes
    ----------
    ppaa : np.ndarray
        Dense ERIs (nmo,nmo,ncas,ncas).
    papa : np.ndarray
        Dense ERIs (nmo,ncas,nmo,ncas).
    vhf_c : np.ndarray
        Core HF potential in MO basis (nmo,nmo).
    j_pc : np.ndarray
        Core Coulomb potential (nmo,ncore).
    k_pc : np.ndarray
        Core Exchange potential (nmo,ncore).
    """

    ppaa: np.ndarray
    papa: np.ndarray
    vhf_c: np.ndarray
    j_pc: np.ndarray
    k_pc: np.ndarray


def build_dense_newton_eris(
    ao_basis: Any,
    atom_coords_bohr: Any,
    mo_coeff: Any,
    *,
    ncore: int,
    ncas: int,
    cache_cpu: DenseERI4cDerivContractionCPUCache | None = None,
    pair_table_threads: int = 0,
    max_tile_bytes: int = 256 << 20,
    threads: int = 0,
) -> DenseNewtonERIs:
    """Build dense ERI intermediates required by the Newton-CASSCF operator.

    Parameters
    ----------
    ao_basis : Any
        AO basis set information.
    atom_coords_bohr : Any
        Atomic coordinates in Bohr.
    mo_coeff : Any
        Molecular orbital coefficients (nao, nmo).
    ncore : int
        Number of core orbitals.
    ncas : int
        Number of active orbitals.
    cache_cpu : DenseERI4cDerivContractionCPUCache | None, optional
        Precomputed cache for contractions.
    pair_table_threads : int, optional
        Number of threads for pair table construction.
    max_tile_bytes : int, optional
        Maximum memory per tile in bytes.
    threads : int, optional
        Number of OMP threads.

    Returns
    -------
    DenseNewtonERIs
        The constructed ERI container.
    """

    mo = _asnumpy_f64(mo_coeff)
    atom_coords_bohr = _asnumpy_f64(atom_coords_bohr)

    ncore = int(ncore)
    ncas = int(ncas)
    if ncore < 0:
        raise ValueError("ncore must be >= 0")
    if ncas <= 0:
        raise ValueError("ncas must be > 0")
    if mo.ndim != 2:
        raise ValueError("mo_coeff must have shape (nao,nmo)")
    nao, nmo = map(int, mo.shape)
    nocc = ncore + ncas
    if nocc > nmo:
        raise ValueError("ncore+ncas exceeds nmo")

    if cache_cpu is None:
        cache_cpu = build_dense_eri4c_deriv_contraction_cache_cpu(
            ao_basis,
            atom_coords_bohr=atom_coords_bohr,
            pair_table_threads=int(pair_table_threads),
        )

    ppaa, papa = dense_ppaa_papa_from_tiles_cpu(
        ao_basis,
        mo,
        ncore=int(ncore),
        ncas=int(ncas),
        atom_coords_bohr=atom_coords_bohr,
        cache_cpu=cache_cpu,
        pair_table_threads=int(pair_table_threads),
        max_tile_bytes=int(max_tile_bytes),
        threads=int(threads),
    )

    # vhf_c in MO basis from core density.
    if ncore:
        mo_core = mo[:, :ncore]
        D_core = 2.0 * (mo_core @ mo_core.T)
        v_ao = dense_vhf_ao_from_tiles_cpu(
            ao_basis,
            D_core,
            atom_coords_bohr=atom_coords_bohr,
            cache_cpu=cache_cpu,
            pair_table_threads=int(pair_table_threads),
            max_tile_bytes=int(max_tile_bytes),
            threads=int(threads),
        )
        vhf_c = np.asarray(mo.T @ v_ao @ mo, dtype=np.float64, order="C")
        j_pc = np.zeros((nmo, ncore), dtype=np.float64)
        k_pc = np.zeros((nmo, ncore), dtype=np.float64)
    else:
        vhf_c = np.zeros((nmo, nmo), dtype=np.float64)
        j_pc = np.zeros((nmo, 0), dtype=np.float64)
        k_pc = np.zeros((nmo, 0), dtype=np.float64)

    return DenseNewtonERIs(ppaa=ppaa, papa=papa, vhf_c=vhf_c, j_pc=j_pc, k_pc=k_pc)


@dataclass
class DenseNewtonCASSCFAdapter:
    """Minimal CASSCF-like adapter for `newton_casscf.gen_g_hop_internal` (dense 4c).

    Attributes
    ----------
    ao_basis : Any
        AO basis set information.
    atom_coords_bohr : np.ndarray
        Atomic coordinates in Bohr.
    hcore_ao : np.ndarray
        Core Hamiltonian in AO basis.
    ncore : int
        Number of core orbitals.
    ncas : int
        Number of active orbitals.
    nelecas : int | tuple[int, int]
        Number of active electrons.
    mo_coeff : np.ndarray
        MO coefficients.
    fcisolver : Any
        FCI solver object.
    eri_cache_cpu : DenseERI4cDerivContractionCPUCache | None
        Cache for ERI contractions.
    pair_table_threads : int
        Threads for pair table.
    max_tile_bytes : int
        Max tile size in bytes.
    eri_threads : int
        Threads for ERI calculation.
    weights : list[float] | None
        State weights for SA-CASSCF.
    frozen : Any | None
        Frozen orbitals.
    internal_rotation : bool
        Whether internal rotation (active-active) is redundant.
    extrasym : Any | None
        Symmetry constraints.
    """

    ao_basis: Any
    atom_coords_bohr: np.ndarray
    hcore_ao: np.ndarray
    ncore: int
    ncas: int
    nelecas: int | tuple[int, int]
    mo_coeff: np.ndarray
    fcisolver: Any
    eri_cache_cpu: DenseERI4cDerivContractionCPUCache | None = None
    pair_table_threads: int = 0
    max_tile_bytes: int = 256 << 20
    eri_threads: int = 0

    # Optional knobs (PySCF-compatible names)
    weights: list[float] | None = None
    frozen: Any | None = None
    internal_rotation: bool = False
    extrasym: Any | None = None

    def get_hcore(self) -> np.ndarray:
        """Return the core Hamiltonian in AO basis."""
        return np.asarray(self.hcore_ao, dtype=np.float64)

    def ao2mo(self, mo_coeff: Any) -> DenseNewtonERIs:
        """Construct the dense ERI object for the given MOs.

        Parameters
        ----------
        mo_coeff : Any
            Molecular orbital coefficients.

        Returns
        -------
        DenseNewtonERIs
            The dense ERI container.
        """
        return build_dense_newton_eris(
            self.ao_basis,
            np.asarray(self.atom_coords_bohr, dtype=np.float64),
            mo_coeff,
            ncore=int(self.ncore),
            ncas=int(self.ncas),
            cache_cpu=self.eri_cache_cpu,
            pair_table_threads=int(self.pair_table_threads),
            max_tile_bytes=int(self.max_tile_bytes),
            threads=int(self.eri_threads),
        )

    def uniq_var_indices(self, nmo: int, ncore: int, ncas: int, frozen: Any | None) -> np.ndarray:
        """Return boolean mask of independent orbital rotation parameters.

        Parameters
        ----------
        nmo : int
            Number of molecular orbitals.
        ncore : int
            Number of core orbitals.
        ncas : int
            Number of active orbitals.
        frozen : Any | None
            Frozen orbitals.

        Returns
        -------
        np.ndarray
            Boolean mask (nmo, nmo) where True elements are independent parameters.
        """
        nmo = int(nmo)
        ncore = int(ncore)
        ncas = int(ncas)
        nocc = ncore + ncas
        mask = np.zeros((nmo, nmo), dtype=bool)
        mask[ncore:nocc, :ncore] = True
        mask[nocc:, :nocc] = True
        if bool(self.internal_rotation):
            mask[ncore:nocc, ncore:nocc][np.tril_indices(ncas, -1)] = True
        if self.extrasym is not None:
            extrasym = np.asarray(self.extrasym)
            extrasym_allowed = extrasym.reshape(-1, 1) == extrasym
            mask = mask & extrasym_allowed
        if frozen is not None:
            if isinstance(frozen, (int, np.integer)):
                mask[: int(frozen)] = False
                mask[:, : int(frozen)] = False
            else:
                frozen_idx = np.asarray(frozen, dtype=np.int32).ravel()
                mask[frozen_idx] = False
                mask[:, frozen_idx] = False
        return mask

    def pack_uniq_var(self, mat: Any) -> np.ndarray:
        """Pack a full anti-symmetric matrix into a flat independent-parameter vector.

        Parameters
        ----------
        mat : Any
            The full matrix.

        Returns
        -------
        np.ndarray
            Flattened vector of independent parameters.
        """
        mat = _asnumpy_f64(mat)
        nmo = int(np.asarray(self.mo_coeff).shape[1])
        idx = self.uniq_var_indices(nmo, int(self.ncore), int(self.ncas), self.frozen)
        return np.asarray(mat[idx], dtype=np.float64)

    def unpack_uniq_var(self, v: Any) -> np.ndarray:
        """Unpack a flat independent-parameter vector into a full anti-symmetric matrix.

        Parameters
        ----------
        v : Any
            The flattened vector.

        Returns
        -------
        np.ndarray
            The full anti-symmetric matrix (nmo, nmo).
        """
        v = _asnumpy_f64(v).ravel()
        nmo = int(np.asarray(self.mo_coeff).shape[1])
        idx = self.uniq_var_indices(nmo, int(self.ncore), int(self.ncas), self.frozen)
        mat = np.zeros((nmo, nmo), dtype=np.float64)
        mat[idx] = v
        return mat - mat.T

    def update_rotate_matrix(self, dx: Any, u0: Any = 1) -> np.ndarray:
        """Apply orbital rotation `dx` to `u0`.

        Parameters
        ----------
        dx : Any
            Parameter update vector (packed).
        u0 : Any, optional
            Current rotation matrix. Defaults to 1.

        Returns
        -------
        np.ndarray
            Updated rotation matrix.
        """
        dr = self.unpack_uniq_var(dx)
        u = cayley_update(np, dr)
        return np.dot(u0, np.asarray(u, dtype=np.float64))

    def update_jk_in_ah(
        self,
        mo: Any,
        r: Any,
        casdm1: Any,
        eris: Any | None = None,
        *,
        return_gpu: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Dense analogue of PySCF `mc1step.CASSCF.update_jk_in_ah`.

        Parameters
        ----------
        mo : Any
            Molecular orbitals.
        r : Any
            Orbital rotation matrix (anti-symmetric).
        casdm1 : Any
            Active space density matrix.
        eris : Any | None, optional
            Integral object (unused, for compatibility).

        Returns
        -------
        tuple
            (va, vc) where va is active-space potential and vc is core potential update.
        """

        _ = eris  # unused (kept for PySCF signature compatibility)

        ncore = int(self.ncore)
        ncas = int(self.ncas)
        nocc = ncore + ncas

        mo = _asnumpy_f64(mo)
        r = _asnumpy_f64(r)
        casdm1 = _asnumpy_f64(casdm1)

        if mo.ndim != 2:
            raise ValueError("mo must be 2D (nao,nmo)")
        nao, nmo = map(int, mo.shape)
        if nocc > nmo:
            raise ValueError("ncore+ncas exceeds nmo")
        if r.shape != (nmo, nmo):
            raise ValueError("r must be (nmo,nmo)")
        if casdm1.shape != (ncas, ncas):
            raise ValueError("casdm1 must be (ncas,ncas)")

        # dm3 = mo_core @ r_core,rest @ mo_rest^T  (+ sym)
        dm3 = mo[:, :ncore] @ r[:ncore, ncore:] @ mo[:, ncore:].T
        dm3 = dm3 + dm3.T

        # dm4 = mo_act @ casdm1 @ r_act,all @ mo^T (+ sym)
        dm4 = mo[:, ncore:nocc] @ casdm1 @ r[ncore:nocc] @ mo.T
        dm4 = dm4 + dm4.T

        cache_cpu = self.eri_cache_cpu
        if cache_cpu is None:
            cache_cpu = build_dense_eri4c_deriv_contraction_cache_cpu(
                self.ao_basis,
                atom_coords_bohr=np.asarray(self.atom_coords_bohr, dtype=np.float64),
                pair_table_threads=int(self.pair_table_threads),
            )

        # 2J - K = 2 * (J - 0.5 K) = 2 * vhf
        v0 = 2.0 * dense_vhf_ao_from_tiles_cpu(
            self.ao_basis,
            dm3,
            atom_coords_bohr=np.asarray(self.atom_coords_bohr, dtype=np.float64),
            cache_cpu=cache_cpu,
            pair_table_threads=int(self.pair_table_threads),
            max_tile_bytes=int(self.max_tile_bytes),
            threads=int(self.eri_threads),
        )
        v1 = 2.0 * dense_vhf_ao_from_tiles_cpu(
            self.ao_basis,
            dm3 * 2.0 + dm4,
            atom_coords_bohr=np.asarray(self.atom_coords_bohr, dtype=np.float64),
            cache_cpu=cache_cpu,
            pair_table_threads=int(self.pair_table_threads),
            max_tile_bytes=int(self.max_tile_bytes),
            threads=int(self.eri_threads),
        )

        mo_act = mo[:, ncore:nocc]
        mo_core = mo[:, :ncore]

        va = casdm1 @ mo_act.T @ v0 @ mo
        vc = mo_core.T @ v1 @ mo[:, ncore:]

        return np.asarray(va, dtype=np.float64, order="C"), np.asarray(vc, dtype=np.float64, order="C")


__all__ = [
    "DenseNewtonERIs",
    "DenseNewtonCASSCFAdapter",
    "build_dense_newton_eris",
]
