from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class AMFIAtomicOccupation:
    """OpenMolcas-style atomic occupations for AMFI mean-field.

    This matches Molcas' default `getocc_ao` behavior at the level of
    - number of fully occupied subshells (per angular momentum l)
    - number of electrons in the partially occupied subshell (per l)
    """

    closed_shells: tuple[int, int, int, int]  # (s,p,d,f): number of full subshells
    open_electrons: tuple[int, int, int, int]  # (s,p,d,f): electrons in the next (open) subshell


def _amfi_atomic_occupation_from_pyscf(Z: int) -> AMFIAtomicOccupation:
    """Return Molcas-style (closed shells, open electrons) for a neutral atom.

    Notes
    -----
    OpenMolcas' AMFI uses a built-in electron configuration table (`getocc_ao`).
    We use PySCF's tabulated ground-state configuration (`pyscf.data.elements.CONFIGURATION`)
    which provides the *total* number of electrons in s/p/d/f shells. From that, we infer:

        closed_shells[l]  = n_electrons[l] // cap(l)
        open_electrons[l] = n_electrons[l] %  cap(l)

    For hydrogen (Z=1), OpenMolcas explicitly disables mean-field ("H: no mean-field"), so we
    return zero occupations.
    """

    Z = int(Z)
    if Z < 0:
        raise ValueError("Z must be non-negative")
    if Z in (0, 1):
        return AMFIAtomicOccupation(closed_shells=(0, 0, 0, 0), open_electrons=(0, 0, 0, 0))

    try:
        from pyscf.data import elements  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PySCF is required for internal AMFI occupations") from e

    try:
        cfg = elements.CONFIGURATION[Z]
    except Exception as e:  # pragma: no cover
        raise ValueError(f"no electron configuration available for Z={Z}") from e

    if len(cfg) < 4:
        raise ValueError(f"invalid CONFIGURATION entry for Z={Z}: {cfg!r}")

    caps = (2, 6, 10, 14)
    closed = tuple(int(cfg[i] // caps[i]) for i in range(4))
    open_e = tuple(int(cfg[i] % caps[i]) for i in range(4))
    return AMFIAtomicOccupation(closed_shells=closed, open_electrons=open_e)


def _build_amfi_atomic_density_matrix(
    mol,  # pyscf.gto.Mole-like
    *,
    occupation: AMFIAtomicOccupation,
) -> np.ndarray:
    """Build the OpenMolcas-style diagonal AO density for AMFI mean-field."""

    # AMFI is defined in a real spherical harmonic basis; for systems without l>=2 this
    # distinction is irrelevant and some Molden loaders may set `mol.cart=True` by default.
    if bool(getattr(mol, "cart", False)):
        max_l = max(int(mol.bas_angular(i)) for i in range(int(mol.nbas))) if int(mol.nbas) else 0
        if max_l >= 2:
            raise ValueError("internal AMFI density builder does not support Cartesian AOs with l>=2 (mol.cart=True)")

    ao_loc = np.asarray(mol.ao_loc_nr(), dtype=np.int64)
    nao = int(mol.nao_nr())
    D = np.zeros((nao, nao), dtype=np.float64)

    # Track "radial contraction index" per l across shells, matching PySCF AO ordering:
    # for each shell (l, nctr), each contraction contributes (2l+1) AOs grouped together.
    contr_count: dict[int, int] = {}

    closed = occupation.closed_shells
    open_e = occupation.open_electrons

    for ib in range(int(mol.nbas)):
        l = int(mol.bas_angular(ib))
        nctr = int(mol.bas_nctr(ib))
        nfunc = 2 * l + 1
        for ic in range(nctr):
            contr_count[l] = int(contr_count.get(l, 0)) + 1
            ridx = int(contr_count[l])  # 1-based radial index within this l block

            occ = 0.0
            if 0 <= l < 4:
                if ridx <= int(closed[l]):
                    occ = 2.0
                elif ridx == int(closed[l]) + 1 and int(open_e[l]) > 0:
                    occ = float(open_e[l]) / float(nfunc)

            if occ == 0.0:
                continue

            start = int(ao_loc[ib] + ic * nfunc)
            stop = int(start + nfunc)
            D[start:stop, start:stop] = np.eye(nfunc, dtype=np.float64) * occ

    return D


def build_openmolcas_amfi_xyz_ao(
    mol,  # pyscf.gto.Mole-like
    *,
    scale: float = 2.0,
    include_mean_field: bool = True,
    atoms: Sequence[int] | None = None,
) -> np.ndarray:
    """Build OpenMolcas-style one-center AMFI integrals in the AO basis.

    Parameters
    ----------
    mol
        A PySCF `gto.Mole` (or compatible) object.
    scale
        Spin prefactor convention. `scale=2.0` matches OpenMolcas RASSI, which multiplies
        AMFI (stored for Pauli matrices) by 2 to use spin operators.
    include_mean_field
        If True, include the 2e spin-orbit mean-field (AMFI/SOMF) contribution.
        If False, return only the 1e nuclear spin-orbit term in the same convention.
    atoms
        Optional list of atom indices to include. If None, include all atoms.

    Returns
    -------
    h_xyz_ao
        Array with shape (3,nao,nao) containing (x,y,z) Cartesian components as real
        antisymmetric matrices (Molcas `ANTITRIP` convention).

    Notes
    -----
    This builder is intended as a **forward-pass parity** tool against OpenMolcas `*.OneInt`.
    It reproduces the one-center AMFI operator by:

    - building an *atomic* AO density from tabulated electron configurations
    - contracting PySCF's 2e SOC integrals (`int2e_p1vxp1`) into an effective 1e operator
    - adding the 1e nuclear SOC integrals (`int1e_pnucxp`)

    The final prefactor is `-scale/(4*c^2)`, consistent with Molcas' Pauli/spin convention.
    """

    if bool(getattr(mol, "cart", False)):
        max_l = max(int(mol.bas_angular(i)) for i in range(int(mol.nbas))) if int(mol.nbas) else 0
        if max_l >= 2:
            raise ValueError("OpenMolcas AMFI uses real spherical harmonics; set mol.cart=False")

    try:
        from pyscf.data import nist  # noqa: PLC0415
        from pyscf import gto  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PySCF is required for internal AMFI integral generation") from e

    nist_c = float(nist.LIGHT_SPEED)
    prefactor = -float(scale) / (4.0 * nist_c * nist_c)

    nao = int(mol.nao_nr())
    h_xyz = np.zeros((3, nao, nao), dtype=np.float64)

    ao_slices = np.asarray(mol.aoslice_by_atom(), dtype=np.int64)
    if ao_slices.shape[0] != int(mol.natm):
        raise ValueError("unexpected aoslice_by_atom shape")

    if atoms is None:
        atoms = list(range(int(mol.natm)))
    atoms = [int(i) for i in atoms]

    # Prefer per-atom basis blocks from molden-loaded objects (keys like "Li1"), falling back
    # to element keys (like "Li") for regular PySCF molecules.
    basis_store = getattr(mol, "_basis", None)

    for ia in atoms:
        if ia < 0 or ia >= int(mol.natm):
            raise ValueError(f"atom index out of range: {ia}")

        # Embed one-center integrals into the global AO block for this atom.
        s0 = int(ao_slices[ia, 2])
        s1 = int(ao_slices[ia, 3])
        if s1 <= s0:
            continue

        atom_key = str(mol.atom_symbol(ia))
        atom_sym = str(getattr(mol, "atom_pure_symbol", mol.atom_symbol)(ia))
        coord_bohr = np.asarray(mol.atom_coord(ia), dtype=np.float64)
        Z = int(mol.atom_charge(ia))

        basis = None
        if isinstance(basis_store, dict):
            basis = basis_store.get(atom_key)
            if basis is None:
                basis = basis_store.get(atom_sym)
        if basis is None:
            raise ValueError(f"failed to locate basis for atom {ia} ({atom_key}) in mol._basis")

        # PySCF requires a consistent (nelec, spin) pair even though integrals don't depend on it.
        spin = int(Z % 2)
        mol_a = gto.M(
            atom=[(atom_sym, coord_bohr)],
            basis=basis,
            unit="Bohr",
            cart=bool(getattr(mol, "cart", False)),
            spin=spin,
        )
        mol_a.build()

        nao_a = int(mol_a.nao_nr())
        if (s1 - s0) != nao_a:
            raise ValueError(
                "AO block size mismatch between global molecule and per-atom molecule: "
                f"atom={ia}, global={s1 - s0}, local={nao_a}"
            )

        h1 = np.asarray(mol_a.intor("int1e_pnucxp", comp=3), dtype=np.float64)
        if h1.shape != (3, nao_a, nao_a):
            raise ValueError("unexpected int1e_pnucxp shape")

        h_mf = 0.0
        if include_mean_field and Z not in (0, 1):
            occ = _amfi_atomic_occupation_from_pyscf(Z)
            D = _build_amfi_atomic_density_matrix(mol_a, occupation=occ)
            if float(np.linalg.norm(D)) != 0.0:
                g2 = np.asarray(mol_a.intor("int2e_p1vxp1", comp=3), dtype=np.float64)
                if g2.shape != (3, nao_a, nao_a, nao_a, nao_a):
                    raise ValueError("unexpected int2e_p1vxp1 shape")

                J = np.einsum("xijkl,lk->xij", g2, D, optimize=True)
                K1 = np.einsum("xikjl,lk->xij", g2, D, optimize=True)
                K2 = np.einsum("xjkil,lk->xij", g2, D, optimize=True)
                h_mf = 0.5 * (2.0 * J - 3.0 * K1 + 3.0 * K2)

        h_loc = prefactor * (h1 + h_mf)
        h_xyz[:, s0:s1, s0:s1] += np.asarray(h_loc, dtype=np.float64)

    # AMFI is stored/used as a real antisymmetric triplet operator.
    h_asym = h_xyz + np.swapaxes(h_xyz, -1, -2)
    if float(np.max(np.abs(h_asym))) > 1e-10:
        # Keep the functional behavior, but make the violation visible for debugging.
        raise RuntimeError("internal AMFI builder produced a non-antisymmetric operator (convention mismatch)")

    return h_xyz


def openmolcas_amfi_antitrip_xyz_ao_to_hso_xyz_ao(
    h_xyz_ao: np.ndarray,
    *,
    rme_scale: float = 3.0,
    phase: complex = 1j,
) -> np.ndarray:
    """Convert OpenMolcas AMFI (ANTITRIP) integrals into the SOC Hamiltonian convention used by cuGUGA.

    OpenMolcas stores AMFI as **real antisymmetric** Cartesian components (x,y,z) of an `ANTITRIP`
    operator and constructs a Hermitian SOC Hamiltonian by multiplying by the imaginary unit in the
    SOEIG assembly (see `OpenMolcas/src/rassi/soeig.f`).

    cuGUGA's SOC contractions assume the provided AO integrals represent the *Hermitian* one-electron
    SOC operator in the AO basis. Therefore, the OpenMolcas `ANTITRIP` AMFI matrices must be mapped
    to an imaginary-antisymmetric (Hermitian) form via a global phase factor (default `phase=+1j`).

    In addition, cuGUGA's current triplet TRDM/RME normalization differs from OpenMolcas' WTDM
    convention by an overall factor of 3; applying `rme_scale=3.0` yields SOC-RASSI eigenvalue parity
    for the forward pass tests.
    """

    h = np.asarray(h_xyz_ao)
    if h.ndim != 3 or int(h.shape[0]) != 3 or int(h.shape[1]) != int(h.shape[2]):
        raise ValueError("h_xyz_ao must have shape (3,nao,nao)")

    return complex(phase) * float(rme_scale) * np.asarray(h, dtype=np.complex128)
