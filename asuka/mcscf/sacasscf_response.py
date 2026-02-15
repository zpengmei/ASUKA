from __future__ import annotations

"""Internal SA-CASSCF response gradient kernels.

This module ports the orbital/CI Lagrange-response nuclear-gradient terms used
by SA-CASSCF analytic derivatives, without importing PySCF at runtime.
"""

from functools import reduce
from typing import Any, Iterator, Sequence

import numpy as np

from asuka.cueri.ao2mo import nr_e2_half_transform


def pack_tril(a: np.ndarray) -> np.ndarray:
    """Pack lower-triangular matrix indices along the last two dimensions.

    Parameters
    ----------
    a : np.ndarray
        Input array (..., n, n).

    Returns
    -------
    np.ndarray
        Packed array (..., n*(n+1)/2).
    """

    arr = np.asarray(a)
    if arr.ndim < 2 or arr.shape[-1] != arr.shape[-2]:
        raise ValueError("pack_tril expects (..., n, n) input")
    n = int(arr.shape[-1])
    tril_i, tril_j = np.tril_indices(n)
    return np.asarray(arr[..., tril_i, tril_j], dtype=arr.dtype)


def shell_prange(mol: Any, shl0: int, shl1: int, blksize: int) -> Iterator[tuple[int, int, int]]:
    """Yield shell blocks `(b0, b1, nf)` similar to PySCF `_shell_prange`.

    Parameters
    ----------
    mol : Any
        Molecule object.
    shl0 : int
        Start shell index.
    shl1 : int
        End shell index.
    blksize : int
        Block size target.

    Yields
    ------
    tuple[int, int, int]
        (start_shell, end_shell, number_of_functions)
    """

    b0 = int(shl0)
    b1_lim = int(shl1)
    blk = max(1, int(blksize))
    ao_loc = np.asarray(mol.ao_loc_nr(), dtype=np.int64)
    while b0 < b1_lim:
        b1 = b0 + 1
        while b1 < b1_lim:
            nf_try = int(ao_loc[b1 + 1] - ao_loc[b0])
            if nf_try > blk:
                break
            b1 += 1
        nf = int(ao_loc[b1] - ao_loc[b0])
        yield b0, b1, nf
        b0 = b1


def Lorb_dot_dgorb_dx(
    Lorb: np.ndarray,
    mc: Any,
    mo_coeff: np.ndarray | None = None,
    ci: Any | None = None,
    atmlst: Sequence[int] | None = None,
    mf_grad: Any | None = None,
    eris: Any | None = None,
    verbose: int | None = None,
) -> np.ndarray:
    """Orbital Lagrange term nuclear gradient contribution.

    Parameters
    ----------
    Lorb : np.ndarray
        Orbital Lagrange multipliers.
    mc : Any
        CASSCF object.
    mo_coeff : np.ndarray | None, optional
        MO coefficients.
    ci : Any | None, optional
        CI vector.
    atmlst : Sequence[int] | None, optional
        Atom list.
    mf_grad : Any | None, optional
        Mean-field gradient object.
    eris : Any | None, optional
        Integral object.
    verbose : int | None, optional
        Verbosity.

    Returns
    -------
    np.ndarray
        Gradient contribution (natm, 3).
    """

    del verbose
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if ci is None:
        ci = mc.ci
    if mf_grad is None:
        mf_grad = mc._scf.nuc_grad_method()
    if eris is None and hasattr(mc, "ao2mo"):
        eris = mc.ao2mo(mo_coeff)
    if eris is None:
        raise ValueError("eris is required for Lorb_dot_dgorb_dx")
    if getattr(mc, "frozen", None) is not None:
        raise NotImplementedError("frozen orbitals are not supported")

    mol = mc.mol
    ncore = int(mc.ncore)
    ncas = int(mc.ncas)
    nocc = ncore + ncas
    nelecas = mc.nelecas
    nao, nmo = np.asarray(mo_coeff).shape
    nao_pair = nao * (nao + 1) // 2

    mo_core = np.asarray(mo_coeff[:, :ncore], dtype=np.float64)
    mo_cas = np.asarray(mo_coeff[:, ncore:nocc], dtype=np.float64)
    moL_coeff = np.asarray(mo_coeff @ np.asarray(Lorb, dtype=np.float64), dtype=np.float64)
    s0_inv = np.asarray(mo_coeff @ mo_coeff.T, dtype=np.float64)
    moL_core = moL_coeff[:, :ncore]
    moL_cas = moL_coeff[:, ncore:nocc]

    casdm1, casdm2 = mc.fcisolver.make_rdm12(ci, ncas, nelecas)
    casdm1 = np.asarray(casdm1, dtype=np.float64)
    casdm2 = np.asarray(casdm2, dtype=np.float64)

    dm_core = mo_core @ mo_core.T * 2.0
    dm_cas = reduce(np.dot, (mo_cas, casdm1, mo_cas.T))
    dmL_core = moL_core @ mo_core.T * 2.0
    dmL_cas = reduce(np.dot, (moL_cas, casdm1, mo_cas.T))
    dmL_core = dmL_core + dmL_core.T
    dmL_cas = dmL_cas + dmL_cas.T
    dm1 = dm_core + dm_cas
    dm1L = dmL_core + dmL_cas

    aapa = np.zeros((ncas, ncas, nmo, ncas), dtype=np.float64)
    aapaL = np.zeros((ncas, ncas, nmo, ncas), dtype=np.float64)
    for i in range(nmo):
        jbuf = np.asarray(eris.ppaa[i], dtype=np.float64)
        kbuf = np.asarray(eris.papa[i], dtype=np.float64)
        aapa[:, :, i, :] = jbuf[ncore:nocc, :, :].transpose(1, 2, 0)
        aapaL[:, :, i, :] += np.tensordot(jbuf, Lorb[:, ncore:nocc], axes=((0), (0)))
        kbuf = np.tensordot(kbuf, Lorb[:, ncore:nocc], axes=((1), (0))).transpose(1, 2, 0)
        aapaL[:, :, i, :] += kbuf + kbuf.transpose(1, 0, 2)

    vj, vk = mc._scf.get_jk(mol, (dm_core, dm_cas))
    vjL, vkL = mc._scf.get_jk(mol, (dmL_core, dmL_cas))
    h1 = np.asarray(mc.get_hcore(), dtype=np.float64)
    vhf_c = np.asarray(vj[0] - vk[0] * 0.5, dtype=np.float64)
    vhf_a = np.asarray(vj[1] - vk[1] * 0.5, dtype=np.float64)
    vhfL_c = np.asarray(vjL[0] - vkL[0] * 0.5, dtype=np.float64)
    vhfL_a = np.asarray(vjL[1] - vkL[1] * 0.5, dtype=np.float64)

    gfock = h1 @ dm1L
    gfock += (vhf_c + vhf_a) @ dmL_core
    gfock += (vhfL_c + vhfL_a) @ dm_core
    gfock += vhfL_c @ dm_cas
    gfock += vhf_c @ dmL_cas
    gfock = s0_inv @ gfock
    gfock += reduce(np.dot, (mo_coeff, np.einsum("uviw,uvtw->it", aapaL, casdm2), mo_cas.T))
    gfock += reduce(np.dot, (mo_coeff, np.einsum("uviw,vuwt->it", aapa, casdm2), moL_cas.T))
    dme0 = (gfock + gfock.T) * 0.5

    vj, vk = mf_grad.get_jk(mol, (dm_core, dm_cas, dmL_core, dmL_cas))
    vhf1c, vhf1a, vhf1cL, vhf1aL = vj - vk * 0.5
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    diag_idx = np.arange(nao, dtype=np.int64)
    diag_idx = diag_idx * (diag_idx + 1) // 2 + diag_idx
    casdm2_cc = casdm2 + casdm2.transpose(0, 1, 3, 2)
    dm2buf = nr_e2_half_transform(
        casdm2_cc.reshape(ncas * ncas, ncas * ncas), mo_cas.T, (0, nao, 0, nao)
    ).reshape(ncas * ncas, nao, nao)

    dm2Lbuf = np.zeros((ncas * ncas, nmo, nmo), dtype=np.float64)
    Lcasdm2 = np.tensordot(Lorb[:, ncore:nocc], casdm2, axes=(1, 2)).transpose(1, 2, 0, 3)
    dm2Lbuf[:, :, ncore:nocc] = Lcasdm2.reshape(ncas * ncas, nmo, ncas)
    Lcasdm2 = np.tensordot(Lorb[:, ncore:nocc], casdm2, axes=(1, 3)).transpose(1, 2, 3, 0)
    dm2Lbuf[:, ncore:nocc, :] += Lcasdm2.reshape(ncas * ncas, ncas, nmo)
    dm2Lbuf += dm2Lbuf.transpose(0, 2, 1)
    dm2Lbuf = nr_e2_half_transform(
        np.ascontiguousarray(dm2Lbuf).reshape(ncas * ncas, nmo * nmo),
        np.asarray(mo_coeff, dtype=np.float64).T,
        (0, nao, 0, nao),
    ).reshape(ncas * ncas, nao, nao)

    dm2buf = pack_tril(dm2buf)
    dm2buf[:, diag_idx] *= 0.5
    dm2buf = dm2buf.reshape(ncas, ncas, nao_pair)
    dm2Lbuf = pack_tril(dm2Lbuf)
    dm2Lbuf[:, diag_idx] *= 0.5
    dm2Lbuf = dm2Lbuf.reshape(ncas, ncas, nao_pair)

    if atmlst is None:
        atmlst = list(range(mol.natm))
    aoslices = mol.aoslice_by_atom()
    de_hcore = np.zeros((len(atmlst), 3), dtype=np.float64)
    de_renorm = np.zeros((len(atmlst), 3), dtype=np.float64)
    de_eri = np.zeros((len(atmlst), 3), dtype=np.float64)

    max_memory = float(getattr(mc, "max_memory", 2000))
    max_p = int((aoslices[:, 3] - aoslices[:, 2]).max())
    blksize = int(max_memory * 0.9e6 / 8 / (4 * max(1, max_p) * nao_pair))
    blksize = min(nao, max(2, blksize))

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[int(ia)]
        h1ao = hcore_deriv(int(ia))
        de_hcore[k] += np.einsum("xij,ij->x", h1ao, dm1L)
        de_renorm[k] -= np.einsum("xij,ij->x", s1[:, p0:p1], dme0[p0:p1]) * 2.0

        q1 = 0
        for b0, b1, nf in shell_prange(mol, 0, mol.nbas, blksize):
            q0, q1 = q1, q1 + nf
            dm2_ao = np.einsum("ijw,pi,qj->pqw", dm2Lbuf, mo_cas[p0:p1], mo_cas[q0:q1])
            dm2_ao += np.einsum("ijw,pi,qj->pqw", dm2buf, moL_cas[p0:p1], mo_cas[q0:q1])
            dm2_ao += np.einsum("ijw,pi,qj->pqw", dm2buf, mo_cas[p0:p1], moL_cas[q0:q1])
            shls_slice = (shl0, shl1, b0, b1, 0, mol.nbas, 0, mol.nbas)
            eri1 = mol.intor("int2e_ip1", comp=3, aosym="s2kl", shls_slice=shls_slice).reshape(
                3, p1 - p0, nf, nao_pair
            )
            de_eri[k] -= np.einsum("xijw,ijw->x", eri1, dm2_ao) * 2.0

        de_eri[k] += np.einsum("xij,ij->x", vhf1c[:, p0:p1], dm1L[p0:p1]) * 2.0
        de_eri[k] += np.einsum("xij,ij->x", vhf1cL[:, p0:p1], dm1[p0:p1]) * 2.0
        de_eri[k] += np.einsum("xij,ij->x", vhf1a[:, p0:p1], dmL_core[p0:p1]) * 2.0
        de_eri[k] += np.einsum("xij,ij->x", vhf1aL[:, p0:p1], dm_core[p0:p1]) * 2.0

    return de_hcore + de_renorm + de_eri


def Lci_dot_dgci_dx(
    Lci: Any,
    weights: Sequence[float] | np.ndarray,
    mc: Any,
    mo_coeff: np.ndarray | None = None,
    ci: Any | None = None,
    atmlst: Sequence[int] | None = None,
    mf_grad: Any | None = None,
    eris: Any | None = None,
    verbose: int | None = None,
) -> np.ndarray:
    """CI Lagrange term nuclear gradient contribution.

    Parameters
    ----------
    Lci : Any
        CI Lagrange multipliers.
    weights : Sequence[float] | np.ndarray
        State weights.
    mc : Any
        CASSCF object.
    mo_coeff : np.ndarray | None, optional
        MO coefficients.
    ci : Any | None, optional
        CI vector.
    atmlst : Sequence[int] | None, optional
        Atom list.
    mf_grad : Any | None, optional
        Mean-field gradient object.
    eris : Any | None, optional
        Integral object.
    verbose : int | None, optional
        Verbosity.

    Returns
    -------
    np.ndarray
        Gradient contribution (natm, 3).
    """

    del weights, verbose
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if ci is None:
        ci = mc.ci
    if mf_grad is None:
        mf_grad = mc._scf.nuc_grad_method()
    if eris is None and hasattr(mc, "ao2mo"):
        eris = mc.ao2mo(mo_coeff)
    if eris is None:
        raise ValueError("eris is required for Lci_dot_dgci_dx")
    if getattr(mc, "frozen", None) is not None:
        raise NotImplementedError("frozen orbitals are not supported")

    mol = mc.mol
    ncore = int(mc.ncore)
    ncas = int(mc.ncas)
    nocc = ncore + ncas
    nelecas = mc.nelecas
    nao, nmo = np.asarray(mo_coeff).shape
    nao_pair = nao * (nao + 1) // 2

    mo_occ = np.asarray(mo_coeff[:, :nocc], dtype=np.float64)
    mo_core = np.asarray(mo_coeff[:, :ncore], dtype=np.float64)
    mo_cas = np.asarray(mo_coeff[:, ncore:nocc], dtype=np.float64)

    casdm1, casdm2 = mc.fcisolver.trans_rdm12(Lci, ci, ncas, nelecas)
    casdm1 = np.asarray(casdm1, dtype=np.float64)
    casdm2 = np.asarray(casdm2, dtype=np.float64)
    casdm1 += casdm1.transpose(1, 0)
    casdm2 += casdm2.transpose(1, 0, 3, 2)

    dm_core = mo_core @ mo_core.T * 2.0
    dm_cas = reduce(np.dot, (mo_cas, casdm1, mo_cas.T))

    aapa = np.zeros((ncas, ncas, nmo, ncas), dtype=np.float64)
    for i in range(nmo):
        aapa[:, :, i, :] = np.asarray(eris.ppaa[i], dtype=np.float64)[ncore:nocc, :, :].transpose(1, 2, 0)
    vj, vk = mc._scf.get_jk(mol, (dm_core, dm_cas))
    h1 = np.asarray(mc.get_hcore(), dtype=np.float64)
    vhf_c = np.asarray(vj[0] - vk[0] * 0.5, dtype=np.float64)
    vhf_a = np.asarray(vj[1] - vk[1] * 0.5, dtype=np.float64)

    gfock = np.zeros_like(dm_cas)
    gfock[:, :nocc] = reduce(np.dot, (mo_coeff.T, vhf_a, mo_occ)) * 2.0
    gfock[:, ncore:nocc] = reduce(np.dot, (mo_coeff.T, h1 + vhf_c, mo_cas, casdm1))
    gfock[:, ncore:nocc] += np.einsum("uvpw,vuwt->pt", aapa, casdm2)
    dme0 = reduce(np.dot, (mo_coeff, (gfock + gfock.T) * 0.5, mo_coeff.T))

    vj, vk = mf_grad.get_jk(mol, (dm_core, dm_cas))
    vhf1c, vhf1a = vj - vk * 0.5
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    diag_idx = np.arange(nao, dtype=np.int64)
    diag_idx = diag_idx * (diag_idx + 1) // 2 + diag_idx
    casdm2_cc = casdm2 + casdm2.transpose(0, 1, 3, 2)
    dm2buf = nr_e2_half_transform(
        casdm2_cc.reshape(ncas * ncas, ncas * ncas), mo_cas.T, (0, nao, 0, nao)
    ).reshape(ncas * ncas, nao, nao)
    dm2buf = pack_tril(dm2buf)
    dm2buf[:, diag_idx] *= 0.5
    dm2buf = dm2buf.reshape(ncas, ncas, nao_pair)

    if atmlst is None:
        atmlst = list(range(mol.natm))
    aoslices = mol.aoslice_by_atom()
    de_hcore = np.zeros((len(atmlst), 3), dtype=np.float64)
    de_renorm = np.zeros((len(atmlst), 3), dtype=np.float64)
    de_eri = np.zeros((len(atmlst), 3), dtype=np.float64)

    max_memory = float(getattr(mc, "max_memory", 2000))
    max_p = int((aoslices[:, 3] - aoslices[:, 2]).max())
    blksize = int(max_memory * 0.9e6 / 8 / (4 * max(1, max_p) * nao_pair))
    blksize = min(nao, max(2, blksize))

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[int(ia)]
        h1ao = hcore_deriv(int(ia))
        de_hcore[k] += np.einsum("xij,ij->x", h1ao, dm_cas)
        de_renorm[k] -= np.einsum("xij,ij->x", s1[:, p0:p1], dme0[p0:p1]) * 2.0

        q1 = 0
        for b0, b1, nf in shell_prange(mol, 0, mol.nbas, blksize):
            q0, q1 = q1, q1 + nf
            dm2_ao = np.einsum("ijw,pi,qj->pqw", dm2buf, mo_cas[p0:p1], mo_cas[q0:q1])
            shls_slice = (shl0, shl1, b0, b1, 0, mol.nbas, 0, mol.nbas)
            eri1 = mol.intor("int2e_ip1", comp=3, aosym="s2kl", shls_slice=shls_slice).reshape(
                3, p1 - p0, nf, nao_pair
            )
            de_eri[k] -= np.einsum("xijw,ijw->x", eri1, dm2_ao) * 2.0

        de_eri[k] += np.einsum("xij,ij->x", vhf1c[:, p0:p1], dm_cas[p0:p1]) * 2.0
        de_eri[k] += np.einsum("xij,ij->x", vhf1a[:, p0:p1], dm_core[p0:p1]) * 2.0

    return de_hcore + de_renorm + de_eri


__all__ = ["Lorb_dot_dgorb_dx", "Lci_dot_dgci_dx", "pack_tril", "shell_prange"]
