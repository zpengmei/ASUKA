from __future__ import annotations

import numpy as np

from asuka.caspt2.gradient.dense_btamp import (
    build_gtoc_dense_term_arrays,
    build_t2ao_dense,
    contract_ordered_ao4_dense_deriv,
)
from asuka.frontend import Molecule
from asuka.frontend.scf import run_hf_df
from asuka.hf.dense_eri import build_ao_eri_dense


def _rand_orthonormal(n: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    q, _r = np.linalg.qr(rng.standard_normal((n, n)))
    return np.asarray(q, dtype=np.float64)


def test_build_gtoc_dense_term_arrays_matches_explicit_q_definition() -> None:
    nmo = 4
    nocc = 2
    scale = 0.125
    mo_coeff = _rand_orthonormal(nmo, seed=7)
    rng = np.random.default_rng(11)
    t2_mo = rng.standard_normal((nocc, nmo, nocc, nmo))

    t2ao = build_t2ao_dense(mo_coeff=mo_coeff, t2_mo=t2_mo)
    got = build_gtoc_dense_term_arrays(
        mo_coeff=mo_coeff,
        t2_mo=t2_mo,
        scale=scale,
    )

    c_occ = np.asarray(mo_coeff[:, :nocc], dtype=np.float64)

    def q_term(x: int, y: int, u: int, v: int) -> float:
        out = 0.0
        for iocc in range(nocc):
            for jocc in range(nocc):
                thbf_ji = t2ao[jocc, v, iocc, u] + t2ao[iocc, u, jocc, v]
                out += c_occ[x, jocc] * c_occ[y, iocc] * thbf_ji
        return float(scale * out)

    exp_ik_to_jl = np.zeros((nmo, nmo, nmo, nmo), dtype=np.float64)
    exp_jl_to_ik = np.zeros_like(exp_ik_to_jl)
    exp_il_to_jk = np.zeros_like(exp_ik_to_jl)
    exp_jk_to_il = np.zeros_like(exp_ik_to_jl)

    for ja in range(nmo):
        for la in range(nmo):
            for ia in range(nmo):
                for ka in range(nmo):
                    exp_ik_to_jl[ja, la, ia, ka] = q_term(ja, la, ia, ka)
                    exp_jl_to_ik[ja, la, ia, ka] = q_term(ia, ka, ja, la)
                    exp_il_to_jk[ja, la, ia, ka] = q_term(ja, ka, ia, la)
                    exp_jk_to_il[ja, la, ia, ka] = q_term(ia, la, ja, ka)

    assert np.allclose(got["ik_to_jl"], exp_ik_to_jl)
    assert np.allclose(got["jl_to_ik"], exp_jl_to_ik)
    assert np.allclose(got["il_to_jk"], exp_il_to_jk)
    assert np.allclose(got["jk_to_il"], exp_jk_to_il)
    assert np.allclose(got["pair_only"], exp_il_to_jk + exp_jk_to_il)
    assert np.allclose(
        got["total"],
        exp_ik_to_jl + exp_jl_to_ik + exp_il_to_jk + exp_jk_to_il,
    )


def test_contract_ordered_ao4_dense_deriv_matches_fd() -> None:
    mol = Molecule.from_atoms(
        "H 0 0 0; H 0 0 1.4",
        unit="Angstrom",
        basis="sto-3g",
        cart=True,
    )
    scf = run_hf_df(mol, method="rhf", backend="cpu", df=True, auxbasis="autoaux")
    nao = int(np.asarray(scf.int1e.S).shape[0])
    coords0 = np.asarray(mol.coords_bohr, dtype=np.float64)
    rng = np.random.default_rng(123)
    bar = rng.standard_normal((nao, nao, nao, nao))

    grad_analytic = contract_ordered_ao4_dense_deriv(
        ao_basis=scf.ao_basis,
        atom_coords_bohr=coords0,
        bar_ao4=bar,
    )

    step = 1.0e-4
    syms = list(mol.elements)
    grad_fd = np.zeros_like(grad_analytic)
    for ia in range(int(coords0.shape[0])):
        for ax in range(3):
            coords_p = coords0.copy()
            coords_m = coords0.copy()
            coords_p[ia, ax] += step
            coords_m[ia, ax] -= step

            def _eri_at(coords_bohr: np.ndarray) -> np.ndarray:
                atoms = "; ".join(
                    f"{syms[i]} {coords_bohr[i,0]:.10f} {coords_bohr[i,1]:.10f} {coords_bohr[i,2]:.10f}"
                    for i in range(len(syms))
                )
                mol_i = Molecule.from_atoms(atoms, unit="bohr", basis="sto-3g", cart=True)
                scf_i = run_hf_df(mol_i, method="rhf", backend="cpu", df=True, auxbasis="autoaux")
                eri = build_ao_eri_dense(scf_i.ao_basis, backend="cpu", eps_ao=0.0).eri_mat
                return np.asarray(eri, dtype=np.float64).reshape((nao, nao, nao, nao))

            der_eri = (_eri_at(coords_p) - _eri_at(coords_m)) / (2.0 * step)
            grad_fd[ia, ax] = float(np.einsum("pqrs,pqrs->", bar, der_eri, optimize=True))

    np.testing.assert_allclose(grad_analytic, grad_fd, atol=1.0e-8, rtol=1.0e-8)
