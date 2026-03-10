from __future__ import annotations

import numpy as np
import pytest


def _has_cpu_eri_ext() -> bool:
    try:
        from asuka.cueri import _eri_rys_cpu as _ext  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_cpu_eri_ext(), reason="requires asuka.cueri._eri_rys_cpu CPU ERI extension")
def test_ricd_df_pair_matrix_matches_dense_h2o_sto3g() -> None:
    """Regression test for RICD(aux) → DF(B) accuracy on a real molecule.

    This targets the AO-pair Coulomb metric V(pq,rs) reconstructed from the
    whitened DF factors, compared to dense AO ERIs.
    """

    try:
        from asuka.frontend import Molecule
        from asuka.frontend.basis_bse import load_basis_shells
        from asuka.frontend.basis_packer import pack_cart_basis
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"frontend basis loader unavailable: {type(e).__name__}: {e}")

    from asuka.hf.dense_eri import build_ao_eri_dense
    from asuka.integrals.cueri_df_cpu import build_df_B_from_cueri_packed_bases_cpu
    from asuka.integrals.ricd_builder import build_ricd_aux_basis
    from asuka.integrals.ricd_types import RICDOptions

    mol = Molecule.from_atoms(
        "O 0 0 0; H 0.757160 0.586260 0; H -0.757160 0.586260 0",
        unit="Angstrom",
        basis="sto-3g",
        cart=True,
    )
    atoms_bohr = list(mol.atoms_bohr)

    shells = load_basis_shells("sto-3g", elements=sorted(set(mol.elements)))
    ao_basis = pack_cart_basis(atoms_bohr, shells, expand_contractions=True)

    dense = build_ao_eri_dense(ao_basis, backend="cpu", threads=0)
    eri_mat = np.asarray(dense.eri_mat, dtype=np.float64)
    nao = int(dense.nao)

    gen = build_ricd_aux_basis(
        ao_basis,
        atoms_bohr,
        options=RICDOptions(mode="accd", threshold=1.0e-4),
        threads=0,
    )
    aux_basis = gen.packed_basis
    B = np.asarray(build_df_B_from_cueri_packed_bases_cpu(ao_basis, aux_basis, threads=0), dtype=np.float64)
    assert tuple(B.shape[:2]) == (nao, nao)

    pairs = [(p, q) for p in range(nao) for q in range(p + 1)]
    idx = np.asarray([p * nao + q for p, q in pairs], dtype=np.int64)
    V_exact = eri_mat[np.ix_(idx, idx)]

    b_pair = np.asarray([B[p, q, :] for p, q in pairs], dtype=np.float64)
    V_df = np.asarray(b_pair @ b_pair.T, dtype=np.float64)

    D = V_df - V_exact
    max_abs = float(np.max(np.abs(D)))
    rel_norm = float(np.linalg.norm(D) / np.linalg.norm(V_exact))

    # Empirically, Molcas RICD (default τ=1e-4) yields ~2.34e-3 max-abs on this
    # system; keep a small cushion for platform differences.
    assert max_abs <= 3.0e-3
    assert rel_norm <= 1.2e-3

