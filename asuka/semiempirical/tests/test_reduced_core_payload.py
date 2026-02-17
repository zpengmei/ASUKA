from __future__ import annotations

import numpy as np
import pytest

from asuka.nddo_core import build_pair_list, compute_all_multipole_params
from asuka.semiempirical.basis import symbol_to_Z
from asuka.semiempirical.core_repulsion import (
    core_core_repulsion,
    core_core_repulsion_from_gamma_ss,
)
from asuka.semiempirical.fock import (
    build_core_hamiltonian,
    build_core_hamiltonian_from_pair_terms,
)
from asuka.semiempirical.nddo_integrals import (
    build_pair_ri_payload,
    build_two_center_integrals,
)
from asuka.semiempirical.overlap import build_overlap_matrix
from asuka.semiempirical.params import ANGSTROM_TO_BOHR, load_params

_CASES = [
    (
        "h2_11",
        ["H", "H"],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float),
    ),
    (
        "hcn_14",
        ["H", "C", "N"],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.06], [0.0, 0.0, 2.22]], dtype=float),
    ),
    (
        "ch4_41",
        ["C", "H", "H", "H", "H"],
        np.array(
            [
                [0.0000, 0.0000, 0.0000],
                [0.6291, 0.6291, 0.6291],
                [0.6291, -0.6291, -0.6291],
                [-0.6291, 0.6291, -0.6291],
                [-0.6291, -0.6291, 0.6291],
            ],
            dtype=float,
        ),
    ),
    (
        "co_44",
        ["C", "O"],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]], dtype=float),
    ),
]


@pytest.mark.parametrize("case_name,symbols,coords_ang", _CASES)
def test_reduced_payload_core_equivalence(case_name, symbols, coords_ang):
    del case_name
    atomic_numbers = [symbol_to_Z(sym) for sym in symbols]
    coords_bohr = np.asarray(coords_ang, dtype=float) * ANGSTROM_TO_BOHR
    elem_params = load_params("AM1").elements

    pair_i, pair_j, _, pair_r = build_pair_list(coords_bohr)
    mp_params = compute_all_multipole_params(elem_params)
    ri_pack, ta_pack, tb_pack, vaa_pack, vbb_pack, gamma_ss = build_pair_ri_payload(
        atomic_numbers=atomic_numbers,
        coords_bohr=coords_bohr,
        pair_i=pair_i,
        pair_j=pair_j,
        mp_params=mp_params,
    )

    npairs = len(pair_i)
    assert ri_pack.shape == (npairs, 22)
    assert ta_pack.shape == (npairs, 16)
    assert tb_pack.shape == (npairs, 16)
    assert vaa_pack.shape == (npairs, 16)
    assert vbb_pack.shape == (npairs, 16)
    assert gamma_ss.shape == (npairs,)
    assert np.all(np.isfinite(ri_pack))
    assert np.all(np.isfinite(ta_pack))
    assert np.all(np.isfinite(tb_pack))
    assert np.all(np.isfinite(vaa_pack))
    assert np.all(np.isfinite(vbb_pack))
    assert np.all(np.isfinite(gamma_ss))

    W_list = build_two_center_integrals(atomic_numbers, coords_bohr, pair_i, pair_j, mp_params)
    gamma_ref = np.asarray([W[0, 0, 0, 0] for W in W_list], dtype=float)
    assert np.allclose(gamma_ss, gamma_ref, atol=1e-12, rtol=0.0)

    S = build_overlap_matrix(atomic_numbers, coords_bohr, elem_params)
    H_ref = build_core_hamiltonian(
        atomic_numbers=atomic_numbers,
        coords_bohr=coords_bohr,
        S=S,
        pair_i=pair_i,
        pair_j=pair_j,
        W_list=W_list,
        elem_params=elem_params,
    )
    H_new = build_core_hamiltonian_from_pair_terms(
        atomic_numbers=atomic_numbers,
        S=S,
        pair_i=pair_i,
        pair_j=pair_j,
        vaa_pack=vaa_pack,
        vbb_pack=vbb_pack,
        elem_params=elem_params,
    )
    assert np.allclose(H_ref, H_new, atol=1e-10, rtol=0.0)

    e_ref = core_core_repulsion(
        atomic_numbers=atomic_numbers,
        coords_bohr=coords_bohr,
        pair_i=pair_i,
        pair_j=pair_j,
        pair_r=pair_r,
        W_list=W_list,
        elem_params=elem_params,
    )
    e_new = core_core_repulsion_from_gamma_ss(
        atomic_numbers=atomic_numbers,
        pair_i=pair_i,
        pair_j=pair_j,
        pair_r=pair_r,
        gamma_ss=gamma_ss,
        elem_params=elem_params,
    )
    assert abs(e_ref - e_new) <= 1e-10
