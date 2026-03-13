from __future__ import annotations

import pytest
import numpy as np

from asuka.hf.direct_jk_cost import (
    DirectJKSymmetryCase,
    estimate_direct_fock_contract_cost,
    estimate_direct_jk_contract_cost,
    symmetry_histogram_from_cases,
)


def test_symmetry_histogram_from_cases_compacts_bins():
    hist = symmetry_histogram_from_cases(
        [
            DirectJKSymmetryCase(ntasks=2, ab_neq=False, cd_neq=False, bra_ket_swap=False),
            DirectJKSymmetryCase(ntasks=3, ab_neq=True, cd_neq=False, bra_ket_swap=True),
            DirectJKSymmetryCase(ntasks=5, ab_neq=True, cd_neq=True, bra_ket_swap=False),
        ]
    )

    expected = np.zeros((8,), dtype=np.int64)
    expected[0b000] = 2
    expected[0b101] = 3
    expected[0b110] = 5
    np.testing.assert_array_equal(hist, expected)


def test_estimate_direct_jk_contract_cost_staged_ssss_exact_counts():
    hist = symmetry_histogram_from_cases(
        [DirectJKSymmetryCase(ntasks=5, ab_neq=False, cd_neq=False, bra_ket_swap=False)]
    )

    cost = estimate_direct_jk_contract_cost(
        histogram=hist,
        nA=1,
        nB=1,
        nC=1,
        nD=1,
        contract_mode="staged",
        want_J=True,
        want_K=True,
        n_dm=1,
    )

    assert cost.ntasks == 5
    assert cost.eri_values == 5
    assert cost.tile_loads == 5
    assert cost.tile_bytes_read == 40
    assert cost.tile_roundtrip_bytes == 80
    assert cost.density_loads == 10
    assert cost.density_bytes == 80
    assert cost.inner_fmuls == 10
    assert cost.scale_fmuls == 5
    assert cost.fmuls == 15
    assert cost.reduction_adds == 0
    assert cost.output_atomics == 10
    assert cost.output_payload_bytes == 80
    assert cost.output_rmw_bytes_lower_bound == 160
    assert cost.flops == 25
    assert cost.arithmetic_intensity_payload == pytest.approx(25.0 / 200.0)
    assert cost.arithmetic_intensity_rmw_lower_bound == pytest.approx(25.0 / 280.0)


def test_estimate_direct_jk_contract_cost_staged_branchy_j_only_exact_counts():
    hist = symmetry_histogram_from_cases(
        [DirectJKSymmetryCase(ntasks=2, ab_neq=True, cd_neq=False, bra_ket_swap=True)]
    )

    cost = estimate_direct_jk_contract_cost(
        histogram=hist,
        nA=3,
        nB=1,
        nC=1,
        nD=1,
        contract_mode="staged",
        want_J=True,
        want_K=False,
        n_dm=1,
    )

    # m = ntasks * nAB * nCD = 2 * 3 * 1 = 6
    assert cost.eri_values == 6
    assert cost.tile_loads == 6
    assert cost.density_loads == 12
    assert cost.inner_fmuls == 12
    assert cost.scale_fmuls == 12
    assert cost.fmuls == 24
    # J atomics per ERI element: (1 + ab_neq) + bk_swap * (1 + cd_neq) = 2 + 1 = 3
    assert cost.output_atomics == 18
    assert cost.flops == 42


def test_estimate_direct_jk_contract_cost_warp_trades_atomics_for_tile_reads():
    hist = symmetry_histogram_from_cases(
        [DirectJKSymmetryCase(ntasks=1, ab_neq=True, cd_neq=True, bra_ket_swap=True)]
    )

    staged = estimate_direct_jk_contract_cost(
        histogram=hist,
        nA=3,
        nB=1,
        nC=3,
        nD=1,
        contract_mode="staged",
        want_J=True,
        want_K=True,
        n_dm=1,
    )
    warp = estimate_direct_jk_contract_cost(
        histogram=hist,
        nA=3,
        nB=1,
        nC=3,
        nD=1,
        contract_mode="warp",
        want_J=True,
        want_K=True,
        n_dm=1,
    )

    assert staged.eri_values == warp.eri_values == 9
    assert staged.density_loads == warp.density_loads == 54
    assert staged.inner_fmuls == warp.inner_fmuls == 54

    # Warp mode re-reads the tile for every reduction family.
    assert staged.tile_loads == 9
    assert warp.tile_loads == 54

    # Warp mode reduces global atomics but pays explicit reduction adds.
    assert staged.output_atomics == 108
    assert warp.output_atomics == 44
    assert staged.reduction_adds == 0
    assert warp.reduction_adds == 32

    # Warp J scaling happens once per reduced output instead of once per ERI element.
    assert staged.scale_fmuls == 18
    assert warp.scale_fmuls == 6
    assert staged.fmuls == 72
    assert warp.fmuls == 60


def test_estimate_direct_jk_contract_cost_multi_density_scales_only_reused_terms():
    hist = symmetry_histogram_from_cases(
        [DirectJKSymmetryCase(ntasks=5, ab_neq=False, cd_neq=False, bra_ket_swap=False)]
    )

    cost = estimate_direct_jk_contract_cost(
        histogram=hist,
        nA=1,
        nB=1,
        nC=1,
        nD=1,
        contract_mode="staged",
        want_J=True,
        want_K=True,
        n_dm=2,
    )

    # Tile is read once per ERI element even though two densities are handled.
    assert cost.tile_loads == 5
    assert cost.tile_bytes_read == 40

    # Density-side work doubles with n_dm.
    assert cost.density_loads == 20
    assert cost.inner_fmuls == 20
    assert cost.scale_fmuls == 10
    assert cost.fmuls == 30
    assert cost.output_atomics == 20
    assert cost.flops == 50


def test_estimate_direct_fock_contract_cost_adds_k_scaling_work():
    hist = symmetry_histogram_from_cases(
        [DirectJKSymmetryCase(ntasks=1, ab_neq=True, cd_neq=True, bra_ket_swap=True)]
    )

    jk = estimate_direct_jk_contract_cost(
        histogram=hist,
        nA=3,
        nB=1,
        nC=3,
        nD=1,
        contract_mode="warp",
        want_J=True,
        want_K=True,
        n_dm=1,
    )
    fock = estimate_direct_fock_contract_cost(
        histogram=hist,
        nA=3,
        nB=1,
        nC=3,
        nD=1,
        contract_mode="warp",
        n_dm=1,
    )

    # Same structural contraction, but Fock scales each reduced K output by alpha=-0.5.
    assert fock.tile_loads == jk.tile_loads
    assert fock.density_loads == jk.density_loads
    assert fock.output_atomics == jk.output_atomics
    assert fock.reduction_adds == jk.reduction_adds
    assert fock.scale_fmuls == jk.scale_fmuls + 16
    assert fock.fmuls == jk.fmuls + 16
    assert fock.flops == jk.flops + 16
