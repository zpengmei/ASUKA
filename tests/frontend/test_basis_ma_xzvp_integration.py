from __future__ import annotations

import numpy as np
import pytest

from asuka.frontend import Molecule
from asuka.frontend.basis_bse import load_autoaux_shells, load_basis_shells
from asuka.frontend.df import build_df_bases_cart


def _shell_shapes(shells: list[tuple[int, np.ndarray, np.ndarray]]) -> list[tuple[int, tuple[int, ...], tuple[int, ...]]]:
    out: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []
    for l, exps, coefs in shells:
        out.append((int(l), tuple(np.asarray(exps).shape), tuple(np.asarray(coefs).shape)))
    return out


@pytest.mark.parametrize(
    "alias,canonical",
    [
        ("ma-SV(P)", "ma-SVPP"),
        ("ma-def2-SV(P)", "ma-SVPP"),
        ("ma-def2-SVPP", "ma-SVPP"),
        ("ma-def2-TZVP", "ma-TZVP"),
        ("ma-def2-TZVPP", "ma-TZVPP"),
    ],
)
def test_ma_basis_aliases_match_canonical_shapes(alias: str, canonical: str) -> None:
    elems = ["H", "O"]
    got_alias = load_basis_shells(alias, elements=elems)
    got_canon = load_basis_shells(canonical, elements=elems)

    for sym in elems:
        assert _shell_shapes(got_alias[sym]) == _shell_shapes(got_canon[sym])


def test_ma_basis_loader_returns_defensive_copies() -> None:
    elems = ["H"]
    s1 = load_basis_shells("ma-TZVP", elements=elems)
    s2 = load_basis_shells("ma-TZVP", elements=elems)

    # Mutating one result must not pollute future calls.
    s1["H"][0][2][0, 0] = 123.456

    assert not np.isclose(s2["H"][0][2][0, 0], 123.456)


def test_bse_loader_matches_pyscf_style_shell_order_for_li_631g() -> None:
    pytest.importorskip("basis_set_exchange")

    shells = load_basis_shells("6-31g", elements=["Li"])["Li"]
    ang_mom = [int(l) for l, _exps, _coefs in shells]
    assert ang_mom == [0, 0, 0, 1, 1]

    exps0 = np.asarray(shells[0][1], dtype=np.float64)
    exps1 = np.asarray(shells[1][1], dtype=np.float64)
    exps2 = np.asarray(shells[2][1], dtype=np.float64)
    exps3 = np.asarray(shells[3][1], dtype=np.float64)
    exps4 = np.asarray(shells[4][1], dtype=np.float64)

    assert np.allclose(exps0, np.asarray([642.418915, 96.7985153, 22.0911212, 6.20107025, 1.93511768, 0.636735789]))
    assert np.allclose(exps1, np.asarray([2.32491841, 0.632430356, 0.0790534347]))
    assert np.allclose(exps2, np.asarray([0.0359619718]))
    assert np.allclose(exps3, np.asarray([2.32491841, 0.632430356, 0.0790534347]))
    assert np.allclose(exps4, np.asarray([0.0359619718]))


def test_ma_basis_autoaux_fallback_and_df_builder_smoke() -> None:
    pytest.importorskip("basis_set_exchange")

    aux_name, aux_shells = load_autoaux_shells("ma-def2-TZVP", elements=["H", "O"])
    assert "autoaux" in aux_name.lower()
    assert "tzvp" in aux_name.lower()
    assert "H" in aux_shells and "O" in aux_shells

    mol = Molecule.from_atoms("H 0 0 0; H 0 0 1.4", unit="Bohr", basis="ma-SVP", cart=True)
    ao_basis, aux_basis, resolved = build_df_bases_cart(
        mol,
        basis="ma-def2-SVP",
        auxbasis="autoaux",
        expand_contractions=False,
    )

    assert int(np.asarray(ao_basis.shell_l, dtype=np.int32).size) > 0
    assert int(np.asarray(aux_basis.shell_l, dtype=np.int32).size) > 0
    assert "autoaux" in str(resolved).lower()
