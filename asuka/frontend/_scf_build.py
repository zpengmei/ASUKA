from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

from asuka.integrals.int1e_cart import Int1eResult

from .basis_bse import load_autoaux_shells, load_basis_shells
from .basis_packer import pack_cart_basis, parse_pyscf_basis_dict
from .periodic_table import atomic_number

if TYPE_CHECKING:  # pragma: no cover
    from asuka.integrals.cart2sph import AOSphericalTransform
    from .molecule import Molecule


def apply_sph_transform(
    mol: Molecule,
    int1e: Int1eResult,
    B,
    ao_basis,
    *,
    df_B_layout: str = "mnQ",
) -> tuple[Int1eResult, Any, AOSphericalTransform | None]:
    """If ``mol.cart=False``, transform int1e and B to spherical AOs."""
    if bool(mol.cart):
        return int1e, B, None

    from asuka.integrals.cart2sph import (  # noqa: PLC0415
        AOSphericalTransform,
        build_cart2sph_matrix,
        compute_sph_layout_from_cart_basis,
        transform_1e_cart_to_sph,
        transform_df_B_cart_to_sph,
    )

    shell_l = np.asarray(ao_basis.shell_l, dtype=np.int32).ravel()
    if int(shell_l.size) and int(np.max(shell_l)) > 5:
        raise ValueError(
            "Spherical AO transform supports basis shells up to l<=5. "
            f"Got max(shell_l)={int(np.max(shell_l))}. Use cart=True for higher angular momentum."
        )
    shell_ao_start_cart = np.asarray(ao_basis.shell_ao_start, dtype=np.int32).ravel()
    shell_ao_start_sph, nao_sph = compute_sph_layout_from_cart_basis(ao_basis)
    nao_cart = int(int1e.S.shape[0])

    T = build_cart2sph_matrix(shell_l, shell_ao_start_cart, shell_ao_start_sph, nao_cart, nao_sph)

    S_sph = transform_1e_cart_to_sph(int1e.S, T)
    T_kin_sph = transform_1e_cart_to_sph(int1e.T, T)
    V_sph = transform_1e_cart_to_sph(int1e.V, T)
    int1e_sph = Int1eResult(S=S_sph, T=T_kin_sph, V=V_sph)

    if B is not None:
        B_sph = transform_df_B_cart_to_sph(
            B,
            T,
            shell_l=shell_l,
            shell_ao_start_cart=shell_ao_start_cart,
            shell_ao_start_sph=shell_ao_start_sph,
            out_layout=str(df_B_layout),
        )
    else:
        B_sph = None

    return int1e_sph, B_sph, AOSphericalTransform(T_c2s=T, nao_cart=nao_cart, nao_sph=nao_sph)


def atom_coords_charges_bohr(mol: Molecule) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray([xyz for _sym, xyz in mol.atoms_bohr], dtype=np.float64).reshape((mol.natm, 3))
    charges = np.asarray([atomic_number(sym) for sym, _xyz in mol.atoms_bohr], dtype=np.float64)
    return coords, charges


def unique_elements(mol: Molecule) -> list[str]:
    return sorted(set(mol.elements))


def build_aux_basis_cart(
    mol: Molecule,
    *,
    basis_in: Any,
    auxbasis: Any,
    expand_contractions: bool,
    ao_basis: Any = None,
) -> tuple[Any, str]:
    """Build (aux_basis, auxbasis_name) as a cuERI packed cart basis."""

    from asuka.integrals.ricd_types import is_ricd_request  # noqa: PLC0415

    if is_ricd_request(auxbasis):
        if ao_basis is None:
            raise ValueError(
                "ao_basis must be provided when using RICD auxiliary basis generation "
                "(auxbasis='ricd'/'acd'/'accd' or an RICDOptions instance)"
            )
        from asuka.integrals.ricd_types import normalize_ricd_options  # noqa: PLC0415
        from asuka.integrals.ricd_builder import build_ricd_aux_basis  # noqa: PLC0415

        opts = normalize_ricd_options(auxbasis)
        atoms_bohr = list(mol.atoms_bohr)
        gen = build_ricd_aux_basis(ao_basis, atoms_bohr, options=opts)
        return gen.packed_basis, gen.basis_name

    elements = unique_elements(mol)
    auxbasis_name = ""

    def _autoaux_basis_name_hint() -> str | None:
        if isinstance(basis_in, str):
            base = str(basis_in).strip()
            return base or None
        try:
            hint = mol.results.get("basis_name")
        except Exception:
            hint = None
        if isinstance(hint, str):
            hint_s = str(hint).strip()
            return hint_s or None
        return None

    if isinstance(auxbasis, str) and str(auxbasis).strip().lower() in ("auto", "autoaux"):
        base = _autoaux_basis_name_hint()
        if base is None:
            raise ValueError(
                "auxbasis='autoaux' requires basis to be a string name "
                "(or set mol.results['basis_name'] to the corresponding basis name)"
            )
        auxbasis_name, aux_shells = load_autoaux_shells(str(base), elements=elements)
    elif isinstance(auxbasis, str):
        auxbasis_name = str(auxbasis)
        try:
            aux_shells = load_basis_shells(auxbasis_name, elements=elements)
        except Exception:
            base = str(auxbasis_name).strip()
            for suf in ("-jkfit", "-jfit", "-rifit", "-ri", "-mp2fit"):
                if base.lower().endswith(suf):
                    base = base[: -len(suf)]
                    break
            base = base or (_autoaux_basis_name_hint() or "")
            if base:
                auxbasis_name, aux_shells = load_autoaux_shells(str(base), elements=elements)
            else:
                raise
    elif isinstance(auxbasis, dict):
        auxbasis_name = "<explicit>"
        aux_shells = parse_pyscf_basis_dict(auxbasis, elements=elements)
    else:
        raise TypeError("auxbasis must be 'autoaux', a string name, or an explicit per-element basis dict")

    aux_basis = pack_cart_basis(list(mol.atoms_bohr), aux_shells, expand_contractions=bool(expand_contractions))
    return aux_basis, auxbasis_name or "<unknown>"


def init_guess_dm_atom_hcore_cart(
    mol: Molecule,
    *,
    ao_basis,
    int1e_cart: Int1eResult,
) -> np.ndarray:
    """Build a SAD-like initial density from per-atom core-Hamiltonian blocks."""

    from asuka.cueri.cart import ncart  # noqa: PLC0415
    from asuka.hf.local_thc_partition import map_shells_to_atoms  # noqa: PLC0415

    S = np.asarray(int1e_cart.S, dtype=np.float64)
    h = np.asarray(int1e_cart.hcore, dtype=np.float64)
    nao = int(S.shape[0])
    if S.shape != (nao, nao) or h.shape != (nao, nao):
        raise ValueError("int1e_cart must have (nao,nao) S/hcore")

    coords, _charges = atom_coords_charges_bohr(mol)
    _sh2a, atom_to_shells = map_shells_to_atoms(np.asarray(ao_basis.shell_cxyz), coords)

    shell_l = np.asarray(ao_basis.shell_l, dtype=np.int32).ravel()
    shell_start = np.asarray(ao_basis.shell_ao_start, dtype=np.int32).ravel()
    if shell_l.shape != shell_start.shape:
        raise ValueError("ao_basis.shell_l and shell_ao_start must have identical shape")
    if int(shell_l.size) == 0:
        return np.zeros((nao, nao), dtype=np.float64)

    nfn_shell = np.asarray([ncart(int(l)) for l in shell_l.tolist()], dtype=np.int32)

    orb_eps: list[float] = []
    orb_idx: list[np.ndarray] = []
    orb_c: list[np.ndarray] = []

    S = 0.5 * (S + S.T)
    h = 0.5 * (h + h.T)

    for ia in range(int(mol.natm)):
        shells = atom_to_shells[int(ia)]
        if not shells:
            continue
        idx: list[int] = []
        for sh in shells:
            s0 = int(shell_start[int(sh)])
            n = int(nfn_shell[int(sh)])
            idx.extend(range(s0, s0 + n))
        if not idx:
            continue
        idx_np = np.asarray(idx, dtype=np.int32)

        Sblk = np.asarray(S[np.ix_(idx_np, idx_np)], dtype=np.float64)
        hblk = np.asarray(h[np.ix_(idx_np, idx_np)], dtype=np.float64)
        Sblk = 0.5 * (Sblk + Sblk.T)
        hblk = 0.5 * (hblk + hblk.T)

        w, U = np.linalg.eigh(Sblk)
        w = np.asarray(w, dtype=np.float64)
        U = np.asarray(U, dtype=np.float64)
        wmax = float(np.max(w)) if int(w.size) else 0.0
        if wmax <= 0.0:
            continue
        keep = w > (1e-12 * wmax)
        if not bool(np.any(keep)):
            continue
        wk = w[keep]
        Uk = U[:, keep]
        Xk = Uk / np.sqrt(wk)[None, :]
        Fp = Xk.T @ hblk @ Xk
        Fp = 0.5 * (Fp + Fp.T)
        e, Cp = np.linalg.eigh(Fp)
        e = np.asarray(e, dtype=np.float64).ravel()
        Cp = np.asarray(Cp, dtype=np.float64)
        C = Xk @ Cp

        for j in range(int(C.shape[1])):
            orb_eps.append(float(e[int(j)]))
            orb_idx.append(idx_np)
            orb_c.append(np.ascontiguousarray(C[:, int(j)], dtype=np.float64))

    nelec = int(mol.nelectron)
    if nelec <= 0 or nelec % 2 != 0:
        raise ValueError("RHF requires an even, positive electron count")
    nocc = int(nelec // 2)
    if int(len(orb_eps)) < int(nocc):
        raise RuntimeError("atom-hcore guess produced too few orbitals to fill nelec")

    order = np.argsort(np.asarray(orb_eps, dtype=np.float64), kind="stable")
    D = np.zeros((nao, nao), dtype=np.float64)
    for k in range(int(nocc)):
        j = int(order[int(k)])
        idx_np = orb_idx[j]
        c = orb_c[j]
        D[np.ix_(idx_np, idx_np)] += 2.0 * np.outer(c, c)
    return 0.5 * (D + D.T)
