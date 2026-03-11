from __future__ import annotations

"""LS-THC DVR grids (CPU build, optional device upload).

This module implements the R-DVR and F-DVR grid constructions described in:

  Parrish et al., J. Chem. Phys. 138, 194107 (2013)

These grids are intended for THC/LS-THC factor construction, and are exposed
via the HF THC factor builders (`asuka.hf.thc_factors`, `asuka.hf.local_thc_factors`).

Notes
-----
- All coordinates are in Bohr.
- The CPU routines return NumPy arrays; `*_device` wrappers upload to CuPy.
- R-DVR is atom-centered and uses Becke partitioning to form molecular weights.
- F-DVR is a global, "molecule-shaped" quadrature with weights from the DVR
  interpolating zeros property. It is global-only and not suitable for per-atom
  local-THC region construction.
"""

from dataclasses import dataclass
from math import erf, gamma
from typing import Any, Iterator

import numpy as np

from asuka.cueri.basis_cart import BasisCartSoA
from asuka.density.grids import _coords_bohr, angular_grid, becke_partition_weights
from asuka.density.types import GridBatch

# Bragg-Slater radii in Bohr, tabulated in PySCF:
#   pyscf.dft.radi.BRAGG_RADII
#
# We embed these values so `asuka.density` remains PySCF-free at runtime.
_BRAGG_SLATER_RADII_BOHR = np.asarray(
    [
        3.779450359403999,
        0.6614041435977716,
        2.645616574391086,
        2.7401028806193395,
        1.984212430793315,
        1.6062672058803025,
        1.322808287195543,
        1.2283219809672903,
        1.133835674739037,
        0.9448630622825309,
        2.8345891868475928,
        3.4015070242171115,
        2.8345891868475928,
        2.3621576557063273,
        2.0786987370215684,
        1.8897261245650618,
        1.8897261245650618,
        1.8897261245650618,
        3.4015070242171115,
        4.157397474043137,
        3.4015070242171115,
        3.0235617993040993,
        2.645616574391086,
        2.551130268162834,
        2.645616574391086,
        2.645616574391086,
        2.645616574391086,
        2.551130268162834,
        2.551130268162834,
        2.551130268162834,
        2.551130268162834,
        2.4566439619345806,
        2.3621576557063273,
        2.1731850432498208,
        2.1731850432498208,
        2.1731850432498208,
        3.590479636673617,
        4.440856392727896,
        3.7794522491301237,
        3.4015070242171115,
        2.929075493075846,
        2.7401028806193395,
        2.7401028806193395,
        2.551130268162834,
        2.4566439619345806,
        2.551130268162834,
        2.645616574391086,
        3.0235617993040993,
        2.929075493075846,
        2.929075493075846,
        2.7401028806193395,
        2.7401028806193395,
        2.645616574391086,
        2.645616574391086,
        3.96842486158663,
        4.913287923869161,
        4.062911167814883,
        3.6849659429018704,
        3.4959933304453648,
        3.4959933304453648,
        3.4959933304453648,
        3.4959933304453648,
        3.4959933304453648,
        3.4959933304453648,
        3.4015070242171115,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        2.929075493075846,
        2.7401028806193395,
        2.551130268162834,
        2.551130268162834,
        2.4566439619345806,
        2.551130268162834,
        2.551130268162834,
        2.551130268162834,
        2.8345891868475928,
        3.590479636673617,
        3.4015070242171115,
        3.0235617993040993,
        3.590479636673617,
        2.7401028806193395,
        3.96842486158663,
        3.4015070242171115,
        4.062911167814883,
        3.6849659429018704,
        3.4015070242171115,
        3.4015070242171115,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
        3.307020717988858,
    ],
    dtype=np.float64,
)

# Lebedev-Laikov "exactness degree" (lmax) -> number of angular points (npts).
#
# This mapping is also taken from PySCF:
#   pyscf.dft.gen_grid.LEBEDEV_ORDER
#
# We omit the trivial (lmax=0,npts=1) rule and always use at least 6 points.
_LEBEDEV_LMAX_TO_NPTS: tuple[tuple[int, int], ...] = (
    (3, 6),
    (5, 14),
    (7, 26),
    (9, 38),
    (11, 50),
    (13, 74),
    (15, 86),
    (17, 110),
    (19, 146),
    (21, 170),
    (23, 194),
    (25, 230),
    (27, 266),
    (29, 302),
    (31, 350),
    (35, 434),
    (41, 590),
    (47, 770),
    (53, 974),
    (59, 1202),
    (65, 1454),
    (71, 1730),
    (77, 2030),
    (83, 2354),
    (89, 2702),
    (95, 3074),
    (101, 3470),
    (107, 3890),
    (113, 4334),
    (119, 4802),
    (125, 5294),
    (131, 5810),
)
_LEBEDEV_NPTS_TO_LMAX = {int(npts): int(lmax) for lmax, npts in _LEBEDEV_LMAX_TO_NPTS}


def _atomic_numbers_or_none(mol_or_coords: Any) -> np.ndarray | None:
    """Best-effort atomic number extraction (for Bragg-Slater pruning)."""

    atoms_bohr = getattr(mol_or_coords, "atoms_bohr", None)
    if atoms_bohr is None:
        return None
    try:
        from asuka.chem.periodic_table import atomic_number  # noqa: PLC0415
    except Exception:  # pragma: no cover
        return None
    try:
        Z = [int(atomic_number(sym)) for sym, _xyz in atoms_bohr]
    except Exception:
        return None
    return np.asarray(Z, dtype=np.int32)


def _lebedev_lmax_from_npts(npts: int) -> int | None:
    return _LEBEDEV_NPTS_TO_LMAX.get(int(npts))


def _lebedev_npts_for_lreq(lreq: int, *, lmax_max: int) -> int:
    """Return the smallest Lebedev npts whose lmax >= lreq, not exceeding lmax_max."""

    lreq = max(0, int(lreq))
    lmax_max = int(lmax_max)
    # Clamp the request to the available maximum.
    if lreq > lmax_max:
        lreq = int(lmax_max)

    for lmax, npts in _LEBEDEV_LMAX_TO_NPTS:
        if int(lmax) >= int(lreq) and int(lmax) <= int(lmax_max):
            return int(npts)

    # Fallback: use the maximum available up to lmax_max.
    for lmax, npts in reversed(_LEBEDEV_LMAX_TO_NPTS):
        if int(lmax) <= int(lmax_max):
            return int(npts)

    return int(_LEBEDEV_LMAX_TO_NPTS[0][1])


def _pruned_lreq_parrish_2013(rho: float, *, rho_bs: float, lmax_max: int) -> int:
    """Heuristic angular pruning envelope from Parrish et al. (2013), Eq. in Sec. D."""

    rho = float(rho)
    rho_bs = float(rho_bs)
    lmax_max = int(lmax_max)
    if not np.isfinite(rho) or rho < 0.0:
        return 0
    if not np.isfinite(rho_bs) or rho_bs <= 0.0:
        return int(lmax_max)

    z = rho / rho_bs
    # L(z) = ceil(Lmax*(erf(1.2*(z-0.0)) - 0.5*erf(1.2*(z-1.5)) - 0.5))
    env = erf(1.2 * (z - 0.0)) - 0.5 * erf(1.2 * (z - 1.5)) - 0.5
    val = float(lmax_max) * float(env)
    if not np.isfinite(val):
        return int(lmax_max)
    lreq = int(np.ceil(val))
    if lreq < 0:
        lreq = 0
    if lreq > int(lmax_max):
        lreq = int(lmax_max)
    return int(lreq)


def _gauss_int(n: int, p: float) -> float:
    """Return ∫_0^∞ r^n exp(-p r^2) dr for n>=0, p>0."""

    n = int(n)
    p = float(p)
    if n < 0:
        raise ValueError("n must be >= 0")
    if not np.isfinite(p) or p <= 0.0:
        raise ValueError("p must be finite and > 0")
    n1 = 0.5 * float(n + 1)
    return 0.5 * float(gamma(n1)) / (p**n1)


def _canonical_orth(S: np.ndarray, *, ortho_cutoff: float) -> np.ndarray:
    """Canonical orthogonalization W such that W^T S W = I.

    Returns W with shape (n, m) where m is the number of retained eigenmodes.
    """

    S = np.asarray(S, dtype=np.float64)
    if S.ndim != 2 or int(S.shape[0]) != int(S.shape[1]):
        raise ValueError("S must be a square 2D matrix")
    n = int(S.shape[0])
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    ortho_cutoff = float(ortho_cutoff)
    if not np.isfinite(ortho_cutoff) or ortho_cutoff <= 0.0:
        raise ValueError("ortho_cutoff must be finite and > 0")

    w, U = np.linalg.eigh(0.5 * (S + S.T))
    w = np.asarray(w, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    wmax = float(np.max(w)) if int(w.size) else 0.0
    if wmax <= 0.0:
        return np.zeros((n, 0), dtype=np.float64)
    keep = w > (ortho_cutoff * wmax)
    if not bool(np.any(keep)):
        return np.zeros((n, 0), dtype=np.float64)
    wk = w[keep]
    Uk = U[:, keep]
    W = Uk / np.sqrt(wk)[None, :]
    return np.ascontiguousarray(W)


def _coords_key(xyz: np.ndarray) -> tuple[float, float, float]:
    x, y, z = map(float, np.asarray(xyz, dtype=np.float64).reshape((3,)))
    return (round(x, 12), round(y, 12), round(z, 12))


def _map_shells_to_atoms(shell_cxyz: np.ndarray, atom_coords_bohr: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    """Map basis shells to atoms by matching center coordinates (Bohr)."""

    shell_cxyz = np.asarray(shell_cxyz, dtype=np.float64).reshape((-1, 3))
    atom_coords_bohr = np.asarray(atom_coords_bohr, dtype=np.float64).reshape((-1, 3))
    natm = int(atom_coords_bohr.shape[0])
    nshell = int(shell_cxyz.shape[0])
    if natm <= 0:
        raise ValueError("no atoms")

    atom_map: dict[tuple[float, float, float], int] = {_coords_key(atom_coords_bohr[i]): i for i in range(natm)}
    shell_to_atom = np.empty((nshell,), dtype=np.int32)
    atom_to_shells: list[list[int]] = [[] for _ in range(natm)]

    for ish in range(nshell):
        key = _coords_key(shell_cxyz[ish])
        ia = atom_map.get(key)
        if ia is None:
            d2 = np.sum((atom_coords_bohr - shell_cxyz[ish][None, :]) ** 2, axis=1)
            ia = int(np.argmin(d2))
            if float(d2[ia]) > 1e-16:
                raise ValueError("failed to map shell centers to atom coordinates (unexpected basis centers)")
        shell_to_atom[ish] = int(ia)
        atom_to_shells[int(ia)].append(int(ish))

    return shell_to_atom, atom_to_shells


def _radial_shell_value(basis: BasisCartSoA, shell: int, rho: np.ndarray) -> np.ndarray:
    """Return radial part χ_shell(ρ) = ρ^l Σ_i c_i exp(-a_i ρ^2)."""

    shell = int(shell)
    rho = np.asarray(rho, dtype=np.float64).ravel()
    if int(rho.size) == 0:
        return rho.copy()

    l = int(np.asarray(basis.shell_l, dtype=np.int32).ravel()[shell])
    p0 = int(np.asarray(basis.shell_prim_start, dtype=np.int32).ravel()[shell])
    npg = int(np.asarray(basis.shell_nprim, dtype=np.int32).ravel()[shell])
    exps = np.asarray(basis.prim_exp[p0 : p0 + npg], dtype=np.float64).ravel()
    coefs = np.asarray(basis.prim_coef[p0 : p0 + npg], dtype=np.float64).ravel()

    r2 = rho * rho
    rad = np.zeros((int(rho.size),), dtype=np.float64)
    for a, c in zip(exps.tolist(), coefs.tolist()):
        rad += float(c) * np.exp(-float(a) * r2)
    if l > 0:
        rad *= rho**int(l)
    return rad


def _rdvr_radial_nodes_weights(
    basis: BasisCartSoA,
    shells: list[int],
    *,
    radial_rmax: float,
    ortho_cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute R-DVR radial nodes/weights for one atom."""

    shells = [int(s) for s in shells]
    n = int(len(shells))
    if n <= 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    shell_l = np.asarray(basis.shell_l, dtype=np.int32).ravel()
    shell_prim_start = np.asarray(basis.shell_prim_start, dtype=np.int32).ravel()
    shell_nprim = np.asarray(basis.shell_nprim, dtype=np.int32).ravel()
    prim_exp = np.asarray(basis.prim_exp, dtype=np.float64).ravel()
    prim_coef = np.asarray(basis.prim_coef, dtype=np.float64).ravel()

    S = np.zeros((n, n), dtype=np.float64)
    Rprime = np.zeros((n, n), dtype=np.float64)

    # Build radial moment matrices in the shell-radial basis.
    for i, shA in enumerate(shells):
        la = int(shell_l[int(shA)])
        p0a = int(shell_prim_start[int(shA)])
        npa = int(shell_nprim[int(shA)])
        exps_a = prim_exp[p0a : p0a + npa]
        coefs_a = prim_coef[p0a : p0a + npa]
        for j, shB in enumerate(shells[: i + 1]):
            lb = int(shell_l[int(shB)])
            p0b = int(shell_prim_start[int(shB)])
            npb = int(shell_nprim[int(shB)])
            exps_b = prim_exp[p0b : p0b + npb]
            coefs_b = prim_coef[p0b : p0b + npb]

            nS = 2 + la + lb
            nR = 3 + la + lb
            s = 0.0
            r = 0.0
            for a, ca in zip(exps_a.tolist(), coefs_a.tolist()):
                a = float(a)
                ca = float(ca)
                for b, cb in zip(exps_b.tolist(), coefs_b.tolist()):
                    p = a + float(b)
                    c = ca * float(cb)
                    s += c * _gauss_int(nS, p)
                    r += c * _gauss_int(nR, p)

            S[i, j] = s
            S[j, i] = s
            Rprime[i, j] = r
            Rprime[j, i] = r

    W = _canonical_orth(S, ortho_cutoff=float(ortho_cutoff))
    if int(W.shape[1]) == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    R = W.T @ Rprime @ W
    R = 0.5 * (R + R.T)
    rho, Q = np.linalg.eigh(R)
    rho = np.asarray(rho, dtype=np.float64).ravel()
    Q = np.asarray(Q, dtype=np.float64)

    # Sort nodes by radius.
    order = np.argsort(rho, kind="stable")
    rho = rho[order]
    Q = Q[:, order]

    # Filter invalid / far nodes.
    radial_rmax = float(radial_rmax)
    if not np.isfinite(radial_rmax) or radial_rmax <= 0.0:
        radial_rmax = float("inf")
    keep = (rho > 0.0) & np.isfinite(rho) & (rho <= radial_rmax)
    if not bool(np.any(keep)):
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    rho = rho[keep]
    Qk = Q[:, keep]  # (m, nnode)

    # Evaluate radial basis at nodes.
    chi = np.empty((int(rho.size), n), dtype=np.float64)
    for col, sh in enumerate(shells):
        chi[:, col] = _radial_shell_value(basis, int(sh), rho)

    psi = chi @ W  # (nnode, m)
    # Only need ξ_P(r_P) (diagonal of psi @ Qk): ξ_diag = sum_B psi[P,B] * Qk[B,P]
    xi_diag = np.sum(psi * Qk.T, axis=1)
    denom = xi_diag * xi_diag
    with np.errstate(divide="ignore", invalid="ignore"):
        w_r = (rho * rho) / denom

    mask = np.isfinite(w_r) & (w_r > 0.0)
    if not bool(np.any(mask)):
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    return np.ascontiguousarray(rho[mask]), np.ascontiguousarray(w_r[mask])


def iter_rdvr_grid(
    mol_or_coords: Any,
    dvr_basis: BasisCartSoA,
    *,
    angular_n: int = 302,
    angular_kind: str = "auto",
    radial_rmax: float = 20.0,
    becke_n: int = 3,
    block_size: int = 20000,
    prune_tol: float = 1e-16,
    ortho_cutoff: float = 1e-10,
    angular_prune: bool = True,
    atom_Z: Any | None = None,
    return_batch: bool = False,
) -> Iterator[Any]:
    """Yield (points, weights) blocks for an R-DVR atom-centered grid.

    If ``angular_prune=True`` (default), the angular grid order is selected
    *per radial node* using the heuristic Bragg-Slater envelope from Parrish
    et al. (JCP 138, 194107 (2013)). In this mode, ``angular_n`` is treated as
    the *maximum* Lebedev grid size (in points) allowed at any radial node.
    """

    R = np.asarray(_coords_bohr(mol_or_coords), dtype=np.float64).reshape((-1, 3))
    natm = int(R.shape[0])
    if natm <= 0:
        raise ValueError("no atoms")

    angular_n_max = int(angular_n)
    if angular_n_max <= 0:
        raise ValueError("angular_n must be > 0")
    block_size = max(1, int(block_size))
    prune_tol = float(prune_tol)
    becke_n = int(becke_n)
    if becke_n < 0:
        raise ValueError("becke_n must be >= 0")

    angular_prune = bool(angular_prune)
    angular_kind_s = str(angular_kind).strip().lower()
    return_batch = bool(return_batch)

    # Angular rule: either shared (no pruning) or cached per selected Lebedev order.
    ang_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    lmax_max: int | None = None
    rho_bs: np.ndarray | None = None
    if angular_prune:
        if angular_kind_s in {"fibonacci", "fib", "fibo"}:
            raise ValueError("angular_prune=True requires Lebedev grids (use angular_kind='auto' or 'lebedev')")
        lmax_max = _lebedev_lmax_from_npts(int(angular_n_max))
        if lmax_max is None:
            raise ValueError(
                f"angular_prune=True requires angular_n to be a supported Lebedev point count. Got angular_n={int(angular_n_max)}."
            )

        if atom_Z is None:
            Z = _atomic_numbers_or_none(mol_or_coords)
        else:
            Z = np.asarray(atom_Z, dtype=np.int32).ravel()
        if Z is None or int(Z.size) != int(natm):
            raise ValueError(
                "angular_prune=True requires atomic numbers. Pass a Molecule with atoms_bohr or provide atom_Z (len=natm)."
            )

        Z = np.asarray(Z, dtype=np.int32).ravel()
        if int(np.min(Z)) < 1 or int(np.max(Z)) >= int(_BRAGG_SLATER_RADII_BOHR.size):
            raise ValueError("unsupported atomic number for Bragg-Slater pruning")
        rho_bs = np.ascontiguousarray(_BRAGG_SLATER_RADII_BOHR[Z])
    else:
        dirs, wang = angular_grid(int(angular_n_max), kind=str(angular_kind))
        dirs = np.asarray(dirs, dtype=np.float64).reshape((-1, 3))
        wang = np.asarray(wang, dtype=np.float64).ravel()
        if int(dirs.shape[0]) != int(wang.shape[0]):
            raise RuntimeError("angular grid returned inconsistent shapes")
        ang_cache[int(angular_n_max)] = (dirs, wang)

    # Precompute inter-atomic distances for Becke partitioning.
    dAB = R[:, None, :] - R[None, :, :]
    RAB = np.linalg.norm(dAB, axis=2)

    # Shell assignment by atom.
    _shell_to_atom, atom_to_shells = _map_shells_to_atoms(np.asarray(dvr_basis.shell_cxyz), R)

    for ia in range(natm):
        shells = atom_to_shells[int(ia)]
        rho, wr = _rdvr_radial_nodes_weights(
            dvr_basis,
            shells,
            radial_rmax=float(radial_rmax),
            ortho_cutoff=float(ortho_cutoff),
        )
        if int(rho.size) == 0:
            continue

        pts_buf: list[np.ndarray] = []
        w_buf: list[np.ndarray] = []
        rid_buf: list[np.ndarray] = []
        nang_buf: list[np.ndarray] = []
        n_buf = 0

        center = np.asarray(R[int(ia)], dtype=np.float64).reshape((3,))
        rho_bs_i = float(rho_bs[int(ia)]) if rho_bs is not None else 0.0

        for inode, (rnode, wnode) in enumerate(zip(rho.tolist(), wr.tolist())):
            if angular_prune:
                assert lmax_max is not None  # for type-checkers
                lreq = _pruned_lreq_parrish_2013(float(rnode), rho_bs=rho_bs_i, lmax_max=int(lmax_max))
                nang = _lebedev_npts_for_lreq(int(lreq), lmax_max=int(lmax_max))
            else:
                nang = int(angular_n_max)

            ang = ang_cache.get(int(nang))
            if ang is None:
                dirs_i, wang_i = angular_grid(int(nang), kind=str(angular_kind))
                dirs_i = np.asarray(dirs_i, dtype=np.float64).reshape((-1, 3))
                wang_i = np.asarray(wang_i, dtype=np.float64).ravel()
                if int(dirs_i.shape[0]) != int(wang_i.shape[0]):
                    raise RuntimeError("angular grid returned inconsistent shapes")
                ang = (dirs_i, wang_i)
                ang_cache[int(nang)] = ang
            dirs_i, wang_i = ang

            pts_node = center[None, :] + float(rnode) * dirs_i  # (nang,3)
            w_node = float(wnode) * wang_i  # (nang,)
            pts_buf.append(pts_node)
            w_buf.append(w_node)
            if return_batch:
                rid_buf.append(np.full((int(w_node.size),), int(inode), dtype=np.int32))
                nang_buf.append(np.full((int(w_node.size),), int(nang), dtype=np.int32))
            n_buf += int(w_node.size)

            if n_buf >= int(block_size):
                pts_blk = np.concatenate(pts_buf, axis=0) if len(pts_buf) else np.zeros((0, 3), dtype=np.float64)
                w_blk = np.concatenate(w_buf, axis=0) if len(w_buf) else np.zeros((0,), dtype=np.float64)
                rid_blk = np.concatenate(rid_buf, axis=0) if len(rid_buf) else np.zeros((0,), dtype=np.int32)
                nang_blk = np.concatenate(nang_buf, axis=0) if len(nang_buf) else np.zeros((0,), dtype=np.int32)
                pts_buf.clear()
                w_buf.clear()
                rid_buf.clear()
                nang_buf.clear()
                n_buf = 0

                if int(w_blk.size):
                    wpart = becke_partition_weights(pts_blk, R, becke_n=int(becke_n), RAB=RAB)
                    w_mol = w_blk * wpart[:, int(ia)]
                    if prune_tol > 0.0:
                        mask = w_mol > prune_tol
                        pts_blk = pts_blk[mask]
                        w_mol = w_mol[mask]
                        if return_batch:
                            rid_blk = rid_blk[mask]
                            nang_blk = nang_blk[mask]
                    if int(w_mol.size):
                        if return_batch:
                            yield GridBatch(
                                points=np.ascontiguousarray(pts_blk),
                                weights=np.ascontiguousarray(w_mol),
                                point_atom=np.full((int(w_mol.size),), int(ia), dtype=np.int32),
                                point_radial_index=np.ascontiguousarray(rid_blk),
                                point_angular_n=np.ascontiguousarray(nang_blk),
                                meta={"grid_kind": "rdvr", "atom": int(ia), "angular_prune": bool(angular_prune)},
                            )
                        else:
                            yield np.ascontiguousarray(pts_blk), np.ascontiguousarray(w_mol)

        # Flush remainder for this atom.
        if n_buf:
            pts_blk = np.concatenate(pts_buf, axis=0) if len(pts_buf) else np.zeros((0, 3), dtype=np.float64)
            w_blk = np.concatenate(w_buf, axis=0) if len(w_buf) else np.zeros((0,), dtype=np.float64)
            rid_blk = np.concatenate(rid_buf, axis=0) if len(rid_buf) else np.zeros((0,), dtype=np.int32)
            nang_blk = np.concatenate(nang_buf, axis=0) if len(nang_buf) else np.zeros((0,), dtype=np.int32)
            if int(w_blk.size):
                wpart = becke_partition_weights(pts_blk, R, becke_n=int(becke_n), RAB=RAB)
                w_mol = w_blk * wpart[:, int(ia)]
                if prune_tol > 0.0:
                    mask = w_mol > prune_tol
                    pts_blk = pts_blk[mask]
                    w_mol = w_mol[mask]
                    if return_batch:
                        rid_blk = rid_blk[mask]
                        nang_blk = nang_blk[mask]
                if int(w_mol.size):
                    if return_batch:
                        yield GridBatch(
                            points=np.ascontiguousarray(pts_blk),
                            weights=np.ascontiguousarray(w_mol),
                            point_atom=np.full((int(w_mol.size),), int(ia), dtype=np.int32),
                            point_radial_index=np.ascontiguousarray(rid_blk),
                            point_angular_n=np.ascontiguousarray(nang_blk),
                            meta={"grid_kind": "rdvr", "atom": int(ia), "angular_prune": bool(angular_prune)},
                        )
                    else:
                        yield np.ascontiguousarray(pts_blk), np.ascontiguousarray(w_mol)


def make_rdvr_grid(
    mol_or_coords: Any,
    dvr_basis: BasisCartSoA,
    *,
    angular_n: int = 302,
    angular_kind: str = "auto",
    radial_rmax: float = 20.0,
    becke_n: int = 3,
    prune_tol: float = 1e-16,
    ortho_cutoff: float = 1e-10,
    angular_prune: bool = True,
    atom_Z: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize a full R-DVR grid as NumPy arrays."""

    pts_list: list[np.ndarray] = []
    w_list: list[np.ndarray] = []
    for pts, w in iter_rdvr_grid(
        mol_or_coords,
        dvr_basis,
        angular_n=int(angular_n),
        angular_kind=str(angular_kind),
        radial_rmax=float(radial_rmax),
        becke_n=int(becke_n),
        block_size=10**9,
        prune_tol=float(prune_tol),
        ortho_cutoff=float(ortho_cutoff),
        angular_prune=bool(angular_prune),
        atom_Z=atom_Z,
    ):
        pts_list.append(pts)
        w_list.append(w)
    if len(pts_list) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    return np.concatenate(pts_list, axis=0), np.concatenate(w_list, axis=0)


def make_rdvr_grid_device(*args: Any, **kwargs: Any):
    """Materialize a full R-DVR grid on the GPU as CuPy arrays."""

    try:
        import cupy as cp  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("make_rdvr_grid_device requires CuPy") from e

    pts, w = make_rdvr_grid(*args, **kwargs)
    return cp.ascontiguousarray(cp.asarray(pts, dtype=cp.float64)), cp.ascontiguousarray(cp.asarray(w, dtype=cp.float64).ravel())


def _joint_diag_jacobi(
    mats: list[np.ndarray],
    *,
    max_sweeps: int,
    tol: float,
) -> np.ndarray:
    """Approximate simultaneous diagonalization of symmetric matrices (Jacobi sweeps).

    This follows the general strategy referenced by Parrish et al. (2013) for F-DVR:
    iteratively apply Givens rotations to reduce the sum of squared off-diagonal
    elements across all matrices.

    Parameters
    ----------
    mats
        List of symmetric (m,m) matrices. Updated in-place.

    Returns
    -------
    Q : (m,m)
        Accumulated orthogonal transformation such that (approximately):
          A_k' = Q^T A_k Q
    """

    if len(mats) == 0:
        return np.zeros((0, 0), dtype=np.float64)

    m = int(mats[0].shape[0])
    if any(A.shape != (m, m) for A in mats):
        raise ValueError("all mats must have identical square shape (m,m)")

    max_sweeps = int(max_sweeps)
    tol = float(tol)
    if max_sweeps <= 0:
        raise ValueError("max_sweeps must be > 0")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and > 0")

    Q = np.eye(m, dtype=np.float64)

    # Stop criterion on total off-diagonal Frobenius norm.
    tol2 = float(tol * tol)
    for _sweep in range(max_sweeps):
        off2 = 0.0
        for A in mats:
            diag2 = float(np.sum(np.diag(A) ** 2))
            off2 += float(np.sum(A * A) - diag2)
        if off2 < tol2:
            break

        for p in range(m - 1):
            for q in range(p + 1, m):
                # Cardoso-Souloumiac style closed-form Jacobi angle:
                # Minimize sum_k (a_k[p,q]')^2 where
                #   a_k[p,q]' = u_k * cos(2phi) + v_k * sin(2phi)
                # with u_k = a_k[p,q], v_k = 0.5*(a_k[p,p] - a_k[q,q]).
                g11 = 0.0  # sum u^2
                g12 = 0.0  # sum u*v
                g22 = 0.0  # sum v^2
                for A in mats:
                    app = float(A[p, p])
                    aqq = float(A[q, q])
                    apq = float(A[p, q])
                    u = apq
                    v = 0.5 * (app - aqq)
                    g11 += u * u
                    g12 += u * v
                    g22 += v * v

                if g12 == 0.0 and g11 == g22:
                    continue

                # Smallest-eigenvalue eigenvector of [[g11,g12],[g12,g22]]:
                # t = [cos(2phi), sin(2phi)].
                theta = 0.5 * float(np.arctan2(2.0 * g12, g11 - g22))
                cth = float(np.cos(theta))
                sth = float(np.sin(theta))
                c2 = -sth
                s2 = cth
                phi = 0.5 * float(np.arctan2(s2, c2))
                # Prefer the small-angle solution (equivalent up to +/- pi/2).
                if phi > 0.25 * np.pi:
                    phi -= 0.5 * np.pi
                elif phi < -0.25 * np.pi:
                    phi += 0.5 * np.pi
                c = float(np.cos(phi))
                s = float(np.sin(phi))
                if abs(s) < 1e-15:
                    continue

                # Accumulate Q <- Q G where G acts on (p,q) columns.
                Qp = Q[:, p].copy()
                Qq = Q[:, q].copy()
                Q[:, p] = c * Qp - s * Qq
                Q[:, q] = s * Qp + c * Qq

                # Apply congruence update A <- G^T A G, preserving symmetry.
                for A in mats:
                    app = float(A[p, p])
                    aqq = float(A[q, q])
                    apq = float(A[p, q])

                    # Update k != p,q entries.
                    for k in range(m):
                        if k == p or k == q:
                            continue
                        aik = float(A[k, p])
                        akq = float(A[k, q])
                        vkp = c * aik - s * akq
                        vkq = s * aik + c * akq
                        A[k, p] = vkp
                        A[p, k] = vkp
                        A[k, q] = vkq
                        A[q, k] = vkq

                    # Update 2x2 minor.
                    A[p, p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
                    A[q, q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
                    A[p, q] = (c * c - s * s) * apq + s * c * (app - aqq)
                    A[q, p] = A[p, q]

    return Q


def make_fdvr_grid(
    dvr_basis: BasisCartSoA,
    *,
    ortho_cutoff: float = 1e-10,
    max_sweeps: int = 50,
    tol: float = 1e-12,
    prune_tol: float = 1e-16,
    validate: bool = True,
    overlap_max_abs_tol: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct an F-DVR grid (nodes + weights) as NumPy arrays.

    Notes
    -----
    F-DVR relies on *approximate* simultaneous diagonalization of the finite-basis
    position operator, and is sensitive to the choice of DVR basis. If
    ``validate=True`` (default), we validate the resulting quadrature against the
    analytic overlap matrix for ``dvr_basis`` and raise if it is not sufficiently
    accurate.
    """

    from asuka.integrals.int1e_dipole_cart import build_overlap_and_dipole_cart  # noqa: PLC0415
    from asuka.orbitals.eval_cart import eval_basis_cart_value_on_points  # noqa: PLC0415

    S, Rx, Ry, Rz = build_overlap_and_dipole_cart(dvr_basis)
    W = _canonical_orth(S, ortho_cutoff=float(ortho_cutoff))
    if int(W.shape[1]) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    X = W.T @ Rx @ W
    Y = W.T @ Ry @ W
    Z = W.T @ Rz @ W
    X = 0.5 * (X + X.T)
    Y = 0.5 * (Y + Y.T)
    Z = 0.5 * (Z + Z.T)

    mats = [X, Y, Z]
    Q = _joint_diag_jacobi(mats, max_sweeps=int(max_sweeps), tol=float(tol))

    pts = np.stack([np.diag(X), np.diag(Y), np.diag(Z)], axis=1).astype(np.float64, copy=False)

    # Compute weights from the interpolating zeros property:
    #   w_P = 1 / (xi_P(r_P))^2, where xi = psi @ Q and psi = chi @ W.
    chi = eval_basis_cart_value_on_points(dvr_basis, pts)  # (m, nfull)
    psi = chi @ W  # (m, m)
    xi_diag = np.sum(psi * Q.T, axis=1)
    denom = xi_diag * xi_diag
    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / denom

    prune_tol = float(prune_tol)
    mask = np.isfinite(w) & (w > max(prune_tol, 0.0)) & np.all(np.isfinite(pts), axis=1)
    if not bool(np.any(mask)):
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    pts = np.ascontiguousarray(pts[mask])
    w = np.ascontiguousarray(w[mask])

    if bool(validate):
        # Validate that the quadrature reproduces the overlap metric for the DVR basis:
        #   S_AB ~= sum_P w_P * chi_A(r_P) * chi_B(r_P)
        chi_v = np.ascontiguousarray(eval_basis_cart_value_on_points(dvr_basis, pts), dtype=np.float64)
        S_quad = chi_v.T @ (chi_v * w[:, None])
        S_ref = np.asarray(S, dtype=np.float64)
        err = float(np.max(np.abs(S_quad - S_ref)))
        if not np.isfinite(err) or err > float(overlap_max_abs_tol):
            raise RuntimeError(
                "F-DVR grid failed overlap-quadrature validation "
                f"(max|S_quad-S|={err:.3e} > {float(overlap_max_abs_tol):.3e}). "
                "Try a different DVR basis, looser canonical-orth cutoff, or use grid_kind='rdvr'/'becke'. "
                "To bypass this check for experimentation, pass validate=False."
            )

    return pts, w


def iter_fdvr_grid(
    dvr_basis: BasisCartSoA,
    *,
    block_size: int = 20_000,
    ortho_cutoff: float = 1e-10,
    max_sweeps: int = 50,
    tol: float = 1e-12,
    prune_tol: float = 1e-16,
    validate: bool = True,
    overlap_max_abs_tol: float = 1e-3,
    return_batch: bool = False,
) -> Iterator[Any]:
    """Iterator wrapper for F-DVR so callers can treat it like other grids.

    Phase 1 still materializes the full F-DVR grid internally, then yields
    slices. That is sufficient to unify the public API.
    """

    pts, w = make_fdvr_grid(
        dvr_basis,
        ortho_cutoff=float(ortho_cutoff),
        max_sweeps=int(max_sweeps),
        tol=float(tol),
        prune_tol=float(prune_tol),
        validate=bool(validate),
        overlap_max_abs_tol=float(overlap_max_abs_tol),
    )

    block_size = max(1, int(block_size))
    return_batch = bool(return_batch)
    for p0 in range(0, int(w.size), int(block_size)):
        p1 = min(int(w.size), p0 + int(block_size))
        pts_blk = np.ascontiguousarray(pts[p0:p1])
        w_blk = np.ascontiguousarray(w[p0:p1])
        if return_batch:
            yield GridBatch(points=pts_blk, weights=w_blk, meta={"grid_kind": "fdvr"})
        else:
            yield pts_blk, w_blk


def make_fdvr_grid_device(*args: Any, **kwargs: Any):
    """Materialize an F-DVR grid on the GPU as CuPy arrays."""

    try:
        import cupy as cp  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("make_fdvr_grid_device requires CuPy") from e

    pts, w = make_fdvr_grid(*args, **kwargs)
    return cp.ascontiguousarray(cp.asarray(pts, dtype=cp.float64)), cp.ascontiguousarray(cp.asarray(w, dtype=cp.float64).ravel())


__all__ = [
    "iter_rdvr_grid",
    "make_rdvr_grid",
    "make_rdvr_grid_device",
    "make_fdvr_grid",
    "iter_fdvr_grid",
    "make_fdvr_grid_device",
]
