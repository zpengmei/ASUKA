from __future__ import annotations

"""Cached DF/Cholesky AO factors and AO->MO transforms.

This module replaces the historical outcore-backed DF "context" with a
cuERI-based implementation:

- AO DF factors are built via cuERI (CPU) as whitened 3-index tensors
  ``B[μ,ν,Q]`` such that ``(μν|λσ) ~= Σ_Q B[μν,Q] B[λσ,Q]``.
- MO transforms are performed in-memory using NumPy (BLAS-backed) contractions.

The public API exposes:

  - `get_df_cholesky_context(...)`
  - `DFCholeskyContext.transform(...)`
  - `DFCholeskyContext.transform_many(...)`

Notes
-----
- This module intentionally contains no external runtime dependencies. It can
  accept a Mole-like object from other libraries without importing them, by
  relying only on mol-like introspection methods.
- For large AO bases, storing `B[μ,ν,Q]` in memory can be expensive. This
  implementation prioritizes compatibility and clarity over peak throughput.
"""

import hashlib
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np


def _mol_fingerprint(mol: Any) -> str:
    """Return a stable fingerprint for a Mole-like object."""

    dumps = getattr(mol, "dumps", None)
    if callable(dumps):
        payload = dumps()
        if not isinstance(payload, str):
            payload = str(payload)
    else:
        as_dict = getattr(mol, "as_dict", None)
        if callable(as_dict):
            payload = repr(as_dict())
        else:
            payload = repr(mol)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def _hash_f64_array(a: Any) -> tuple[bytes, tuple[int, ...]]:
    arr = np.asarray(a, dtype=np.float64)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    h = hashlib.blake2b(memoryview(arr), digest_size=16)
    shape = tuple(int(x) for x in arr.shape)
    return h.digest(), shape


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if not val:
        return int(default)
    try:
        return int(val)
    except Exception:
        return int(default)


def _atoms_bohr_from_mol_like(mol: Any) -> list[tuple[str, np.ndarray]]:
    """Extract [(sym, xyz_bohr), ...] from a Mole-like object without external imports."""

    try:
        from asuka.frontend.molecule import Molecule  # noqa: PLC0415
    except Exception:  # pragma: no cover
        Molecule = None  # type: ignore[assignment]

    if Molecule is not None and isinstance(mol, Molecule):
        return [(str(sym), np.asarray(xyz, dtype=np.float64).reshape((3,))) for sym, xyz in mol.atoms_bohr]

    natm = getattr(mol, "natm", None)
    atom_symbol = getattr(mol, "atom_symbol", None)
    atom_coord = getattr(mol, "atom_coord", None)
    if natm is None or not callable(atom_symbol) or not callable(atom_coord):
        raise TypeError(
            "mol must be an asuka.frontend.molecule.Molecule or a Mole-like object with "
            "natm/atom_symbol(i)/atom_coord(i)"
        )

    atoms: list[tuple[str, np.ndarray]] = []
    for i in range(int(natm)):
        sym_raw = str(atom_symbol(int(i)))
        # Some upstream sources (e.g. Molden imports) may label atoms as "N1", "C2", etc.
        # Normalize to a plain element symbol for basis lookups / aux basis dict keys.
        m = re.match(r"^([A-Za-z]{1,2})", sym_raw.strip())
        sym = (m.group(1) if m is not None else sym_raw).capitalize()
        xyz = np.asarray(atom_coord(int(i)), dtype=np.float64).reshape((3,))
        atoms.append((sym, xyz))
    return atoms


def _unique_elements(atoms_bohr: list[tuple[str, np.ndarray]]) -> list[str]:
    return sorted({sym for sym, _xyz in atoms_bohr})


def _build_aux_basis_cart_from_name(
    *,
    auxbasis_name: str,
    elements: list[str],
    orbital_basis_name: str | None,
) -> tuple[str, dict[str, list[tuple[int, np.ndarray, np.ndarray]]]]:
    """Return (resolved_aux_name, aux_shells_by_element) from a string spec."""

    from asuka.frontend.basis_bse import load_autoaux_shells, load_basis_shells  # noqa: PLC0415

    name = str(auxbasis_name).strip()
    norm = name.lower()

    if norm in ("auto", "autoaux"):
        if not orbital_basis_name:
            raise ValueError("auxbasis='autoaux' requires a string orbital basis name")
        aux_name, aux_shells = load_autoaux_shells(str(orbital_basis_name), elements=elements)
        return str(aux_name), aux_shells

    try:
        aux_shells = load_basis_shells(name, elements=elements)
        return str(name), aux_shells
    except Exception as e_bse:
        # Basis Set Exchange does not always expose fitted aux bases as standalone
        # names. Treat common JKFIT-like names as aliases for BSE autoaux.
        if orbital_basis_name:
            base = str(name).strip()
            for suf in ("-jkfit", "-jfit", "-rifit", "-ri", "-mp2fit"):
                if base.lower().endswith(suf):
                    base = base[: -len(suf)]
                    break
            base = base or str(orbital_basis_name)
            try:
                aux_name, aux_shells = load_autoaux_shells(str(base), elements=elements)
                return str(aux_name), aux_shells
            except Exception:
                pass

        # Legacy PySCF-style universal aux basis names are not guaranteed to exist
        # in BSE. For the common "weigend(+etb)" family, PySCF's basis library is
        # identical to BSE's `def2-universal-jfit`, so treat it as an alias to
        # preserve parity without requiring PySCF.
        if norm in ("weigend+etb", "weigend", "etb"):
            alias = "def2-universal-jfit"
            aux_shells = load_basis_shells(alias, elements=elements)
            return alias, aux_shells

        raise RuntimeError(
            f"failed to load auxiliary basis '{name}' via Basis Set Exchange. "
            "ASUKA no longer falls back to PySCF's basis library. "
            "Install `basis_set_exchange` (e.g. `pip install asuka[frontend]`) "
            "or provide an explicit per-element aux basis dict."
        ) from e_bse


def _build_df_bases_cart(mol: Any, *, auxbasis: Any, expand_contractions: bool) -> tuple[Any, Any, str]:
    """Return (ao_basis, aux_basis, auxbasis_name) as cuERI packed cart bases."""

    from asuka.cueri.mol_basis import pack_cart_shells_from_mol  # noqa: PLC0415
    from asuka.frontend.basis_packer import pack_cart_basis, parse_pyscf_basis_dict  # noqa: PLC0415

    atoms_bohr = _atoms_bohr_from_mol_like(mol)
    elements = _unique_elements(atoms_bohr)

    # AO basis: prefer direct packing from a PySCF-like mol when possible, since
    # it preserves exactly the basis as used by the upstream SCF.
    ao_basis = None
    if bool(getattr(mol, "cart", True)) and hasattr(mol, "nbas") and callable(getattr(mol, "bas_exp", None)):
        ao_basis = pack_cart_shells_from_mol(mol, expand_contractions=bool(expand_contractions))
        orbital_basis_name = getattr(mol, "basis", None)
        orbital_basis_name = str(orbital_basis_name) if isinstance(orbital_basis_name, str) else None
    else:
        # asuka.frontend.molecule.Molecule path
        orbital_basis_name = getattr(mol, "basis", None)
        if isinstance(orbital_basis_name, str):
            from asuka.frontend.basis_bse import load_basis_shells  # noqa: PLC0415

            ao_shells = load_basis_shells(str(orbital_basis_name), elements=elements)
        elif isinstance(orbital_basis_name, dict):
            ao_shells = parse_pyscf_basis_dict(orbital_basis_name, elements=elements)
        else:
            raise TypeError("mol.basis must be a string name or an explicit per-element basis dict")
        ao_basis = pack_cart_basis(atoms_bohr, ao_shells, expand_contractions=bool(expand_contractions))

    # Aux basis
    if hasattr(auxbasis, "shell_cxyz") and hasattr(auxbasis, "shell_l") and hasattr(auxbasis, "prim_exp"):
        aux_basis = auxbasis
        auxbasis_name = "<packed>"
    elif isinstance(auxbasis, dict):
        aux_shells = parse_pyscf_basis_dict(auxbasis, elements=elements)
        aux_basis = pack_cart_basis(atoms_bohr, aux_shells, expand_contractions=bool(expand_contractions))
        auxbasis_name = "<explicit>"
    elif isinstance(auxbasis, str):
        auxbasis_name, aux_shells = _build_aux_basis_cart_from_name(
            auxbasis_name=str(auxbasis),
            elements=elements,
            orbital_basis_name=orbital_basis_name,
        )
        aux_basis = pack_cart_basis(atoms_bohr, aux_shells, expand_contractions=bool(expand_contractions))
    else:
        raise TypeError("auxbasis must be a string name, an explicit basis dict, or a packed aux basis object")

    return ao_basis, aux_basis, str(auxbasis_name)


@dataclass
class DFCholeskyContext:
    """Cached AO DF tensor for repeated MO transforms (cuERI-backed)."""

    mol_fingerprint: str
    auxbasis: Any
    auxbasis_name: str
    ao_basis: Any
    aux_basis: Any
    B_ao: np.ndarray  # (nao, nao, naux)
    max_memory: int
    verbose: int

    _mo_cache: "OrderedDict[tuple[Any, ...], np.ndarray]"
    _mo_cache_bytes: int
    _mo_cache_max_bytes: int
    _mo_cache_max_entry_bytes: int

    @classmethod
    def build(
        cls,
        mol: Any,
        *,
        auxbasis: Any = "autoaux",
        max_memory: int = 2000,
        verbose: int = 0,
        expand_contractions: bool = True,
        threads: int = 0,
    ) -> "DFCholeskyContext":
        mol_fp = _mol_fingerprint(mol)
        ao_basis, aux_basis, auxbasis_name = _build_df_bases_cart(
            mol,
            auxbasis=auxbasis,
            expand_contractions=bool(expand_contractions),
        )

        from asuka.integrals.cueri_df_cpu import build_df_B_from_cueri_packed_bases_cpu  # noqa: PLC0415

        B_ao = build_df_B_from_cueri_packed_bases_cpu(ao_basis, aux_basis, threads=int(threads))
        B_ao = np.asarray(B_ao, dtype=np.float64, order="C")

        # If spherical AOs requested, transform B to spherical basis.
        if not bool(getattr(mol, "cart", True)):
            from asuka.integrals.cart2sph import (  # noqa: PLC0415
                build_cart2sph_matrix,
                compute_sph_layout_from_cart_basis,
                transform_df_B_cart_to_sph,
            )

            shell_l = np.asarray(ao_basis.shell_l, dtype=np.int32).ravel()
            shell_ao_start_cart = np.asarray(ao_basis.shell_ao_start, dtype=np.int32).ravel()
            shell_ao_start_sph, nao_sph = compute_sph_layout_from_cart_basis(ao_basis)
            nao_cart = int(B_ao.shape[0])
            T = build_cart2sph_matrix(shell_l, shell_ao_start_cart, shell_ao_start_sph, nao_cart, nao_sph)
            B_ao = transform_df_B_cart_to_sph(B_ao, T)
            B_ao = np.asarray(B_ao, dtype=np.float64, order="C")

        max_bytes = _env_int("CUGUGA_DF_MO_CACHE_BYTES", 512 * 1024 * 1024)
        max_entry = _env_int("CUGUGA_DF_MO_CACHE_ENTRY_BYTES", 256 * 1024 * 1024)
        return cls(
            mol_fingerprint=mol_fp,
            auxbasis=auxbasis,
            auxbasis_name=str(auxbasis_name),
            ao_basis=ao_basis,
            aux_basis=aux_basis,
            B_ao=B_ao,
            max_memory=int(max_memory),
            verbose=int(verbose),
            _mo_cache=OrderedDict(),
            _mo_cache_bytes=0,
            _mo_cache_max_bytes=max(0, int(max_bytes)),
            _mo_cache_max_entry_bytes=max(0, int(max_entry)),
        )

    def _cache_get(self, key: tuple[Any, ...]) -> np.ndarray | None:
        hit = self._mo_cache.get(key)
        if hit is None:
            return None
        self._mo_cache.move_to_end(key)
        return hit

    def _cache_put(self, key: tuple[Any, ...], val: np.ndarray) -> None:
        if self._mo_cache_max_bytes <= 0:
            return
        if self._mo_cache_max_entry_bytes > 0 and int(val.nbytes) > int(self._mo_cache_max_entry_bytes):
            return
        if key in self._mo_cache:
            return
        self._mo_cache[key] = val
        self._mo_cache_bytes += int(val.nbytes)
        while self._mo_cache_bytes > self._mo_cache_max_bytes and self._mo_cache:
            _k, _v = self._mo_cache.popitem(last=False)
            self._mo_cache_bytes -= int(_v.nbytes)

    def _transform_block(
        self,
        mo_x: np.ndarray,
        mo_y: np.ndarray,
        *,
        compact: bool,
        same_input: bool,
        max_memory: int,
    ) -> np.ndarray:
        mo_x = np.asarray(mo_x, dtype=np.float64, order="C")
        mo_y = np.asarray(mo_y, dtype=np.float64, order="C")
        if mo_x.ndim != 2 or mo_y.ndim != 2:
            raise ValueError("mo_x and mo_y must be 2D arrays (nao, nmo)")

        nao = int(self.B_ao.shape[0])
        if int(mo_x.shape[0]) != nao or int(mo_y.shape[0]) != nao:
            raise ValueError("mo_x/mo_y nao mismatch with DF context AO basis")

        naux = int(self.B_ao.shape[2])
        n1 = int(mo_x.shape[1])
        n2 = int(mo_y.shape[1])
        if n1 <= 0 or n2 <= 0:
            raise ValueError("mo_x/mo_y must have at least 1 column")

        # Use the caller-provided identity flag (pre-asarray) since we may copy
        # into distinct contiguous buffers above.
        compact_s2 = bool(compact and same_input and n1 == n2)
        if compact_s2:
            npair = n1 * (n1 + 1) // 2
            out = np.empty((naux, npair), dtype=np.float64, order="C")
            tri_i, tri_j = np.tril_indices(n1)
        else:
            out = np.empty((naux, n1 * n2), dtype=np.float64, order="C")

        # Chunk over aux dimension to control peak memory.
        mem_mb = int(max_memory)
        if mem_mb <= 0:
            mem_mb = 256
        mem_bytes = int(mem_mb) * (1024**2)
        per_aux = int(8 * (nao * n2 + n1 * n2))
        if per_aux <= 0:
            per_aux = 1
        block_naux = max(1, min(naux, mem_bytes // per_aux))
        block_naux = int(min(block_naux, 1024))

        for q0 in range(0, naux, block_naux):
            q1 = min(naux, q0 + block_naux)
            B_blk = self.B_ao[:, :, q0:q1]  # (nao, nao, qb)
            qb = int(q1 - q0)

            # res[u,v,q] = mo_x^T @ B[:,:,q] @ mo_y
            tmp = np.tensordot(B_blk, mo_y, axes=([1], [0]))  # (nao, qb, n2)
            tmp = np.transpose(tmp, (0, 2, 1))  # (nao, n2, qb)
            res = np.tensordot(mo_x.T, tmp, axes=([1], [0]))  # (n1, n2, qb)

            if compact_s2:
                packed = res[tri_i, tri_j, :]  # (npair, qb)
                out[q0:q1, :] = np.asarray(packed.T, dtype=np.float64, order="C")
            else:
                out[q0:q1, :] = np.asarray(res.reshape(n1 * n2, qb).T, dtype=np.float64, order="C")

        return out

    def transform(
        self,
        mo_x: np.ndarray,
        mo_y: np.ndarray,
        *,
        compact: bool = True,
        cache: bool = True,
        max_memory: int | None = None,
    ) -> np.ndarray:
        same_input = bool(mo_y is mo_x)
        hx, sx = _hash_f64_array(mo_x)
        hy, sy = (hx, sx) if same_input else _hash_f64_array(mo_y)
        key = ("mo", bool(compact), hx, sx, hy, sy)
        if cache:
            hit = self._cache_get(key)
            if hit is not None:
                return hit

        mem = int(self.max_memory) if max_memory is None else int(max_memory)
        out = self._transform_block(mo_x, mo_y, compact=bool(compact), same_input=same_input, max_memory=mem)
        if cache:
            self._cache_put(key, out)
        return out

    def transform_many(
        self,
        blocks: list[tuple[np.ndarray, np.ndarray, bool]],
        *,
        cache: bool = True,
        max_memory: int | None = None,
    ) -> list[np.ndarray]:
        if not blocks:
            return []
        mem = int(self.max_memory) if max_memory is None else int(max_memory)
        return [self.transform(mx, my, compact=bool(compact), cache=bool(cache), max_memory=int(mem)) for mx, my, compact in blocks]


_DF_CTX_CACHE: dict[tuple[Any, ...], DFCholeskyContext] = {}


def get_df_cholesky_context(
    mol: Any,
    *,
    auxbasis: Any = "autoaux",
    max_memory: int = 2000,
    verbose: int = 0,
    threads: int = 0,
) -> DFCholeskyContext:
    mol_fp = _mol_fingerprint(mol)
    key = (mol_fp, repr(auxbasis), int(max_memory), int(verbose), int(threads))
    ctx = _DF_CTX_CACHE.get(key)
    if ctx is not None:
        return ctx

    ctx = DFCholeskyContext.build(
        mol,
        auxbasis=auxbasis,
        max_memory=int(max_memory),
        verbose=int(verbose),
        threads=int(threads),
    )
    _DF_CTX_CACHE[key] = ctx
    return ctx


__all__ = ["DFCholeskyContext", "get_df_cholesky_context"]
