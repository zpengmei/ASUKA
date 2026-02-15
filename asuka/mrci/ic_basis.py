from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class OrbitalSpaces:
    """Orbital index partitions for internally contracted MR-CI.

    Indices refer to a *correlated* MO space (after any frozen-core folding).
    Recommended ordering for correlated orbitals is `[I][V]`.

    Attributes
    ----------
    internal : np.ndarray
        Indices of internal (active) orbitals. Shape: (nI,).
    external : np.ndarray
        Indices of external (virtual) orbitals. Shape: (nV,).
    orbsym : np.ndarray | None
        Orbital symmetry labels (PySCF XOR irreps). Shape: (norb,).
    """

    internal: np.ndarray  # (nI,), int32
    external: np.ndarray  # (nV,), int32
    orbsym: np.ndarray | None = None  # (norb,), int32, PySCF XOR irreps

    def __post_init__(self) -> None:
        internal = np.asarray(self.internal, dtype=np.int32).ravel()
        external = np.asarray(self.external, dtype=np.int32).ravel()
        object.__setattr__(self, "internal", internal)
        object.__setattr__(self, "external", external)

        if np.any(internal < 0) or np.any(external < 0):
            raise ValueError("orbital indices must be >= 0")

        if self.orbsym is not None:
            orbsym = np.asarray(self.orbsym, dtype=np.int32).ravel()
            object.__setattr__(self, "orbsym", orbsym)
            norb = int(orbsym.size)
            if np.any(internal >= norb) or np.any(external >= norb):
                raise ValueError("orbital indices out of range for orbsym length")

        # Disjointness is required for well-defined excitation labels.
        if np.intersect1d(internal, external).size:
            raise ValueError("internal and external orbital sets must be disjoint")

    @property
    def n_internal(self) -> int:
        return int(self.internal.size)

    @property
    def n_external(self) -> int:
        return int(self.external.size)


@dataclass(frozen=True)
class ICSingles:
    """Label set for fully internally contracted singles: |a r> = E_ar |Psi0>.

    Attributes
    ----------
    a : np.ndarray
        External orbital indices. Shape: (nlab,).
    r : np.ndarray
        Internal orbital indices. Shape: (nlab,).
    a_group_offsets : np.ndarray
        Offsets into `a_group_order` for grouping labels by `a`. Shape: (n_groups+1,).
    a_group_order : np.ndarray
        Permutation indices that sort labels by `a`. Shape: (nlab,).
    a_group_keys : np.ndarray
        Unique external orbital indices present in the label set. Shape: (n_groups,).
    """

    a: np.ndarray  # (nlab,), int32, orbital index in external space (global orbital id)
    r: np.ndarray  # (nlab,), int32, orbital index in internal space (global orbital id)
    a_group_offsets: np.ndarray  # (n_groups+1,), int32 offsets into `a_group_order`
    a_group_order: np.ndarray  # (nlab,), int32 permutation that groups labels by `a`
    a_group_keys: np.ndarray  # (n_groups,), int32 unique external orbital ids (sorted)

    def __post_init__(self) -> None:
        a = np.asarray(self.a, dtype=np.int32).ravel()
        r = np.asarray(self.r, dtype=np.int32).ravel()
        if a.size != r.size:
            raise ValueError("a and r must have the same length")
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "r", r)

        off = np.asarray(self.a_group_offsets, dtype=np.int32).ravel()
        order = np.asarray(self.a_group_order, dtype=np.int32).ravel()
        keys = np.asarray(self.a_group_keys, dtype=np.int32).ravel()
        if off.size < 1 or int(off[0]) != 0 or int(off[-1]) != int(a.size):
            raise ValueError("invalid a_group_offsets")
        if int(order.size) != int(a.size):
            raise ValueError("a_group_order must have length nlab")
        if int(keys.size) != int(off.size - 1):
            raise ValueError("a_group_keys length must be len(offsets)-1")
        object.__setattr__(self, "a_group_offsets", off)
        object.__setattr__(self, "a_group_order", order)
        object.__setattr__(self, "a_group_keys", keys)

    @property
    def nlab(self) -> int:
        return int(self.a.size)

    @property
    def n_groups(self) -> int:
        return int(self.a_group_keys.size)


@dataclass(frozen=True)
class ICDoubles:
    """Label set for fully internally contracted doubles: |ab;rs> = E_ar E_bs |Psi0>.

    Attributes
    ----------
    a : np.ndarray
        First external orbital indices. Shape: (nlab,).
    b : np.ndarray
        Second external orbital indices. Shape: (nlab,).
    r : np.ndarray
        First internal orbital indices. Shape: (nlab,).
    s : np.ndarray
        Second internal orbital indices. Shape: (nlab,).
    ab_group_offsets : np.ndarray
        Offsets into `ab_group_order` for grouping labels by (a,b). Shape: (n_groups+1,).
    ab_group_order : np.ndarray
        Permutation indices that sort labels by (a,b). Shape: (nlab,).
    ab_group_keys : np.ndarray
        Unique external pairs (a,b) present in the label set. Shape: (n_groups, 2).
    """

    a: np.ndarray  # (nlab,), int32
    b: np.ndarray  # (nlab,), int32
    r: np.ndarray  # (nlab,), int32
    s: np.ndarray  # (nlab,), int32

    ab_group_offsets: np.ndarray  # (n_groups+1,), int32 offsets into ab_group_order
    ab_group_order: np.ndarray  # (nlab,), int32 permutation grouping labels by (a,b)
    ab_group_keys: np.ndarray  # (n_groups,2), int32 unique (a,b) pairs (sorted)

    def __post_init__(self) -> None:
        a = np.asarray(self.a, dtype=np.int32).ravel()
        b = np.asarray(self.b, dtype=np.int32).ravel()
        r = np.asarray(self.r, dtype=np.int32).ravel()
        s = np.asarray(self.s, dtype=np.int32).ravel()
        if not (a.size == b.size == r.size == s.size):
            raise ValueError("a,b,r,s must have the same length")
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "r", r)
        object.__setattr__(self, "s", s)

        off = np.asarray(self.ab_group_offsets, dtype=np.int32).ravel()
        order = np.asarray(self.ab_group_order, dtype=np.int32).ravel()
        keys = np.asarray(self.ab_group_keys, dtype=np.int32)
        if keys.ndim != 2 or keys.shape[1] != 2:
            raise ValueError("ab_group_keys must have shape (n_groups, 2)")
        if off.size < 1 or int(off[0]) != 0 or int(off[-1]) != int(a.size):
            raise ValueError("invalid ab_group_offsets")
        if int(order.size) != int(a.size):
            raise ValueError("ab_group_order must have length nlab")
        if int(keys.shape[0]) != int(off.size - 1):
            raise ValueError("ab_group_keys length must be len(offsets)-1")
        object.__setattr__(self, "ab_group_offsets", off)
        object.__setattr__(self, "ab_group_order", order)
        object.__setattr__(self, "ab_group_keys", keys)

    @property
    def nlab(self) -> int:
        return int(self.a.size)

    @property
    def n_groups(self) -> int:
        return int(self.ab_group_keys.shape[0])


@dataclass(frozen=True)
class SCSingles:
    """Label set for strongly contracted singles: |a> = sum_r E_ar |Psi0>.

    Attributes
    ----------
    a : np.ndarray
        External orbital indices. Shape: (nlab,).
    """

    a: np.ndarray  # (nlab,), int32 (global external orbital id)

    def __post_init__(self) -> None:
        a = np.asarray(self.a, dtype=np.int32).ravel()
        object.__setattr__(self, "a", a)

    @property
    def nlab(self) -> int:
        return int(self.a.size)


@dataclass(frozen=True)
class SCDoubles:
    """Label set for strongly contracted doubles: |ab> = sum_{r,s} E_ar E_bs |Psi0>.

    Attributes
    ----------
    a : np.ndarray
        First external orbital indices. Shape: (nlab,).
    b : np.ndarray
        Second external orbital indices. Shape: (nlab,).
    """

    a: np.ndarray  # (nlab,), int32
    b: np.ndarray  # (nlab,), int32

    def __post_init__(self) -> None:
        a = np.asarray(self.a, dtype=np.int32).ravel()
        b = np.asarray(self.b, dtype=np.int32).ravel()
        if a.size != b.size:
            raise ValueError("a and b must have the same length")
        if int(a.size) and not bool(np.all(a <= b)):
            raise ValueError("SCDoubles requires a <= b for all labels")
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)

    @property
    def nlab(self) -> int:
        return int(self.a.size)


def enumerate_ic_singles(
    spaces: OrbitalSpaces,
    *,
    symmetry: bool = True,
) -> ICSingles:
    """Enumerate fully internally contracted singles labels (a,r).

    Parameters
    ----------
    spaces : OrbitalSpaces
        Definition of internal and external orbital spaces.
    symmetry : bool, optional
        If True and `spaces.orbsym` is available, filter labels by symmetry
        (orbsym[a] ^ orbsym[r] == 0). Default is True.

    Returns
    -------
    ICSingles
        Enumerated singles labels.
    """

    internal = np.asarray(spaces.internal, dtype=np.int32).ravel()
    external = np.asarray(spaces.external, dtype=np.int32).ravel()
    if internal.size == 0 or external.size == 0:
        return ICSingles(
            a=np.zeros(0, dtype=np.int32),
            r=np.zeros(0, dtype=np.int32),
            a_group_offsets=np.asarray([0], dtype=np.int32),
            a_group_order=np.zeros(0, dtype=np.int32),
            a_group_keys=np.zeros(0, dtype=np.int32),
        )

    use_sym = bool(symmetry) and (spaces.orbsym is not None)
    orbsym = None if spaces.orbsym is None else np.asarray(spaces.orbsym, dtype=np.int32).ravel()

    a_list: list[int] = []
    r_list: list[int] = []
    if use_sym:
        for a in external.tolist():
            sym_a = int(orbsym[int(a)])
            for r in internal.tolist():
                if (sym_a ^ int(orbsym[int(r)])) != 0:
                    continue
                a_list.append(int(a))
                r_list.append(int(r))
    else:
        for a in external.tolist():
            for r in internal.tolist():
                a_list.append(int(a))
                r_list.append(int(r))

    a_arr = np.asarray(a_list, dtype=np.int32)
    r_arr = np.asarray(r_list, dtype=np.int32)

    if int(a_arr.size) == 0:
        return ICSingles(
            a=a_arr,
            r=r_arr,
            a_group_offsets=np.asarray([0], dtype=np.int32),
            a_group_order=np.zeros(0, dtype=np.int32),
            a_group_keys=np.zeros(0, dtype=np.int32),
        )

    # Group labels by external index a (for overlap applications).
    order = np.argsort(a_arr, kind="mergesort").astype(np.int32, copy=False)
    a_sorted = a_arr[order]
    change = np.nonzero(a_sorted[1:] != a_sorted[:-1])[0] + 1
    offsets = np.concatenate(
        (
            np.asarray([0], dtype=np.int32),
            change.astype(np.int32, copy=False),
            np.asarray([int(a_arr.size)], dtype=np.int32),
        )
    )
    keys = a_sorted[offsets[:-1]].copy()

    return ICSingles(
        a=a_arr,
        r=r_arr,
        a_group_offsets=offsets,
        a_group_order=order,
        a_group_keys=keys,
    )


def enumerate_ic_doubles(
    spaces: OrbitalSpaces,
    *,
    symmetry: bool = True,
    allow_same_external: bool = True,
    allow_same_internal: bool = True,
) -> ICDoubles:
    """Enumerate fully internally contracted doubles labels (a,b,r,s).

    Parameters
    ----------
    spaces : OrbitalSpaces
        Definition of internal and external orbital spaces.
    symmetry : bool, optional
        If True and `spaces.orbsym` is available, filter labels by symmetry.
        Default is True.
    allow_same_external : bool, optional
        If False, exclude pairs with a == b. Default is True.
    allow_same_internal : bool, optional
        If False, exclude pairs with r == s. Default is True.

    Returns
    -------
    ICDoubles
        Enumerated and grouped doubles labels with canonical ordering (a,r) <= (b,s).
    """

    internal = np.asarray(spaces.internal, dtype=np.int32).ravel()
    external = np.asarray(spaces.external, dtype=np.int32).ravel()
    if internal.size == 0 or external.size == 0:
        return ICDoubles(
            a=np.zeros(0, dtype=np.int32),
            b=np.zeros(0, dtype=np.int32),
            r=np.zeros(0, dtype=np.int32),
            s=np.zeros(0, dtype=np.int32),
            ab_group_offsets=np.asarray([0], dtype=np.int32),
            ab_group_order=np.zeros(0, dtype=np.int32),
            ab_group_keys=np.zeros((0, 2), dtype=np.int32),
        )

    use_sym = bool(symmetry) and (spaces.orbsym is not None)
    orbsym = None if spaces.orbsym is None else np.asarray(spaces.orbsym, dtype=np.int32).ravel()

    allow_same_external = bool(allow_same_external)
    allow_same_internal = bool(allow_same_internal)

    # Build all possible single-excitation pair labels (a,r), then form unordered
    # pair-of-pairs (a,r)(b,s) with canonical (a,r) <= (b,s).
    ext = external.tolist()
    intl = internal.tolist()

    pairs: list[tuple[int, int, int]] = []  # (key, a, r)
    for a in ext:
        for r in intl:
            key = (int(a) << 32) | int(r)
            pairs.append((key, int(a), int(r)))
    pairs.sort(key=lambda t: t[0])

    a_list: list[int] = []
    b_list: list[int] = []
    r_list: list[int] = []
    s_list: list[int] = []

    for i, (_k1, a, r) in enumerate(pairs):
        sym_ar = 0
        if use_sym:
            sym_ar = int(orbsym[int(a)]) ^ int(orbsym[int(r)])
        for j in range(i, len(pairs)):
            _k2, b, s = pairs[j]
            if (not allow_same_external) and int(a) == int(b):
                continue
            if (not allow_same_internal) and int(r) == int(s):
                continue
            if use_sym:
                if (sym_ar ^ int(orbsym[int(b)]) ^ int(orbsym[int(s)])) != 0:
                    continue
            a_list.append(int(a))
            b_list.append(int(b))
            r_list.append(int(r))
            s_list.append(int(s))

    a_arr = np.asarray(a_list, dtype=np.int32)
    b_arr = np.asarray(b_list, dtype=np.int32)
    r_arr = np.asarray(r_list, dtype=np.int32)
    s_arr = np.asarray(s_list, dtype=np.int32)

    if int(a_arr.size) == 0:
        return ICDoubles(
            a=a_arr,
            b=b_arr,
            r=r_arr,
            s=s_arr,
            ab_group_offsets=np.asarray([0], dtype=np.int32),
            ab_group_order=np.zeros(0, dtype=np.int32),
            ab_group_keys=np.zeros((0, 2), dtype=np.int32),
        )

    # Group by external pair (a,b) for staged expansions/projections.
    ab_key = (a_arr.astype(np.int64) << 32) ^ b_arr.astype(np.int64)
    order = np.argsort(ab_key, kind="mergesort").astype(np.int32, copy=False)
    ab_sorted = ab_key[order]
    change = np.nonzero(ab_sorted[1:] != ab_sorted[:-1])[0] + 1
    offsets = np.concatenate(
        (
            np.asarray([0], dtype=np.int32),
            change.astype(np.int32, copy=False),
            np.asarray([int(a_arr.size)], dtype=np.int32),
        )
    )

    keys = np.empty((int(offsets.size - 1), 2), dtype=np.int32)
    for g in range(int(offsets.size - 1)):
        idx0 = int(order[int(offsets[g])])
        keys[g, 0] = int(a_arr[idx0])
        keys[g, 1] = int(b_arr[idx0])

    return ICDoubles(
        a=a_arr,
        b=b_arr,
        r=r_arr,
        s=s_arr,
        ab_group_offsets=offsets,
        ab_group_order=order,
        ab_group_keys=keys,
    )


def enumerate_sc_singles(
    spaces: OrbitalSpaces,
    *,
    symmetry: bool = True,
) -> SCSingles:
    """Enumerate strongly contracted singles labels (a).

    Parameters
    ----------
    spaces : OrbitalSpaces
        Definition of internal and external orbital spaces.
    symmetry : bool, optional
        If True and `spaces.orbsym` is available, include only `a` for which
        there exists at least one internal `r` satisfying symmetry. Default is True.

    Returns
    -------
    SCSingles
        Enumerated singles labels.
    """

    internal = np.asarray(spaces.internal, dtype=np.int32).ravel()
    external = np.asarray(spaces.external, dtype=np.int32).ravel()
    if internal.size == 0 or external.size == 0:
        return SCSingles(a=np.zeros(0, dtype=np.int32))

    use_sym = bool(symmetry) and (spaces.orbsym is not None)
    if not use_sym:
        a = np.unique(external.astype(np.int32, copy=False))
        a.sort()
        return SCSingles(a=a)

    orbsym = np.asarray(spaces.orbsym, dtype=np.int32).ravel()

    a_list: list[int] = []
    intl = internal.tolist()
    for a in external.tolist():
        sym_a = int(orbsym[int(a)])
        ok = False
        for r in intl:
            if (sym_a ^ int(orbsym[int(r)])) == 0:
                ok = True
                break
        if ok:
            a_list.append(int(a))

    a_arr = np.unique(np.asarray(a_list, dtype=np.int32))
    a_arr.sort()
    return SCSingles(a=a_arr)


def enumerate_sc_doubles(
    spaces: OrbitalSpaces,
    *,
    symmetry: bool = True,
    allow_same_external: bool = True,
    allow_same_internal: bool = True,
) -> SCDoubles:
    """Enumerate strongly contracted doubles labels (a,b).

    Parameters
    ----------
    spaces : OrbitalSpaces
        Definition of internal and external orbital spaces.
    symmetry : bool, optional
        If True and `spaces.orbsym` is available, include only pairs `(a,b)` for
        which there exists at least one internal pair `(r,s)` satisfying symmetry.
        Default is True.
    allow_same_external : bool, optional
        If False, exclude pairs with a == b. Default is True.
    allow_same_internal : bool, optional
        If False, require r != s for existence check. Default is True.

    Returns
    -------
    SCDoubles
        Enumerated doubles labels with canonical ordering a <= b.
    """

    internal = np.unique(np.asarray(spaces.internal, dtype=np.int32).ravel())
    internal.sort()
    external = np.unique(np.asarray(spaces.external, dtype=np.int32).ravel())
    external.sort()
    if internal.size == 0 or external.size == 0:
        return SCDoubles(a=np.zeros(0, dtype=np.int32), b=np.zeros(0, dtype=np.int32))

    allow_same_external = bool(allow_same_external)
    allow_same_internal = bool(allow_same_internal)

    nI = int(internal.size)
    if (not allow_same_internal) and nI < 2:
        return SCDoubles(a=np.zeros(0, dtype=np.int32), b=np.zeros(0, dtype=np.int32))

    use_sym = bool(symmetry) and (spaces.orbsym is not None)
    orbsym = None if spaces.orbsym is None else np.asarray(spaces.orbsym, dtype=np.int32).ravel()

    ext = external.tolist()
    intl = internal.tolist()

    a_list: list[int] = []
    b_list: list[int] = []

    for ia, a in enumerate(ext):
        sym_a = 0 if not use_sym else int(orbsym[int(a)])
        start_b = ia if allow_same_external else ia + 1
        for b in ext[start_b:]:
            sym_b = 0 if not use_sym else int(orbsym[int(b)])

            ok = False
            if not use_sym:
                ok = True
            else:
                for ir, r in enumerate(intl):
                    sym_ar = sym_a ^ int(orbsym[int(r)])
                    if int(a) == int(b):
                        start_s = ir if allow_same_internal else ir + 1
                        for s in intl[start_s:]:
                            if (sym_ar ^ sym_b ^ int(orbsym[int(s)])) == 0:
                                ok = True
                                break
                        if ok:
                            break
                    else:
                        for s in intl:
                            if (not allow_same_internal) and int(r) == int(s):
                                continue
                            if (sym_ar ^ sym_b ^ int(orbsym[int(s)])) == 0:
                                ok = True
                                break
                        if ok:
                            break

            if ok:
                a_list.append(int(a))
                b_list.append(int(b))

    a_arr = np.asarray(a_list, dtype=np.int32)
    b_arr = np.asarray(b_list, dtype=np.int32)
    return SCDoubles(a=a_arr, b=b_arr)


def filter_ic_singles_by_norm(
    labels: ICSingles,
    *,
    gamma: np.ndarray,
    norm_min: float,
) -> ICSingles:
    """Filter singles labels by overlap norm ||E_ar|Psi0>||^2 = gamma[r,r]."""

    gamma = np.asarray(gamma, dtype=np.float64)
    if gamma.ndim != 2 or gamma.shape[0] != gamma.shape[1]:
        raise ValueError("gamma must be square")

    r = np.asarray(labels.r, dtype=np.int64)
    if np.any(r < 0) or np.any(r >= int(gamma.shape[0])):
        raise ValueError("labels.r out of range for gamma")

    diag = np.diag(gamma)
    keep = diag[r] >= float(norm_min)

    a_new = np.asarray(labels.a[keep], dtype=np.int32)
    r_new = np.asarray(labels.r[keep], dtype=np.int32)

    if int(a_new.size) == 0:
        return ICSingles(
            a=a_new,
            r=r_new,
            a_group_offsets=np.asarray([0], dtype=np.int32),
            a_group_order=np.zeros(0, dtype=np.int32),
            a_group_keys=np.zeros(0, dtype=np.int32),
        )

    order = np.argsort(a_new, kind="mergesort").astype(np.int32, copy=False)
    a_sorted = a_new[order]
    change = np.nonzero(a_sorted[1:] != a_sorted[:-1])[0] + 1
    offsets = np.concatenate(
        (
            np.asarray([0], dtype=np.int32),
            change.astype(np.int32, copy=False),
            np.asarray([int(a_new.size)], dtype=np.int32),
        )
    )
    keys = a_sorted[offsets[:-1]].copy()

    return ICSingles(
        a=a_new,
        r=r_new,
        a_group_offsets=offsets,
        a_group_order=order,
        a_group_keys=keys,
    )


def filter_ic_doubles_by_norm(
    labels: ICDoubles,
    *,
    dm2: np.ndarray,
    norm_min: float,
) -> ICDoubles:
    """Filter doubles labels by overlap norms <ab;rs|ab;rs>.

    Parameters
    ----------
    labels : ICDoubles
        The initial set of doubles labels.
    dm2 : np.ndarray
        2-RDM on internal orbitals. Shape: (nI, nI, nI, nI).
    norm_min : float
        Minimum norm threshold.

    Returns
    -------
    ICDoubles
        Filtered set of doubles labels.
    """

    dm2 = np.asarray(dm2, dtype=np.float64)
    if dm2.ndim != 4 or dm2.shape[0] != dm2.shape[1] or dm2.shape[0] != dm2.shape[2] or dm2.shape[0] != dm2.shape[3]:
        raise ValueError("dm2 must have shape (nI, nI, nI, nI)")
    nI = int(dm2.shape[0])

    r = np.asarray(labels.r, dtype=np.int64)
    s = np.asarray(labels.s, dtype=np.int64)
    if np.any(r < 0) or np.any(r >= nI) or np.any(s < 0) or np.any(s >= nI):
        raise ValueError("labels.r/labels.s out of range for dm2")

    diag = dm2[r, r, s, s].copy()
    same_ext = np.asarray(labels.a == labels.b, dtype=bool)
    if bool(np.any(same_ext)):
        diag[same_ext] += dm2[s[same_ext], r[same_ext], r[same_ext], s[same_ext]]

    keep = diag >= float(norm_min)

    a_new = np.asarray(labels.a[keep], dtype=np.int32)
    b_new = np.asarray(labels.b[keep], dtype=np.int32)
    r_new = np.asarray(labels.r[keep], dtype=np.int32)
    s_new = np.asarray(labels.s[keep], dtype=np.int32)

    if int(a_new.size) == 0:
        return ICDoubles(
            a=a_new,
            b=b_new,
            r=r_new,
            s=s_new,
            ab_group_offsets=np.asarray([0], dtype=np.int32),
            ab_group_order=np.zeros(0, dtype=np.int32),
            ab_group_keys=np.zeros((0, 2), dtype=np.int32),
        )

    ab_key = (a_new.astype(np.int64) << 32) ^ b_new.astype(np.int64)
    order = np.argsort(ab_key, kind="mergesort").astype(np.int32, copy=False)
    ab_sorted = ab_key[order]
    change = np.nonzero(ab_sorted[1:] != ab_sorted[:-1])[0] + 1
    offsets = np.concatenate(
        (
            np.asarray([0], dtype=np.int32),
            change.astype(np.int32, copy=False),
            np.asarray([int(a_new.size)], dtype=np.int32),
        )
    )

    keys = np.empty((int(offsets.size - 1), 2), dtype=np.int32)
    for g in range(int(offsets.size - 1)):
        idx0 = int(order[int(offsets[g])])
        keys[g, 0] = int(a_new[idx0])
        keys[g, 1] = int(b_new[idx0])

    return ICDoubles(
        a=a_new,
        b=b_new,
        r=r_new,
        s=s_new,
        ab_group_offsets=offsets,
        ab_group_order=order,
        ab_group_keys=keys,
    )
