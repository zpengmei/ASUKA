from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RASSpec:
    nras1: int
    nras2: int
    nras3: int
    max_hole: int
    max_particle: int


@dataclass(frozen=True)
class GASSpec:
    sizes: tuple[int, ...]
    bounds: tuple[tuple[int, int], ...]  # (emin, emax) per block


def ras_ne_constraints(norb: int, nelec: int, spec: RASSpec) -> dict[int, tuple[int, int]]:
    norb = int(norb)
    nelec = int(nelec)
    if norb < 0:
        raise ValueError("norb must be >= 0")
    if nelec < 0:
        raise ValueError("nelec must be >= 0")
    if nelec > 2 * norb:
        raise ValueError("nelec must be <= 2*norb")

    n1 = int(spec.nras1)
    n2 = int(spec.nras2)
    n3 = int(spec.nras3)
    if n1 < 0 or n2 < 0 or n3 < 0:
        raise ValueError("nras1/nras2/nras3 must be >= 0")
    if n1 + n2 + n3 != norb:
        raise ValueError("norb must equal nras1+nras2+nras3")

    max_hole = int(spec.max_hole)
    max_particle = int(spec.max_particle)
    if max_hole < 0:
        raise ValueError("max_hole must be >= 0")
    if max_particle < 0:
        raise ValueError("max_particle must be >= 0")
    if max_hole > 2 * n1:
        raise ValueError("max_hole must be <= 2*nras1")
    if max_particle > 2 * n3:
        raise ValueError("max_particle must be <= 2*nras3")
    if 2 * n1 - max_hole > nelec:
        raise ValueError("RAS1 hole constraint is incompatible with nelec")
    if max_particle > nelec:
        raise ValueError("max_particle must be <= nelec")

    constraints: dict[int, tuple[int, int]] = {}

    if n1 > 0:
        k = n1
        ne_min = 2 * n1 - max_hole
        ne_max = min(2 * n1, nelec)
        constraints[k] = (ne_min, ne_max)

    if n3 > 0:
        k = n1 + n2
        if 0 < k < norb:
            ne_min = nelec - max_particle
            ne_max = nelec
            if k in constraints:
                old_min, old_max = constraints[k]
                ne_min = max(old_min, ne_min)
                ne_max = min(old_max, ne_max)
                if ne_min > ne_max:
                    raise ValueError("RAS constraints are incompatible at the RAS2 boundary")
            constraints[k] = (ne_min, ne_max)

    for k, (ne_min, ne_max) in constraints.items():
        if k < 0 or k > norb:
            raise ValueError("invalid constraint boundary k")
        if ne_min < 0 or ne_max < 0:
            raise ValueError("constraint bounds must be >= 0")
        if ne_min > ne_max:
            raise ValueError("constraint must satisfy ne_min <= ne_max")
        if ne_max > nelec:
            raise ValueError("constraint ne_max must be <= nelec")

    return constraints


def gas_ne_constraints(norb: int, nelec: int, spec: GASSpec) -> dict[int, tuple[int, int]]:
    norb = int(norb)
    nelec = int(nelec)
    if norb < 0:
        raise ValueError("norb must be >= 0")
    if nelec < 0:
        raise ValueError("nelec must be >= 0")
    if nelec > 2 * norb:
        raise ValueError("nelec must be <= 2*norb")

    sizes = tuple(int(x) for x in spec.sizes)
    bounds = tuple((int(b[0]), int(b[1])) for b in spec.bounds)
    if len(sizes) != len(bounds):
        raise ValueError("sizes and bounds must have the same length")
    if any(s <= 0 for s in sizes):
        raise ValueError("GAS block sizes must be > 0")
    if sum(sizes) != norb:
        raise ValueError("norb must equal sum(sizes)")

    sum_emin = 0
    sum_emax = 0
    for size, (emin, emax) in zip(sizes, bounds):
        if emin < 0 or emax < 0:
            raise ValueError("GAS bounds must be >= 0")
        if emin > emax:
            raise ValueError("GAS bounds must satisfy emin <= emax")
        if emax > 2 * size:
            raise ValueError("GAS emax must be <= 2*block_size")
        sum_emin += emin
        sum_emax += emax

    if sum_emin > nelec or sum_emax < nelec:
        raise ValueError("GAS bounds are incompatible with nelec")

    constraints: dict[int, tuple[int, int]] = {}
    k = 0
    cum_emin = 0
    cum_emax = 0
    for idx, (size, (emin, emax)) in enumerate(zip(sizes, bounds)):
        k += int(size)
        cum_emin += int(emin)
        cum_emax += int(emax)
        if idx == len(sizes) - 1:
            break  # skip k=norb (redundant and may violate ne_max<=nelec)

        ne_min = cum_emin
        ne_max = min(cum_emax, nelec)
        if ne_min < 0 or ne_max < 0:
            raise ValueError("GAS constraint bounds must be >= 0")
        if ne_min > ne_max:
            raise ValueError("GAS constraints are infeasible at a boundary")
        if ne_max > nelec:
            raise ValueError("GAS constraint ne_max must be <= nelec")
        constraints[k] = (ne_min, ne_max)

    return constraints


def merge_ne_constraints(
    *dicts: dict[int, tuple[int, int]] | None,
) -> dict[int, tuple[int, int]] | None:
    merged: dict[int, tuple[int, int]] = {}
    any_non_none = False
    for d in dicts:
        if not d:
            continue
        any_non_none = True
        for k, bounds in dict(d).items():
            kk = int(k)
            if bounds is None or len(bounds) != 2:
                raise ValueError(f"ne_constraints[{kk}] must be a (ne_min, ne_max) tuple")
            ne_min = int(bounds[0])
            ne_max = int(bounds[1])
            if ne_min < 0 or ne_max < 0:
                raise ValueError(f"ne_constraints[{kk}] bounds must be >= 0")
            if ne_min > ne_max:
                raise ValueError(f"ne_constraints[{kk}] must satisfy ne_min <= ne_max")
            if kk in merged:
                old_min, old_max = merged[kk]
                ne_min = max(old_min, ne_min)
                ne_max = min(old_max, ne_max)
                if ne_min > ne_max:
                    raise ValueError(f"infeasible merged ne_constraints at k={kk}")
            merged[kk] = (ne_min, ne_max)

    if not any_non_none:
        return None
    return None if not merged else merged

