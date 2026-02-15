from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asuka.cuguga.overlap.overlap_segment import Generator


def V_from_eri(eri: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    """Return Dobrautz' V_{ij,kl} from chemist-notation ERI (pq|rs).

    Convention (per Dobrautz/Smart/Alavi):
      V_{ij,kl} = < i k | j l >

    With `eri[p,q,r,s] = (pq|rs)`, that is:
      V_{ij,kl} = eri[i, k, j, l]
    """

    return float(np.asarray(eri, dtype=np.float64)[int(i), int(k), int(j), int(l)])


@dataclass(frozen=True)
class TwoBodyContribution:
    """Canonical two-body wiring for overlap-table evaluation.

    This stores:
    - ordered one-body generators (gen1, gen2) used to evaluate overlap products -> (w0, w1)
    - the ERI permutations needed to combine (w0, w1) into the Hamiltonian matrix element

    Notes:
    - `eri_a`/`eri_b` are 4-index tuples into the *chemist* ERI tensor eri[p,q,r,s]=(pq|rs).
    - The final formulas here intentionally match the Appendix-B style "w0/w1 + ERI combos"
      used for overlap-range implementations.
    """

    type_code: str
    kind: str  # "like" or "mixed"
    gen1: Generator
    gen2: Generator
    eri_a: tuple[int, int, int, int]
    eri_b: tuple[int, int, int, int]

    def h2_value(self, eri: np.ndarray, *, w0: float, w1: float) -> float:
        eri = np.asarray(eri, dtype=np.float64)
        a = float(eri[self.eri_a])
        b = float(eri[self.eri_b])
        if self.kind == "like":
            # Type (3a) prototype:
            #   H = w0*(V1+V2) + w1*(V1-V2)
            return float(w0 * (a + b) + w1 * (a - b))
        if self.kind == "mixed":
            # Mixed-generator overlap prototypes (types 3c–3f):
            #   H = w0*(Vcoul - 0.5*Vexch) + w1*Vexch
            # with (eri_a, eri_b) = (Vcoul, Vexch).
            return float(w0 * (a - 0.5 * b) + w1 * b)
        raise ValueError(f"unknown kind {self.kind!r}")


def canonical_two_body_contribution(type_code: str, i: int, j: int, k: int, l: int) -> TwoBodyContribution:
    """Return canonical generator order + ERI wiring for a double-excitation type.

    Inputs must be the four distinct orbitals involved, in ascending order i<j<k<l.
    `type_code` is the thesis/excitation classifier label (e.g. "3a", "3c", ...).

    Implemented:
    - like generators: type "3a" prototype
    - mixed generators: types "3c".."3f" prototype wiring

    This is intentionally small and explicit; extend as more excitation types are implemented.
    """

    i = int(i)
    j = int(j)
    k = int(k)
    l = int(l)
    if not (i < j < k < l):
        raise ValueError("expected i<j<k<l for canonical double-excitation wiring")

    type_code = str(type_code)
    if type_code == "3a":
        # Standard operator (Table VI style): e_{jl,ik} / e_{jk,il}
        # Hamiltonian combo uses V_{jlik} and V_{jkil}.
        gen1 = Generator(j, l)  # R
        gen2 = Generator(i, k)  # R
        # V_{jlik} = V(j,l,i,k) = eri[j, i, l, k]
        # V_{jkil} = V(j,k,i,l) = eri[j, i, k, l]
        return TwoBodyContribution(
            type_code=type_code,
            kind="like",
            gen1=gen1,
            gen2=gen2,
            eri_a=(j, i, l, k),
            eri_b=(j, i, k, l),
        )

    if type_code in ("3c", "3d", "3e", "3f"):
        # Mixed-generator overlap wiring:
        #   H = w0*(V_{ijkl} - 0.5*V_{ilkj}) + w1*V_{ilkj}
        gen1 = Generator(i, l)  # R
        gen2 = Generator(k, j)  # L
        # V_{ijkl} = V(i,j,k,l) = eri[i, k, j, l]
        # V_{ilkj} = V(i,l,k,j) = eri[i, k, l, j]
        return TwoBodyContribution(
            type_code=type_code,
            kind="mixed",
            gen1=gen1,
            gen2=gen2,
            eri_a=(i, k, j, l),  # "coul"
            eri_b=(i, k, l, j),  # "exch"
        )

    raise NotImplementedError(f"unsupported type_code {type_code!r}")
