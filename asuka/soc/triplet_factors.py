from __future__ import annotations

import math
from typing import Final

from asuka.soc.wigner import wigner_6j_twos

TWOS_HALF: Final[int] = 1  # 2*(1/2)


def phase_from_twos_sum(twos_sum: int) -> float:
    """Return (-1)**(twos_sum/2) for even twos_sum; return 0.0 if twos_sum is odd."""

    twos_sum = int(twos_sum)
    if twos_sum & 1:
        return 0.0
    return -1.0 if ((twos_sum // 2) & 1) else 1.0


def A_factor(
    twos_km1: int,
    twos_k: int,
    twos_km1_p: int,
    twos_k_p: int,
    *,
    occ_case: str,
) -> float:
    """Triplet ICE endpoint factor A_t for an annihilated orbital.

    Implements Lang et al. (JCTC 2025) eqs. (63)–(64).

    Parameters are doubled-integer intermediate spins around orbital level t:
    - ket:  twos_km1 = 2*S_{t-1}, twos_k = 2*S_t
    - bra:  twos_km1_p = 2*S'_{t-1}, twos_k_p = 2*S'_t

    `occ_case` is determined by the **ket occupancy** at this orbital:
    - "DOMO": ket DOMO (spin-0), bra SOMO  -> eq. (63)
    - "SOMO": ket SOMO, bra VMO (spin-0)   -> eq. (64)
    """

    occ_case = str(occ_case).upper()
    if occ_case not in ("SOMO", "DOMO"):
        raise ValueError("occ_case must be 'SOMO' or 'DOMO'")

    twos_km1 = int(twos_km1)
    twos_k = int(twos_k)
    twos_km1_p = int(twos_km1_p)
    twos_k_p = int(twos_k_p)

    if occ_case == "DOMO":
        # Eq. (63): A_t^DOMO = sqrt(2*S'_t + 1).
        # DOMO on ket side implies S_t = S_{t-1}.
        if twos_k != twos_km1:
            return 0.0
        return math.sqrt(float(twos_k_p + 1))

    # Eq. (64): A_t^SOMO = (-1)^{S'_{t-1} - S_t - 1/2} * sqrt(2*S_t + 1).
    # VMO on bra side implies S'_t = S'_{t-1}.
    if twos_k_p != twos_km1_p:
        return 0.0
    phase = phase_from_twos_sum(twos_km1_p - twos_k - TWOS_HALF)
    if phase == 0.0:
        return 0.0
    return float(phase * math.sqrt(float(twos_k + 1)))


def B_factor(
    twos_km1: int,
    twos_k: int,
    twos_km1_p: int,
    twos_k_p: int,
    *,
    occ_case: str,
) -> float:
    """Triplet ICE endpoint factor B_t for a created orbital.

    Implements Lang et al. (JCTC 2025) eqs. (63)–(64).

    `occ_case` is determined by the **bra occupancy** at this orbital:
    - "SOMO": bra SOMO, ket VMO (spin-0)   -> eq. (63)
    - "DOMO": bra DOMO (spin-0), ket SOMO  -> eq. (64)
    """

    occ_case = str(occ_case).upper()
    if occ_case not in ("SOMO", "DOMO"):
        raise ValueError("occ_case must be 'SOMO' or 'DOMO'")

    twos_km1 = int(twos_km1)
    twos_k = int(twos_k)
    twos_km1_p = int(twos_km1_p)
    twos_k_p = int(twos_k_p)

    if occ_case == "SOMO":
        # Eq. (63): B_t^SOMO = sqrt(2*S'_t + 1).
        # VMO on ket side implies S_t = S_{t-1}.
        if twos_k != twos_km1:
            return 0.0
        return math.sqrt(float(twos_k_p + 1))

    # Eq. (64): B_t^DOMO = (-1)^{S'_{t-1} - S_t - 1/2} * sqrt(2*S_t + 1).
    # DOMO on bra side implies S'_t = S'_{t-1}.
    if twos_k_p != twos_km1_p:
        return 0.0
    phase = phase_from_twos_sum(twos_km1_p - twos_k - TWOS_HALF)
    if phase == 0.0:
        return 0.0
    return float(phase * math.sqrt(float(twos_k + 1)))


def Atilde_factor(
    twos_km1: int,
    twos_k: int,
    twos_km1_p: int,
    twos_k_p: int,
    *,
    twos_opline: int,
    occ_case: str,
) -> float:
    """Triplet ICE endpoint factor Ã_t^(k).

    Implements Lang et al. (JCTC 2025) eqs. (65)–(66).

    `occ_case` is determined by the **ket occupancy** at this orbital ("SOMO" or "DOMO").
    """

    occ_case = str(occ_case).upper()
    if occ_case not in ("SOMO", "DOMO"):
        raise ValueError("occ_case must be 'SOMO' or 'DOMO'")

    twos_km1 = int(twos_km1)
    twos_k = int(twos_k)
    twos_km1_p = int(twos_km1_p)
    twos_k_p = int(twos_k_p)
    twos_opline = int(twos_opline)
    if twos_km1 < 0 or twos_k < 0 or twos_km1_p < 0 or twos_k_p < 0 or twos_opline < 0:
        return 0.0

    twos_kmhalf = twos_opline - TWOS_HALF  # 2*(k-1/2)
    if twos_kmhalf < 0:
        return 0.0

    if occ_case == "DOMO":
        # Eq. (65): Ã_t^(k),DOMO = (-1)^{S'_t + S_t - k - 1} * sqrt(2*S'_t + 1)
        #           * { k  1/2  k-1/2 ; S'_{t-1}  S_t  S'_t }
        # DOMO on ket side implies S_t = S_{t-1}.
        if twos_k != twos_km1:
            return 0.0
        phase = phase_from_twos_sum(twos_k_p + twos_k - twos_opline - 2)
        if phase == 0.0:
            return 0.0
        sixj = wigner_6j_twos(twos_opline, TWOS_HALF, twos_kmhalf, twos_km1_p, twos_k, twos_k_p)
        if sixj == 0.0:
            return 0.0
        return float(phase * math.sqrt(float(twos_k_p + 1)) * sixj)

    # Eq. (66): Ã_t^(k),SOMO = (-1)^{S'_{t-1} + S_{t-1} - k + 1/2} * sqrt(2*S_t + 1)
    #           * { k  1/2  k-1/2 ; S_{t-1}  S'_{t-1}  S_t }
    # VMO on bra side implies S'_t = S'_{t-1}.
    if twos_k_p != twos_km1_p:
        return 0.0
    phase = phase_from_twos_sum(twos_km1_p + twos_km1 - twos_opline + TWOS_HALF)
    if phase == 0.0:
        return 0.0
    sixj = wigner_6j_twos(twos_opline, TWOS_HALF, twos_kmhalf, twos_km1, twos_km1_p, twos_k)
    if sixj == 0.0:
        return 0.0
    return float(phase * math.sqrt(float(twos_k + 1)) * sixj)


def Btilde_factor(
    twos_km1: int,
    twos_k: int,
    twos_km1_p: int,
    twos_k_p: int,
    *,
    twos_opline: int,
    occ_case: str,
) -> float:
    """Triplet ICE endpoint factor B̃_t^(k).

    Implements Lang et al. (JCTC 2025) eqs. (65)–(66).

    `occ_case` is determined by the **bra occupancy** at this orbital ("SOMO" or "DOMO").
    """

    occ_case = str(occ_case).upper()
    if occ_case not in ("SOMO", "DOMO"):
        raise ValueError("occ_case must be 'SOMO' or 'DOMO'")

    # Lang et al. show Ã^(k),DOMO == B̃^(k),SOMO and Ã^(k),SOMO == B̃^(k),DOMO.
    mapped = "DOMO" if occ_case == "SOMO" else "SOMO"
    return Atilde_factor(
        int(twos_km1),
        int(twos_k),
        int(twos_km1_p),
        int(twos_k_p),
        twos_opline=int(twos_opline),
        occ_case=mapped,
    )


def T_factor(
    twos_Skm1: int,
    twos_Sk: int,
    twos_Skm1_p: int,
    twos_Sk_p: int,
    twos_opline: int,
) -> float:
    """Triplet ICE propagation factor T_t^(k) through a SOMO pair.

    Implements Lang et al. (JCTC 2025) eq. (67).
    """

    twos_Skm1 = int(twos_Skm1)
    twos_Sk = int(twos_Sk)
    twos_Skm1_p = int(twos_Skm1_p)
    twos_Sk_p = int(twos_Sk_p)
    twos_opline = int(twos_opline)
    if twos_Skm1 < 0 or twos_Sk < 0 or twos_Skm1_p < 0 or twos_Sk_p < 0 or twos_opline < 0:
        return 0.0

    # (-1)^{S'_{t-1} + S_t + k + 1/2}
    phase = phase_from_twos_sum(twos_Skm1_p + twos_Sk + twos_opline + TWOS_HALF)
    if phase == 0.0:
        return 0.0

    # { k  S'_t  S_t ;  1/2  S_{t-1}  S'_{t-1} }
    sixj = wigner_6j_twos(twos_opline, twos_Sk_p, twos_Sk, TWOS_HALF, twos_Skm1, twos_Skm1_p)
    if sixj == 0.0:
        return 0.0

    pref = math.sqrt(float((twos_Sk + 1) * (twos_Sk_p + 1)))
    return float(phase * pref * sixj)

