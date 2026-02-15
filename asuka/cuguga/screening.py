from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RowScreening:
    """Optional tuning knobs for sparse row oracles.

    All thresholds are absolute-value thresholds and default to 0.0 (disabled),
    meaning the oracle should match the unscreened reference up to FP roundoff.

    Notes
    -----
    These are heuristic pruning/acceleration controls. They are intended for performance /
    scaling studies and for approximate workflows.
    They do not guarantee a bound on the total error unless paired
    with additional norm-based logic (not implemented here).
    """

    thresh_h1e: float = 0.0
    thresh_rs_coeff: float = 0.0
    thresh_rs_pairnorm: float = 0.0
    thresh_gpq: float = 0.0
    thresh_contrib: float = 0.0
    # DF-only acceleration knob: if >0, allow materializing the DF ERI matrix
    # in pair space (size (norb^2)^2) up to this many bytes.
    df_eri_mat_max_bytes: int = 0

    def __post_init__(self) -> None:
        for name in (
            "thresh_h1e",
            "thresh_rs_coeff",
            "thresh_rs_pairnorm",
            "thresh_gpq",
            "thresh_contrib",
        ):
            val = float(getattr(self, name))
            if val < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if int(self.df_eri_mat_max_bytes) < 0:
            raise ValueError("df_eri_mat_max_bytes must be >= 0")

    @property
    def enabled(self) -> bool:
        return any(
            float(getattr(self, name)) > 0.0
            for name in (
                "thresh_h1e",
                "thresh_rs_coeff",
                "thresh_rs_pairnorm",
                "thresh_gpq",
                "thresh_contrib",
            )
        ) or int(self.df_eri_mat_max_bytes) > 0
