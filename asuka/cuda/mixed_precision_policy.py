"""Per-operation precision policy for SCF/CASSCF DF pipeline.

Defines which TF32 error-pruning algorithm to use for each DF operation,
based on numerical sensitivity analysis.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PrecisionPolicy:
    """Per-operation precision policy for SCF/CASSCF DF pipeline.

    Each field specifies the precision mode for a particular operation.
    Valid modes:
        - "fp64": standard FP64 (baseline, ~1e-15)
        - "fp32_kchunked": k-chunked FP32 + FP64 external accum (~1e-5)
        - "tf32_refined": Ozaki-2 + k-chunked main term (~1e-6)
        - "tf32_pure": pure TF32, no correction (~1e-3)
        - "ozaki2": alias for tf32_refined (~1e-6)
        - "ozaki2_kahan": Ozaki-2 + Kahan compensated accumulation
        - "ozaki3": 3-way Ozaki splitting (~1e-9)

    Attributes
    ----------
    metric_cholesky : str
        Auxiliary metric Cholesky. Always FP64 — catastrophic TF32 error.
    whitening_trsm : str
        Whitening triangular solve. Always FP64 (uses FP64 L).
    df_j_projection : str
        J Coulomb projection GEMV. Medium sensitivity.
    df_j_matrix : str
        J matrix assembly GEMV. Medium sensitivity.
    df_k_syrk : str
        K exchange SYRK accumulation. High sensitivity.
    active_space_df : str
        Active-space DF 2-stage contraction. Medium sensitivity.
    eri_matrix : str
        ERI matrix from l_full. Medium-high sensitivity.
    """

    metric_cholesky: str = "fp64"
    whitening_trsm: str = "fp64"
    df_j_projection: str = "tf32_refined"
    df_j_matrix: str = "tf32_refined"
    df_k_syrk: str = "ozaki2_kahan"
    active_space_df: str = "tf32_refined"
    eri_matrix: str = "ozaki2"

    # ERI kernel-level precision controls (Phase 1-3 mixed-precision).
    eri_tile_dtype: str = "fp64"        # "fp64" | "fp32"
    eri_component_mode: str = "fp64"    # "fp64" | "mixed"
    eri_accumulator_mode: str = "fp64"  # "fp64" | "fp32_light"

    def __post_init__(self) -> None:
        valid = {"fp64", "fp32_kchunked", "tf32_refined", "tf32_pure", "ozaki2", "ozaki2_kahan", "ozaki3"}
        for fld_name in (
            "metric_cholesky",
            "whitening_trsm",
            "df_j_projection",
            "df_j_matrix",
            "df_k_syrk",
            "active_space_df",
            "eri_matrix",
        ):
            val = getattr(self, fld_name)
            if val not in valid:
                raise ValueError(
                    f"PrecisionPolicy.{fld_name}={val!r} is not valid. "
                    f"Must be one of: {sorted(valid)}"
                )
        # Validate ERI kernel precision fields.
        if self.eri_tile_dtype not in ("fp64", "fp32"):
            raise ValueError(f"PrecisionPolicy.eri_tile_dtype={self.eri_tile_dtype!r} must be 'fp64' or 'fp32'")
        if self.eri_component_mode not in ("fp64", "mixed"):
            raise ValueError(f"PrecisionPolicy.eri_component_mode={self.eri_component_mode!r} must be 'fp64' or 'mixed'")
        if self.eri_accumulator_mode not in ("fp64", "fp32_light"):
            raise ValueError(f"PrecisionPolicy.eri_accumulator_mode={self.eri_accumulator_mode!r} must be 'fp64' or 'fp32_light'")

    @classmethod
    def fp64(cls) -> PrecisionPolicy:
        """All-FP64 baseline policy (no TF32)."""
        return cls(
            metric_cholesky="fp64",
            whitening_trsm="fp64",
            df_j_projection="fp64",
            df_j_matrix="fp64",
            df_k_syrk="fp64",
            active_space_df="fp64",
            eri_matrix="fp64",
            eri_tile_dtype="fp64",
            eri_component_mode="fp64",
            eri_accumulator_mode="fp64",
        )

    @classmethod
    def tf32_conservative(cls) -> PrecisionPolicy:
        """Conservative TF32 policy: refinement everywhere except critical ops."""
        return cls(
            metric_cholesky="fp64",
            whitening_trsm="fp64",
            df_j_projection="tf32_refined",
            df_j_matrix="tf32_refined",
            df_k_syrk="ozaki2_kahan",
            active_space_df="tf32_refined",
            eri_matrix="ozaki2",
            eri_tile_dtype="fp64",
            eri_component_mode="fp64",
            eri_accumulator_mode="fp64",
        )

    @classmethod
    def tf32_aggressive(cls) -> PrecisionPolicy:
        """Aggressive TF32 policy: pure TF32 for SCF-convergent ops.

        Use when SCF self-correction absorbs per-step errors (J/K per-iteration).
        Best with Delta-D rebasing (periodic FP64 recalibration).
        """
        return cls(
            metric_cholesky="fp64",
            whitening_trsm="fp64",
            df_j_projection="tf32_pure",
            df_j_matrix="tf32_pure",
            df_k_syrk="tf32_pure",
            active_space_df="tf32_refined",
            eri_matrix="ozaki2",
            eri_tile_dtype="fp64",
            eri_component_mode="fp64",
            eri_accumulator_mode="fp64",
        )

    @classmethod
    def eri_conservative(cls) -> PrecisionPolicy:
        """Conservative ERI mixed-precision: FP32 tile + mixed components, FP64 accum.

        Phase 1+2: Halves tile bandwidth and gets FP32 throughput on component
        evaluation while maintaining FP64 accumulation. Total J/K error < 1e-6.
        """
        return cls(
            metric_cholesky="fp64",
            whitening_trsm="fp64",
            df_j_projection="fp64",
            df_j_matrix="fp64",
            df_k_syrk="fp64",
            active_space_df="fp64",
            eri_matrix="fp64",
            eri_tile_dtype="fp32",
            eri_component_mode="mixed",
            eri_accumulator_mode="fp64",
        )

    @classmethod
    def eri_aggressive(cls) -> PrecisionPolicy:
        """Aggressive ERI mixed-precision: + FP32 accum for light quartets.

        Phase 3: Additionally uses FP32 accumulators for light quartets
        (nPairAB * nPairCD * NROOTS < 50). SCF energy error < 1e-7 Eh.
        NOT safe for gradients.
        """
        return cls(
            metric_cholesky="fp64",
            whitening_trsm="fp64",
            df_j_projection="fp64",
            df_j_matrix="fp64",
            df_k_syrk="fp64",
            active_space_df="fp64",
            eri_matrix="fp64",
            eri_tile_dtype="fp32",
            eri_component_mode="mixed",
            eri_accumulator_mode="fp32_light",
        )

    @classmethod
    def from_env(cls) -> PrecisionPolicy:
        """Construct from environment variables.

        Reads ASUKA_PRECISION_POLICY (one of: "fp64", "tf32_conservative",
        "tf32_aggressive") and per-operation overrides like
        ASUKA_PRECISION_DF_J="tf32_pure".
        """
        base_name = os.environ.get("ASUKA_PRECISION_POLICY", "fp64").strip().lower()
        if base_name == "fp64":
            policy = cls.fp64()
        elif base_name in ("tf32", "tf32_conservative"):
            policy = cls.tf32_conservative()
        elif base_name == "tf32_aggressive":
            policy = cls.tf32_aggressive()
        elif base_name == "eri_conservative":
            policy = cls.eri_conservative()
        elif base_name == "eri_aggressive":
            policy = cls.eri_aggressive()
        else:
            policy = cls.fp64()

        # Per-operation overrides.
        overrides = {
            "ASUKA_PRECISION_DF_J": ("df_j_projection", "df_j_matrix"),
            "ASUKA_PRECISION_DF_K": ("df_k_syrk",),
            "ASUKA_PRECISION_ACTIVE_DF": ("active_space_df",),
            "ASUKA_PRECISION_ERI": ("eri_matrix",),
        }
        for env_key, fields in overrides.items():
            val = os.environ.get(env_key)
            if val is not None:
                val = val.strip().lower()
                for f in fields:
                    object.__setattr__(policy, f, val)

        # ERI kernel-level overrides (mixed precision ON by default).
        eri_mixed = os.environ.get("ASUKA_ERI_MIXED_PRECISION", "1")
        if eri_mixed.strip() != "0":
            object.__setattr__(policy, "eri_component_mode", "mixed")

        eri_tile = os.environ.get("ASUKA_ERI_TILE_F32")
        if eri_tile is not None and eri_tile.strip() == "1":
            object.__setattr__(policy, "eri_tile_dtype", "fp32")

        eri_comp = os.environ.get("ASUKA_ERI_COMPONENT_MODE")
        if eri_comp is not None:
            object.__setattr__(policy, "eri_component_mode", eri_comp.strip().lower())

        policy.__post_init__()  # Validate.
        return policy

    def summary(self) -> dict[str, str]:
        """Return a dict of operation -> precision mode."""
        return {
            "metric_cholesky": self.metric_cholesky,
            "whitening_trsm": self.whitening_trsm,
            "df_j_projection": self.df_j_projection,
            "df_j_matrix": self.df_j_matrix,
            "df_k_syrk": self.df_k_syrk,
            "active_space_df": self.active_space_df,
            "eri_matrix": self.eri_matrix,
            "eri_tile_dtype": self.eri_tile_dtype,
            "eri_component_mode": self.eri_component_mode,
            "eri_accumulator_mode": self.eri_accumulator_mode,
        }


__all__ = ["PrecisionPolicy"]
