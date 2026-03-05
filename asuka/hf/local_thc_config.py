from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LocalTHCConfig:
    """Configuration for local-THC factor construction.

    This follows the spirit of Song & Martinez (JCP 146, 034104 (2017)):
    - Partition atoms into blocks (primary set).
    - For each block, include a secondary set from neighboring blocks
      (overlap-based). For HF J/K assembly we include both earlier and later
      blocks for density contraction, but apply an ownership mask on the output
      Fock elements to avoid double counting.
    - Build per-block THC factors (X,Z) on a local fitting region controlled by
      a Schwarz-bound threshold.
    """

    # Blocking
    block_max_ao: int = 256

    # Auxiliary region selection (Schwarz threshold, a.u.)
    aux_schwarz_thr: float = 5e-2

    # Secondary selection: keep later-block atoms that overlap with the primary
    # AO set above this threshold (|S|).
    sec_overlap_thr: float = 1e-6

    # Per-block number of grid points:
    # - If `thc_npt` is provided at call site, it overrides this.
    # - Otherwise use npt = clamp(npt_factor * n_primary_ao, npt_min, npt_max).
    npt_factor: int = 8
    npt_min: int = 512
    npt_max: int = 2048

    # Central metric storage dtype: 'float64' or 'float32'
    z_dtype: str = "float64"


__all__ = ["LocalTHCConfig"]
