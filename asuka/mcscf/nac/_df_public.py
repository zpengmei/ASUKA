from __future__ import annotations

"""SA-CASSCF nonadiabatic couplings (DF).

This module exposes a *production* NAC driver with a small,
stable API surface.

Implementation details
--------------------
The underlying implementation lives in :mod:`asuka.mcscf.nac_df`. That module
also contains finite-difference Jacobian backends used for parity validation.
Here we intentionally expose only the recommended response term:

  - ``response_term="split_orbfd"`` (default)
"""

from typing import Any, Literal, Sequence

import numpy as np

from ._df import sacasscf_nonadiabatic_couplings_df as _impl
from asuka.solver import GUGAFCISolver

_ResponseTerm = Literal["split_orbfd"]


def sacasscf_nonadiabatic_couplings_df(
    scf_out: Any,
    casscf: Any,
    *,
    pairs: Sequence[tuple[int, int]] | None = None,
    atmlst: Sequence[int] | None = None,
    use_etfs: bool = False,
    mult_ediff: bool = False,
    fcisolver: GUGAFCISolver | None = None,
    twos: int | None = None,
    df_backend: Literal["cpu", "cuda"] = "cpu",
    df_config: Any | None = None,
    df_threads: int = 0,
    response_term: _ResponseTerm = "split_orbfd",
    z_tol: float = 1e-10,
    z_maxiter: int = 200,
    delta_bohr: float = 1e-4,
) -> np.ndarray:
    """Compute SA-CASSCF NACVs (<bra|d/dR|ket>) using ASUKA DF integrals.

    This is the public NAC entry point. It supports only the
    ``split_orbfd`` response backend. ``delta_bohr`` controls the finite-
    difference step only if the DF derivative contraction falls back to FD.
    """

    if str(response_term).lower() != "split_orbfd":
        raise NotImplementedError("Public DF NAC only supports response_term='split_orbfd'.")

    return _impl(
        scf_out,
        casscf,
        pairs=pairs,
        atmlst=atmlst,
        use_etfs=use_etfs,
        mult_ediff=mult_ediff,
        fcisolver=fcisolver,
        twos=twos,
        df_backend=df_backend,
        df_config=df_config,
        df_threads=df_threads,
        response_term="split_orbfd",
        z_tol=z_tol,
        z_maxiter=z_maxiter,
        delta_bohr=delta_bohr,
    )


__all__ = ["sacasscf_nonadiabatic_couplings_df"]
