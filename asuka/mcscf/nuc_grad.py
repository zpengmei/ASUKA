from __future__ import annotations

"""Nuclear gradients for CASCI/CASSCF.

This module provides a default implementation based on DF
gradient kernels in :mod:`asuka.mcscf.nuc_grad_df`.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .nuc_grad_df import (
    DFNucGradMultirootResult,
    DFNucGradResult,
    casscf_nuc_grad_df,
    casscf_nuc_grad_df_per_root,
    casci_nuc_grad_df_relaxed,
    casci_nuc_grad_df_unrelaxed,
)
from .nuc_grad_direct import casscf_nuc_grad_direct, casscf_nuc_grad_direct_per_root
from .nuc_grad_thc import casscf_nuc_grad_thc, casscf_nuc_grad_thc_per_root


@dataclass(frozen=True)
class NucGradResult:
    """Nuclear gradient container (Eh/Bohr).

    Mirrors the historical return type from the PySCF-backed wrappers.

    Attributes
    ----------
    e_tot : float
        Total energy in Hartree.
    e_nuc : float
        Nuclear repulsion energy in Hartree.
    grad : np.ndarray
        Gradient array (natm, 3) in Eh/Bohr.
    """

    e_tot: float
    e_nuc: float
    grad: np.ndarray


def _from_df(res: DFNucGradResult) -> NucGradResult:
    return NucGradResult(
        e_tot=float(res.e_tot),
        e_nuc=float(res.e_nuc),
        grad=np.asarray(res.grad, dtype=np.float64),
    )


def casci_nuc_grad(*args: Any, **kwargs: Any) -> NucGradResult:
    """Compute CASCI nuclear gradients.

    Call pattern
    ------------
    ``casci_nuc_grad(scf_out, casci, *, relaxed=True, **df_kwargs)``

    Delegates to :func:`asuka.mcscf.nuc_grad_df.casci_nuc_grad_df_relaxed` (or
    `_unrelaxed` when ``relaxed=False``).

    Parameters
    ----------
    *args : Any
        Positional arguments (scf_out, casci).
    **kwargs : Any
        Keyword arguments passed to backend.

    Returns
    -------
    NucGradResult
        Gradient result container.

    Raises
    ------
    TypeError
        If arguments do not match the expected pattern.
    """

    if len(args) == 2:
        scf_out, casci = args
        relaxed = bool(kwargs.pop("relaxed", True))
        if relaxed:
            return _from_df(casci_nuc_grad_df_relaxed(scf_out, casci, **kwargs))
        return _from_df(casci_nuc_grad_df_unrelaxed(scf_out, casci, **kwargs))

    if len(args) == 1:
        raise TypeError(
            "casci_nuc_grad no longer supports the legacy PySCF workflow "
            "`casci_nuc_grad(mol, ...)`. Use `casci_nuc_grad(scf_out, casci, ...)` "
            "or call `casci_nuc_grad_df_relaxed/unrelaxed` directly."
        )

    raise TypeError("casci_nuc_grad expects (scf_out, casci, ...)")


def casscf_nuc_grad(*args: Any, **kwargs: Any) -> NucGradResult:
    """Compute CASSCF nuclear gradients.

    Call pattern
    ------------
    ``casscf_nuc_grad(scf_out, casscf, **df_kwargs)``

    Delegates to :func:`asuka.mcscf.nuc_grad_df.casscf_nuc_grad_df`.

    Parameters
    ----------
    *args : Any
        Positional arguments (scf_out, casscf).
    **kwargs : Any
        Keyword arguments passed to backend.

    Returns
    -------
    NucGradResult
        Gradient result container.

    Raises
    ------
    TypeError
        If arguments do not match the expected pattern.
    """

    if len(args) == 2:
        scf_out, casscf = args
        backend = str(kwargs.pop("backend", "auto")).strip().lower()
        if backend not in {"auto", "df", "thc", "direct"}:
            raise ValueError("backend must be one of: 'auto', 'df', 'thc', 'direct'")

        use_direct = bool(backend == "direct")
        use_thc = bool(backend == "thc")
        if backend == "auto":
            use_direct = bool(
                getattr(scf_out, "direct_jk_ctx", None) is not None
                or str(getattr(scf_out, "two_e_backend", "") or "").strip().lower() == "direct"
            )
            if not use_direct:
                # THC-SCF runs do not cache DF factors; default to THC gradients in that case.
                use_thc = getattr(scf_out, "df_B", None) is None and getattr(scf_out, "thc_factors", None) is not None

        if use_direct:
            return _from_df(casscf_nuc_grad_direct(scf_out, casscf, **kwargs))

        if use_thc:
            res = casscf_nuc_grad_thc(scf_out, casscf, **kwargs)
            if isinstance(res, tuple):
                res = res[0]
            return _from_df(res)

        return _from_df(casscf_nuc_grad_df(scf_out, casscf, **kwargs))

    if len(args) == 1:
        raise TypeError(
            "casscf_nuc_grad no longer supports the legacy PySCF workflow "
            "`casscf_nuc_grad(mol, ...)`. Use `casscf_nuc_grad(scf_out, casscf, ...)` "
            "or call `casscf_nuc_grad_df` directly."
        )

    raise TypeError("casscf_nuc_grad expects (scf_out, casscf, ...)")


def casscf_nuc_grad_per_root(*args: Any, **kwargs: Any) -> DFNucGradMultirootResult:
    """Compute per-root SA-CASSCF nuclear gradients.

    Call pattern
    ------------
    ``casscf_nuc_grad_per_root(scf_out, casscf, *, backend='auto', **kwargs)``
    """

    if len(args) == 2:
        scf_out, casscf = args
        backend = str(kwargs.pop("backend", "auto")).strip().lower()
        if backend not in {"auto", "df", "thc", "direct"}:
            raise ValueError("backend must be one of: 'auto', 'df', 'thc', 'direct'")

        use_direct = bool(backend == "direct")
        use_thc = bool(backend == "thc")
        if backend == "auto":
            use_direct = bool(
                getattr(scf_out, "direct_jk_ctx", None) is not None
                or str(getattr(scf_out, "two_e_backend", "") or "").strip().lower() == "direct"
            )
            if not use_direct:
                use_thc = getattr(scf_out, "df_B", None) is None and getattr(scf_out, "thc_factors", None) is not None

        if use_direct:
            return casscf_nuc_grad_direct_per_root(scf_out, casscf, **kwargs)

        import warnings
        warnings.warn(
            "casscf_nuc_grad_per_root: using the DF gradient backend. "
            "The DF per-root gradient path has a known CUDA numerical issue "
            "for systems with virtual orbitals. For production use (e.g. SHARC "
            "dynamics), use two_e_backend='direct' in run_hf_df to enable the "
            "exact 4c gradient path, which is validated against PySCF to ~5e-5.",
            stacklevel=2,
        )
        if use_thc:
            return casscf_nuc_grad_thc_per_root(scf_out, casscf, **kwargs)
        return casscf_nuc_grad_df_per_root(scf_out, casscf, **kwargs)

    if len(args) == 1:
        raise TypeError(
            "casscf_nuc_grad_per_root no longer supports legacy PySCF workflows. "
            "Use `casscf_nuc_grad_per_root(scf_out, casscf, ...)`."
        )

    raise TypeError("casscf_nuc_grad_per_root expects (scf_out, casscf, ...)")


def sacasscf_nonadiabatic_couplings(*args: Any, **kwargs: Any) -> np.ndarray:
    """Compute SA-CASSCF nonadiabatic coupling vectors.

    Call pattern
    ------------
    ``sacasscf_nonadiabatic_couplings(scf_out, casscf, *, backend='auto', **kwargs)``

    When ``backend='auto'``, selects 'direct' (exact 4c) if
    ``scf_out.two_e_backend == 'direct'``, else 'df'.

    Returns
    -------
    np.ndarray
        NACV array indexed as ``result[i, j]`` → ``(natm, 3)``.
    """

    if len(args) != 2:
        raise TypeError("sacasscf_nonadiabatic_couplings expects (scf_out, casscf, ...)")

    scf_out, casscf = args
    backend = str(kwargs.pop("backend", "auto")).strip().lower()
    if backend not in {"auto", "df", "direct"}:
        raise ValueError("backend must be one of: 'auto', 'df', 'direct'")

    use_direct = bool(backend == "direct")
    if backend == "auto":
        use_direct = bool(
            getattr(scf_out, "direct_jk_ctx", None) is not None
            or str(getattr(scf_out, "two_e_backend", "") or "").strip().lower() == "direct"
        )

    if use_direct:
        from .nac._dense import sacasscf_nonadiabatic_couplings_dense  # noqa: PLC0415

        return sacasscf_nonadiabatic_couplings_dense(scf_out, casscf, **kwargs)

    import warnings
    warnings.warn(
        "sacasscf_nonadiabatic_couplings: using the DF backend. "
        "The DF NACV path has a known CUDA numerical issue for systems "
        "with virtual orbitals. For production use (e.g. SHARC dynamics), "
        "use two_e_backend='direct' in run_hf_df to enable the exact 4c "
        "NACV path, which is validated against PySCF to ~1e-5.",
        stacklevel=2,
    )
    from .nac._df import sacasscf_nonadiabatic_couplings_df  # noqa: PLC0415

    return sacasscf_nonadiabatic_couplings_df(scf_out, casscf, **kwargs)


__all__ = [
    "NucGradResult",
    "casci_nuc_grad",
    "casscf_nuc_grad",
    "casscf_nuc_grad_per_root",
    "sacasscf_nonadiabatic_couplings",
]
