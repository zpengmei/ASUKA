from __future__ import annotations

"""Convenience wrapper that computes SA-CASSCF energies, forces, and NACVs in one call.

Typical usage::

    result = sacasscf_properties(scf_out, mc, df_backend="cuda")
    result.e_roots      # (nroots,)        energies in Eh
    result.forces       # (nroots, natm, 3) forces  in Eh/Bohr  (= -grads)
    result.nacvs        # (nroots, nroots, natm, 3) NACVs in 1/Bohr
"""

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np


@dataclass(frozen=True)
class SACASSCFPropertiesResult:
    """Energies, per-root forces, and non-adiabatic coupling vectors for SA-CASSCF.

    Attributes
    ----------
    e_roots : np.ndarray
        Per-root energies, shape ``(nroots,)`` in Eh.
    e_sa : float
        State-averaged energy in Eh.
    e_nuc : float
        Nuclear repulsion energy in Eh.
    root_weights : np.ndarray
        SA averaging weights, shape ``(nroots,)``.
    grads : np.ndarray or None
        Per-root nuclear gradients dE_I/dR, shape ``(nroots, natm, 3)`` in Eh/Bohr.
        ``None`` when ``compute_grads=False``.
    grad_sa : np.ndarray or None
        State-averaged gradient, shape ``(natm, 3)`` in Eh/Bohr.
        ``None`` when ``compute_grads=False``.
    nacvs : np.ndarray or None
        Non-adiabatic coupling vectors ``<I|∂/∂R|J>``,
        shape ``(nroots, nroots, natm, 3)`` in 1/Bohr.
        Diagonal entries are zero.  ``None`` when ``compute_nacvs=False``.
    nacv_pairs : list of (int, int) or None
        The (bra, ket) pairs that were computed.  Off-diagonal entries not in
        this list are zero in ``nacvs``.  ``None`` when ``compute_nacvs=False``.
    """

    e_roots: np.ndarray
    e_sa: float
    e_nuc: float
    root_weights: np.ndarray
    grads: np.ndarray | None
    grad_sa: np.ndarray | None
    nacvs: np.ndarray | None
    nacv_pairs: list[tuple[int, int]] | None

    @property
    def forces(self) -> np.ndarray | None:
        """Per-root forces ``-dE_I/dR``, shape ``(nroots, natm, 3)`` in Eh/Bohr."""
        if self.grads is None:
            return None
        return -self.grads


def sacasscf_properties(
    scf_out: Any,
    casscf: Any,
    *,
    compute_grads: bool = True,
    compute_nacvs: bool = False,
    nacv_pairs: Sequence[tuple[int, int]] | None = None,
    mult_ediff: bool = False,
    use_etfs: bool = False,
    df_backend: Literal["cpu", "cuda", "auto"] = "auto",
    int1e_backend: Literal["auto", "cpu", "cuda"] = "auto",
    df_threads: int = 0,
    z_tol: float = 1e-10,
    z_maxiter: int = 200,
    nacv_response_term: Literal["split_orbfd", "fd_jacobian"] = "split_orbfd",
    grad_kwargs: dict[str, Any] | None = None,
    nacv_kwargs: dict[str, Any] | None = None,
) -> SACASSCFPropertiesResult:
    """Compute SA-CASSCF energies, per-root forces, and NACVs in one call.

    This is a convenience wrapper around :func:`casscf_nuc_grad_df_per_root`
    and :func:`sacasscf_nonadiabatic_couplings_df`.  It collects all
    quantities needed for non-adiabatic molecular dynamics into a single
    :class:`SACASSCFPropertiesResult`.

    Parameters
    ----------
    scf_out
        DF-SCF result (provides DF tensors, ``mol``, etc.).
    casscf
        SA-CASSCF result (``mo_coeff``, ``ci``, ``e_roots``, etc.).
    compute_grads : bool, default ``True``
        Whether to compute per-root nuclear gradients.
    compute_nacvs : bool, default ``True``
        Whether to compute non-adiabatic coupling vectors.
    nacv_pairs : list of (bra, ket) pairs, optional
        State pairs for NACVs.  ``None`` computes all off-diagonal pairs.
    mult_ediff : bool, default ``False``
        If ``True``, NACVs are multiplied by the energy gap ``E_J - E_I``
        (returns the numerator ``<I|∂H/∂R|J>`` instead of the coupling).
    use_etfs : bool, default ``False``
        Include electron-translation-factor corrections to NACVs.
    df_backend : ``"cpu"``, ``"cuda"``, or ``"auto"``, default ``"auto"``
        Backend for DF 2e derivative contractions.  ``"auto"`` detects from
        ``scf_out``: uses ``"cuda"`` if the DF tensor is a CuPy array,
        otherwise ``"cpu"``.
    int1e_backend : ``"auto"``, ``"cpu"``, or ``"cuda"``, default ``"auto"``
        Backend for 1e AO derivative contractions (hcore, overlap).
        ``"auto"`` picks CUDA fused kernels when available, CPU otherwise.
    df_threads : int, default ``0``
        Number of threads for CPU DF backend.
    z_tol : float, default ``1e-10``
        Convergence tolerance for the CP-MCSCF Z-vector solve.
    z_maxiter : int, default ``200``
        Maximum iterations for the Z-vector solve.
    nacv_response_term : ``"split_orbfd"`` or ``"fd_jacobian"``, default ``"split_orbfd"``
        Response backend for NACVs.
    grad_kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`casscf_nuc_grad_df_per_root`.
    nacv_kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`sacasscf_nonadiabatic_couplings_df`.

    Returns
    -------
    SACASSCFPropertiesResult
        Container with ``.e_roots``, ``.forces`` (``-grads``), ``.nacvs``,
        ``.root_weights``, ``.e_sa``, ``.e_nuc``, ``.grad_sa``.

    Examples
    --------
    >>> from asuka.frontend import Molecule, run_hf
    >>> from asuka.mcscf import run_casscf
    >>> from asuka.mcscf.properties import sacasscf_properties
    >>>
    >>> mol = Molecule.from_atoms([("O",(0,0,0)),("H",(0,1.43,-0.89)),("H",(0,-1.43,-0.89))],
    ...                           unit="Bohr", basis="sto-3g")
    >>> scf = run_hf(mol, backend="cuda")
    >>> mc  = run_casscf(scf, ncore=4, ncas=2, nelecas=2, nroots=3, backend="cuda")
    >>> res = sacasscf_properties(scf, mc, df_backend="cuda")
    >>> res.e_roots        # energies for all 3 states
    >>> res.forces         # (3, natm, 3) forces
    >>> res.nacvs          # (3, 3, natm, 3) NACVs
    """
    from asuka.mcscf.nuc_grad_df import casscf_nuc_grad_df_per_root
    from asuka.mcscf.nac._df import sacasscf_nonadiabatic_couplings_df
    from asuka.mcscf.state_average import normalize_weights

    grad_kwargs = dict(grad_kwargs or {})
    nacv_kwargs = dict(nacv_kwargs or {})

    # ── resolve df_backend ───────────────────────────────────────────────────
    if str(df_backend) == "auto":
        try:
            import cupy as cp  # type: ignore[import-not-found]
            _b = getattr(scf_out, "df_B", None) or getattr(scf_out, "B_ao", None)
            df_backend = "cuda" if (_b is not None and isinstance(_b, cp.ndarray)) else "cpu"
        except Exception:
            df_backend = "cpu"

    # ── energies (always available) ──────────────────────────────────────────
    e_raw = getattr(casscf, "e_roots", None)
    if e_raw is None:
        e_raw = getattr(casscf, "e_states", None)
    if e_raw is None:
        raise ValueError("casscf must expose per-root energies as .e_roots or .e_states")
    e_roots = np.asarray(e_raw, dtype=np.float64).ravel()
    nroots = int(e_roots.size)

    weights_raw = getattr(casscf, "root_weights", None)
    if weights_raw is None:
        weights_raw = getattr(casscf, "weights", None)
    root_weights = normalize_weights(weights_raw, nroots=nroots)
    e_sa = float(np.dot(root_weights, e_roots))

    mol = getattr(scf_out, "mol", None)
    if mol is None:
        raise TypeError("scf_out must have a .mol attribute")

    # nuclear repulsion
    e_nuc = float(mol.energy_nuc())

    # ── per-root gradients ───────────────────────────────────────────────────
    grads: np.ndarray | None = None
    grad_sa: np.ndarray | None = None
    if compute_grads:
        _grad_result = casscf_nuc_grad_df_per_root(
            scf_out,
            casscf,
            df_backend=df_backend,
            int1e_contract_backend=int1e_backend,
            df_threads=df_threads,
            z_tol=z_tol,
            z_maxiter=z_maxiter,
            **grad_kwargs,
        )
        grads = np.asarray(_grad_result.grads, dtype=np.float64)
        grad_sa = np.asarray(_grad_result.grad_sa, dtype=np.float64)
        e_nuc = float(_grad_result.e_nuc)

    # ── NACVs ────────────────────────────────────────────────────────────────
    nacvs: np.ndarray | None = None
    computed_pairs: list[tuple[int, int]] | None = None
    if compute_nacvs and nroots > 1:
        _pairs_use: list[tuple[int, int]] | None
        if nacv_pairs is not None:
            _pairs_use = [tuple(p) for p in nacv_pairs]  # type: ignore[misc]
        else:
            _pairs_use = None  # all off-diagonal

        _nacv_raw = sacasscf_nonadiabatic_couplings_df(
            scf_out,
            casscf,
            pairs=_pairs_use,
            mult_ediff=bool(mult_ediff),
            use_etfs=bool(use_etfs),
            df_backend=df_backend,
            df_threads=df_threads,
            z_tol=z_tol,
            z_maxiter=z_maxiter,
            response_term=nacv_response_term,
            **nacv_kwargs,
        )
        # sacasscf_nonadiabatic_couplings_df returns (nroots, nroots, natm, 3)
        nacvs = np.asarray(_nacv_raw, dtype=np.float64)

        # record which pairs were computed
        natm = int(nacvs.shape[2]) if nacvs.ndim == 4 else 0
        if _pairs_use is not None:
            computed_pairs = _pairs_use
        else:
            computed_pairs = [
                (I, J) for I in range(nroots) for J in range(nroots) if I != J
            ]
    elif compute_nacvs and nroots <= 1:
        # single state: no couplings
        natm_guess = int(getattr(mol, "natm", 0))
        nacvs = np.zeros((nroots, nroots, natm_guess, 3), dtype=np.float64)
        computed_pairs = []

    return SACASSCFPropertiesResult(
        e_roots=e_roots,
        e_sa=e_sa,
        e_nuc=e_nuc,
        root_weights=root_weights,
        grads=grads,
        grad_sa=grad_sa,
        nacvs=nacvs,
        nacv_pairs=computed_pairs,
    )
