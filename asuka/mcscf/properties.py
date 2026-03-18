from __future__ import annotations

"""Convenience wrapper that computes SA-CASSCF energies, forces, and NACVs in one call.

Typical usage::

    result = sacasscf_properties(scf_out, mc, df_backend="cuda")
    result.e_roots      # (nroots,)        energies in Eh
    result.forces       # (nroots, natm, 3) forces  in Eh/Bohr  (= -grads)
    result.nacvs        # (nroots, nroots, natm, 3) NACVs in 1/Bohr
    result.ci           # (nroots, ncsf)   CI vectors (for time-derivative coupling)
    result.mo_coeff     # (nao, nmo)       MO coefficients
    result.sigma        # (nroots, nroots) <ψ_I(t-dt)|ψ_J(t)> overlaps (from callback only)
"""

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np


def _maybe_asnumpy(x: Any) -> Any:
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except Exception:
        cp = None

    if cp is not None and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    if isinstance(x, list):
        return [_maybe_asnumpy(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_maybe_asnumpy(v) for v in x)
    return x


def _materialize_array(x: Any, *, dtype: Any | None = None) -> np.ndarray:
    return np.asarray(_maybe_asnumpy(x), dtype=dtype)


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
    ci : np.ndarray or None
        CI vectors in the CSF basis, shape ``(nroots, ncsf)``.
        Useful for computing time-derivative couplings
        ``σ_IJ = <ψ_I(t-dt)|ψ_J(t)>`` between consecutive NAMD steps.
    mo_coeff : np.ndarray or None
        MO coefficient matrix, shape ``(nao, nmo)``.
    sigma : np.ndarray or None
        CI-vector overlap matrix ``σ_IJ = <ψ_I(t-dt)|ψ_J(t)>``,
        shape ``(nroots, nroots)``.  Populated by :func:`make_df_sacasscf_properties_eval`
        on the second and subsequent trajectory steps; ``None`` otherwise.
        This is the primary coupling quantity for FSSH propagation —
        phase-invariant and finite at conical intersections.
    """

    e_roots: np.ndarray
    e_sa: float
    e_nuc: float
    root_weights: np.ndarray
    grads: np.ndarray | None
    grad_sa: np.ndarray | None
    nacvs: np.ndarray | None
    nacv_pairs: list[tuple[int, int]] | None
    ci: np.ndarray | None = None
    mo_coeff: np.ndarray | None = None
    sigma: np.ndarray | None = None

    @property
    def forces(self) -> np.ndarray | None:
        """Per-root forces ``-dE_I/dR``, shape ``(nroots, natm, 3)`` in Eh/Bohr."""
        if self.grads is None:
            return None
        return -self.grads


_VALID_GRAD_BACKENDS = {"auto", "df", "dense", "direct", "direct_df", "thc"}

# Both "dense" (precomputed ao_eri) and "direct" (on-the-fly 4c via
# direct_jk_ctx) use the same gradient driver (nuc_grad_direct) which
# always computes ERI derivatives on-the-fly.  We canonicalise both to
# "_4c" internally so the dispatcher has a single code-path.
_4C = "_4c"  # internal sentinel — not user-facing


def _resolve_grad_backend(
    grad_backend: str,
    scf_out: Any,
) -> str:
    """Resolve *grad_backend* to a canonical value.

    Returns one of ``"df"``, ``"_4c"``, ``"direct_df"``, ``"thc"``.

    Mapping
    -------
    - ``"df"``        → DF 3-centre derivative contractions (needs ``scf_out.df_B``).
    - ``"dense"``     → 4-centre ERI derivative path (precomputed ``ao_eri``).
    - ``"direct"``    → 4-centre ERI derivative path (integral-direct ``direct_jk_ctx``).
    - ``"direct_df"`` → DF gradient with on-the-fly ``df_B`` materialisation
                         (for ``direct_df`` SCF results that have ``df_L`` but no ``df_B``).
    - ``"thc"``       → THC gradient (needs ``scf_out.thc_factors``).
    - ``"auto"``      → Detect from ``scf_out.two_e_backend`` and available tensors.
    """
    grad_backend = str(grad_backend).strip().lower()

    if grad_backend in ("dense", "direct"):
        return _4C
    if grad_backend in ("df", "direct_df", "thc"):
        return grad_backend
    if grad_backend != "auto":
        raise ValueError(
            f"grad_backend must be one of {sorted(_VALID_GRAD_BACKENDS)}, got {grad_backend!r}"
        )

    # ── auto-detect ──────────────────────────────────────────────────────────
    two_e = str(getattr(scf_out, "two_e_backend", "") or "").strip().lower()

    if two_e == "direct_df":
        return "direct_df"
    if two_e in ("direct", "dense") or getattr(scf_out, "direct_jk_ctx", None) is not None:
        return _4C
    if getattr(scf_out, "thc_factors", None) is not None and getattr(scf_out, "df_B", None) is None:
        return "thc"
    return "df"


def _ensure_df_B(scf_out: Any) -> Any:
    """Return *scf_out* with ``df_B`` materialised if it was ``None``.

    ``direct_df`` SCF results store only the Cholesky factor ``df_L`` to avoid
    the O(nao^2 * naux) memory footprint during the SCF iterations.  The DF
    gradient path, however, needs the full ``df_B`` tensor.  This helper builds
    it on-the-fly and attaches it to ``scf_out`` (mutating in-place for
    subsequent calls).
    """
    if getattr(scf_out, "df_B", None) is not None:
        return scf_out

    ao_basis = getattr(scf_out, "ao_basis", None)
    aux_basis = getattr(scf_out, "aux_basis", None)
    df_L = getattr(scf_out, "df_L", None)
    if ao_basis is None or aux_basis is None or df_L is None:
        raise ValueError(
            "direct_df grad_backend requires scf_out to have ao_basis, aux_basis, and df_L "
            "(Cholesky factor) so that df_B can be materialised for the DF gradient."
        )

    mol = getattr(scf_out, "mol", None)
    is_sph = mol is not None and not bool(getattr(mol, "cart", True))
    ao_rep = "sph" if is_sph else "cart"

    try:
        import cupy as cp  # type: ignore[import-not-found]
        _has_gpu = cp is not None
    except Exception:
        _has_gpu = False

    if _has_gpu:
        from asuka.integrals.cueri_df import build_df_B_from_cueri_packed_bases  # noqa: PLC0415

        B, _L = build_df_B_from_cueri_packed_bases(
            ao_basis,
            aux_basis,
            layout="mnQ",
            ao_rep=ao_rep,
            return_L=True,
        )
    else:
        from asuka.integrals.cueri_df_cpu import build_df_B_from_cueri_packed_bases_cpu  # noqa: PLC0415

        B = build_df_B_from_cueri_packed_bases_cpu(
            ao_basis,
            aux_basis,
            layout="mnQ",
            ao_rep=ao_rep,
        )

    # Try packed Qp layout for GPU path.
    try:
        import cupy as _cp  # type: ignore[import-not-found]
        from asuka.integrals.df_packed_s2 import ao_packed_s2_enabled, pack_B_to_Qp  # noqa: PLC0415

        if isinstance(B, _cp.ndarray) and B.ndim == 3 and ao_packed_s2_enabled():
            int1e = getattr(scf_out, "int1e", None)
            nao = int(int1e.S.shape[0]) if int1e is not None else int(B.shape[0])
            B = pack_B_to_Qp(B, layout="mnQ", nao=nao)
    except Exception:
        pass

    # Attach to scf_out for reuse.
    try:
        scf_out.df_B = B
    except (AttributeError, TypeError):
        # Frozen dataclass — wrap in a simple namespace.
        import types  # noqa: PLC0415
        ns = types.SimpleNamespace(**{k: getattr(scf_out, k) for k in dir(scf_out) if not k.startswith("_")})
        ns.df_B = B
        return ns

    return scf_out


def _dispatch_per_root_grad(
    grad_backend: str,
    scf_out: Any,
    casscf: Any,
    *,
    grad_roots: Sequence[int] | None,
    z_tol: float,
    z_maxiter: int,
    grad_kwargs: dict[str, Any],
) -> Any:
    """Call the appropriate per-root gradient function for *grad_backend*."""
    resolved = _resolve_grad_backend(grad_backend, scf_out)

    common = dict(
        z_tol=float(z_tol),
        z_maxiter=int(z_maxiter),
        grad_roots=grad_roots,
    )

    if resolved == _4C:
        from asuka.mcscf.nuc_grad_direct import casscf_nuc_grad_direct_per_root  # noqa: PLC0415

        kw = {**common, **grad_kwargs}
        for _k in ("df_backend", "int1e_contract_backend", "df_threads"):
            kw.pop(_k, None)
        return casscf_nuc_grad_direct_per_root(scf_out, casscf, **kw)

    if resolved == "thc":
        from asuka.mcscf.nuc_grad_thc import casscf_nuc_grad_thc_per_root  # noqa: PLC0415

        kw = {**common, **grad_kwargs}
        for _k in ("df_backend", "direct_eri_deriv_backend"):
            kw.pop(_k, None)
        return casscf_nuc_grad_thc_per_root(scf_out, casscf, **kw)

    if resolved == "direct_df":
        scf_out = _ensure_df_B(scf_out)

    # DF path (also handles direct_df after df_B materialisation)
    from asuka.mcscf.nuc_grad_df import casscf_nuc_grad_df_per_root  # noqa: PLC0415

    kw = {**common, **grad_kwargs}
    return casscf_nuc_grad_df_per_root(scf_out, casscf, **kw)


def sacasscf_properties(
    scf_out: Any,
    casscf: Any,
    *,
    compute_grads: bool = True,
    compute_nacvs: bool = False,
    grad_roots: Sequence[int] | None = None,
    grad_backend: Literal["auto", "df", "dense", "direct", "direct_df", "thc"] = "auto",
    nacv_pairs: Sequence[tuple[int, int]] | None = None,
    mult_ediff: bool = False,
    use_etfs: bool = False,
    df_backend: Literal["cpu", "cuda", "auto"] = "auto",
    int1e_backend: Literal["auto", "cpu", "cuda"] = "auto",
    df_threads: int = 0,
    z_tol: float = 1e-10,
    z_maxiter: int = 100,
    nacv_response_term: Literal["split_orbfd", "fd_jacobian"] = "split_orbfd",
    grad_kwargs: dict[str, Any] | None = None,
    nacv_kwargs: dict[str, Any] | None = None,
) -> SACASSCFPropertiesResult:
    """Compute SA-CASSCF energies, per-root forces, and NACVs in one call.

    This is a convenience wrapper around the per-root gradient and NACV
    drivers.  It collects all quantities needed for non-adiabatic molecular
    dynamics into a single :class:`SACASSCFPropertiesResult`.

    Parameters
    ----------
    scf_out
        SCF result object (provides DF tensors / direct JK context / THC factors).
    casscf
        SA-CASSCF result (``mo_coeff``, ``ci``, ``e_roots``, etc.).
    compute_grads : bool, default ``True``
        Whether to compute per-root nuclear gradients.
    compute_nacvs : bool, default ``False``
        Whether to compute non-adiabatic coupling vectors.
    grad_roots : sequence of int or None, default ``None``
        If given, only compute gradients for the listed root indices.
        Skipped roots get zero gradients.  Useful in dynamics when only
        the active-state gradient is needed (e.g. ``grad_roots=[1]``
        for 3 states computes only root 1's gradient).
    grad_backend : str, default ``"auto"``
        Two-electron integral backend for per-root gradients.  Mirrors the
        SCF-level ``two_e_backend`` vocabulary.

        - ``"df"``        — DF 3-centre derivative contractions (requires ``scf_out.df_B``).
        - ``"direct"``    — On-the-fly 4-centre ERI derivatives (requires
                            ``scf_out.direct_jk_ctx``).
        - ``"dense"``     — 4-centre ERI derivatives from precomputed ERIs (requires
                            ``scf_out.ao_eri`` or ``scf_out.direct_jk_ctx``).
        - ``"direct_df"`` — DF gradient path with on-the-fly ``df_B`` materialisation
                            (for ``direct_df`` SCF results that store only ``df_L``).
        - ``"thc"``       — Tensor hypercontraction (requires ``scf_out.thc_factors``).
        - ``"auto"``      — Detect from ``scf_out.two_e_backend`` and available tensors.
    nacv_pairs : list of (bra, ket) pairs, optional
        State pairs for NACVs.  ``None`` computes all off-diagonal pairs.
    mult_ediff : bool, default ``False``
        If ``True``, NACVs are multiplied by the energy gap ``E_J - E_I``
        (returns the numerator ``<I|∂H/∂R|J>`` instead of the coupling).
    use_etfs : bool, default ``False``
        Include electron-translation-factor corrections to NACVs.
    df_backend : ``"cpu"``, ``"cuda"``, or ``"auto"``, default ``"auto"``
        Device backend for DF 2e derivative contractions (only used when
        ``grad_backend="df"``).  ``"auto"`` detects from ``scf_out``:
        uses ``"cuda"`` if the DF tensor is a CuPy array, otherwise ``"cpu"``.
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
        Extra keyword arguments forwarded to the per-root gradient driver.
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
    from asuka.mcscf.nac._df import sacasscf_nonadiabatic_couplings_df
    from asuka.mcscf.state_average import normalize_weights

    grad_kwargs_use = dict(grad_kwargs or {})
    nacv_kwargs = dict(nacv_kwargs or {})

    # ── resolve df_backend (device selector for DF path) ─────────────────────
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
    e_roots = _materialize_array(e_raw, dtype=np.float64).ravel()
    nroots = int(e_roots.size)

    weights_raw = getattr(casscf, "root_weights", None)
    if weights_raw is None:
        weights_raw = getattr(casscf, "weights", None)
    root_weights = normalize_weights(_maybe_asnumpy(weights_raw), nroots=nroots)
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
        # Inject df_backend / int1e_backend / df_threads into grad_kwargs for
        # the DF path; they are harmlessly stripped for direct/THC paths.
        _gk = dict(grad_kwargs_use)
        _gk.setdefault("df_backend", df_backend)
        _gk.setdefault("int1e_contract_backend", int1e_backend)
        _gk.setdefault("df_threads", df_threads)

        _grad_result = _dispatch_per_root_grad(
            grad_backend=str(grad_backend),
            scf_out=scf_out,
            casscf=casscf,
            grad_roots=grad_roots,
            z_tol=z_tol,
            z_maxiter=z_maxiter,
            grad_kwargs=_gk,
        )
        grads = _materialize_array(_grad_result.grads, dtype=np.float64)
        grad_sa = _materialize_array(_grad_result.grad_sa, dtype=np.float64)
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
        nacvs = _materialize_array(_nacv_raw, dtype=np.float64)

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

    # ── CI vectors and MO coefficients ──────────────────────────────────────
    ci_arr: np.ndarray | None = None
    ci_raw = getattr(casscf, "ci", None)
    if ci_raw is not None:
        _ci = _materialize_array(ci_raw, dtype=np.float64)
        ci_arr = _ci.reshape(nroots, -1) if _ci.ndim == 1 else _ci

    mo_coeff_arr: np.ndarray | None = None
    mo_raw = getattr(casscf, "mo_coeff", None)
    if mo_raw is not None:
        mo_coeff_arr = _materialize_array(mo_raw, dtype=np.float64)

    return SACASSCFPropertiesResult(
        e_roots=e_roots,
        e_sa=e_sa,
        e_nuc=e_nuc,
        root_weights=root_weights,
        grads=grads,
        grad_sa=grad_sa,
        nacvs=nacvs,
        nacv_pairs=computed_pairs,
        ci=ci_arr,
        mo_coeff=mo_coeff_arr,
        sigma=None,  # populated by make_df_sacasscf_properties_eval across steps
    )
