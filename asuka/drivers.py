"""High-level drivers for ASUKA (cuGUGA core) quantum chemistry methods.

This module provides simplified, high-level entry points for the various
quantum chemistry methods implemented in ASUKA. These drivers are designed to
work directly with PySCF CASCI/CASSCF objects.

Available Methods
-----------------
- FCI/CASCI: Full Configuration Interaction using CSF basis
- NEVPT2: Strongly-contracted N-electron valence perturbation theory
- Selected CI: Iterative selected CI with PT2 correction
- FCIQMC: Full CI Quantum Monte Carlo (stochastic projector)
- FCI-FRI: Full CI with Fast Randomized Iteration
- SOC: Spin-orbit coupling via state interaction

Example Usage
-------------
>>> from pyscf import gto, scf, mcscf
>>> from asuka import drivers
>>>
>>> mol = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='cc-pvdz')
>>> mf = scf.RHF(mol).run()
>>> mc = mcscf.CASSCF(mf, 6, 6).run()
>>>
>>> # Run NEVPT2
>>> result = drivers.nevpt2(mc)
>>> print(f"NEVPT2 correlation energy: {result.e_corr:.6f}")
"""

from __future__ import annotations

from collections import OrderedDict
import contextlib
from dataclasses import dataclass, replace
import threading
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np

if TYPE_CHECKING:
    from asuka.solver import GUGAFCISolver


@dataclass(frozen=True)
class DriverConfig:
    """Global defaults for ASUKA drivers."""

    # Default ERI backend for mc-based drivers ("auto" selects best available).
    eri_backend: str = "auto"
    # Default matvec backend for FCI/CASCI.
    matvec_backend: str = "contract"

    # cuERI integration
    prefer_cueri: bool = True
    auto_patch_cueri: bool = True

    # Integral caching
    cache_integrals: bool = True
    integral_cache_size: int = 8

    # cuERI dense CPU options
    cueri_eps_ao: float = 0.0
    cueri_eps_mo: float = 0.0
    # If None, let cuERI CPU auto-resolve to the basis max-l.
    cueri_max_l: int | None = None
    cueri_threads: int = 0
    cueri_max_tile_bytes: int = 256 * 1024 * 1024
    cueri_blas_nthreads: int | None = None


_DRIVER_CONFIG = DriverConfig()


def get_driver_config() -> DriverConfig:
    return _DRIVER_CONFIG


def set_driver_config(**kwargs: Any) -> DriverConfig:
    """Update the global driver config."""

    global _DRIVER_CONFIG
    _DRIVER_CONFIG = replace(_DRIVER_CONFIG, **kwargs)
    return _DRIVER_CONFIG


@contextlib.contextmanager
def driver_config(**kwargs: Any):
    """Temporarily override the global driver config."""

    global _DRIVER_CONFIG
    prev = _DRIVER_CONFIG
    _DRIVER_CONFIG = replace(_DRIVER_CONFIG, **kwargs)
    try:
        yield _DRIVER_CONFIG
    finally:
        _DRIVER_CONFIG = prev


@dataclass(frozen=True)
class _ActiveSpaceIntegrals:
    h1e: np.ndarray
    eri: Any
    ncas: int
    nelec: int
    ecore: float


class _IntegralCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: OrderedDict[tuple[Any, ...], _ActiveSpaceIntegrals] = OrderedDict()

    def get(self, key: tuple[Any, ...]) -> _ActiveSpaceIntegrals | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            self._data.move_to_end(key)
            return entry

    def put(self, key: tuple[Any, ...], value: _ActiveSpaceIntegrals, *, max_size: int) -> None:
        if max_size <= 0:
            return
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > max_size:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class _CuERIBuildCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: OrderedDict[tuple[Any, ...], Any] = OrderedDict()

    def get(self, key: tuple[Any, ...]) -> Any | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            self._data.move_to_end(key)
            return entry

    def put(self, key: tuple[Any, ...], value: Any, *, max_size: int) -> None:
        if max_size <= 0:
            return
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > max_size:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_INTEGRAL_CACHE = _IntegralCache()
_CUERI_BUILDER_CACHE = _CuERIBuildCache()


def clear_driver_caches() -> None:
    """Clear driver-level caches (integrals + cuERI builders)."""

    _INTEGRAL_CACHE.clear()
    _CUERI_BUILDER_CACHE.clear()


def _resolve_config(config: DriverConfig | None) -> DriverConfig:
    return _DRIVER_CONFIG if config is None else config


def _normalize_nelec(nelecas: int | tuple[int, int] | Sequence[int]) -> int:
    if isinstance(nelecas, (tuple, list)):
        return int(nelecas[0]) + int(nelecas[1])
    return int(nelecas)


def _infer_twos_from_mc(mc: Any, *, twos: int | None) -> int:
    if twos is not None:
        return int(twos)
    fcisolver = getattr(mc, "fcisolver", None)
    if fcisolver is not None and hasattr(fcisolver, "twos"):
        return int(getattr(fcisolver, "twos"))
    nelecas = getattr(mc, "nelecas", None)
    if isinstance(nelecas, (tuple, list)):
        return abs(int(nelecas[0]) - int(nelecas[1]))
    return 0


def _resolve_cueri_opts(
    cfg: DriverConfig,
    *,
    eps_ao: float | None = None,
    eps_mo: float | None = None,
    max_l: int | None = None,
    threads: int | None = None,
    max_tile_bytes: int | None = None,
    blas_nthreads: int | None = None,
) -> dict[str, Any]:
    if max_l is None:
        if cfg.cueri_max_l is None:
            max_l_i = None
        else:
            max_l_i = int(cfg.cueri_max_l)
    else:
        max_l_i = int(max_l)
    return {
        "eps_ao": float(cfg.cueri_eps_ao if eps_ao is None else eps_ao),
        "eps_mo": float(cfg.cueri_eps_mo if eps_mo is None else eps_mo),
        "max_l": (None if max_l_i is None else int(max_l_i)),
        "threads": int(cfg.cueri_threads if threads is None else threads),
        "max_tile_bytes": int(cfg.cueri_max_tile_bytes if max_tile_bytes is None else max_tile_bytes),
        "blas_nthreads": cfg.cueri_blas_nthreads if blas_nthreads is None else blas_nthreads,
    }


def _can_use_cueri_cpu(mc: Any) -> bool:
    mol = getattr(mc, "mol", None)
    if mol is None or not bool(getattr(mol, "cart", False)):
        return False
    try:
        from asuka.cueri.active_space_dense_cpu import CuERIActiveSpaceDenseCPUBuilder  # noqa: F401
    except Exception:
        return False
    return True


def _select_eri_backend(mc: Any, *, eri_backend: str | None, cfg: DriverConfig) -> str:
    backend = (eri_backend or cfg.eri_backend or "auto").strip().lower()
    if backend == "pyscf_ao2mo":
        raise ValueError(
            "eri_backend='pyscf_ao2mo' has been removed from core drivers; "
            "use eri_backend='mc_get_h2eff' or eri_backend='cueri_cpu'."
        )
    if backend == "auto":
        if cfg.prefer_cueri and _can_use_cueri_cpu(mc):
            return "cueri_cpu"
        if hasattr(mc, "get_h2eff"):
            return "mc_get_h2eff"
        return "mc_get_h2eff"
    return backend


def _make_integral_cache_key(
    mc: Any,
    *,
    mo_coeff: Any,
    ncas: int,
    nelec: int,
    eri_backend: str,
    cueri_opts: dict[str, Any],
) -> tuple[Any, ...]:
    return (
        id(mc),
        id(mo_coeff),
        int(ncas),
        int(nelec),
        str(eri_backend),
        float(cueri_opts["eps_ao"]),
        float(cueri_opts["eps_mo"]),
        None if cueri_opts["max_l"] is None else int(cueri_opts["max_l"]),
        int(cueri_opts["threads"]),
        int(cueri_opts["max_tile_bytes"]),
        None if cueri_opts["blas_nthreads"] is None else int(cueri_opts["blas_nthreads"]),
    )


def _get_cueri_builder(mc: Any, *, cueri_opts: dict[str, Any], cache_size: int) -> Any:
    mol = getattr(mc, "mol", None)
    if mol is None:
        raise ValueError("mc.mol is required for cuERI")
    key = (
        id(mol),
        None if cueri_opts["max_l"] is None else int(cueri_opts["max_l"]),
        int(cueri_opts["max_tile_bytes"]),
        int(cueri_opts["threads"]),
    )
    builder = _CUERI_BUILDER_CACHE.get(key)
    if builder is not None:
        return builder
    from asuka.cueri.active_space_dense_cpu import CuERIActiveSpaceDenseCPUBuilder
    from asuka.cueri.mol_basis import get_cached_or_pack_cart_ao_basis

    ao_basis = get_cached_or_pack_cart_ao_basis(cache_owner=mc, mol=mol, cache_attr="_asuka_ao_basis")
    if ao_basis is None:
        raise NotImplementedError(
            "cuERI CPU builder requires ao_basis. Provide a mol-like object with "
            "cartesian basis introspection APIs or set mc._asuka_ao_basis explicitly."
        )

    builder = CuERIActiveSpaceDenseCPUBuilder(
        ao_basis=ao_basis,
        max_l=(None if cueri_opts["max_l"] is None else int(cueri_opts["max_l"])),
        max_tile_bytes=int(cueri_opts["max_tile_bytes"]),
        threads=int(cueri_opts["threads"]),
    )
    _CUERI_BUILDER_CACHE.put(key, builder, max_size=cache_size)
    return builder


@contextlib.contextmanager
def _maybe_patch_mc_get_h2eff(mc: Any, *, cfg: DriverConfig, cueri_opts: dict[str, Any]):
    if not cfg.auto_patch_cueri or not _can_use_cueri_cpu(mc):
        yield
        return
    try:
        from asuka.cueri import patch_mc_get_h2eff_to_cueri_dense_cpu
    except Exception:
        yield
        return
    try:
        with patch_mc_get_h2eff_to_cueri_dense_cpu(
            mc,
            eps_ao=float(cueri_opts["eps_ao"]),
            eps_mo=float(cueri_opts["eps_mo"]),
            max_l=(None if cueri_opts["max_l"] is None else int(cueri_opts["max_l"])),
            max_tile_bytes=int(cueri_opts["max_tile_bytes"]),
            threads=int(cueri_opts["threads"]),
            blas_nthreads=cueri_opts["blas_nthreads"],
        ):
            yield
    except Exception:
        # Fall back to unpatched behavior.
        yield


def _get_active_space_integrals_from_mc(
    mc: Any,
    *,
    eri_backend: str | None,
    cfg: DriverConfig,
    cache_integrals: bool | None,
    cueri_opts: dict[str, Any],
) -> _ActiveSpaceIntegrals:
    ncas = int(getattr(mc, "ncas"))
    nelec = _normalize_nelec(getattr(mc, "nelecas"))
    h1e_eff, ecore = mc.h1e_for_cas()
    h1e = np.asarray(h1e_eff, dtype=np.float64, order="C")
    backend = _select_eri_backend(mc, eri_backend=eri_backend, cfg=cfg)

    use_cache = cfg.cache_integrals if cache_integrals is None else bool(cache_integrals)
    cache_key = None
    if use_cache:
        mo_coeff = getattr(mc, "mo_coeff", None)
        if mo_coeff is not None:
            cache_key = _make_integral_cache_key(
                mc,
                mo_coeff=mo_coeff,
                ncas=ncas,
                nelec=nelec,
                eri_backend=backend,
                cueri_opts=cueri_opts,
            )
            cached = _INTEGRAL_CACHE.get(cache_key)
            if cached is not None:
                return cached

    if backend == "mc_get_h2eff":
        get_h2eff = getattr(mc, "get_h2eff", None)
        if get_h2eff is None or not callable(get_h2eff):
            raise NotImplementedError(
                "eri_backend='mc_get_h2eff' requires a callable mc.get_h2eff(). "
                "Provide explicit (h1e, eri) to drivers.fci or use eri_backend='cueri_cpu' "
                "when mol.cart=True."
            )
        with _maybe_patch_mc_get_h2eff(mc, cfg=cfg, cueri_opts=cueri_opts):
            eri = mc.get_h2eff()
    elif backend == "cueri_cpu":
        if not _can_use_cueri_cpu(mc):
            raise NotImplementedError("eri_backend='cueri_cpu' requires mol.cart=True and cuERI")
        builder = _get_cueri_builder(mc, cueri_opts=cueri_opts, cache_size=cfg.integral_cache_size)
        mo_cas = np.asarray(mc.mo_coeff[:, mc.ncore : mc.ncore + ncas], dtype=np.float64, order="C")
        eri = builder.build_eri_packed(
            mo_cas,
            eps_ao=float(cueri_opts["eps_ao"]),
            eps_mo=float(cueri_opts["eps_mo"]),
            blas_nthreads=cueri_opts["blas_nthreads"],
        )
    else:
        raise ValueError(f"unsupported eri_backend={backend!r}")

    result = _ActiveSpaceIntegrals(
        h1e=h1e,
        eri=eri,
        ncas=ncas,
        nelec=nelec,
        ecore=float(ecore),
    )
    if use_cache and cache_key is not None:
        _INTEGRAL_CACHE.put(cache_key, result, max_size=cfg.integral_cache_size)
    return result


# =============================================================================
# FCI / CASCI Driver
# =============================================================================


@dataclass(frozen=True)
class FCIResult:
    """Result from FCI/CASCI calculation."""

    e_tot: float
    e_cas: float
    ci: np.ndarray
    nroots: int
    converged: bool


def fci(
    mc: Any | None = None,
    *,
    h1e: np.ndarray | None = None,
    eri: Any | None = None,
    norb: int | None = None,
    nelec: int | tuple[int, int] | None = None,
    ecore: float = 0.0,
    nroots: int = 1,
    twos: int | None = None,
    matvec_backend: str | None = None,
    eri_backend: Literal["auto", "mc_get_h2eff", "cueri_cpu"] | None = None,
    cueri_eps_ao: float | None = None,
    cueri_eps_mo: float | None = None,
    cueri_max_l: int | None = None,
    cueri_threads: int | None = None,
    cueri_max_tile_bytes: int | None = None,
    cueri_blas_nthreads: int | None = None,
    conv_tol: float = 1e-10,
    max_cycle: int = 100,
    max_space: int = 30,
    verbose: int = 0,
    config: DriverConfig | None = None,
    cache_integrals: bool | None = None,
) -> FCIResult:
    """Run FCI/CASCI calculation using GUGA CSF basis.

    Parameters
    ----------
    mc
        PySCF CASCI/CASSCF object with converged orbitals.
        If omitted, provide ``h1e``/``eri``/``norb``/``nelec`` instead.
    h1e, eri
        Optional MO-basis integrals (use when ``mc`` is None).
    norb, nelec
        Active-space orbital and electron counts (use when ``mc`` is None).
    ecore
        Frozen-core energy shift (use when ``mc`` is None).
    nroots
        Number of roots to compute.
    twos
        Target 2S value. If None, inferred from mc or defaults to 0 (singlet).
    matvec_backend
        Backend for matrix-vector product: "contract" (small systems),
        "row_oracle_df" (memory-scalable), or "cuda" (GPU).
        If None, uses driver config.
    eri_backend
        How to build the active-space 2e integrals:
        - "auto": choose best available (default)
        - "mc_get_h2eff": call `mc.get_h2eff()` (respects cuERI patch if enabled)
        - "cueri_cpu": use cuERI CPU dense active-space builder (requires `mol.cart=True`)
    conv_tol
        Convergence tolerance for Davidson iterations.
    max_cycle
        Maximum number of Davidson iterations.
    max_space
        Maximum Davidson subspace size.
    verbose
        Verbosity level.

    Returns
    -------
    FCIResult
        Result object containing energies and CI vector.

    Notes
    -----
    Use ``set_driver_config(...)`` or the ``config`` argument to control
    auto-backend selection and integral caching.
    """

    from asuka.solver import GUGAFCISolver

    cfg = _resolve_config(config)
    cueri_opts = _resolve_cueri_opts(
        cfg,
        eps_ao=cueri_eps_ao,
        eps_mo=cueri_eps_mo,
        max_l=cueri_max_l,
        threads=cueri_threads,
        max_tile_bytes=cueri_max_tile_bytes,
        blas_nthreads=cueri_blas_nthreads,
    )

    if mc is not None:
        integrals = _get_active_space_integrals_from_mc(
            mc,
            eri_backend=eri_backend,
            cfg=cfg,
            cache_integrals=cache_integrals,
            cueri_opts=cueri_opts,
        )
        h1e = integrals.h1e
        eri = integrals.eri
        ncas = integrals.ncas
        nelec = integrals.nelec
        ecore = integrals.ecore
        if twos is None:
            twos = _infer_twos_from_mc(mc, twos=twos)
    else:
        if h1e is None or eri is None:
            raise ValueError("fci requires either mc or (h1e, eri)")
        if norb is None:
            h1e = np.asarray(h1e, dtype=np.float64)
            if h1e.ndim != 2 or h1e.shape[0] != h1e.shape[1]:
                raise ValueError("h1e must be square (norb, norb)")
            ncas = int(h1e.shape[0])
        else:
            ncas = int(norb)
        if nelec is None:
            raise ValueError("nelec is required when mc is None")
        nelec = _normalize_nelec(nelec)
        if h1e is None:
            raise ValueError("h1e is required when mc is None")
        h1e = np.asarray(h1e, dtype=np.float64, order="C")
        ecore = float(ecore)
        if twos is None:
            twos = 0

    if matvec_backend is None or str(matvec_backend).strip().lower() == "auto":
        matvec_backend = cfg.matvec_backend

    # Run solver
    solver = GUGAFCISolver(twos=int(twos), nroots=nroots)
    solver.conv_tol = conv_tol
    solver.matvec_backend = matvec_backend
    solver.verbose = verbose

    e, ci = solver.kernel(h1e, eri, ncas, nelec, ecore=ecore, max_cycle=max_cycle, max_space=max_space)

    # The kernel returns total energy including ecore
    if nroots == 1:
        e_tot = float(e)
        e_cas = float(e) - ecore
    else:
        e_tot = float(e[0])
        e_cas = float(e[0]) - ecore

    return FCIResult(
        e_tot=e_tot,
        e_cas=e_cas,
        ci=ci,
        nroots=nroots,
        converged=True,
    )


# =============================================================================
# NEVPT2 Driver
# =============================================================================


def nevpt2(
    mc: Any,
    *,
    twos: int = 0,
    auxbasis: Any = "weigend+etb",
    semicanonicalize: bool = True,
    backend: str = "cpu",
    cuda_device: int | None = None,
    max_memory_mb: float = 4000.0,
    verbose: int = 0,
):
    """Run SC-NEVPT2 calculation with density fitting.

    Parameters
    ----------
    mc
        PySCF CASCI/CASSCF object with converged CAS reference.
    twos
        Target 2S value (default: 0 for singlet).
    auxbasis
        Auxiliary basis for density fitting.
    semicanonicalize
        Whether to semicanonicalize core/virtual orbitals.
    backend
        Computation backend: "cpu" or "cuda".
    cuda_device
        CUDA device ID (if backend="cuda").
    max_memory_mb
        Maximum memory in MB for intermediate arrays.
    verbose
        Verbosity level.

    Returns
    -------
    NEVPT2SCDFResult
        Result object containing correlation energy and breakdown by class.

    Example
    -------
    >>> from pyscf import gto, scf, mcscf
    >>> from asuka import drivers
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='cc-pvdz')
    >>> mf = scf.RHF(mol).run()
    >>> mc = mcscf.CASSCF(mf, 6, 6).run()
    >>> result = drivers.nevpt2(mc)
    >>> print(f"NEVPT2 E_corr = {result.e_corr:.6f}")
    """

    from asuka.mrpt2 import nevpt2_sc_df_from_mc

    return nevpt2_sc_df_from_mc(
        mc,
        auxbasis=auxbasis,
        twos=twos,
        semicanonicalize=semicanonicalize,
        pt2_backend=backend,
        cuda_device=cuda_device,
        max_memory_mb=max_memory_mb,
        verbose=verbose,
    )


def nevpt2_grad(
    mc: Any,
    *,
    twos: int = 0,
    auxbasis: Any = "weigend+etb",
    grad_backend: str = "fd",
    verbose: int = 0,
):
    """Compute SC-NEVPT2 nuclear gradients.

    Parameters
    ----------
    mc
        PySCF CASCI/CASSCF object.
    twos
        Target 2S value.
    auxbasis
        Auxiliary basis for density fitting.
    grad_backend
        Gradient backend: "fd" for finite differences.
    verbose
        Verbosity level.

    Returns
    -------
    NEVPT2SCDFGradResult
        Result object containing gradients.
    """

    from asuka.mrpt2 import nevpt2_sc_df_grad_from_mc

    return nevpt2_sc_df_grad_from_mc(
        mc,
        auxbasis=auxbasis,
        twos=twos,
        grad_backend=grad_backend,
        verbose=verbose,
    )


# =============================================================================
# MRCI Driver
# =============================================================================


def mrci(
    mc: Any,
    *,
    method: Literal["mrcisd", "ic_mrcisd"] = "mrcisd",
    state: int = 0,
    n_virt: int | None = None,
    twos: int | None = None,
    plus_q: bool = False,
    tol: float = 1e-10,
    max_cycle: int = 400,
    max_space: int = 30,
):
    """Run MRCI from a converged reference object."""

    from asuka.mrci import run_mrci  # noqa: PLC0415

    return run_mrci(
        mc,
        method=method,
        state=int(state),
        n_virt=n_virt,
        twos=twos,
        plus_q=bool(plus_q),
        tol=float(tol),
        max_cycle=int(max_cycle),
        max_space=int(max_space),
    )


def mrci_grad(
    mc: Any,
    *,
    method: Literal["mrcisd", "ic_mrcisd"] = "mrcisd",
    backend: Literal["fd", "analytic"] = "analytic",
    state: int = 0,
    n_virt: int | None = None,
    twos: int | None = None,
    **kwargs: Any,
):
    """Run MRCI gradient driver from a converged reference object."""

    from asuka.mrci import run_mrci_grad  # noqa: PLC0415

    return run_mrci_grad(
        mc,
        method=method,
        backend=backend,
        state=int(state),
        n_virt=n_virt,
        twos=twos,
        **kwargs,
    )


def dftmrci(
    mf: Any,
    *,
    nroots: int = 1,
    twos: int = 0,
    orb_list: Sequence[int] | None = None,
    n_act: int,
    n_virt: int | None = None,
    nelec: int | None = None,
    params: Any = "QE8",
    max_virt_e: int = 2,
    eri_backend: Literal["dense", "df"] = "df",
    auxbasis: Any = "weigend+etb",
    tmpdir: str | None = None,
    max_memory_mb: int = 2000,
    max_out: int = 200_000,
    backend: Literal["cpu_csr", "cuda_fixed_sell", "cuda_fixed_ell"] = "cpu_csr",
    slice_height: int = 32,
    symmetrize_csr: bool = False,
    strict: bool = True,
    tol: float = 1e-10,
    max_cycle: int = 200,
    max_space: int = 30,
    verbose: int = 0,
    **solver_kwargs: Any,
):
    """DFT/MRCI is not part of the core ASUKA distribution."""

    raise NotImplementedError(
        "DFT/MRCI has been removed from the core ASUKA distribution. "
        "The experimental implementation lives under `asuka.research.dftmrci` in the source tree "
        "and is intentionally excluded from the runtime wheel."
    )


def dftmrci_auto(
    mf: Any,
    *,
    nroots: int = 1,
    twos: int = 0,
    orb_list: Sequence[int] | None = None,
    n_act: int,
    n_virt: int | None = None,
    nelec: int | None = None,
    params: Any = "QE8",
    max_virt_e: int = 2,
    eri_backend: Literal["dense", "df"] = "df",
    auxbasis: Any = "weigend+etb",
    tmpdir: str | None = None,
    max_memory_mb: int = 2000,
    max_out: int = 200_000,
    max_ref_cycles: int = 10,
    foi_rank_max: int = 2,
    e_cut_init: float = 0.0,
    e_cut_margin: float = 0.0,
    w_ref_min: float = 1e-4,
    n_promote: int = 200,
    max_ref_cfg: int = 10_000,
    max_fois_cfg: int = 2_000_000,
    max_sel_cfg: int = 200_000,
    symmetrize_csr: bool = True,
    return_ci_full: bool = False,
    strict: bool = True,
    tol: float = 1e-10,
    max_cycle: int = 200,
    verbose: int = 0,
):
    """DFT/MRCI is not part of the core ASUKA distribution."""

    raise NotImplementedError(
        "DFT/MRCI has been removed from the core ASUKA distribution. "
        "The experimental implementation lives under `asuka.research.dftmrci` in the source tree "
        "and is intentionally excluded from the runtime wheel."
    )


# =============================================================================
# Selected CI Driver
# =============================================================================


def selected_ci(
    h1e: np.ndarray | None = None,
    eri: Any | None = None,
    norb: int | None = None,
    nelec: int | tuple[int, int] | None = None,
    *,
    mc: Any | None = None,
    twos: int | None = None,
    nroots: int = 1,
    max_iter: int = 20,
    nsel_init: int | None = None,
    nsel_max: int | None = None,
    nsel_add: int = 100,
    pt2_selection: bool = True,
    tol_var: float = 1e-6,
    verbose: int = 0,
    eri_backend: Literal["auto", "mc_get_h2eff", "cueri_cpu"] | None = None,
    config: DriverConfig | None = None,
    cache_integrals: bool | None = None,
):
    """Run selected CI calculation.

    Parameters
    ----------
    h1e
        One-electron integrals (norb, norb).
    eri
        Two-electron integrals: dense (norb, norb, norb, norb) or DFMOIntegrals.
    norb
        Number of orbitals.
    nelec
        Number of electrons.
    mc
        Optional PySCF CASCI/CASSCF object. If provided, integrals are built under the hood.
    mc
        Optional PySCF CASCI/CASSCF object. If provided, integrals are built under the hood.
    mc
        Optional PySCF CASCI/CASSCF object. If provided, integrals are built under the hood.
    eri_backend
        ERI backend selection when ``mc`` is provided (default: auto).
    config
        Optional driver config override.
    twos
        Target 2S value.
    nroots
        Number of roots to compute.
    max_iter
        Maximum selection iterations.
    nsel_init
        Initial number of CSFs (default: nroots * 10).
    nsel_max
        Maximum CSF space size (default: unlimited).
    nsel_add
        Number of CSFs to add per iteration.
    pt2_selection
        Use EN-PT2 selection criterion (vs. first-order amplitude).
    tol_var
        Variational energy convergence tolerance.
    verbose
        Verbosity level.

    Returns
    -------
    SCIResult
        Result object with variational and PT2 energies.

    Example
    -------
    >>> from asuka import drivers
    >>> # Assuming h1e, eri are prepared
    >>> result = drivers.selected_ci(h1e, eri, norb=10, nelec=10)
    >>> print(f"SCI E_var = {result.e_var[0]:.6f}")
    >>> print(f"SCI E_tot = {result.e_tot[0]:.6f}")
    """

    from asuka.sci import selected_ci as sci_kernel

    cfg = _resolve_config(config)
    cueri_opts = _resolve_cueri_opts(cfg)

    if mc is not None:
        integrals = _get_active_space_integrals_from_mc(
            mc,
            eri_backend=eri_backend,
            cfg=cfg,
            cache_integrals=cache_integrals,
            cueri_opts=cueri_opts,
        )
        h1e = integrals.h1e
        eri = integrals.eri
        norb = integrals.ncas
        nelec = integrals.nelec
        if twos is None:
            twos = _infer_twos_from_mc(mc, twos=twos)
    else:
        if h1e is None or eri is None:
            raise ValueError("selected_ci requires either mc or (h1e, eri)")
        if norb is None:
            h1e = np.asarray(h1e, dtype=np.float64)
            if h1e.ndim != 2 or h1e.shape[0] != h1e.shape[1]:
                raise ValueError("h1e must be square (norb, norb)")
            norb = int(h1e.shape[0])
        if nelec is None:
            raise ValueError("nelec is required when mc is None")
        nelec = _normalize_nelec(nelec)
        if twos is None:
            twos = 0

    return sci_kernel(
        h1e=h1e,
        eri=eri,
        norb=norb,
        nelec=nelec,
        twos=twos,
        nroots=nroots,
        max_iter=max_iter,
        nsel_init=nsel_init,
        nsel_max=nsel_max,
        nsel_add=nsel_add,
        pt2_selection=pt2_selection,
        tol_var=tol_var,
        verbose=verbose,
    )


def selected_ci_from_mc(
    mc: Any,
    *,
    twos: int | None = None,
    nroots: int = 1,
    max_iter: int = 20,
    nsel_add: int = 100,
    verbose: int = 0,
    eri_backend: Literal["auto", "mc_get_h2eff", "cueri_cpu"] | None = None,
    config: DriverConfig | None = None,
    cache_integrals: bool | None = None,
):
    """Run selected CI from a PySCF CASCI/CASSCF object.

    Parameters
    ----------
    mc
        PySCF CASCI/CASSCF object.
    twos
        Target 2S value. If None, inferred from mc.
    nroots
        Number of roots to compute.
    max_iter
        Maximum selection iterations.
    nsel_add
        Number of CSFs to add per iteration.
    verbose
        Verbosity level.

    Returns
    -------
    SCIResult
        Result object with selected CI results.

    Notes
    -----
    This wrapper uses the same auto-backend selection as :func:`selected_ci`.
    """

    return selected_ci(
        mc=mc,
        twos=twos,
        nroots=nroots,
        max_iter=max_iter,
        nsel_add=nsel_add,
        verbose=verbose,
        eri_backend=eri_backend,
        config=config,
        cache_integrals=cache_integrals,
    )


# =============================================================================
# QMC Drivers
# =============================================================================


def fciqmc(
    h1e: np.ndarray | None = None,
    eri: Any | None = None,
    norb: int | None = None,
    nelec: int | tuple[int, int] | None = None,
    *,
    mc: Any | None = None,
    twos: int | None = None,
    dt: float = 0.001,
    niter: int = 10000,
    nspawn_one: int = 100,
    nspawn_two: int = 100,
    seed: int = 12345,
    target_population: float = 10000.0,
    shift_damping: float = 0.0,
    shift_stride: int = 1,
    shift_start: int = 0,
    verbose: int = 0,
    eri_backend: Literal["auto", "mc_get_h2eff", "cueri_cpu"] | None = None,
    config: DriverConfig | None = None,
    cache_integrals: bool | None = None,
):
    """Run FCIQMC calculation.

    Parameters
    ----------
    h1e
        One-electron integrals (norb, norb).
    eri
        Two-electron integrals.
    norb
        Number of orbitals.
    nelec
        Number of electrons.
    twos
        Target 2S value.
    dt
        Time step.
    niter
        Number of iterations.
    nspawn_one
        Number of one-body spawn attempts per walker.
    nspawn_two
        Number of two-body spawn attempts per walker.
    seed
        Random seed.
    target_population
        Target walker population.
    shift_damping
        Damping factor for shift update. Use 0.0 to disable shift control.
    shift_stride
        Update shift every `shift_stride` iterations.
    shift_start
        Start shift updates at iteration `shift_start`.
    verbose
        Verbosity level.

    Returns
    -------
    FCIQMCRun
        Result object with energy trajectory and final wavefunction.

    Example
    -------
    >>> from asuka import drivers
    >>> result = drivers.fciqmc(h1e, eri, norb=6, nelec=6, niter=5000)
    >>> print(f"Final energy: {result.energies[-1]:.6f}")
    """

    from asuka.cuguga import build_drt
    from asuka.qmc import run_fciqmc

    if mc is not None:
        cfg = _resolve_config(config)
        cueri_opts = _resolve_cueri_opts(cfg)
        integrals = _get_active_space_integrals_from_mc(
            mc,
            eri_backend=eri_backend,
            cfg=cfg,
            cache_integrals=cache_integrals,
            cueri_opts=cueri_opts,
        )
        h1e = integrals.h1e
        eri = integrals.eri
        norb = integrals.ncas
        nelec = integrals.nelec
        if twos is None:
            twos = _infer_twos_from_mc(mc, twos=twos)
    else:
        if h1e is None or eri is None:
            raise ValueError("fciqmc requires either mc or (h1e, eri)")
        if norb is None:
            h1e = np.asarray(h1e, dtype=np.float64)
            if h1e.ndim != 2 or h1e.shape[0] != h1e.shape[1]:
                raise ValueError("h1e must be square (norb, norb)")
            norb = int(h1e.shape[0])
        if nelec is None:
            raise ValueError("nelec is required when mc is None")
        nelec = _normalize_nelec(nelec)
        if twos is None:
            twos = 0

    drt = build_drt(norb, nelec, twos_target=twos)

    # Start from HF-like initial guess (first CSF)
    x_idx = np.array([0], dtype=np.int32)
    x_val = np.array([1.0], dtype=np.float64)

    return run_fciqmc(
        drt,
        h1e,
        eri,
        x_idx,
        x_val,
        dt=dt,
        niter=niter,
        nspawn_one=nspawn_one,
        nspawn_two=nspawn_two,
        seed=seed,
        target_population=target_population,
        shift_damping=shift_damping,
        shift_stride=shift_stride,
        shift_start=shift_start,
    )


def fcifri(
    h1e: np.ndarray | None = None,
    eri: Any | None = None,
    norb: int | None = None,
    nelec: int | tuple[int, int] | None = None,
    *,
    mc: Any | None = None,
    twos: int | None = None,
    m: int = 1000,
    eps: float = 0.05,
    niter: int = 500,
    nspawn_one: int = 256,
    nspawn_two: int = 256,
    seed: int = 12345,
    nroots: int = 1,
    backend: str = "stochastic",
    energy_estimator: Literal["projected", "rayleigh"] = "projected",
    verbose: int = 0,
    eri_backend: Literal["auto", "mc_get_h2eff", "cueri_cpu"] | None = None,
    config: DriverConfig | None = None,
    cache_integrals: bool | None = None,
):
    """Run FCI-FRI (Fast Randomized Iteration) calculation.

    Parameters
    ----------
    h1e
        One-electron integrals (norb, norb).
    eri
        Two-electron integrals.
    norb
        Number of orbitals.
    nelec
        Number of electrons.
    twos
        Target 2S value.
    m
        Target compressed vector size.
    eps
        Time step parameter for projector.
    niter
        Number of iterations.
    nspawn_one
        Number of one-body spawn samples.
    nspawn_two
        Number of two-body spawn samples.
    seed
        Random seed.
    nroots
        Number of roots (for subspace method).
    backend
        Projector backend: "stochastic" (CPU) or "cuda" (GPU). For multi-root
        subspace iteration, "exact" is also available for validation on tiny spaces.
    energy_estimator
        Ground-state energy estimator for ``nroots=1``:
        ``"projected"`` or ``"rayleigh"``.
    verbose
        Verbosity level.

    Returns
    -------
    FCIFRIRun or FCIFRISubspaceRun
        Result object with energy trajectory and compressed wavefunction.

    Example
    -------
    >>> from asuka import drivers
    >>> result = drivers.fcifri(h1e, eri, norb=6, nelec=6, m=500, niter=50)
    >>> print(f"Final energy: {result.energies[-1]:.6f}")
    """

    from asuka.cuguga import build_drt
    from asuka.qmc import run_fcifri_ground, run_fcifri_subspace

    if mc is not None:
        cfg = _resolve_config(config)
        cueri_opts = _resolve_cueri_opts(cfg)
        integrals = _get_active_space_integrals_from_mc(
            mc,
            eri_backend=eri_backend,
            cfg=cfg,
            cache_integrals=cache_integrals,
            cueri_opts=cueri_opts,
        )
        h1e = integrals.h1e
        eri = integrals.eri
        norb = integrals.ncas
        nelec = integrals.nelec
        if twos is None:
            twos = _infer_twos_from_mc(mc, twos=twos)
    else:
        if h1e is None or eri is None:
            raise ValueError("fcifri requires either mc or (h1e, eri)")
        if norb is None:
            h1e = np.asarray(h1e, dtype=np.float64)
            if h1e.ndim != 2 or h1e.shape[0] != h1e.shape[1]:
                raise ValueError("h1e must be square (norb, norb)")
            norb = int(h1e.shape[0])
        if nelec is None:
            raise ValueError("nelec is required when mc is None")
        nelec = _normalize_nelec(nelec)
        if twos is None:
            twos = 0

    energy_estimator = str(energy_estimator).lower()
    if energy_estimator not in ("projected", "rayleigh"):
        raise ValueError("energy_estimator must be 'projected' or 'rayleigh'")

    drt = build_drt(norb, nelec, twos_target=twos)

    # Start from HF-like initial guess
    x_idx = np.array([0], dtype=np.int32)
    x_val = np.array([1.0], dtype=np.float64)

    if nroots == 1:
        return run_fcifri_ground(
            drt,
            h1e,
            eri,
            x_idx,
            x_val,
            m=m,
            eps=eps,
            niter=niter,
            nspawn_one=nspawn_one,
            nspawn_two=nspawn_two,
            seed=seed,
            backend=backend,
            energy_estimator=energy_estimator,
        )
    else:
        # Multi-root subspace method
        x0 = [(x_idx.copy(), x_val.copy()) for _ in range(nroots)]
        return run_fcifri_subspace(
            drt,
            h1e,
            eri,
            nroots=nroots,
            x0=x0,
            m=m,
            eps=eps,
            niter=niter,
            nspawn_one=nspawn_one,
            nspawn_two=nspawn_two,
            seed=seed,
            backend=backend,
        )


# =============================================================================
# SOC Driver
# =============================================================================


@dataclass(frozen=True)
class SOCCIZVectorCIResponseResult:
    """CI-response intermediates for a SOC-SI root via an MCSCF Z-vector solve (CI-first)."""

    si_energies: np.ndarray
    si_vectors: np.ndarray
    si_basis: list[tuple[int, int]]

    w_state: np.ndarray
    eta: np.ndarray

    response: Any


def soc_state_interaction(
    states: Sequence[Any],
    soc_integrals: Any,
    *,
    verbose: int = 0,
):
    """Compute spin-orbit coupling via state interaction.

    Parameters
    ----------
    states
        List of SpinFreeState objects (or tuples of (twos, energy, drt, ci)).
    soc_integrals
        SOCIntegrals object containing h_xyz or h_m integrals.
    verbose
        Verbosity level.

    Returns
    -------
    tuple
        (eigenvalues, eigenvectors, SI_basis, SI_hamiltonian)

    Example
    -------
    >>> from asuka import drivers
    >>> from asuka.soc import SpinFreeState, SOCIntegrals
    >>> # Prepare states and SOC integrals
    >>> result = drivers.soc_state_interaction(states, soc_integrals)
    >>> energies, vectors, basis, H_SI = result
    """

    from asuka.soc import soc_state_interaction as _soc_si

    return _soc_si(states, soc_integrals)


def soc_ci_zvector_response(
    mc: Any,
    soc_integrals: Any,
    *,
    so_root: int = 0,
    states: Sequence[Any] | None = None,
    h_m_ao: np.ndarray | None = None,
    h_xyz_ao: np.ndarray | None = None,
    block_nops: int = 8,
    symmetrize: bool = True,
    imag_tol: float = 1e-10,
    project_normalized: bool = True,
    z_tol: float = 1e-10,
    z_maxiter: int = 200,
    verbose: int = 0,
) -> SOCCIZVectorCIResponseResult:
    """Build the SOC adjoint RHS for a selected SO root and solve the MCSCF Z-vector.

    Notes
    -----
    This is a developer-facing helper for connecting SOC-SI adjoints to the existing MCSCF
    Z-vector machinery.

    - If `h_m_ao`/`h_xyz_ao` is provided, this helper also builds the SOC-driven *orbital*
      RHS from the effective SOC density `rho_m` (AO-level SOC integrals are required for
      a nonzero orbital RHS).
    - AO-level SOC integral *derivatives* (`dh/dR`) are not included here; this helper only
      constructs the Z-vector and CI-response effective RDMs.
    """

    _ = int(verbose)  # reserved
    from asuka.soc import (
        SOCIntegrals,
        compute_si_adjoint_weights,
        soc_state_interaction as _soc_si,
        soc_xyz_to_spherical,
    )
    from asuka.soc.grad import solve_soc_ci_zvector_response, spinfree_states_from_mc

    if states is None:
        states_sf = spinfree_states_from_mc(mc)
    else:
        # Assume user provided SpinFreeState objects.
        states_sf = list(states)  # type: ignore[arg-type]

    if not isinstance(soc_integrals, SOCIntegrals):
        raise TypeError("soc_integrals must be a asuka.soc.SOCIntegrals")
    h_m = soc_integrals.h_m
    if h_m is None:
        if soc_integrals.h_xyz is None:
            raise ValueError("soc_integrals must provide h_m or h_xyz")
        h_m = soc_xyz_to_spherical(soc_integrals.h_xyz)

    if h_m_ao is not None and h_xyz_ao is not None:
        raise ValueError("provide at most one of h_m_ao or h_xyz_ao")
    if h_xyz_ao is not None:
        h_m_ao = soc_xyz_to_spherical(h_xyz_ao)

    e_si, c_si, basis = _soc_si(states_sf, soc_integrals, block_nops=int(block_nops), symmetrize=bool(symmetrize))
    so_root = int(so_root)
    if so_root < 0 or so_root >= int(e_si.size):
        raise ValueError("so_root out of range for SI eigenvectors")

    w_state, eta = compute_si_adjoint_weights(states_sf, basis, c_si[:, so_root])

    response = solve_soc_ci_zvector_response(
        mc,
        states=states_sf,
        eta=eta,
        h_m=h_m,
        h_m_ao=h_m_ao,
        block_nops=int(block_nops),
        imag_tol=float(imag_tol),
        project_normalized=bool(project_normalized),
        z_tol=float(z_tol),
        z_maxiter=int(z_maxiter),
    )

    return SOCCIZVectorCIResponseResult(
        si_energies=e_si,
        si_vectors=c_si,
        si_basis=basis,
        w_state=w_state,
        eta=eta,
        response=response,
    )


def soc_ci_zvector_response_multi_spin(
    mcs: Sequence[Any],
    soc_integrals: Any,
    *,
    so_root: int = 0,
    h_m_ao: np.ndarray | None = None,
    h_xyz_ao: np.ndarray | None = None,
    block_nops: int = 8,
    symmetrize: bool = True,
    eps: float = 0.0,
    imag_tol: float = 1e-10,
    project_normalized: bool = True,
    z_tol: float = 1e-10,
    z_maxiter: int = 200,
    verbose: int = 0,
):
    """Multi-spin variant of `soc_ci_zvector_response` (cross-spin SOC-SI support).

    Notes
    -----
    - This helper assumes all manifolds share the same active orbital basis used by `soc_integrals`.
    - If `h_m_ao`/`h_xyz_ao` is provided, each manifold's Z-vector includes an orbital RHS term
      driven by the global SOC effective density `rho_m`.
    """

    _ = int(verbose)  # reserved
    from asuka.soc import SOCIntegrals, soc_xyz_to_spherical
    from asuka.soc.grad import solve_soc_ci_zvector_response_multi_spin

    if not isinstance(soc_integrals, SOCIntegrals):
        raise TypeError("soc_integrals must be a asuka.soc.SOCIntegrals")

    if h_m_ao is not None and h_xyz_ao is not None:
        raise ValueError("provide at most one of h_m_ao or h_xyz_ao")
    if h_xyz_ao is not None:
        h_m_ao = soc_xyz_to_spherical(h_xyz_ao)

    return solve_soc_ci_zvector_response_multi_spin(
        mcs,
        soc_integrals,
        so_root=int(so_root),
        h_m_ao=h_m_ao,
        block_nops=int(block_nops),
        symmetrize=bool(symmetrize),
        eps=float(eps),
        imag_tol=float(imag_tol),
        project_normalized=bool(project_normalized),
        z_tol=float(z_tol),
        z_maxiter=int(z_maxiter),
    )


# =============================================================================
# Vibrational analysis + geometry optimization
# =============================================================================


def vib_fd_hessian(
    grad_fn: Any,
    coords0_bohr: np.ndarray,
    *,
    step_bohr: float = 1e-3,
    method: str = "central",
    symmetrize: bool = True,
    verbose: int = 0,
):
    """Finite-difference Cartesian Hessian (Eh/Bohr^2) from a gradient callable."""

    from asuka.vib.hessian_fd import fd_cartesian_hessian  # noqa: PLC0415

    return fd_cartesian_hessian(
        grad_fn,
        np.asarray(coords0_bohr, dtype=np.float64),
        step_bohr=float(step_bohr),
        method=str(method),
        symmetrize=bool(symmetrize),
        verbose=int(verbose),
    )


def vib_frequency_analysis(
    coords0_bohr: np.ndarray,
    masses_amu: Sequence[float],
    hess_cart: np.ndarray,
    *,
    linear: bool | None = None,
    tr_tol: float = 1e-10,
    symmetrize: bool = True,
    seed: int = 0,
):
    """Harmonic frequency analysis from a Cartesian Hessian."""

    from asuka.vib.frequency import frequency_analysis  # noqa: PLC0415

    return frequency_analysis(
        hessian_cart=np.asarray(hess_cart, dtype=np.float64),
        coords_bohr=np.asarray(coords0_bohr, dtype=np.float64),
        masses_amu=list(masses_amu),
        mol=None,
        linear=linear,
        tr_tol=float(tr_tol),
        symmetrize=bool(symmetrize),
        seed=int(seed),
    )


def geomopt_cartesian(
    energy_grad: Any,
    coords0_bohr: np.ndarray,
    *,
    settings: Any | None = None,
):
    """Optimize geometry in Cartesian coordinates using an (E,grad) callback."""

    from asuka.geomopt.optimizer import GeomOptSettings, optimize_cartesian  # noqa: PLC0415

    st = GeomOptSettings() if settings is None else settings
    return optimize_cartesian(energy_grad, np.asarray(coords0_bohr, dtype=np.float64), settings=st)


# =============================================================================
# Convenience exports
# =============================================================================

__all__ = [
    "FCIResult",
    "fci",
    "nevpt2",
    "nevpt2_grad",
    "mrci",
    "mrci_grad",
    "selected_ci",
    "selected_ci_from_mc",
    "fciqmc",
    "fcifri",
    "soc_state_interaction",
    "SOCCIZVectorCIResponseResult",
    "soc_ci_zvector_response",
    "soc_ci_zvector_response_multi_spin",
    "vib_fd_hessian",
    "vib_frequency_analysis",
    "geomopt_cartesian",
]
