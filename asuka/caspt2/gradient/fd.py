"""Finite-difference CASPT2 nuclear gradients.

The multistate analytic CASPT2 gradient machinery is not yet integrated in the
top-level driver. This module provides a correct fallback by finite
differences of the existing SS/MS/XMS CASPT2 energy driver.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import os
import numpy as np

from asuka.caspt2.driver_asuka import _caspt2_spinfree_states_for_soc, caspt2_from_casscf
from asuka.frontend.analysis import _clone_molecule_with_coords
from asuka.frontend.scf import RHFDFRunResult, ROHFDFRunResult, UHFDFRunResult, run_hf_df
from asuka.mcscf import run_casscf
from asuka.mcscf.state_average import ci_as_list, fix_ci_phases, match_roots_by_overlap


BuildFn = Callable[[np.ndarray], tuple[Any, Any]]
EnergyFn = Callable[[Any, Any, int], Any]

_DEFAULT_MSXMS_FD_STEP_BOHR = 1.0e-4


def _asnumpy_f64(a: Any) -> np.ndarray:
    try:
        import cupy as cp  # noqa: PLC0415

        if isinstance(a, cp.ndarray):  # type: ignore[attr-defined]
            return np.asarray(cp.asnumpy(a), dtype=np.float64)
    except Exception:
        pass
    return np.asarray(a, dtype=np.float64)


def _fd_step_bohr_from_env(default: float = _DEFAULT_MSXMS_FD_STEP_BOHR) -> float:
    try:
        step = float(os.environ.get("ASUKA_CASPT2_MSXMS_GRAD_FD_STEP_BOHR", str(default)))
    except Exception:
        step = float(default)
    step = abs(float(step))
    if step <= 0.0:
        raise ValueError("FD step must be positive")
    return step


def _bool_env(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _normalize_which(
    which: Iterable[tuple[int, int]] | None,
    *,
    natm: int,
) -> list[tuple[int, int]]:
    if which is None:
        return [(ia, xyz) for ia in range(int(natm)) for xyz in range(3)]
    out: list[tuple[int, int]] = []
    for item in which:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(f"invalid FD coordinate selector entry: {item!r}")
        ia = int(item[0])
        xyz = int(item[1])
        if ia < 0 or ia >= int(natm):
            raise ValueError(f"atom index out of range in FD selector: {ia}")
        if xyz not in (0, 1, 2):
            raise ValueError(f"axis index out of range in FD selector: {xyz}")
        out.append((ia, xyz))
    return out


def _result_energy_array(res: Any, *, method: str) -> np.ndarray:
    method_u = str(method).upper().strip()
    e_tot = getattr(res, "e_tot")
    if method_u == "SS":
        return np.asarray([float(np.asarray(e_tot, dtype=np.float64).ravel()[0])], dtype=np.float64)
    return np.asarray(e_tot, dtype=np.float64).ravel()


def _result_root_ci_vectors(ref: Any, scf_out: Any, res: Any, *, method: str) -> list[np.ndarray]:
    method_u = str(method).upper().strip()
    if method_u == "SS":
        return [np.asarray(ci_as_list(getattr(ref, "ci"), nroots=1)[0], dtype=np.float64).ravel()]
    states = _caspt2_spinfree_states_for_soc(ref, scf_out, res)
    return [np.asarray(st.ci, dtype=np.float64).ravel() for st in states]


def _ordered_target_energy(
    *,
    ref_obj: Any,
    scf_obj: Any,
    res: Any,
    method: str,
    base_ci: list[np.ndarray] | None,
    iroot: int,
) -> tuple[float, np.ndarray]:
    e = _result_energy_array(res, method=method)
    if base_ci is None or int(e.size) <= 1:
        perm = np.arange(int(e.size), dtype=np.int32)
        return float(e[int(iroot)]), perm

    cur_ci = _result_root_ci_vectors(ref_obj, scf_obj, res, method=method)
    perm = np.asarray(match_roots_by_overlap(base_ci, cur_ci), dtype=np.int32).ravel()
    cur_ci_perm = [np.asarray(cur_ci[int(p)], dtype=np.float64).copy() for p in perm.tolist()]
    fix_ci_phases(base_ci, cur_ci_perm)
    e_ord = np.asarray(e[perm], dtype=np.float64)
    return float(e_ord[int(iroot)]), perm


def _fd_gradient_core(
    build_fn: BuildFn,
    energy_fn: EnergyFn,
    coords_bohr: np.ndarray,
    *,
    method: str,
    nstates: int,
    iroot: int,
    caspt2_kwargs: dict[str, Any] | None,
    delta: float,
    which: Iterable[tuple[int, int]] | None,
    verbose: int = 0,
    base_pair: tuple[Any, Any] | None = None,
) -> tuple[Any, Any, Any, float, np.ndarray, list[dict[str, Any]]]:
    coords0 = np.asarray(coords_bohr, dtype=np.float64).reshape((-1, 3))
    natm = int(coords0.shape[0])
    delta_b = abs(float(delta))
    if delta_b <= 0.0:
        raise ValueError("delta must be positive")

    coords_sel = _normalize_which(which, natm=natm)
    kwargs = dict(caspt2_kwargs or {})
    method_u = str(method).upper().strip()
    if method_u not in {"SS", "MS", "XMS"}:
        raise ValueError(f"unsupported method for FD gradient: {method!r}")

    if base_pair is None:
        scf0, ref0 = build_fn(coords0)
    else:
        scf0, ref0 = base_pair
    res0 = energy_fn(scf0, ref0, int(iroot))
    base_ci = None
    # For SS-CASPT2 with nroots>1 CASSCF, store all CASSCF CI vectors for
    # root tracking at displaced geometries.  States can reorder when symmetry
    # is broken by the finite displacement (e.g. NH3 C3v degenerate states),
    # and without tracking we pick the wrong root, causing catastrophic errors.
    ss_base_casscf_ci: list[np.ndarray] | None = None
    if method_u != "SS" and int(nstates) > 1:
        base_ci = _result_root_ci_vectors(ref0, scf0, res0, method=method_u)
    elif method_u == "SS" and int(nstates) > 1:
        ss_base_casscf_ci = [
            np.asarray(ci, dtype=np.float64).ravel()
            for ci in ci_as_list(getattr(ref0, "ci"), nroots=int(nstates))
        ]
    e0, perm0 = _ordered_target_energy(
        ref_obj=ref0,
        scf_obj=scf0,
        res=res0,
        method=method_u,
        base_ci=base_ci,
        iroot=int(iroot),
    )
    grad = np.zeros((natm, 3), dtype=np.float64)
    points: list[dict[str, Any]] = []

    for ia, xyz in coords_sel:
        if verbose:
            print(f"[CASPT2 grad FD] ({ia},{xyz})")
        coords_p = np.asarray(coords0, dtype=np.float64).copy()
        coords_m = np.asarray(coords0, dtype=np.float64).copy()
        coords_p[ia, xyz] += delta_b
        coords_m[ia, xyz] -= delta_b

        scf_p, ref_p = build_fn(coords_p)
        if ss_base_casscf_ci is not None:
            cur_ci_p = [
                np.asarray(ci, dtype=np.float64).ravel()
                for ci in ci_as_list(getattr(ref_p, "ci"), nroots=int(nstates))
            ]
            perm_p_arr = np.asarray(match_roots_by_overlap(ss_base_casscf_ci, cur_ci_p), dtype=np.int32)
            iroot_p = int(perm_p_arr[int(iroot)])
        else:
            perm_p_arr = np.arange(max(1, int(nstates)), dtype=np.int32)
            iroot_p = int(iroot)
        res_p = energy_fn(scf_p, ref_p, int(iroot_p))
        e_p, perm_p = _ordered_target_energy(
            ref_obj=ref_p,
            scf_obj=scf_p,
            res=res_p,
            method=method_u,
            base_ci=base_ci,
            iroot=0 if ss_base_casscf_ci is not None else int(iroot),
        )

        scf_m, ref_m = build_fn(coords_m)
        if ss_base_casscf_ci is not None:
            cur_ci_m = [
                np.asarray(ci, dtype=np.float64).ravel()
                for ci in ci_as_list(getattr(ref_m, "ci"), nroots=int(nstates))
            ]
            perm_m_arr = np.asarray(match_roots_by_overlap(ss_base_casscf_ci, cur_ci_m), dtype=np.int32)
            iroot_m = int(perm_m_arr[int(iroot)])
        else:
            perm_m_arr = np.arange(max(1, int(nstates)), dtype=np.int32)
            iroot_m = int(iroot)
        res_m = energy_fn(scf_m, ref_m, int(iroot_m))
        e_m, perm_m = _ordered_target_energy(
            ref_obj=ref_m,
            scf_obj=scf_m,
            res=res_m,
            method=method_u,
            base_ci=base_ci,
            iroot=0 if ss_base_casscf_ci is not None else int(iroot),
        )

        grad[ia, xyz] = float((e_p - e_m) / (2.0 * delta_b))
        points.append(
            {
                "atom": int(ia),
                "axis": int(xyz),
                "delta_bohr": float(delta_b),
                "e_plus": float(e_p),
                "e_minus": float(e_m),
                "perm0": np.asarray(perm0, dtype=np.int32).tolist(),
                "perm_plus": np.asarray(perm_p, dtype=np.int32).tolist(),
                "perm_minus": np.asarray(perm_m, dtype=np.int32).tolist(),
                "iroot_plus": int(iroot_p),
                "iroot_minus": int(iroot_m),
            }
        )

    return scf0, ref0, res0, float(e0), np.asarray(grad, dtype=np.float64), points


def fd_nuclear_gradient_asuka(
    build_fn: BuildFn,
    coords_bohr: np.ndarray,
    *,
    method: str,
    nstates: int,
    iroot: int = 0,
    caspt2_kwargs: dict[str, Any] | None = None,
    delta: float = _DEFAULT_MSXMS_FD_STEP_BOHR,
    which: Iterable[tuple[int, int]] | None = None,
    verbose: int = 0,
    base_pair: tuple[Any, Any] | None = None,
) -> tuple[float, np.ndarray, list[dict[str, Any]]]:
    """Finite-difference CASPT2 nuclear gradient from an ASUKA build callback."""
    kwargs = dict(caspt2_kwargs or {})

    def _energy(scf_obj: Any, ref_obj: Any, iroot_sel: int):
        return caspt2_from_casscf(
            scf_obj,
            ref_obj,
            method=str(method).upper().strip(),
            nstates=int(nstates),
            iroot=int(iroot_sel),
            **kwargs,
        )

    _scf0, _ref0, _res0, e0, grad, points = _fd_gradient_core(
        build_fn,
        _energy,
        coords_bohr,
        method=method,
        nstates=int(nstates),
        iroot=int(iroot),
        caspt2_kwargs=caspt2_kwargs,
        delta=float(delta),
        which=which,
        verbose=int(verbose),
        base_pair=base_pair,
    )
    return float(e0), np.asarray(grad, dtype=np.float64), points


def _infer_hf_method(scf_out: Any) -> str:
    if isinstance(scf_out, RHFDFRunResult):
        return "rhf"
    if isinstance(scf_out, ROHFDFRunResult):
        return "rohf"
    if isinstance(scf_out, UHFDFRunResult):
        return "uhf"
    name = type(scf_out).__name__.lower()
    if "rohf" in name:
        return "rohf"
    if "uhf" in name:
        return "uhf"
    return "rhf"


def _infer_backend(scf_out: Any) -> str:
    probe = getattr(scf_out, "df_B", None)
    if probe is None:
        probe = getattr(getattr(scf_out, "scf", None), "mo_coeff", None)
    try:
        import cupy as cp  # noqa: PLC0415

        if isinstance(probe, cp.ndarray):  # type: ignore[attr-defined]
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _casscf_root_energy(casscf: Any, *, iroot: int) -> float:
    # Prefer per-root energies if available; fall back to .e_tot.
    for attr in ("e_roots", "e_states", "e_state", "e_root"):
        if hasattr(casscf, attr):
            try:
                arr = np.asarray(getattr(casscf, attr), dtype=np.float64).ravel()
                if int(arr.size) > int(iroot):
                    return float(arr[int(iroot)])
            except Exception:
                pass
    return float(np.asarray(getattr(casscf, "e_tot"), dtype=np.float64).ravel()[0])


def _build_cuda_df_blocks_for_ss(
    *,
    B_ao: np.ndarray,
    C: np.ndarray,
    ncore: int,
    ncas: int,
    nvirt: int,
    max_memory_mb: float,
) -> Any:
    from asuka.caspt2.cuda.rhs_df_cuda import CASPT2DFBlocks  # noqa: PLC0415
    from asuka.mrpt2.df_pair_block import DFPairBlock, build_df_pair_blocks_from_df_B  # noqa: PLC0415

    B_full = np.asarray(B_ao, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    nao = int(C.shape[0])
    if B_full.ndim == 2:
        from asuka.integrals.df_packed_s2 import unpack_Qp_to_mnQ  # noqa: PLC0415

        B_full = np.asarray(unpack_Qp_to_mnQ(B_full, nao=nao), dtype=np.float64)
    if B_full.ndim != 3:
        raise ValueError(f"invalid DF tensor shape for CUDA PT2 FD energy: {B_full.shape}")

    naux = int(B_full.shape[2])
    C_core = np.asarray(C[:, :ncore], dtype=np.float64, order="C")
    C_act = np.asarray(C[:, ncore : ncore + ncas], dtype=np.float64, order="C")
    C_virt = np.asarray(C[:, ncore + ncas :], dtype=np.float64, order="C")

    def _empty(nx: int, ny: int) -> DFPairBlock:
        return DFPairBlock(
            nx=int(nx),
            ny=int(ny),
            l_full=np.zeros((int(nx) * int(ny), naux), dtype=np.float64),
            pair_norm=None,
        )

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    labels: list[str] = []
    if ncore > 0:
        pairs.append((C_core, C_core)); labels.append("ii")
    if ncore > 0 and ncas > 0:
        pairs.append((C_core, C_act)); labels.append("it")
    if ncore > 0 and nvirt > 0:
        pairs.append((C_core, C_virt)); labels.append("ia")
    if nvirt > 0 and ncas > 0:
        pairs.append((C_virt, C_act)); labels.append("at")
    if ncas > 0:
        pairs.append((C_act, C_act)); labels.append("tu")
    if nvirt > 0:
        pairs.append((C_virt, C_virt)); labels.append("ab")

    by_label: dict[str, DFPairBlock] = {}
    if pairs:
        built = build_df_pair_blocks_from_df_B(
            B_full,
            pairs,
            max_memory=int(max(1.0, float(max_memory_mb))),
            compute_pair_norm=False,
        )
        by_label = dict(zip(labels, built))

    return CASPT2DFBlocks(
        l_it=by_label.get("it", _empty(ncore, ncas)),
        l_ia=by_label.get("ia", _empty(ncore, nvirt)),
        l_at=by_label.get("at", _empty(nvirt, ncas)),
        l_tu=by_label.get("tu", _empty(ncas, ncas)),
        l_ii=by_label.get("ii", None if ncore == 0 else _empty(ncore, ncore)),
        l_ab=by_label.get("ab", None if nvirt == 0 else _empty(nvirt, nvirt)),
    )


def _caspt2_energy_ss_cpu(
    scf_out: Any,
    casscf: Any,
    *,
    iroot: int,
    nstates: int,
    caspt2_kwargs: dict[str, Any],
) -> Any:
    """CPU-only SS CASPT2 energy helper for FD gradients.

    This bypasses `caspt2_from_casscf` (GPU-first) by reusing the existing
    pure-NumPy SS energy drivers:
      - IC: `asuka.caspt2.energy.caspt2_energy_ss`
      - SST: `asuka.caspt2.sst.sst_caspt2_energy_ss`
    """
    from types import SimpleNamespace  # noqa: PLC0415

    from asuka.caspt2.energy import caspt2_energy_ss  # noqa: PLC0415
    from asuka.caspt2.f3 import CASPT2CIContext  # noqa: PLC0415
    from asuka.caspt2.fock_df import build_caspt2_fock_ao  # noqa: PLC0415
    from asuka.caspt2.superindex import build_superindex  # noqa: PLC0415
    from asuka.cuguga.drt import build_drt  # noqa: PLC0415
    from asuka.rdm.rdm123 import _make_rdm123_pyscf, _reorder_dm123_molcas  # noqa: PLC0415

    pt2_backend_norm = str(caspt2_kwargs.get("pt2_backend", "ic")).strip().lower()
    imag_shift = float(caspt2_kwargs.get("imag_shift", 0.0))
    real_shift = float(caspt2_kwargs.get("real_shift", 0.0))
    tol = float(caspt2_kwargs.get("tol", 1e-8))
    maxiter = int(caspt2_kwargs.get("maxiter", 200))
    threshold = float(caspt2_kwargs.get("threshold", 1e-10))
    threshold_s = float(caspt2_kwargs.get("threshold_s", 1e-8))
    cuda_device = caspt2_kwargs.get("cuda_device", None)
    cuda_mode = str(caspt2_kwargs.get("cuda_mode", "hybrid"))
    cuda_f3_cache_bytes = int(caspt2_kwargs.get("cuda_f3_cache_bytes", 512 * 1024 * 1024))
    cuda_profile = bool(caspt2_kwargs.get("cuda_profile", False))
    max_memory_mb = float(caspt2_kwargs.get("max_memory_mb", 1024.0))
    verbose = int(caspt2_kwargs.get("verbose", 0))

    ncore = int(getattr(casscf, "ncore"))
    ncas = int(getattr(casscf, "ncas"))
    C = np.asarray(getattr(casscf, "mo_coeff"), dtype=np.float64)
    nao, nmo = C.shape
    nvirt = int(nmo - ncore - ncas)

    B_ao = getattr(scf_out, "df_B", None)
    if B_ao is None:
        raise ValueError("CPU FD CASPT2 energy requires scf_out.df_B (DF factors)")
    B_ao = np.asarray(B_ao.get() if hasattr(B_ao, "get") else B_ao, dtype=np.float64)
    if B_ao.ndim == 2:
        from asuka.integrals.df_packed_s2 import unpack_Qp_to_mnQ  # noqa: PLC0415

        B_ao = np.asarray(unpack_Qp_to_mnQ(B_ao, nao=int(nao)), dtype=np.float64)

    h_ao = getattr(getattr(scf_out, "int1e", None), "hcore", None)
    if h_ao is None:
        raise ValueError("scf_out.int1e.hcore is missing for CPU FD CASPT2 energy")
    h_ao = np.asarray(h_ao.get() if hasattr(h_ao, "get") else h_ao, dtype=np.float64)

    mol = getattr(scf_out, "mol", None)
    if mol is None:
        raise ValueError("scf_out.mol is missing for CPU FD CASPT2 energy")
    e_nuc = float(getattr(mol, "energy_nuc")())
    e_ref = _casscf_root_energy(casscf, iroot=int(iroot))

    nelecas = getattr(casscf, "nelecas")
    nelec_total = int(nelecas) if isinstance(nelecas, (int, np.integer)) else int(sum(nelecas))
    twos = int(getattr(mol, "spin", 0))
    drt = build_drt(norb=int(ncas), nelec=int(nelec_total), twos_target=int(twos))
    ci_vec = np.asarray(ci_as_list(getattr(casscf, "ci"), nroots=int(nstates))[int(iroot)], dtype=np.float64).ravel()
    dm1, dm2, dm3 = _make_rdm123_pyscf(drt, ci_vec, reorder=False)
    dm1, dm2, dm3 = _reorder_dm123_molcas(dm1, dm2, dm3, inplace=True)

    fock = build_caspt2_fock_ao(
        h_ao,
        B_ao,
        C,
        dm1,
        int(ncore),
        int(ncas),
        int(nvirt),
        e_nuc=float(e_nuc),
    )
    smap = build_superindex(int(ncore), int(ncas), int(nvirt))
    ci_ctx = CASPT2CIContext(drt=drt, ci_csf=ci_vec)

    if pt2_backend_norm in {"sst", "sst-ic", "sst-full"}:
        from asuka.caspt2.sst import sst_caspt2_energy_ss  # noqa: PLC0415
        from asuka.caspt2.sst.types import SSTConfig, SSTInput  # noqa: PLC0415

        sst_mode = "ic" if pt2_backend_norm == "sst-ic" else "full"
        cfg = SSTConfig(
            imag_shift=float(imag_shift),
            real_shift=float(real_shift),
            tol=float(tol),
            maxiter=int(maxiter),
            threshold=float(threshold),
            threshold_s=float(threshold_s),
            verbose=int(verbose),
        )
        inp = SSTInput(
            ncore=int(ncore),
            ncas=int(ncas),
            nvirt=int(nvirt),
            mo_coeff=np.asarray(C, dtype=np.float64),
            dm1_act=np.asarray(dm1, dtype=np.float64),
            dm2_act=np.asarray(dm2, dtype=np.float64),
            fock=fock,
            semicanonical=None,
            e_ref=float(e_ref),
            e_nuc=float(e_nuc),
            dm3_act=np.asarray(dm3, dtype=np.float64),
            ci_context=ci_ctx,
            smap=smap,
            B_ao=np.asarray(B_ao, dtype=np.float64),
            eri_mo=None,
        )
        sst_res = sst_caspt2_energy_ss(inp, cfg, sst_mode=str(sst_mode))
        return SimpleNamespace(
            e_ref=float(e_ref),
            e_pt2=float(getattr(sst_res, "e_pt2")),
            e_tot=float(getattr(sst_res, "e_tot")),
            method="SS",
            breakdown=dict(getattr(sst_res, "breakdown", {}) or {}),
        )

    if pt2_backend_norm in {"cuda", "cupy", "gpu", "df", "df-cuda"}:
        df_blocks = _build_cuda_df_blocks_for_ss(
            B_ao=np.asarray(B_ao, dtype=np.float64),
            C=np.asarray(C, dtype=np.float64),
            ncore=int(ncore),
            ncas=int(ncas),
            nvirt=int(nvirt),
            max_memory_mb=float(max_memory_mb),
        )
        return caspt2_energy_ss(
            smap=smap,
            fock=fock,
            eri_mo=np.empty((0, 0, 0, 0), dtype=np.float64),
            dm1=np.asarray(dm1, dtype=np.float64),
            dm2=np.asarray(dm2, dtype=np.float64),
            dm3=np.asarray(dm3, dtype=np.float64),
            e_ref=float(e_ref),
            ci_context=ci_ctx,
            pt2_backend="cuda",
            cuda_device=None if cuda_device is None else int(cuda_device),
            cuda_mode=str(cuda_mode),
            cuda_f3_cache_bytes=int(cuda_f3_cache_bytes),
            cuda_profile=bool(cuda_profile),
            df_blocks=df_blocks,
            imag_shift=float(imag_shift),
            real_shift=float(real_shift),
            tol=float(tol),
            maxiter=int(maxiter),
            threshold=float(threshold),
            threshold_s=float(threshold_s),
            verbose=int(verbose),
        )

    # IC SS energy on CPU
    b_mo = np.einsum("mi,mnP,nj->ijP", C, B_ao, C, optimize=True)
    naux = int(b_mo.shape[2])
    b2 = b_mo.reshape(nmo * nmo, naux)
    eri_mo = np.asarray((b2 @ b2.T).reshape(nmo, nmo, nmo, nmo), dtype=np.float64)

    res = caspt2_energy_ss(
        smap=smap,
        fock=fock,
        eri_mo=eri_mo,
        dm1=np.asarray(dm1, dtype=np.float64),
        dm2=np.asarray(dm2, dtype=np.float64),
        dm3=np.asarray(dm3, dtype=np.float64),
        e_ref=float(e_ref),
        ci_context=ci_ctx,
        pt2_backend="cpu",
        imag_shift=float(imag_shift),
        real_shift=float(real_shift),
        tol=float(tol),
        maxiter=int(maxiter),
        threshold=float(threshold),
        threshold_s=float(threshold_s),
        verbose=int(verbose),
    )
    return res


def _caspt2_energy_ms_cpu_ic(
    scf_out: Any,
    casscf: Any,
    *,
    method: str,
    iroot: int,
    nstates: int,
    caspt2_kwargs: dict[str, Any],
    base_ci_vecs: list[np.ndarray] | None = None,
) -> Any:
    """CPU-only IC MS/XMS-CASPT2 energy for deterministic FD gradients.

    Uses the same IC energy + CPU Heff path as `caspt2_ms_gradient_native`
    with ``ASUKA_SS_ERI_MODE=dense``, ensuring FD and analytic are consistent.
    Activated via ``ASUKA_MS_CUDA=0``.

    Parameters
    ----------
    base_ci_vecs : list of arrays, optional
        Reference CI vectors for phase-fixing.  When provided, the signs of
        each CI vector extracted from ``casscf`` are adjusted so that
        ``<base_ci[i] | ci[i]> >= 0`` before running PT2.  This ensures
        consistent off-diagonal Heff coupling signs across displaced-geometry
        evaluations, removing a primary source of FD non-determinism.
    """
    from types import SimpleNamespace  # noqa: PLC0415

    from asuka.caspt2.energy import caspt2_energy_ss  # noqa: PLC0415
    from asuka.caspt2.f3 import CASPT2CIContext  # noqa: PLC0415
    from asuka.caspt2.fock import build_caspt2_fock  # noqa: PLC0415
    from asuka.caspt2.fock_df import build_caspt2_fock_ao  # noqa: PLC0415
    from asuka.caspt2.multistate import build_heff, diagonalize_heff  # noqa: PLC0415
    from asuka.caspt2.superindex import build_superindex  # noqa: PLC0415
    from asuka.cuguga.drt import build_drt  # noqa: PLC0415
    from asuka.mcscf.state_average import ci_as_list  # noqa: PLC0415
    from asuka.rdm.rdm123 import _make_rdm123_pyscf, _reorder_dm123_molcas  # noqa: PLC0415

    method_u = str(method).upper().strip()
    imag_shift = float(caspt2_kwargs.get("imag_shift", 0.0))
    real_shift = float(caspt2_kwargs.get("real_shift", 0.0))
    tol = float(caspt2_kwargs.get("tol", 1e-8))
    maxiter = int(caspt2_kwargs.get("maxiter", 200))
    threshold = float(caspt2_kwargs.get("threshold", 1e-10))
    threshold_s = float(caspt2_kwargs.get("threshold_s", 1e-8))
    verbose = int(caspt2_kwargs.get("verbose", 0))

    def _to_np(x: Any) -> np.ndarray:
        """Convert CuPy or numpy array to numpy float64."""
        if hasattr(x, "get"):
            x = x.get()
        return np.asarray(x, dtype=np.float64)

    ncore = int(getattr(casscf, "ncore"))
    ncas = int(getattr(casscf, "ncas"))
    C = _to_np(getattr(casscf, "mo_coeff"))
    nao, nmo = C.shape
    nvirt = int(nmo - ncore - ncas)
    nocc = ncore + ncas

    h_ao = _to_np(getattr(getattr(scf_out, "int1e", None), "hcore"))
    mol = getattr(scf_out, "mol", None)
    e_nuc = float(getattr(mol, "energy_nuc")())

    nelecas = getattr(casscf, "nelecas")
    nelec_total = int(nelecas) if isinstance(nelecas, (int, np.integer)) else int(sum(nelecas))
    twos = int(getattr(mol, "spin", 0))
    drt = build_drt(norb=int(ncas), nelec=int(nelec_total), twos_target=int(twos))
    smap = build_superindex(int(ncore), int(ncas), int(nvirt))
    ci_vecs = [
        np.asarray(v, dtype=np.float64).ravel()
        for v in ci_as_list(getattr(casscf, "ci"), nroots=int(nstates))
    ]

    # Fix CI phases against reference (base_ci_vecs) to ensure consistent
    # off-diagonal Heff coupling signs across displaced-geometry evaluations.
    # Without this, SA-CASSCF CI vectors at displaced geometries can have
    # arbitrary sign relative to the reference, causing the Heff off-diagonal
    # to flip sign and making the FD gradient non-deterministic.
    if base_ci_vecs is not None and len(base_ci_vecs) == len(ci_vecs):
        for i in range(len(ci_vecs)):
            if float(np.dot(base_ci_vecs[i], ci_vecs[i])) < 0.0:
                ci_vecs[i] = -ci_vecs[i]

    # Build MO ERIs (dense 4-index, matching gradient path)
    eri_mode = str(os.environ.get("ASUKA_SS_ERI_MODE", "df")).strip().lower()
    if eri_mode in {"dense", "full", "eri_dense"}:
        from asuka.hf.dense_eri import build_ao_eri_dense  # noqa: PLC0415
        ao_basis = getattr(scf_out, "ao_basis", None)
        if ao_basis is None:
            raise TypeError("scf_out.ao_basis required for dense ERIs")
        dense_backend = str(os.environ.get("ASUKA_SS_DENSE_ERI_BACKEND", "cuda")).strip().lower()
        eri_res = build_ao_eri_dense(ao_basis, backend=dense_backend, eps_ao=0.0)
        eri_mat = _to_np(getattr(eri_res, "eri_mat")).reshape(nao, nao, nao, nao)
        eri_mo = np.asarray(
            np.einsum("mp,nq,lr,ks,mnlk->pqrs", C, C, C, C, eri_mat, optimize=True),
            dtype=np.float64,
        )
    else:
        B_ao = getattr(scf_out, "df_B", None)
        if B_ao is None:
            raise ValueError("scf_out.df_B required for DF ERIs in CPU MS energy")
        B_ao = np.asarray(B_ao.get() if hasattr(B_ao, "get") else B_ao, dtype=np.float64)
        if B_ao.ndim == 2:
            from asuka.integrals.df_packed_s2 import unpack_Qp_to_mnQ  # noqa: PLC0415
            B_ao = np.asarray(unpack_Qp_to_mnQ(B_ao, nao=int(nao)), dtype=np.float64)
        b_mo = np.einsum("mi,mnP,nj->ijP", C, B_ao, C, optimize=True)
        naux = b_mo.shape[2]
        b2 = b_mo.reshape(nmo * nmo, naux)
        eri_mo = np.asarray((b2 @ b2.T).reshape(nmo, nmo, nmo, nmo), dtype=np.float64)

    # Build per-state RDMs and Fock
    dm1_list, dm2_list, dm3_list = [], [], []
    fock_list = []
    e_ref_list = []
    if method_u == "XMS":
        from asuka.caspt2.xms import xms_rotate_states  # noqa: PLC0415
        _dm1_orig = []
        for _I in range(nstates):
            d1, _, _ = _make_rdm123_pyscf(drt, ci_vecs[_I], reorder=False)
            _dm1_orig.append(d1)
        dm1_sa = np.mean(np.stack(_dm1_orig), axis=0)
        h_mo = np.asarray(C.T @ h_ao @ C, dtype=np.float64)
        fock_sa = build_caspt2_fock(h_mo, eri_mo, dm1_sa, ncore, ncas, nvirt, e_nuc=e_nuc)
        _casscf_e_roots = np.asarray(getattr(casscf, "e_roots"), dtype=np.float64).ravel()
        rotated_ci, u0, _ = xms_rotate_states(drt, ci_vecs, _dm1_orig, fock_sa, ncore, ncas, nstates)
        ci_vecs = [np.asarray(c, dtype=np.float64).ravel() for c in rotated_ci]
        e_ref_list = [float(_casscf_e_roots[I]) for I in range(nstates)]
        for I in range(nstates):
            d1, d2, d3 = _make_rdm123_pyscf(drt, ci_vecs[I], reorder=False)
            d1, d2, d3 = _reorder_dm123_molcas(d1, d2, d3, inplace=True)
            dm1_list.append(d1); dm2_list.append(d2); dm3_list.append(d3)
            fock_list.append(fock_sa)
    else:
        for I in range(nstates):
            d1, d2, d3 = _make_rdm123_pyscf(drt, ci_vecs[I], reorder=False)
            d1, d2, d3 = _reorder_dm123_molcas(d1, d2, d3, inplace=True)
            dm1_list.append(d1); dm2_list.append(d2); dm3_list.append(d3)
            h_mo_I = np.asarray(C.T @ h_ao @ C, dtype=np.float64)
            fock_I = build_caspt2_fock(h_mo_I, eri_mo, d1, ncore, ncas, nvirt, e_nuc=e_nuc)
            fock_list.append(fock_I)
            eri_act = eri_mo[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc]
            e_ref_I = (
                float(fock_I.e_core)
                + float(np.einsum("tu,tu->", fock_I.fimo[ncore:nocc, ncore:nocc], d1))
                + 0.5 * float(np.einsum("tuvx,tuvx->", eri_act, d2.reshape(ncas, ncas, ncas, ncas)))
            )
            e_ref_list.append(e_ref_I)

    # SS-CASPT2 energy per state (CPU IC backend, stores amplitudes for Heff)
    ss_results = []
    for I in range(nstates):
        ci_ctx = CASPT2CIContext(drt=drt, ci_csf=ci_vecs[I])
        res_I = caspt2_energy_ss(
            smap, fock_list[I], eri_mo, dm1_list[I], dm2_list[I], dm3_list[I], e_ref_list[I],
            ci_context=ci_ctx,
            pt2_backend="cpu",
            imag_shift=float(imag_shift),
            real_shift=float(real_shift),
            tol=float(tol),
            maxiter=int(maxiter),
            threshold=float(threshold),
            threshold_s=float(threshold_s),
            store_sb_decomp=True,
            verbose=int(verbose),
        )
        ss_results.append(res_I)

    # Build MS Heff (CPU, dense ERIs) and diagonalize
    _heff_fock = fock_list[0] if method_u == "XMS" else fock_list
    heff = build_heff(
        nstates, ss_results, ci_vecs, drt, smap,
        _heff_fock, eri_mo, dm1_list, dm2_list, dm3_list,
        verbose=int(verbose),
    )
    ms_energies, ueff = diagonalize_heff(heff)

    bd: dict[str, Any] = {
        "ss_energies": [float(r.e_tot) for r in ss_results],
        "ms_energies": ms_energies.tolist(),
        "heff_backend": "cpu_ic",
    }
    if method_u == "XMS":
        bd["u0"] = np.asarray(u0, dtype=np.float64)

    return SimpleNamespace(
        e_ref=e_ref_list,
        e_pt2=[float(ms_energies[i] - float(e_ref_list[i])) for i in range(nstates)],
        e_tot=ms_energies.tolist(),
        heff=np.asarray(heff, dtype=np.float64),
        ueff=np.asarray(ueff, dtype=np.float64),
        method=method_u,
        breakdown=bd,
    )


def fd_nuclear_gradient_from_casscf(
    scf_out: Any,
    casscf: Any,
    *,
    method: str,
    iroot: int,
    verbose: int = 0,
    step_bohr: float | None = None,
    caspt2_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, np.ndarray, list[dict[str, Any]]]:
    """Finite-difference CASPT2 gradient from an ASUKA CASSCF reference object."""
    mol0 = getattr(scf_out, "mol", None)
    if mol0 is None:
        raise TypeError("scf_out must provide .mol for FD CASPT2 gradients")
    if getattr(scf_out, "df_B", None) is None:
        raise NotImplementedError("FD CASPT2 gradient wrapper currently expects a DF SCF result")

    method_u = str(method).upper().strip()
    nstates = int(getattr(casscf, "nroots", 1))
    if method_u in {"MS", "XMS"} and nstates <= 1:
        raise ValueError(f"{method_u} gradients require nroots > 1")

    hf_backend = _infer_backend(scf_out)
    hf_method = _infer_hf_method(scf_out)
    auxbasis_name_raw = str(getattr(scf_out, "auxbasis_name", "autoaux"))
    # Dense mode: scf_out has ao_eri but no real aux basis (df_B may have been
    # built from ao_eri for PT2 internal use, but FD displaced points need dense HF).
    _use_df = getattr(scf_out, "ao_eri", None) is None
    auxbasis_arg = "autoaux" if auxbasis_name_raw.strip().lower().endswith("_autoaux") else auxbasis_name_raw
    coords0 = np.asarray(getattr(mol0, "coords_bohr"), dtype=np.float64).reshape((-1, 3))
    root_weights = getattr(casscf, "root_weights", None)
    ncore = int(getattr(casscf, "ncore"))
    ncas = int(getattr(casscf, "ncas"))
    nelecas = getattr(casscf, "nelecas")

    cas_kwargs: dict[str, Any] = {
        "ncore": int(ncore),
        "ncas": int(ncas),
        "nelecas": nelecas,
        "backend": str(hf_backend),
        "df": bool(_use_df),
    }
    reuse_guess = _bool_env("ASUKA_CASPT2_FD_REUSE_CASSCF_GUESS", True)
    tracking_method = str(os.environ.get("ASUKA_CASPT2_FD_ORBITAL_TRACKING_METHOD", "subspace")).strip().lower()
    if tracking_method not in {"subspace", "hungarian"}:
        tracking_method = "subspace"
    if int(nstates) > 1:
        cas_kwargs["nroots"] = int(nstates)
        if root_weights is not None:
            cas_kwargs["root_weights"] = tuple(float(x) for x in np.asarray(root_weights, dtype=np.float64).ravel().tolist())
    for name in ("frozen", "internal_rotation"):
        if hasattr(casscf, name):
            val = getattr(casscf, name)
            if val is not None:
                cas_kwargs[name] = val

    prev_mol = mol0
    _mo_coeff_raw = getattr(casscf, "mo_coeff")
    prev_casscf_mo = np.asarray(_mo_coeff_raw.get() if hasattr(_mo_coeff_raw, "get") else _mo_coeff_raw, dtype=np.float64)
    prev_ci = getattr(casscf, "ci", None)

    def _tracked_guess_mo_coeff0(*, mol_new, scf_out_new) -> np.ndarray | None:
        if not bool(reuse_guess):
            return None
        try:
            from asuka.frontend.one_electron import build_ao_basis_cart  # noqa: PLC0415
            from asuka.integrals.cross_geometry import build_S_cross  # noqa: PLC0415
            from asuka.mcscf.orbital_tracking import (  # noqa: PLC0415
                align_orbital_phases,
                assign_active_orbitals_by_overlap,
                reorder_mo_to_active_space,
            )
        except Exception:
            return None

        try:
            basis_prev, _ = build_ao_basis_cart(prev_mol)
            basis_new, _ = build_ao_basis_cart(mol_new)
            T_prev = None
            T_new = None
            if not bool(getattr(mol_new, "cart", True)):
                from asuka.integrals.cart2sph import build_cart2sph_matrix, compute_sph_layout_from_cart_basis  # noqa: PLC0415
                from asuka.integrals.int1e_cart import nao_cart_from_basis  # noqa: PLC0415

                sh0_sph_prev, nao_sph_prev = compute_sph_layout_from_cart_basis(basis_prev)
                T_prev = build_cart2sph_matrix(
                    basis_prev.shell_l,
                    basis_prev.shell_ao_start,
                    sh0_sph_prev,
                    nao_cart_from_basis(basis_prev),
                    nao_sph_prev,
                )
                sh0_sph_new, nao_sph_new = compute_sph_layout_from_cart_basis(basis_new)
                T_new = build_cart2sph_matrix(
                    basis_new.shell_l,
                    basis_new.shell_ao_start,
                    sh0_sph_new,
                    nao_cart_from_basis(basis_new),
                    nao_sph_new,
                )
            S_cross = build_S_cross(basis_prev, basis_new, T_bra=T_prev, T_ket=T_new)
            scf_mo = np.asarray(getattr(getattr(scf_out_new, "scf", None), "mo_coeff", None), dtype=np.float64)
            prev_active_idx = list(range(int(ncore), int(ncore) + int(ncas)))
            new_active_idx = assign_active_orbitals_by_overlap(
                prev_casscf_mo,
                scf_mo,
                S_cross,
                prev_active_idx,
                int(ncas),
                method=tracking_method,
            )
            scf_mo_reordered = reorder_mo_to_active_space(scf_mo, new_active_idx, int(ncore))
            scf_mo_aligned = align_orbital_phases(
                prev_casscf_mo,
                scf_mo_reordered,
                S_cross,
                alignment_idx=range(int(ncore), int(ncore) + int(ncas)),
            )
            return np.asarray(scf_mo_aligned, dtype=np.float64)
        except Exception:
            return None

    def _build(coords_bohr_in: np.ndarray) -> tuple[Any, Any]:
        mol_eval = _clone_molecule_with_coords(mol0, coords_bohr_in)
        scf_kw: dict[str, Any] = {
            "method": str(hf_method),
            "backend": str(hf_backend),
            "df": bool(_use_df),
        }
        if bool(_use_df):
            scf_kw["auxbasis"] = str(auxbasis_arg)
        scf_eval = run_hf_df(mol_eval, **scf_kw)
        if not bool(_use_df):
            # Augment dense HF result with exact DF factors for PT2 internal use.
            # Factorize M[(mu,nu),(la,si)] = ao_eri via eigendecomposition so that
            # B[mu,nu,Q] satisfies M = B @ B^T (Cholesky-like exact factor).
            ao_eri_eval = getattr(scf_eval, "ao_eri", None)
            if ao_eri_eval is not None:
                try:
                    from dataclasses import replace as _dc_replace  # noqa: PLC0415
                    eri_raw = np.asarray(ao_eri_eval, dtype=np.float64)
                    if eri_raw.ndim == 4:
                        nao_e = int(eri_raw.shape[0])
                        m = eri_raw.reshape(nao_e * nao_e, nao_e * nao_e)
                    else:
                        # 2D: shape (nao², nao²)
                        n2 = int(eri_raw.shape[0])
                        nao_e = int(round(np.sqrt(float(n2))))
                        m = eri_raw
                    m = np.asarray(0.5 * (m + m.T), dtype=np.float64)
                    w, u = np.linalg.eigh(m)
                    thr = max(1e-14, 1e-14 * float(max(abs(float(w.max())), abs(float(w.min())))))
                    keep = w > thr
                    b_exact = np.asarray((u[:, keep] * np.sqrt(w[keep])).reshape(nao_e, nao_e, int(keep.sum())), dtype=np.float64)
                    scf_eval = _dc_replace(scf_eval, df_B=b_exact)
                except Exception:
                    pass
        casscf_kw = {k: v for k, v in cas_kwargs.items() if k not in {"guess"}}
        if bool(reuse_guess):
            mo0 = _tracked_guess_mo_coeff0(mol_new=mol_eval, scf_out_new=scf_eval)
            if mo0 is not None:
                casscf_kw["mo_coeff0"] = np.asarray(mo0, dtype=np.float64)
            if prev_ci is not None and mo0 is not None:
                casscf_kw["ci0"] = prev_ci
        cas_eval = run_casscf(scf_eval, **casscf_kw)
        return scf_eval, cas_eval

    kwargs = dict(caspt2_kwargs or {})
    if method_u in {"MS", "XMS"}:
        kwargs.setdefault("heff_backend", str(os.environ.get("ASUKA_CASPT2_MSXMS_HEFF_BACKEND", "cuda")))
        kwargs.setdefault("cuda_mode", str(os.environ.get("ASUKA_CASPT2_MSXMS_CUDA_MODE", "strict")))
    if step_bohr is None:
        step = _fd_step_bohr_from_env()
    else:
        step = abs(float(step_bohr))

    _ms_cpu_ic = (
        method_u in {"MS", "XMS"}
        and str(os.environ.get("ASUKA_MS_CUDA", "1")).strip() == "0"
    )

    # Base CASSCF CI vectors for phase-fixing at displaced geometries.
    # Extracted lazily after the first build (reference geometry).
    _base_ci_for_phase_fix: list[np.ndarray] | None = None

    def _energy(scf_obj: Any, ref_obj: Any, iroot_sel: int):
        nonlocal _base_ci_for_phase_fix
        # GPU-first driver fails on some environments (e.g. no CUDA). Fall back
        # to pure-NumPy SS energy drivers when the SCF/CASSCF objects are on CPU.
        if method_u != "SS" and not _ms_cpu_ic:
            return caspt2_from_casscf(
                scf_obj,
                ref_obj,
                method=method_u,
                nstates=int(nstates),
                iroot=int(iroot_sel),
                **kwargs,
            )
        if method_u != "SS" and _ms_cpu_ic:
            # CPU-only IC MS energy for deterministic FD (matches gradient path).
            # Lazily build base CI vectors on first call (reference geometry) and
            # use them to fix CI phases at displaced geometries — this prevents
            # sign flips in the off-diagonal Heff coupling from run to run.
            cur_ci = [
                np.asarray(v, dtype=np.float64).ravel()
                for v in ci_as_list(getattr(ref_obj, "ci"), nroots=int(nstates))
            ]
            if _base_ci_for_phase_fix is None:
                _base_ci_for_phase_fix = [np.asarray(c, dtype=np.float64).copy() for c in cur_ci]
            return _caspt2_energy_ms_cpu_ic(
                scf_obj,
                ref_obj,
                method=method_u,
                iroot=int(iroot_sel),
                nstates=int(nstates),
                caspt2_kwargs=dict(kwargs),
                base_ci_vecs=_base_ci_for_phase_fix,
            )
        if str(hf_backend).strip().lower() == "cpu":
            return _caspt2_energy_ss_cpu(
                scf_obj,
                ref_obj,
                iroot=int(iroot_sel),
                nstates=int(nstates),
                caspt2_kwargs=dict(kwargs),
            )
        return caspt2_from_casscf(
            scf_obj,
            ref_obj,
            method=method_u,
            nstates=int(nstates),
            iroot=int(iroot_sel),
            **kwargs,
        )

    _scf0, _ref0, res0, _e0, grad, points = _fd_gradient_core(
        _build,
        _energy,
        coords0,
        method=method_u,
        nstates=int(nstates),
        iroot=int(iroot),
        caspt2_kwargs=kwargs,
        delta=float(step),
        which=None,
        verbose=int(verbose),
        base_pair=(scf_out, casscf),
    )
    return res0, np.asarray(grad, dtype=np.float64), points


__all__ = [
    "fd_nuclear_gradient_asuka",
    "fd_nuclear_gradient_from_casscf",
]
