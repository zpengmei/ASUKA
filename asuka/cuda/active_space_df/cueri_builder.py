from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Any
import time

import numpy as np

from .builder import DeviceActiveDF


def _ncart(l: int) -> int:
    l = int(l)
    if l < 0:
        raise ValueError("l must be >= 0")
    return (l + 1) * (l + 2) // 2


def _gaussian_int(n: int, alpha: np.ndarray) -> np.ndarray:
    """Compute ∫_0^∞ x^n exp(-alpha x^2) dx for vector alpha (float64)."""

    from math import gamma

    n = int(n)
    if n < 0:
        raise ValueError("n must be >= 0")
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim != 1:
        raise ValueError("alpha must be a 1D array")
    n1 = 0.5 * float(n + 1)
    return (gamma(n1) / 2.0) / np.power(alpha, n1)


def _gto_norm_radial(l: int, exp: np.ndarray) -> np.ndarray:
    l = int(l)
    if l < 0:
        raise ValueError("l must be >= 0")
    exp = np.asarray(exp, dtype=np.float64)
    if exp.ndim != 1:
        raise ValueError("exp must be a 1D array")
    return 1.0 / np.sqrt(_gaussian_int(l * 2 + 2, 2.0 * exp))


def _primitive_norm_cart_like_pyscf(l: int, exp: np.ndarray) -> np.ndarray:
    """Return primitive norms matching PySCF/libcint cart=True conventions."""

    l = int(l)
    if l < 0:
        raise ValueError("l must be >= 0")
    exp = np.asarray(exp, dtype=np.float64)
    if exp.ndim != 1:
        raise ValueError("exp must be a 1D array")
    if l <= 1:
        # N_l = (2a/pi)^(3/4) * (4a)^(l/2)
        return (2.0 * exp / pi) ** 0.75 * (4.0 * exp) ** (0.5 * l)
    return _gto_norm_radial(l, exp)


def _pack_cart_shells_from_pyscf(mol, *, expand_contractions: bool = True):
    """Pack a PySCF Mole (cart=True) into a cuERI BasisCartSoA."""

    if not bool(getattr(mol, "cart", False)):
        raise ValueError("requires mol.cart=True")

    from asuka.cueri.basis_cart import BasisCartSoA  # noqa: PLC0415

    shell_cxyz: list[np.ndarray] = []
    shell_prim_start: list[int] = []
    shell_nprim: list[int] = []
    shell_l: list[int] = []
    shell_ao_start: list[int] = []
    prim_exp: list[float] = []
    prim_coef: list[float] = []

    bas_id_out: list[int] = []
    ctr_id_out: list[int] = []

    ao_loc = np.asarray(mol.ao_loc_nr(cart=True), dtype=np.int32)

    for bas_id in range(int(mol.nbas)):
        l = int(mol.bas_angular(bas_id))
        nprim = int(mol.bas_nprim(bas_id))
        nctr = int(mol.bas_nctr(bas_id))

        exp = np.asarray(mol.bas_exp(bas_id), dtype=np.float64)
        ctr_coeff = np.asarray(mol.bas_ctr_coeff(bas_id), dtype=np.float64)  # (nprim, nctr)
        if exp.shape != (nprim,):
            raise RuntimeError("unexpected bas_exp shape from PySCF")
        if ctr_coeff.shape != (nprim, nctr):
            raise RuntimeError("unexpected bas_ctr_coeff shape from PySCF")

        norm = _primitive_norm_cart_like_pyscf(l, exp)
        if np.any(~np.isfinite(norm)):
            raise ValueError("non-finite primitive normalization")

        center = np.asarray(mol.bas_coord(bas_id), dtype=np.float64)
        if center.shape != (3,):
            raise RuntimeError("unexpected bas_coord shape from PySCF")

        ctr_iter = range(nctr) if expand_contractions else range(1)
        for ctr_id in ctr_iter:
            shell_cxyz.append(center)
            shell_prim_start.append(len(prim_exp))
            shell_nprim.append(nprim)
            shell_l.append(l)
            shell_ao_start.append(int(ao_loc[bas_id] + ctr_id * _ncart(l)))
            bas_id_out.append(bas_id)
            ctr_id_out.append(ctr_id)

            col = ctr_coeff[:, ctr_id]
            prim_exp.extend(exp.tolist())
            prim_coef.extend((col * norm).tolist())

    shell_cxyz_arr = np.asarray(shell_cxyz, dtype=np.float64)
    shell_prim_start_arr = np.asarray(shell_prim_start, dtype=np.int32)
    shell_nprim_arr = np.asarray(shell_nprim, dtype=np.int32)
    shell_l_arr = np.asarray(shell_l, dtype=np.int32)
    shell_ao_start_arr = np.asarray(shell_ao_start, dtype=np.int32)
    prim_exp_arr = np.asarray(prim_exp, dtype=np.float64)
    prim_coef_arr = np.asarray(prim_coef, dtype=np.float64)

    bas_id_arr = np.asarray(bas_id_out, dtype=np.int32) if bas_id_out else np.empty((0,), dtype=np.int32)
    ctr_id_arr = np.asarray(ctr_id_out, dtype=np.int32) if ctr_id_out else np.empty((0,), dtype=np.int32)

    return BasisCartSoA(
        shell_cxyz=shell_cxyz_arr,
        shell_prim_start=shell_prim_start_arr,
        shell_nprim=shell_nprim_arr,
        shell_l=shell_l_arr,
        shell_ao_start=shell_ao_start_arr,
        prim_exp=prim_exp_arr,
        prim_coef=prim_coef_arr,
        source_bas_id=bas_id_arr,
        source_ctr_id=ctr_id_arr,
    )


def _infer_nao_from_basis_cart_soa(basis: Any) -> int:
    """Infer nao from a cuERI BasisCartSoA (assumes shell_ao_start follows cart ordering)."""

    shell_ao_start = np.asarray(getattr(basis, "shell_ao_start"), dtype=np.int64).ravel()
    shell_l = np.asarray(getattr(basis, "shell_l"), dtype=np.int64).ravel()
    if shell_ao_start.shape != shell_l.shape:
        raise ValueError("basis shell_ao_start and shell_l shapes mismatch")
    if shell_ao_start.size == 0:
        return 0
    ncart = ((shell_l + 1) * (shell_l + 2) // 2).astype(np.int64, copy=False)
    end = shell_ao_start + ncart
    nao = int(np.max(end))
    if nao < 0:
        raise ValueError("invalid inferred nao from basis")
    return nao


@dataclass
class CuERIActiveSpaceDFBuilder:
    """Build active-space DF integrals on GPU via cuERI streamed DF (in-house).

    This builder calls `asuka.cueri.df.active_Lfull_streamed_basis(...)` to produce
    `L_full[pq, L]` directly from AO+aux packed basis objects, and then optionally
    builds `ERI_mat`, `J_ps`, and `pair_norm` on GPU.

    Notes
    -----
    - This path currently requires `mol.cart=True` and (for `backend='gpu_rys'`)
      AO/aux shells within compiled cuERI CUDA angular limits.
    - The packed basis objects are kept alive on the builder instance so cuERI's
      internal plan caches can hit across macro-iterations.
    """

    mol: Any | None
    # Optional standalone mode: provide pre-packed basis objects directly.
    ao_basis: Any | None = None
    aux_basis: Any | None = None
    auxbasis: Any = "auto"
    backend: str = "gpu_rys"
    threads: int = 256
    mode: str = "auto"
    work_small_max: int = 512
    work_large_min: int = 200_000
    blocks_per_task: int = 4
    aux_block_naux: int = 256
    max_tile_bytes: int = 256 * 1024 * 1024
    df_strategy: str = "auto"
    ao_contract_mode: str = "auto"
    graph_capture: bool = False
    stream: Any | None = None

    def __post_init__(self) -> None:
        try:
            import cupy as cp
        except Exception as e:  # pragma: no cover
            raise RuntimeError("CuPy is required for CuERIActiveSpaceDFBuilder") from e
        try:
            from asuka.cueri.gpu import CUDA_MAX_L as _cuda_max_l  # noqa: PLC0415
        except Exception:
            _cuda_max_l = 5

        try:
            from asuka.cueri import df as cueri_df
        except Exception as e:  # pragma: no cover
            raise RuntimeError("cuERI is required for CuERIActiveSpaceDFBuilder") from e

        backend = str(self.backend).lower().strip()
        if backend not in ("gpu_ss", "gpu_sp", "gpu_rys"):
            raise ValueError("backend must be one of: 'gpu_ss', 'gpu_sp', 'gpu_rys'")
        df_strategy = str(self.df_strategy).lower().strip().replace("-", "_")
        if df_strategy not in ("auto", "x_block", "digest"):
            raise ValueError("df_strategy must be one of: 'auto', 'x_block', 'digest'")
        object.__setattr__(self, "df_strategy", df_strategy)
        ao_contract_mode = str(self.ao_contract_mode).lower().strip()
        if ao_contract_mode not in ("auto", "expanded", "native_contracted"):
            raise ValueError("ao_contract_mode must be one of: 'auto', 'expanded', 'native_contracted'")
        object.__setattr__(self, "ao_contract_mode", ao_contract_mode)
        object.__setattr__(self, "graph_capture", bool(self.graph_capture))

        ao_basis = self.ao_basis
        aux_basis = self.aux_basis
        if (ao_basis is not None) or (aux_basis is not None):
            # Standalone mode: AO/aux bases are provided by the caller.
            if ao_basis is None or aux_basis is None:
                raise ValueError("ao_basis and aux_basis must be provided together")

            lmax_ao = int(np.max(np.asarray(getattr(ao_basis, "shell_l"), dtype=np.int32))) if int(getattr(ao_basis, "shell_l").size) else 0  # type: ignore[attr-defined]
            lmax_aux = int(np.max(np.asarray(getattr(aux_basis, "shell_l"), dtype=np.int32))) if int(getattr(aux_basis, "shell_l").size) else 0  # type: ignore[attr-defined]
            if backend == "gpu_sp":
                if lmax_ao > 1 or lmax_aux > 1:
                    raise NotImplementedError("backend='gpu_sp' currently requires s/p-only shells (lmax <= 1)")
            elif backend == "gpu_rys":
                if lmax_ao > int(_cuda_max_l) or lmax_aux > int(_cuda_max_l):
                    raise NotImplementedError(
                        f"backend='gpu_rys' currently requires lmax <= {_cuda_max_l} (AO and aux)"
                    )

            object.__setattr__(self, "_ao_basis", ao_basis)
            object.__setattr__(self, "_aux_basis", aux_basis)
            object.__setattr__(self, "_nao", _infer_nao_from_basis_cart_soa(ao_basis))
            object.__setattr__(self, "_naux", _infer_nao_from_basis_cart_soa(aux_basis))
        else:
            mol = self.mol
            if mol is None:
                raise ValueError("mol must be provided when ao_basis/aux_basis are not given")
            if not bool(getattr(mol, "cart", False)):
                raise NotImplementedError("CuERIActiveSpaceDFBuilder currently requires mol.cart=True")

            # Standalone mode: build packed AO/AUX bases from a mol-like object.
            #
            # auxbasis="auto" uses Basis Set Exchange `get_aux` (autoaux) based on the
            # orbital basis name, so it requires `basis_set_exchange` to be installed.
            try:
                from asuka.integrals.df_context import _build_df_bases_cart  # noqa: PLC0415
            except Exception as e:  # pragma: no cover
                raise RuntimeError("failed to import ASUKA DF basis builder") from e

            try:
                ao_basis, aux_basis, _aux_name = _build_df_bases_cart(
                    mol,
                    auxbasis=self.auxbasis,
                    expand_contractions=True,
                )
            except Exception as e:
                raise RuntimeError(
                    "failed to build packed AO/AUX bases (standalone mode). "
                    "If using auxbasis='auto', install `basis_set_exchange` (e.g. `pip install asuka[frontend]`). "
                    "Otherwise pass an explicit auxbasis dict or provide pre-packed bases via ao_basis/aux_basis."
                ) from e

            lmax_ao = int(np.max(np.asarray(getattr(ao_basis, "shell_l"), dtype=np.int32))) if int(getattr(ao_basis, "shell_l").size) else 0  # type: ignore[attr-defined]
            lmax_aux = int(np.max(np.asarray(getattr(aux_basis, "shell_l"), dtype=np.int32))) if int(getattr(aux_basis, "shell_l").size) else 0  # type: ignore[attr-defined]
            if backend == "gpu_sp":
                if lmax_ao > 1 or lmax_aux > 1:
                    raise NotImplementedError("backend='gpu_sp' currently requires s/p-only shells (lmax <= 1)")
            elif backend == "gpu_rys":
                if lmax_ao > int(_cuda_max_l) or lmax_aux > int(_cuda_max_l):
                    raise NotImplementedError(
                        f"backend='gpu_rys' currently requires lmax <= {_cuda_max_l} (AO and aux)"
                    )

            object.__setattr__(self, "_ao_basis", ao_basis)
            object.__setattr__(self, "_aux_basis", aux_basis)
            object.__setattr__(self, "_nao", _infer_nao_from_basis_cart_soa(ao_basis))
            object.__setattr__(self, "_naux", _infer_nao_from_basis_cart_soa(aux_basis))
        object.__setattr__(self, "_device_id", int(cp.cuda.Device().id))

    def allocate(
        self,
        norb: int,
        *,
        want_eri_mat: bool = True,
        want_j_ps: bool = True,
        want_pair_norm: bool = True,
    ) -> DeviceActiveDF:
        import cupy as cp

        norb = int(norb)
        if norb <= 0:
            raise ValueError("norb must be > 0")

        naux = int(getattr(self, "_naux"))
        nops = norb * norb

        l_full = cp.empty((nops, naux), dtype=cp.float64)
        eri_mat = cp.empty((nops, nops), dtype=cp.float64) if bool(want_eri_mat) else None
        j_ps = cp.empty((norb, norb), dtype=cp.float64) if bool(want_j_ps) else None
        pair_norm = cp.empty((nops,), dtype=cp.float64) if bool(want_pair_norm) else None
        return DeviceActiveDF(norb=norb, naux=naux, l_full=l_full, j_ps=j_ps, pair_norm=pair_norm, eri_mat=eri_mat)

    def build(
        self,
        c_cas,
        *,
        want_eri_mat: bool = True,
        want_j_ps: bool = True,
        want_pair_norm: bool = True,
        out: DeviceActiveDF | None = None,
        profile: dict | None = None,
        cached_b_whitened: Any | None = None,
        cache_out: dict | None = None,
    ) -> DeviceActiveDF:
        import cupy as cp

        if int(cp.cuda.Device().id) != int(getattr(self, "_device_id")):
            raise RuntimeError("CuERIActiveSpaceDFBuilder must be used on the same CUDA device it was created on")

        c_cas = cp.asarray(c_cas)
        if cp.iscomplexobj(c_cas):
            raise NotImplementedError("CuERIActiveSpaceDFBuilder currently supports real orbitals only (float64)")
        c_cas = cp.asarray(c_cas, dtype=cp.float64)
        c_cas = cp.ascontiguousarray(c_cas)
        if c_cas.ndim != 2:
            raise ValueError("c_cas must be 2D (nao,norb)")

        nao, norb = map(int, c_cas.shape)
        nao_expected = int(getattr(self, "_nao"))
        if nao != nao_expected:
            raise ValueError(f"c_cas has nao={nao}, expected {nao_expected} (mol.nao_nr(cart=True))")
        if norb <= 0:
            raise ValueError("norb must be > 0")

        naux = int(getattr(self, "_naux"))
        nops = norb * norb

        if out is not None:
            if int(out.norb) != norb:
                raise ValueError(f"out.norb={int(out.norb)} does not match norb={norb}")
            if int(out.naux) != naux:
                raise ValueError(f"out.naux={int(out.naux)} does not match naux={naux}")
            if tuple(out.l_full.shape) != (nops, naux):
                raise ValueError(f"out.l_full has shape {tuple(out.l_full.shape)}, expected {(nops, naux)}")
            if out.l_full.dtype != cp.float64:
                raise ValueError("out.l_full must have dtype float64")
            if not bool(out.l_full.flags.c_contiguous):
                raise ValueError("out.l_full must be C-contiguous")
            if bool(want_eri_mat) != (out.eri_mat is not None):
                raise ValueError("out.eri_mat presence does not match want_eri_mat")
            if bool(want_j_ps) != (out.j_ps is not None):
                raise ValueError("out.j_ps presence does not match want_j_ps")
            if bool(want_pair_norm) != (out.pair_norm is not None):
                raise ValueError("out.pair_norm presence does not match want_pair_norm")

        try:
            from asuka.cueri import df as cueri_df
        except Exception as e:  # pragma: no cover
            raise RuntimeError("cuERI is required for CuERIActiveSpaceDFBuilder") from e

        df_prof: dict | None = {} if profile is not None else None
        t0 = time.perf_counter() if profile is not None else 0.0
        prof = profile.setdefault("cueri_active_df", {}) if profile is not None else None
        cache_hit = False
        cache_populated = False

        def _as_l_full(arr):
            if out is not None:
                cp.copyto(out.l_full, arr)
                return out.l_full
            return cp.ascontiguousarray(arr)

        if cached_b_whitened is not None:
            B_whitened = cp.asarray(cached_b_whitened, dtype=cp.float64)
            if B_whitened.ndim != 3:
                raise ValueError("cached_b_whitened must have shape (nao, nao, naux)")
            if tuple(B_whitened.shape) != (nao, nao, naux):
                raise ValueError(
                    f"cached_b_whitened has shape {tuple(B_whitened.shape)}, expected {(nao, nao, naux)}"
                )
            if not bool(B_whitened.flags.c_contiguous):
                B_whitened = cp.ascontiguousarray(B_whitened)
            l_full = _as_l_full(cueri_df.active_Lfull_from_cached_B_whitened(B_whitened, c_cas))
            cache_hit = True
        elif cache_out is not None:
            try:
                V = cueri_df.metric_2c2e_basis(
                    getattr(self, "_aux_basis"),
                    stream=self.stream,
                    backend=str(self.backend),
                    mode=str(self.mode),
                    threads=int(self.threads),
                )
                L = cueri_df.cholesky_metric(V)
                X = cueri_df.int3c2e_basis(
                    getattr(self, "_ao_basis"),
                    getattr(self, "_aux_basis"),
                    stream=self.stream,
                    backend=str(self.backend),
                    mode=str(self.mode),
                    threads=int(self.threads),
                )
                B_whitened = cueri_df.whiten_3c2e(X, L)
                B_whitened = cp.ascontiguousarray(B_whitened)
                cache_out["cached_b_whitened"] = B_whitened
                l_full = _as_l_full(cueri_df.active_Lfull_from_cached_B_whitened(B_whitened, c_cas))
                cache_populated = True
            except Exception as e:
                cache_out["cache_build_failed"] = str(e)
                l_full = cueri_df.active_Lfull_streamed_basis(
                    getattr(self, "_ao_basis"),
                    getattr(self, "_aux_basis"),
                    c_cas,
                    stream=self.stream,
                    backend=str(self.backend),
                    threads=int(self.threads),
                    mode=str(self.mode),
                    work_small_max=int(self.work_small_max),
                    work_large_min=int(self.work_large_min),
                    blocks_per_task=int(self.blocks_per_task),
                    aux_block_naux=int(self.aux_block_naux),
                    max_tile_bytes=int(self.max_tile_bytes),
                    strategy=str(self.df_strategy),
                    ao_contract_mode=str(self.ao_contract_mode),
                    graph_capture=bool(self.graph_capture),
                    profile=df_prof,
                    out=None if out is None else out.l_full,
                )
                l_full = _as_l_full(l_full)
        else:
            l_full = cueri_df.active_Lfull_streamed_basis(
                getattr(self, "_ao_basis"),
                getattr(self, "_aux_basis"),
                c_cas,
                stream=self.stream,
                backend=str(self.backend),
                threads=int(self.threads),
                mode=str(self.mode),
                work_small_max=int(self.work_small_max),
                work_large_min=int(self.work_large_min),
                blocks_per_task=int(self.blocks_per_task),
                aux_block_naux=int(self.aux_block_naux),
                max_tile_bytes=int(self.max_tile_bytes),
                strategy=str(self.df_strategy),
                ao_contract_mode=str(self.ao_contract_mode),
                graph_capture=bool(self.graph_capture),
                profile=df_prof,
                out=None if out is None else out.l_full,
            )
            l_full = _as_l_full(l_full)

        if profile is not None:
            # Ensure any queued kernels are completed before timing (profile is opt-in).
            cp.cuda.get_current_stream().synchronize()
            prof["t_l_full_s"] = float(time.perf_counter() - t0)
            prof["b_cache_hit"] = bool(cache_hit)
            prof["b_cache_populated"] = bool(cache_populated)
            t0 = time.perf_counter()

        eri_mat = None
        if bool(want_eri_mat):
            if out is not None:
                assert out.eri_mat is not None
                cp.dot(l_full, l_full.T, out=out.eri_mat)
                eri_mat = out.eri_mat
            else:
                eri_mat = cp.ascontiguousarray(l_full @ l_full.T)
            if profile is not None:
                cp.cuda.get_current_stream().synchronize()
                prof["t_eri_mat_s"] = float(time.perf_counter() - t0)
                t0 = time.perf_counter()

        j_ps = None
        if bool(want_j_ps):
            if eri_mat is not None:
                eri4 = eri_mat.reshape(norb, norb, norb, norb)
                j_ps_val = eri4.diagonal(axis1=1, axis2=2).sum(axis=2)
            else:
                l3 = l_full.reshape(norb, norb, naux)
                j_ps_val = cp.einsum("pql,qsl->ps", l3, l3, optimize=True)
            if out is not None:
                assert out.j_ps is not None
                out.j_ps[...] = j_ps_val
                j_ps = out.j_ps
            else:
                j_ps = cp.ascontiguousarray(j_ps_val)
            if profile is not None:
                cp.cuda.get_current_stream().synchronize()
                prof["t_j_ps_s"] = float(time.perf_counter() - t0)
                t0 = time.perf_counter()

        pair_norm = None
        if bool(want_pair_norm):
            pair_norm_val = cp.linalg.norm(l_full, axis=1)
            if out is not None:
                assert out.pair_norm is not None
                out.pair_norm[...] = pair_norm_val
                pair_norm = out.pair_norm
            else:
                pair_norm = cp.ascontiguousarray(pair_norm_val)
            if profile is not None:
                cp.cuda.get_current_stream().synchronize()
                prof["t_pair_norm_s"] = float(time.perf_counter() - t0)

        if profile is not None:
            prof.update(df_prof or {})

        if out is not None:
            return out

        return DeviceActiveDF(
            norb=norb,
            naux=naux,
            l_full=l_full,
            j_ps=j_ps,
            pair_norm=pair_norm,
            eri_mat=eri_mat,
        )
