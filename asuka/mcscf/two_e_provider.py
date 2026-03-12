from __future__ import annotations

"""Two-electron provider abstraction for MCSCF backends.

The provider layer centralizes J/K and mixed-index integral dispatch so higher
level MCSCF code does not need backend-specific branching.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


def _probe_array_from_provider(provider: Any) -> Any | None:
    probe = getattr(provider, "probe_array", None)
    if callable(probe):
        try:
            return probe()
        except Exception:
            return None
    return None


@dataclass
class DFProvider:
    df_B: Any

    kind: str = "df"
    backend: str = "cpu"

    def __post_init__(self) -> None:
        from asuka.hf import df_scf as _df_scf  # noqa: PLC0415

        _xp, is_gpu = _df_scf._get_xp(self.df_B)  # noqa: SLF001
        self.backend = "cuda" if bool(is_gpu) else "cpu"

    def probe_array(self) -> Any:
        return self.df_B

    def dense_builder(self) -> Any | None:
        return None

    def jk(
        self,
        D: Any,
        *,
        want_J: bool = True,
        want_K: bool = True,
        profile: dict | None = None,
    ) -> tuple[Any | None, Any | None]:
        from asuka.hf import df_scf as _df_scf  # noqa: PLC0415

        _ = profile
        return _df_scf._df_JK(self.df_B, D, want_J=bool(want_J), want_K=bool(want_K))  # noqa: SLF001

    def jk_multi2(
        self,
        Da: Any,
        Db: Any,
        *,
        want_J: bool = True,
        want_K: bool = True,
        profile: dict | None = None,
    ) -> tuple[Any | None, Any | None, Any | None, Any | None]:
        Ja, Ka = self.jk(Da, want_J=want_J, want_K=want_K, profile=profile)
        Jb, Kb = self.jk(Db, want_J=want_J, want_K=want_K, profile=profile)
        return Ja, Ka, Jb, Kb


@dataclass
class DenseERIProvider:
    ao_eri: Any

    kind: str = "dense"
    backend: str = "cpu"

    def __post_init__(self) -> None:
        from asuka.hf import df_scf as _df_scf  # noqa: PLC0415

        _xp, is_gpu = _df_scf._get_xp(self.ao_eri)  # noqa: SLF001
        self.backend = "cuda" if bool(is_gpu) else "cpu"

    def probe_array(self) -> Any:
        return self.ao_eri

    def dense_builder(self) -> Any | None:
        return None

    def jk(
        self,
        D: Any,
        *,
        want_J: bool = True,
        want_K: bool = True,
        profile: dict | None = None,
    ) -> tuple[Any | None, Any | None]:
        from asuka.hf.dense_jk import dense_JK_from_eri_mat_D  # noqa: PLC0415

        _ = profile
        return dense_JK_from_eri_mat_D(self.ao_eri, D, want_J=bool(want_J), want_K=bool(want_K))

    def jk_multi2(
        self,
        Da: Any,
        Db: Any,
        *,
        want_J: bool = True,
        want_K: bool = True,
        profile: dict | None = None,
    ) -> tuple[Any | None, Any | None, Any | None, Any | None]:
        Ja, Ka = self.jk(Da, want_J=want_J, want_K=want_K, profile=profile)
        Jb, Kb = self.jk(Db, want_J=want_J, want_K=want_K, profile=profile)
        return Ja, Ka, Jb, Kb


@dataclass
class DirectHybridProvider:
    scf_out: Any
    _builder: Any | None = None
    _builder_mol: Any | None = None
    _threads: int = 256
    _max_tile_bytes: int = 256 << 20
    _eps_ao: float = 0.0
    _ao_rep: str = "auto"

    kind: str = "direct_hybrid"
    backend: str = "cuda"

    def __post_init__(self) -> None:
        if getattr(self.scf_out, "direct_jk_ctx", None) is None:
            raise ValueError("DirectHybridProvider requires scf_out.direct_jk_ctx")

    @property
    def ctx(self) -> Any:
        return getattr(self.scf_out, "direct_jk_ctx")

    def probe_array(self) -> Any:
        probe = getattr(self.ctx, "sp_A_dev", None)
        if probe is not None:
            return probe
        return getattr(self.scf_out, "df_B", None)

    def dense_builder(self) -> Any:
        if self._builder is None:
            from asuka.cueri.active_space_dense_gpu import CuERIActiveSpaceDenseGPUBuilder  # noqa: PLC0415

            ao_basis = getattr(self.scf_out, "ao_basis", None)
            if ao_basis is None:
                raise ValueError("direct-hybrid provider requires scf_out.ao_basis")
            mol = self._builder_mol if self._builder_mol is not None else getattr(self.scf_out, "mol", None)
            self._builder = CuERIActiveSpaceDenseGPUBuilder(
                mol=mol,
                ao_basis=ao_basis,
                ao_rep=str(self._ao_rep),
                threads=int(self._threads),
                max_tile_bytes=int(self._max_tile_bytes),
                eps_ao=float(self._eps_ao),
            )
        return self._builder

    def jk(
        self,
        D: Any,
        *,
        want_J: bool = True,
        want_K: bool = True,
        profile: dict | None = None,
    ) -> tuple[Any | None, Any | None]:
        from asuka.hf.direct_jk import direct_JK  # noqa: PLC0415

        return direct_JK(self.ctx, D, want_J=bool(want_J), want_K=bool(want_K), profile=profile)

    def jk_multi2(
        self,
        Da: Any,
        Db: Any,
        *,
        want_J: bool = True,
        want_K: bool = True,
        profile: dict | None = None,
    ) -> tuple[Any | None, Any | None, Any | None, Any | None]:
        from asuka.hf.direct_jk import direct_JK_multi  # noqa: PLC0415

        return direct_JK_multi(
            self.ctx,
            Da,
            Db,
            want_J=bool(want_J),
            want_K=bool(want_K),
            profile=profile,
        )

    def build_pq_uv(self, C_mo: Any, C_act: Any, *, out: Any | None = None, profile: dict | None = None) -> Any:
        return self.dense_builder().build_pq_uv_eri_mat(C_mo, C_act, out=out, profile=profile)

    def build_pu_qv(self, C_mo: Any, C_act: Any, *, out: Any | None = None, profile: dict | None = None) -> Any:
        return self.dense_builder().build_pu_qv_eri_mat(C_mo, C_act, out=out, profile=profile)

    def build_pu_wx(self, C_mo: Any, C_act: Any, *, out: Any | None = None, profile: dict | None = None) -> Any:
        return self.dense_builder().build_pu_wx_eri_mat(C_mo, C_act, out=out, profile=profile)

    def contract_pu_wx_dm2(
        self,
        C_mo: Any,
        C_act: Any,
        dm2_wxuv: Any,
        *,
        out: Any | None = None,
        profile: dict | None = None,
    ) -> Any:
        return self.dense_builder().contract_pu_wx_dm2(
            C_mo,
            C_act,
            dm2_wxuv,
            out=out,
            profile=profile,
        )


@dataclass
class StreamedDFHybridProvider:
    scf_out: Any
    _builder: Any | None = None
    _builder_mol: Any | None = None
    _threads: int = 256
    _max_tile_bytes: int = 256 << 20
    _eps_ao: float = 0.0
    _ao_rep: str = "auto"
    _df_mode: str = "auto"
    _df_aux_block_naux: int = 256

    kind: str = "streamed_df_hybrid"
    backend: str = "cuda"
    _ctx: Any | None = None

    def __post_init__(self) -> None:
        if getattr(self.scf_out, "ao_basis", None) is None or getattr(self.scf_out, "aux_basis", None) is None:
            raise ValueError("StreamedDFHybridProvider requires scf_out.ao_basis and scf_out.aux_basis")
        if getattr(self.scf_out, "df_L", None) is None:
            raise ValueError("StreamedDFHybridProvider requires scf_out.df_L")

    def probe_array(self) -> Any:
        return getattr(self.ctx, "L_metric", None)

    @property
    def ctx(self) -> Any:
        if self._ctx is None:
            from asuka.hf.df_jk_streamed import make_streamed_df_jk_context  # noqa: PLC0415

            df_run_config = getattr(self.scf_out, "df_run_config", None)
            backend = getattr(df_run_config, "backend", None)
            backend_s = str(backend or "gpu_rys").strip().lower()
            if backend_s in {"cuda", "gpu", "cupy"}:
                backend_s = "gpu_rys"
            elif backend_s not in {"gpu_ss", "gpu_sp", "gpu_rys"}:
                backend_s = "gpu_rys"
            ao_rep = str(self._ao_rep).strip().lower()
            if ao_rep == "auto":
                ao_rep = "cart" if bool(getattr(getattr(self.scf_out, "mol", None), "cart", False)) else "sph"
            aux_block_naux = int(self._df_aux_block_naux)
            try:
                prof_stream = getattr(self.scf_out, "profile", None)
                if isinstance(prof_stream, dict):
                    aux_block_naux = int(prof_stream.get("jk_streamed", {}).get("aux_block_naux", aux_block_naux))
            except Exception:
                pass
            self._ctx = make_streamed_df_jk_context(
                getattr(self.scf_out, "ao_basis"),
                getattr(self.scf_out, "aux_basis"),
                L_metric=getattr(self.scf_out, "df_L"),
                backend=str(backend_s),
                threads=int(self._threads),
                mode=str(self._df_mode),
                ao_rep=str(ao_rep),
                aux_block_naux=int(aux_block_naux),
                profile=None,
            )
        return self._ctx

    def dense_builder(self) -> Any:
        if self._builder is None:
            from asuka.cueri.active_space_dense_gpu import CuERIActiveSpaceDenseGPUBuilder  # noqa: PLC0415

            ao_basis = getattr(self.scf_out, "ao_basis", None)
            if ao_basis is None:
                raise ValueError("streamed-df provider requires scf_out.ao_basis")
            mol = self._builder_mol if self._builder_mol is not None else getattr(self.scf_out, "mol", None)
            self._builder = CuERIActiveSpaceDenseGPUBuilder(
                mol=mol,
                ao_basis=ao_basis,
                ao_rep=str(self._ao_rep),
                threads=int(self._threads),
                max_tile_bytes=int(self._max_tile_bytes),
                eps_ao=float(self._eps_ao),
            )
        return self._builder

    @staticmethod
    def _density_to_occ(D: Any) -> tuple[Any, Any]:
        from asuka.hf import df_scf as _df_scf  # noqa: PLC0415

        xp, _is_gpu = _df_scf._get_xp(D)  # noqa: SLF001
        D_xp = xp.asarray(D, dtype=xp.float64)
        D_xp = 0.5 * (D_xp + D_xp.T)
        evals, evecs = xp.linalg.eigh(D_xp)
        keep = evals > 1e-10
        if int(getattr(keep, "sum", lambda: np.sum(np.asarray(keep)))()) == 0:
            return xp.zeros((int(D_xp.shape[0]), 0), dtype=xp.float64), xp.zeros((0,), dtype=xp.float64)
        occ = evals[keep]
        coeff = evecs[:, keep]
        order = xp.argsort(occ)[::-1]
        occ = xp.asarray(occ[order], dtype=xp.float64)
        coeff = xp.asarray(coeff[:, order], dtype=xp.float64)
        return coeff, occ

    def jk(
        self,
        D: Any,
        *,
        want_J: bool = True,
        want_K: bool = True,
        profile: dict | None = None,
    ) -> tuple[Any | None, Any | None]:
        from asuka.hf.df_jk_streamed import df_JK_streamed  # noqa: PLC0415

        C_occ, occ_vals = self._density_to_occ(D)
        J, K = df_JK_streamed(
            self.ctx,
            D,
            C_occ,
            occ_vals,
            k_q_block=int(getattr(getattr(self.scf_out, "df_run_config", None), "k_q_block", 128)),
            cublas_math_mode=getattr(getattr(self.scf_out, "df_run_config", None), "cublas_math_mode", None),
            work=None,
            profile=profile,
        )
        if not want_J:
            J = None
        if not want_K:
            K = None
        return J, K

    def jk_multi2(
        self,
        Da: Any,
        Db: Any,
        *,
        want_J: bool = True,
        want_K: bool = True,
        profile: dict | None = None,
    ) -> tuple[Any | None, Any | None, Any | None, Any | None]:
        Ja, Ka = self.jk(Da, want_J=want_J, want_K=want_K, profile=profile)
        Jb, Kb = self.jk(Db, want_J=want_J, want_K=want_K, profile=profile)
        return Ja, Ka, Jb, Kb

    def build_pq_uv(self, C_mo: Any, C_act: Any, *, out: Any | None = None, profile: dict | None = None) -> Any:
        return self.dense_builder().build_pq_uv_eri_mat(C_mo, C_act, out=out, profile=profile)

    def build_pu_qv(self, C_mo: Any, C_act: Any, *, out: Any | None = None, profile: dict | None = None) -> Any:
        return self.dense_builder().build_pu_qv_eri_mat(C_mo, C_act, out=out, profile=profile)

    def build_pu_wx(self, C_mo: Any, C_act: Any, *, out: Any | None = None, profile: dict | None = None) -> Any:
        return self.dense_builder().build_pu_wx_eri_mat(C_mo, C_act, out=out, profile=profile)

    def contract_pu_wx_dm2(
        self,
        C_mo: Any,
        C_act: Any,
        dm2_wxuv: Any,
        *,
        out: Any | None = None,
        profile: dict | None = None,
    ) -> Any:
        return self.dense_builder().contract_pu_wx_dm2(
            C_mo,
            C_act,
            dm2_wxuv,
            out=out,
            profile=profile,
        )


def resolve_two_e_provider(
    scf_out: Any,
    *,
    dense_gpu_builder: Any | None = None,
    dense_gpu_builder_mol: Any | None = None,
    dense_gpu_threads: int = 256,
    dense_max_tile_bytes: int = 256 << 20,
    dense_eps_ao: float = 0.0,
    dense_gpu_ao_rep: str = "auto",
) -> Any | None:
    """Resolve the best available two-electron provider from an SCF output."""

    two_e_backend = str(getattr(scf_out, "two_e_backend", "") or "").strip().lower()
    direct_ctx = getattr(scf_out, "direct_jk_ctx", None)

    if direct_ctx is not None or two_e_backend == "direct":
        if direct_ctx is None:
            raise ValueError("two_e_backend='direct' requires scf_out.direct_jk_ctx")
        return DirectHybridProvider(
            scf_out=scf_out,
            _builder=dense_gpu_builder,
            _builder_mol=dense_gpu_builder_mol,
            _threads=int(dense_gpu_threads),
            _max_tile_bytes=int(dense_max_tile_bytes),
            _eps_ao=float(dense_eps_ao),
            _ao_rep=str(dense_gpu_ao_rep),
        )

    if two_e_backend == "direct_df":
        return StreamedDFHybridProvider(
            scf_out=scf_out,
            _builder=dense_gpu_builder,
            _builder_mol=dense_gpu_builder_mol,
            _threads=int(dense_gpu_threads),
            _max_tile_bytes=int(dense_max_tile_bytes),
            _eps_ao=float(dense_eps_ao),
            _ao_rep=str(dense_gpu_ao_rep),
        )

    df_B = getattr(scf_out, "df_B", None)
    if df_B is not None:
        return DFProvider(df_B=df_B)

    ao_eri = getattr(scf_out, "ao_eri", None)
    if ao_eri is not None:
        return DenseERIProvider(ao_eri=ao_eri)

    return None


__all__ = [
    "DFProvider",
    "DenseERIProvider",
    "DirectHybridProvider",
    "StreamedDFHybridProvider",
    "resolve_two_e_provider",
    "_probe_array_from_provider",
]
