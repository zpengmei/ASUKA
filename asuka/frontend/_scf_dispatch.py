from __future__ import annotations

from typing import Any, Callable, Mapping


def _maybe_flush_cuda_pool_after_scf() -> None:
    # Optional post-SCF cleanup: reduce driver-visible peak VRAM by releasing
    # cached CUDA workspaces and returning unused CuPy pool blocks.
    try:
        import os as _os_scf  # noqa: PLC0415

        _flag = _os_scf.environ.get("ASUKA_FLUSH_GPU_POOL_AFTER_SCF")
        if _flag is None:
            def _env_true(name: str) -> bool:
                v = _os_scf.environ.get(name)
                if v is None:
                    return False
                return str(v).strip().lower() not in ("0", "false", "no", "off", "")

            _flag = "1" if (_env_true("ASUKA_VRAM_DEBUG") or _env_true("ASUKA_GPU_MEM_CAP_GB")) else "0"
        _v = str(_flag).strip().lower()
        if _v not in ("0", "false", "no", "off", ""):
            import cupy as _cp_scf  # noqa: PLC0415
            from asuka.hf import df_jk as _df_jk  # noqa: PLC0415

            _cp_scf.cuda.Device().synchronize()
            try:
                _df_jk.release_cuda_ext_workspace_cache()
            except Exception:
                pass
            _cp_scf.get_default_memory_pool().free_all_blocks()
            try:
                _cp_scf.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass
    except Exception:
        pass


def run_hf_df_dispatch(
    *,
    mol: Any,
    method: str,
    backend: str,
    df: bool,
    two_e_backend: str | None,
    guess: Any | None,
    dm0: Any | None,
    mo_coeff0: Any | None,
    kwargs: Mapping[str, Any],
    ops: Mapping[str, Callable[..., Any]],
    with_two_e_metadata: Callable[..., Any],
) -> Any:
    """Backend/integral dispatch implementation for frontend.run_hf_df."""

    method_s = str(method).strip().lower()
    backend_s = str(backend).strip().lower()
    if method_s not in {"rhf", "uhf", "rohf", "rks", "uks"}:
        raise ValueError("method must be one of: 'rhf', 'uhf', 'rohf', 'rks', 'uks'")
    if backend_s not in {"cpu", "cuda"}:
        raise ValueError("backend must be 'cpu' or 'cuda'")

    if mo_coeff0 is None and dm0 is None and guess is not None:
        scf_guess = getattr(guess, "scf", guess)
        try:
            mo_coeff0 = getattr(scf_guess, "mo_coeff", None)
        except Exception:
            mo_coeff0 = None

    if two_e_backend is None:
        two_e = "df" if bool(df) else "dense"
    else:
        two_e = str(two_e_backend).strip().lower()
        if two_e not in {"df", "dense", "thc", "direct", "direct_df"}:
            raise ValueError("two_e_backend must be one of: 'df', 'dense', 'thc', 'direct', 'direct_df'")

    kwargs_local = dict(kwargs)

    if method_s in {"rks", "uks"}:
        if backend_s != "cuda":
            raise NotImplementedError("RKS/UKS currently require backend='cuda'")
        if two_e in {"dense", "direct", "direct_df"}:
            raise NotImplementedError("RKS/UKS currently support only two_e_backend='df' or 'thc'")

        dft_kwargs = dict(kwargs_local)
        xc_name = dft_kwargs.pop("functional", "mn15")

        if two_e == "df":
            if method_s == "rks":
                return with_two_e_metadata(
                    ops["run_rks_df"](mol, functional=str(xc_name), dm0=dm0, mo_coeff0=mo_coeff0, **dft_kwargs),
                    two_e_backend="df",
                )
            return with_two_e_metadata(
                ops["run_uks_df"](mol, functional=str(xc_name), dm0=dm0, mo_coeff0=mo_coeff0, **dft_kwargs),
                two_e_backend="df",
            )

        if method_s == "rks":
            return with_two_e_metadata(
                ops["run_rhf_thc"](mol, functional=str(xc_name), dm0=dm0, mo_coeff0=mo_coeff0, **dft_kwargs),
                two_e_backend="thc",
            )
        return with_two_e_metadata(
            ops["run_uhf_thc"](mol, functional=str(xc_name), dm0=dm0, mo_coeff0=mo_coeff0, **dft_kwargs),
            two_e_backend="thc",
        )

    if two_e == "dense":
        dense_kwargs = dict(kwargs_local)
        for key in (
            "auxbasis",
            "df_config",
            "df_int3c_plan_policy",
            "df_int3c_work_small_max",
            "df_int3c_work_large_min",
            "df_int3c_blocks_per_task",
            "df_k_cache_max_mb",
            "df_threads",
            "jk_mode",
            "k_engine",
            "k_q_block",
            "cublas_math_mode",
            "ao_basis",
            "aux_basis",
            "df_backend",
            "df_mode",
            "df_aux_block_naux",
            "L_metric",
        ):
            dense_kwargs.pop(key, None)
        if method_s == "rhf":
            return with_two_e_metadata(
                ops["run_rhf_dense"](mol, backend=backend_s, dm0=dm0, mo_coeff0=mo_coeff0, **dense_kwargs),
                two_e_backend="dense",
            )
        if method_s == "uhf":
            return with_two_e_metadata(
                ops["run_uhf_dense"](mol, backend=backend_s, dm0=dm0, mo_coeff0=mo_coeff0, **dense_kwargs),
                two_e_backend="dense",
            )
        return with_two_e_metadata(
            ops["run_rohf_dense"](mol, backend=backend_s, dm0=dm0, mo_coeff0=mo_coeff0, **dense_kwargs),
            two_e_backend="dense",
        )

    if two_e == "direct":
        if backend_s != "cuda":
            raise NotImplementedError("Direct integral backend currently requires backend='cuda'")
        direct_kwargs = dict(kwargs_local)
        for key in (
            "auxbasis",
            "df_config",
            "df_int3c_plan_policy",
            "df_int3c_work_small_max",
            "df_int3c_work_large_min",
            "df_int3c_blocks_per_task",
            "df_k_cache_max_mb",
            "df_threads",
            "jk_mode",
            "k_engine",
            "k_q_block",
            "cublas_math_mode",
            "ao_basis",
            "aux_basis",
            "df_backend",
            "df_mode",
            "df_aux_block_naux",
            "L_metric",
            "dense_threads",
            "dense_max_tile_bytes",
            "dense_eps_ao",
            "dense_max_l",
            "dense_mem_budget_gib",
        ):
            direct_kwargs.pop(key, None)
        if method_s == "rhf":
            out = ops["run_rhf_direct"](mol, dm0=dm0, mo_coeff0=mo_coeff0, **direct_kwargs)
            return with_two_e_metadata(out, two_e_backend="direct", direct_jk_ctx=getattr(out, "direct_jk_ctx", None))
        if method_s == "uhf":
            out = ops["run_uhf_direct"](mol, dm0=dm0, mo_coeff0=mo_coeff0, **direct_kwargs)
            return with_two_e_metadata(out, two_e_backend="direct", direct_jk_ctx=getattr(out, "direct_jk_ctx", None))
        out = ops["run_rohf_direct"](mol, dm0=dm0, mo_coeff0=mo_coeff0, **direct_kwargs)
        return with_two_e_metadata(out, two_e_backend="direct", direct_jk_ctx=getattr(out, "direct_jk_ctx", None))

    if two_e == "direct_df":
        if backend_s != "cuda":
            raise NotImplementedError("Direct DF backend currently requires backend='cuda'")
        direct_df_kwargs = dict(kwargs_local)
        for key in (
            "df_layout",
            "jk_mode",
            "k_engine",
            "ao_basis",
            "aux_basis",
            "dense_threads",
            "dense_max_tile_bytes",
            "dense_eps_ao",
            "dense_max_l",
            "dense_mem_budget_gib",
            "eps_schwarz",
            "direct_threads",
            "direct_max_tile_bytes",
        ):
            direct_df_kwargs.pop(key, None)
        if method_s == "rhf":
            out = ops["run_rhf_direct_df"](mol, dm0=dm0, mo_coeff0=mo_coeff0, **direct_df_kwargs)
            return with_two_e_metadata(out, two_e_backend="direct_df")
        if method_s == "uhf":
            out = ops["run_uhf_direct_df"](mol, dm0=dm0, mo_coeff0=mo_coeff0, **direct_df_kwargs)
            return with_two_e_metadata(out, two_e_backend="direct_df")
        out = ops["run_rohf_direct_df"](mol, dm0=dm0, mo_coeff0=mo_coeff0, **direct_df_kwargs)
        return with_two_e_metadata(out, two_e_backend="direct_df")

    if two_e == "thc":
        if backend_s != "cuda":
            raise NotImplementedError("THC backend currently requires backend='cuda'")
        if method_s == "rhf":
            return with_two_e_metadata(
                ops["run_rhf_thc"](mol, dm0=dm0, mo_coeff0=mo_coeff0, **kwargs_local),
                two_e_backend="thc",
            )
        if method_s == "uhf":
            return with_two_e_metadata(
                ops["run_uhf_thc"](mol, dm0=dm0, mo_coeff0=mo_coeff0, **kwargs_local),
                two_e_backend="thc",
            )
        return with_two_e_metadata(
            ops["run_rohf_thc"](mol, dm0=dm0, mo_coeff0=mo_coeff0, **kwargs_local),
            two_e_backend="thc",
        )

    if backend_s == "cuda":
        if method_s == "rhf":
            out = ops["run_rhf_df"](mol, dm0=dm0, mo_coeff0=mo_coeff0, **kwargs_local)
        elif method_s == "uhf":
            out = ops["run_uhf_df"](mol, dm0=dm0, mo_coeff0=mo_coeff0, **kwargs_local)
        else:
            out = ops["run_rohf_df"](mol, dm0=dm0, mo_coeff0=mo_coeff0, **kwargs_local)

        _maybe_flush_cuda_pool_after_scf()
        return with_two_e_metadata(out, two_e_backend="df")

    cpu_kwargs = dict(kwargs_local)
    for key in (
        "df_int3c_plan_policy",
        "df_int3c_work_small_max",
        "df_int3c_work_large_min",
        "df_int3c_blocks_per_task",
        "df_k_cache_max_mb",
    ):
        cpu_kwargs.pop(key, None)
    if method_s == "rhf":
        return with_two_e_metadata(ops["run_rhf_df_cpu"](mol, dm0=dm0, mo_coeff0=mo_coeff0, **cpu_kwargs), two_e_backend="df")
    if method_s == "uhf":
        return with_two_e_metadata(ops["run_uhf_df_cpu"](mol, dm0=dm0, mo_coeff0=mo_coeff0, **cpu_kwargs), two_e_backend="df")
    return with_two_e_metadata(ops["run_rohf_df_cpu"](mol, dm0=dm0, mo_coeff0=mo_coeff0, **cpu_kwargs), two_e_backend="df")
