from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass
class AutoTuneTrial:
    """Single autotune trial result."""

    index: int
    config: dict[str, Any]
    wall_s: float | None
    casci_s: float | None
    hop_total_s: float | None
    hop_time_s: float | None
    hop_calls: float | None
    csr_build_s: float | None
    offdiag_gemm_s: float | None
    offdiag_apply_s: float | None
    csr_prefilter_s: float | None
    score: float
    metric_used: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AutoTuneResult:
    """Autotune summary and recommendation."""

    metric: str
    baseline: AutoTuneTrial
    best: AutoTuneTrial
    trials: list[AutoTuneTrial]
    recommended_solver_kwargs: dict[str, Any]
    ncsf_hint: int | None
    gpu_info: dict[str, Any] | None
    gpu_autofill_applied: dict[str, Any]
    gpu_profile_preset: str | None
    warm_start_fallback: bool
    search_stats: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "metric": self.metric,
            "baseline": self.baseline.to_dict(),
            "best": self.best.to_dict(),
            "trials": [t.to_dict() for t in self.trials],
            "recommended_solver_kwargs": dict(self.recommended_solver_kwargs),
            "ncsf_hint": self.ncsf_hint,
            "gpu_info": dict(self.gpu_info) if isinstance(self.gpu_info, dict) else None,
            "gpu_autofill_applied": dict(self.gpu_autofill_applied),
            "gpu_profile_preset": self.gpu_profile_preset,
            "warm_start_fallback": bool(self.warm_start_fallback),
        }
        if self.search_stats is not None:
            d["search_stats"] = dict(self.search_stats)
        return d


def _freeze_cfg(cfg: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    items: list[tuple[str, Any]] = []
    for k in sorted(cfg):
        v = cfg[k]
        if isinstance(v, float):
            v = round(float(v), 12)
        items.append((str(k), v))
    return tuple(items)


def _sum_hop_metric(hop: Any, key: str) -> float | None:
    if isinstance(hop, Mapping):
        v = hop.get(key)
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            return float(v)
        total = 0.0
        found = False
        for subkey in ("low_precision", "full_precision"):
            sub = hop.get(subkey)
            if isinstance(sub, Mapping):
                sv = sub.get(key)
                if isinstance(sv, (int, float)) and math.isfinite(float(sv)):
                    total += float(sv)
                    found = True
        if found:
            return total
    return None


def _extract_metrics(profile: Mapping[str, Any]) -> dict[str, float | None]:
    stats = profile.get("davidson_stats") if isinstance(profile.get("davidson_stats"), Mapping) else {}
    hop = profile.get("matvec_cuda_hop_profile") if isinstance(profile.get("matvec_cuda_hop_profile"), Mapping) else {}
    hop_calls = stats.get("hop_calls")
    out = {
        "hop_total_s": _sum_hop_metric(hop, "total_s"),
        "hop_time_s": float(stats["hop_time_s"]) if isinstance(stats.get("hop_time_s"), (int, float)) else None,
        "hop_calls": float(hop_calls) if isinstance(hop_calls, (int, float)) else None,
        "csr_build_s": _sum_hop_metric(hop, "csr_build_s"),
        "offdiag_gemm_s": _sum_hop_metric(hop, "offdiag_gemm_s"),
        "offdiag_apply_s": _sum_hop_metric(hop, "offdiag_apply_s"),
        "csr_prefilter_s": _sum_hop_metric(hop, "csr_prefilter_s"),
    }
    return out


def _score_from_metrics(metric: str, metrics: Mapping[str, float | None], wall_s: float | None) -> tuple[float, str]:
    order = [metric, "hop_total_s", "hop_time_s"]
    if metric != "casci_s":
        order.append("casci_s")
    if metric != "wall_s":
        order.append("wall_s")
    seen: set[str] = set()
    for key in order:
        if key in seen:
            continue
        seen.add(key)
        if key == "wall_s":
            if wall_s is not None and math.isfinite(float(wall_s)):
                return float(wall_s), "wall_s"
            continue
        v = metrics.get(key)
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            return float(v), str(key)
    return float("inf"), "inf"


def _ncsf_hint(solver: Any, norb: int, nelec: int | tuple[int, int]) -> int | None:
    try:
        neleca, nelecb, nelec_total, _ = solver._normalize_nelec(nelec)
        twos = solver._get_twos_target(neleca, nelecb)
        drt = solver._get_drt(
            int(norb),
            int(nelec_total),
            int(twos),
            orbsym=getattr(solver, "orbsym", None),
            wfnsym=getattr(solver, "wfnsym", None),
            ne_constraints=getattr(solver, "ne_constraints", None),
        )
        return int(getattr(drt, "ncsf"))
    except Exception:
        return None


def _dict_get_any(d: Mapping[str, Any], key: str) -> Any:
    if key in d:
        return d[key]
    bkey = key.encode("utf-8")
    if bkey in d:
        return d[bkey]
    return None


NOISE_MARGIN = 0.03  # 3% noise margin for winner selection
EARLY_STOP_REL_THRESHOLD = 0.02  # 2% improvement threshold for early stopping


def _get_mem_gib(gpu_info: Mapping[str, Any] | None) -> float | None:
    """Extract total GPU memory in GiB from gpu_info dict."""
    if not isinstance(gpu_info, Mapping):
        return None
    v = gpu_info.get("total_mem_gib")
    if isinstance(v, (int, float)) and float(v) > 0.0:
        return float(v)
    return None


def _adaptive_j_tile_candidates(
    ncsf: int | None, gpu_info: Mapping[str, Any] | None
) -> list[int]:
    """Generate j_tile candidates adapted to problem size and GPU memory."""
    all_vals = [512, 1024, 1536, 2048, 3072, 4096]
    if ncsf is not None:
        all_vals = [v for v in all_vals if v < ncsf]
        if ncsf < 50_000:
            all_vals = [v for v in all_vals if v <= 1024]
    mem_gib = _get_mem_gib(gpu_info)
    if mem_gib is not None and mem_gib <= 12.0:
        all_vals = [v for v in all_vals if v <= 2048]
    return all_vals or [min(ncsf or 1024, 1024)]


def _adaptive_max_g_candidates(gpu_info: Mapping[str, Any] | None) -> list[int]:
    """Generate max_g_mib candidates adapted to GPU memory."""
    all_vals = [128, 256, 512, 1024]
    mem_gib = _get_mem_gib(gpu_info)
    if mem_gib is not None:
        if mem_gib <= 8.0:
            all_vals = [v for v in all_vals if v <= 128]
        elif mem_gib <= 12.0:
            all_vals = [v for v in all_vals if v <= 256]
        elif mem_gib <= 16.0:
            all_vals = [v for v in all_vals if v <= 512]
    return all_vals


def _adaptive_csr_host_cache_candidates(ncsf: int | None) -> list[dict[str, Any]] | None:
    """Generate csr_host_cache candidates; returns None if not applicable."""
    if ncsf is not None and ncsf >= 1_000_000:
        return [
            {"matvec_cuda_csr_host_cache": "off"},
            {"matvec_cuda_csr_host_cache": "auto"},
        ]
    return None


def detect_cuda_device_info() -> dict[str, Any] | None:
    """Best-effort CUDA device query via CuPy runtime."""

    try:
        import cupy as cp  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        dev_id = int(cp.cuda.runtime.getDevice())
        props = cp.cuda.runtime.getDeviceProperties(dev_id)
    except Exception:
        return None

    if not isinstance(props, Mapping):
        return None

    name = _dict_get_any(props, "name")
    if isinstance(name, (bytes, bytearray)):
        try:
            name = bytes(name).decode("utf-8", errors="ignore").strip()
        except Exception:
            name = str(name)
    else:
        name = str(name) if name is not None else ""

    total_mem_b = _dict_get_any(props, "totalGlobalMem")
    sm_count = _dict_get_any(props, "multiProcessorCount")
    major = _dict_get_any(props, "major")
    minor = _dict_get_any(props, "minor")
    warp = _dict_get_any(props, "warpSize")

    try:
        total_mem_b_i = int(total_mem_b)
    except Exception:
        total_mem_b_i = 0
    try:
        sm_i = int(sm_count)
    except Exception:
        sm_i = 0
    try:
        major_i = int(major)
    except Exception:
        major_i = -1
    try:
        minor_i = int(minor)
    except Exception:
        minor_i = -1
    try:
        warp_i = int(warp)
    except Exception:
        warp_i = 32

    out = {
        "device_id": int(dev_id),
        "name": str(name),
        "total_mem_bytes": int(total_mem_b_i),
        "total_mem_gib": float(total_mem_b_i) / float(1024**3) if total_mem_b_i > 0 else None,
        "sm_count": int(sm_i),
        "compute_capability": f"{major_i}.{minor_i}" if major_i >= 0 and minor_i >= 0 else None,
        "warp_size": int(warp_i),
    }
    return out


def _gpu_autofill_overrides(
    base_cfg: Mapping[str, Any],
    *,
    ncsf: int | None,
    gpu_info: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Heuristic CUDA config autofill from device properties."""

    if not isinstance(gpu_info, Mapping):
        return {}

    total_mem_gib_v = gpu_info.get("total_mem_gib")
    total_mem_gib: float | None
    if isinstance(total_mem_gib_v, (int, float)) and float(total_mem_gib_v) > 0.0:
        total_mem_gib = float(total_mem_gib_v)
    else:
        total_mem_gib = None

    sm_v = gpu_info.get("sm_count")
    sm_count = int(sm_v) if isinstance(sm_v, (int, float)) else 0
    use_epq = bool(base_cfg.get("matvec_cuda_use_epq_table", False))
    agg = bool(base_cfg.get("matvec_cuda_aggregate_offdiag", True))

    out: dict[str, Any] = {}

    # Keep a safety margin from full VRAM to reduce OOM risk and allocator churn.
    if total_mem_gib is not None:
        reserve_gib = 1.5 if total_mem_gib >= 20.0 else (1.0 if total_mem_gib >= 12.0 else 0.75)
        hard_cap = min(total_mem_gib * 0.90, total_mem_gib - reserve_gib)
        hard_cap = max(2.0, hard_cap)
        out["matvec_cuda_mem_hard_cap_gib"] = float(round(hard_cap, 3))

    if total_mem_gib is not None:
        if total_mem_gib <= 8.0:
            out["matvec_cuda_max_g_mib"] = 96.0
        elif total_mem_gib <= 12.0:
            out["matvec_cuda_max_g_mib"] = 128.0
        elif total_mem_gib <= 16.0:
            out["matvec_cuda_max_g_mib"] = 192.0
        elif total_mem_gib <= 24.0:
            out["matvec_cuda_max_g_mib"] = 256.0
        else:
            out["matvec_cuda_max_g_mib"] = 384.0

    if ncsf is not None and ncsf > 0:
        if int(ncsf) >= 1_000_000:
            if total_mem_gib is not None and total_mem_gib >= 20.0:
                out["matvec_cuda_j_tile"] = 2048
            elif total_mem_gib is not None and total_mem_gib >= 14.0:
                out["matvec_cuda_j_tile"] = 1536
            else:
                out["matvec_cuda_j_tile"] = 1024
        else:
            out["matvec_cuda_j_tile"] = int(min(int(ncsf), 1024))

    if sm_count >= 64:
        threads_g = 256
    elif sm_count >= 40:
        threads_g = 192
    else:
        threads_g = 128
    out["matvec_cuda_threads_g"] = int(threads_g)
    out["matvec_cuda_threads_w"] = int(threads_g)

    if ncsf is not None and int(ncsf) >= 1_000_000:
        out["matvec_cuda_threads_apply"] = 64
    else:
        out["matvec_cuda_threads_apply"] = 128 if sm_count >= 64 else 64

    if (not use_epq) and agg and ncsf is not None and int(ncsf) >= 1_000_000:
        out["matvec_cuda_prefilter_trivial_tasks"] = "off"
    else:
        out["matvec_cuda_prefilter_trivial_tasks"] = "auto"

    # Keep host cache conservative by default on smaller-memory devices.
    if total_mem_gib is not None and total_mem_gib <= 12.5:
        out["matvec_cuda_csr_host_cache"] = "off"
        out["matvec_cuda_csr_host_cache_budget_gib"] = 0.0

    # Mixed-precision: enable TF32 GEMM on Ampere+ (cc >= 8.0) for the AH
    # orbital optimizer's contract_2e calls. The CASCI energy step stays FP64.
    cc_str = gpu_info.get("compute_capability")
    if isinstance(cc_str, str) and "." in cc_str:
        try:
            cc_major = int(cc_str.split(".")[0])
        except Exception:
            cc_major = 0
    else:
        cc_major = 0
    if cc_major >= 8:
        out["matvec_cuda_gemm_backend"] = "gemmex_tf32"
        out["matvec_cuda_fp32_coeff_data"] = True

    return out


def list_gpu_profile_presets() -> tuple[str, ...]:
    """Return supported fixed GPU profile preset names."""

    return ("ada_12gb_laptop", "ada_24gb_desktop")


def _gpu_profile_preset_overrides(
    preset: str | None,
    *,
    base_cfg: Mapping[str, Any],
    ncsf: int | None,
) -> dict[str, Any]:
    """Return static tuning overrides for common GPU classes."""

    if preset is None:
        return {}
    p = str(preset).strip().lower()
    if p == "":
        return {}
    use_epq = bool(base_cfg.get("matvec_cuda_use_epq_table", False))
    agg = bool(base_cfg.get("matvec_cuda_aggregate_offdiag", True))
    out: dict[str, Any] = {}

    if p == "ada_12gb_laptop":
        out.update(
            {
                "matvec_cuda_mem_hard_cap_gib": 9.5,
                "matvec_cuda_max_g_mib": 128.0,
                "matvec_cuda_j_tile": 1024,
                "matvec_cuda_threads_g": 256,
                "matvec_cuda_threads_w": 256,
                "matvec_cuda_threads_apply": 64,
                "matvec_cuda_gemm_backend": "gemmex_tf32",
                "matvec_cuda_fp32_coeff_data": True,
                "matvec_cuda_csr_host_cache": "off",
                "matvec_cuda_csr_host_cache_budget_gib": 0.0,
                "matvec_cuda_csr_host_cache_min_ncsf": 1_000_000,
            }
        )
        if (not use_epq) and agg and ncsf is not None and int(ncsf) >= 1_000_000:
            out["matvec_cuda_prefilter_trivial_tasks"] = "off"
        else:
            out["matvec_cuda_prefilter_trivial_tasks"] = "auto"
        return out

    if p == "ada_24gb_desktop":
        out.update(
            {
                "matvec_cuda_mem_hard_cap_gib": 20.5,
                "matvec_cuda_max_g_mib": 256.0,
                "matvec_cuda_j_tile": 2048 if (ncsf is None or int(ncsf) >= 1_000_000) else 1024,
                "matvec_cuda_threads_g": 256,
                "matvec_cuda_threads_w": 256,
                "matvec_cuda_threads_apply": 64,
                "matvec_cuda_gemm_backend": "gemmex_tf32",
                "matvec_cuda_fp32_coeff_data": True,
                "matvec_cuda_csr_host_cache": "auto",
                "matvec_cuda_csr_host_cache_budget_gib": 4.0,
                "matvec_cuda_csr_host_cache_min_ncsf": 1_000_000,
            }
        )
        if (not use_epq) and agg and ncsf is not None and int(ncsf) >= 1_000_000:
            out["matvec_cuda_prefilter_trivial_tasks"] = "off"
        else:
            out["matvec_cuda_prefilter_trivial_tasks"] = "auto"
        return out

    raise ValueError(
        f"unknown gpu_profile_preset={preset!r}; supported: {', '.join(list_gpu_profile_presets())}"
    )


def _make_ci0(ncsf: int, seed: int) -> np.ndarray:
    rs = np.random.default_rng(int(seed))
    vec = rs.standard_normal(int(ncsf), dtype=np.float64)
    nrm = float(np.linalg.norm(vec))
    if nrm > 0.0:
        vec /= nrm
    return vec


def _default_base_kwargs(solver: Any, *, max_cycle: int, max_space: int | None) -> dict[str, Any]:
    backend = str(getattr(solver, "matvec_backend", "cuda_eri_mat")).strip().lower()
    if backend not in ("cuda_eri_mat", "cuda"):
        raise ValueError(
            f"autotune currently supports CUDA ERI-mat backend only; got matvec_backend={backend!r}"
        )
    threads_g = int(getattr(solver, "matvec_cuda_threads_g", 256))
    threads_w = int(getattr(solver, "matvec_cuda_threads_w", 0))
    if threads_w <= 0:
        threads_w = int(threads_g)
    threads_apply = int(getattr(solver, "matvec_cuda_threads_apply", 0))
    if threads_apply <= 0:
        threads_apply = 64
    j_tile = int(getattr(solver, "matvec_cuda_j_tile", 0))
    if j_tile <= 0:
        j_tile = 1024
    agg = getattr(solver, "matvec_cuda_aggregate_offdiag", None)
    if agg is None:
        agg = True
    out = {
        "matvec_backend": backend,
        "max_cycle": int(max_cycle),
        "kernel_profile": True,
        "kernel_profile_print": False,
        "kernel_profile_cuda_sync": False,
        "matvec_cuda_hop_profile": True,
        "matvec_cuda_dtype": str(getattr(solver, "matvec_cuda_dtype", "float64")),
        "matvec_cuda_gemm_backend": str(getattr(solver, "matvec_cuda_gemm_backend", "gemmex_fp64")),
        "matvec_cuda_fp32_coeff_data": bool(getattr(solver, "matvec_cuda_fp32_coeff_data", False)),
        "matvec_cuda_use_epq_table": bool(getattr(solver, "matvec_cuda_use_epq_table", False)),
        "matvec_cuda_aggregate_offdiag": bool(agg),
        "matvec_cuda_mem_hard_cap_gib": float(getattr(solver, "matvec_cuda_mem_hard_cap_gib", 11.5)),
        "matvec_cuda_max_g_mib": float(getattr(solver, "matvec_cuda_max_g_mib", 256.0)),
        "matvec_cuda_j_tile": int(j_tile),
        "matvec_cuda_threads_g": int(threads_g),
        "matvec_cuda_threads_w": int(threads_w),
        "matvec_cuda_threads_apply": int(threads_apply),
        "matvec_cuda_prefilter_trivial_tasks": str(
            getattr(solver, "matvec_cuda_prefilter_trivial_tasks", "auto")
        ),
        "matvec_cuda_prefilter_trivial_tasks_min_ncsf": int(
            getattr(solver, "matvec_cuda_prefilter_trivial_tasks_min_ncsf", 1_000_000)
        ),
        "matvec_cuda_csr_host_cache": str(getattr(solver, "matvec_cuda_csr_host_cache", "off")),
        "matvec_cuda_csr_host_cache_budget_gib": float(
            getattr(solver, "matvec_cuda_csr_host_cache_budget_gib", 0.0)
        ),
        "matvec_cuda_csr_host_cache_min_ncsf": int(
            getattr(solver, "matvec_cuda_csr_host_cache_min_ncsf", 1_000_000)
        ),
    }
    if max_space is not None:
        out["max_space"] = int(max_space)
    return out


def _default_steps(
    base_cfg: Mapping[str, Any],
    ncsf: int | None,
    gpu_info: Mapping[str, Any] | None = None,
) -> list[list[dict[str, Any]]]:
    use_epq = bool(base_cfg.get("matvec_cuda_use_epq_table", False))
    agg = bool(base_cfg.get("matvec_cuda_aggregate_offdiag", True))
    steps: list[list[dict[str, Any]]] = []

    if (not use_epq) and agg:
        steps.append(
            [
                {"matvec_cuda_prefilter_trivial_tasks": "off"},
                {"matvec_cuda_prefilter_trivial_tasks": "auto"},
            ]
        )
    else:
        steps.append(
            [
                {"matvec_cuda_prefilter_trivial_tasks": "auto"},
                {"matvec_cuda_prefilter_trivial_tasks": "on"},
                {"matvec_cuda_prefilter_trivial_tasks": "off"},
            ]
        )

    j_tile_vals = _adaptive_j_tile_candidates(ncsf, gpu_info)
    if j_tile_vals:
        steps.append([{"matvec_cuda_j_tile": int(v)} for v in j_tile_vals])

    steps.append([{"matvec_cuda_threads_apply": int(v)} for v in (64, 128, 32)])
    steps.append(
        [
            {"matvec_cuda_threads_g": int(v), "matvec_cuda_threads_w": int(v)}
            for v in (256, 192, 128)
        ]
    )

    max_g_vals = _adaptive_max_g_candidates(gpu_info)
    if len(max_g_vals) > 1:
        steps.append(
            [{"matvec_cuda_max_g_mib": float(v)} for v in max_g_vals]
        )

    csr_cands = _adaptive_csr_host_cache_candidates(ncsf)
    if csr_cands is not None:
        steps.append(csr_cands)

    # Mixed-precision: TF32 GEMM can be ~2× faster for large CAS spaces.
    # Only offer when compute capability >= 8.0 (Ampere+) and ncsf is large
    # enough that GEMM is a meaningful fraction of the hop cost.
    if ncsf is not None and int(ncsf) >= 4096:
        steps.append([
            {"matvec_cuda_gemm_backend": "gemmex_fp64", "matvec_cuda_fp32_coeff_data": False},
            {"matvec_cuda_gemm_backend": "gemmex_tf32", "matvec_cuda_fp32_coeff_data": True},
        ])

    return steps


def autotune(
    solver: Any,
    h1e: Any,
    eri: Any,
    norb: int,
    nelec: int | tuple[int, int],
    *,
    ci0: np.ndarray | None = None,
    ecore: float = 0.0,
    nroots: int | None = None,
    max_cycle: int = 2,
    max_space: int | None = None,
    metric: str = "hop_total_s",
    seed: int = 123,
    n_repeats: int = 1,
    warm_start_fallback: bool = False,
    warm_state_context: Mapping[str, Any] | None = None,
    gpu_autofill: bool = True,
    gpu_info_override: Mapping[str, Any] | None = None,
    gpu_profile_preset: str | None = None,
    extra_kernel_kwargs: Mapping[str, Any] | None = None,
    candidate_overrides: Sequence[Mapping[str, Any]] | None = None,
    apply_best: bool = False,
    refine: bool = False,
    verbose: bool = False,
) -> AutoTuneResult:
    """Autotune CUDA matvec settings and suggest the best configuration.

    The function runs a bounded sweep and ranks trials by ``metric`` (lower is better).
    Defaults are tuned for fast debug iteration, not exhaustive search.
    """

    if int(n_repeats) <= 0:
        raise ValueError("n_repeats must be >= 1")

    ncsf = _ncsf_hint(solver, int(norb), nelec)
    ci0_use = ci0
    if (
        ci0_use is None
        and (not bool(warm_start_fallback))
        and nroots in (None, 1)
        and ncsf is not None
        and ncsf > 0
    ):
        ci0_use = _make_ci0(int(ncsf), int(seed))

    base_cfg = _default_base_kwargs(solver, max_cycle=int(max_cycle), max_space=max_space)
    preset_applied: dict[str, Any] = _gpu_profile_preset_overrides(
        gpu_profile_preset,
        base_cfg=base_cfg,
        ncsf=ncsf,
    )
    if preset_applied:
        base_cfg.update(dict(preset_applied))
    gpu_info: dict[str, Any] | None = None
    gpu_autofill_applied: dict[str, Any] = {}
    if bool(gpu_autofill):
        if gpu_info_override is not None:
            gpu_info = dict(gpu_info_override)
        else:
            gpu_info = detect_cuda_device_info()
        gpu_autofill_applied = _gpu_autofill_overrides(base_cfg, ncsf=ncsf, gpu_info=gpu_info)
        if gpu_autofill_applied:
            base_cfg.update(dict(gpu_autofill_applied))
    if extra_kernel_kwargs:
        base_cfg.update(dict(extra_kernel_kwargs))
    if bool(verbose) and gpu_info is not None:
        print(f"[autotune] gpu_info={gpu_info}")
        if preset_applied:
            print(f"[autotune] gpu_profile_preset={gpu_profile_preset} applied={preset_applied}")
        if gpu_autofill_applied:
            print(f"[autotune] gpu_autofill_applied={gpu_autofill_applied}")

    trials: list[AutoTuneTrial] = []
    cache: dict[tuple[tuple[str, Any], ...], AutoTuneTrial] = {}

    def _run_config(cfg: Mapping[str, Any]) -> AutoTuneTrial:
        frozen = _freeze_cfg(cfg)
        if frozen in cache:
            return cache[frozen]

        wall_vals: list[float] = []
        casci_vals: list[float] = []
        hop_total_vals: list[float] = []
        hop_time_vals: list[float] = []
        hop_calls_vals: list[float] = []
        csr_build_vals: list[float] = []
        offdiag_gemm_vals: list[float] = []
        offdiag_apply_vals: list[float] = []
        csr_prefilter_vals: list[float] = []

        err_msg: str | None = None
        for _ in range(int(n_repeats)):
            t0 = time.perf_counter()
            try:
                call_kwargs = dict(cfg)
                if bool(warm_start_fallback):
                    call_kwargs["warm_state_enable"] = True
                    call_kwargs["warm_state_update"] = True
                    if warm_state_context is not None:
                        call_kwargs["warm_state_context"] = dict(warm_state_context)
                solver.kernel(
                    h1e,
                    eri,
                    int(norb),
                    nelec,
                    ci0=ci0_use,
                    ecore=float(ecore),
                    nroots=nroots,
                    **call_kwargs,
                )
                wall = float(time.perf_counter() - t0)
                prof = getattr(solver, "_last_kernel_profile", None) or {}
                mm = _extract_metrics(prof if isinstance(prof, Mapping) else {})
                wall_vals.append(wall)
                casci_vals.append(wall)
                if mm["hop_total_s"] is not None:
                    hop_total_vals.append(float(mm["hop_total_s"]))
                if mm["hop_time_s"] is not None:
                    hop_time_vals.append(float(mm["hop_time_s"]))
                if mm["hop_calls"] is not None:
                    hop_calls_vals.append(float(mm["hop_calls"]))
                if mm["csr_build_s"] is not None:
                    csr_build_vals.append(float(mm["csr_build_s"]))
                if mm["offdiag_gemm_s"] is not None:
                    offdiag_gemm_vals.append(float(mm["offdiag_gemm_s"]))
                if mm["offdiag_apply_s"] is not None:
                    offdiag_apply_vals.append(float(mm["offdiag_apply_s"]))
                if mm["csr_prefilter_s"] is not None:
                    csr_prefilter_vals.append(float(mm["csr_prefilter_s"]))
            except Exception as exc:  # noqa: BLE001
                err_msg = f"{type(exc).__name__}: {exc}"
                break

        def _avg(vals: list[float]) -> float | None:
            if not vals:
                return None
            return float(sum(vals) / len(vals))

        metrics_for_score = {
            "casci_s": _avg(casci_vals),
            "hop_total_s": _avg(hop_total_vals),
            "hop_time_s": _avg(hop_time_vals),
        }
        score, metric_used = _score_from_metrics(metric=str(metric), metrics=metrics_for_score, wall_s=_avg(wall_vals))
        if err_msg is not None:
            score = float("inf")
            metric_used = "error"

        trial = AutoTuneTrial(
            index=len(trials),
            config=dict(cfg),
            wall_s=_avg(wall_vals),
            casci_s=_avg(casci_vals),
            hop_total_s=_avg(hop_total_vals),
            hop_time_s=_avg(hop_time_vals),
            hop_calls=_avg(hop_calls_vals),
            csr_build_s=_avg(csr_build_vals),
            offdiag_gemm_s=_avg(offdiag_gemm_vals),
            offdiag_apply_s=_avg(offdiag_apply_vals),
            csr_prefilter_s=_avg(csr_prefilter_vals),
            score=float(score),
            metric_used=str(metric_used),
            error=err_msg,
        )
        trials.append(trial)
        cache[frozen] = trial
        if bool(verbose):
            label = "ERR" if err_msg else "OK "
            print(f"[autotune] {label} trial={trial.index} score={trial.score:.6f} cfg={dict(cfg)}")
            if err_msg:
                print(f"[autotune] error: {err_msg}")
        return trial

    best_cfg = dict(base_cfg)
    baseline = _run_config(best_cfg)
    best_trial = baseline
    early_stopped = False
    steps_run = 0

    if candidate_overrides is None:
        steps = _default_steps(best_cfg, ncsf, gpu_info=gpu_info)
        for step in steps:
            step_best_cfg = dict(best_cfg)
            step_best_trial = best_trial
            for delta in step:
                cand = dict(best_cfg)
                cand.update(dict(delta))
                trial = _run_config(cand)
                if trial.score < step_best_trial.score * (1.0 - NOISE_MARGIN):
                    step_best_trial = trial
                    step_best_cfg = cand
            best_cfg = step_best_cfg
            best_trial = step_best_trial
            steps_run += 1
            # Early stopping: if no meaningful improvement over baseline
            if best_trial.score >= baseline.score * (1.0 - EARLY_STOP_REL_THRESHOLD):
                early_stopped = True
                if bool(verbose):
                    print(f"[autotune] early stop after step {steps_run}: "
                          f"best={best_trial.score:.6f} baseline={baseline.score:.6f}")
                break

        # Optional refine pass: re-test first two dimensions with final config
        if bool(refine) and not early_stopped and len(steps) >= 2:
            if bool(verbose):
                print("[autotune] refine pass: re-testing first 2 dimensions")
            for step in steps[:2]:
                step_best_cfg = dict(best_cfg)
                step_best_trial = best_trial
                for delta in step:
                    cand = dict(best_cfg)
                    cand.update(dict(delta))
                    trial = _run_config(cand)
                    if trial.score < step_best_trial.score * (1.0 - NOISE_MARGIN):
                        step_best_trial = trial
                        step_best_cfg = cand
                best_cfg = step_best_cfg
                best_trial = step_best_trial
                steps_run += 1
    else:
        for ov in candidate_overrides:
            cand = dict(base_cfg)
            cand.update(dict(ov))
            trial = _run_config(cand)
            if trial.score < best_trial.score:
                best_cfg = cand
                best_trial = trial

    search_stats = {
        "total_trials": len(trials),
        "early_stopped": early_stopped,
        "steps_run": steps_run,
        "cache_hits": len(trials) - len(cache),
    }

    if apply_best:
        for k, v in best_cfg.items():
            if not str(k).startswith("matvec_cuda_") and k != "matvec_backend":
                continue
            if hasattr(solver, str(k)):
                setattr(solver, str(k), v)

    return AutoTuneResult(
        metric=str(metric),
        baseline=baseline,
        best=best_trial,
        trials=trials,
        recommended_solver_kwargs=dict(best_cfg),
        ncsf_hint=ncsf,
        gpu_info=gpu_info,
        gpu_autofill_applied=dict(gpu_autofill_applied),
        gpu_profile_preset=None if gpu_profile_preset is None else str(gpu_profile_preset),
        warm_start_fallback=bool(warm_start_fallback),
        search_stats=search_stats,
    )


__all__ = [
    "AutoTuneTrial",
    "AutoTuneResult",
    "detect_cuda_device_info",
    "list_gpu_profile_presets",
    "autotune",
]
