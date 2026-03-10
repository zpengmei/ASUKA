from __future__ import annotations

from typing import Any

from asuka.hf.dense_eri import build_ao_eri_dense


def dense_default_threads(backend: str) -> int:
    backend_s = str(backend).strip().lower()
    return 0 if backend_s == "cpu" else 256


def build_dense_ao_eri(
    ao_basis: Any,
    *,
    backend: str,
    dense_threads: int | None,
    dense_max_tile_bytes: int,
    dense_eps_ao: float,
    dense_max_l: int | None,
    dense_mem_budget_gib: float | None,
    default_mem_budget_gib: float,
    profile: dict | None = None,
):
    backend_s = str(backend).strip().lower()
    if backend_s not in {"cpu", "cuda"}:
        raise ValueError("backend must be 'cpu' or 'cuda'")
    threads_i = dense_default_threads(backend_s) if dense_threads is None else int(dense_threads)
    budget_gib = float(default_mem_budget_gib) if dense_mem_budget_gib is None else float(dense_mem_budget_gib)
    dense_prof = profile.setdefault("dense_eri_build", {}) if profile is not None else None
    return build_ao_eri_dense(
        ao_basis,
        backend=str(backend_s),
        threads=int(threads_i),
        max_tile_bytes=int(dense_max_tile_bytes),
        eps_ao=float(dense_eps_ao),
        max_l=dense_max_l,
        mem_budget_gib=float(budget_gib),
        profile=dense_prof,
    )
