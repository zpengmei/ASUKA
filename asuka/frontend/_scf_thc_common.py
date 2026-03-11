from __future__ import annotations

from typing import Any


def resolve_thc_grid_and_dvr_basis(
    *,
    mol,
    aux_basis,
    thc_grid_kind: str,
    thc_dvr_basis: Any | None,
    expand_contractions: bool,
    build_ao_basis_cart_fn: Any,
):
    """Resolve THC grid kind and optional DVR basis materialization."""
    grid_kind_s = str(thc_grid_kind).strip().lower()
    dvr_basis_cart = None
    if grid_kind_s in {"rdvr", "r-dvr", "r_dvr", "fdvr", "f-dvr", "f_dvr"}:
        if thc_dvr_basis is None:
            dvr_basis_cart = aux_basis
        elif isinstance(thc_dvr_basis, str) and str(thc_dvr_basis).strip().lower() in {"aux", "auxbasis"}:
            dvr_basis_cart = aux_basis
        else:
            dvr_basis_cart, _unused_name = build_ao_basis_cart_fn(
                mol,
                basis=thc_dvr_basis,
                expand_contractions=bool(expand_contractions),
            )
    return grid_kind_s, dvr_basis_cart


def resolve_thc_mp_store_policy(
    *,
    thc_mp_mode: str,
    thc_store_Z: bool | None,
    thc_profile: dict | None,
) -> tuple[str, bool]:
    """Normalize THC mixed-precision mode and `store_Z` policy."""
    thc_mp_mode_s = str(thc_mp_mode).strip().lower()
    if thc_mp_mode_s not in {"fp64", "tf32"}:
        raise ValueError("thc_mp_mode must be 'fp64' or 'tf32'")
    if thc_store_Z is None:
        store_Z_eff = bool(thc_mp_mode_s != "tf32")
    else:
        store_Z_eff = bool(thc_store_Z)
    if thc_profile is not None:
        thc_profile.setdefault("mp_mode", str(thc_mp_mode_s))
        thc_profile.setdefault("store_Z", bool(store_Z_eff))
    return thc_mp_mode_s, bool(store_Z_eff)


def coerce_local_thc_config(
    *,
    thc_local_config: Any | None,
    local_config_type: Any,
):
    """Coerce user-provided local THC config into the expected config type."""
    if thc_local_config is None:
        return local_config_type()
    if isinstance(thc_local_config, local_config_type):
        return thc_local_config
    if isinstance(thc_local_config, dict):
        return local_config_type(**thc_local_config)
    raise TypeError("thc_local_config must be a LocalTHCConfig or a dict")
