"""Shared NDDO core utilities used by semiempirical frontends.

Exports are loaded lazily to avoid import cycles between ``asuka.nddo_core``
and ``asuka.semiempirical`` during package initialization.
"""

from __future__ import annotations

from importlib import import_module

_EXPORT_MAP = {
    "build_ao_offsets": ("asuka.nddo_core.basis", "build_ao_offsets"),
    "nao_for_Z": ("asuka.nddo_core.basis", "nao_for_Z"),
    "symbol_to_Z": ("asuka.nddo_core.basis", "symbol_to_Z"),
    "valence_electrons": ("asuka.nddo_core.basis", "valence_electrons"),
    "build_local_frames": ("asuka.nddo_core.pairs", "build_local_frames"),
    "build_pair_list": ("asuka.nddo_core.pairs", "build_pair_list"),
    "block_rotation": ("asuka.nddo_core.pairs", "block_rotation"),
    "MultipoleParams": ("asuka.nddo_core.multipole", "MultipoleParams"),
    "compute_all_multipole_params": ("asuka.nddo_core.multipole", "compute_all_multipole_params"),
    "derive_multipole_params": ("asuka.nddo_core.multipole", "derive_multipole_params"),
    "build_pair_ri_payload": ("asuka.nddo_core.ri", "build_pair_ri_payload"),
    "build_two_center_integrals": ("asuka.nddo_core.ri", "build_two_center_integrals"),
    "extract_electron_nuclear": ("asuka.nddo_core.ri", "extract_electron_nuclear"),
    "build_core_hamiltonian": ("asuka.nddo_core.core_h", "build_core_hamiltonian"),
    "build_core_hamiltonian_from_pair_terms": (
        "asuka.nddo_core.core_h",
        "build_core_hamiltonian_from_pair_terms",
    ),
    "build_fock": ("asuka.nddo_core.fock_ref", "build_fock"),
    "build_onecenter_eris": ("asuka.nddo_core.fock_ref", "build_onecenter_eris"),
    "core_core_repulsion": ("asuka.nddo_core.energy", "core_core_repulsion"),
    "core_core_repulsion_from_gamma_ss": (
        "asuka.nddo_core.energy",
        "core_core_repulsion_from_gamma_ss",
    ),
    "AtomicData": ("asuka.nddo_core.types", "AtomicData"),
    "PairData": ("asuka.nddo_core.types", "PairData"),
}

__all__ = list(_EXPORT_MAP.keys())


def __getattr__(name: str):
    try:
        mod_name, attr_name = _EXPORT_MAP[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    mod = import_module(mod_name)
    obj = getattr(mod, attr_name)
    globals()[name] = obj
    return obj
