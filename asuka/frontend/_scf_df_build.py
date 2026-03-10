from __future__ import annotations

from typing import Any

from asuka.integrals.cueri_df import CuERIDFConfig
from asuka.integrals.int1e_cart import build_int1e_cart

from ._scf_build import (
    apply_sph_transform,
    atom_coords_charges_bohr,
    build_aux_basis_cart,
)
from ._scf_config import resolve_df_config_overrides
from .one_electron import build_ao_basis_cart


def build_df_metric_cholesky(
    aux_basis,
    *,
    df_config: CuERIDFConfig | None = None,
    profile: dict | None = None,
):
    """Build only the DF metric Cholesky needed by streamed/direct-DF SCF."""

    from asuka.cueri import df as cueri_df  # noqa: PLC0415

    cfg = CuERIDFConfig() if df_config is None else df_config
    df_prof = profile.setdefault("df_build", {}) if profile is not None else None
    if df_prof is not None:
        df_prof.setdefault("metric_only", True)
        df_prof.setdefault("materialized_B", False)
        df_prof.setdefault("backend", str(cfg.backend))
        df_prof.setdefault("mode", str(cfg.mode))
        df_prof.setdefault("threads", int(cfg.threads))

    V = cueri_df.metric_2c2e_basis(
        aux_basis,
        stream=cfg.stream,
        backend=str(cfg.backend),
        mode=str(cfg.mode),
        threads=int(cfg.threads),
    )
    L = cueri_df.cholesky_metric(V)
    del V

    if df_prof is not None:
        try:
            df_prof["L_shape"] = list(map(int, L.shape))
        except Exception:
            pass

    return L


def prepare_direct_df_inputs(
    mol,
    *,
    basis_in: Any,
    auxbasis: Any,
    expand_contractions: bool,
    df_config: CuERIDFConfig | None,
    df_backend: str | None = None,
    df_mode: str | None = None,
    df_threads: int | None = None,
    L_metric=None,
    profile: dict | None = None,
):
    cfg = resolve_df_config_overrides(
        df_config,
        backend=df_backend,
        mode=df_mode,
        threads=df_threads,
    )

    ao_basis, basis_name = build_ao_basis_cart(mol, basis=basis_in, expand_contractions=bool(expand_contractions))
    coords, charges = atom_coords_charges_bohr(mol)
    int1e = build_int1e_cart(ao_basis, atom_coords_bohr=coords, atom_charges=charges)

    aux_basis, auxbasis_name = build_aux_basis_cart(
        mol,
        basis_in=basis_in,
        auxbasis=auxbasis,
        expand_contractions=bool(expand_contractions),
        ao_basis=ao_basis,
    )

    int1e_scf, _B_unused, sph_map = apply_sph_transform(mol, int1e, None, ao_basis, df_B_layout="mnQ")
    df_ao_rep = "cart" if bool(mol.cart) else "sph"

    if L_metric is None:
        L_metric = build_df_metric_cholesky(aux_basis, df_config=cfg, profile=profile)

    return cfg, ao_basis, str(basis_name), int1e_scf, aux_basis, str(auxbasis_name), sph_map, str(df_ao_rep), L_metric
