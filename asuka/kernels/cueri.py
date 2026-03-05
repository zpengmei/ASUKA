from __future__ import annotations

"""cuERI CUDA extension kernel registry.

This module centralizes:
- importing `asuka.cueri._cueri_cuda_ext`
- probing for the presence of key kernel entry points
- providing stable, documented symbol names for downstream dispatch/docs
"""

from typing import Any

from ._base import KernelSymbol, try_import


EXT_MODULE = "asuka.cueri._cueri_cuda_ext"


def load_ext() -> Any | None:
    """Return the imported cuERI CUDA extension module, or None if unavailable."""

    mod, _err = try_import(EXT_MODULE)
    return mod


def require_ext() -> Any:
    """Return the cuERI CUDA extension module or raise with a build hint."""

    mod, err = try_import(EXT_MODULE)
    if mod is None:
        msg = "cuERI CUDA extension is unavailable"
        if err is not None:
            msg += f" ({type(err).__name__}: {err})"
        msg += ". Build via `python -m asuka.cueri.build_cuda_ext`."
        raise RuntimeError(msg)
    return mod


KERNELS: list[KernelSymbol] = [
    # ---- Packed-Qp helpers (DF 3c factors) ----
    KernelSymbol(
        EXT_MODULE,
        "df_pack_mnq_to_qp_device",
        category="df_packed_qp",
        purpose="Pack mnQ (nao,nao,naux) -> Qp (naux,ntri) on device.",
        io="in: mnQ f64 (flattened) -> out: Qp f64 (flattened)",
    ),
    KernelSymbol(
        EXT_MODULE,
        "df_pack_qmn_to_qp_device",
        category="df_packed_qp",
        purpose="Pack Qmn (naux,nao,nao) -> Qp (naux,ntri) on device.",
        io="in: Qmn f64 (flattened) -> out: Qp f64 (flattened)",
    ),
    KernelSymbol(
        EXT_MODULE,
        "df_pack_qmn_block_to_qp_device",
        category="df_packed_qp",
        purpose="Pack a streamed Qmn block (q,nao,nao) -> Qp slice (q,ntri).",
        io="in: Qmn block f64 -> out: Qp slice f64",
    ),
    KernelSymbol(
        EXT_MODULE,
        "df_pack_lf_block_to_qp_device",
        category="df_packed_qp",
        purpose="Pack Lf_block (nao, q*nao) -> Qp slice for q0..q1.",
        io="in: Lf_block f64 -> out: Qp f64",
    ),
    KernelSymbol(
        EXT_MODULE,
        "df_unpack_qp_to_qmn_block_device",
        category="df_packed_qp",
        purpose="Unpack Qp (naux,ntri) -> streamed Qmn block (q,nao,nao).",
        io="in: Qp (flattened) -> out: Qmn block f64",
    ),
    KernelSymbol(
        EXT_MODULE,
        "df_unpack_qp_to_mnq_device",
        category="df_packed_qp",
        purpose="Unpack Qp (naux,ntri) -> mnQ (nao,nao,naux) on device.",
        io="in: Qp (flattened) -> out: mnQ f64 (flattened)",
    ),
    # ---- DF int3c tile scatter helpers (DF build) ----
    KernelSymbol(
        EXT_MODULE,
        "scatter_df_int3c2e_tiles_inplace_device",
        category="df_int3c_scatter",
        purpose="Scatter (A,B) tile outputs into global DF int3c buffer on device.",
        io="in: tile buffer -> out: global 3c tensor buffer",
    ),
    KernelSymbol(
        EXT_MODULE,
        "scatter_df_int3c2e_tiles_cart_to_sph_inplace_device",
        category="df_int3c_scatter",
        purpose="Scatter DF int3c tiles while transforming Cartesian->spherical on device.",
        io="in: cart tile buffer -> out: spherical global 3c tensor buffer",
    ),
    # ---- Symmetry / layout helpers ----
    KernelSymbol(
        EXT_MODULE,
        "df_symmetrize_qmn_inplace_device",
        category="df_sym",
        purpose="In-place symmetrize Qmn (naux,nao,nao) over (m,n).",
        io="in/out: Qmn f64 (flattened)",
    ),
    KernelSymbol(
        EXT_MODULE,
        "df_symmetrize_mnq_inplace_device",
        category="df_sym",
        purpose="In-place symmetrize mnQ (nao,nao,naux) over (m,n).",
        io="in/out: mnQ f64 (flattened)",
    ),
    KernelSymbol(
        EXT_MODULE,
        "df_symmetrize_qmn_to_mnq_device",
        category="df_sym",
        purpose="Fuse Qmn->mnQ transpose with (m,n) symmetrization.",
        io="in: Qmn f64 -> out: mnQ f64",
    ),
    KernelSymbol(
        EXT_MODULE,
        "df_symmetrize_mnq_to_f32_device",
        category="df_sym",
        purpose="Symmetrize mnQ and convert to float32 (bandwidth/VRAM reduction).",
        io="in: mnQ f64 -> out: mnQ f32",
    ),
    # ---- DF B/cart<->sph transforms (used by cart2sph + DF grad paths) ----
    KernelSymbol(
        EXT_MODULE,
        "df_B_cart_to_sph_sym_device",
        category="df_cart2sph",
        purpose="Transform DF B from Cartesian mnQ to spherical mnQ (sym shell-pairs).",
        io="in: cart mnQ f64 -> out: sph mnQ f64",
    ),
    KernelSymbol(
        EXT_MODULE,
        "df_B_cart_to_sph_qmn_sym_device",
        category="df_cart2sph",
        purpose="Transform DF B from Cartesian mnQ to spherical Qmn (sym shell-pairs).",
        io="in: cart mnQ f64 -> out: sph Qmn f64",
    ),
    KernelSymbol(
        EXT_MODULE,
        "df_bar_x_sph_to_cart_sym_device",
        category="df_cart2sph",
        purpose="Transform bar_X from spherical mnQ to Cartesian mnQ (sym shell-pairs).",
        io="in: sph mnQ f64 -> out: cart mnQ f64",
    ),
    KernelSymbol(
        EXT_MODULE,
        "df_bar_x_sph_qmn_to_cart_sym_device",
        category="df_cart2sph",
        purpose="Transform bar_X from spherical Qmn to Cartesian mnQ (sym shell-pairs).",
        io="in: sph Qmn f64 -> out: cart mnQ f64",
    ),
    # ---- DF gradient contraction (3c derivative) ----
    KernelSymbol(
        EXT_MODULE,
        "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qp_inplace_device",
        category="df_grad_3c",
        purpose="3c derivative contraction for sph-bar adjoint in packed-Qp layout (single bar).",
        io="in: bar_X Qp (flattened) -> out: grad (natm,3) f64",
    ),
    KernelSymbol(
        EXT_MODULE,
        "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qp_abtile_inplace_device",
        category="df_grad_3c",
        purpose="Packed-Qp sph-bar 3c derivative contraction with AB tiling (single bar).",
        io="in: bar_X Qp -> out: grad f64",
    ),
    KernelSymbol(
        EXT_MODULE,
        "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qp_multibar2_inplace_device",
        category="df_grad_3c",
        purpose="Packed-Qp sph-bar 3c derivative contraction for 2 bars in one pass.",
        io="in: 2x bar_X Qp -> out: grad (2,natm,3) f64",
    ),
    KernelSymbol(
        EXT_MODULE,
        "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qp_abtile_multibar2_inplace_device",
        category="df_grad_3c",
        purpose="Packed-Qp sph-bar 3c derivative contraction (AB tiled) for 2 bars.",
        io="in: 2x bar_X Qp -> out: grad (2,natm,3) f64",
    ),
    KernelSymbol(
        EXT_MODULE,
        "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qp_multibar3_inplace_device",
        category="df_grad_3c",
        purpose="Packed-Qp sph-bar 3c derivative contraction for 3 bars in one pass.",
        io="in: 3x bar_X Qp -> out: grad (3,natm,3) f64",
    ),
    KernelSymbol(
        EXT_MODULE,
        "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qp_abtile_multibar3_inplace_device",
        category="df_grad_3c",
        purpose="Packed-Qp sph-bar 3c derivative contraction (AB tiled) for 3 bars.",
        io="in: 3x bar_X Qp -> out: grad (3,natm,3) f64",
    ),
    KernelSymbol(
        EXT_MODULE,
        "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_inplace_device",
        category="df_grad_3c",
        purpose="3c derivative contraction for sph-bar adjoint in Qmn layout (single bar, non-streamed).",
        io="in: bar_X Qmn -> out: grad f64",
    ),
    KernelSymbol(
        EXT_MODULE,
        "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_abtile_inplace_device",
        category="df_grad_3c",
        purpose="Qmn sph-bar 3c derivative contraction with AB tiling (single bar).",
        io="in: bar_X Qmn -> out: grad f64",
    ),
    KernelSymbol(
        EXT_MODULE,
        "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_streamed_inplace_device",
        category="df_grad_3c",
        purpose="Streamed-Q aux-block 3c derivative contraction in Qmn layout (single bar).",
        io="in: streamed bar_X Qmn blocks -> out: grad f64",
    ),
    KernelSymbol(
        EXT_MODULE,
        "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_streamed_abtile_inplace_device",
        category="df_grad_3c",
        purpose="Streamed-Q Qmn sph-bar 3c derivative contraction with AB tiling (single bar).",
        io="in: streamed bar_X Qmn blocks -> out: grad f64",
    ),
    KernelSymbol(
        EXT_MODULE,
        "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_streamed_multibar_inplace_device",
        category="df_grad_3c",
        purpose="Streamed-Q Qmn sph-bar 3c derivative contraction for multiple bars.",
        io="in: stacked Qmn blocks -> out: grad (nbar,natm,3) f64",
    ),
    KernelSymbol(
        EXT_MODULE,
        "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_streamed_abtile_multibar_inplace_device",
        category="df_grad_3c",
        purpose="Streamed-Q Qmn sph-bar 3c derivative contraction (AB tiled) for multiple bars.",
        io="in: stacked Qmn blocks -> out: grad (nbar,natm,3) f64",
    ),
    # ---- DF gradient contraction (metric 2c derivative) ----
    KernelSymbol(
        EXT_MODULE,
        "df_metric_2c2e_deriv_contracted_cart_allsp_atomgrad_tril_inplace_device",
        category="df_grad_2c",
        purpose="Metric (2c2e) derivative contraction using aux-triangular symmetry (fast path).",
        io="in: bar_V (flattened) -> out: grad (natm,3) f64",
    ),
    KernelSymbol(
        EXT_MODULE,
        "df_metric_2c2e_deriv_contracted_cart_sp_batch_inplace_device",
        category="df_grad_2c",
        purpose="Metric (2c2e) derivative contraction using sp-batch output + atom scatter (fallback).",
        io="in: bar_V -> out: (nt,2,3) f64",
    ),
    # ---- 1e spherical prebuilt contractions ----
    KernelSymbol(
        EXT_MODULE,
        "int1e_dS_deriv_contracted_sph_inplace_device",
        category="int1e_sph",
        purpose="Contract prebuilt spherical dS tensor with an AO matrix on device.",
        io="in: dS (natm,3,nao,nao), M (nao,nao) -> out: grad (natm,3)",
    ),
    KernelSymbol(
        EXT_MODULE,
        "int1e_dhcore_deriv_contracted_sph_inplace_device",
        category="int1e_sph",
        purpose="Contract prebuilt spherical dT/dV tensors with an AO matrix on device.",
        io="in: dT,dV (natm,3,nao,nao), M (nao,nao) -> out: grad (natm,3)",
    ),
]


def probe() -> dict[str, Any]:
    """Return a structured capability report for the cuERI CUDA extension."""

    mod, err = try_import(EXT_MODULE)
    out: dict[str, Any] = {
        "module": str(EXT_MODULE),
        "present": bool(mod is not None),
        "import_error": None if err is None else f"{type(err).__name__}: {err}",
        "symbols": {},
        "categories": {},
    }
    cats: dict[str, list[str]] = {}
    syms: dict[str, bool] = {}
    for ks in KERNELS:
        ok = bool(ks.available())
        syms[str(ks.symbol)] = ok
        cats.setdefault(str(ks.category), []).append(str(ks.symbol))
    out["symbols"] = syms
    out["categories"] = cats
    return out


__all__ = [
    "EXT_MODULE",
    "KERNELS",
    "load_ext",
    "require_ext",
    "probe",
]
