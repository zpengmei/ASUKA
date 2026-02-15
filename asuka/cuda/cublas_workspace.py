from __future__ import annotations

from dataclasses import dataclass


def _ceildiv(a: int, b: int) -> int:
    a = int(a)
    b = int(b)
    if b <= 0:
        raise ValueError("ceildiv denominator must be > 0")
    return (a + b - 1) // b


@dataclass(frozen=True)
class FixedPointWorkspaceEstimate:
    bytes_required: int
    max_mantissa_bits_used: int
    mantissa_control: str
    batch_count: int


def estimate_cublas_fp64_fixedpoint_workspace_bytes(
    *,
    m: int,
    n: int,
    k: int,
    batch_count: int = 1,
    is_complex: bool = False,
    mantissa_control: str = "dynamic",
    max_mantissa_bit_count: int = 0,
    default_max_mantissa_bit_count: int = 52,
) -> FixedPointWorkspaceEstimate:
    """Safe bound estimate for FP64 fixed-point emulation workspace (cuBLAS docs).

    This implements the reference function shown in the cuBLAS "Fixed-Point Workspace Requirements"
    documentation, with minor Python-ization and input validation.

    Notes
    -----
    - The estimate is intentionally conservative (safe bound).
    - If `max_mantissa_bit_count<=0`, we substitute `default_max_mantissa_bit_count` to avoid
      under-estimating workspace when the library default is unknown.
    """

    m = int(m)
    n = int(n)
    k = int(k)
    batch_count = int(batch_count)
    if m < 0 or n < 0 or k < 0:
        raise ValueError("m/n/k must be >= 0")
    if batch_count < 1:
        raise ValueError("batch_count must be >= 1")

    mantissa_control = str(mantissa_control).strip().lower()
    if mantissa_control not in ("dynamic", "fixed"):
        raise ValueError("mantissa_control must be 'dynamic' or 'fixed'")

    max_mantissa_bit_count = int(max_mantissa_bit_count)
    if max_mantissa_bit_count <= 0:
        max_mantissa_bit_count = int(default_max_mantissa_bit_count)
    if max_mantissa_bit_count < 0:
        raise ValueError("max_mantissa_bit_count must be >= 0")

    # Reference constants from docs.
    multiplier = 1.25
    constant_size = 128 * 1024 * 1024
    mult = 2 if bool(is_complex) else 1

    num_slices = _ceildiv(max_mantissa_bit_count + 1, 8)
    padded_m = _ceildiv(m, 1024) * 1024
    padded_n = _ceildiv(n, 1024) * 1024
    padded_k = _ceildiv(k, 128) * 128
    num_blocks_k = _ceildiv(k, 64)

    # int8 A/B slices + int32 scaling factors (see docs).
    gemm_workspace = 1 * ((padded_m * padded_k) + (padded_n * padded_k)) * mult * num_slices
    gemm_workspace += 4 * (padded_m + padded_n) * mult
    if bool(is_complex):
        gemm_workspace += 8 * (m * n) * mult * mult

    adp_workspace = 0
    if mantissa_control == "dynamic":
        adp_workspace = 4 * ((m * num_blocks_k) + (n * num_blocks_k) + (m * n)) * mult

    bytes_required = int(max(gemm_workspace, adp_workspace) * batch_count * multiplier) + int(constant_size)
    return FixedPointWorkspaceEstimate(
        bytes_required=int(bytes_required),
        max_mantissa_bits_used=int(max_mantissa_bit_count),
        mantissa_control=str(mantissa_control),
        batch_count=int(batch_count),
    )


def recommend_cublas_workspace_bytes_for_emulated_fp64_gemm(
    *,
    ws_info: dict[str, object],
    gemm_shapes: list[tuple[int, int, int]],
    batch_count: int = 1,
    is_complex: bool = False,
    cap_bytes: int | None = None,
    default_max_mantissa_bit_count: int = 52,
) -> int:
    """Return a conservative recommended cuBLAS workspace size for emulated-FP64 GEMMEx."""

    if not gemm_shapes:
        return 0

    gemm_backend = str(ws_info.get("gemm_backend", "")).strip().lower()
    if gemm_backend != "gemmex_emulated_fixedpoint":
        return 0

    mantissa_control = str(ws_info.get("fixed_point_mantissa_control_name", "dynamic")).strip().lower()
    max_bits = int(ws_info.get("fixed_point_max_mantissa_bits", 0) or 0)

    req = 0
    for (m, n, k) in gemm_shapes:
        est = estimate_cublas_fp64_fixedpoint_workspace_bytes(
            m=int(m),
            n=int(n),
            k=int(k),
            batch_count=int(batch_count),
            is_complex=bool(is_complex),
            mantissa_control=str(mantissa_control),
            max_mantissa_bit_count=int(max_bits),
            default_max_mantissa_bit_count=int(default_max_mantissa_bit_count),
        )
        req = max(req, int(est.bytes_required))

    if cap_bytes is not None:
        cap = int(cap_bytes)
        if cap > 0:
            req = min(req, cap)

    return int(req)

