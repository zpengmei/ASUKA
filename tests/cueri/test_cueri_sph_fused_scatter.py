import numpy as np
import pytest


@pytest.mark.cuda
@pytest.mark.parametrize("aosym", ["s8", "s4"])
def test_cart2sph_left_scatter_fused_matches_two_stage(aosym: str):
    cp = pytest.importorskip("cupy")
    try:
        _ = cp.zeros((1,), dtype=cp.float64)
        _ = int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        pytest.skip("CuPy is present but a CUDA device is unavailable")

    try:
        from asuka.cueri import _cueri_cuda_ext as _ext  # noqa: PLC0415
    except Exception:
        pytest.skip("cuERI CUDA extension is unavailable")
    if aosym == "s8" and not hasattr(_ext, "cart2sph_eri_left_scatter_s8_inplace_device"):
        pytest.skip("cuERI CUDA extension lacks fused sph s8 left-scatter kernel")
    if aosym == "s4" and not hasattr(_ext, "cart2sph_eri_left_scatter_s4_inplace_device"):
        pytest.skip("cuERI CUDA extension lacks fused sph s4 left-scatter kernel")

    from asuka.cueri.eri_utils import npair
    from asuka.cueri.gpu import (
        DeviceShellPairs,
        cart2sph_eri_tiles_device,
        cart2sph_eri_tiles_scatter_sph_s4_inplace_device,
        cart2sph_eri_tiles_scatter_sph_s8_inplace_device,
        scatter_eri_tiles_sph_s4_inplace_device,
        scatter_eri_tiles_sph_s8_inplace_device,
    )
    from asuka.cueri.tasks import TaskList

    la = 1
    lb = 0
    lc = 1
    ld = 0
    nA_sph = 2 * la + 1
    nB_sph = 2 * lb + 1
    nC_sph = 2 * lc + 1
    nD_sph = 2 * ld + 1

    # Two shell-pairs of p-s shells in distinct shell blocks.
    dsp = DeviceShellPairs(
        sp_A=cp.asarray(np.array([0, 2], dtype=np.int32)),
        sp_B=cp.asarray(np.array([1, 3], dtype=np.int32)),
        sp_npair=cp.asarray(np.array([0, 0], dtype=np.int32)),
        sp_pair_start=cp.asarray(np.array([0, 0, 0], dtype=np.int32)),
    )
    shell_ao_start_sph = cp.asarray(np.array([0, 3, 4, 7], dtype=np.int32))
    nao_sph = 8

    tasks = TaskList(
        task_spAB=np.array([0, 1], dtype=np.int32),
        task_spCD=np.array([0, 1], dtype=np.int32),
    )

    rng = np.random.default_rng(20260309)
    tile_cart = cp.asarray(rng.normal(size=(2, 3, 3)), dtype=cp.float64)

    tile_sph = cart2sph_eri_tiles_device(tile_cart, la=la, lb=lb, lc=lc, ld=ld)

    if aosym == "s8":
        out_ref = cp.zeros((int(npair(npair(nao_sph))),), dtype=cp.float64)
        out_fused = cp.zeros_like(out_ref)
        scatter_eri_tiles_sph_s8_inplace_device(
            tasks,
            dsp,
            shell_ao_start_sph=shell_ao_start_sph,
            nao_sph=nao_sph,
            nA=nA_sph,
            nB=nB_sph,
            nC=nC_sph,
            nD=nD_sph,
            tile_vals=tile_sph,
            out_s8=out_ref,
        )
        cart2sph_eri_tiles_scatter_sph_s8_inplace_device(
            tasks,
            dsp,
            shell_ao_start_sph=shell_ao_start_sph,
            nao_sph=nao_sph,
            la=la,
            lb=lb,
            lc=lc,
            ld=ld,
            tile_cart=tile_cart,
            out_s8=out_fused,
        )
    else:
        nao_pair = int(npair(nao_sph))
        out_ref = cp.zeros((nao_pair, nao_pair), dtype=cp.float64)
        out_fused = cp.zeros_like(out_ref)
        scatter_eri_tiles_sph_s4_inplace_device(
            tasks,
            dsp,
            shell_ao_start_sph=shell_ao_start_sph,
            nao_sph=nao_sph,
            nA=nA_sph,
            nB=nB_sph,
            nC=nC_sph,
            nD=nD_sph,
            tile_vals=tile_sph,
            out_s4=out_ref,
        )
        cart2sph_eri_tiles_scatter_sph_s4_inplace_device(
            tasks,
            dsp,
            shell_ao_start_sph=shell_ao_start_sph,
            nao_sph=nao_sph,
            la=la,
            lb=lb,
            lc=lc,
            ld=ld,
            tile_cart=tile_cart,
            out_s4=out_fused,
        )

    denom = float(cp.linalg.norm(out_ref).item())
    denom = max(1.0, denom)
    rel = float((cp.linalg.norm(out_fused - out_ref) / denom).item())
    assert rel < 1e-12

