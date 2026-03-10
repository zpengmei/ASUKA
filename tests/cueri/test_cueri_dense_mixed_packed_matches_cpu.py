"""Verify that the packed-accumulation mixed-index ERI builders (pu_wx, pq_uv)
produce the same output as the CPU dense reference builder.

After Finding 7 partial fix: _build_pu_wx_eri_mat_dense_rys_from_cached now uses
packed ket coefficients (npair_act) + end-of-loop unpack; _build_pq_uv_eri_mat_dense_rys_from_cached
uses packed bra (npair_mo) + ket (npair_act) + end-of-loop unpack for both.
"""
from __future__ import annotations

import numpy as np
import pytest


def _build_reference_pu_wx(eri4, C_mo, C_act):
    """CPU reference: (pu|wx) = einsum puwx via full 4-index tensor."""
    nao, nmo = C_mo.shape
    nao2, ncas = C_act.shape
    # eri4 shape: (nao, nao, nao, nao)
    # (pu|wx) = sum_{mu,nu,lam,sig} C_mo[mu,p] * C_act[nu,u] * C_act[lam,w] * C_act[sig,x] * eri4[mu,nu,lam,sig]
    tmp = np.einsum("mnls,mp,nu->puls", eri4, C_mo, C_act, optimize=True)
    out = np.einsum("puls,lw,sx->puwx", tmp, C_act, C_act, optimize=True)
    return out.reshape(nmo * ncas, ncas * ncas)


def _build_reference_pq_uv(eri4, C_mo, C_act):
    """CPU reference: (pq|uv) = einsum pquv via full 4-index tensor."""
    nao, nmo = C_mo.shape
    nao2, ncas = C_act.shape
    tmp = np.einsum("mnls,mp,nq->pqls", eri4, C_mo, C_mo, optimize=True)
    out = np.einsum("pqls,lu,sv->pquv", tmp, C_act, C_act, optimize=True)
    return out.reshape(nmo * nmo, ncas * ncas)


@pytest.mark.cuda
def test_pu_wx_packed_ket_matches_cpu_reference():
    cp = pytest.importorskip("cupy")
    if int(cp.cuda.runtime.getDeviceCount()) <= 0:
        pytest.skip("no CUDA device")

    try:
        from asuka.cueri.gpu import has_cuda_ext
    except Exception as e:
        pytest.skip(f"cuERI GPU module unavailable ({type(e).__name__}: {e})")
    if not bool(has_cuda_ext()):
        pytest.skip("cuERI CUDA extension unavailable")

    from asuka.frontend.molecule import Molecule
    from asuka.frontend.one_electron import build_ao_basis_cart
    from asuka.cueri.active_space_dense_gpu import CuERIActiveSpaceDenseGPUBuilder

    mol = Molecule.from_atoms(
        [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4)), ("H", (0.0, 1.4, 0.0))],
        unit="Bohr",
        basis="sto-3g",
        cart=True,
    )
    ao_basis, _ = __import__("asuka.frontend.one_electron", fromlist=["build_ao_basis_cart"]).build_ao_basis_cart(mol, basis=mol.basis, expand_contractions=True)

    rng = np.random.default_rng(20260310)
    nao = int(ao_basis.shell_ao_start[-1]) + 3  # approximate nao
    # Get actual nao from basis
    from asuka.integrals.int1e_cart import nao_cart_from_basis
    nao = int(nao_cart_from_basis(ao_basis))
    nmo = nao
    ncas = min(3, nao)

    C_mo_np = rng.standard_normal((nao, nmo))
    C_act_np = rng.standard_normal((nao, ncas))
    # Orthogonalize for stability
    C_mo_np, _ = np.linalg.qr(C_mo_np)
    C_act_np, _ = np.linalg.qr(C_act_np)

    C_mo = cp.asarray(C_mo_np, dtype=cp.float64)
    C_act = cp.asarray(C_act_np, dtype=cp.float64)

    builder = CuERIActiveSpaceDenseGPUBuilder(ao_basis=ao_basis, threads=256, max_tile_bytes=64 << 20)

    gpu_result = cp.asnumpy(builder.build_pu_wx_eri_mat(C_mo, C_act))

    # CPU reference via full 4-index tensor
    from asuka.cueri.dense_cpu import build_active_eri_mat_dense_cpu
    # Use the dense CPU builder or a direct einsum over PySCF integrals
    # Build eri4 from dense CPU
    from asuka.cueri.active_space_dense_cpu import CuERIActiveSpaceDenseCPUBuilder
    cpu_builder = CuERIActiveSpaceDenseCPUBuilder(ao_basis=ao_basis)
    # Build full ordered ERI in AO basis: use identity coefficients
    C_identity = np.eye(nao)
    eri_ao_mat = cpu_builder.build_eri_mat(C_identity, eps_ao=0.0, eps_mo=0.0, blas_nthreads=1, profile=None)
    eri4 = eri_ao_mat.reshape(nao, nao, nao, nao)

    cpu_ref = _build_reference_pu_wx(eri4, C_mo_np, C_act_np)

    np.testing.assert_allclose(
        gpu_result, cpu_ref, rtol=1e-10, atol=1e-10,
        err_msg="pu_wx GPU (packed ket) does not match CPU reference",
    )


@pytest.mark.cuda
def test_pq_uv_packed_bra_ket_matches_cpu_reference():
    cp = pytest.importorskip("cupy")
    if int(cp.cuda.runtime.getDeviceCount()) <= 0:
        pytest.skip("no CUDA device")

    try:
        from asuka.cueri.gpu import has_cuda_ext
    except Exception as e:
        pytest.skip(f"cuERI GPU module unavailable ({type(e).__name__}: {e})")
    if not bool(has_cuda_ext()):
        pytest.skip("cuERI CUDA extension unavailable")

    from asuka.frontend.molecule import Molecule
    from asuka.frontend.one_electron import build_ao_basis_cart
    from asuka.cueri.active_space_dense_gpu import CuERIActiveSpaceDenseGPUBuilder
    from asuka.integrals.int1e_cart import nao_cart_from_basis
    from asuka.cueri.active_space_dense_cpu import CuERIActiveSpaceDenseCPUBuilder

    mol = Molecule.from_atoms(
        [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4)), ("H", (0.0, 1.4, 0.0))],
        unit="Bohr",
        basis="sto-3g",
        cart=True,
    )
    ao_basis, _ = build_ao_basis_cart(mol, basis=mol.basis, expand_contractions=True)

    rng = np.random.default_rng(20260311)
    nao = int(nao_cart_from_basis(ao_basis))
    nmo = nao
    ncas = min(3, nao)

    C_mo_np = rng.standard_normal((nao, nmo))
    C_act_np = rng.standard_normal((nao, ncas))
    C_mo_np, _ = np.linalg.qr(C_mo_np)
    C_act_np, _ = np.linalg.qr(C_act_np)

    C_mo = cp.asarray(C_mo_np, dtype=cp.float64)
    C_act = cp.asarray(C_act_np, dtype=cp.float64)

    builder = CuERIActiveSpaceDenseGPUBuilder(ao_basis=ao_basis, threads=256, max_tile_bytes=64 << 20)
    gpu_result = cp.asnumpy(builder.build_pq_uv_eri_mat(C_mo, C_act))

    cpu_builder = CuERIActiveSpaceDenseCPUBuilder(ao_basis=ao_basis)
    C_identity = np.eye(nao)
    eri_ao_mat = cpu_builder.build_eri_mat(C_identity, eps_ao=0.0, eps_mo=0.0, blas_nthreads=1, profile=None)
    eri4 = eri_ao_mat.reshape(nao, nao, nao, nao)

    cpu_ref = _build_reference_pq_uv(eri4, C_mo_np, C_act_np)

    np.testing.assert_allclose(
        gpu_result, cpu_ref, rtol=1e-10, atol=1e-10,
        err_msg="pq_uv GPU (packed bra+ket) does not match CPU reference",
    )
