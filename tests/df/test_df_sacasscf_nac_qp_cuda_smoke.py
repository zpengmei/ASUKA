import numpy as np
import pytest


@pytest.mark.cuda
def test_df_sacasscf_nac_qp_cuda_smoke(monkeypatch):
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

    # This test specifically exercises the packed-Qp DF factor layout (B_Qp) used by CUDA DF-SCF.
    if not hasattr(_ext, "df_unpack_qp_to_qmn_block_device"):
        pytest.skip("CUDA extension lacks packed-Qp unpack support (df_unpack_qp_to_qmn_block_device)")

    monkeypatch.setenv("ASUKA_DF_AO_PACKED_S2", "1")

    from asuka.frontend import Molecule, run_hf
    from asuka.mcscf import run_casscf
    from asuka.mcscf.nac import sacasscf_nonadiabatic_couplings_df

    mol = Molecule.from_atoms(
        atoms=[
            ("Li", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, 3.0)),
        ],
        unit="Bohr",
        basis="sto-3g",
        cart=True,
    )

    scf_out = run_hf(mol, method="rhf", backend="cuda", max_cycle=50, conv_tol=1e-10, conv_tol_dm=1e-8)
    assert bool(getattr(scf_out.scf, "converged", False))

    B = getattr(scf_out, "df_B", None)
    assert B is not None
    assert isinstance(B, cp.ndarray)
    assert int(getattr(B, "ndim", 0)) == 2  # packed Qp

    mc = run_casscf(
        scf_out,
        ncore=1,
        ncas=2,
        nelecas=2,
        backend="cuda",
        max_cycle_macro=30,
        nroots=2,
        root_weights=[0.5, 0.5],
    )
    assert bool(getattr(mc, "converged", False))

    nac_num = sacasscf_nonadiabatic_couplings_df(
        scf_out,
        mc,
        pairs=[(0, 1)],
        mult_ediff=True,
        df_backend="cuda",
        response_term="split_orbfd",
        z_tol=1e-9,
        z_maxiter=120,
    )
    h01 = np.asarray(nac_num[0, 1], dtype=np.float64)
    h10 = np.asarray(nac_num[1, 0], dtype=np.float64)
    assert h01.shape == (int(mol.natm), 3)
    assert bool(np.isfinite(h01).all())
    assert float(np.linalg.norm(h01)) > 1.0e-10
    assert bool(np.allclose(h10, 0.0, atol=1.0e-14, rtol=0.0))
