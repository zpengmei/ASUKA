import numpy as np
import pytest


pytestmark = pytest.mark.cuda


def test_rdvr_grid_device_matches_cpu_moments_no_prune():
    cp = pytest.importorskip("cupy")

    try:
        from asuka import _orbitals_cuda_ext as _orb_ext  # noqa: F401
    except Exception:
        pytest.skip("asuka._orbitals_cuda_ext is not available (build via python -m asuka.build.orbitals_cuda_ext)")

    from asuka.frontend.molecule import Molecule
    from asuka.frontend.scf import _build_aux_basis_cart
    from asuka.density.dvr_grids import make_rdvr_grid
    from asuka.density.dvr_grids_device import make_rdvr_grid_device

    mol = Molecule.from_atoms(
        [
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.0, -0.757, 0.587)),
            ("H", (0.0, 0.757, 0.587)),
        ],
        unit="Angstrom",
        charge=0,
        spin=0,
        basis="sto-3g",
        cart=False,
    )

    # Use aux basis as the DVR basis (common in LS-THC R-DVR setups).
    aux_basis, _aux_name = _build_aux_basis_cart(
        mol,
        basis_in="sto-3g",
        auxbasis="autoaux",
        expand_contractions=True,
    )

    opts = dict(
        angular_n=74,  # Lebedev order (small)
        angular_kind="lebedev",
        radial_rmax=10.0,
        becke_n=3,
        angular_prune=False,  # avoid atomic-number dependency for this unit test
        prune_tol=0.0,  # avoid threshold instabilities
        ortho_cutoff=1e-10,
    )

    pts_cpu, w_cpu = make_rdvr_grid(mol, aux_basis, **opts)
    pts_gpu, w_gpu = make_rdvr_grid_device(mol, aux_basis, threads=256, stream=None, **opts)

    pts_gpu_h = cp.asnumpy(pts_gpu)
    w_gpu_h = cp.asnumpy(w_gpu)

    assert pts_gpu_h.shape == pts_cpu.shape
    assert w_gpu_h.shape == w_cpu.shape
    assert pts_cpu.ndim == 2 and pts_cpu.shape[1] == 3
    assert w_cpu.ndim == 1 and w_cpu.shape[0] == pts_cpu.shape[0]

    # Points should match closely (same radial nodes + same angular dirs).
    np.testing.assert_allclose(pts_gpu_h, pts_cpu, rtol=0.0, atol=1e-12)

    # Weights may differ slightly due to GPU vs CPU Becke partition arithmetic;
    # compare integrated moments rather than elementwise equality.
    sum_cpu = float(np.sum(w_cpu))
    sum_gpu = float(np.sum(w_gpu_h))
    np.testing.assert_allclose(sum_gpu, sum_cpu, rtol=1e-9, atol=1e-9)

    m1_cpu = np.sum(w_cpu[:, None] * pts_cpu, axis=0)
    m1_gpu = np.sum(w_gpu_h[:, None] * pts_gpu_h, axis=0)
    np.testing.assert_allclose(m1_gpu, m1_cpu, rtol=1e-9, atol=1e-9)

    r2_cpu = np.sum(w_cpu * np.sum(pts_cpu * pts_cpu, axis=1))
    r2_gpu = np.sum(w_gpu_h * np.sum(pts_gpu_h * pts_gpu_h, axis=1))
    np.testing.assert_allclose(float(r2_gpu), float(r2_cpu), rtol=1e-9, atol=1e-9)

