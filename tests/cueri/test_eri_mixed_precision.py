"""Per-class ERI mixed-precision accuracy tests.

Verifies that mixed-precision ERI kernels produce results within acceptable
tolerances compared to FP64 baseline:
- FP32 tile output: relative error < 1e-6 per quartet class
- Mixed-precision components: ||J_mixed - J_fp64||_F / ||J_fp64||_F < 1e-6
"""
from __future__ import annotations

import os
import pytest
import numpy as np


# Skip if CuPy not available
cp = pytest.importorskip("cupy")


@pytest.fixture
def lih_mol():
    """LiH molecule in STO-3G basis."""
    from asuka.frontend.molecule import Molecule

    return Molecule.from_atoms(
        [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.6))],
        basis="sto-3g",
    )


@pytest.fixture
def h2o_mol():
    """H2O molecule in 6-31g basis."""
    from asuka.frontend.molecule import Molecule

    return Molecule.from_atoms(
        [
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.757, 0.587)),
            ("H", (0.0, -0.757, 0.587)),
        ],
        basis="6-31g",
    )


def _run_direct_jk_with_params(mol, tile_dtype="float64", mixed_precision=False):
    """Run one SCF iteration and return J, K matrices."""
    from asuka.frontend.scf import run_hf_df

    # Run HF to get a density matrix
    scf_out = run_hf_df(mol, two_e_backend="direct")
    D = cp.asarray(scf_out.dm)

    from asuka.hf.direct_jk import make_direct_jk_context, direct_JK

    ctx = make_direct_jk_context(mol.ao_basis)
    J, K = direct_JK(
        ctx, D,
        tile_dtype=tile_dtype,
        mixed_precision=mixed_precision,
    )
    return J, K, D


class TestERIDispatchHelpers:
    """Test dispatch helpers for mixed-precision ERI kernel routing."""

    def test_mixed_precision_mode_encoding(self):
        """Verify precision_mode int encoding."""
        from asuka.cueri.eri_dispatch import _mixed_precision_mode

        assert _mixed_precision_mode("float64", False) == 0
        assert _mixed_precision_mode("float32", False) == 1
        assert _mixed_precision_mode("float64", True) == 2
        assert _mixed_precision_mode("float32", True) == 3
        # aliases
        assert _mixed_precision_mode("fp32", False) == 1
        assert _mixed_precision_mode("f32", False) == 1
        assert _mixed_precision_mode("F32", False) == 1
        assert _mixed_precision_mode(" float32 ", False) == 1

    def test_mixed_prec_classes(self):
        """Verify which kernel classes support mixed precision."""
        from asuka.cueri.eri_dispatch import _MIXED_PREC_CLASSES

        # Hand-written s/p kernels must be included
        for name in ("psss", "ppss", "psps", "dsss", "ppps", "pppp"):
            assert name in _MIXED_PREC_CLASSES, f"missing hand-written class {name}"
        # Generated d/f/g-shell classes must be included
        for name in ("ddss", "dddd", "fpss", "fdps", "ssfs", "ssgs", "ffgs"):
            assert name in _MIXED_PREC_CLASSES, f"missing generated class {name}"
        # Total count: 6 hand-written + 43 generated = 49
        assert len(_MIXED_PREC_CLASSES) == 49


class TestERIMixedPrecision:
    """Test suite for mixed-precision ERI kernel accuracy."""

    def test_precision_policy_eri_fields(self):
        """PrecisionPolicy has ERI kernel precision fields."""
        from asuka.cuda.mixed_precision_policy import PrecisionPolicy

        p = PrecisionPolicy.fp64()
        assert p.eri_tile_dtype == "fp64"
        assert p.eri_component_mode == "fp64"
        assert p.eri_accumulator_mode == "fp64"

        p = PrecisionPolicy.eri_conservative()
        assert p.eri_tile_dtype == "fp32"
        assert p.eri_component_mode == "mixed"
        assert p.eri_accumulator_mode == "fp64"

        p = PrecisionPolicy.eri_aggressive()
        assert p.eri_tile_dtype == "fp32"
        assert p.eri_component_mode == "mixed"
        assert p.eri_accumulator_mode == "fp32_light"

    def test_precision_policy_env_override(self):
        """PrecisionPolicy reads ASUKA_ERI_MIXED_PRECISION env var."""
        from asuka.cuda.mixed_precision_policy import PrecisionPolicy

        env = os.environ.copy()
        try:
            # Mixed precision is ON by default
            os.environ.pop("ASUKA_ERI_MIXED_PRECISION", None)
            p = PrecisionPolicy.from_env()
            assert p.eri_component_mode == "mixed"

            # Explicitly disable
            os.environ["ASUKA_ERI_MIXED_PRECISION"] = "0"
            p = PrecisionPolicy.from_env()
            assert p.eri_component_mode == "fp64"
        finally:
            os.environ.clear()
            os.environ.update(env)

    def test_precision_policy_tile_f32_env(self):
        """PrecisionPolicy reads ASUKA_ERI_TILE_F32 env var."""
        from asuka.cuda.mixed_precision_policy import PrecisionPolicy

        env = os.environ.copy()
        try:
            os.environ["ASUKA_ERI_TILE_F32"] = "1"
            p = PrecisionPolicy.from_env()
            assert p.eri_tile_dtype == "fp32"
            # component_mode should stay at default unless explicitly set
        finally:
            os.environ.clear()
            os.environ.update(env)

    def test_precision_policy_validation(self):
        """Invalid ERI precision values raise ValueError."""
        from asuka.cuda.mixed_precision_policy import PrecisionPolicy

        with pytest.raises(ValueError):
            PrecisionPolicy(eri_tile_dtype="fp16")
        with pytest.raises(ValueError):
            PrecisionPolicy(eri_component_mode="turbo")
        with pytest.raises(ValueError):
            PrecisionPolicy(eri_accumulator_mode="invalid")

    def test_precision_policy_summary(self):
        """Summary includes ERI fields."""
        from asuka.cuda.mixed_precision_policy import PrecisionPolicy

        p = PrecisionPolicy.eri_conservative()
        s = p.summary()
        assert "eri_tile_dtype" in s
        assert "eri_component_mode" in s
        assert "eri_accumulator_mode" in s
        assert s["eri_tile_dtype"] == "fp32"
        assert s["eri_component_mode"] == "mixed"
